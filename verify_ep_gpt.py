#!/usr/bin/env python3
"""Verify full GPT EP integration against an independent functional full MoE.

Run from nanoGPT (this file has NO dependency on previous verification scripts):
    python verify_ep_gpt.py --project . --device cpu --ep-size 2 --threads 1 --report ep_gpt_report.json

Project API:
    model.GPT(config, ep_group=None)
    model.GPTConfig(... use_moe, n_experts, moe_top_k, moe_aux_loss_coef ...)
    model.Block(config, ep_group=None)
    parallel_ep.setup_ep()
    parallel_ep.split_ep_parameters(model)
    parallel_ep.sync_shared_parameters(model, parallel)
    parallel_ep.build_ep_optimizer(model, learning_rate, weight_decay, beta1, beta2)
    parallel_ep.step_ep_optimizer(model, optimizer, parallel, grad_clip=0.0)
The optimizer step owns ONE gradient synchronization, global clipping,
conditional AdamW update, and zero_grad(set_to_none=True).

Both a full local MoE GPT (default constructor) and an EP GPT are constructed.
Full-model weights are loaded by GLOBAL expert IDs into local EP state keys.
The numerical reference is independent functional embedding, LayerNorm,
causal attention, GELU MLP, all-expert masked routing, and a tied output head.
It never calls project Router/MoE/Block/GPT forward to compute reference values.
The project default constructor's forward is itself checked against that oracle.

Checks: every layer's expert ownership, complete shared parameters, tied head,
strict full-to-local loading, shared broadcast preserving local experts,
full logits, CE/auxiliary sum, train/eval/no-target contracts, weighted gradient
accumulation, every local clipped gradient, AdamW parameters/moments/steps,
and sparse inactive-parameter semantics. Two optimizer updates per case use
clipping disabled and enabled. Bias on/off, top-k 1/2/E, auxiliary coefficient
0/nonzero, sequence length 1/block_size, and plain-MLP GPT are included.

The objective is mean_ranks(sum_microbatches(local_sample_weight * local_loss)).
Auxiliary statistics stay LOCAL to each microbatch. All samples in a batch
have the same number of valid targets, including ignore_index=-1 test cases.
One EP group covers WORLD, with the same group for shared-parameter DP.
Each worker has its own data; a complete GPT is replicated except experts.

CPU: FileStore/Gloo, no TCPStore/libuv. CUDA: one NCCL worker per visible device.
No project source is edited and no dataset is required. Production training
entry, data loading, AMP, compile, checkpoint, multi-node, extra expert replicas,
and stochastic-dropout equivalence are NOT verified.
"""


from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full GPT expert-parallel numerical verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=90, help="Collective timeout in seconds.")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true", help="Keep temporary per-rank reports.")
    args = parser.parse_args(argv)
    for name in ("ep_size", "threads", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and nonnegative")
    args.project = args.project.resolve()
    return args

def configure_threads(threads):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def load_project(project):
    sys.path.insert(0, str(project))
    importlib.invalidate_caches()
    names = ("parallel_ep", "moe", "moe_ep", "model")
    for name in names:
        sys.modules.pop(name, None)
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        ensure(Path(module.__file__).resolve() == (project / (name + ".py")).resolve(),
               f"Imported {name} from the wrong project: {module.__file__}")
        modules[name] = module
    return modules

class Checks:
    def __init__(self, torch, dist, args, rank, device, record, rank_path):
        self.torch, self.dist = torch, dist
        self.args, self.rank, self.device = args, rank, device
        self.record, self.rank_path = record, rank_path
        self.rtol = args.rtol if args.rtol is not None else (3e-5 if args.dtype == "float64" else 3e-4)
        self.atol = args.atol if args.atol is not None else (3e-8 if args.dtype == "float64" else 3e-5)

    def phase(self, label):
        self.record["active_phase"] = label
        write_json(self.rank_path, self.record)

    def close(self, actual, expected, label, errors, exact=False):
        try:
            ensure(actual is not None and expected is not None, f"{label}: tensor/gradient is None")
            self.torch.testing.assert_close(actual, expected, rtol=0 if exact else self.rtol,
                                           atol=0 if exact else self.atol)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    def agree(self, errors, label):
        flag = self.torch.tensor(int(bool(errors)), dtype=self.torch.int32, device=self.device)
        self.dist.all_reduce(flag, op=self.dist.ReduceOp.MAX)
        if flag.item():
            for message in errors[:8]:
                print(f"[FAIL rank={self.rank}] {label}: {message}", flush=True)
            self.record["local_errors"] = errors
            raise AssertionError(f"Verification failed: {label}; inspect the per-rank errors.")

    def passed(self, label, details=None):
        self.record["passed"].append({"label": label, "details": details or {}})
        if self.rank == 0:
            print(f"[PASS] {label}", flush=True)

def check_groups(checks, utils, info):
    checks.phase("group mapping and expert ranges")
    errors = []
    try:
        for key, expected in (("world_rank", checks.rank), ("ep_rank", checks.rank),
                              ("world_size", checks.args.ep_size), ("ep_size", checks.args.ep_size)):
            ensure(info[key] == expected, f"{key}: expected {expected}, got {info[key]}")
        for key in ("ep_group", "shared_dp_group"):
            group = info[key]
            ensure(checks.dist.get_rank(group=group) == checks.rank, f"Wrong rank in {key}")
            ensure(checks.dist.get_world_size(group=group) == checks.args.ep_size, f"Wrong size of {key}")
        for size in (1, 2, 3, 4, checks.args.ep_size):
            for local in (1, 2, 3):
                for rank in range(size):
                    actual = utils.get_expert_range(size * local, size, rank)
                    ensure(tuple(actual) == (rank * local, (rank + 1) * local),
                           f"Wrong expert range E={size * local}, EP={size}, rank={rank}: {actual}")
        for arguments in ((0, 1, 0), (4, 0, 0), (5, 2, 0), (4, 2, -1), (4, 2, 2)):
            try:
                utils.get_expert_range(*arguments)
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError(f"Invalid expert range was accepted: {arguments}")
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, "WORLD EP / shared-DP mapping and expert ownership")
    checks.passed("WORLD EP / shared-DP mapping and expert ownership")

def tensor_close(checks, actual, expected, label, errors, *, atol=None, exact=False, equal_nan=False):
    try:
        ensure(actual is not None and expected is not None, "tensor/gradient is None")
        checks.torch.testing.assert_close(actual, expected,
            rtol=0.0 if exact else checks.rtol,
            atol=0.0 if exact else checks.atol if atol is None else atol,
            equal_nan=equal_nan)
    except Exception as exc:
        errors.append(f"{label}: {exc}")

def scalar_close(checks, actual, expected, label, errors, *, exact=False):
    try:
        ensure(actual is not None, "returned None")
        actual = float(actual)
        ensure(math.isfinite(actual), f"nonfinite result: {actual}")
        ensure(math.isclose(actual, float(expected), rel_tol=0 if exact else checks.rtol,
                            abs_tol=0 if exact else checks.atol), f"expected {expected}, got {actual}")
    except Exception as exc:
        errors.append(f"{label}: {exc}")

def state_snapshot(optimizer, model):
    return {name: {key: value.detach().clone() if hasattr(value, "detach") else value
                   for key, value in optimizer.state.get(param, {}).items()}
            for name, param in model.named_parameters()}

def compare_states(checks, actual, expected, label, errors, *, exact=False):
    if set(actual) != set(expected):
        errors.append(f"{label}: state keys differ: {set(actual)} vs {set(expected)}")
        return
    for key in expected:
        value, reference = actual[key], expected[key]
        if hasattr(reference, "detach"):
            if exact or key == "step":
                tensor_close(checks, value, reference, label + "." + key, errors, exact=True)
            else:
                defaults = ((1e-10, 1e-13) if checks.args.dtype == "float64" else (1e-7, 1e-10))
                atol = checks.args.atol if checks.args.atol is not None else defaults[key == "exp_avg_sq"]
                tensor_close(checks, value, reference, label + "." + key, errors, atol=atol)
        elif value != reference:
            errors.append(f"{label}.{key}: expected {reference}, got {value}")

def linear(parameters, prefix, x):
    return torch.nn.functional.linear(
        x, parameters[prefix + ".weight"], parameters.get(prefix + ".bias"),
    )

def layer_norm(parameters, prefix, x, width):
    return torch.nn.functional.layer_norm(
        x, (width,), parameters[prefix + ".weight"],
        parameters.get(prefix + ".bias"), 1e-5,
    )

def dense_mlp(parameters, prefix, x):
    h = torch.nn.functional.gelu(linear(parameters, prefix + ".c_fc", x))
    return linear(parameters, prefix + ".c_proj", h)

def sorted_routes(logits, top_k):
    # Independent selection implementation: sort rather than torch.topk.
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    indices = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :top_k]
    mask = torch.nn.functional.one_hot(
        indices, num_classes=probabilities.shape[-1]
    ).sum(dim=1).to(probabilities.dtype)
    fractions = mask.mean(dim=0) / top_k
    auxiliary = probabilities.shape[-1] * (
        fractions * probabilities.mean(dim=0)
    ).sum()
    return probabilities, indices, mask, auxiliary

def functional_gpt(parameters, cfg, token_ids, targets=None):
    B, T = token_ids.shape
    C = cfg.n_embd
    positions = torch.arange(T, device=token_ids.device)
    x = (
        torch.nn.functional.embedding(token_ids, parameters["transformer.wte.weight"])
        + torch.nn.functional.embedding(positions, parameters["transformer.wpe.weight"])
    )
    auxiliary_sum = x.new_zeros((), dtype=torch.float32)
    layer_counts = []
    causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    for layer in range(cfg.n_layer):
        prefix = f"transformer.h.{layer}"
        h = layer_norm(parameters, prefix + ".ln_1", x, C)
        q, k, value = linear(parameters, prefix + ".attn.c_attn", h).split(C, dim=-1)
        q = q.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        k = k.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        value = value.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(C // cfg.n_head)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        h = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(B, T, C)
        x = x + linear(parameters, prefix + ".attn.c_proj", h)
        h = layer_norm(parameters, prefix + ".ln_2", x, C)
        if cfg.use_moe:
            flat = h.reshape(-1, C)
            logits = linear(parameters, prefix + ".mlp.router.gate", flat)
            probabilities, _, mask, auxiliary = sorted_routes(logits, cfg.moe_top_k)
            experts = torch.stack([
                dense_mlp(parameters, prefix + f".mlp.experts.{expert}", flat)
                for expert in range(cfg.n_experts)
            ], dim=1)
            weighted = (probabilities * mask).to(experts.dtype).unsqueeze(-1)
            mlp_output = (experts * weighted).sum(dim=1).reshape(B, T, C)
            auxiliary_sum = auxiliary_sum + auxiliary
            layer_counts.append(mask.sum(dim=0).to(torch.long))
        else:
            mlp_output = dense_mlp(parameters, prefix + ".mlp", h)
        x = x + mlp_output
    x = layer_norm(parameters, "transformer.ln_f", x, C)
    logits = linear(parameters, "lm_head", x)
    ce = None
    if targets is not None:
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
        )
    return logits, ce, auxiliary_sum, layer_counts


def preflight(project):
    contracts = {
        "parallel_ep.py": {"get_expert_range", "setup_ep", "exchange_counts", "AllToAll",
                          "split_ep_parameters", "sync_shared_parameters", "sync_ep_gradients",
                          "clip_ep_gradients", "build_ep_optimizer", "step_ep_optimizer"},
        "moe_ep.py": {"MoEEP"},
        "moe.py": {"MoE", "Router"},
        "model.py": {"GPT", "GPTConfig", "Block", "MLP"},
    }
    hashes = {}
    for filename, required in contracts.items():
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing project file: {path}")
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, filename=str(path))
        compile(tree, str(path), "exec")
        names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        ensure(required <= names, f"{filename} needs top-level definitions: {sorted(required - names)}")
        if filename == "model.py":
            for class_name in ("GPT", "Block"):
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
                init = next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
                ensure(init is not None, f"{class_name} must define __init__")
                args = init.args
                names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
                ensure("ep_group" in names or args.kwarg is not None,
                       f"model.py: {class_name}.__init__ must accept ep_group")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def cases(size):
    experts = 2 * size
    result = [{"use_moe": True, "bias": bias, "k": k, "coef": coefficient, "sequence": sequence}
              for bias in (True, False) for k in sorted({1, min(2, experts), experts})
              for coefficient in (0.0, 0.07) for sequence in (1, 7)]
    result += [{"use_moe": False, "bias": bias, "k": 1, "coef": 0.07, "sequence": 7}
               for bias in (True, False)]
    return result


def expert_key(name):
    parts = name.split(".")
    if len(parts) >= 7 and parts[:2] == ["transformer", "h"] and parts[3:5] == ["mlp", "experts"]:
        return int(parts[2]), int(parts[5])
    return None


def model_batch(cfg, rank, micro, event, samples, sequence, device):
    ids = (torch.arange(samples * sequence, device=device, dtype=torch.long).reshape(samples, sequence)
           + 7 * rank + 5 * micro + 3 * event) % cfg.vocab_size
    targets = (ids + 3) % cfg.vocab_size
    if sequence > 1:
        targets[:, 0] = -1
    return ids, targets


def make_reference_parameters(prototype):
    unique = {name: torch.nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad)
              for name, param in prototype.named_parameters()}
    parameters = dict(unique)
    parameters["lm_head.weight"] = parameters["transformer.wte.weight"]
    return unique, parameters


def reference_optimizer(unique):
    groups = []
    for decay, predicate in ((0.13, lambda p: p.ndim >= 2), (0.0, lambda p: p.ndim < 2)):
        params = [param for param in unique.values() if param.requires_grad and predicate(param)]
        if params:
            groups.append({"params": params, "weight_decay": decay})
    return torch.optim.AdamW(groups, lr=7e-4, betas=(0.8, 0.93), eps=1e-8, foreach=False, fused=False)


def construct(checks, modules, info, case):
    cfg = modules["model"].GPTConfig(
        n_layer=2, n_head=3, n_embd=12, block_size=7, vocab_size=31,
        dropout=0.0, bias=case["bias"], use_moe=case["use_moe"],
        n_experts=2 * checks.args.ep_size, moe_top_k=case["k"], moe_aux_loss_coef=case["coef"],
    )
    expected_config = dataclasses.asdict(cfg)
    dtype = getattr(torch, checks.args.dtype)
    torch.manual_seed(checks.args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        prototype = modules["model"].GPT(cfg).to(device=checks.device, dtype=dtype)
    # Distinct ordered gate logits reduce accidental FP32 top-k ties. They
    # intentionally route sparsely for k=1/2 and to every expert for k=E.
    if cfg.use_moe:
        with torch.no_grad():
            for layer, block in enumerate(prototype.transformer.h):
                weight = block.mlp.router.gate.weight
                weight.zero_()
                ramp = torch.linspace(-0.15, 0.15, cfg.n_experts, device=checks.device, dtype=dtype)
                weight[:, 0].copy_(ramp if layer % 2 == 0 else ramp.flip(0))
    torch.manual_seed(checks.args.seed + 100 + checks.rank)
    with contextlib.redirect_stdout(io.StringIO()):
        actual = modules["model"].GPT(cfg, ep_group=info["ep_group"]).to(device=checks.device, dtype=dtype)
    ensure(dataclasses.asdict(cfg) == expected_config, "Constructors must preserve global GPTConfig values")
    json.dumps(expected_config)  # Runtime ProcessGroups do not belong in GPTConfig.
    return cfg, actual, prototype


def check_structure_and_load(checks, modules, info, cfg, actual, prototype, label):
    checks.phase(label + ": module ownership / full-state loading")
    errors = []
    try:
        for model, name in ((actual, "EP"), (prototype, "full")):
            ensure(model.transformer.wte.weight is model.lm_head.weight, f"{name} head/embedding must share ONE Parameter")
            ensure(len(model.transformer.h) == cfg.n_layer, f"{name} has the wrong layer count")
        start, end = checks.rank * 2, (checks.rank + 1) * 2
        for index, (block, full_block) in enumerate(zip(actual.transformer.h, prototype.transformer.h)):
            if cfg.use_moe:
                ensure(isinstance(block.mlp, modules["moe_ep"].MoEEP), f"Layer {index} did not use MoEEP")
                ensure(isinstance(full_block.mlp, modules["moe"].MoE), "GPT(config) must keep all experts local")
                ensure(set(block.mlp.experts.keys()) == {str(i) for i in range(start, end)},
                       f"Layer {index}: wrong GLOBAL local expert IDs")
                ensure(block.mlp.expert_start == start and block.mlp.expert_end == end, "Incorrect expert range attributes")
                ensure(all(isinstance(expert, modules["model"].MLP) for expert in block.mlp.experts.values()),
                       f"Layer {index}: EP experts must use the existing MLP")
                ensure(len(full_block.mlp.experts) == cfg.n_experts, "Full-model expert count is wrong")
            else:
                ensure(isinstance(block.mlp, modules["model"].MLP), "Plain GPT must use ordinary MLP")
                ensure(isinstance(full_block.mlp, modules["model"].MLP), "Default plain GPT must use MLP")
        full_state = prototype.state_dict()
        expected_keys = {name for name in full_state
                         if expert_key(name) is None or start <= expert_key(name)[1] < end}
        ensure(set(actual.state_dict()) == expected_keys, "EP state keys must include complete shared tensors and only owned experts")
        actual.load_state_dict({name: value for name, value in full_state.items() if name in expected_keys}, strict=True)
        for name, value in actual.state_dict().items():
            checks.close(value, full_state[name], name + " full-to-local load", errors, exact=True)
        named = dict(actual.named_parameters())
        expected_shared = {name for name in named if expert_key(name) is None}
        expected_experts = set(named) - expected_shared
        shared, experts = modules["parallel_ep"].split_ep_parameters(actual)
        shared, experts = list(shared), list(experts)
        ensure([name for name, _ in shared] == sorted(expected_shared), "Wrong shared parameters or collective order")
        ensure({name for name, _ in experts} == expected_experts, "Wrong expert parameter partition")
        ensure(len(shared) + len(experts) == len(named), "Partition omitted or duplicated parameters")
        ensure(len({id(param) for _, param in shared + experts}) == len(named), "Shared head was counted more than once")
        ensure(all(named[name] is param for name, param in shared + experts), "Partition returned wrong Parameter objects")
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + ": expert ownership / complete shared state / tied head / strict loading")
    checks.passed(label + ": expert ownership / complete shared state / tied head / strict loading")


def check_shared_initialization(checks, modules, info, actual, prototype, label):
    checks.phase(label + ": shared broadcast preserves local experts")
    full_params = dict(prototype.named_parameters())
    named = dict(actual.named_parameters())
    with torch.no_grad():
        for name, param in named.items():
            if expert_key(name) is not None:
                param.add_(0.03125 * (checks.rank + 1))
            elif checks.rank != 0:
                param.add_(0.125 * checks.rank)
    local_experts = {name: param.detach().clone() for name, param in named.items() if expert_key(name) is not None}
    modules["parallel_ep"].sync_shared_parameters(actual, info)
    errors = []
    for name, param in named.items():
        expected = local_experts[name] if name in local_experts else full_params[name]
        checks.close(param, expected, name + " initialization broadcast", errors, exact=True)
    if actual.transformer.wte.weight is not actual.lm_head.weight:
        errors.append("Broadcast broke embedding/head sharing")
    checks.agree(errors, label + ": shared broadcast preserves local experts")
    actual.load_state_dict({name: value for name, value in prototype.state_dict().items()
                            if name in actual.state_dict()}, strict=True)


def check_default_constructor(checks, cfg, prototype, parameters, case, label):
    checks.phase(label + ": default GPT(config) compatibility")
    ids, targets = model_batch(cfg, checks.rank, 0, 0, 2, case["sequence"], checks.device)
    errors = []
    with torch.no_grad():
        expected_logits, ce, auxiliary, _ = functional_gpt(parameters, cfg, ids, targets)
        prototype.train()
        logits, loss = prototype(ids, targets)
        checks.close(logits, expected_logits, "default GPT full logits", errors)
        expected_loss = ce + cfg.moe_aux_loss_coef * auxiliary if cfg.use_moe else ce
        checks.close(loss, expected_loss, "default GPT training loss", errors)
        prototype.eval()
        _, eval_loss = prototype(ids, targets)
        checks.close(eval_loss, ce, "default GPT eval CE", errors)
    checks.agree(errors, label + ": default GPT(config) compatibility")
    checks.passed(label + ": default GPT(config) compatibility")


def check_eval(checks, cfg, actual, parameters, case, event, label):
    checks.phase(label + ": EP evaluation / last-token inference")
    ids, targets = model_batch(cfg, checks.rank, 0, event, 1 + checks.rank % 2, case["sequence"], checks.device)
    errors = []
    actual.eval()
    with torch.no_grad():
        expected_logits, ce, auxiliary, _ = functional_gpt(parameters, cfg, ids, targets)
        logits, loss = actual(ids, targets)
        inference_logits, inference_loss = actual(ids)
        checks.close(logits, expected_logits, "EP eval full logits", errors)
        checks.close(loss, ce, "EP eval loss must be CE only", errors)
        checks.close(inference_logits, expected_logits[:, [-1], :], "EP no-target last-position logits", errors)
        if inference_loss is not None:
            errors.append("No-target inference must return loss=None")
    if any(param.grad is not None for param in actual.parameters()):
        errors.append("Evaluation materialized or retained parameter gradients")
    actual.train()
    checks.agree(errors, label + ": EP eval CE / last-token inference")
    checks.passed(label + ": EP eval CE / last-token inference",
                  {"reference_auxiliary_sum": float(auxiliary.item()), "sequence_length": case["sequence"]})


def average_reference_gradients(checks, unique, global_counts):
    for name, param in sorted(unique.items()):
        key = expert_key(name)
        if key is not None and global_counts[key[0], key[1]].item() == 0:
            param.grad = None
        present = torch.tensor(int(param.grad is not None), dtype=torch.int32, device=checks.device)
        checks.dist.all_reduce(present, op=checks.dist.ReduceOp.MAX)
        if not present.item():
            param.grad = None
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
        checks.dist.all_reduce(grad, op=checks.dist.ReduceOp.SUM)
        param.grad = grad / checks.args.ep_size


def reference_norm_and_clip(unique, threshold):
    norm = math.sqrt(math.fsum(float(param.grad.detach().to(torch.float64).square().sum().item())
                               for param in unique.values() if param.grad is not None))
    coefficient = min(1.0, threshold / (norm + 1e-6)) if threshold > 0 else 1.0
    with torch.no_grad():
        if threshold > 0:
            for param in unique.values():
                if param.grad is not None:
                    param.grad.mul_(coefficient)
    return norm, coefficient


def train_update(checks, modules, info, cfg, actual, unique, parameters, optimizer,
                 dense_optimizer, case, event, label):
    checks.phase(label + ": accumulated full-GPT forward/backward")
    actual.train()
    optimizer.zero_grad(set_to_none=True)
    dense_optimizer.zero_grad(set_to_none=True)
    sizes = [1 + checks.rank % 2, 2, 1]
    local_loss = 0.0
    counts_history = []
    for micro, samples in enumerate(sizes):
        ids, targets = model_batch(cfg, checks.rank, micro, event, samples, case["sequence"], checks.device)
        logits, loss = actual(ids, targets)
        reference_logits, ce, auxiliary, counts = functional_gpt(parameters, cfg, ids, targets)
        reference_loss = ce + cfg.moe_aux_loss_coef * auxiliary if cfg.use_moe else ce
        errors = []
        checks.close(logits, reference_logits, f"microbatch {micro} full logits", errors)
        checks.close(loss, reference_loss, f"microbatch {micro} CE + coefficient * layer auxiliary sum", errors)
        checks.agree(errors, label + f": microbatch {micro} full logits / loss")
        weight = samples / sum(sizes)
        (loss * weight).backward()
        (reference_loss * weight).backward()
        local_loss += float(loss.detach().item()) * weight
        if cfg.use_moe:
            counts_history.append(torch.stack(counts))
    global_counts = (torch.stack(counts_history).sum(dim=0) if cfg.use_moe else
                     torch.zeros((cfg.n_layer, cfg.n_experts), dtype=torch.long, device=checks.device))
    checks.dist.all_reduce(global_counts, op=checks.dist.ReduceOp.SUM)
    average_reference_gradients(checks, unique, global_counts)
    threshold = 0.0 if event == 0 else 0.01
    expected_norm, coefficient = reference_norm_and_clip(unique, threshold)
    expected_grads = {name: None if param.grad is None else param.grad.detach().clone()
                      for name, param in unique.items()}
    params_before = {name: param.detach().clone() for name, param in actual.named_parameters()}
    states_before = state_snapshot(optimizer, actual)
    captures = []

    def capture(used_optimizer, args, kwargs):
        captures.append({name: None if param.grad is None else param.grad.detach().clone()
                         for name, param in actual.named_parameters()})

    handle = optimizer.register_step_pre_hook(capture)
    try:
        checks.phase(label + ": project EP sync / clipping / AdamW")
        result = modules["parallel_ep"].step_ep_optimizer(actual, optimizer, info, grad_clip=threshold)
    finally:
        handle.remove()
    dense_optimizer.step()
    errors = []
    if not isinstance(result, dict) or result.get("stepped") is not True:
        errors.append(f"Finite GPT update must return stepped=True, got {result}")
    scalar_close(checks, result.get("grad_norm") if isinstance(result, dict) else None,
                 expected_norm, "logical GPT norm; embedding/head counted ONCE", errors)
    if len(captures) != 1:
        errors.append(f"Expected exactly one AdamW call, got {len(captures)}")
    actual_grads = captures[0] if captures else {}
    parameter_atol = checks.args.atol if checks.args.atol is not None else (2e-6 if checks.args.dtype == "float64" else 3e-5)
    max_error = 0.0
    for name, param in actual.named_parameters():
        expected = expected_grads[name]
        if expected is None:
            if actual_grads.get(name) is not None:
                errors.append(f"{name}: inactive expert must retain grad=None at AdamW entry")
            checks.close(param, params_before[name], name + " inactive parameter preserved", errors, exact=True)
            compare_states(checks, optimizer.state.get(param, {}), states_before[name],
                           name + " inactive state preserved", errors, exact=True)
        else:
            checks.close(actual_grads.get(name), expected, name + " full-GPT synchronized/clipped gradient", errors)
        tensor_close(checks, param, unique[name], name + " AdamW parameter vs full GPT", errors, atol=parameter_atol)
        max_error = max(max_error, float((param.detach() - unique[name].detach()).abs().max().item()))
        compare_states(checks, optimizer.state.get(param, {}), dense_optimizer.state.get(unique[name], {}),
                       name + " AdamW state vs full GPT", errors)
        if param.grad is not None:
            errors.append(f"{name}: update must clear gradients to None")
    if actual.transformer.wte.weight is not actual.lm_head.weight:
        errors.append("AdamW update broke embedding/head sharing")
    checks.agree(errors, label + ": full-GPT gradients / norm / AdamW parameters and moments")
    checks.passed(label + ": full-GPT gradients / norm / AdamW parameters and moments",
                  {"local_microbatch_sizes": sizes, "local_loss": local_loss, "grad_clip": threshold,
                   "expected_preclip_norm": expected_norm, "clip_coefficient": coefficient,
                   "global_routes_per_layer": global_counts.tolist(), "max_parameter_abs_error": max_error})
    dense_optimizer.zero_grad(set_to_none=True)


def run_case(checks, modules, info, case):
    label = (f"use_moe={case['use_moe']} bias={case['bias']} k={case['k']} "
             f"aux_coef={case['coef']} T={case['sequence']}")
    if checks.rank == 0:
        print(f"[CASE] {label}", flush=True)
    checks.phase(label + ": constructors")
    errors, built = [], None
    try:
        built = construct(checks, modules, info, case)
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + ": constructors / runtime ep_group")
    cfg, actual, prototype = built
    check_structure_and_load(checks, modules, info, cfg, actual, prototype, label)
    check_shared_initialization(checks, modules, info, actual, prototype, label)
    unique, parameters = make_reference_parameters(prototype)
    check_default_constructor(checks, cfg, prototype, parameters, case, label)
    optimizer = modules["parallel_ep"].build_ep_optimizer(
        actual, learning_rate=7e-4, weight_decay=0.13, beta1=0.8, beta2=0.93)
    dense_optimizer = reference_optimizer(unique)
    optimizer.zero_grad(set_to_none=True)
    check_eval(checks, cfg, actual, parameters, case, 0, label + ": initial")
    for event in (0, 1):
        train_update(checks, modules, info, cfg, actual, unique, parameters, optimizer,
                     dense_optimizer, case, event, label + f": update {event}")
    check_eval(checks, cfg, actual, parameters, case, 2, label + ": after updates")


def worker(rank, args, workdir):
    global torch
    configure_threads(args.threads)
    record = {"rank": rank, "status": "running", "passed": [], "active_phase": "import / initialization"}
    rank_path = Path(workdir) / f"rank_{rank}.json"
    write_json(rank_path, record)
    dist = None
    try:
        import torch
        import torch.distributed as dist
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if args.device == "cuda":
            torch.cuda.set_device(rank)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        store = dist.FileStore(str(Path(workdir) / "rendezvous.store"), args.ep_size)
        store.set_timeout(timedelta(seconds=args.timeout))
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo", store=store,
                                rank=rank, world_size=args.ep_size, timeout=timedelta(seconds=args.timeout))
        record["torch_version"], record["device"] = torch.__version__, str(device)
        modules = load_project(args.project)
        checks = Checks(torch, dist, args, rank, device, record, rank_path)
        info = modules["parallel_ep"].setup_ep()
        check_groups(checks, modules["parallel_ep"], info)
        for case in cases(args.ep_size):
            run_case(checks, modules, info, case)
        dist.barrier()
        record["status"], record["active_phase"] = "passed", "complete"
    except BaseException:
        record["status"], record["exception"] = "failed", traceback.format_exc()
        print(f"[FAIL rank={rank}] phase: {record['active_phase']}", flush=True)
        raise
    finally:
        write_json(rank_path, record)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": {"full_GPT_EP_wiring_and_global_expert_IDs": True,
                  "strict_dense_weight_loading_and_shared_broadcast": True,
                  "tied_embedding_head_parameter": True, "functional_attention_and_all_expert_MLP_reference": True,
                  "train_loss_local_auxiliary_sum_eval_CE_last_token_inference": True,
                  "sample_weighted_microbatch_accumulation": True, "two_global_AdamW_updates_and_moments": True,
                  "clipping_disabled_then_enabled": True, "sparse_inactive_parameters": True,
                  "bias_topk_auxcoef_sequence_boundaries_plain_GPT": True,
                  "dropout_zero_for_numerical_comparison": True, "one_EP_group_covers_WORLD": True,
                  "objective": "mean_ranks(sum_microbatches(local_sample_weight * local_loss))",
                  "production_training_data_AMP_compile_checkpoint_multinode_extra_expert_replicas": False},
        "tolerance_note": "Router probabilities remain FP32 in float64 tests. Parameters default to absolute tolerance 2e-6 (float64) or 3e-5 (float32); moments use tighter tolerances. CLI rtol/atol override defaults.",
        "workers": []}
    workdir, code = None, 1
    try:
        report["source_sha256"] = preflight(args.project)
        print("[PASS] source syntax / GPT and Block ep_group / EP optimizer definitions (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; use the Python environment used for nanoGPT.") from exc
        ensure(torch.distributed.is_available(), "This PyTorch build has no distributed support")
        if args.device == "cuda":
            ensure(torch.cuda.is_available(), "CUDA is not available")
            ensure(torch.cuda.device_count() >= args.ep_size, "CUDA verification needs one visible device per worker")
            ensure(torch.distributed.is_nccl_available(), "This PyTorch build has no NCCL backend")
        else:
            ensure(torch.distributed.is_gloo_available(), "This PyTorch build has no Gloo backend")
        report["torch_version"] = torch.__version__
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}; full-GPT cases={len(cases(args.ep_size))}.", flush=True)
        print("[INFO] Independent functional full-GPT reference; local auxiliary statistics; clipped/unclipped AdamW.", flush=True)
        print("[INFO] No dataset is needed. Production training entry and checkpoint are pending.", flush=True)
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_gpt_")
        mp.spawn(worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        records = [json.loads((Path(workdir) / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(args.ep_size)]
        ensure(all(record["status"] == "passed" for record in records), "A worker did not complete")
        report["checks_per_worker"] = len(records[0]["passed"])
        report["status"], code = "passed", 0
        print("[PASS] Full GPT EP numerical verification completed.", flush=True)
    except BaseException:
        report["status"], report["exception"] = "failed", traceback.format_exc()
        traceback.print_exc()
    finally:
        if workdir is not None:
            for rank in range(args.ep_size):
                path = Path(workdir) / f"rank_{rank}.json"
                report["workers"].append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                                         else {"rank": rank, "status": "missing_worker_report"})
            if args.keep_workdir:
                report["workdir"] = workdir
                print(f"[INFO] Temporary reports kept: {workdir}", flush=True)
            else:
                shutil.rmtree(workdir, ignore_errors=True)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if args.report is not None:
            write_json(args.report, report)
            print(f"[INFO] Report saved: {args.report.resolve()}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
