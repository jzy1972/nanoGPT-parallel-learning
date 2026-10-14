#!/usr/bin/env python3
"""Numerical verification of local-expert MoE with replicated data parallelism.

Place this file beside verify_moe_model.py and run from the nanoGPT directory:
    python verify_moe_dp.py --project . --device cpu --dp-size 2 --threads 1 --report moe_dp_report.json

All experts are replicated on every worker. This is DP verification, not EP.
The DP size is configurable; the CPU default launches small Gloo workers with
one BLAS/OpenMP thread each. CUDA launches one NCCL worker per visible device.
Direct FileStore avoids TCPStore/libuv and hostname rendezvous on Windows.

The actual micro-step loop, clipping/optimizer update statements, DDP unused
parameter setting, and estimate_loss function are extracted from train.py's
AST. Project files are never edited; the rest of train.py is not executed.
The verifier supplies deterministic batches, an unfused AdamW, and a no-op
FP32/FP64 scaler. It compares against independently computed per-worker,
per-microbatch losses before averaging gradients across DP workers.

The reference deliberately does NOT concatenate all samples for an auxiliary
loss: the current model defines that loss on each local microbatch.

Not covered: EP, AMP/overflow handling, torch.compile, fused CUDA AdamW,
launcher/data-loader setup, checkpoint saving/resume in the actual train.py
main loop, and multi-node launch. Two local scripts are required; no dataset.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import traceback
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MoE + DP numerical verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--timeout", type=int, default=120, help="Collective timeout in seconds.")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    for name in ("dp_size", "threads", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and nonnegative")
    return args


def configure_threads(threads):
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def load_helper():
    path = Path(__file__).resolve().with_name("verify_moe_model.py")
    if not path.is_file():
        raise FileNotFoundError(
            "Place verify_moe_model.py beside verify_moe_dp.py before running."
        )
    spec = importlib.util.spec_from_file_location("_moe_model_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_call(node, object_name, method_name):
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == object_name and node.func.attr == method_name
    )


def training_source(project):
    path = project / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "micro_step"
    ]
    ensure(len(loops) == 1, "train.py must have exactly one micro_step loop.")
    micro_loop = loops[0]
    parent = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.While) and micro_loop in node.body),
        None,
    )
    ensure(parent is not None, "Expected micro_step loop directly inside the training while loop.")
    start = parent.body.index(micro_loop)
    end = None
    for index in range(start + 1, len(parent.body)):
        if any(is_call(node, "optimizer", "zero_grad") for node in ast.walk(parent.body[index])):
            end = index + 1
            break
    ensure(end is not None, "Could not locate optimizer.zero_grad after the micro-step loop.")
    update_nodes = copy.deepcopy(parent.body[start:end])
    ensure(any(is_call(node, "scaler", "step")
               for statement in update_nodes for node in ast.walk(statement)),
           "Expected scaler.step(optimizer) in the actual training update.")
    eval_functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "estimate_loss"
    ]
    ensure(len(eval_functions) == 1, "train.py must define top-level estimate_loss().")
    eval_calls = [
        node for node in parent.body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id == "estimate_loss" for child in ast.walk(node))
    ]
    ensure(len(eval_calls) == 1, "Could not locate the evaluation condition in the training loop.")
    ensure(not any(isinstance(node, ast.Name) and node.id == "master_process"
                   for node in ast.walk(eval_calls[0].test)),
           "All DP ranks must enter estimate_loss; remove master_process from its outer condition.")
    assignments = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ddp_find_unused_parameters"
                for target in node.targets)
    ]
    ensure(len(assignments) == 1,
           "Expected ddp_find_unused_parameters = model.config.use_moe before DDP wrapping.")
    ddp_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DDP"
    ]
    ensure(len(ddp_calls) == 1, "Expected one DDP(model, ...) constructor.")
    ddp_keywords = {keyword.arg: keyword.value for keyword in ddp_calls[0].keywords}
    ensure("find_unused_parameters" in ddp_keywords,
           "DDP must receive find_unused_parameters for dynamically selected experts.")
    update_module = ast.fix_missing_locations(ast.Module(body=update_nodes, type_ignores=[]))
    eval_module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(eval_functions[0])], type_ignores=[])
    )
    return {
        "update": compile(update_module, f"{path}::<actual training update>", "exec"),
        "evaluation": compile(eval_module, f"{path}::<actual estimate_loss>", "exec"),
        "unused_assignment": compile(
            ast.Expression(body=copy.deepcopy(assignments[0].value)), str(path), "eval"
        ),
        "unused_keyword": compile(
            ast.Expression(body=copy.deepcopy(ddp_keywords["find_unused_parameters"])),
            str(path), "eval",
        ),
        "static_graph": (
            compile(ast.Expression(body=copy.deepcopy(ddp_keywords["static_graph"])),
                    str(path), "eval")
            if "static_graph" in ddp_keywords else None
        ),
    }


def agree(torch, dist, v, rank, label, function, records):
    error = None
    detail = {}
    try:
        detail = function() or {}
    except Exception:
        error = traceback.format_exc()
    flag = torch.tensor(int(error is not None), dtype=torch.int32, device=v.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, {"rank": rank, "error": error})
        records.append({"name": label, "status": "failed"})
        if rank == 0:
            for item in errors:
                if item["error"]:
                    print(f"[FAIL rank={item['rank']}] {label}\n{item['error']}", flush=True)
        raise AssertionError(f"Distributed verification failed: {label}")
    records.append({"name": label, "status": "passed", **detail})
    if rank == 0:
        print(f"[PASS] {label}", flush=True)


def snapshot_gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


class NoopScaler:
    """Instrument the real update without AMP scaling or overflow behavior."""
    def __init__(self, model):
        self.model = model
        self.scale_calls = self.unscale_calls = self.step_calls = self.update_calls = 0
        self.before_clip = None
        self.after_clip = None

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def unscale_(self, _optimizer):
        self.unscale_calls += 1
        self.before_clip = snapshot_gradients(self.model)

    def step(self, optimizer):
        self.step_calls += 1
        if self.before_clip is None:
            self.before_clip = snapshot_gradients(self.model)
        self.after_clip = snapshot_gradients(self.model)
        return optimizer.step()

    def update(self):
        self.update_calls += 1


def make_probe(torch, MoE, helper):
    ToyExpert = helper.make_toy_expert()

    class MoEProbe(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.moe = MoE(config, expert_cls=ToyExpert)

        def forward(self, x, targets):
            output, auxiliary = self.moe(x)
            loss = (
                0.5 * torch.nn.functional.mse_loss(output, targets)
                + self.config.moe_aux_loss_coef * auxiliary
            )
            return output, loss
    return MoEProbe


def training_data(torch, helper, v, cfg, kind, case, dp_rank, step, micro):
    if kind == "probe":
        pattern = case["pattern"]
        if pattern == "earlier_only":
            rotation = step + (2 if micro == case["microbatches"] - 1 else dp_rank % 2)
            token_pattern = "sparse"
        elif pattern == "globally_unused":
            rotation, token_pattern = step, "sparse"
        elif pattern == "rank_disjoint":
            rotation, token_pattern = dp_rank + step, "sparse"
        else:
            rotation, token_pattern = dp_rank + step + micro, "mixed"
        x = helper.controlled_input(v, cfg.n_experts, token_pattern, rotation)
        targets = torch.linspace(-0.3, 0.5, x.numel(), dtype=v.dtype, device=v.device).reshape_as(x)
        targets = targets + 0.03 * dp_rank + 0.01 * micro
        return x, targets
    generator = torch.Generator(device="cpu")
    generator.manual_seed(v.args.seed + 10000 + 113 * dp_rank + 41 * step + 7 * micro)
    ids = torch.randint(cfg.vocab_size, (2, 5), generator=generator).to(v.device)
    targets = torch.randint(cfg.vocab_size, (2, 5), generator=generator).to(v.device)
    return ids, targets


def expected_update(torch, helper, v, reference, kind, case, step, rank):
    """Average local accumulated gradients AFTER each local backward.

    This preserves the local microbatch auxiliary objective, and the float32
    softmax's backward scale, before the DP gradient averaging operation.
    """
    names = dict(reference.named_parameters())
    gradients = {name: torch.zeros_like(parameter) for name, parameter in names.items()}
    route_totals = {}
    own_outputs = []
    objective = 0.0
    for dp_rank in range(v.args.dp_size):
        if kind == "probe":
            local_reference = copy.deepcopy(reference)
            local_parameters = dict(local_reference.named_parameters())
        else:
            parameters = helper.parameter_reference(reference)
            local_parameters = {name: parameters[name] for name in names}
        for micro in range(case["microbatches"]):
            x, targets = training_data(
                torch, helper, v, reference.config, kind, case, dp_rank, step, micro
            )
            if kind == "probe":
                output, auxiliary, counts = helper.dense_toy_moe(local_reference.moe, x)
                loss = (
                    0.5 * torch.nn.functional.mse_loss(output, targets)
                    + reference.config.moe_aux_loss_coef * auxiliary
                )
                routes = {"moe.experts": counts}
            else:
                output, ce, auxiliary, counts = helper.functional_gpt(
                    parameters, reference.config, x, targets
                )
                loss = ce + reference.config.moe_aux_loss_coef * auxiliary
                routes = {
                    f"transformer.h.{layer}.mlp.experts": count
                    for layer, count in enumerate(counts)
                }
            for prefix, counts in routes.items():
                for expert, count in enumerate(counts.tolist()):
                    key = f"{prefix}.{expert}"
                    route_totals[key] = route_totals.get(key, 0) + count
            if dp_rank == rank:
                own_outputs.append((output.detach().clone(), loss.detach().clone()))
            objective += float(loss.detach().item()) / (
                case["microbatches"] * v.args.dp_size
            )
            (loss / case["microbatches"]).backward()
        for name, parameter in local_parameters.items():
            if parameter.grad is not None:
                gradients[name].add_(parameter.grad.detach() / v.args.dp_size)
    for name in gradients:
        match = re.match(r"^(.*\.experts\.\d+)\.", name)
        if match and route_totals.get(match.group(1), 0) == 0:
            gradients[name] = None
    return gradients, own_outputs, objective, route_totals


def run_actual_update(torch, v, ddp_model, optimizer, source, batches, grad_clip):
    microbatches = len(batches)
    prefetched = iter([*batches[1:], batches[0]])
    fetch_count = 0

    def get_batch(split):
        nonlocal fetch_count
        ensure(split == "train", "Training update fetched an unexpected split.")
        fetch_count += 1
        return next(prefetched)

    outputs = []
    sync_flags = []

    def capture(_module, _inputs, result):
        ensure(isinstance(result, tuple) and len(result) == 2, "Expected (output, loss).")
        outputs.append((result[0].detach().clone(), result[1].detach().clone()))
        sync_flags.append(bool(ddp_model.require_backward_grad_sync))

    scaler = NoopScaler(ddp_model.module)
    handle = ddp_model.module.register_forward_hook(capture)
    namespace = {
        "torch": torch, "ddp": True, "model": ddp_model, "optimizer": optimizer,
        "ctx": nullcontext(), "get_batch": get_batch,
        "gradient_accumulation_steps": microbatches,
        "grad_clip": grad_clip, "scaler": scaler,
        "X": batches[0][0], "Y": batches[0][1],
    }
    try:
        exec(source["update"], namespace)
    finally:
        handle.remove()
    return {
        "scaler": scaler, "outputs": outputs, "sync_flags": sync_flags,
        "fetch_count": fetch_count, "final_loss": namespace.get("loss"),
    }


def compare_gradient_maps(v, actual, expected, label):
    ensure(actual is not None, f"{label}: no gradients were captured.")
    ensure(actual.keys() == expected.keys(), f"{label}: parameter names differ.")
    for name, reference in expected.items():
        if reference is None:
            ensure(actual[name] is None,
                   f"{label}: globally unused expert parameter received a gradient: {name}")
        else:
            ensure(actual[name] is not None, f"{label}: missing global gradient: {name}")
            v.close(actual[name], reference, f"{label}: {name}")


def gradient_norm(torch, gradients, device):
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients.values():
        if gradient is not None:
            squared.add_(gradient.detach().to(torch.float64).square().sum())
    return squared.sqrt()


def check_replicas(torch, dist, v, model, label):
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    root = flat.clone()
    dist.broadcast(root, src=0)
    v.close(flat, root, label)


def wrap_actual_ddp(torch, source, model, device):
    namespace = {"model": model, "use_moe": model.config.use_moe}
    namespace["ddp_find_unused_parameters"] = eval(source["unused_assignment"], namespace)
    unused = eval(source["unused_keyword"], namespace)
    ensure(bool(unused), "Actual train.py DDP settings did not enable unused-parameter detection.")
    static_graph = eval(source["static_graph"], namespace) if source["static_graph"] else False
    ensure(not static_graph, "Dynamic MoE routing requires static_graph=False.")
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=bool(unused), static_graph=False,
    )


def verify_case(torch, dist, helper, v, rank, GPT, GPTConfig, MoE,
                source, case, records):
    kind = case["kind"]
    cfg = helper.tiny_config(
        GPTConfig, bias=case["bias"], experts=4 if kind == "probe" else 3,
        top_k=case["top_k"], coef=0.11,
    )
    torch.manual_seed(v.args.seed + rank + 29 * case["id"])
    if kind == "probe":
        model = make_probe(torch, MoE, helper)(cfg).to(device=v.device, dtype=v.dtype)
        helper.identity_gate(model.moe.router.gate)
    else:
        model = GPT(cfg).to(device=v.device, dtype=v.dtype)
    wrapped = wrap_actual_ddp(torch, source, model, v.device)
    reference = copy.deepcopy(wrapped.module)
    optimizer = helper.build_unfused_optimizer(wrapped.module)
    ref_optimizer = helper.build_unfused_optimizer(reference)
    prefix = (
        f"{kind}, bias={case['bias']}, k={case['top_k']}, "
        f"microbatches={case['microbatches']}, clip={case['clip']}, {case['pattern']}"
    )
    agree(torch, dist, v, rank, prefix + ": initial replicas",
          lambda: check_replicas(torch, dist, v, wrapped.module, "Initial parameter replicas"), records)
    for step in range(3):
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        expected = {}

        def prepare():
            gradients, own_outputs, loss, route_totals = expected_update(
                torch, helper, v, reference, kind, case, step, rank
            )
            expected.update(gradients=gradients, outputs=own_outputs, loss=loss,
                            route_totals=route_totals)
            return {"reference_mean_loss": loss}

        agree(torch, dist, v, rank, prefix + f", step={step}: independent objective",
              prepare, records)
        batches = [
            training_data(torch, helper, v, cfg, kind, case, rank, step, micro)
            for micro in range(case["microbatches"])
        ]
        # This executes real DDP forwards/backwards and real train.py statements.
        # Errors inside a collective are bounded by the process-group timeout.
        actual = run_actual_update(torch, v, wrapped, optimizer, source, batches, case["clip"])

        def check_update():
            scaler = actual["scaler"]
            ensure(len(actual["outputs"]) == case["microbatches"], "Wrong forward count.")
            for micro, (output, loss) in enumerate(actual["outputs"]):
                ref_output, ref_loss = expected["outputs"][micro]
                v.close(output, ref_output, f"Local output, microbatch={micro}")
                v.close(loss, ref_loss, f"Local loss, microbatch={micro}")
            ensure(actual["sync_flags"] == (
                [False] * (case["microbatches"] - 1) + [True]
            ), f"Unexpected gradient-sync flags: {actual['sync_flags']}")
            ensure(actual["fetch_count"] == case["microbatches"], "Wrong prefetch count.")
            ensure(scaler.scale_calls == case["microbatches"], "Wrong scaled backward count.")
            ensure(scaler.unscale_calls == int(case["clip"] > 0), "Wrong unscale count.")
            ensure(scaler.step_calls == 1 and scaler.update_calls == 1,
                   "Expected exactly one optimizer/scaler update.")
            v.close(actual["final_loss"], expected["outputs"][-1][1] / case["microbatches"],
                    "Actual accumulation loss divisor")
            compare_gradient_maps(v, scaler.before_clip, expected["gradients"],
                                  "DP-averaged accumulated gradients")
            norm = gradient_norm(torch, expected["gradients"], v.device)
            for name, parameter in reference.named_parameters():
                parameter.grad = expected["gradients"][name]
            if case["clip"] > 0:
                ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), case["clip"])
                v.close(ref_norm.to(torch.float64), norm, "Reference replicated-model gradient norm")
            clipped = snapshot_gradients(reference)
            compare_gradient_maps(v, scaler.after_clip, clipped, "Gradient clipping")
            ref_optimizer.step()
            ref_parameters = dict(reference.named_parameters())
            for name, parameter in wrapped.module.named_parameters():
                v.close(parameter, ref_parameters[name], f"AdamW update: {name}")
            helper.compare_optimizer_states(
                v, optimizer, ref_optimizer, list(wrapped.module.named_parameters()), ref_parameters
            )
            ensure(all(parameter.grad is None for parameter in wrapped.module.parameters()),
                   "Actual update did not clear gradients with set_to_none=True.")
            return {
                "gradient_norm": float(norm.item()),
                "globally_unused_experts": sum(value == 0 for value in expected["route_totals"].values()),
            }

        agree(torch, dist, v, rank, prefix + f", step={step}: actual train.py accumulation / gradients / AdamW",
              check_update, records)
        agree(torch, dist, v, rank, prefix + f", step={step}: updated replicas",
              lambda: check_replicas(torch, dist, v, wrapped.module, "Updated parameter replicas"), records)
    if kind == "gpt":
        verify_actual_evaluation(torch, dist, helper, v, rank, wrapped,
                                 reference, source, case, records)
    del wrapped, optimizer, ref_optimizer, reference, model


def evaluation_data(torch, v, cfg, split, dp_rank, iteration):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        v.args.seed + 50000 + (1000 if split == "val" else 0) + 67 * dp_rank + iteration
    )
    ids = torch.randint(cfg.vocab_size, (2, 5), generator=generator).to(v.device)
    targets = torch.randint(cfg.vocab_size, (2, 5), generator=generator).to(v.device)
    return ids, targets


def verify_actual_evaluation(torch, dist, helper, v, rank, wrapped,
                             reference, source, case, records):
    eval_iters = 2
    expected = {}
    parameters = helper.parameter_reference(reference)
    with torch.no_grad():
        for split in ("train", "val"):
            loss_sum = 0.0
            for dp_rank in range(v.args.dp_size):
                for iteration in range(eval_iters):
                    ids, targets = evaluation_data(
                        torch, v, reference.config, split, dp_rank, iteration
                    )
                    _, ce, _, _ = helper.functional_gpt(parameters, reference.config, ids, targets)
                    loss_sum += float(ce.item())
            expected[split] = loss_sum / (v.args.dp_size * eval_iters)
    for initial_training in (True, False):
        wrapped.train(initial_training)
        calls = {"train": 0, "val": 0}

        def get_batch(split):
            iteration = calls[split]
            calls[split] += 1
            return evaluation_data(torch, v, reference.config, split, rank, iteration)

        namespace = {
            "torch": torch, "dist": dist, "model": wrapped, "ctx": nullcontext(),
            "device": v.device, "ddp": True, "ddp_world_size": v.args.dp_size,
            "eval_iters": eval_iters, "get_batch": get_batch,
        }
        exec(source["evaluation"], namespace)
        actual = namespace["estimate_loss"]()

        def check():
            ensure(set(actual) == {"train", "val"}, "Evaluation must return train/val losses.")
            ensure(calls == {"train": eval_iters, "val": eval_iters}, "Wrong eval batch count.")
            ensure(wrapped.training == initial_training and wrapped.module.training == initial_training,
                   "estimate_loss did not restore the previous mode.")
            for split in ("train", "val"):
                v.close(torch.as_tensor(actual[split], dtype=torch.float64, device=v.device),
                        torch.tensor(expected[split], dtype=torch.float64, device=v.device),
                        f"Actual distributed {split} CE mean")
            return {"losses": {split: float(actual[split]) for split in actual}}

        agree(torch, dist, v, rank,
              f"GPT k={case['top_k']}, bias={case['bias']}: actual estimate_loss / global CE mean / mode={initial_training}",
              check, records)
    wrapped.train()


def test_cases():
    configurations = [
        ("probe", True, 1, 1, 0.0, "rank_disjoint"),
        ("probe", True, 1, 3, 0.0, "earlier_only"),
        ("probe", True, 2, 3, 0.01, "earlier_only"),
        ("probe", True, 1, 2, 0.0, "globally_unused"),
        ("probe", False, 2, 2, 0.01, "rank_disjoint"),
        ("probe", False, 4, 2, 0.0, "mixed_tokens"),
        ("gpt", True, 1, 1, 0.0, "random_tokens"),
        ("gpt", True, 2, 2, 0.01, "random_tokens"),
        ("gpt", False, 1, 3, 0.01, "random_tokens"),
        ("gpt", False, 3, 2, 0.0, "random_tokens"),
    ]
    return [
        {"id": index, "kind": kind, "bias": bias, "top_k": top_k,
         "microbatches": microbatches, "clip": clip, "pattern": pattern}
        for index, (kind, bias, top_k, microbatches, clip, pattern) in enumerate(configurations)
    ]


def worker(rank, args, temporary):
    configure_threads(args.threads)
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    helper = load_helper()
    helper.torch = torch
    args.device = f"cuda:{rank}" if args.device == "cuda" else "cpu"
    report = {"rank": rank, "status": "running", "checks": []}
    try:
        device = torch.device(args.device)
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        store = dist.FileStore(str(Path(temporary) / "rendezvous"), args.dp_size)
        dist.init_process_group(
            backend=backend, store=store, rank=rank, world_size=args.dp_size,
            timeout=timedelta(seconds=args.timeout),
        )
        GPT, GPTConfig, _, _, MoE = helper.load_project(args.project)
        source = training_source(args.project)
        v = helper.Verifier(args, report)
        report.update(device=str(v.device), dtype=args.dtype,
                      rtol=v.rtol, atol=v.atol, torch_version=torch.__version__)
        for case in test_cases():
            verify_case(torch, dist, helper, v, rank, GPT, GPTConfig, MoE,
                        source, case, report["checks"])
        report.update(status="passed", tensor_comparisons=v.comparisons,
                      max_abs_error=v.max_abs_error)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        (Path(temporary) / f"rank_{rank}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


def read_worker_reports(temporary, world):
    reports = []
    for rank in range(world):
        path = Path(temporary) / f"rank_{rank}.json"
        if path.is_file():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            reports.append({"rank": rank, "status": "missing"})
    return reports


def main(argv=None):
    args = parse_args(argv)
    args.project = args.project.expanduser().resolve()
    report = {
        "schema_version": 1, "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(args.project), "device": args.device,
        "dp_size": args.dp_size, "dtype": args.dtype, "threads": args.threads,
        "scope": "real DDP + actual train.py update fragment and estimate_loss",
        "auxiliary_objective": "mean of local per-worker/per-microbatch losses",
        "not_tested": [
            "EP", "AMP/overflow", "torch.compile", "fused CUDA optimizer",
            "production launcher/data loader", "actual main-loop checkpoint/resume", "multi-node launch",
        ],
        "workers": [],
    }
    success = False
    try:
        configure_threads(args.threads)
        helper = load_helper()
        report["source_sha256"] = helper.preflight(args.project)
        training_source(args.project)
        helper_path = Path(__file__).resolve().with_name("verify_moe_model.py")
        report["reference_script_sha256"] = hashlib.sha256(helper_path.read_bytes()).hexdigest()
        print("[PASS] project syntax / actual training and evaluation fragments (static only).", flush=True)
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is not installed in this Python environment.") from exc
        dist = importlib.import_module("torch.distributed")
        multiprocessing = importlib.import_module("torch.multiprocessing")
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not include torch.distributed.")
        if args.device == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() < args.dp_size:
                raise RuntimeError("--dp-size exceeds available visible CUDA devices.")
            if not dist.is_nccl_available():
                raise RuntimeError("CUDA verification requires a PyTorch build with NCCL.")
        elif not dist.is_gloo_available():
            raise RuntimeError("CPU verification requires a PyTorch build with Gloo.")
        print(f"[INFO] {args.dp_size} {args.device} workers; threads per worker={args.threads}; dtype={args.dtype}.", flush=True)
        print("[INFO] Replicated experts + DP; eager execution; three optimizer steps per case.", flush=True)
        with tempfile.TemporaryDirectory(prefix="moe_dp_verification_") as temporary:
            try:
                multiprocessing.spawn(
                    worker, args=(args, temporary), nprocs=args.dp_size, join=True,
                )
            finally:
                report["workers"] = read_worker_reports(temporary, args.dp_size)
        ensure(all(item["status"] == "passed" for item in report["workers"]),
               "At least one worker did not complete verification.")
        report["status"] = "passed"
        success = True
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        print(f"[FAIL] {report['error']}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if args.report is not None:
            path = args.report.expanduser().resolve()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
                print(f"[INFO] Report saved: {path}", flush=True)
            except OSError as exc:
                print(f"[FAIL] Cannot write report: {exc}", file=sys.stderr, flush=True)
                success = False
    if success:
        print("[PASS] MoE+DP numerical verification completed.", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
