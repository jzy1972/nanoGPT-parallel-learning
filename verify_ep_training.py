#!/usr/bin/env python3
"""Verify train_ep.py data loading, microbatches, and the real training main().

Put this file beside verify_ep_gpt.py, then run in nanoGPT:
    python verify_ep_training.py --project . --device cpu --ep-size 2 --threads 1 --report ep_training_report.json

Required project API (the verifier does not edit these files):
    train_ep.parse_args()
    train_ep.initialize_ep(args)
    train_ep.get_ep_batch(data_path, batch_size, seq_length, device, generator)
    train_ep.train_ep_step(model, optimizer, parallel, token_ids, targets,
                           micro_batch_size, grad_clip=0.0)
    train_ep.main()
    model / moe / moe_ep / parallel_ep APIs used by verify_ep_gpt.py

The independent functional full-MoE oracle comes from verify_ep_gpt.py. The
verifier calls the project's ORIGINAL train_ep_step and ORIGINAL main. It
observes optimizer entry, parameters, and moments without replacing their
mathematics. Data and tiny test configurations are temporary, not user data.

Objective: mean_ranks(sum_microbatches(micro_samples / local_batch_samples *
    (micro_CE + auxiliary_coefficient * micro_local_auxiliary_sum))).
All samples have equally many valid targets. Auxiliary statistics are local
to each microbatch, not recomputed across the entire batch or across ranks.
One EP group covers WORLD and also synchronizes the shared parameters.

main() is tested in fresh workers and must initialize and destroy its process
group. Only its rendezvous transport is redirected from env:// to FileStore
to avoid Windows TCPStore/libuv/hostname problems. Backend, rank, device,
collectives, model, data reader, training step, and optimizer remain real.
Thus TCP/env:// rendezvous itself, multi-node launching, stochastic dropout,
AMP, compile, evaluation, and checkpoint/resume are outside this test.

Default float64 improves numerical comparison; router probabilities remain
FP32. --dtype float32 also works. CUDA needs one visible GPU per EP worker.
No training dataset is required; no earlier verifier besides verify_ep_gpt.py
is required. Per-rank failures and full tracebacks are retained in the report.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


LOG_PATTERN = re.compile(
    r"^iter\s+(\d+):\s+loss=([^\s,]+),\s+grad_norm=([^\s,]+),\s+stepped=(True|False)\s*$"
)


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def load_base():
    path = Path(__file__).resolve().with_name("verify_ep_gpt.py")
    if not path.is_file():
        raise FileNotFoundError(
            "Keep verify_ep_gpt.py in the same folder as verify_ep_training.py: " + str(path)
        )
    spec = importlib.util.spec_from_file_location("_ep_training_gpt_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EP data/training-step/actual-main verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=120, help="Collective timeout in seconds.")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
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


def preflight(base, project):
    hashes = base.preflight(project)
    path = project / "train_ep.py"
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {"parse_args", "initialize_ep", "get_ep_batch", "train_ep_step", "main"}
    ensure(required <= definitions.keys(),
           f"train_ep.py needs top-level functions: {sorted(required - definitions.keys())}")
    signatures = {
        "get_ep_batch": {"data_path", "batch_size", "seq_length", "device", "generator"},
        "train_ep_step": {"model", "optimizer", "parallel", "token_ids", "targets",
                          "micro_batch_size", "grad_clip"},
    }
    for name, expected in signatures.items():
        args = definitions[name].args
        names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
        ensure(expected <= names or args.kwarg is not None,
               f"train_ep.{name} must accept: {sorted(expected)}")
    guarded = False
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__" and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"):
                guarded = True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            ensure(not isinstance(function, ast.Name) or function.id != "main",
                   "train_ep.py must guard main() with if __name__ == '__main__'")
    ensure(guarded, "train_ep.py needs if __name__ == '__main__': main()")
    hashes["train_ep.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_project(base, project):
    modules = base.load_project(project)
    sys.modules.pop("train_ep", None)
    trainer = importlib.import_module("train_ep")
    ensure(Path(trainer.__file__).resolve() == (project / "train_ep.py").resolve(),
           "Imported train_ep.py from the wrong directory")
    modules["train_ep"] = trainer
    return modules


def create_data(workdir, seed):
    import numpy as np
    directory = Path(workdir) / "data"
    directory.mkdir()
    rng = np.random.default_rng(seed % (2 ** 32))
    # Random windows expose resetting/replacing the supplied Generator more
    # reliably than periodic arange data.
    rng.integers(0, 31, size=4096, dtype=np.uint16).tofile(directory / "train.bin")
    np.arange(8, dtype=np.uint16).tofile(directory / "minimal.bin")
    np.arange(7, dtype=np.uint16).tofile(directory / "short.bin")
    (directory / "empty.bin").write_bytes(b"")
    (directory / "odd.bin").write_bytes(b"\x00\x01\x02")
    return directory


def expected_batch(torch, np, path, batch, sequence, generator, device):
    # Independent byte decoding/window construction; no project batch reader.
    tokens = np.frombuffer(Path(path).read_bytes(), dtype=np.uint16).astype(np.int64)
    starts = torch.randint(len(tokens) - sequence, (batch,), generator=generator,
                           device="cpu").tolist()
    x = np.stack([tokens[start:start + sequence] for start in starts])
    y = np.stack([tokens[start + 1:start + sequence + 1] for start in starts])
    return (torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device), starts)


def invoke_parser(trainer, argv):
    previous = sys.argv
    try:
        sys.argv = ["train_ep.py"] + list(argv)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return trainer.parse_args()
    finally:
        sys.argv = previous


def main_specs(size):
    return [
        {"name": "main_tail_clip", "bias": True, "k": 1, "coef": 0.07,
         "batch": 7, "micro": 3, "sequence": 7, "clip": 0.01, "iters": 4, "interval": 2},
        {"name": "main_full_topk_no_clip", "bias": False, "k": 2 * size, "coef": 0.07,
         "batch": 3, "micro": 8, "sequence": 1, "clip": 0.0, "iters": 2, "interval": 5},
    ]


def trainer_argv(args, data_dir, case):
    values = [
        "--device-type", args.device, "--seed", str(args.seed),
        "--n-layer", "2", "--n-head", "3", "--n-embd", "12",
        "--block-size", "7", "--vocab-size", "31", "--dropout", "0",
        "--n-experts", str(2 * args.ep_size), "--moe-top-k", str(case["k"]),
        "--moe-aux-loss-coef", str(case["coef"]),
        "--learning-rate", "0.0007", "--weight-decay", "0.13",
        "--beta1", "0.8", "--beta2", "0.93", "--grad-clip", str(case["clip"]),
        "--data-dir", str(data_dir), "--batch-size", str(case["batch"]),
        "--micro-batch-size", str(case["micro"]), "--seq-length", str(case["sequence"]),
        "--max-iters", str(case["iters"]), "--log-interval", str(case["interval"]),
    ]
    if not case["bias"]:
        values.append("--no-bias")
    return values


def check_parser(checks, trainer, data_dir):
    errors = []
    case = main_specs(checks.args.ep_size)[0]
    argv = trainer_argv(checks.args, data_dir, case)
    try:
        namespace = invoke_parser(trainer, argv)
        for name, value in (("batch_size", 7), ("micro_batch_size", 3),
                            ("seq_length", 7), ("max_iters", 4), ("log_interval", 2)):
            ensure(getattr(namespace, name) == value, f"parse_args ignored {name}")
        oversized = list(argv)
        oversized[oversized.index("--micro-batch-size") + 1] = "12"
        ensure(invoke_parser(trainer, oversized).micro_batch_size == 12,
               "A microbatch larger than the batch is valid")
        for flag, value in (("--batch-size", "0"), ("--micro-batch-size", "0"),
                            ("--seq-length", "8"), ("--max-iters", "0"),
                            ("--log-interval", "0"), ("--moe-top-k", str(2 * checks.args.ep_size + 1))):
            invalid = list(argv)
            invalid[invalid.index(flag) + 1] = value
            try:
                invoke_parser(trainer, invalid)
            except SystemExit as exc:
                ensure(exc.code != 0, f"Invalid {flag} exited successfully")
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid CLI option accepted: {flag} {value}")
    except Exception:
        errors.append(traceback.format_exc())
    checks.agree(errors, "train_ep CLI options / validation / nondivisible microbatches")
    checks.passed("train_ep CLI options / validation / nondivisible microbatches")


def check_data_reader(base, checks, trainer, data_dir):
    import numpy as np
    torch = checks.torch
    errors = []
    seed = checks.args.seed + 77 + checks.rank
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reference = torch.Generator(device="cpu").manual_seed(seed)
    observed = []
    try:
        for batch, sequence in ((7, 7), (3, 1), (1, 7), (7, 7)):
            before = torch.get_rng_state().clone()
            x, y = trainer.get_ep_batch(data_path=str(data_dir / "train.bin"),
                batch_size=batch, seq_length=sequence, device=checks.device, generator=generator)
            expected_x, expected_y, starts = expected_batch(
                torch, np, data_dir / "train.bin", batch, sequence, reference, checks.device)
            base.tensor_close(checks, x, expected_x, "sampled token IDs", errors, exact=True)
            base.tensor_close(checks, y, expected_y, "one-token-shifted targets", errors, exact=True)
            base.tensor_close(checks, generator.get_state(), reference.get_state(),
                              "passed Generator advanced exactly once", errors, exact=True)
            base.tensor_close(checks, torch.get_rng_state(), before,
                              "batch reader must preserve global CPU RNG", errors, exact=True)
            observed.append(starts)
        replay = torch.Generator(device="cpu").manual_seed(seed)
        x, y = trainer.get_ep_batch(str(data_dir / "train.bin"), 7, 7, checks.device, replay)
        replay_reference = torch.Generator(device="cpu").manual_seed(seed)
        ex, ey, _ = expected_batch(torch, np, data_dir / "train.bin", 7, 7,
                                   replay_reference, checks.device)
        base.tensor_close(checks, x, ex, "same-seed replay x", errors, exact=True)
        base.tensor_close(checks, y, ey, "same-seed replay y", errors, exact=True)
        x, y = trainer.get_ep_batch(str(data_dir / "minimal.bin"), 3, 7,
                                    checks.device, generator)
        ex = torch.arange(7, dtype=torch.long, device=checks.device).repeat(3, 1)
        base.tensor_close(checks, x, ex, "N=sequence+1 boundary x", errors, exact=True)
        base.tensor_close(checks, y, ex + 1, "N=sequence+1 boundary y", errors, exact=True)
        invalid = [("train.bin", 0, 7, ValueError), ("train.bin", 1, 0, ValueError),
                   ("short.bin", 1, 7, ValueError), ("empty.bin", 1, 7, ValueError),
                   ("odd.bin", 1, 1, ValueError), ("missing.bin", 1, 7, OSError)]
        for filename, batch, sequence, exception in invalid:
            try:
                trainer.get_ep_batch(str(data_dir / filename), batch, sequence,
                                     checks.device, generator)
            except exception:
                pass
            else:
                raise AssertionError(f"get_ep_batch accepted invalid input: {filename}, {batch}, {sequence}")
    except Exception:
        errors.append(traceback.format_exc())
    checks.agree(errors, "uint16 data / independent rank RNG / shifted targets / boundaries")
    checks.passed("uint16 data / independent rank RNG / shifted targets / boundaries",
                  {"generator_seed": seed, "sampled_starts": observed})


def expected_parameter_shapes(cfg):
    C = cfg.n_embd
    result = {"transformer.wte.weight": (cfg.vocab_size, C),
              "transformer.wpe.weight": (cfg.block_size, C),
              "transformer.ln_f.weight": (C,)}
    if cfg.bias:
        result["transformer.ln_f.bias"] = (C,)
    for layer in range(cfg.n_layer):
        prefix = f"transformer.h.{layer}"
        for norm in ("ln_1", "ln_2"):
            result[f"{prefix}.{norm}.weight"] = (C,)
            if cfg.bias:
                result[f"{prefix}.{norm}.bias"] = (C,)
        for name, shape in (("c_attn", (3 * C, C)), ("c_proj", (C, C))):
            result[f"{prefix}.attn.{name}.weight"] = shape
            if cfg.bias:
                result[f"{prefix}.attn.{name}.bias"] = (shape[0],)
        result[f"{prefix}.mlp.router.gate.weight"] = (cfg.n_experts, C)
        for expert in range(cfg.n_experts):
            for name, shape in (("c_fc", (4 * C, C)), ("c_proj", (C, 4 * C))):
                result[f"{prefix}.mlp.experts.{expert}.{name}.weight"] = shape
                if cfg.bias:
                    result[f"{prefix}.mlp.experts.{expert}.{name}.bias"] = (shape[0],)
    return result


def check_optimizer(checks, model, optimizer, errors):
    torch = checks.torch
    ensure(isinstance(optimizer, torch.optim.AdamW), "Training main must use the AdamW EP builder")
    named = {id(param): (name, param) for name, param in model.named_parameters() if param.requires_grad}
    seen = []
    for group in optimizer.param_groups:
        for key, expected in (("lr", 7e-4), ("eps", 1e-8)):
            ensure(math.isclose(float(group[key]), expected, rel_tol=1e-12),
                   f"main optimizer {key}: expected {expected}, got {group.get(key)}")
        ensure(tuple(group["betas"]) == (0.8, 0.93), "main optimizer ignored beta arguments")
        ensure(not group.get("amsgrad", False) and not group.get("maximize", False),
               "Unexpected AdamW variant")
        for param in group["params"]:
            ensure(id(param) in named, "Optimizer contains a foreign or frozen parameter")
            name, _ = named[id(param)]
            expected_decay = 0.13 if param.ndim >= 2 else 0.0
            ensure(float(group["weight_decay"]) == expected_decay, f"Wrong weight decay for {name}")
            seen.append(id(param))
    ensure(len(seen) == len(set(seen)) and set(seen) == set(named),
           "Optimizer omitted or duplicated local parameters (including the tied head)")


def assemble_main_reference(base, checks, model, optimizer, parallel, case):
    """Assemble one logical full GPT from initial real main() parameter shards.

    This consumes no random numbers and constructs no additional project GPT.
    Shared values must already be equal; gathering never repairs the model.
    """
    torch, dist = checks.torch, checks.dist
    errors = []
    cfg = model.config
    expected_cfg = {"n_layer": 2, "n_head": 3, "n_embd": 12, "block_size": 7,
                    "vocab_size": 31, "dropout": 0.0, "bias": case["bias"],
                    "use_moe": True, "n_experts": 2 * checks.args.ep_size,
                    "moe_top_k": case["k"], "moe_aux_loss_coef": case["coef"]}
    try:
        for key, value in expected_cfg.items():
            ensure(getattr(cfg, key) == value, f"main ignored config {key}={value}")
        ensure(model.transformer.wte.weight is model.lm_head.weight,
               "main must preserve embedding/output-head sharing")
        ensure(all(param.device == checks.device for param in model.parameters()), "Wrong model device")
        ensure(all(param.dtype == getattr(torch, checks.args.dtype) for param in model.parameters()),
               "Unexpected model dtype")
        check_optimizer(checks, model, optimizer, errors)
        ensure(not optimizer.state, "A fresh training main has nonempty optimizer state")
    except Exception:
        errors.append(traceback.format_exc())
    checks.agree(errors, case["name"] + ": config / fresh optimizer / tied head")
    base.check_groups(checks, sys.modules["parallel_ep"], parallel)
    local = {name: (param.detach().cpu().clone(), bool(param.requires_grad))
             for name, param in model.named_parameters()}
    gathered = [None] * checks.args.ep_size
    dist.all_gather_object(gathered, local, group=parallel["ep_group"])
    errors, assembled = [], {}
    shapes = expected_parameter_shapes(cfg)
    for rank, shard in enumerate(gathered):
        start, end = rank * 2, (rank + 1) * 2
        expected_names = {name for name in shapes
                          if base.expert_key(name) is None or start <= base.expert_key(name)[1] < end}
        if set(shard) != expected_names:
            errors.append(f"rank {rank}: missing={sorted(expected_names - set(shard))}; "
                          f"extra={sorted(set(shard) - expected_names)}")
        for name, (value, trainable) in shard.items():
            if name not in shapes:
                continue
            if tuple(value.shape) != shapes[name] or not trainable:
                errors.append(f"rank {rank}: wrong shape or frozen parameter: {name}")
            if name in assembled:
                base.tensor_close(checks, value, assembled[name],
                                  f"initial shared broadcast: rank {rank} {name}", errors, exact=True)
            else:
                assembled[name] = value
    if set(assembled) != set(shapes):
        errors.append("Cannot reconstruct every expert of the full GPT from main() shards")
    checks.agree(errors, case["name"] + ": global expert IDs / initial shared synchronization")
    unique = {name: torch.nn.Parameter(assembled[name].to(checks.device).clone())
              for name in sorted(shapes)}
    parameters = dict(unique)
    parameters["lm_head.weight"] = parameters["transformer.wte.weight"]
    checks.passed(case["name"] + ": global expert IDs / initial shared synchronization")
    return cfg, unique, parameters, base.reference_optimizer(unique)


def audit_step(base, checks, original_step, model, optimizer, parallel,
               token_ids, targets, micro_batch_size, grad_clip, cfg,
               unique, parameters, reference_optimizer, label, inject_nonfinite=False):
    torch, dist = checks.torch, checks.dist
    checks.phase(label + ": independent full-MoE microbatch reference")
    reference_optimizer.zero_grad(set_to_none=True)
    batch = token_ids.shape[0]
    sizes, expected_outputs, losses, counts = [], [], [], []
    local_loss = 0.0
    for start in range(0, batch, micro_batch_size):
        end = min(start + micro_batch_size, batch)
        logits, ce, auxiliary, layer_counts = base.functional_gpt(
            parameters, cfg, token_ids[start:end], targets[start:end])
        loss = ce + cfg.moe_aux_loss_coef * auxiliary
        weight = (end - start) / batch
        (loss * weight).backward()
        local_loss += float(loss.detach().item()) * weight
        sizes.append(end - start)
        expected_outputs.append(logits.detach())
        losses.append(loss.detach())
        counts.append(torch.stack(layer_counts))
    global_counts = torch.stack(counts).sum(dim=0)
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=parallel["ep_group"])
    base.average_reference_gradients(checks, unique, global_counts)
    expected_norm, coefficient = base.reference_norm_and_clip(unique, float(grad_clip))
    expected_grads = {name: None if param.grad is None else param.grad.detach().clone()
                      for name, param in unique.items()}
    expected_loss = torch.tensor(local_loss, dtype=torch.float64, device=checks.device)
    dist.all_reduce(expected_loss, op=dist.ReduceOp.SUM, group=parallel["shared_dp_group"])
    expected_loss.div_(checks.args.ep_size)
    params_before = {name: param.detach().clone() for name, param in model.named_parameters()}
    states_before = base.state_snapshot(optimizer, model)
    captures, calls, forward_errors = [], [], []

    def capture(used_optimizer, args, kwargs):
        captures.append({name: None if param.grad is None else param.grad.detach().clone()
                         for name, param in model.named_parameters()})

    def capture_forward(module, inputs, kwargs, output):
        micro = len(calls)
        calls.append(True)
        if micro >= len(sizes):
            forward_errors.append("train_ep_step made an extra GPT forward call")
            return
        start = sum(sizes[:micro])
        end = start + sizes[micro]
        try:
            actual_ids = inputs[0] if inputs else kwargs["idx"]
            actual_targets = inputs[1] if len(inputs) > 1 else kwargs.get("targets")
            ensure(module.training and torch.is_grad_enabled(), "Training forward must enable train mode and autograd")
            base.tensor_close(checks, actual_ids, token_ids[start:end],
                              f"micro {micro} input slice", forward_errors, exact=True)
            base.tensor_close(checks, actual_targets, targets[start:end],
                              f"micro {micro} target slice", forward_errors, exact=True)
            logits, loss = output
            base.tensor_close(checks, logits, expected_outputs[micro],
                              f"micro {micro} full logits", forward_errors)
            base.tensor_close(checks, loss, losses[micro],
                              f"micro {micro} CE + local auxiliary", forward_errors)
        except Exception:
            forward_errors.append(traceback.format_exc())

    step_handle = optimizer.register_step_pre_hook(capture)
    forward_handle = model.register_forward_hook(capture_forward, with_kwargs=True)
    anomaly_handle = None
    if inject_nonfinite and checks.rank == checks.args.ep_size - 1:
        anomaly_handle = model.transformer.wte.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan")))
    try:
        checks.phase(label + ": original train_ep_step")
        result = original_step(model=model, optimizer=optimizer, parallel=parallel,
            token_ids=token_ids, targets=targets, micro_batch_size=micro_batch_size,
            grad_clip=grad_clip)
    finally:
        step_handle.remove()
        forward_handle.remove()
        if anomaly_handle is not None:
            anomaly_handle.remove()
    if not inject_nonfinite:
        reference_optimizer.step()
    errors = list(forward_errors)
    if len(calls) != len(sizes):
        errors.append(f"Expected microbatch sizes {sizes}, got {len(calls)} GPT forwards")
    if not isinstance(result, dict):
        errors.append(f"train_ep_step must return a dict, got {type(result).__name__}")
        result_for_checks = {}
    else:
        result_for_checks = result
    if result_for_checks.get("stepped") is not (not inject_nonfinite):
        errors.append(f"Wrong stepped result: {result_for_checks}")
    base.scalar_close(checks, result_for_checks.get("loss"), float(expected_loss.item()),
                      "global mean of sample-weighted microbatch losses", errors)
    if inject_nonfinite:
        if result_for_checks.get("grad_norm", "missing") is not None:
            errors.append("A global nonfinite skip must return grad_norm=None")
        if captures:
            errors.append("AdamW ran despite a nonfinite gradient on one rank")
    else:
        base.scalar_close(checks, result_for_checks.get("grad_norm"), expected_norm,
                          "logical full-model preclip norm", errors)
        if len(captures) != 1:
            errors.append(f"Expected one AdamW call per batch, got {len(captures)}")
    actual_grads = captures[0] if captures else {}
    parameter_atol = checks.args.atol if checks.args.atol is not None else (
        2e-6 if checks.args.dtype == "float64" else 3e-5)
    maximum_error = 0.0
    for name, param in model.named_parameters():
        reference = unique[name]
        gradient = expected_grads[name]
        if inject_nonfinite or gradient is None:
            base.tensor_close(checks, param, params_before[name], name + " unchanged", errors, exact=True)
            base.compare_states(checks, optimizer.state.get(param, {}), states_before[name],
                                name + " unchanged AdamW state", errors, exact=True)
            if not inject_nonfinite and actual_grads.get(name) is not None:
                errors.append(name + ": unrouted expert must retain grad=None at AdamW entry")
        else:
            base.tensor_close(checks, actual_grads.get(name), gradient,
                              name + " synchronized/clipped gradient", errors)
        base.tensor_close(checks, param, reference, name + " full-reference AdamW parameter",
                          errors, atol=parameter_atol)
        base.compare_states(checks, optimizer.state.get(param, {}),
                            reference_optimizer.state.get(reference, {}), name + " full-reference AdamW state", errors)
        maximum_error = max(maximum_error, float((param.detach() - reference.detach()).abs().max().item()))
        if param.grad is not None:
            errors.append(name + ": gradients must be cleared after update or skip")
    if not model.training:
        errors.append("train_ep_step did not switch the model into training mode")
    if model.transformer.wte.weight is not model.lm_head.weight:
        errors.append("Training broke embedding/output-head sharing")
    checks.agree(errors, label + ": loss / microbatches / gradients / AdamW / cleanup")
    details = {"microbatch_sizes": sizes, "sample_weights": [size / batch for size in sizes],
               "local_loss": local_loss, "global_loss": float(expected_loss.item()),
               "returned_loss": float(result_for_checks["loss"]),
               "returned_grad_norm": None if result_for_checks["grad_norm"] is None else float(result_for_checks["grad_norm"]),
               "grad_clip": float(grad_clip), "expected_preclip_norm": None if inject_nonfinite else expected_norm,
               "clip_coefficient": None if inject_nonfinite else coefficient,
               "injected_nonfinite_on_rank": checks.args.ep_size - 1 if inject_nonfinite else None,
               "stepped": not inject_nonfinite, "global_routes_per_layer": global_counts.tolist(),
               "max_parameter_abs_error": maximum_error}
    checks.passed(label + ": loss / microbatches / gradients / AdamW / cleanup", details)
    reference_optimizer.zero_grad(set_to_none=True)
    return result, details


def direct_cases(size):
    return [
        {"bias": True, "k": 1, "coef": 0.07, "batch": 7, "micro": 3, "sequence": 7},
        {"bias": False, "k": min(2, 2 * size), "coef": 0.0, "batch": 5, "micro": 2, "sequence": 7},
        {"bias": True, "k": 2 * size, "coef": 0.07, "batch": 3, "micro": 8, "sequence": 7},
        {"bias": False, "k": 1, "coef": 0.07, "batch": 1, "micro": 1, "sequence": 1},
    ]


def run_direct_cases(base, checks, modules, parallel):
    torch = checks.torch
    for index, case in enumerate(direct_cases(checks.args.ep_size)):
        label = f"direct bias={case['bias']} k={case['k']} B={case['batch']} micro={case['micro']} T={case['sequence']}"
        checks.phase(label + ": constructors")
        cfg, model, prototype = base.construct(checks, modules, parallel, dict(case, use_moe=True))
        base.check_structure_and_load(checks, modules, parallel, cfg, model, prototype, label)
        unique, parameters = base.make_reference_parameters(prototype)
        reference = base.reference_optimizer(unique)
        optimizer = modules["parallel_ep"].build_ep_optimizer(model=model,
            learning_rate=7e-4, weight_decay=0.13, beta1=0.8, beta2=0.93)
        # The first call must clear these stale gradients before accumulating.
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        events = [(False, 0.0), (False, 0.01)]
        if index == 0:
            events += [(True, 0.01), (False, 0.01)]
        for event, (inject, threshold) in enumerate(events):
            ids, targets = base.model_batch(cfg, checks.rank, 0, event,
                case["batch"], case["sequence"], checks.device)
            model.eval()  # train_ep_step must restore training mode itself.
            audit_step(base, checks, modules["train_ep"].train_ep_step,
                model, optimizer, parallel, ids, targets, case["micro"], threshold,
                cfg, unique, parameters, reference, label + f" event={event}", inject)


def init_worker_runtime(args, rank):
    global torch
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(getattr(torch, args.dtype))
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    base = load_base()
    base.torch = torch  # functional oracle uses its module-level torch binding.
    return torch, dist, device, base


def file_init(dist, args, rank, store_path, **kwargs):
    store = dist.FileStore(str(store_path), args.ep_size)
    store.set_timeout(timedelta(seconds=args.timeout))
    dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo", store=store,
        rank=rank, world_size=args.ep_size, timeout=timedelta(seconds=args.timeout), **kwargs)


def function_worker(rank, args, workdir):
    base = load_base()
    base.configure_threads(args.threads)
    record = {"rank": rank, "status": "running", "active_phase": "import / initialization", "passed": []}
    rank_path = Path(workdir) / "functions" / f"rank_{rank}.json"
    base.write_json(rank_path, record)
    dist = None
    try:
        torch, dist, device, base = init_worker_runtime(args, rank)
        file_init(dist, args, rank, Path(workdir) / "functions" / "rendezvous.store")
        record["torch_version"], record["device"] = torch.__version__, str(device)
        modules = load_project(base, args.project)
        checks = base.Checks(torch, dist, args, rank, device, record, rank_path)
        parallel = modules["parallel_ep"].setup_ep()
        base.check_groups(checks, modules["parallel_ep"], parallel)
        check_parser(checks, modules["train_ep"], Path(workdir) / "data")
        check_data_reader(base, checks, modules["train_ep"], Path(workdir) / "data")
        run_direct_cases(base, checks, modules, parallel)
        dist.barrier()
        record["status"], record["active_phase"] = "passed", "complete"
    except BaseException:
        record["status"], record["exception"] = "failed", traceback.format_exc()
        print(f"[FAIL rank={rank}] {record['active_phase']}", flush=True)
        raise
    finally:
        base.write_json(rank_path, record)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def check_main_logs(text, case, rank, steps):
    matches = []
    for line in text.splitlines():
        if line.startswith("iter "):
            match = LOG_PATTERN.fullmatch(line)
            ensure(match is not None, f"Malformed training log: {line}")
            matches.append(match)
    expected_indices = [index for index in range(case["iters"])
                        if index % case["interval"] == 0 or index == case["iters"] - 1] if rank == 0 else []
    actual_indices = [int(match.group(1)) for match in matches]
    ensure(actual_indices == expected_indices,
           f"rank {rank} log iterations: expected {expected_indices}, got {actual_indices}")
    for match in matches:
        index = int(match.group(1))
        expected = steps[index]
        # Numerical equivalence to the oracle was already checked in audit_step.
        # Here check that the logger prints that actual result to six decimals.
        logged_loss = expected.get("returned_loss", expected["global_loss"])
        logged_norm = expected.get("returned_grad_norm", expected["expected_preclip_norm"])
        ensure(math.isclose(float(match.group(2)), logged_loss, rel_tol=0, abs_tol=1.1e-6),
               f"iter {index}: printed loss does not match the updated batch")
        norm = match.group(3)
        if logged_norm is None:
            ensure(norm == "n/a", f"iter {index}: skipped norm should be n/a")
        else:
            ensure(math.isclose(float(norm), logged_norm, rel_tol=0, abs_tol=1.1e-6),
                   f"iter {index}: printed norm differs from the training result")
        ensure((match.group(4) == "True") == expected["stepped"], f"iter {index}: incorrect stepped log")
    return expected_indices


def main_worker(rank, args, workdir, case):
    base = load_base()
    base.configure_threads(args.threads)
    directory = Path(workdir) / case["name"]
    rank_path = directory / f"rank_{rank}.json"
    record = {"rank": rank, "status": "running", "active_phase": "import", "passed": [], "case": case}
    base.write_json(rank_path, record)
    dist, original_init, original_destroy = None, None, None
    original_argv = sys.argv
    output = io.StringIO()
    handles = {}
    try:
        torch, dist, device, base = init_worker_runtime(args, rank)
        import numpy as np
        modules = load_project(base, args.project)
        trainer = modules["train_ep"]
        checks = base.Checks(torch, dist, args, rank, device, record, rank_path)
        original_init, original_destroy = dist.init_process_group, dist.destroy_process_group
        original_batch, original_step = trainer.get_ep_batch, trainer.train_ep_step
        reference_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000 + rank)
        data_path = Path(workdir) / "data" / "train.bin"
        data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
        steps, batch_calls, init_calls, destroy_calls = [], [], [], []
        reference = None

        @functools.wraps(original_init)
        def observed_init(*positional, **keywords):
            bound = inspect.signature(original_init).bind(*positional, **keywords)
            requested = bound.arguments
            backend = requested.get("backend")
            expected_backend = "nccl" if args.device == "cuda" else "gloo"
            ensure(str(backend).lower() == expected_backend, f"main initialized {backend}, expected {expected_backend}")
            ensure(not dist.is_initialized(), "main initialized a process group twice")
            init_calls.append({"backend": str(backend), "requested_init_method": requested.get("init_method")})
            store = dist.FileStore(str(directory / "main_rendezvous.store"), args.ep_size)
            store.set_timeout(timedelta(seconds=args.timeout))
            forwarded = {"backend": backend, "store": store, "rank": rank, "world_size": args.ep_size,
                         "timeout": timedelta(seconds=args.timeout)}
            if requested.get("device_id") is not None:
                forwarded["device_id"] = requested["device_id"]
            return original_init(**forwarded)

        @functools.wraps(original_destroy)
        def observed_destroy(*positional, **keywords):
            destroy_calls.append(True)
            return original_destroy(*positional, **keywords)

        @functools.wraps(original_batch)
        def observed_batch(*positional, **keywords):
            bound = inspect.signature(original_batch).bind(*positional, **keywords)
            bound.apply_defaults()
            values = bound.arguments
            before = torch.get_rng_state().clone()
            actual = original_batch(*positional, **keywords)
            errors = []
            try:
                ensure(Path(values["data_path"]).resolve() == data_path.resolve(), "main used the wrong train.bin path")
                ensure(values["batch_size"] == case["batch"] and values["seq_length"] == case["sequence"],
                       "main ignored batch-size or seq-length arguments")
                ensure(torch.device(values["device"]) == device, "main used the wrong batch device")
                x, y, starts = expected_batch(torch, np, data_path, case["batch"],
                    case["sequence"], reference_generator, device)
                base.tensor_close(checks, actual[0], x, "main sampled x", errors, exact=True)
                base.tensor_close(checks, actual[1], y, "main shifted y", errors, exact=True)
                base.tensor_close(checks, values["generator"].get_state(), reference_generator.get_state(),
                                  "main independent per-rank Generator state", errors, exact=True)
                base.tensor_close(checks, torch.get_rng_state(), before,
                                  "main data reader preserved global CPU RNG", errors, exact=True)
                batch_calls.append({"starts": starts, "generator_seed": args.seed + 10_000 + rank})
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + f": actual data call {len(batch_calls)}")
            return actual

        @functools.wraps(original_step)
        def observed_step(*positional, **keywords):
            nonlocal reference
            bound = inspect.signature(original_step).bind(*positional, **keywords)
            bound.apply_defaults()
            values = bound.arguments
            errors = []
            try:
                ensure(values["micro_batch_size"] == case["micro"], "main ignored micro-batch-size")
                ensure(float(values["grad_clip"]) == case["clip"], "main ignored grad-clip")
                ensure(len(batch_calls) == len(steps) + 1, "main must load exactly one batch before each step")
                ensure(tuple(values["token_ids"].shape) == (case["batch"], case["sequence"]), "Wrong training batch shape")
                expected_x, expected_y = handles["last_batch"]
                base.tensor_close(checks, values["token_ids"], expected_x, "main passes sampled inputs to step", errors, exact=True)
                base.tensor_close(checks, values["targets"], expected_y, "main passes sampled targets to step", errors, exact=True)
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + ": main training-step arguments")
            if reference is None:
                reference = assemble_main_reference(base, checks, values["model"], values["optimizer"],
                                                     values["parallel"], case)
                handles["model"], handles["optimizer"] = values["model"], values["optimizer"]
            else:
                ensure(values["model"] is handles["model"] and values["optimizer"] is handles["optimizer"],
                       "main recreated the model or optimizer between iterations")
            cfg, unique, parameters, reference_optimizer = reference
            result, details = audit_step(base, checks, original_step, values["model"], values["optimizer"],
                values["parallel"], values["token_ids"], values["targets"], values["micro_batch_size"],
                values["grad_clip"], cfg, unique, parameters, reference_optimizer,
                case["name"] + f" iter={len(steps)}")
            steps.append(details)
            return result

        @functools.wraps(original_batch)
        def remember_batch(*positional, **keywords):
            result = observed_batch(*positional, **keywords)
            # Keep independent copies to catch changing/replacing the returned
            # data before the main loop passes it to train_ep_step.
            handles["last_batch"] = (result[0].detach().clone(), result[1].detach().clone())
            return result

        dist.init_process_group, dist.destroy_process_group = observed_init, observed_destroy
        trainer.get_ep_batch, trainer.train_ep_step = remember_batch, observed_step
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.ep_size), LOCAL_RANK=str(rank),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        sys.argv = [str(args.project / "train_ep.py")] + trainer_argv(args, Path(workdir) / "data", case)
        record["active_phase"] = case["name"] + ": original main()"
        with contextlib.redirect_stdout(output):
            trainer.main()
        record["stdout"] = output.getvalue()
        ensure(len(init_calls) == 1, f"main initialized {len(init_calls)} process groups")
        ensure(len(destroy_calls) == 1 and not dist.is_initialized(),
               "main must destroy its process group in finally")
        ensure(len(batch_calls) == case["iters"], f"Expected {case['iters']} data calls, got {len(batch_calls)}")
        ensure(len(steps) == case["iters"], f"Expected {case['iters']} updates, got {len(steps)}")
        log_indices = check_main_logs(output.getvalue(), case, rank, steps)
        ensure(hashlib.sha256(data_path.read_bytes()).hexdigest() == data_sha256, "main modified the temporary dataset")
        record.update(status="passed", active_phase="complete", updates=steps, data_calls=batch_calls,
                      init_calls=init_calls, destroy_calls=len(destroy_calls), logged_iterations=log_indices)
        if rank == 0:
            print(f"[PASS] {case['name']}: actual main / {len(steps)} AdamW updates / data RNG / logs / cleanup", flush=True)
    except BaseException:
        record["status"], record["exception"] = "failed", traceback.format_exc()
        record["stdout"] = output.getvalue()
        if output.getvalue():
            print(f"[CAPTURED rank={rank}]\n{output.getvalue()}", flush=True)
        print(f"[FAIL rank={rank}] {record['active_phase']}", flush=True)
        raise
    finally:
        sys.argv = original_argv
        try:
            if dist is not None and original_init is not None:
                dist.init_process_group, dist.destroy_process_group = original_init, original_destroy
                if dist.is_initialized():
                    original_destroy()
        except BaseException:
            record["status"], record["cleanup_exception"] = "failed", traceback.format_exc()
            raise
        finally:
            base.write_json(rank_path, record)


def collect_records(base, workdir, size, phases):
    records = {}
    for phase in phases:
        records[phase] = []
        for rank in range(size):
            path = Path(workdir) / phase / f"rank_{rank}.json"
            records[phase].append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                                  else {"rank": rank, "status": "missing_worker_report"})
    return records


def main(argv=None):
    args = parse_args(argv)
    # Thread limits must be set before NumPy / PyTorch import and before spawn.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": {"actual_get_ep_batch": True, "independent_rank_data_rng": True,
                  "actual_train_ep_step": True, "tail_microbatch_sample_weighting": True,
                  "independent_functional_full_MoE_reference": True,
                  "every_local_gradient_parameter_AdamW_moment": True,
                  "nonfinite_skip_then_recovery": True, "actual_training_main": True,
                  "main_config_data_calls_logs_group_cleanup": True,
                  "main_initial_shared_broadcast_and_expert_ownership": True,
                  "dropout_zero_for_comparison": True,
                  "rendezvous": "FileStore replaces env:// transport; real backend/collectives are retained",
                  "objective": "mean_ranks(sum_microbatches(samples/local_batch * local_micro_loss))",
                  "AMP_compile_eval_checkpoint_multinode_env_TCP_rendezvous": False},
        "worker_phases": {}}
    workdir, base, code = None, None, 1
    phases = ["functions"] + [case["name"] for case in main_specs(args.ep_size)]
    try:
        base = load_base()
        report["oracle_sha256"] = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
        report["source_sha256"] = preflight(base, args.project)
        print("[PASS] source syntax / EP training helper definitions (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Use the nanoGPT Python environment with both PyTorch and NumPy installed.") from exc
        ensure(torch.distributed.is_available(), "PyTorch distributed is unavailable")
        if args.device == "cuda":
            ensure(torch.cuda.is_available() and torch.cuda.device_count() >= args.ep_size,
                   "CUDA verification requires one visible GPU per EP worker")
            ensure(torch.distributed.is_nccl_available(), "This PyTorch build has no NCCL backend")
        else:
            ensure(torch.distributed.is_gloo_available(), "This PyTorch build has no Gloo backend")
        report["torch_version"], report["numpy_version"] = torch.__version__, np.__version__
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_training_")
        for phase in phases:
            (Path(workdir) / phase).mkdir()
        data_dir = create_data(workdir, args.seed)
        data_hash = hashlib.sha256((data_dir / "train.bin").read_bytes()).hexdigest()
        report["temporary_data_sha256"] = data_hash
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}.", flush=True)
        print("[INFO] Temporary uint16 data; independent full-MoE oracle; sample-weighted tail microbatches.", flush=True)
        print("[INFO] Real main() in fresh workers; FileStore rendezvous; EP eval/checkpoint are pending.", flush=True)
        mp.spawn(function_worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        for case in main_specs(args.ep_size):
            mp.spawn(main_worker, args=(args, workdir, case), nprocs=args.ep_size, join=True)
        records = collect_records(base, workdir, args.ep_size, phases)
        ensure(all(record["status"] == "passed" for phase in records.values() for record in phase),
               "At least one worker did not finish all checks")
        ensure(preflight(base, args.project) == report["source_sha256"], "A project source file changed during verification")
        ensure(hashlib.sha256((data_dir / "train.bin").read_bytes()).hexdigest() == data_hash,
               "Temporary training data changed")
        report["status"], code = "passed", 0
        print("[PASS] EP training entry verification completed.", flush=True)
    except BaseException:
        report["status"], report["exception"] = "failed", traceback.format_exc()
        traceback.print_exc()
    finally:
        if workdir is not None:
            report["worker_phases"] = collect_records(base, workdir, args.ep_size, phases)
            if args.keep_workdir:
                report["workdir"] = workdir
                print(f"[INFO] Temporary reports kept: {workdir}", flush=True)
            else:
                shutil.rmtree(workdir, ignore_errors=True)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if args.report is not None:
            if base is not None:
                base.write_json(args.report, report)
            else:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Report saved: {args.report.resolve()}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
