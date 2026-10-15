#!/usr/bin/env python3
"""Numerically verify EP clipping and AdamW updates against a full MoE.

Keep verify_ep_forward.py and verify_ep_grad_sync.py in THIS script's folder.
Run from the nanoGPT project:
    python verify_ep_optimizer.py --project . --device cpu --ep-size 2 --threads 1 --report ep_optimizer_report.json

Functions under test, all top-level in parallel_ep.py:
    build_ep_optimizer(model, learning_rate, weight_decay, beta1=0.9, beta2=0.95)
    clip_ep_gradients(model, max_norm, parallel) -> pre-clip norm or None
    step_ep_optimizer(model, optimizer, parallel, grad_clip=0.0)
        -> {"stepped": bool, "grad_norm": float | None}

step_ep_optimizer owns ONE sync_ep_gradients call after accumulation, clipping,
the conditional optimizer.step, and zero_grad(set_to_none=True) on both paths.
max_norm/grad_clip == 0 disables clipping, but still computes the norm and
checks for nonfinite gradients. Negative/nonfinite thresholds are invalid.

The previous scripts provide a two-layer nested EP fixture with tied, frozen,
optional, and unused shared parameters; its dense counterpart evaluates all
experts with independent routing masks. Reference gradients are averaged over
rank-local objectives, using sample-weighted local microbatch losses and local
auxiliary losses. Globally unrouted expert gradients are restored to None in
the dense reference, to match sparse dispatch and AdamW's skip semantics.

Checks include parameter groups/deduplication, shared norm counted once,
expert norm across all owners, common clipping scale, weighted accumulation,
all local gradients at optimizer entry, AdamW parameters/moments/step counts,
inactive experts with existing state, global NaN/Inf skips, and recovery.
The project optimizer builder is used for EP; reference AdamW is independent.

No project files are edited. No dataset or production training entry is used.
The topology is one EP group covering WORLD, with the same shared-DP group.
AMP, compile, GPT training/eval, checkpoint, multi-node, and extra expert
replica DP groups are NOT verified. CPU uses FileStore/Gloo, CUDA NCCL.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
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
    parser = argparse.ArgumentParser(description="EP global clipping and AdamW numerical verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=90)
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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_helpers():
    folder = Path(__file__).resolve().parent
    paths = {name: folder / (name + ".py") for name in ("verify_ep_grad_sync", "verify_ep_forward")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Keep {path.name} in the SAME folder as verify_ep_optimizer.py.")
    spec = importlib.util.spec_from_file_location("_ep_grad_reference_for_optimizer", paths["verify_ep_grad_sync"])
    sync = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sync
    spec.loader.exec_module(sync)
    base, _ = sync.load_base()
    for name in ("preflight", "build_models", "expected_partition", "reference_forward", "micro_sizes", "expert_key"):
        ensure(hasattr(sync, name), f"verify_ep_grad_sync.py is missing {name}; use the supplied version.")
    hashes = {name + ".py": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return sync, base, hashes


def preflight(project, sync, base):
    hashes = sync.preflight(project, base)
    path = project / "parallel_ep.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    needed = {"build_ep_optimizer", "clip_ep_gradients", "step_ep_optimizer"}
    ensure(needed <= functions, f"parallel_ep.py needs top-level functions: {sorted(needed - functions)}")
    return hashes


def cases(size):
    experts = 2 * size
    return [{"family": family, "bias": bias, "k": k, "clip": clip, "microsteps": 3}
            for family in ("toy", "MLP") for bias in (True, False)
            for k in sorted({1, min(2, experts), experts}) for clip in (0.0, 0.01)]


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


def independent_optimizer(torch, model, hyper):
    groups = []
    params = [param for param in model.parameters() if param.requires_grad]
    for decay, predicate in ((hyper["weight_decay"], lambda p: p.ndim >= 2), (0.0, lambda p: p.ndim < 2)):
        selected = [param for param in params if predicate(param)]
        if selected:
            groups.append({"params": selected, "weight_decay": decay})
    return torch.optim.AdamW(groups, lr=hyper["learning_rate"], betas=(hyper["beta1"], hyper["beta2"]),
                             eps=1e-8, foreach=False, fused=False)


def hyperparameters():
    return {"learning_rate": 7e-4, "weight_decay": 0.13, "beta1": 0.8, "beta2": 0.93}


def check_builder(checks, utils, stage, optimizer, hyper):
    errors = []
    params = {id(param): (name, param) for name, param in stage.named_parameters() if param.requires_grad}
    actual = [param for group in optimizer.param_groups for param in group["params"]]
    try:
        ensure(isinstance(optimizer, checks.torch.optim.AdamW), "Builder must return torch.optim.AdamW")
        ensure(len({id(param) for param in actual}) == len(actual), "A tied Parameter occurs multiple times")
        ensure({id(param) for param in actual} == set(params), "Optimizer must contain ALL and ONLY local trainable parameters")
        ensure(stage.offset is stage.offset_alias, "Shared weight tie is broken")
        for group in optimizer.param_groups:
            ensure(group["lr"] == hyper["learning_rate"], "Incorrect learning rate")
            ensure(tuple(group["betas"]) == (hyper["beta1"], hyper["beta2"]), "Incorrect betas")
            ensure(group["eps"] == 1e-8, "Incorrect AdamW epsilon")
            ensure(group["foreach"] is False and group["fused"] is False, "Use foreach=False and fused=False for this baseline")
            for param in group["params"]:
                expected = hyper["weight_decay"] if param.ndim >= 2 else 0.0
                ensure(group["weight_decay"] == expected, f"Incorrect weight decay for {params[id(param)][0]}")
        ensure(len(optimizer.state) == 0, "New optimizer must not have initialized moments")
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, "AdamW groups / local ownership / tied deduplication / frozen exclusion")


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


def check_invalid_arguments(checks, utils, stage, optimizer, info):
    checks.phase("invalid optimizer and clipping arguments")
    errors = []
    hyper = hyperparameters()
    invalid_hyper = [(key, value) for key in ("learning_rate", "weight_decay")
                     for value in (-1.0, math.nan, math.inf)]
    invalid_hyper += [(key, value) for key in ("beta1", "beta2")
                      for value in (-0.1, 1.0, math.nan, math.inf)]
    for key, value in invalid_hyper:
        try:
            utils.build_ep_optimizer(model=stage, **{**hyper, key: value})
        except ValueError:
            pass
        except Exception as exc:
            errors.append(f"Invalid {key}: expected ValueError, got {exc}")
        else:
            errors.append(f"Invalid {key}={value} was accepted")
    for value in (-1.0, math.nan, math.inf):
        for name, call in (
            ("clip_ep_gradients", lambda: utils.clip_ep_gradients(stage, value, info)),
            ("step_ep_optimizer", lambda: utils.step_ep_optimizer(stage, optimizer, info, grad_clip=value)),
        ):
            try:
                call()
            except ValueError:
                pass
            except Exception as exc:
                errors.append(f"{name}({value}): expected ValueError, got {exc}")
            else:
                errors.append(f"{name} accepted invalid threshold={value}")
    checks.agree(errors, "invalid optimizer and clipping arguments")
    checks.passed("invalid optimizer and clipping arguments")


def synthetic_gradients(torch, stage, size):
    """Already-synchronized gradients: shared 3, expert 0 -> 4, last expert -> 12."""
    stage.zero_grad(set_to_none=True)
    names = dict(stage.named_parameters())
    names["offset"].grad = torch.zeros_like(names["offset"])
    names["offset"].grad[0] = 3.0
    for expert_id, value in ((0, 4.0), (2 * size - 1, 12.0)):
        prefix = f"blocks.0.experts.{expert_id}."
        matching = [(name, param) for name, param in names.items() if name.startswith(prefix) and param.ndim == 2]
        if matching:
            _, param = sorted(matching)[0]
            grad = torch.zeros_like(param)
            grad.reshape(-1)[0] = value
            param.grad = grad.t().contiguous().t()


def check_synthetic_clipping(checks, utils, stage, info):
    torch = checks.torch
    params_before = {name: param.detach().clone() for name, param in stage.named_parameters()}
    for threshold in (0.0, 0.01, 100.0):
        label = f"synthetic GLOBAL norm=13, max_norm={threshold}"
        checks.phase(label)
        synthetic_gradients(torch, stage, checks.args.ep_size)
        before = {name: None if param.grad is None else param.grad.detach().clone()
                  for name, param in stage.named_parameters()}
        result = utils.clip_ep_gradients(stage, threshold, info)
        errors = []
        scalar_close(checks, result, 13.0, "pre-clip norm; shared tie counted ONCE", errors)
        coefficient = min(1.0, threshold / (13.0 + 1e-6)) if threshold > 0 else 1.0
        for name, param in stage.named_parameters():
            if before[name] is None:
                if param.grad is not None:
                    errors.append(f"{name}: clipping materialized an unused gradient")
            else:
                checks.close(param.grad, before[name] * coefficient, name + " uniform clipping", errors)
            checks.close(param, params_before[name], name + " unchanged parameter", errors, exact=True)
        checks.agree(errors, label)
        checks.passed(label)
    stage.zero_grad(set_to_none=True)
    errors = []
    zero_norm = utils.clip_ep_gradients(stage, 0.01, info)
    scalar_close(checks, zero_norm, 0.0, "all-None norm", errors, exact=True)
    if any(param.grad is not None for param in stage.parameters()):
        errors.append("All-None clipping must preserve None gradients")
    checks.agree(errors, "all-None gradients / zero norm")
    checks.passed("all-None gradients / zero norm")

    tests = [("nan", math.nan), ("inf", math.inf), ("negative inf", -math.inf)]
    if checks.args.dtype == "float64":
        tests.append(("finite values whose squared norm overflows", 1e200))
    for label, value in tests:
        checks.phase("direct clipping rejection: " + label)
        synthetic_gradients(torch, stage, checks.args.ep_size)
        if checks.rank == checks.args.ep_size - 1:
            local_experts = [(name, param) for name, param in stage.named_parameters()
                             if ".experts." in name and param.requires_grad]
            name, param = sorted(local_experts)[0]
            grad = torch.zeros_like(param)
            grad.reshape(-1)[0] = value
            param.grad = grad
        before = {name: None if param.grad is None else param.grad.detach().clone()
                  for name, param in stage.named_parameters()}
        result = utils.clip_ep_gradients(stage, 0.01, info)
        errors = []
        if result is not None:
            errors.append(f"Every rank must return None for {label}")
        for name, param in stage.named_parameters():
            if before[name] is None:
                if param.grad is not None:
                    errors.append(f"{name}: unused gradient changed on rejection")
            else:
                tensor_close(checks, param.grad, before[name], name + " rejected gradient preserved", errors,
                             exact=True, equal_nan=True)
            checks.close(param, params_before[name], name + " rejected parameter preserved", errors, exact=True)
        checks.agree(errors, "direct clipping rejection: " + label)
        checks.passed("direct clipping rejection: " + label)
    stage.zero_grad(set_to_none=True)


def input_for(checks, base, config, pattern, micro, samples):
    values = base.controlled_input(checks.torch, config, checks.rank,
        "rotating" if pattern == "rotating" else "one_owner", checks.device, getattr(checks.torch, checks.args.dtype))
    values = values.repeat(1, math.ceil(samples / values.shape[1]), 1)[:, :samples].contiguous()
    if pattern == "last" or (pattern == "early" and micro > 0):
        values[:, :, :config.n_experts] = checks.torch.roll(values[:, :, :config.n_experts],
                                                         shifts=config.n_experts - 1, dims=-1)
    return values


def accumulate(checks, sync, base, config, stage, reference, pattern, event):
    torch = checks.torch
    stage.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    sizes = sync.micro_sizes(checks.rank, 3)
    total = sum(sizes)
    history, inputs = [], []
    local_loss = 0.0
    for micro, samples in enumerate(sizes):
        values = input_for(checks, base, config, pattern, micro, samples)
        x, reference_x = values.clone().requires_grad_(True), values.clone().requires_grad_(True)
        use_optional = pattern != "last" and (checks.rank + micro) % checks.args.ep_size == 0
        expected_output, expected_auxiliary, counts = sync.reference_forward(torch, base, reference,
                                                                           reference_x, config, use_optional)
        output, auxiliary = stage(x, use_optional)
        errors = []
        checks.close(output, expected_output, f"microbatch {micro} output", errors)
        checks.close(auxiliary, expected_auxiliary, f"microbatch {micro} local auxiliary", errors)
        checks.agree(errors, f"accumulated forward, event={event}, microbatch={micro}")
        weight = samples / total
        loss = weight * base.objective(torch, output, auxiliary, checks.rank + 11 * micro + 7 * event, "mixed")
        reference_loss = weight * base.objective(torch, expected_output, expected_auxiliary,
                                                checks.rank + 11 * micro + 7 * event, "mixed")
        loss.backward()
        reference_loss.backward()
        local_loss += float(loss.detach().item())
        inputs.append((x, reference_x))
        history.append(counts)
    history = torch.stack(history)
    checks.dist.all_reduce(history, op=checks.dist.ReduceOp.SUM, group=checks.dist.group.WORLD)
    return {"sizes": sizes, "local_loss": local_loss, "inputs": inputs, "history": history,
            "global_counts": history.sum(dim=0)}


def average_reference_gradients(checks, sync, reference, global_counts):
    """Full-model independent global mean; full-reference-only unused experts keep None."""
    torch, dist = checks.torch, checks.dist
    for name, param in sorted(reference.named_parameters()):
        key = sync.expert_key(name)
        if key is not None and global_counts[int(key[0]), key[1]].item() == 0:
            param.grad = None
        present = torch.tensor(int(param.grad is not None), dtype=torch.int32, device=checks.device)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        if not present.item():
            param.grad = None
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        param.grad = grad / checks.args.ep_size


def reference_norm_and_clip(torch, reference, threshold):
    # All logical parameters are present exactly once in this dense model.
    norm = math.sqrt(math.fsum(float(param.grad.detach().to(dtype=torch.float64).square().sum().item())
                               for param in reference.parameters() if param.grad is not None))
    coefficient = min(1.0, threshold / (norm + 1e-6)) if threshold > 0 else 1.0
    with torch.no_grad():
        if threshold > 0:
            for param in reference.parameters():
                if param.grad is not None:
                    param.grad.mul_(coefficient)
    return norm, coefficient


def finite_step(checks, sync, base, utils, info, config, stage, reference,
                optimizer, reference_optimizer, case, pattern, event, label):
    checks.phase(label + ": accumulation")
    record = accumulate(checks, sync, base, config, stage, reference, pattern, event)
    average_reference_gradients(checks, sync, reference, record["global_counts"])
    expected_norm, coefficient = reference_norm_and_clip(checks.torch, reference, case["clip"])
    expected_grads = {name: None if param.grad is None else param.grad.detach().clone()
                      for name, param in reference.named_parameters()}
    before_params = {name: param.detach().clone() for name, param in stage.named_parameters()}
    before_states = state_snapshot(optimizer, stage)
    calls = []

    def capture(used_optimizer, args, kwargs):
        calls.append({name: None if param.grad is None else param.grad.detach().clone()
                      for name, param in stage.named_parameters()})

    handle = optimizer.register_step_pre_hook(capture)
    try:
        checks.phase(label + ": project sync / clip / AdamW / clear gradients")
        result = utils.step_ep_optimizer(stage, optimizer, info, grad_clip=case["clip"])
    finally:
        handle.remove()
    reference_optimizer.step()
    errors = []
    if not isinstance(result, dict) or result.get("stepped") is not True:
        errors.append(f"Finite update must return stepped=True, got {result}")
    scalar_close(checks, result.get("grad_norm") if isinstance(result, dict) else None,
                 expected_norm, "whole-model pre-clip norm", errors)
    if len(calls) != 1:
        errors.append(f"Expected exactly one AdamW call, got {len(calls)}")
    actual_grads = calls[0] if calls else {}
    reference_params = dict(reference.named_parameters())
    parameter_atol = checks.args.atol if checks.args.atol is not None else (2e-6 if checks.args.dtype == "float64" else 3e-5)
    max_parameter_error = 0.0
    inactive_expert_states = sum(
        sync.expert_key(name) is not None
        and expected_grads[name] is None
        and bool(reference_optimizer.state.get(param, {}))
        for name, param in reference.named_parameters()
    )
    early_only = (record["history"][0, 0] > 0) & (
        record["history"][1:, 0].sum(dim=0) == 0
    )
    if pattern == "last" and case["k"] == 1 and inactive_expert_states == 0:
        errors.append("Fixture did not exercise an inactive expert with existing AdamW moments")
    if pattern == "early" and case["k"] == 1 and not early_only.any().item():
        errors.append("Fixture did not exercise an expert used only in an earlier microbatch")
    for name, param in stage.named_parameters():
        expected = expected_grads[name]
        actual = actual_grads.get(name)
        if expected is None:
            if actual is not None:
                errors.append(f"{name}: AdamW entry must keep globally unused/frozen grad=None")
            checks.close(param, before_params[name], name + " inactive parameter unchanged", errors, exact=True)
            compare_states(checks, optimizer.state.get(param, {}), before_states[name],
                           name + " inactive optimizer state unchanged", errors, exact=True)
        else:
            checks.close(actual, expected, name + " synchronized and clipped AdamW gradient", errors)
        tensor_close(checks, param, reference_params[name], name + " updated parameter vs dense", errors, atol=parameter_atol)
        max_parameter_error = max(max_parameter_error, float((param.detach() - reference_params[name].detach()).abs().max().item()))
        compare_states(checks, optimizer.state.get(param, {}), reference_optimizer.state.get(reference_params[name], {}),
                       name + " AdamW state vs dense", errors)
        if param.grad is not None:
            errors.append(f"{name}: step must finish with grad=None")
    for index, (x, expected_x) in enumerate(record["inputs"]):
        checks.close(x.grad, expected_x.grad, f"microbatch {index} LOCAL input gradient", errors)
    if stage.offset is not stage.offset_alias:
        errors.append("Optimizer update broke a tied Parameter")
    checks.agree(errors, label)
    checks.passed(label, {"pattern": pattern, "local_microbatch_sizes": record["sizes"],
        "local_loss": record["local_loss"], "expected_preclip_norm": expected_norm,
        "expected_clip_coefficient": coefficient, "global_routes_per_layer": record["global_counts"].tolist(),
        "inactive_experts_parameter_states": inactive_expert_states,
        "early_only_first_layer_experts": checks.torch.where(early_only)[0].tolist(),
        "max_parameter_abs_error": max_parameter_error})
    reference_optimizer.zero_grad(set_to_none=True)


def anomaly_step(checks, sync, base, utils, info, config, stage, reference,
                 optimizer, reference_optimizer, case, kind, value, event, label):
    checks.phase(label + ": finite accumulation before anomaly injection")
    accumulate(checks, sync, base, config, stage, reference, "rotating", event)
    before_params = {name: param.detach().clone() for name, param in stage.named_parameters()}
    before_states = state_snapshot(optimizer, stage)
    before_state_count = len(optimizer.state)
    if checks.rank == checks.args.ep_size - 1:
        if kind == "shared":
            param = stage.offset
        else:
            _, param = sorted((name, param) for name, param in stage.named_parameters()
                              if ".experts." in name and param.requires_grad)[0]
        grad = checks.torch.zeros_like(param)
        grad.reshape(-1)[0] = value
        param.grad = grad
    calls = []

    def capture(used_optimizer, args, kwargs):
        calls.append(True)

    handle = optimizer.register_step_pre_hook(capture)
    try:
        checks.phase(label + ": global skip")
        result = utils.step_ep_optimizer(stage, optimizer, info, grad_clip=case["clip"])
    finally:
        handle.remove()
    errors = []
    if not isinstance(result, dict) or result.get("stepped") is not False or result.get("grad_norm") is not None:
        errors.append(f"Nonfinite gradients must return stepped=False and grad_norm=None, got {result}")
    if calls:
        errors.append("AdamW.step was called during a global anomaly skip")
    if len(optimizer.state) != before_state_count:
        errors.append("Skipped update created or removed AdamW states")
    for name, param in stage.named_parameters():
        checks.close(param, before_params[name], name + " skipped parameter unchanged", errors, exact=True)
        compare_states(checks, optimizer.state.get(param, {}), before_states[name], name + " skipped state unchanged", errors, exact=True)
        if param.grad is not None:
            errors.append(f"{name}: rejected gradients were not cleared to None")
    checks.agree(errors, label)
    checks.passed(label, {"injected_kind": kind, "injected_rank": checks.args.ep_size - 1})
    reference_optimizer.zero_grad(set_to_none=True)


def check_case(checks, sync, base, modules, info, case):
    utils = modules["parallel_ep"]
    label = f"{case['family']} bias={case['bias']} k={case['k']} grad_clip={case['clip']}"
    checks.phase(label + ": model / optimizer construction")
    errors, built, optimizer, reference_optimizer = [], None, None, None
    try:
        built = sync.build_models(checks, base, modules, info, case)
        config, stage, reference = built
        hyper = hyperparameters()
        optimizer = utils.build_ep_optimizer(model=stage, **hyper)
        reference_optimizer = independent_optimizer(checks.torch, reference, hyper)
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + ": model / optimizer construction")
    check_builder(checks, utils, stage, optimizer, hyper)
    optimizer.zero_grad(set_to_none=True)
    for event, pattern in enumerate(("first", "last", "early")):
        finite_step(checks, sync, base, utils, info, config, stage, reference,
                    optimizer, reference_optimizer, case, pattern, event, label + f": normal step {event}")
    if case["family"] == "MLP" and case["bias"] and case["k"] == 1:
        for event, (kind, value) in enumerate((("shared", math.nan), ("expert", math.inf), ("expert", -math.inf)), start=3):
            anomaly_step(checks, sync, base, utils, info, config, stage, reference, optimizer,
                         reference_optimizer, case, kind, value, event, label + f": global {kind} anomaly {value}")
        finite_step(checks, sync, base, utils, info, config, stage, reference,
                    optimizer, reference_optimizer, case, "rotating", 6, label + ": finite recovery after anomalies")


def worker(rank, args, workdir):
    sync, base, _ = load_helpers()
    base.configure_threads(args.threads)
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
        modules = base.load_project(args.project)
        utils = modules["parallel_ep"]
        checks = base.Checks(torch, dist, args, rank, device, record, rank_path)
        info = utils.setup_ep()
        base.check_groups(checks, utils, info)
        checks.phase("synthetic fixture construction")
        errors, built, optimizer = [], None, None
        try:
            built = sync.build_models(checks, base, modules, info,
                                     {"family": "MLP", "bias": True, "k": 1, "microsteps": 3})
            _, stage, _ = built
            optimizer = utils.build_ep_optimizer(model=stage, **hyperparameters())
        except Exception as exc:
            errors.append(str(exc))
        checks.agree(errors, "synthetic fixture construction")
        check_invalid_arguments(checks, utils, stage, optimizer, info)
        check_synthetic_clipping(checks, utils, stage, info)
        for case in cases(args.ep_size):
            check_case(checks, sync, base, modules, info, case)
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
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": {"project_AdamW_builder": True, "global_norm_and_clipping": True,
                  "shared_parameters_counted_once": True, "expert_owner_contributions_summed": True,
                  "weighted_accumulation": True, "all_AdamW_parameters_moments_steps": True,
                  "unused_parameters_and_inactive_state": True, "global_anomaly_skip_and_recovery": True,
                  "one_EP_group_covers_WORLD": True,
                  "objective": "mean_ranks(sum_microbatches(local_sample_weight * local_loss))",
                  "auxiliary": "weighted local-microbatch auxiliary, not global concatenated routing statistics",
                  "AMP_compile_GPT_training_eval_checkpoint_multinode_extra_expert_replicas": False},
        "tolerance_note": "Router softmax is FP32 even in float64 tests. Parameter absolute tolerance defaults to 2e-6 (float64) or 3e-5 (float32); moments use tighter absolute tolerances. CLI rtol/atol override defaults.",
        "workers": []}
    workdir, exit_code = None, 1
    try:
        sync, base, report["reference_script_sha256"] = load_helpers()
        base.configure_threads(args.threads)
        report["source_sha256"] = preflight(args.project, sync, base)
        print("[PASS] source syntax / EP optimizer helper definitions (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; use the Python environment used for nanoGPT.") from exc
        if args.device == "cuda":
            ensure(torch.cuda.is_available(), "CUDA is not available")
            ensure(torch.cuda.device_count() >= args.ep_size, "CUDA verification requires one visible device per worker")
        report["torch_version"] = torch.__version__
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}; optimizer cases={len(cases(args.ep_size))}.", flush=True)
        print("[INFO] Checking global norm/clipping, AdamW parameters/moments, inactive state, anomaly skips and recovery.", flush=True)
        print("[INFO] Production GPT training, eval and checkpoint are pending.", flush=True)
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_optimizer_")
        mp.spawn(worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        records = [json.loads((Path(workdir) / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(args.ep_size)]
        ensure(all(record["status"] == "passed" for record in records), "A worker did not finish verification")
        report["checks_per_worker"] = len(records[0]["passed"])
        report["status"], exit_code = "passed", 0
        print("[PASS] EP global clipping and AdamW verification completed.", flush=True)
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
