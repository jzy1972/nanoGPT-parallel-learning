#!/usr/bin/env python3
"""Verify EP checkpoint helpers and the real train_ep.main() save/resume path.

Keep verify_ep_gpt.py, verify_ep_training.py, and verify_ep_eval.py beside this
file. Run in nanoGPT:
    python verify_ep_checkpoint.py --project . --device cpu --ep-size 2 --threads 1 --report ep_checkpoint_report.json

No project source is edited. All data/checkpoints are private temporary files.
The original save/load helpers and original main() are executed. A controlled
interruption occurs AFTER a successful checkpoint and BEFORE reading the next
training batch; resume uses a fresh set of spawned processes. The interrupted
run keeps its original target iteration count, avoiding an artificial extra
"last iteration" evaluation at the interruption point.

Checks include helper round-trip, optimizer moments, tied embedding/head,
CPU/CUDA/default/data-generator RNG, nonzero dropout, continued training,
all-rank error propagation, incomplete checkpoints, rank/config/order/progress
checks, main save/eval ordering, best validation loss, iteration/log offsets,
periodic/final saving, increasing max_iters, and already-completed resume.
Both biased sparse top-1 and bias-free full top-k GPTs are tested.

This is a checkpoint consistency test. Independent model/gradient/AdamW
mathematics remain the responsibility of the earlier GPT/training verifiers.
CPU trajectory comparisons are exact; CUDA floating-point comparisons use
tolerances, but RNG states and direct save/load round-trips remain exact.

One EP group covers WORLD and also provides shared-parameter DP. Only main's
rendezvous transport is replaced by FileStore to avoid Windows TCPStore/libuv
and hostname issues. Real Gloo/NCCL collectives, initialization, and cleanup
remain active. Multi-node/env rendezvous, EP resizing, AMP, compile, operating
system crashes, and alternative checkpoint formats are outside scope.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
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


TRAIN_LOG = re.compile(r"^iter (\d+): loss=([^\s,]+), grad_norm=([^\s,]+), stepped=(True|False)\s*$")
EVAL_LOG = re.compile(r"^eval after (\d+) steps: val_loss=([^\s,]+)\s*$")
SAVE_LOG = re.compile(r"^saved checkpoint: (.+)$")


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def load_helpers():
    path = Path(__file__).resolve().with_name("verify_ep_eval.py")
    ensure(path.is_file(), "Keep verify_ep_eval.py beside verify_ep_checkpoint.py")
    spec = importlib.util.spec_from_file_location("_ep_checkpoint_eval_helpers", path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    support = evaluation.load_support()
    base = support.load_base()
    return evaluation, support, base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EP checkpoint helper and actual-main verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=120)
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


def preflight(evaluation, support, base, project):
    hashes = evaluation.preflight(support, base, project)
    path = project / "checkpoint_ep.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    compile(tree, str(path), "exec")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {"capture_ep_rng_state", "restore_ep_rng_state", "check_ep_checkpoint_errors",
                "save_ep_checkpoint", "load_ep_checkpoint"}
    ensure(required <= functions.keys(), f"checkpoint_ep.py needs: {sorted(required - functions.keys())}")
    common = {"model", "optimizer", "parallel", "checkpoint_dir",
              "train_data_generator", "eval_data_generator", "device", "train_args"}
    for name, expected in (("save_ep_checkpoint", common | {"completed_iters", "best_val_loss"}),
                           ("load_ep_checkpoint", common)):
        args = functions[name].args
        names = {item.arg for item in args.posonlyargs + args.args + args.kwonlyargs}
        ensure(expected <= names or args.kwarg is not None, f"{name} must accept {sorted(expected)}")
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_project(support, base, project):
    sys.modules.pop("checkpoint_ep", None)
    modules = support.load_project(base, project)
    checkpoint = importlib.import_module("checkpoint_ep")
    ensure(Path(checkpoint.__file__).resolve() == (project / "checkpoint_ep.py").resolve(),
           "Imported checkpoint_ep.py from the wrong directory")
    trainer = modules["train_ep"]
    for name in ("save_ep_checkpoint", "load_ep_checkpoint"):
        ensure(getattr(trainer, name, None) is getattr(checkpoint, name),
               f"train_ep.py must import {name} from checkpoint_ep.py")
    modules["checkpoint_ep"] = checkpoint
    return modules


def runtime(args, rank):
    evaluation, support, _ = load_helpers()
    torch, dist, device, base = support.init_worker_runtime(args, rank)
    return torch, dist, device, evaluation, support, base


def bind(function, positional, keywords):
    bound = inspect.signature(function).bind(*positional, **keywords)
    bound.apply_defaults()
    return bound.arguments


def native_rng(torch, device, train_generator, eval_generator):
    return {"torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
            "train_data": train_generator.get_state().clone(),
            "eval_data": eval_generator.get_state().clone() if eval_generator is not None else None}


def optimizer_names(model, optimizer):
    names = {id(param): name for name, param in model.named_parameters()}
    return [[names[id(param)] for param in group["params"]] for group in optimizer.param_groups]


def capture(torch, evaluation, model, optimizer, device, train_generator, eval_generator, best):
    return evaluation.cpu_copy({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_param_names": optimizer_names(model, optimizer),
        "rng": native_rng(torch, device, train_generator, eval_generator),
        "grad_is_none": {name: param.grad is None for name, param in model.named_parameters()},
        "tied_head": model.transformer.wte.weight is model.lm_head.weight,
        "modes": {name: module.training for name, module in model.named_modules()},
        "model_args": dataclasses.asdict(model.config), "best_val_loss": float(best),
    })


def compare_nested(torch, args, actual, expected, errors, path="state", exact=True):
    try:
        if isinstance(expected, torch.Tensor):
            ensure(isinstance(actual, torch.Tensor), "expected a Tensor")
            strict = exact or not expected.is_floating_point() or path.endswith("/step") or "/rng/" in path
            rtol = 0 if strict else args.rtol if args.rtol is not None else (3e-5 if args.dtype == "float64" else 3e-4)
            if strict:
                atol = 0
            elif args.atol is not None:
                atol = args.atol
            elif path.endswith("/exp_avg_sq"):
                atol = 1e-13 if args.dtype == "float64" else 1e-10
            elif path.endswith("/exp_avg"):
                atol = 1e-10 if args.dtype == "float64" else 1e-7
            else:
                atol = 2e-6 if args.dtype == "float64" else 3e-5
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        elif isinstance(expected, dict):
            ensure(isinstance(actual, dict) and actual.keys() == expected.keys(), "dictionary keys differ")
            for key, value in expected.items():
                compare_nested(torch, args, actual[key], value, errors, path + "/" + str(key), exact)
        elif isinstance(expected, (tuple, list)):
            ensure(isinstance(actual, type(expected)) and len(actual) == len(expected), "sequence type/length differs")
            for index, value in enumerate(expected):
                compare_nested(torch, args, actual[index], value, errors, path + "/" + str(index), exact)
        elif isinstance(expected, float):
            if exact or not math.isfinite(expected):
                ensure(actual == expected, f"expected {expected}, got {actual}")
            else:
                ensure(math.isclose(float(actual), expected,
                    rel_tol=args.rtol if args.rtol is not None else (3e-5 if args.dtype == "float64" else 3e-4),
                    abs_tol=args.atol if args.atol is not None else (3e-8 if args.dtype == "float64" else 3e-5)),
                    f"expected {expected}, got {actual}")
        else:
            ensure(type(actual) is type(expected) and actual == expected, f"expected {expected!r}, got {actual!r}")
    except Exception as exc:
        errors.append(f"{path}: {exc}")


def assert_same(checks, actual, expected, label, exact=True):
    errors = []
    compare_nested(checks.torch, checks.args, actual, expected, errors, label, exact)
    checks.agree(errors, label)


def check_payload(torch, args, payload, manifest, state, parallel, completed, train_args, errors):
    expected = {"format_version": 1, "parallelism": "ep_shared_dp",
                "world_size": parallel["world_size"], "ep_size": parallel["ep_size"],
                "completed_iters": completed, "model_args": state["model_args"],
                "dtype": str(next(iter(state["model"].values())).dtype), "device_type": args.device}
    for key, value in expected.items():
        if payload.get(key) != value or manifest.get(key) != value:
            errors.append(f"checkpoint common field {key} differs")
    for key, value in (("world_rank", parallel["world_rank"]), ("ep_rank", parallel["ep_rank"])):
        if payload.get(key) != value:
            errors.append(f"wrong {key} in rank checkpoint")
    for key in ("model", "optimizer", "optimizer_param_names", "rng", "best_val_loss"):
        compare_nested(torch, args, payload.get(key), state[key], errors, "payload/" + key, exact=True)
    compare_nested(torch, args, payload.get("train_args"), dict(train_args), errors, "payload/train_args", exact=True)
    files = [f"rank_{rank:05d}.pt" for rank in range(parallel["world_size"])]
    if manifest.get("rank_files") != files:
        errors.append("manifest rank_files differs")


def read_checkpoint(torch, directory, rank):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(directory / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
    return manifest, payload


def main_cases(size):
    return [
        {"name": "sparse_dropout", "bias": True, "k": 1, "coef": 0.17, "dropout": 0.25,
         "batch": 7, "micro": 3, "sequence": 7, "clip": 0.01, "iters": 5,
         "interrupted_target": 4, "cut": 2, "interval": 1, "eval_interval": 2, "eval_iters": 2, "save_interval": 2},
        {"name": "full_topk_no_eval", "bias": False, "k": 2 * size, "coef": 0.0, "dropout": 0.25,
         "batch": 3, "micro": 8, "sequence": 1, "clip": 0.0, "iters": 3,
         "interrupted_target": 3, "cut": 2, "interval": 2, "eval_interval": 0, "eval_iters": 1, "save_interval": 2},
    ]


def period_schedule(total, interval):
    return [] if interval == 0 else [step for step in range(1, total + 1) if step % interval == 0 or step == total]


def phase_target(case, phase):
    return case["interrupted_target"] if phase == "interrupted" else case["cut"] if phase == "done" else case["iters"]


def phase_schedule(case, phase, kind):
    start = case["cut"] if phase in ("resume", "done") else 0
    limit = case["cut"] if phase == "interrupted" else phase_target(case, phase)
    return [step for step in period_schedule(phase_target(case, phase), case[kind + "_interval"]) if start < step <= limit]


def report_number(value):
    return float(value) if math.isfinite(float(value)) else str(value)


def expect_failure(checks, evaluation, function, model, optimizer, device, train_gen, eval_gen,
                   best, label, unchanged=True):
    checks.phase(label)
    before = capture(checks.torch, evaluation, model, optimizer, device, train_gen, eval_gen, best)
    caught, errors = None, []
    try:
        function()
    except Exception as exc:
        caught = exc
    if not isinstance(caught, RuntimeError):
        errors.append(f"Expected a collective RuntimeError on every rank, got {caught!r}")
    if unchanged:
        after = capture(checks.torch, evaluation, model, optimizer, device, train_gen, eval_gen, best)
        compare_nested(checks.torch, checks.args, after, before, errors, label + "/unchanged", exact=True)
    checks.agree(errors, label)
    checks.passed(label)


def fixture_variant(checks, source, destination, mutation):
    errors = []
    if checks.rank == 0:
        try:
            shutil.copytree(source, destination)
            mutation(Path(destination))
        except Exception:
            errors.append(traceback.format_exc())
    checks.agree(errors, "prepare invalid-checkpoint fixture: " + Path(destination).name)


def failure_cases(checks, evaluation, checkpoint, kwargs, best, source, root):
    torch, args = checks.torch, checks.args
    model, optimizer = kwargs["model"], kwargs["optimizer"]
    train_gen, eval_gen, device = kwargs["train_data_generator"], kwargs["eval_data_generator"], kwargs["device"]
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in Path(source).iterdir() if file.is_file()}

    def bad_save(directory):
        return checkpoint.save_ep_checkpoint(**dict(kwargs, checkpoint_dir=str(directory),
            completed_iters=4, best_val_loss=best))

    expect_failure(checks, evaluation, lambda: bad_save(source), model, optimizer, device, train_gen,
                   eval_gen, best, "completed checkpoint is not overwritten")
    after = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in Path(source).iterdir() if file.is_file()}
    checks.agree([] if hashes == after else ["Completed checkpoint bytes changed"], "overwrite refusal preserves files")

    dirty_dir = root / "dirty_gradients"
    if checks.rank == args.ep_size - 1:
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    expect_failure(checks, evaluation, lambda: bad_save(dirty_dir), model, optimizer, device, train_gen,
                   eval_gen, best, "pending gradients on one rank reject saving collectively")
    model.zero_grad(set_to_none=True)
    checks.agree([] if not (dirty_dir / "manifest.json").exists() else ["Incomplete checkpoint has a manifest"], "dirty-gradient save has no complete marker")

    failed_dir = root / "write_failure"
    original_save = torch.save

    def rejected_save(*positional, **keywords):
        raise OSError("Verifier injected a rank-file write failure")

    if checks.rank == args.ep_size - 1:
        torch.save = rejected_save
    try:
        expect_failure(checks, evaluation, lambda: bad_save(failed_dir), model, optimizer, device, train_gen,
                       eval_gen, best, "rank-file write failure propagates to all ranks")
    finally:
        torch.save = original_save
    checks.agree([] if not (failed_dir / "manifest.json").exists() else ["Failed save was marked complete"], "write failure has no complete marker")

    manifest_dir = root / "manifest_failure"
    original_replace = os.replace

    def rejected_manifest(source_path, destination_path):
        if Path(destination_path).name == "manifest.json":
            raise OSError("Verifier injected a manifest rename failure")
        return original_replace(source_path, destination_path)

    if checks.rank == 0:
        os.replace = rejected_manifest
    try:
        expect_failure(checks, evaluation, lambda: bad_save(manifest_dir), model, optimizer, device, train_gen,
                       eval_gen, best, "manifest write failure propagates to all ranks")
    finally:
        os.replace = original_replace
    checks.agree([] if not (manifest_dir / "manifest.json").exists() else ["Failed manifest was committed"], "manifest failure has no complete marker")

    def rewrite(directory, modify):
        path = directory / f"rank_{args.ep_size - 1:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        modify(payload)
        torch.save(payload, path)

    variants = [
        ("missing_manifest", lambda folder: (folder / "manifest.json").unlink()),
        ("missing_rank_file", lambda folder: (folder / f"rank_{args.ep_size - 1:05d}.pt").unlink()),
        ("wrong_rank", lambda folder: rewrite(folder, lambda payload: payload.update(world_rank=args.ep_size))),
        ("wrong_size", lambda folder: rewrite(folder, lambda payload: payload.update(ep_size=args.ep_size + 1))),
        ("wrong_dtype", lambda folder: rewrite(folder, lambda payload: payload.update(dtype="torch.float16"))),
        ("wrong_progress", lambda folder: rewrite(folder, lambda payload: payload.update(completed_iters=3))),
        ("wrong_optimizer_order", lambda folder: rewrite(folder, lambda payload: payload["optimizer_param_names"][0].__setitem__(0, "wrong.parameter"))),
    ]
    if args.ep_size > 1:
        variants.append(("inconsistent_best_loss", lambda folder: rewrite(folder,
            lambda payload: payload.update(best_val_loss=float(payload["best_val_loss"]) + 0.5))))
    for name, mutation in variants:
        directory = root / name
        fixture_variant(checks, source, directory, mutation)
        expect_failure(checks, evaluation,
            lambda directory=directory: checkpoint.load_ep_checkpoint(**dict(kwargs, checkpoint_dir=str(directory))),
            model, optimizer, device, train_gen, eval_gen, best, name + " rejects loading collectively")

    changed_args = dict(kwargs["train_args"])
    if checks.rank == args.ep_size - 1:
        changed_args["grad_clip"] += 0.5
    expect_failure(checks, evaluation,
        lambda: checkpoint.load_ep_checkpoint(**dict(kwargs, train_args=changed_args)),
        model, optimizer, device, train_gen, eval_gen, best, "changed training configuration rejects loading collectively")

    directory = root / "invalid_rng"
    fixture_variant(checks, source, directory, lambda folder: rewrite(folder,
        lambda payload: payload["rng"].update(train_data=torch.zeros(1, dtype=torch.uint8))))
    expect_failure(checks, evaluation,
        lambda: checkpoint.load_ep_checkpoint(**dict(kwargs, checkpoint_dir=str(directory))),
        model, optimizer, device, train_gen, eval_gen, best, "invalid RNG restoration fails collectively",
        unchanged=False)  # A failed restore aborts; transactional rollback is outside this format.
    checks.agree([] if hashes == {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in Path(source).iterdir() if file.is_file()} else ["Valid fixture changed"], "valid checkpoint remains unchanged")


def direct_step(checks, evaluation, support, trainer, kwargs, completed, best):
    import numpy as np
    torch = checks.torch
    train_args = kwargs["train_args"]
    train_gen, eval_gen = kwargs["train_data_generator"], kwargs["eval_data_generator"]
    expected_gen = torch.Generator(device="cpu")
    expected_gen.set_state(train_gen.get_state())
    expected_x, expected_y, starts = support.expected_batch(torch, np,
        Path(train_args["data_dir"]) / "train.bin", train_args["batch_size"], train_args["seq_length"],
        expected_gen, checks.device)
    ids, targets = trainer.get_ep_batch(data_path=str(Path(train_args["data_dir"]) / "train.bin"),
        batch_size=train_args["batch_size"], seq_length=train_args["seq_length"], device=checks.device, generator=train_gen)
    errors = []
    base = support.load_base()
    base.tensor_close(checks, ids, expected_x, "direct training sampled inputs", errors, exact=True)
    base.tensor_close(checks, targets, expected_y, "direct training shifted targets", errors, exact=True)
    base.tensor_close(checks, train_gen.get_state(), expected_gen.get_state(), "direct Generator advance", errors, exact=True)
    checks.agree(errors, f"direct helper batch {completed}")
    result = trainer.train_ep_step(model=kwargs["model"], optimizer=kwargs["optimizer"], parallel=kwargs["parallel"],
        token_ids=ids, targets=targets, micro_batch_size=train_args["micro_batch_size"], grad_clip=train_args["grad_clip"])
    errors = []
    ensure(isinstance(result, dict), "train_ep_step must return a dict")
    if result.get("stepped") is not True or not math.isfinite(float(result.get("loss", float("nan")))):
        errors.append(f"Finite warm-up/continuation must update successfully: {result}")
    checks.agree(errors, f"direct helper update {completed}")
    val_loss = None
    if completed in period_schedule(train_args["max_iters"], train_args["eval_interval"]):
        preserved = evaluation.snapshot(base, checks, kwargs["model"], kwargs["optimizer"], train_gen)
        val_loss = trainer.evaluate_ep(model=kwargs["model"], parallel=kwargs["parallel"],
            data_path=str(Path(train_args["data_dir"]) / "val.bin"), batch_size=train_args["batch_size"],
            seq_length=train_args["seq_length"], eval_iters=train_args["eval_iters"], device=checks.device, generator=eval_gen)
        errors = []
        evaluation.check_preserved(base, checks, kwargs["model"], kwargs["optimizer"], train_gen, preserved, errors)
        if not math.isfinite(float(val_loss)):
            errors.append("Validation returned a nonfinite loss")
        checks.agree(errors, f"direct helper evaluation {completed}")
        best = min(best, float(val_loss))
    state = capture(torch, evaluation, kwargs["model"], kwargs["optimizer"], checks.device, train_gen, eval_gen, best)
    checks.agree([] if all(state["grad_is_none"].values()) and state["tied_head"] else
                 ["Gradients were not cleared or the tied head was broken"], f"direct update boundary {completed}")
    return {"completed_iters": completed, "starts": starts, "result": dict(result),
            "val_loss": val_loss, "state": state}, best


def helper_roundtrip(checks, evaluation, support, base, modules, parallel, root, index):
    torch = checks.torch
    case = {"use_moe": True, "bias": index == 0, "k": 1 if index == 0 else 2 * checks.args.ep_size, "coef": 0.17}
    label = f"helper-roundtrip bias={case['bias']} k={case['k']} dropout=0.25"
    checks.phase(label + ": constructors")
    cfg, model, prototype = base.construct(checks, modules, parallel, case)
    base.check_structure_and_load(checks, modules, parallel, cfg, model, prototype, label)
    cfg.dropout = 0.25
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.25
    for block in model.transformer.h:
        block.attn.dropout = 0.25
    train_args = {"seed": checks.args.seed, "data_dir": str(root / "data"), "batch_size": 5,
        "micro_batch_size": 2, "seq_length": 7, "learning_rate": 7e-4, "weight_decay": 0.13,
        "beta1": 0.8, "beta2": 0.93, "grad_clip": 0.01 if index == 0 else 0.0,
        "eval_interval": 2, "eval_iters": 2, "max_iters": 4}
    optimizer = modules["parallel_ep"].build_ep_optimizer(model=model, learning_rate=7e-4,
        weight_decay=0.13, beta1=0.8, beta2=0.93)
    optimizer.zero_grad(set_to_none=True)
    train_gen = torch.Generator(device="cpu").manual_seed(checks.args.seed + 10_000 + checks.rank)
    eval_gen = torch.Generator(device="cpu").manual_seed(checks.args.seed + 20_000 + checks.rank)
    directory = root / "functions" / f"valid_{index}"
    kwargs = dict(model=model, optimizer=optimizer, parallel=parallel, checkpoint_dir=str(directory),
        train_data_generator=train_gen, eval_data_generator=eval_gen, device=checks.device, train_args=train_args)
    best = float("inf")
    for completed in (1, 2):
        _, best = direct_step(checks, evaluation, support, modules["train_ep"], kwargs, completed, best)
    before = capture(torch, evaluation, model, optimizer, checks.device, train_gen, eval_gen, best)
    checks.phase(label + ": native save")
    returned = modules["checkpoint_ep"].save_ep_checkpoint(**kwargs, completed_iters=2, best_val_loss=best)
    errors = []
    ensure(Path(returned).resolve() == directory.resolve(), "save_ep_checkpoint returned the wrong directory")
    manifest, payload = read_checkpoint(torch, directory, checks.rank)
    check_payload(torch, checks.args, payload, manifest, before, parallel, 2, train_args, errors)
    for filename in manifest["rank_files"]:
        if not (directory / filename).is_file():
            errors.append("Missing rank file: " + filename)
    checks.agree(errors, label + ": rank payload / manifest / optimizer names / RNG")
    assert_same(checks, capture(torch, evaluation, model, optimizer, checks.device, train_gen, eval_gen, best),
                before, label + ": saving preserves training state")
    baseline = []
    for completed in (3, 4):
        event, best = direct_step(checks, evaluation, support, modules["train_ep"], kwargs, completed, best)
        baseline.append(event)

    # Recreate and deliberately reseed everything before restoring.
    torch.manual_seed(checks.args.seed + 900 + checks.rank)
    restored_model = modules["model"].GPT(cfg, ep_group=parallel["ep_group"]).to(
        device=checks.device, dtype=getattr(torch, checks.args.dtype))
    modules["parallel_ep"].sync_shared_parameters(restored_model, parallel)
    restored_optimizer = modules["parallel_ep"].build_ep_optimizer(model=restored_model, learning_rate=7e-4,
        weight_decay=0.13, beta1=0.8, beta2=0.93)
    restored_train = torch.Generator(device="cpu").manual_seed(81 + checks.rank)
    restored_eval = torch.Generator(device="cpu").manual_seed(91 + checks.rank)
    loaded_kwargs = dict(kwargs, model=restored_model, optimizer=restored_optimizer,
        train_data_generator=restored_train, eval_data_generator=restored_eval,
        train_args=dict(train_args, max_iters=6))  # max_iters may increase.
    checks.phase(label + ": native load")
    restored = modules["checkpoint_ep"].load_ep_checkpoint(**loaded_kwargs)
    checks.agree([] if restored == {"completed_iters": 2, "best_val_loss": before["best_val_loss"]}
                 else [f"Wrong load result: {restored}"], label + ": progress and best loss")
    assert_same(checks, capture(torch, evaluation, restored_model, restored_optimizer, checks.device,
        restored_train, restored_eval, restored["best_val_loss"]), before, label + ": exact state/RNG round-trip")
    best_restored = restored["best_val_loss"]
    # Keep the original comparison horizon/cadence for the two continuation steps.
    loaded_kwargs["train_args"] = train_args
    for completed, expected in zip((3, 4), baseline):
        actual, best_restored = direct_step(checks, evaluation, support, modules["train_ep"],
            loaded_kwargs, completed, best_restored)
        assert_same(checks, actual, expected, label + f": continuation {completed}",
                    exact=checks.args.device == "cpu")
    checks.passed(label + ": native round-trip and continued training",
                  {"completed_iters": 2, "best_val_loss": report_number(before["best_val_loss"])})
    if index == 0:
        failure_cases(checks, evaluation, modules["checkpoint_ep"], loaded_kwargs, best_restored,
                      directory, root / "functions")


def helper_worker(rank, args, workdir):
    _, _, base = load_helpers()
    base.configure_threads(args.threads)
    path = Path(workdir) / "functions" / f"rank_{rank}.json"
    record = {"rank": rank, "status": "running", "active_phase": "initialization", "passed": []}
    base.write_json(path, record)
    dist = None
    try:
        torch, dist, device, evaluation, support, base = runtime(args, rank)
        support.file_init(dist, args, rank, Path(workdir) / "functions" / "rendezvous.store")
        checks = base.Checks(torch, dist, args, rank, device, record, path)
        modules = load_project(support, base, args.project)
        parallel = modules["parallel_ep"].setup_ep()
        base.check_groups(checks, modules["parallel_ep"], parallel)
        case = main_cases(args.ep_size)[0]
        argv = support.trainer_argv(args, Path(workdir) / "data", case)
        namespace = support.invoke_parser(modules["train_ep"], argv)
        checks.agree([] if namespace.save_interval == 0 and namespace.resume is None
                     else ["save_interval must default to 0 and resume to None"], "checkpoint CLI defaults")
        selected = support.invoke_parser(modules["train_ep"], argv + [
            "--save-interval", "2", "--out-dir", str(Path(workdir) / "selected-output"),
            "--resume", str(Path(workdir) / "selected-input")])
        errors = []
        if (selected.save_interval != 2
                or Path(selected.out_dir) != Path(workdir) / "selected-output"
                or Path(selected.resume) != Path(workdir) / "selected-input"):
            errors.append("Checkpoint CLI values were ignored")
        try:
            support.invoke_parser(modules["train_ep"], argv + ["--save-interval", "-1"])
        except (ValueError, AssertionError, SystemExit):
            pass
        else:
            errors.append("A negative save_interval was accepted")
        checks.agree(errors, "checkpoint CLI values and validation")
        for index in (0, 1):
            helper_roundtrip(checks, evaluation, support, base, modules, parallel, Path(workdir), index)
        dist.barrier()
        record.update(status="passed", active_phase="complete", torch_version=torch.__version__)
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc())
        print(f"[FAIL rank={rank}] {record['active_phase']}", flush=True)
        raise
    finally:
        base.write_json(path, record)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


class StopAfterCheckpoint(RuntimeError):
    pass


def check_objects(checks, support, base, model, optimizer, parallel, case):
    errors = []
    try:
        cfg = model.config
        expected = {"n_layer": 2, "n_head": 3, "n_embd": 12, "block_size": 7, "vocab_size": 31,
                    "dropout": case["dropout"], "bias": case["bias"], "use_moe": True,
                    "n_experts": 2 * checks.args.ep_size, "moe_top_k": case["k"], "moe_aux_loss_coef": case["coef"]}
        for name, value in expected.items():
            ensure(getattr(cfg, name) == value, f"main ignored model config {name}={value}")
        ensure(model.transformer.wte.weight is model.lm_head.weight, "Embedding/head must share one Parameter")
        shapes = support.expected_parameter_shapes(cfg)
        start, end = 2 * checks.rank, 2 * (checks.rank + 1)
        expected_names = {name for name in shapes if base.expert_key(name) is None or start <= base.expert_key(name)[1] < end}
        named = dict(model.named_parameters())
        ensure(set(named) == expected_names, "Wrong global expert IDs or shared parameter ownership")
        for name, param in named.items():
            ensure(tuple(param.shape) == shapes[name] and param.device == checks.device
                   and param.dtype == getattr(checks.torch, checks.args.dtype), "Wrong shape/device/dtype: " + name)
        support.check_optimizer(checks, model, optimizer, errors)
    except Exception:
        errors.append(traceback.format_exc())
    checks.agree(errors, "actual main model / optimizer / expert ownership")
    base.check_groups(checks, sys.modules["parallel_ep"], parallel)


def check_main_logs(text, case, phase, steps, evaluations, saves, out_dir):
    start = case["cut"] if phase in ("resume", "done") else 0
    end = case["cut"] if phase == "interrupted" else phase_target(case, phase)
    target = phase_target(case, phase)
    expected_train = [index for index in range(start, end) if index % case["interval"] == 0 or index == target - 1]
    train_matches, eval_matches, save_matches = [], [], []
    for line in text.splitlines():
        if line.startswith("iter "):
            match = TRAIN_LOG.fullmatch(line)
            ensure(match is not None, "Malformed training log: " + line)
            train_matches.append(match)
        elif line.startswith("eval after "):
            match = EVAL_LOG.fullmatch(line)
            ensure(match is not None, "Malformed evaluation log: " + line)
            eval_matches.append(match)
        elif line.startswith("saved checkpoint: "):
            match = SAVE_LOG.fullmatch(line)
            ensure(match is not None, "Malformed checkpoint log: " + line)
            save_matches.append(match)
    ensure([int(match.group(1)) for match in train_matches] == expected_train,
           f"Wrong train iteration logs; expected {expected_train}")
    lookup = {entry["iter_num"]: entry["result"] for entry in steps}
    for match in train_matches:
        actual = lookup[int(match.group(1))]
        ensure(math.isclose(float(match.group(2)), actual["loss"], rel_tol=0, abs_tol=1.1e-6), "Printed training loss differs")
        if actual["grad_norm"] is None:
            ensure(match.group(3) == "n/a", "Wrong skipped norm log")
        else:
            ensure(math.isclose(float(match.group(3)), actual["grad_norm"], rel_tol=0, abs_tol=1.1e-6), "Printed grad norm differs")
        ensure((match.group(4) == "True") == actual["stepped"], "Wrong stepped log")
    expected_eval = phase_schedule(case, phase, "eval")
    ensure([int(match.group(1)) for match in eval_matches] == expected_eval, f"Wrong evaluation logs; expected {expected_eval}")
    for match, entry in zip(eval_matches, evaluations):
        ensure(math.isclose(float(match.group(2)), entry["loss"], rel_tol=0, abs_tol=1.1e-6), "Printed validation loss differs")
    expected_saves = phase_schedule(case, phase, "save")
    actual_paths = [Path(match.group(1)).resolve() for match in save_matches]
    ensure(actual_paths == [(out_dir / f"iter_{step:08d}").resolve() for step in expected_saves], "Wrong checkpoint path/iteration logs")
    resumed = [line for line in text.splitlines() if line.startswith("resumed checkpoint: ")]
    ensure(len(resumed) == int(phase in ("resume", "done")), "Wrong number of resume logs")
    if resumed:
        ensure(f"completed_iters={case['cut']}" in resumed[0], "Resume log has the wrong iteration offset")
    return {"train": expected_train, "eval": expected_eval, "save": expected_saves}


def main_worker(rank, args, workdir, case, phase):
    _, _, base = load_helpers()
    base.configure_threads(args.threads)
    directory = Path(workdir) / case["name"] / phase
    path = directory / f"rank_{rank}.json"
    record = {"rank": rank, "status": "running", "active_phase": "initialization", "passed": [],
              "case": case, "run_phase": phase}
    base.write_json(path, record)
    output = io.StringIO()
    previous_argv = sys.argv
    dist, original_init, original_destroy = None, None, None
    try:
        torch, dist, device, evaluation, support, base = runtime(args, rank)
        import numpy as np
        checks = base.Checks(torch, dist, args, rank, device, record, path)
        modules = load_project(support, base, args.project)
        trainer = modules["train_ep"]
        original_init, original_destroy = dist.init_process_group, dist.destroy_process_group
        original_batch, original_step = trainer.get_ep_batch, trainer.train_ep_step
        original_eval, original_save, original_load = trainer.evaluate_ep, trainer.save_ep_checkpoint, trainer.load_ep_checkpoint
        data_dir, out_dir = Path(workdir) / "data", directory / "checkpoints"
        resume_dir = Path(workdir) / case["name"] / "interrupted" / "checkpoints" / f"iter_{case['cut']:08d}"
        start = case["cut"] if phase in ("resume", "done") else 0
        limit = case["cut"] if phase == "interrupted" else phase_target(case, phase)
        expected_generators = {stream: torch.Generator(device="cpu").manual_seed(args.seed + offset + rank)
                               for stream, offset in (("train", 10_000), ("val", 20_000))}
        objects, data_calls, last_batch = {}, {"train": [], "val": []}, {}
        steps, evaluations, saves = [], [], []
        init_calls, destroy_calls, load_calls = [], [], []
        in_evaluation, stop_ready, stopped, inspected = False, False, False, False
        best = float("inf")
        loaded_snapshot = None

        def state_now():
            return capture(torch, evaluation, objects["model"], objects["optimizer"], device,
                           objects["train_gen"], objects.get("eval_gen"), best)

        def register_generator(stream, generator):
            key = "train_gen" if stream == "train" else "eval_gen"
            if key in objects:
                ensure(objects[key] is generator, f"main replaced the {stream} Generator")
            objects[key] = generator
            if "train_gen" in objects and "eval_gen" in objects:
                ensure(objects["train_gen"] is not objects["eval_gen"], "Training/evaluation Generators must be distinct")

        @functools.wraps(original_init)
        def observed_init(*positional, **keywords):
            values = bind(original_init, positional, keywords)
            backend = values.get("backend")
            ensure(str(backend).lower() == ("nccl" if args.device == "cuda" else "gloo"), "Wrong backend")
            ensure(not dist.is_initialized(), "main initialized twice")
            init_calls.append(str(backend))
            store = dist.FileStore(str(directory / "rendezvous.store"), args.ep_size)
            store.set_timeout(timedelta(seconds=args.timeout))
            forwarded = dict(backend=backend, store=store, rank=rank, world_size=args.ep_size,
                             timeout=timedelta(seconds=args.timeout))
            if values.get("device_id") is not None:
                forwarded["device_id"] = values["device_id"]
            return original_init(**forwarded)

        @functools.wraps(original_destroy)
        def observed_destroy(*positional, **keywords):
            destroy_calls.append(True)
            return original_destroy(*positional, **keywords)

        @functools.wraps(original_batch)
        def observed_batch(*positional, **keywords):
            if stop_ready and not in_evaluation:
                raise StopAfterCheckpoint("Controlled interruption before reading the next training batch")
            values = bind(original_batch, positional, keywords)
            stream = "val" if in_evaluation else "train"
            result = original_batch(*positional, **keywords)
            errors = []
            try:
                ensure(phase not in ("resume", "done") or len(load_calls) == 1, "main sampled before restoring")
                ensure(Path(values["data_path"]).resolve() == (data_dir / (stream + ".bin")).resolve(), "Wrong data path")
                ensure(values["batch_size"] == case["batch"] and values["seq_length"] == case["sequence"], "Wrong batch/sequence arguments")
                register_generator(stream, values["generator"])
                expected_x, expected_y, starts = support.expected_batch(torch, np, data_dir / (stream + ".bin"),
                    case["batch"], case["sequence"], expected_generators[stream], device)
                base.tensor_close(checks, result[0], expected_x, stream + " inputs", errors, exact=True)
                base.tensor_close(checks, result[1], expected_y, stream + " targets", errors, exact=True)
                base.tensor_close(checks, values["generator"].get_state(), expected_generators[stream].get_state(),
                                  stream + " sampling state", errors, exact=True)
                data_calls[stream].append(starts)
                last_batch[stream] = (result[0].detach().clone(), result[1].detach().clone())
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + "/" + phase + ": " + stream + " batch")
            return result

        @functools.wraps(original_step)
        def observed_step(*positional, **keywords):
            nonlocal inspected
            values = bind(original_step, positional, keywords)
            errors = []
            try:
                ensure(not in_evaluation and len(data_calls["train"]) == len(steps) + 1, "Wrong training data/update order")
                ensure(values["micro_batch_size"] == case["micro"] and float(values["grad_clip"]) == case["clip"], "Wrong microbatch/clip arguments")
                for key, value in (("model", values["model"]), ("optimizer", values["optimizer"])):
                    if key in objects:
                        ensure(objects[key] is value, "main replaced " + key)
                    objects[key] = value
                for key, actual, expected in (("inputs", values["token_ids"], last_batch["train"][0]),
                                              ("targets", values["targets"], last_batch["train"][1])):
                    base.tensor_close(checks, actual, expected, "training " + key, errors, exact=True)
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, "main training-step arguments")
            if not inspected:
                check_objects(checks, support, base, values["model"], values["optimizer"], values["parallel"], case)
                inspected = True
            result = original_step(*positional, **keywords)
            errors = []
            ensure(isinstance(result, dict), "train_ep_step must return a dict")
            if result.get("stepped") is not True or not math.isfinite(float(result.get("loss", float("nan")))):
                errors.append(f"Finite main update failed: {result}")
            checks.agree(errors, "finite actual-main update")
            item = {"iter_num": start + len(steps), "completed_iters": start + len(steps) + 1,
                    "result": {"loss": float(result["loss"]), "grad_norm": None if result["grad_norm"] is None else float(result["grad_norm"]),
                               "stepped": result["stepped"]}, "state": state_now(),
                    "starts": data_calls["train"][-1]}
            ensure(all(item["state"]["grad_is_none"].values()), "main update left gradients uncleared")
            steps.append(item)
            return result

        @functools.wraps(original_eval)
        def observed_eval(*positional, **keywords):
            nonlocal in_evaluation, best
            values = bind(original_eval, positional, keywords)
            completed = start + len(steps)
            errors = []
            try:
                schedule = phase_schedule(case, phase, "eval")
                ensure(len(evaluations) < len(schedule) and completed == schedule[len(evaluations)], "Wrong evaluation position")
                ensure(values["model"] is objects["model"] and values["eval_iters"] == case["eval_iters"], "Wrong evaluation arguments")
                register_generator("val", values["generator"])
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, "main evaluation schedule")
            preserved = evaluation.snapshot(base, checks, objects["model"], objects["optimizer"], objects["train_gen"])
            in_evaluation = True
            try:
                result = original_eval(*positional, **keywords)
            finally:
                in_evaluation = False
            errors = []
            evaluation.check_preserved(base, checks, objects["model"], objects["optimizer"], objects["train_gen"], preserved, errors)
            if not math.isfinite(float(result)):
                errors.append("main evaluation returned a nonfinite loss")
            checks.agree(errors, "main evaluation preserves training state")
            best = min(best, float(result))
            evaluations.append({"completed_iters": completed, "loss": float(result), "state": state_now()})
            return result

        @functools.wraps(original_save)
        def observed_save(*positional, **keywords):
            nonlocal stop_ready
            values = bind(original_save, positional, keywords)
            completed = start + len(steps)
            errors = []
            try:
                schedule = phase_schedule(case, phase, "save")
                ensure(len(saves) < len(schedule) and completed == schedule[len(saves)], "Wrong save schedule")
                ensure(values["completed_iters"] == completed, "Save used iter_num instead of completed_iters")
                ensure(values["model"] is objects["model"] and values["optimizer"] is objects["optimizer"], "Wrong save objects")
                register_generator("train", values["train_data_generator"])
                register_generator("val", values["eval_data_generator"])
                ensure(float(values["best_val_loss"]) == best, "main did not update best_val_loss on every rank")
                ensure(len(evaluations) == sum(step <= completed for step in phase_schedule(case, phase, "eval")),
                       "Checkpoint must be saved after the scheduled evaluation")
                expected_dir = out_dir / f"iter_{completed:08d}"
                ensure(Path(values["checkpoint_dir"]).resolve() == expected_dir.resolve(), "Wrong checkpoint directory counter")
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, "main save schedule / evaluation-before-save / best loss")
            before = state_now()
            result = original_save(*positional, **keywords)
            assert_same(checks, state_now(), before, "actual-main save preserves state")
            manifest, payload = read_checkpoint(torch, expected_dir, rank)
            errors = []
            check_payload(torch, args, payload, manifest, before, values["parallel"], completed, values["train_args"], errors)
            ensure(Path(result).resolve() == expected_dir.resolve(), "Wrong save return path")
            checks.agree(errors, "actual-main checkpoint contents and RNG")
            saves.append({"completed_iters": completed, "path": str(expected_dir), "best_val_loss": report_number(best)})
            if phase == "interrupted" and completed == case["cut"]:
                stop_ready = True
            return result

        @functools.wraps(original_load)
        def observed_load(*positional, **keywords):
            nonlocal best, loaded_snapshot, inspected
            values = bind(original_load, positional, keywords)
            errors = []
            try:
                ensure(phase in ("resume", "done") and not load_calls and not steps and not data_calls["train"],
                       "main must restore once, before sampling")
                ensure(Path(values["checkpoint_dir"]).resolve() == resume_dir.resolve(), "Wrong resume directory")
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, "main resume arguments and position")
            result = original_load(*positional, **keywords)
            manifest, payload = read_checkpoint(torch, resume_dir, rank)
            objects.update(model=values["model"], optimizer=values["optimizer"],
                           train_gen=values["train_data_generator"], eval_gen=values["eval_data_generator"])
            best = float(payload["best_val_loss"])
            errors = []
            compare_nested(torch, args, result, {"completed_iters": case["cut"], "best_val_loss": best}, errors, "load_result", True)
            loaded_snapshot = state_now()
            for key in ("model", "optimizer", "optimizer_param_names", "rng", "best_val_loss"):
                compare_nested(torch, args, loaded_snapshot[key], payload[key], errors, "loaded/" + key, True)
            checks.agree(errors, "actual-main exact model/AdamW/RNG restore")
            for stream, key in (("train", "train_data"), ("val", "eval_data")):
                expected_generators[stream].set_state(payload["rng"][key])
            check_objects(checks, support, base, values["model"], values["optimizer"], values["parallel"], case)
            inspected = True
            load_calls.append({"completed_iters": case["cut"], "best_val_loss": report_number(best)})
            return result

        dist.init_process_group, dist.destroy_process_group = observed_init, observed_destroy
        trainer.get_ep_batch, trainer.train_ep_step, trainer.evaluate_ep = observed_batch, observed_step, observed_eval
        trainer.save_ep_checkpoint, trainer.load_ep_checkpoint = observed_save, observed_load
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.ep_size), LOCAL_RANK=str(rank),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        argv_case = dict(case, iters=phase_target(case, phase))
        argv = support.trainer_argv(args, data_dir, argv_case)
        argv[argv.index("--dropout") + 1] = str(case["dropout"])
        argv += ["--eval-interval", str(case["eval_interval"]), "--eval-iters", str(case["eval_iters"]),
                 "--save-interval", str(case["save_interval"]), "--out-dir", str(out_dir)]
        if phase in ("resume", "done"):
            argv += ["--resume", str(resume_dir)]
        sys.argv = [str(args.project / "train_ep.py")] + argv
        record["active_phase"] = case["name"] + "/" + phase + ": original main()"
        with contextlib.redirect_stdout(output):
            try:
                trainer.main()
            except StopAfterCheckpoint:
                ensure(phase == "interrupted" and stop_ready, "Unexpected controlled interruption")
                stopped = True
        ensure(stopped == (phase == "interrupted"), "Interrupted main did not stop at its checkpoint boundary")
        ensure(len(init_calls) == 1 and len(destroy_calls) == 1 and not dist.is_initialized(), "main initialization/cleanup count is wrong")
        ensure(len(steps) == limit - start and len(data_calls["train"]) == len(steps), "Wrong resumed iteration/data count")
        ensure([item["completed_iters"] for item in evaluations] == phase_schedule(case, phase, "eval"), "Wrong evaluation schedule")
        ensure([item["completed_iters"] for item in saves] == phase_schedule(case, phase, "save"), "Wrong save schedule")
        ensure(len(data_calls["val"]) == len(evaluations) * case["eval_iters"], "Wrong validation batch count")
        ensure(len(load_calls) == int(phase in ("resume", "done")), "Wrong checkpoint load count")
        if rank == 0:
            logged = check_main_logs(output.getvalue(), case, phase, steps, evaluations, saves, out_dir)
        else:
            ensure(not any(line.startswith(("iter ", "eval after ", "saved checkpoint: ", "resumed checkpoint: "))
                           for line in output.getvalue().splitlines()), "A non-master rank printed training/checkpoint logs")
            logged = {}
        artifact = {"steps": steps, "evaluations": evaluations, "final": state_now(), "loaded": loaded_snapshot}
        torch.save(artifact, directory / f"state_rank_{rank}.pt")
        record.update(status="passed", active_phase="complete", stdout=output.getvalue(),
                      steps=[{key: value for key, value in item.items() if key != "state"} for item in steps],
                      evaluations=[{key: value for key, value in item.items() if key != "state"} for item in evaluations],
                      saves=saves, load_calls=load_calls, data_calls=data_calls, logged=logged,
                      best_val_loss=report_number(best), controlled_interruption=stopped)
        if rank == 0:
            print(f"[PASS] {case['name']}/{phase}: actual main / iterations [{start}, {limit}) / save-resume logs", flush=True)
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc(), stdout=output.getvalue())
        if output.getvalue():
            print(f"[CAPTURED rank={rank}]\n{output.getvalue()}", flush=True)
        print(f"[FAIL rank={rank}] {record['active_phase']}", flush=True)
        raise
    finally:
        sys.argv = previous_argv
        try:
            if dist is not None and original_init is not None:
                dist.init_process_group, dist.destroy_process_group = original_init, original_destroy
                if dist.is_initialized():
                    original_destroy()
        except BaseException:
            record.update(status="failed", cleanup_exception=traceback.format_exc())
            raise
        finally:
            base.write_json(path, record)


def compare_main_runs(torch, args, workdir):
    comparisons, errors = [], []
    exact = args.device == "cpu"
    for case in main_cases(args.ep_size):
        for rank in range(args.ep_size):
            root = Path(workdir) / case["name"]

            def read(phase):
                directory = root / phase
                state = torch.load(directory / f"state_rank_{rank}.pt", map_location="cpu", weights_only=True)
                record = json.loads((directory / f"rank_{rank}.json").read_text(encoding="utf-8"))
                return state, record

            baseline, baseline_record = read("baseline")
            interrupted, interrupted_record = read("interrupted")
            resumed, resumed_record = read("resume")
            label = f"{case['name']}/rank_{rank}"
            combined_steps = interrupted["steps"] + resumed["steps"]
            combined_evals = interrupted["evaluations"] + resumed["evaluations"]
            compare_nested(torch, args, combined_steps, baseline["steps"], errors,
                           label + "/continuous_vs_resumed_steps", exact)
            compare_nested(torch, args, combined_evals, baseline["evaluations"], errors,
                           label + "/continuous_vs_resumed_evaluations", exact)
            compare_nested(torch, args, resumed["final"], baseline["final"], errors,
                           label + "/continuous_vs_resumed_final", exact)
            compare_nested(torch, args, resumed["loaded"], interrupted["final"], errors,
                           label + "/exact_checkpoint_boundary", True)
            for stream in ("train", "val"):
                combined = interrupted_record["data_calls"][stream] + resumed_record["data_calls"][stream]
                compare_nested(torch, args, combined, baseline_record["data_calls"][stream], errors,
                               label + "/sampled_windows/" + stream, True)
            if rank == 0:
                for stream in ("train", "eval", "save"):
                    combined = interrupted_record["logged"][stream] + resumed_record["logged"][stream]
                    compare_nested(torch, args, combined, baseline_record["logged"][stream], errors,
                                   label + "/global_log_counters/" + stream, True)
            maximum = 0.0
            for name, value in baseline["final"]["model"].items():
                if value.is_floating_point() and value.numel():
                    maximum = max(maximum, float((resumed["final"]["model"][name] - value).abs().max().item()))
            if case["name"] == "sparse_dropout":
                done, done_record = read("done")
                compare_nested(torch, args, done["final"], interrupted["final"], errors,
                               label + "/already_completed_final", True)
                compare_nested(torch, args, done["loaded"], interrupted["final"], errors,
                               label + "/already_completed_load", True)
                if done["steps"] or done["evaluations"] or any(done_record["data_calls"].values()):
                    errors.append(label + ": an already-completed resume performed work")
            comparisons.append({"case": case["name"], "rank": rank,
                "CPU_exact": exact, "RNG_exact": True, "direct_restore_exact": True,
                "max_parameter_abs_error": maximum,
                "interrupted_target": case["interrupted_target"], "resumed_target": case["iters"]})
        if errors:
            raise AssertionError("Continuous/resumed trajectories differ:\n" + "\n".join(errors[:24]))
        print(f"[PASS] {case['name']}: continuous vs fresh-process resume / all steps, AdamW, RNG and sampling", flush=True)
    return comparisons


def main(argv=None):
    args = parse_args(argv)
    # Set these before importing NumPy/PyTorch and before launching workers.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    cases = main_cases(args.ep_size)
    phases = ["functions"]
    for case in cases:
        phases += [case["name"] + "/" + phase for phase in ("baseline", "interrupted", "resume")]
        if case["name"] == "sparse_dropout":
            phases.append(case["name"] + "/done")
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": {"original_save_load_and_actual_main": True, "fresh_process_resume": True,
                  "model_buffers_local_experts_tied_head_AdamW": True,
                  "CPU_CUDA_default_and_train_eval_generator_RNG": True, "nonzero_dropout": True,
                  "sample_windows_all_updates_and_validation_trajectory": True,
                  "collective_read_write_and_RNG_restore_failures": True,
                  "incomplete_checkpoint_and_overwrite_refusal": True,
                  "rank_config_optimizer_order_and_progress_validation": True,
                  "save_after_eval_and_best_loss_on_all_ranks": True,
                  "periodic_final_save_and_global_iteration_logs": True,
                  "increasing_max_iters_and_already_completed_resume": True,
                  "CPU_trajectory_comparison": "exact", "CUDA_trajectory_comparison": "tolerance",
                  "save_load_roundtrip_and_RNG_comparison": "exact",
                  "temporary_data_and_checkpoints_only": True,
                  "rendezvous": "FileStore replaces env:// transport; original backend/collectives are retained",
                  "independent_model_gradient_AdamW_math": "covered by earlier verifiers, not repeated here",
                  "AMP_compile_EP_resize_multinode_OS_crash": False}, "worker_phases": {}}
    evaluation, support, base, workdir, code = None, None, None, None, 1
    try:
        evaluation, support, base = load_helpers()
        report["helper_sha256"] = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (evaluation, support, base)}
        report["source_sha256"] = preflight(evaluation, support, base, args.project)
        print("[PASS] source syntax / EP checkpoint and training signatures (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Use the nanoGPT Python environment with PyTorch and NumPy installed.") from exc
        ensure(torch.distributed.is_available(), "PyTorch distributed is unavailable")
        if args.device == "cuda":
            ensure(torch.cuda.is_available() and torch.cuda.device_count() >= args.ep_size,
                   "CUDA verification requires one visible GPU per EP worker")
            ensure(torch.distributed.is_nccl_available(), "This PyTorch build has no NCCL backend")
        else:
            ensure(torch.distributed.is_gloo_available(), "This PyTorch build has no Gloo backend")
        report.update(torch_version=torch.__version__, numpy_version=np.__version__)
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_checkpoint_")
        for phase in phases:
            (Path(workdir) / phase).mkdir(parents=True)
        data_dir = evaluation.create_data(workdir, args.seed)
        data_hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                       for name in ("train.bin", "val.bin")}
        report["temporary_data_sha256"] = data_hashes
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}.", flush=True)
        print("[INFO] Nonzero dropout; checkpoint model/AdamW/RNG round-trip; collective failure injection.", flush=True)
        print("[INFO] Real main(): continuous / interrupted / fresh-process resume / already-completed resume.", flush=True)
        print("[INFO] Only temporary uint16 data and temporary checkpoint directories are used.", flush=True)
        mp.spawn(helper_worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        for case in cases:
            run_phases = ["baseline", "interrupted", "resume"]
            if case["name"] == "sparse_dropout":
                run_phases.append("done")
            for phase in run_phases:
                mp.spawn(main_worker, args=(args, workdir, case, phase), nprocs=args.ep_size, join=True)
        records = support.collect_records(base, workdir, args.ep_size, phases)
        ensure(all(record["status"] == "passed" for group in records.values() for record in group),
               "At least one worker did not finish all checks")
        report["trajectory_comparisons"] = compare_main_runs(torch, args, workdir)
        ensure(preflight(evaluation, support, base, args.project) == report["source_sha256"],
               "A project source file changed during verification")
        ensure({name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in data_hashes} == data_hashes,
               "A temporary dataset changed")
        report["status"], code = "passed", 0
        print("[PASS] EP checkpoint verification completed.", flush=True)
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        traceback.print_exc()
    finally:
        if workdir is not None:
            report["worker_phases"] = support.collect_records(base, workdir, args.ep_size, phases)
            if args.keep_workdir:
                report["workdir"] = workdir
                print(f"[INFO] Temporary files kept: {workdir}", flush=True)
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
