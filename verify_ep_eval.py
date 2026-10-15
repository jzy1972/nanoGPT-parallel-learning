#!/usr/bin/env python3
"""Verify EP validation and its integration into the real train_ep.main().

Keep verify_ep_gpt.py and verify_ep_training.py beside this file. Run in nanoGPT:
    python verify_ep_eval.py --project . --device cpu --ep-size 2 --threads 1 --report ep_eval_report.json

Required API:
    train_ep.evaluate_ep(model, parallel, data_path, batch_size, seq_length,
                         eval_iters, device, generator) -> global mean CE
    --eval-interval INT (default 0 = disabled), --eval-iters POSITIVE_INT
    The other model / optimizer / training APIs checked by verify_ep_training.

The original evaluator, data reader, training step, and main() are executed.
An independent functional full-MoE GPT supplies reference logits and CE.
Temporary uint16 train.bin and val.bin contain distinct random token streams.
Function cases vary rank-local token counts, initial training mode, dropout,
bias, top-k, and auxiliary coefficient. Nonempty AdamW state and existing
gradients must survive evaluation unchanged. Global CPU/CUDA RNG state and
the independent training Generator must also survive unchanged. The passed
evaluation Generator must advance by the expected sampling operations.

Fresh-worker main() cases compare evaluation disabled/enabled, checking
every training update against the full-MoE oracle, sampling streams, evaluation
schedule and rank-zero logs. Final parameters, AdamW states, and training RNG
are compared between paired runs. CPU trajectory comparisons are exact;
CUDA parameter/moment comparisons allow the earlier numerical tolerances,
because atomic GPU accumulation can differ between independent runs.

One EP group covers WORLD, with the same group for shared-parameter DP.
Only the main() rendezvous transport is redirected from env:// to FileStore;
real Gloo/NCCL collectives and initialization/cleanup are retained. Multi-node
launching, TCP/env rendezvous, AMP, compile, checkpoint/resume, and extra expert
replica groups are outside scope. Main() comparisons use dropout=0. Separate
evaluation-function cases enable dropout=0.35 to check eval-mode handling.
Targets from these bin files are all valid: ignore_index masks are not tested.
No project source or user dataset is edited. Failures retain full tracebacks.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import functools
import hashlib
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


EVAL_LOG = re.compile(r"^eval after (\d+) steps: val_loss=([^\s,]+)\s*$")


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def load_support():
    path = Path(__file__).resolve().with_name("verify_ep_training.py")
    ensure(path.is_file(), "Keep verify_ep_training.py in the same folder as verify_ep_eval.py")
    spec = importlib.util.spec_from_file_location("_ep_eval_training_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EP validation function and actual-main verification.")
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


def preflight(support, base, project):
    hashes = support.preflight(base, project)
    path = project / "train_ep.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "evaluate_ep"), None)
    ensure(function is not None, "train_ep.py needs a top-level evaluate_ep()")
    args = function.args
    names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    expected = {"model", "parallel", "data_path", "batch_size", "seq_length",
                "eval_iters", "device", "generator"}
    ensure(expected <= names or args.kwarg is not None,
           f"evaluate_ep must accept: {sorted(expected)}")
    return hashes


def phase_specs(size):
    tail = {"bias": True, "k": 1, "coef": 0.17, "batch": 7, "micro": 3,
            "sequence": 7, "clip": 0.01, "iters": 5, "interval": 2, "eval_iters": 2}
    full = {"bias": False, "k": 2 * size, "coef": 0.17, "batch": 3, "micro": 8,
            "sequence": 1, "clip": 0.0, "iters": 2, "interval": 5, "eval_iters": 1}
    return [dict(tail, name="baseline_tail", eval_interval=0),
            dict(tail, name="eval_tail", eval_interval=2),
            dict(full, name="baseline_full", eval_interval=0),
            dict(full, name="eval_every", eval_interval=1),
            dict(full, name="eval_final", eval_interval=10)]


def expected_schedule(case):
    interval = case["eval_interval"]
    if interval == 0:
        return []
    return [completed for completed in range(1, case["iters"] + 1)
            if completed % interval == 0 or completed == case["iters"]]


def create_data(workdir, seed):
    import numpy as np
    directory = Path(workdir) / "data"
    directory.mkdir()
    for index, name in enumerate(("train.bin", "val.bin")):
        rng = np.random.default_rng((seed + 1931 * index) % (2 ** 32))
        rng.integers(0, 31, size=4096 - 521 * index, dtype=np.uint16).tofile(directory / name)
    return directory


def rng_snapshot(torch, device):
    result = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def parameter_groups(model, optimizer):
    names = {id(param): name for name, param in model.named_parameters()}
    return [{key: [names[id(param)] for param in value] if key == "params" else copy.deepcopy(value)
             for key, value in group.items()} for group in optimizer.param_groups]


def snapshot(base, checks, model, optimizer, train_generator):
    return {"state": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "grads": {name: (param.grad, None if param.grad is None else param.grad.detach().clone())
                      for name, param in model.named_parameters()},
            "optimizer": base.state_snapshot(optimizer, model),
            "groups": parameter_groups(model, optimizer),
            "modes": {name: module.training for name, module in model.named_modules()},
            "rng": rng_snapshot(checks.torch, checks.device),
            "train_generator": train_generator.get_state().clone(),
            "config": copy.deepcopy(vars(model.config))}


def check_preserved(base, checks, model, optimizer, train_generator, before, errors):
    if {name: module.training for name, module in model.named_modules()} != before["modes"]:
        errors.append("Evaluation failed to restore the original model/submodule modes")
    if vars(model.config) != before["config"]:
        errors.append("Evaluation modified GPTConfig")
    state = model.state_dict()
    if set(state) != set(before["state"]):
        errors.append("Evaluation changed model state keys")
    for name, value in state.items():
        if name in before["state"]:
            base.tensor_close(checks, value, before["state"][name], name + " parameter/buffer unchanged", errors, exact=True)
    for name, param in model.named_parameters():
        original_object, original_value = before["grads"][name]
        if param.grad is not original_object:
            errors.append(name + ": evaluation replaced or cleared an existing .grad")
        if original_value is not None:
            base.tensor_close(checks, param.grad, original_value, name + " existing gradient unchanged", errors, exact=True)
        base.compare_states(checks, optimizer.state.get(param, {}), before["optimizer"][name],
                            name + " optimizer state unchanged", errors, exact=True)
    if parameter_groups(model, optimizer) != before["groups"]:
        errors.append("Evaluation modified optimizer parameter groups")
    for key, value in rng_snapshot(checks.torch, checks.device).items():
        base.tensor_close(checks, value, before["rng"][key], key + " global RNG unchanged", errors, exact=True)
    base.tensor_close(checks, train_generator.get_state(), before["train_generator"],
                      "training Generator unchanged by evaluation", errors, exact=True)


def expected_batches(support, torch, data_path, batch, sequence, iters, generator, device):
    import numpy as np
    cloned = torch.Generator(device="cpu")
    cloned.set_state(generator.get_state())
    batches = [support.expected_batch(torch, np, data_path, batch, sequence, cloned, device)
               for _ in range(iters)]
    return batches, cloned.get_state().clone()


def audit_evaluation(support, base, checks, original_evaluate, model, optimizer, parallel,
                     data_path, batch_size, seq_length, eval_iters, generator, train_generator,
                     cfg, parameters, label):
    torch, dist = checks.torch, checks.dist
    checks.phase(label + ": original evaluate_ep / full-MoE CE reference")
    batches, expected_generator_state = expected_batches(support, torch, data_path, batch_size,
        seq_length, eval_iters, generator, checks.device)
    before = snapshot(base, checks, model, optimizer, train_generator)
    statistics = torch.zeros(2, dtype=torch.float64, device=checks.device)
    errors, calls = [], []

    def observe_forward(module, inputs, kwargs, output):
        index = len(calls)
        calls.append(True)
        if index >= eval_iters:
            errors.append("evaluate_ep made an extra GPT forward call")
            return
        try:
            ids = inputs[0] if inputs else kwargs["idx"]
            targets = inputs[1] if len(inputs) > 1 else kwargs.get("targets")
            expected_x, expected_y, _ = batches[index]
            base.tensor_close(checks, ids, expected_x, f"eval batch {index} sampled inputs", errors, exact=True)
            base.tensor_close(checks, targets, expected_y, f"eval batch {index} shifted targets", errors, exact=True)
            ensure(not torch.is_grad_enabled(), "Evaluation must disable gradient recording")
            ensure(all(not child.training for child in module.modules()), "Evaluation must put the whole GPT in eval mode")
            logits, loss = output
            ensure(not logits.requires_grad and not loss.requires_grad, "Evaluation output retained an autograd graph")
            with torch.no_grad():
                expected_logits, ce, auxiliary, _ = base.functional_gpt(parameters, cfg, expected_x, expected_y)
                base.tensor_close(checks, logits, expected_logits, f"eval batch {index} full logits", errors)
                base.tensor_close(checks, loss, ce, f"eval batch {index} CE only; no auxiliary penalty", errors)
                count = expected_y.numel()
                statistics[0].add_(ce.to(torch.float64) * count)
                statistics[1].add_(count)
        except Exception:
            errors.append(traceback.format_exc())

    handle = model.register_forward_hook(observe_forward, with_kwargs=True)
    try:
        result = original_evaluate(model=model, parallel=parallel, data_path=str(data_path),
            batch_size=batch_size, seq_length=seq_length, eval_iters=eval_iters,
            device=checks.device, generator=generator)
    finally:
        handle.remove()
    if len(calls) != eval_iters:
        errors.append(f"Expected {eval_iters} evaluation forwards, got {len(calls)}")
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=parallel["shared_dp_group"])
    expected = float((statistics[0] / statistics[1]).item())
    base.scalar_close(checks, result, expected, "global token-weighted validation CE", errors)
    check_preserved(base, checks, model, optimizer, train_generator, before, errors)
    base.tensor_close(checks, generator.get_state(), expected_generator_state,
                      "evaluation Generator advanced exactly by eval_iters batches", errors, exact=True)
    # Agree before converting a possibly invalid return value on one rank.
    checks.agree(errors, label + ": CE / token weighting / modes / gradients / AdamW / RNG")
    values = [None] * checks.args.ep_size
    dist.all_gather_object(values, float(result), group=parallel["shared_dp_group"])
    checks.agree([] if all(value == values[0] for value in values) else
                 [f"Validation result must be identical on all ranks: {values}"], label + ": rank agreement")
    details = {"loss": float(result), "reference_global_ce": expected,
               "global_tokens": int(statistics[1].item()), "local_batch_size": batch_size,
               "local_sequence_length": seq_length, "eval_iters": eval_iters,
               "initial_training_mode": before["modes"][""],
               "sampled_starts": [starts for _, _, starts in batches]}
    checks.passed(label + ": CE / token weighting / modes / gradients / AdamW / RNG", details)
    return result, details


def check_cli(support, checks, trainer, data_dir):
    errors = []
    try:
        case = phase_specs(checks.args.ep_size)[0]
        argv = support.trainer_argv(checks.args, data_dir, case)
        namespace = support.invoke_parser(trainer, argv)
        ensure(namespace.eval_interval == 0, "--eval-interval must default to 0")
        namespace = support.invoke_parser(trainer, argv + ["--eval-interval", "2", "--eval-iters", "3"])
        ensure(namespace.eval_interval == 2 and namespace.eval_iters == 3, "Evaluation CLI flags were ignored")
        for options in (["--eval-interval", "-1"], ["--eval-iters", "0"], ["--eval-iters", "-1"]):
            try:
                support.invoke_parser(trainer, argv + options)
            except SystemExit as exc:
                ensure(exc.code != 0, "Invalid eval option exited successfully")
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid evaluation options accepted: {options}")
    except Exception:
        errors.append(traceback.format_exc())
    checks.agree(errors, "evaluation CLI / disabled default / invalid counts")
    checks.passed("evaluation CLI / disabled default / invalid counts")


class ForcedEvaluationError(RuntimeError):
    pass


def check_failure_restore(base, checks, trainer, model, optimizer, parallel, data_path, train_generator):
    torch = checks.torch
    for mode in (True, False):
        model.train(mode)
        before = snapshot(base, checks, model, optimizer, train_generator)
        generator = torch.Generator(device="cpu").manual_seed(checks.args.seed + 35_000 + checks.rank)

        def fail_after_forward(module, inputs, output):
            raise ForcedEvaluationError("Intentional failure after all ranks completed one EP forward")

        handle = model.register_forward_hook(fail_after_forward)
        raised, errors = False, []
        try:
            trainer.evaluate_ep(model=model, parallel=parallel, data_path=str(data_path),
                batch_size=2, seq_length=3, eval_iters=2, device=checks.device, generator=generator)
        except ForcedEvaluationError:
            raised = True
        finally:
            handle.remove()
        if not raised:
            errors.append("evaluate_ep swallowed the intentional forward exception")
        check_preserved(base, checks, model, optimizer, train_generator, before, errors)
        checks.agree(errors, f"evaluation exception restores training={mode}")
        checks.passed(f"evaluation exception restores training={mode}")
    for invalid in (0, -1):
        errors = []
        try:
            trainer.evaluate_ep(model=model, parallel=parallel, data_path=str(data_path),
                batch_size=2, seq_length=3, eval_iters=invalid, device=checks.device, generator=generator)
        except ValueError:
            pass
        else:
            errors.append(f"evaluate_ep accepted eval_iters={invalid}")
        checks.agree(errors, f"evaluate_ep rejects eval_iters={invalid}")
    checks.passed("evaluate_ep rejects zero/negative batch counts")


def function_worker(rank, args, workdir):
    support = load_support()
    base = support.load_base()
    base.configure_threads(args.threads)
    path = Path(workdir) / "functions" / f"rank_{rank}.json"
    record = {"rank": rank, "status": "running", "active_phase": "initialization", "passed": []}
    base.write_json(path, record)
    dist = None
    try:
        torch, dist, device, base = support.init_worker_runtime(args, rank)
        support.file_init(dist, args, rank, Path(workdir) / "functions" / "rendezvous.store")
        checks = base.Checks(torch, dist, args, rank, device, record, path)
        modules = support.load_project(base, args.project)
        trainer = modules["train_ep"]
        parallel = modules["parallel_ep"].setup_ep()
        base.check_groups(checks, modules["parallel_ep"], parallel)
        check_cli(support, checks, trainer, Path(workdir) / "data")
        cases = [(True, 1, 0.19), (False, 2 * args.ep_size, 0.19),
                 (True, 2 * args.ep_size, 0.0), (False, 1, 0.0)]
        for case_index, (bias, k, coef) in enumerate(cases):
            case = {"use_moe": True, "bias": bias, "k": k, "coef": coef}
            label = f"eval-function bias={bias} k={k} aux_coef={coef}"
            checks.phase(label + ": constructors")
            cfg, model, prototype = base.construct(checks, modules, parallel, case)
            base.check_structure_and_load(checks, modules, parallel, cfg, model, prototype, label)
            unique, parameters = base.make_reference_parameters(prototype)
            optimizer = modules["parallel_ep"].build_ep_optimizer(model=model,
                learning_rate=7e-4, weight_decay=0.13, beta1=0.8, beta2=0.93)
            reference_optimizer = base.reference_optimizer(unique)
            # Populate AdamW state with deterministic GLOBAL-name gradients.
            for named in (dict(model.named_parameters()), unique):
                for name, param in named.items():
                    param.grad = torch.full_like(param, 0.001 * (1 + len(name) % 7))
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            # Nonzero dropout catches a missing model.eval().
            cfg.dropout = 0.35
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.35
            for block in model.transformer.h:
                block.attn.dropout = 0.35
            # Preserve both None and non-None existing gradients.
            for index, (_, param) in enumerate(model.named_parameters()):
                param.grad = None if index % 2 else torch.full_like(param, 0.125)
            train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000 + rank)
            for initial_mode in (True, False):
                model.train(initial_mode)
                seed = args.seed + 20_000 + 101 * case_index + rank
                generator = torch.Generator(device="cpu").manual_seed(seed)
                batch = 1 + rank % 3
                sequence = 7 if (rank + case_index) % 2 == 0 else 1
                for call in range(2):
                    audit_evaluation(support, base, checks, trainer.evaluate_ep, model, optimizer, parallel,
                        Path(workdir) / "data" / "val.bin", batch, sequence, 3, generator, train_generator,
                        cfg, parameters, label + f" initial_training={initial_mode} call={call}")
            if case_index == 0:
                check_failure_restore(base, checks, trainer, model, optimizer, parallel,
                    Path(workdir) / "data" / "val.bin", train_generator)
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


def cpu_copy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def check_eval_logs(text, case, rank, evaluations):
    matches = []
    for line in text.splitlines():
        if line.startswith("eval after "):
            match = EVAL_LOG.fullmatch(line)
            ensure(match is not None, f"Malformed evaluation log: {line}")
            matches.append(match)
    expected = expected_schedule(case) if rank == 0 else []
    ensure([int(match.group(1)) for match in matches] == expected,
           f"rank {rank}: expected evaluation log steps {expected}")
    for match, result in zip(matches, evaluations):
        ensure(math.isclose(float(match.group(2)), result["loss"], rel_tol=0, abs_tol=1.1e-6),
               "Printed validation loss differs from evaluate_ep's result")
    return expected


def main_worker(rank, args, workdir, case):
    support = load_support()
    base = support.load_base()
    base.configure_threads(args.threads)
    directory = Path(workdir) / case["name"]
    path = directory / f"rank_{rank}.json"
    record = {"rank": rank, "status": "running", "active_phase": "initialization", "passed": [], "case": case}
    base.write_json(path, record)
    output = io.StringIO()
    dist, original_init, original_destroy = None, None, None
    previous_argv = sys.argv
    try:
        torch, dist, device, base = support.init_worker_runtime(args, rank)
        import numpy as np
        checks = base.Checks(torch, dist, args, rank, device, record, path)
        modules = support.load_project(base, args.project)
        trainer = modules["train_ep"]
        original_init, original_destroy = dist.init_process_group, dist.destroy_process_group
        original_batch, original_step, original_evaluate = trainer.get_ep_batch, trainer.train_ep_step, trainer.evaluate_ep
        init_calls, destroy_calls, steps, evaluations = [], [], [], []
        data_calls, generators, last_batch, objects = {"train": [], "val": []}, {}, {}, {}
        expected_generators = {stream: torch.Generator(device="cpu").manual_seed(args.seed + offset + rank)
                               for stream, offset in (("train", 10_000), ("val", 20_000))}
        reference, in_evaluation = None, False
        data_dir = Path(workdir) / "data"

        @functools.wraps(original_init)
        def observed_init(*positional, **keywords):
            requested = inspect.signature(original_init).bind(*positional, **keywords).arguments
            backend = requested.get("backend")
            ensure(str(backend).lower() == ("nccl" if args.device == "cuda" else "gloo"), "main used the wrong backend")
            ensure(not dist.is_initialized(), "main initialized a process group twice")
            init_calls.append(str(backend))
            store = dist.FileStore(str(directory / "rendezvous.store"), args.ep_size)
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
            stream = "val" if in_evaluation else "train"
            result = original_batch(*positional, **keywords)
            errors = []
            try:
                ensure(Path(values["data_path"]).resolve() == (data_dir / (stream + ".bin")).resolve(),
                       f"main {stream} read the wrong bin file")
                ensure(values["batch_size"] == case["batch"] and values["seq_length"] == case["sequence"],
                       "main ignored batch-size / seq-length")
                actual_generator = values["generator"]
                if stream in generators:
                    ensure(actual_generator is generators[stream], f"main recreated the {stream} Generator")
                generators[stream] = actual_generator
                if len(generators) == 2:
                    ensure(generators["train"] is not generators["val"], "Training and evaluation must use distinct Generators")
                expected_x, expected_y, starts = support.expected_batch(torch, np,
                    data_dir / (stream + ".bin"), case["batch"], case["sequence"], expected_generators[stream], device)
                base.tensor_close(checks, result[0], expected_x, f"main {stream} sampled inputs", errors, exact=True)
                base.tensor_close(checks, result[1], expected_y, f"main {stream} shifted targets", errors, exact=True)
                base.tensor_close(checks, actual_generator.get_state(), expected_generators[stream].get_state(),
                                  f"main {stream} independent Generator state", errors, exact=True)
                data_calls[stream].append(starts)
                last_batch[stream] = (result[0].detach().clone(), result[1].detach().clone())
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + f": main {stream} data call")
            return result

        @functools.wraps(original_step)
        def observed_step(*positional, **keywords):
            nonlocal reference
            bound = inspect.signature(original_step).bind(*positional, **keywords)
            bound.apply_defaults()
            values = bound.arguments
            errors = []
            try:
                ensure(not in_evaluation, "main called the training step from evaluation")
                ensure(len(data_calls["train"]) == len(steps) + 1, "One training batch is required per step")
                ensure(values["micro_batch_size"] == case["micro"] and float(values["grad_clip"]) == case["clip"],
                       "main ignored micro-batch-size / grad-clip")
                for key, actual, expected in (("inputs", values["token_ids"], last_batch["train"][0]),
                                              ("targets", values["targets"], last_batch["train"][1])):
                    base.tensor_close(checks, actual, expected, "main training " + key, errors, exact=True)
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + ": training-step arguments")
            if reference is None:
                reference = support.assemble_main_reference(base, checks, values["model"], values["optimizer"],
                                                           values["parallel"], case)
                objects.update(model=values["model"], optimizer=values["optimizer"])
            ensure(values["model"] is objects["model"] and values["optimizer"] is objects["optimizer"],
                   "main recreated its model or optimizer")
            cfg, unique, parameters, ref_optimizer = reference
            result, details = support.audit_step(base, checks, original_step, values["model"], values["optimizer"],
                values["parallel"], values["token_ids"], values["targets"], values["micro_batch_size"],
                values["grad_clip"], cfg, unique, parameters, ref_optimizer, case["name"] + f" step={len(steps)}")
            steps.append(details)
            return result

        @functools.wraps(original_evaluate)
        def observed_evaluate(*positional, **keywords):
            nonlocal in_evaluation
            bound = inspect.signature(original_evaluate).bind(*positional, **keywords)
            bound.apply_defaults()
            values = bound.arguments
            schedule = expected_schedule(case)
            errors = []
            try:
                ensure(reference is not None, "main evaluated before completing the first training update")
                ensure(len(evaluations) < len(schedule) and len(steps) == schedule[len(evaluations)],
                       f"Wrong evaluation schedule: after {len(steps)} steps; expected {schedule}")
                ensure(values["model"] is objects["model"], "main evaluated a different GPT")
                ensure(values["batch_size"] == case["batch"] and values["seq_length"] == case["sequence"],
                       "main ignored evaluation batch-size / seq-length")
                ensure(values["eval_iters"] == case["eval_iters"], "main ignored eval-iters")
                ensure(Path(values["data_path"]).resolve() == (data_dir / "val.bin").resolve(), "main used the wrong validation path")
                base.tensor_close(checks, values["generator"].get_state(), expected_generators["val"].get_state(),
                                  "main evaluation Generator before evaluation", errors, exact=True)
            except Exception:
                errors.append(traceback.format_exc())
            checks.agree(errors, case["name"] + ": evaluation call schedule / arguments")
            cfg, unique, parameters, ref_optimizer = reference
            in_evaluation = True
            try:
                result, details = audit_evaluation(support, base, checks, original_evaluate, values["model"],
                    objects["optimizer"], values["parallel"], data_dir / "val.bin", values["batch_size"],
                    values["seq_length"], values["eval_iters"], values["generator"], generators["train"],
                    cfg, parameters, case["name"] + f" eval_after={len(steps)}")
            finally:
                in_evaluation = False
            details["completed_iters"] = len(steps)
            evaluations.append(details)
            return result

        dist.init_process_group, dist.destroy_process_group = observed_init, observed_destroy
        trainer.get_ep_batch, trainer.train_ep_step, trainer.evaluate_ep = observed_batch, observed_step, observed_evaluate
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.ep_size), LOCAL_RANK=str(rank),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        sys.argv = [str(args.project / "train_ep.py")] + support.trainer_argv(args, data_dir, case) + [
            "--eval-interval", str(case["eval_interval"]), "--eval-iters", str(case["eval_iters"])]
        record["active_phase"] = case["name"] + ": original main()"
        with contextlib.redirect_stdout(output):
            trainer.main()
        ensure(len(init_calls) == 1 and len(destroy_calls) == 1 and not dist.is_initialized(),
               "main must initialize and destroy its process group once")
        ensure(len(steps) == case["iters"] and len(data_calls["train"]) == case["iters"], "Wrong number of training updates/data calls")
        ensure([entry["completed_iters"] for entry in evaluations] == expected_schedule(case), "Wrong actual evaluation schedule")
        ensure(len(data_calls["val"]) == len(evaluations) * case["eval_iters"], "Wrong number of validation data calls")
        support.check_main_logs(output.getvalue(), case, rank, steps)
        logged_evaluations = check_eval_logs(output.getvalue(), case, rank, evaluations)
        model, optimizer = objects["model"], objects["optimizer"]
        state = {"parameters": {name: param for name, param in model.named_parameters()},
                 "optimizer": base.state_snapshot(optimizer, model), "groups": parameter_groups(model, optimizer),
                 "training_generator": generators["train"].get_state(), "rng": rng_snapshot(torch, device)}
        torch.save(cpu_copy(state), directory / f"state_rank_{rank}.pt")
        record.update(status="passed", active_phase="complete", stdout=output.getvalue(), steps=steps,
                      evaluations=evaluations, data_calls=data_calls, logged_evaluations=logged_evaluations)
        if rank == 0:
            print(f"[PASS] {case['name']}: actual main / training reference / evaluation steps {expected_schedule(case)} / logs", flush=True)
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


def compare_trajectories(torch, args, workdir):
    pairs = [("baseline_tail", "eval_tail"), ("baseline_full", "eval_every"), ("baseline_full", "eval_final")]
    comparisons = []
    exact = args.device == "cpu"
    rtol = args.rtol if args.rtol is not None else (3e-5 if args.dtype == "float64" else 3e-4)
    atol = args.atol if args.atol is not None else (2e-6 if args.dtype == "float64" else 3e-5)
    for baseline, enabled in pairs:
        for rank in range(args.ep_size):
            left = torch.load(Path(workdir) / baseline / f"state_rank_{rank}.pt", map_location="cpu", weights_only=True)
            right = torch.load(Path(workdir) / enabled / f"state_rank_{rank}.pt", map_location="cpu", weights_only=True)
            ensure(left["groups"] == right["groups"], "Evaluation changed the optimizer parameter groups between runs")
            ensure(left["parameters"].keys() == right["parameters"].keys(), "Paired runs have different parameter keys")
            maximum = 0.0
            for name, value in left["parameters"].items():
                torch.testing.assert_close(right["parameters"][name], value, rtol=0 if exact else rtol, atol=0 if exact else atol,
                                           msg=f"{enabled} rank {rank}: parameter trajectory changed: {name}")
                maximum = max(maximum, float((right["parameters"][name] - value).abs().max().item()))
                expected_state, actual_state = left["optimizer"][name], right["optimizer"][name]
                ensure(actual_state.keys() == expected_state.keys(), "Paired optimizer state keys differ: " + name)
                for key, item in expected_state.items():
                    if hasattr(item, "detach"):
                        state_atol = args.atol if args.atol is not None else (
                            (1e-13 if key == "exp_avg_sq" else 1e-10) if args.dtype == "float64" else
                            (1e-10 if key == "exp_avg_sq" else 1e-7))
                        strict = exact or key == "step"
                        torch.testing.assert_close(actual_state[key], item, rtol=0 if strict else rtol,
                            atol=0 if strict else state_atol, msg=f"{enabled} rank {rank}: AdamW trajectory changed: {name}.{key}")
                    else:
                        ensure(actual_state[key] == item, "Paired optimizer state differs")
            torch.testing.assert_close(left["training_generator"], right["training_generator"], rtol=0, atol=0)
            for key in left["rng"]:
                torch.testing.assert_close(left["rng"][key], right["rng"][key], rtol=0, atol=0)
            left_report = json.loads((Path(workdir) / baseline / f"rank_{rank}.json").read_text(encoding="utf-8"))
            right_report = json.loads((Path(workdir) / enabled / f"rank_{rank}.json").read_text(encoding="utf-8"))
            ensure(left_report["data_calls"]["train"] == right_report["data_calls"]["train"], "Evaluation changed the training windows")
            for before, after in zip(left_report["steps"], right_report["steps"]):
                ensure(math.isclose(before["returned_loss"], after["returned_loss"],
                    rel_tol=0 if exact else rtol, abs_tol=0 if exact else atol), "Evaluation changed the training loss trajectory")
            comparisons.append({"baseline": baseline, "enabled": enabled, "rank": rank,
                                "exact_CPU_comparison": exact, "max_parameter_abs_error": maximum})
        print(f"[PASS] {baseline} vs {enabled}: training windows / parameters / AdamW / RNG", flush=True)
    return comparisons


def main(argv=None):
    args = parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": {"original_evaluate_ep_and_main": True, "independent_functional_full_MoE_CE_reference": True,
                  "unequal_rank_token_counts": args.ep_size > 1, "auxiliary_penalty_excluded": True,
                  "dropout_disabled_and_modes_restored": True, "exception_mode_restoration": True,
                  "parameters_buffers_existing_gradients_AdamW_preserved": True,
                  "independent_train_eval_and_global_RNG": True, "actual_training_updates_against_oracle": True,
                  "periodic_final_every_step_disabled_eval_schedule": True, "rank_zero_logging": True,
                  "paired_training_trajectories_eval_enabled_disabled": True,
                  "CPU_trajectory_comparison": "exact", "CUDA_trajectory_comparison": "tolerance",
                  "all_targets_valid": True, "rendezvous": "FileStore replaces env:// transport only",
                  "AMP_compile_checkpoint_multinode_expert_replica_groups": False}, "worker_phases": {}}
    support, base, workdir, code = None, None, None, 1
    phases = ["functions"] + [case["name"] for case in phase_specs(args.ep_size)]
    try:
        support = load_support()
        base = support.load_base()
        report["helper_sha256"] = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (support, base)}
        report["source_sha256"] = preflight(support, base, args.project)
        print("[PASS] source syntax / evaluate_ep signature (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Use the nanoGPT Python environment with PyTorch and NumPy installed.") from exc
        ensure(torch.distributed.is_available(), "PyTorch distributed is unavailable")
        if args.device == "cuda":
            ensure(torch.cuda.is_available() and torch.cuda.device_count() >= args.ep_size, "CUDA requires one visible GPU per EP worker")
            ensure(torch.distributed.is_nccl_available(), "This PyTorch build has no NCCL backend")
        else:
            ensure(torch.distributed.is_gloo_available(), "This PyTorch build has no Gloo backend")
        report.update(torch_version=torch.__version__, numpy_version=np.__version__)
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_eval_")
        for phase in phases:
            (Path(workdir) / phase).mkdir()
        data_dir = create_data(workdir, args.seed)
        hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("train.bin", "val.bin")}
        report["temporary_data_sha256"] = hashes
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}.", flush=True)
        print("[INFO] Temporary train/val data; unequal-rank token counts; mode/gradient/AdamW/RNG preservation.", flush=True)
        print("[INFO] Real main() in five fresh-worker runs; eval off/on paired trajectories; checkpoint is pending.", flush=True)
        mp.spawn(function_worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        for case in phase_specs(args.ep_size):
            mp.spawn(main_worker, args=(args, workdir, case), nprocs=args.ep_size, join=True)
        records = support.collect_records(base, workdir, args.ep_size, phases)
        ensure(all(record["status"] == "passed" for group in records.values() for record in group), "A worker did not complete")
        report["trajectory_comparisons"] = compare_trajectories(torch, args, workdir)
        ensure(preflight(support, base, args.project) == report["source_sha256"], "A project source file changed")
        ensure({name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in hashes} == hashes,
               "A temporary dataset changed")
        report["status"], code = "passed", 0
        print("[PASS] EP evaluation verification completed.", flush=True)
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        traceback.print_exc()
    finally:
        if workdir is not None:
            report["worker_phases"] = support.collect_records(base, workdir, args.ep_size, phases)
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
