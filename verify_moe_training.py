#!/usr/bin/env python3
"""Run real nanoGPT train.py in isolated CPU processes and verify its checkpoints.

Requires verify_moe_model.py beside this script, plus the project files.
Example:
    python verify_moe_training.py --project . --threads 1 --report moe_training_report.json

Expected training contract:
* iter_num counts completed optimizer updates; max_iters is the total goal.
* Final ckpt.pt contains the last completed update, optimizer, model_args,
  best_val_loss, config, and scaler state.
* Resume restores the saved model architecture / MoE objective and optimizer.
* eval_only evaluates once even on resume, makes no updates, and saves nothing.
* Completed goals make no additional updates.
* Old dense checkpoints without MoE/scaler fields remain compatible.

The script runs the actual unmodified train.py using runpy in child processes.
Hooks and a read-only profiler record inputs, outputs and optimizer state;
no training loop is replaced. A functional GPT independently checks the
observations. Temporary uint16 datasets and output folders are generated.

Continuous/resumed trajectory comparisons use constant tokens and dropout=0.
They do NOT test RNG/data-stream restoration, CUDA/NCCL, AMP numerical behavior,
torch.compile, EP or multi-node launch. CUDA implementation is not constrained
by this CPU verification. Only the report is retained unless --keep-workdir.
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
import pickle
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

torch = None


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Actual nanoGPT MoE training/checkpoint verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per child process in seconds.")
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--_child-manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("threads", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("rtol", "atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
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
        raise FileNotFoundError("Place verify_moe_model.py beside verify_moe_training.py.")
    spec = importlib.util.spec_from_file_location("_moe_training_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preflight(project, helper):
    sources = helper.preflight(project)
    path = project / "configurator.py"
    ensure(path.is_file(), f"Missing {path}")
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    sources["configurator.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse((project / "train.py").read_text(encoding="utf-8-sig"))
    loops = [node for node in tree.body if isinstance(node, ast.While)]
    ensure(len(loops) == 1, "Expected one top-level training while loop.")
    has_total_goal = any(
        isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "iter_num" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Lt) and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "max_iters"
        for node in ast.walk(loops[0].test)
    )
    ensure(has_total_goal,
           "Use 'while iter_num < max_iters or eval_only:' and remove the old > max_iters break.")
    return sources


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def gradients(model):
    return {
        name: None if parameter.grad is None else clone_cpu(parameter.grad)
        for name, parameter in model.named_parameters()
    }


class Observer:
    def __init__(self, project):
        self.train_path = str((project / "train.py").resolve())
        self.model_path = str((project / "model.py").resolve())
        self.model = self.optimizer = self.namespace = None
        self.completed = 0
        self.pending = None
        self.clip_events = []
        self.evaluation_start = 0
        self.packet = {
            "status": "running", "steps": [], "forwards": [], "batches": [],
            "evaluations": [], "saves": [], "optimizer_loads": 0,
            "scaler_loads": 0, "mfu_calls": 0, "pretrained_called": False,
        }

    def bind(self, model, optimizer, namespace):
        ensure(self.model is None, "Production optimizer was constructed more than once.")
        self.model, self.optimizer, self.namespace = model, optimizer, namespace
        self.packet.update(
            model_config=asdict(model.config),
            initial_iter=int(namespace["iter_num"]),
            initial_best=float(namespace["best_val_loss"]),
            initial_model=clone_cpu(model.state_dict()),
            initial_optimizer=clone_cpu(optimizer.state_dict()),
        )
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        self.packet["parameter_groups"] = [
            [names[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
        model.register_forward_hook(self.forward)
        optimizer.register_step_pre_hook(self.before_step)
        optimizer.register_step_post_hook(self.after_step)
        original_load = optimizer.load_state_dict

        def load(state, *args, **kwargs):
            result = original_load(state, *args, **kwargs)
            self.packet["optimizer_loads"] += 1
            self.packet["loaded_optimizer"] = clone_cpu(optimizer.state_dict())
            return result
        optimizer.load_state_dict = load

    def forward(self, model, inputs, output):
        ensure(len(inputs) == 2, "Expected GPT token IDs and targets as positional inputs.")
        logits, loss = output
        self.packet["forwards"].append({
            "local_step": self.completed, "iter": int(self.namespace["iter_num"]),
            "training": bool(model.training),
            "x": clone_cpu(inputs[0]), "y": clone_cpu(inputs[1]),
            "logits": clone_cpu(logits), "loss": clone_cpu(loss),
        })

    def before_step(self, optimizer, _args, _kwargs):
        ensure(self.pending is None, "Overlapping optimizer updates were observed.")
        self.pending = {
            "iter": int(self.namespace["iter_num"]),
            "model_before": clone_cpu(self.model.state_dict()),
            "optimizer_before": clone_cpu(optimizer.state_dict()),
            "gradients": gradients(self.model),
            "clips": self.clip_events,
        }
        self.clip_events = []

    def after_step(self, optimizer, _args, _kwargs):
        ensure(self.pending is not None, "Post hook ran without a pre hook.")
        self.pending.update(
            model_after=clone_cpu(self.model.state_dict()),
            optimizer_after=clone_cpu(optimizer.state_dict()),
        )
        self.packet["steps"].append(self.pending)
        self.pending = None
        self.completed += 1

    def profile(self, frame, event, returned):
        name, filename = frame.f_code.co_name, frame.f_code.co_filename
        if event == "call" and filename == self.train_path and name == "estimate_loss":
            self.evaluation_start = len(self.packet["forwards"])
        if event == "return" and filename == self.train_path:
            if name == "get_batch" and isinstance(returned, tuple):
                self.packet["batches"].append({
                    "split": frame.f_locals["split"],
                    "indices": clone_cpu(frame.f_locals["ix"]),
                    "x": clone_cpu(returned[0]), "y": clone_cpu(returned[1]),
                })
            elif name == "estimate_loss" and isinstance(returned, dict):
                self.packet["evaluations"].append({
                    "iter": int(frame.f_globals["iter_num"]),
                    "losses": {key: float(value) for key, value in returned.items()},
                    "forward_start": self.evaluation_start,
                    "forward_end": len(self.packet["forwards"]),
                })
        if event == "call":
            if filename == self.model_path and name == "estimate_mfu":
                self.packet["mfu_calls"] += 1
            if name == "load_state_dict":
                instance = frame.f_locals.get("self")
                if instance is not None and type(instance).__name__ == "GradScaler":
                    self.packet["scaler_loads"] += 1

    def finish(self, namespace):
        self.packet.update(
            final_iter=int(namespace["iter_num"]),
            final_best=float(namespace["best_val_loss"]),
            final_config=clone_cpu(namespace["config"]),
            final_model_args=clone_cpu(namespace["model_args"]),
            mfu_supported=bool(namespace["mfu_supported"]),
        )
        if self.model is not None:
            self.packet["final_model"] = clone_cpu(self.model.state_dict())
            self.packet["final_optimizer"] = clone_cpu(self.optimizer.state_dict())
            self.packet["gradients_cleared"] = all(
                parameter.grad is None for parameter in self.model.parameters()
            )
            self.packet["final_scaler"] = clone_cpu(namespace["scaler"].state_dict())
        self.packet["status"] = "passed"


def child(manifest_path):
    global torch
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configure_threads(manifest["threads"])
    torch = importlib.import_module("torch")
    torch.set_num_threads(manifest["threads"])
    torch.set_num_interop_threads(manifest["threads"])
    helper = load_helper()
    helper.torch = torch
    project = Path(manifest["project"])
    GPT, _, _, _, _ = helper.load_project(project)
    observer = Observer(project)
    original_configure = GPT.configure_optimizers
    original_save = torch.save
    original_clip = torch.nn.utils.clip_grad_norm_
    previous_profile, previous_argv = sys.getprofile(), sys.argv
    previous_directory = Path.cwd()

    def configure(model, *args, **kwargs):
        namespace = sys._getframe(1).f_globals
        optimizer = original_configure(model, *args, **kwargs)
        observer.bind(model, optimizer, namespace)
        return optimizer

    def save(value, destination, *args, **kwargs):
        if (isinstance(value, dict) and "model" in value and "model_args" in value
                and isinstance(destination, (str, os.PathLike))):
            observer.packet["saves"].append({
                "completed": observer.packet["initial_iter"] + observer.completed,
                "payload": clone_cpu(value), "path": str(destination),
                "evaluation_count": len(observer.packet["evaluations"]),
            })
        return original_save(value, destination, *args, **kwargs)

    def clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        before = gradients(observer.model)
        norm = original_clip(parameters, max_norm, *args, **kwargs)
        observer.clip_events.append({
            "max_norm": float(max_norm), "before": before, "norm": clone_cpu(norm),
        })
        return norm

    def blocked_pretrained(*_args, **_kwargs):
        observer.packet["pretrained_called"] = True
        raise RuntimeError("Verifier blocked a pretrained download: add the MoE/GPT-2 guard first.")

    GPT.configure_optimizers = configure
    GPT.from_pretrained = blocked_pretrained
    torch.save = save
    torch.nn.utils.clip_grad_norm_ = clip
    success = False
    try:
        os.chdir(project)
        sys.argv = [str(project / "train.py"), *manifest["train_arguments"]]
        sys.setprofile(observer.profile)
        namespace = runpy.run_path(str(project / "train.py"), run_name="__main__")
        sys.setprofile(previous_profile)
        observer.finish(namespace)
        success = True
    except Exception as exc:
        observer.packet.update(
            status="failed", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        sys.setprofile(previous_profile)
        sys.argv = previous_argv
        os.chdir(previous_directory)
        torch.save = original_save
        torch.nn.utils.clip_grad_norm_ = original_clip
        original_save(observer.packet, manifest["observations"])
    return 0 if success else 1


def scheduled_lr(iteration):
    maximum, minimum, warmup, end = 0.002, 0.0002, 2, 6
    if iteration < warmup:
        return maximum * (iteration + 1) / (warmup + 1)
    if iteration > end:
        return minimum
    fraction = (iteration - warmup) / (end - warmup)
    return minimum + (maximum - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))


def compare_tree(v, actual, expected, label, exact=False):
    if isinstance(expected, torch.Tensor):
        ensure(isinstance(actual, torch.Tensor), f"{label}: tensor missing.")
        if exact:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            v.close(actual, expected, label)
    elif isinstance(expected, dict):
        ensure(isinstance(actual, dict) and actual.keys() == expected.keys(),
               f"{label}: dictionary keys differ.")
        for key in expected:
            compare_tree(v, actual[key], expected[key], f"{label}/{key}", exact)
    elif isinstance(expected, (list, tuple)):
        ensure(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
               f"{label}: sequence length differs.")
        for index, item in enumerate(expected):
            compare_tree(v, actual[index], item, f"{label}/{index}", exact)
    else:
        ensure(actual == expected, f"{label}: {actual!r} != {expected!r}")


def compare_gradients(v, actual, expected, label):
    ensure(actual.keys() == expected.keys(), f"{label}: gradient names differ.")
    for name, value in expected.items():
        if value is None:
            ensure(actual[name] is None, f"{label}: unused expert has a gradient: {name}")
        else:
            ensure(actual[name] is not None, f"{label}: missing gradient: {name}")
            v.close(actual[name], value, f"{label}/{name}")


def audit_batches(v, packet, tokens, block_size=5):
    ensure(packet["batches"], "The actual data loader did not return any batches.")
    for index, batch in enumerate(packet["batches"]):
        data = tokens[batch["split"]]
        starts = batch["indices"].tolist()
        ensure(len(starts) == 2, "Unexpected batch_size.")
        ensure(all(0 <= start < len(data) - block_size for start in starts),
               "Data-loader sample index is out of range.")
        expected_x = torch.tensor(
            [data[start:start + block_size] for start in starts], dtype=torch.long
        )
        expected_y = torch.tensor(
            [data[start + 1:start + block_size + 1] for start in starts], dtype=torch.long
        )
        compare_tree(v, batch["x"], expected_x, f"Data batch {index}/x", exact=True)
        compare_tree(v, batch["y"], expected_y, f"Data batch {index}/shifted targets", exact=True)


def audit_phase(v, helper, GPT, GPTConfig, packet, phase, tokens, before_checkpoint=None):
    ensure(packet["status"] == "passed", packet.get("traceback", "Actual train.py failed."))
    cfg = GPTConfig(**packet["model_config"])
    initial = packet["initial_iter"]
    expected_start = before_checkpoint["iter_num"] if before_checkpoint else 0
    ensure(initial == expected_start, "Resume started at the wrong saved iteration.")
    count = 0 if phase["eval_only"] else max(phase["goal"] - initial, 0)
    ensure(len(packet["steps"]) == count,
           f"Expected {count} updates, observed {len(packet['steps'])}; check the loop stop condition.")
    expected_final = initial if phase["eval_only"] else max(initial, phase["goal"])
    ensure(packet["final_iter"] == expected_final, "Final completed-update counter is wrong.")
    audit_batches(v, packet, tokens)
    if before_checkpoint:
        compare_tree(v, packet["initial_model"], before_checkpoint["model"],
                     "Actual resumed initial weights", exact=True)
        ensure(packet["optimizer_loads"] == 1, "Resume did not load AdamW state once.")
        compare_tree(v, packet["loaded_optimizer"], before_checkpoint["optimizer"],
                     "Actual restored AdamW state", exact=True)
        ensure(packet["scaler_loads"] == int("scaler" in before_checkpoint),
               "Saved scaler state was not restored, or legacy fallback is incorrect.")
        defaults = asdict(GPTConfig())
        keys = ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size",
                "use_moe", "n_experts", "moe_top_k", "moe_aux_loss_coef")
        for key in keys:
            expected = before_checkpoint["model_args"].get(key, defaults[key])
            ensure(packet["model_config"][key] == expected, f"Restored model config is wrong: {key}")
    else:
        ensure(cfg.use_moe == phase["use_moe"] and cfg.bias == phase["bias"],
               "Scratch initialization ignored the requested model mode.")
    ensure(cfg.dropout == 0.0, "This deterministic verification requires dropout=0.")
    for key, value in packet["final_model_args"].items():
        ensure(packet["final_config"].get(key) == value,
               f"Logging/checkpoint config is stale: {key}; call config.update(model_args).")
    if cfg.use_moe:
        ensure(not packet["mfu_supported"] and packet["mfu_calls"] == 0,
               "MoE called the dense MFU estimator.")
    if phase["eval_only"]:
        ensure(packet["evaluations"], "eval_only did not evaluate on resume.")
        ensure(not packet["saves"], "eval_only unexpectedly wrote a checkpoint.")
    else:
        ensure(packet["saves"], "No final checkpoint was saved.")
        ensure(packet["saves"][-1]["payload"]["iter_num"] == expected_final,
               "The final checkpoint does not contain the last update.")
    ensure(packet["gradients_cleared"], "Gradients were not cleared after optimizer updates.")

    reference = GPT(cfg).to(device="cpu", dtype=torch.float32)
    reference.load_state_dict(packet["initial_model"], strict=True)
    optimizer = helper.build_unfused_optimizer(reference)
    initial_optimizer = packet.get("loaded_optimizer", packet["initial_optimizer"])
    optimizer.load_state_dict(initial_optimizer)
    all_parameters = dict(reference.named_parameters())
    initial_groups = initial_optimizer["param_groups"]
    observed_names = packet["parameter_groups"]
    registered = [name for group in observed_names for name in group]
    ensure(len(registered) == len(set(registered)) and set(registered) == set(all_parameters),
           "Production AdamW does not register every unique trainable parameter once.")
    for names, group in zip(observed_names, initial_groups):
        for name in names:
            decay = 0.1 if all_parameters[name].dim() >= 2 else 0.0
            ensure(group["weight_decay"] == decay, f"Wrong weight decay for {name}.")
        ensure(group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8,
               "Unexpected production AdamW hyperparameters.")
        ensure(not group.get("fused"), "CPU verification expects the unfused optimizer.")

    parameter_history = {initial: clone_cpu(reference.state_dict())}
    optimizer_history = {initial: clone_cpu(optimizer.state_dict())}
    for local_step in range(count + 1):
        parameters = helper.parameter_reference(reference)
        records = [item for item in packet["forwards"] if item["local_step"] == local_step]
        for record in records:
            logits, ce, auxiliary, _ = helper.functional_gpt(
                parameters, cfg, record["x"], record["y"]
            )
            expected_loss = ce
            if record["training"] and cfg.use_moe:
                expected_loss = ce + cfg.moe_aux_loss_coef * auxiliary
            v.close(record["logits"], logits.detach(), "Actual main forward logits")
            v.close(record["loss"], expected_loss.detach(), "Actual main forward loss")
        training = [item for item in records if item["training"]]
        if local_step == count:
            ensure(not training, "Forward/backward continued after the expected final update.")
            break
        ensure(len(training) == 2, "Actual main did not use two microbatches per update.")
        optimizer.zero_grad(set_to_none=True)
        parameters = helper.parameter_reference(reference)
        route_counts = {}
        for record in training:
            _, ce, auxiliary, counts = helper.functional_gpt(
                parameters, cfg, record["x"], record["y"]
            )
            loss = ce + cfg.moe_aux_loss_coef * auxiliary if cfg.use_moe else ce
            (loss / 2).backward()
            for layer, layer_counts in enumerate(counts):
                for expert, number in enumerate(layer_counts.tolist()):
                    key = (layer, expert)
                    route_counts[key] = route_counts.get(key, 0) + number
        expected_gradients = {}
        for name, parameter in reference.named_parameters():
            match = re.match(r"^transformer\.h\.(\d+)\.mlp\.experts\.(\d+)\.", name)
            unused = match and route_counts.get(tuple(map(int, match.groups())), 0) == 0
            parameter.grad = None if unused else parameters[name].grad
            expected_gradients[name] = clone_cpu(parameter.grad)
        actual = packet["steps"][local_step]
        absolute_step = initial + local_step
        ensure(actual["iter"] == absolute_step, "Optimizer update iteration is off by one.")
        compare_tree(v, actual["model_before"], reference.state_dict(),
                     "Actual main weights before the update")
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr(absolute_step)
        compare_tree(v, actual["optimizer_before"], optimizer.state_dict(),
                     "Actual main AdamW state / scheduled LR before the update")
        if phase["clip"] > 0:
            ensure(len(actual["clips"]) == 1, "Expected one clipping call per optimizer update.")
            clipping = actual["clips"][0]
            ensure(clipping["max_norm"] == phase["clip"], "Wrong clipping threshold.")
            compare_gradients(v, clipping["before"], expected_gradients, "Actual main accumulated gradients")
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), phase["clip"])
            v.close(clipping["norm"], norm, "Actual main clipping norm")
        else:
            ensure(not actual["clips"], "Clipping=0 should disable clipping.")
        compare_gradients(v, actual["gradients"], gradients(reference),
                          "Actual main gradients passed to AdamW")
        optimizer.step()
        compare_tree(v, actual["model_after"], reference.state_dict(), "Actual main AdamW update")
        compare_tree(v, actual["optimizer_after"], optimizer.state_dict(), "Actual main AdamW moments")
        parameter_history[absolute_step + 1] = actual["model_after"]
        optimizer_history[absolute_step + 1] = actual["optimizer_after"]
    if phase["eval_only"]:
        # The original script sets the iteration LR before entering evaluation.
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr(initial)
    compare_tree(v, packet["final_model"], reference.state_dict(), "Actual final model")
    compare_tree(v, packet["final_optimizer"], optimizer.state_dict(), "Actual final optimizer")
    best = packet["initial_best"]
    for evaluation in packet["evaluations"]:
        forwards = packet["forwards"][
            evaluation["forward_start"]:evaluation["forward_end"]
        ]
        ensure(len(forwards) == 4 and not any(item["training"] for item in forwards),
               "Evaluation did not use two eval batches for each split.")
        for split, records in (("train", forwards[:2]), ("val", forwards[2:])):
            mean = sum(float(item["loss"].item()) for item in records) / 2
            v.close(torch.tensor(evaluation["losses"][split], dtype=torch.float64),
                    torch.tensor(mean, dtype=torch.float64),
                    f"Actual main evaluation mean/{split}")
        best = min(best, evaluation["losses"]["val"])
    ensure(packet["final_best"] == best, "best_val_loss is not the historical minimum.")
    for saved in packet["saves"]:
        payload, completed = saved["payload"], saved["completed"]
        ensure(payload["iter_num"] == completed, "Saved counter does not match completed updates.")
        ensure("scaler" in payload, "Save scaler.state_dict() in the checkpoint.")
        compare_tree(v, payload["model"], parameter_history[completed],
                     "Checkpoint model at the recorded update", exact=True)
        # At an iteration boundary, LR may have been set for the next update.
        saved_optimizer = copy.deepcopy(payload["optimizer"])
        history_optimizer = copy.deepcopy(optimizer_history[completed])
        for group in saved_optimizer["param_groups"]:
            group["lr"] = 0.0
        for group in history_optimizer["param_groups"]:
            group["lr"] = 0.0
        compare_tree(v, saved_optimizer, history_optimizer,
                     "Checkpoint moments at the recorded update", exact=True)
        minimum = packet["initial_best"]
        for evaluation in packet["evaluations"][:saved["evaluation_count"]]:
            minimum = min(minimum, evaluation["losses"]["val"])
        ensure(payload["best_val_loss"] == minimum, "Checkpoint lost historical best validation loss.")
    return {"updates": count, "final_iter": expected_final,
            "data_batches_checked": len(packet["batches"])}


def fixture_data(root, varied=False):
    np = importlib.import_module("numpy")
    directory = root / ("varied_data" if varied else "constant_data")
    directory.mkdir()
    tokens = {
        "train": [index % 23 for index in range(257)] if varied else [0] * 257,
        "val": [(3 * index + 1) % 23 for index in range(257)] if varied else [1] * 257,
    }
    for split, values in tokens.items():
        np.asarray(values, dtype=np.uint16).tofile(directory / f"{split}.bin")
    with (directory / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": 23}, handle)
    return directory, tokens


def phase_arguments(phase, data, out):
    values = {
        "device": "cpu", "dtype": "float32", "compile": False, "wandb_log": False,
        "dataset": str(data), "out_dir": str(out), "init_from": phase["init_from"],
        "max_iters": phase["goal"], "eval_interval": 2, "eval_iters": 2,
        "log_interval": 1, "eval_only": phase["eval_only"], "always_save_checkpoint": True,
        "batch_size": 2, "block_size": 5, "gradient_accumulation_steps": 2,
        "n_layer": 2, "n_head": 3, "n_embd": 12, "bias": phase["bias"], "dropout": 0.0,
        "use_moe": phase["use_moe"], "n_experts": 3, "moe_top_k": 2,
        "moe_aux_loss_coef": 0.11,
        "learning_rate": 0.002, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
        "grad_clip": phase["clip"], "decay_lr": True, "warmup_iters": 2,
        "lr_decay_iters": 6, "min_lr": 0.0002,
    }
    if phase["init_from"] == "resume":
        # Restored architecture and MoE objective must override these CLI values.
        values.update(n_layer=1, n_head=2, n_embd=8, bias=not phase["bias"],
                      use_moe=not phase["use_moe"], n_experts=4,
                      moe_top_k=1, moe_aux_loss_coef=0.0)
    return [f"--{key}={value}" for key, value in values.items()]


def run_phase(args, root, helper, v, GPT, GPTConfig, name, data, tokens, *,
              goal=6, bias=True, use_moe=True, init_from="scratch",
              checkpoint=None, eval_only=False, clip=0.01, expected_guard=False):
    directory = root / name
    directory.mkdir()
    output = directory / "out"
    output.mkdir()
    prior = None
    if checkpoint is not None:
        shutil.copy2(checkpoint, output / "ckpt.pt")
        prior = torch.load(output / "ckpt.pt", map_location="cpu", weights_only=True)
    phase = {
        "goal": goal, "bias": bias, "use_moe": use_moe, "init_from": init_from,
        "eval_only": eval_only, "clip": clip,
    }
    observations = directory / "observations.pt"
    manifest = {
        "project": str(args.project), "threads": args.threads,
        "train_arguments": phase_arguments(phase, data, output),
        "observations": str(observations),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "LOCAL_WORLD_SIZE"):
        environment.pop(key, None)
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="")
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_child-manifest", str(manifest_path)],
        cwd=args.project, env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=args.timeout,
    )
    ensure(observations.is_file(), f"{name}: child did not write observations.\n{process.stderr[-6000:]}")
    packet = torch.load(observations, map_location="cpu", weights_only=True)
    summary = {
        "name": name, "status": "running", "returncode": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr,
    }
    v.report["phases"].append(summary)
    if expected_guard:
        ensure(process.returncode != 0 and packet.get("error_type") == "ValueError",
               "GPT-2/MoE guard must raise ValueError before loading pretrained weights.")
        ensure(not packet["pretrained_called"], "Missing guard attempted pretrained loading.")
        summary["status"] = "passed"
        print(f"[PASS] {name}: pretrained guard (no download).", flush=True)
        return None, packet
    ensure(process.returncode == 0, f"{name}: actual train.py failed.\n{packet.get('traceback', process.stderr)}")
    detail = audit_phase(v, helper, GPT, GPTConfig, packet, phase, tokens, prior)
    summary.update(status="passed", **detail)
    checkpoint_path = output / "ckpt.pt"
    if eval_only:
        ensure(checkpoint_path.is_file(), "eval_only lost the supplied checkpoint.")
        after = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        compare_tree(v, after, prior, "eval_only checkpoint remains unchanged", exact=True)
    else:
        ensure(checkpoint_path.is_file(), "Final ckpt.pt is missing.")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        compare_tree(v, saved["model"], packet["final_model"], "Actual final on-disk weights", exact=True)
        compare_tree(v, saved["optimizer"], packet["final_optimizer"], "Actual final on-disk moments", exact=True)
        ensure(saved["iter_num"] == packet["final_iter"], "Final on-disk counter is stale.")
    print(f"[PASS] {name}: actual data / forward / gradients / AdamW / checkpoint.", flush=True)
    return checkpoint_path, packet


def compare_continuation(v, baseline, resumed, include_model_args=True):
    first = torch.load(baseline, map_location="cpu", weights_only=True)
    second = torch.load(resumed, map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "iter_num", "best_val_loss", "scaler"):
        compare_tree(v, second[key], first[key], f"Continuous vs resumed/{key}", exact=True)
    if include_model_args:
        compare_tree(v, second["model_args"], first["model_args"],
                     "Continuous vs resumed/model_args", exact=True)


def exercise(args, root, helper, v, GPT, GPTConfig):
    data, tokens = fixture_data(root)
    for bias in (True, False):
        label = "bias_true" if bias else "bias_false"
        baseline, _ = run_phase(args, root, helper, v, GPT, GPTConfig,
                                label + "_continuous", data, tokens, bias=bias)
        cut = 2 if bias else 3
        prefix, _ = run_phase(args, root, helper, v, GPT, GPTConfig,
                              label + "_prefix", data, tokens, goal=cut, bias=bias)
        run_phase(args, root, helper, v, GPT, GPTConfig, label + "_eval_only",
                  data, tokens, bias=bias, init_from="resume",
                  checkpoint=prefix, eval_only=True)
        resumed, _ = run_phase(args, root, helper, v, GPT, GPTConfig,
                               label + "_resumed", data, tokens, bias=bias,
                               init_from="resume", checkpoint=prefix)
        compare_continuation(v, baseline, resumed)
        print(f"[PASS] {label}: continuous vs resumed trajectory (fixed data, dropout=0).", flush=True)
        if bias:
            run_phase(args, root, helper, v, GPT, GPTConfig, "already_complete",
                      data, tokens, bias=bias, init_from="resume", checkpoint=baseline)
    dense, _ = run_phase(args, root, helper, v, GPT, GPTConfig, "dense_continuous",
                         data, tokens, use_moe=False)
    dense_prefix, _ = run_phase(args, root, helper, v, GPT, GPTConfig, "dense_prefix",
                                data, tokens, goal=2, use_moe=False)
    legacy = torch.load(dense_prefix, map_location="cpu", weights_only=True)
    for key in ("use_moe", "n_experts", "moe_top_k", "moe_aux_loss_coef"):
        legacy["model_args"].pop(key, None)
    legacy.pop("scaler", None)
    legacy_path = root / "legacy_dense.pt"
    torch.save(legacy, legacy_path)
    dense_resumed, _ = run_phase(args, root, helper, v, GPT, GPTConfig, "legacy_dense_resume",
                                 data, tokens, use_moe=False, init_from="resume",
                                 checkpoint=legacy_path)
    compare_continuation(v, dense, dense_resumed, include_model_args=False)
    print("[PASS] Legacy dense checkpoint without MoE/scaler fields.", flush=True)
    varied, varied_tokens = fixture_data(root, varied=True)
    run_phase(args, root, helper, v, GPT, GPTConfig, "varied_token_loader",
              varied, varied_tokens, goal=1, clip=0.0)
    run_phase(args, root, helper, v, GPT, GPTConfig, "gpt2_moe_guard",
              data, tokens, init_from="gpt2", expected_guard=True)


def main(argv=None):
    global torch
    args = parse_args(argv)
    if args._child_manifest is not None:
        return child(args._child_manifest.resolve())
    args.project = args.project.expanduser().resolve()
    report = {
        "schema_version": 1, "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(args.project), "scope": "real single-process CPU train.py, float32/eager",
        "not_tested": ["RNG/data-stream restoration", "CUDA/NCCL full main",
                       "AMP numerical behavior", "torch.compile", "EP", "multi-node launch"],
        "checks": [], "phases": [],
    }
    root = None
    success = False
    try:
        configure_threads(args.threads)
        helper = load_helper()
        report["source_sha256"] = preflight(args.project, helper)
        print("[PASS] source syntax / completed-update loop contract (static only).", flush=True)
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is not installed in this Python environment.") from exc
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        helper.torch = torch
        GPT, GPTConfig, _, _, _ = helper.load_project(args.project)
        # Reuse the model verifier's comparison utilities at the actual dtype.
        args.device, args.dtype, args.seed = "cpu", "float32", 1337
        v = helper.Verifier(args, report)
        report.update(torch_version=torch.__version__, rtol=v.rtol, atol=v.atol,
                      threads=args.threads)
        root = Path(tempfile.mkdtemp(prefix="moe_training_verification_"))
        print("[INFO] Actual train.py; one CPU child at a time; tiny temporary datasets.", flush=True)
        print("[INFO] Fixed-data resume comparison; separate varied-token data-loader case.", flush=True)
        exercise(args, root, helper, v, GPT, GPTConfig)
        report.update(status="passed", tensor_comparisons=v.comparisons,
                      max_abs_error=v.max_abs_error)
        success = True
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        print(f"[FAIL] {report['error']}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if root is not None:
            if args.keep_workdir:
                report["workdir"] = str(root)
                print(f"[INFO] Retained workdir: {root}", flush=True)
            else:
                shutil.rmtree(root)
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
        print("[PASS] MoE actual training/checkpoint verification completed.", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
