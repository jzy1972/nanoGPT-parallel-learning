"""Check actual CUDA train.py: continuous training vs save-and-resume."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from check_project import check_project
from verification_common import compare_trees


def compare_checkpoint_dirs(left: Path, right: Path, tp_size: int, dp_size: int,
                            *, atol=0.0, rtol=0.0, expected_iter=None):
    import torch

    model_fields = (
        "model", "optimizer", "scaler", "model_args", "iter_num", "best_val_loss",
        "tp_size", "dp_size", "tp_rank", "dtype",
    )
    config_fields = (
        "gradient_accumulation_steps", "batch_size", "block_size", "n_layer", "n_head",
        "n_embd", "dropout", "bias", "learning_rate", "min_lr", "weight_decay",
        "beta1", "beta2", "grad_clip", "decay_lr", "warmup_iters", "lr_decay_iters",
        "eval_interval", "eval_iters", "dtype", "compile", "dataset", "tp_size",
    )
    iteration = None
    for t in range(tp_size):
        filename = f"ckpt_tp{t:03d}.pt"
        if not (left / filename).is_file() or not (right / filename).is_file():
            raise AssertionError(f"Missing checkpoint shard: {filename}")
        a = torch.load(left / filename, map_location="cpu", weights_only=True)
        b = torch.load(right / filename, map_location="cpu", weights_only=True)
        if a["tp_size"] != tp_size or a["dp_size"] != dp_size or a["tp_rank"] != t:
            raise AssertionError(f"Unexpected TP/DP metadata in {filename}")
        if iteration is None:
            iteration = a["iter_num"]
        if a["iter_num"] != iteration:
            raise AssertionError("Checkpoint shards are from different iterations")
        if expected_iter is not None and iteration != expected_iter:
            raise AssertionError(f"Expected checkpoint at iter {expected_iter}, got {iteration}")
        for field in model_fields:
            compare_trees(a[field], b[field], f"{filename}.{field}", atol=atol, rtol=rtol)
        for field in config_fields:
            compare_trees(
                a["config"][field], b["config"][field],
                f"{filename}.config.{field}",
            )
        print(f"[PASS] {filename}: model, optimizer, scaler and configuration")
    for rank in range(tp_size * dp_size):
        filename = f"runtime_rank{rank:03d}.pt"
        if not (left / filename).is_file() or not (right / filename).is_file():
            raise AssertionError(f"Missing runtime snapshot: {filename}")
        a = torch.load(left / filename, map_location="cpu", weights_only=True)
        b = torch.load(right / filename, map_location="cpu", weights_only=True)
        required = {
            "world_rank", "tp_size", "dp_size", "iter_num", "cpu_rng", "cuda_rng",
            "tp_rng", "train_generator", "eval_generator", "X", "Y",
        }
        if not required.issubset(a) or not required.issubset(b):
            raise AssertionError(f"Runtime fields missing in {filename}")
        if a["world_rank"] != rank or a["iter_num"] != iteration:
            raise AssertionError(f"Unexpected rank/iteration in {filename}")
        # RNG bytes and prefetched token batches always require an exact match.
        compare_trees(a, b, filename, atol=0.0, rtol=0.0)
        print(f"[PASS] {filename}: RNG states, generators and prefetched X/Y")
    return iteration


def terminate_owned_process(proc):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def run_training(project, command, log_path, timeout):
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, cwd=project, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            return_code = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            terminate_owned_process(proc)
            raise RuntimeError(f"Training stopped; inspect {log_path}") from None
    if return_code:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Training exited with {return_code}. Log: {log_path}\n{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--compare-only", type=Path, nargs=2, metavar=("CONTINUOUS", "RESUMED"))
    parser.add_argument("--total-iters", type=int, default=6)
    parser.add_argument("--split-iter", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.tp_size, args.dp_size, args.grad_accum, args.timeout) <= 0:
        parser.error("TP/DP degrees, accumulation and timeout must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    if not (0 < args.split_iter < args.total_iters):
        parser.error("Require 0 < --split-iter < --total-iters")
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("[ERROR] PyTorch is not installed; see README_zh.md.")
        return 2
    if args.compare_only:
        iteration = compare_checkpoint_dirs(
            args.compare_only[0], args.compare_only[1], args.tp_size, args.dp_size,
            atol=args.atol, rtol=args.rtol,
        )
        print(f"[PASS] Checkpoint/resume equivalence at iter {iteration}")
        return 0
    project = args.project.resolve()
    errors = check_project(project)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    world_size = args.tp_size * args.dp_size
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
        or not dist.is_nccl_available()
    ):
        print(
            f"[NEEDS GPU] Actual train.py resume check needs {world_size} visible GPUs "
            "and NCCL. Run verify_tp_dp.py --device cpu for CPU numerical checks."
        )
        return 2
    try:
        import numpy as np
    except ImportError:
        print("[ERROR] NumPy is required by train.py.")
        return 2
    if not (project / "configurator.py").is_file():
        parser.error(f"configurator.py missing in {project}")
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = project / "verification_runs" / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
    run_dir = run_dir.resolve()
    if run_dir.exists():
        parser.error("--run-dir must be a new directory")
    run_dir.mkdir(parents=True)
    data_dir = run_dir / "data"
    data_dir.mkdir()
    generator = np.random.default_rng(2026)
    generator.integers(0, 64, size=32768, dtype=np.uint16).tofile(data_dir / "train.bin")
    generator.integers(0, 64, size=8192, dtype=np.uint16).tofile(data_dir / "val.bin")
    with (data_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"vocab_size": 64}, f)
    common = {
        "tp_size": args.tp_size, "dataset": str(data_dir),
        "n_layer": 2, "n_head": 2 * args.tp_size, "n_embd": 16 * args.tp_size,
        "block_size": 8, "batch_size": 2, "bias": True, "dropout": 0.2,
        "gradient_accumulation_steps": args.grad_accum,
        "dtype": args.dtype, "compile": False, "wandb_log": False,
        "eval_interval": 1, "eval_iters": 2, "log_interval": 1,
        "always_save_checkpoint": True, "eval_only": False,
        "learning_rate": 0.001, "min_lr": 0.0001, "weight_decay": 0.01,
        "grad_clip": 1.0, "decay_lr": True, "warmup_iters": 2,
        # Same schedule in all three runs, including the shorter first part.
        "lr_decay_iters": args.total_iters + 4,
    }
    launcher = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc_per_node={world_size}", "train.py",
    ]
    jobs = [
        ("continuous", run_dir / "continuous", "scratch", args.total_iters),
        ("first_part", run_dir / "resumed", "scratch", args.split_iter),
        ("resume", run_dir / "resumed", "resume", args.total_iters),
    ]
    for name, out_dir, init_from, stop in jobs:
        config = dict(common, out_dir=str(out_dir), init_from=init_from, max_iters=stop)
        command = launcher + [f"--{key}={value}" for key, value in config.items()]
        print(f"[RUN] {name}; log={run_dir / (name + '.log')}", flush=True)
        run_training(project, command, run_dir / f"{name}.log", args.timeout)
    iteration = compare_checkpoint_dirs(
        run_dir / "continuous", run_dir / "resumed", args.tp_size, args.dp_size,
        atol=args.atol, rtol=args.rtol, expected_iter=args.total_iters,
    )
    report = {
        "passed": True, "iteration": iteration, "tp_size": args.tp_size,
        "dp_size": args.dp_size, "dtype": args.dtype, "dropout": 0.2,
        "scope": "Actual train.py checkpoint boundary, model/optimizer/scaler/RNG/data continuity",
        "atol": args.atol, "rtol": args.rtol,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PASS] Actual train.py save/resume equivalence. Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
