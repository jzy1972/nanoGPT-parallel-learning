"""Compare the user's TP+DP model against a dense nanoGPT reference."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from check_project import check_project
from verification_common import (
    CollectiveChecks,
    load_module,
    load_project_function,
    math_attention_context,
    oracle_shard,
    setup_tracker,
)


def configure_cpu_threads(threads):
    """Limit native thread pools before importing PyTorch or NumPy.

    The spawn parent must do this too, so children inherit the limits before
    Python loads their modules. Explicit assignment also replaces large limits
    already present in the environment.
    """
    for name in (
        "OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS",
        "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(threads)


def worker(local_rank, args, store_path=None):
    configure_cpu_threads(args.threads)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.set_num_threads(args.threads)
    rank = int(os.environ["RANK"]) if store_path is None else local_rank
    world_size = int(os.environ["WORLD_SIZE"]) if store_path is None else args.tp_size * args.dp_size
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if store_path is None:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
            timeout=timedelta(seconds=args.timeout),
        )
    else:
        # Direct FileStore supports Unicode project paths and avoids TCPStore/libuv setup.
        store = dist.FileStore(store_path, world_size)
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            store=store, rank=rank, world_size=world_size,
            timeout=timedelta(seconds=args.timeout),
        )
    try:
        project = Path(args.project)
        candidate = load_module(project / "model.py", "_verification_candidate_model")
        reference = load_module(Path(args.reference), "_verification_dense_model")
        factory = load_project_function(project, "build_parallel_groups", candidate)
        tp_group, dp_group = factory(args.tp_size)
        tp_rank, dp_rank = dist.get_rank(tp_group), dist.get_rank(dp_group)
        checks = CollectiveChecks(device, rank, args.atol, args.rtol)
        own_rank = torch.tensor([rank], dtype=torch.int64, device=device)
        tp_members = [torch.empty_like(own_rank) for _ in range(args.tp_size)]
        dp_members = [torch.empty_like(own_rank) for _ in range(args.dp_size)]
        dist.all_gather(tp_members, own_rank, group=tp_group)
        dist.all_gather(dp_members, own_rank, group=dp_group)
        expected_t, expected_d = rank % args.tp_size, rank // args.tp_size
        checks.condition(tp_rank == expected_t and dp_rank == expected_d, "TP/DP local rank mapping")
        checks.condition(
            [x.item() for x in tp_members]
            == [expected_d * args.tp_size + t for t in range(args.tp_size)],
            "TP group membership",
        )
        checks.condition(
            [x.item() for x in dp_members]
            == [d * args.tp_size + expected_t for d in range(args.dp_size)],
            "DP group membership",
        )
        checks.finish("TP/DP group mapping")

        dtype = torch.float64 if args.precision == "double" else torch.float32
        config_values = dict(
            block_size=args.seq_len, vocab_size=args.vocab_size, n_layer=args.layers,
            n_head=2 * args.tp_size, n_embd=16 * args.tp_size,
            dropout=0.0, bias=not args.no_bias,
        )
        torch.manual_seed(args.seed)
        dense = reference.GPT(reference.GPTConfig(**config_values)).to(device=device, dtype=dtype)
        parallel = candidate.GPT(
            candidate.GPTConfig(**config_values), tp_group=tp_group,
        ).to(device=device, dtype=dtype)
        setup_tracker(candidate, device, rank)
        if args.attention == "manual":
            for model in (dense, parallel):
                for block in model.transformer.h:
                    block.attn.flash = False
                    block.attn.register_buffer(
                        "bias", torch.tril(torch.ones(args.seq_len, args.seq_len, device=device))
                        .view(1, 1, args.seq_len, args.seq_len),
                    )
        parallel.load_dense_state_dict(dense.state_dict())
        checks.condition(
            parallel.transformer.wte.weight is parallel.lm_head.weight,
            "Token embedding / lm_head weight tying",
        )
        for name, actual in parallel.state_dict().items():
            wanted = oracle_shard(name, dense.state_dict()[name], parallel, candidate, tp_rank, args.tp_size)
            checks.close(actual, wanted, f"initial parameter {name}", exact=True)
        checks.finish("Dense weight loading, QKV layout and replicated parameters")
        wrapped = DDP(
            parallel, device_ids=[local_rank] if device.type == "cuda" else None,
            process_group=dp_group,
        )
        dense.train()
        wrapped.train()
        dense_optimizer = torch.optim.AdamW(
            dense.parameters(), lr=0.002, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01, foreach=False,
        )
        parallel_optimizer = torch.optim.AdamW(
            parallel.parameters(), lr=0.002, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01, foreach=False,
        )
        if args.clip_norm is not None:
            mark = load_project_function(project, "mark_tp_parameters", candidate)
            clip = load_project_function(project, "clip_grad_norm_tp", candidate)
            mark(parallel)
        if args.check_overflow:
            has_nonfinite = load_project_function(project, "has_nonfinite_grad", candidate)
        batch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 71)

        for step in range(args.steps):
            dense_optimizer.zero_grad(set_to_none=True)
            parallel_optimizer.zero_grad(set_to_none=True)
            shape = (args.grad_accum, args.dp_size, args.batch_size, args.seq_len)
            x_all = torch.randint(args.vocab_size, shape, generator=batch_generator).to(device)
            y_all = torch.randint(args.vocab_size, shape, generator=batch_generator).to(device)
            with math_attention_context():
                dense_logits, dense_loss = dense(
                    x_all.reshape(-1, args.seq_len), y_all.reshape(-1, args.seq_len),
                )
                dense_loss.backward()
            logits_by_micro = dense_logits.detach().reshape(
                *shape, args.vocab_size,
            )
            local_mean_loss = torch.zeros((), dtype=dtype, device=device)
            for micro in range(args.grad_accum):
                if tp_rank == 0:
                    x = x_all[micro, dp_rank].clone()
                    y = y_all[micro, dp_rank].clone()
                else:
                    x = torch.empty((args.batch_size, args.seq_len), dtype=torch.long, device=device)
                    y = torch.empty_like(x)
                dist.broadcast(x, src=dp_rank * args.tp_size, group=tp_group)
                dist.broadcast(y, src=dp_rank * args.tp_size, group=tp_group)
                checks.close(x, x_all[micro, dp_rank], "TP batch broadcast X", exact=True)
                checks.close(y, y_all[micro, dp_rank], "TP batch broadcast Y", exact=True)
                sync = wrapped.no_sync() if micro < args.grad_accum - 1 else nullcontext()
                with sync, math_attention_context():
                    logits, loss = wrapped(x, y)
                    checks.close(
                        logits.detach(), logits_by_micro[micro, dp_rank],
                        f"step {step}, micro {micro}, logits",
                    )
                    local_mean_loss += loss.detach() / args.grad_accum
                    (loss / args.grad_accum).backward()
            dist.all_reduce(local_mean_loss, op=dist.ReduceOp.SUM, group=dp_group)
            local_mean_loss /= args.dp_size
            checks.close(local_mean_loss, dense_loss.detach(), f"step {step}, global mean loss")
            checks.finish(f"step {step}: forward / loss / accumulated backward")

            dense_parameters = dict(dense.named_parameters())
            for name, actual in parallel.named_parameters():
                reference_grad = dense_parameters[name].grad
                checks.condition(actual.grad is not None, f"Missing TP gradient: {name}")
                checks.condition(reference_grad is not None, f"Missing dense gradient: {name}")
                if actual.grad is not None and reference_grad is not None:
                    wanted = oracle_shard(
                        name, reference_grad, parallel, candidate, tp_rank, args.tp_size,
                    )
                    checks.close(actual.grad, wanted, f"step {step}, gradient {name}")
            checks.finish(f"step {step}: all parameter gradients vs global dense batch")

            if args.check_overflow:
                first_grad = next(parallel.parameters()).grad
                saved_grad = first_grad.clone()
                if rank == 0:
                    first_grad.view(-1)[0] = float("inf")
                checks.condition(has_nonfinite(parallel), "All ranks must see the injected Inf")
                first_grad.copy_(saved_grad)
                checks.condition(not has_nonfinite(parallel), "Finite gradients must clear the flag")
                checks.finish(f"step {step}: global nonfinite-gradient consensus")

            if args.clip_norm is not None:
                dense_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), args.clip_norm)
                parallel_norm = clip(parallel, args.clip_norm, tp_group)
                checks.close(parallel_norm.double(), dense_norm.double(), f"step {step}, global clip norm")
                for name, actual in parallel.named_parameters():
                    wanted = oracle_shard(
                        name, dense_parameters[name].grad, parallel, candidate, tp_rank, args.tp_size,
                    )
                    checks.close(actual.grad, wanted, f"step {step}, clipped gradient {name}")
                checks.finish(f"step {step}: TP gradient clipping")

            dense_optimizer.step()
            parallel_optimizer.step()
            for name, actual in parallel.named_parameters():
                wanted = oracle_shard(
                    name, dense_parameters[name].detach(), parallel, candidate, tp_rank, args.tp_size,
                )
                checks.close(actual.detach(), wanted, f"step {step}, updated parameter {name}")
                full_state = dense_optimizer.state[dense_parameters[name]]
                shard_state = parallel_optimizer.state[actual]
                for field in ("exp_avg", "exp_avg_sq"):
                    wanted_state = oracle_shard(
                        name, full_state[field], parallel, candidate, tp_rank, args.tp_size,
                    )
                    checks.close(shard_state[field], wanted_state, f"AdamW {field}: {name}")
                checks.close(shard_state["step"], full_state["step"], f"AdamW step: {name}", exact=True)
            checks.finish(f"step {step}: AdamW update and optimizer moments")
        if rank == 0:
            report = {
                "passed": True, "device": args.device, "precision": args.precision,
                "tp_size": args.tp_size, "dp_size": args.dp_size,
                "world_size": world_size, "dropout": 0.0, "attention": args.attention,
                "grad_accum": args.grad_accum, "steps": args.steps,
                "clip_norm": args.clip_norm, "check_overflow": args.check_overflow,
                "threads_per_worker": args.threads,
                "stages": checks.completed,
                "scope": "Model/group math; CPU mode substitutes CUDA RNG tracking only at dropout=0.",
            }
            print("[PASS] TP+DP numerical verification completed.", flush=True)
            if args.report:
                report_path = Path(args.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("model_dense.py"))
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("double", "float32"), default="double")
    parser.add_argument("--attention", choices=("sdpa", "manual"), default="sdpa")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--threads", type=int, default=1,
        help="BLAS/OpenMP and PyTorch intra-op thread limit per process (default: 1)",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--clip-norm", type=float)
    parser.add_argument("--check-overflow", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.project = str(args.project.resolve())
    args.reference = str(args.reference.resolve())
    if args.report:
        args.report = str(args.report.resolve())
    if args.tp_size <= 0:
        parser.error("--tp-size must be positive")
    under_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if under_torchrun:
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size % args.tp_size:
            parser.error("WORLD_SIZE must be divisible by --tp-size")
        expected_dp = world_size // args.tp_size
        if args.dp_size is not None and args.dp_size != expected_dp:
            parser.error("--dp-size does not match WORLD_SIZE / --tp-size")
        args.dp_size = expected_dp
    elif args.dp_size is None or args.dp_size <= 0:
        parser.error("--dp-size must be positive for the local spawn launcher")
    for field in ("grad_accum", "steps", "batch_size", "seq_len", "vocab_size", "layers", "threads", "timeout"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.atol < 0 or args.rtol < 0 or (args.clip_norm is not None and args.clip_norm <= 0):
        parser.error("Tolerances must be nonnegative; --clip-norm must be positive")
    errors = check_project(Path(args.project), model_only=True)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    if not Path(args.reference).is_file():
        parser.error(f"Reference model not found: {args.reference}")
    configure_cpu_threads(args.threads)
    if not under_torchrun or int(os.environ["RANK"]) == 0:
        print(
            f"[INFO] {args.tp_size * args.dp_size} {args.device} workers; "
            f"BLAS/OpenMP thread limit per process: {args.threads}.",
            flush=True,
        )
    try:
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
    except ImportError:
        print("[ERROR] PyTorch is not installed. See README_zh.md for the CPU installation command.")
        return 2
    if not dist.is_available():
        parser.error("This PyTorch build does not include torch.distributed")
    if args.device == "cpu" and not dist.is_gloo_available():
        parser.error("This PyTorch build does not include Gloo")
    if args.device == "cuda":
        needed = int(os.environ.get("LOCAL_WORLD_SIZE", args.tp_size * args.dp_size))
        if not torch.cuda.is_available() or torch.cuda.device_count() < needed:
            parser.error(f"CUDA launch needs {needed} visible local GPUs; CPU mode uses CPU processes.")
        if not dist.is_nccl_available():
            parser.error("CUDA verification requires a PyTorch build with NCCL")
    if under_torchrun:
        worker(int(os.environ["LOCAL_RANK"]), args)
    else:
        with tempfile.TemporaryDirectory(prefix="nanogpt_verify_") as temporary:
            store_path = str(Path(temporary) / "rendezvous")
            mp.spawn(worker, args=(args, store_path), nprocs=args.tp_size * args.dp_size, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
