import argparse
import os

import math
from contextlib import nullcontext



import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from zero1 import Zero1Optimizer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device-type",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/shakespeare_char",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)

    args = parser.parse_args()

    for name in (
        "n_layer", "n_head", "n_embd", "vocab_size", "block_size"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须大于 0")

    if args.n_embd % args.n_head != 0:
        parser.error("n_embd 必须能被 n_head 整除")

    if not 0 <= args.dropout < 1:
        parser.error("dropout 必须在 [0, 1) 内")

    if args.batch_size <= 0:
        parser.error("batch_size 必须大于 0")

    if not 1 <= args.seq_length <= args.block_size:
        parser.error("seq_length 必须在 1 到 block_size 之间")

    for name in ("micro_batch_size", "max_iters", "log_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须大于 0")

    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("grad_clip 必须是有限的非负数")

    return args


def get_zero_batch(
    data_path,
    batch_size,
    seq_length,
    device,
    generator,
):
    if batch_size <= 0 or seq_length <= 0:
        raise ValueError("batch_size 和 seq_length 必须大于 0")

    data = np.memmap(
        data_path,
        dtype=np.uint16,
        mode="r",
    )

    if len(data) <= seq_length:
        raise ValueError("数据的 token 数必须大于 seq_length")

    starts = torch.randint(
        high=len(data) - seq_length,
        size=(batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()

    x = torch.stack([
        torch.from_numpy(
            data[start:start + seq_length].astype(np.int64)
        )
        for start in starts
    ])

    y = torch.stack([
        torch.from_numpy(
            data[start + 1:start + 1 + seq_length].astype(np.int64)
        )
        for start in starts
    ])

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

def train_zero_step(
    model,
    optimizer,
    x,
    y,
    micro_batch_size,
    grad_clip=0.0,
):
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size 必须大于 0")

    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x、y 必须是形状相同的二维张量")

    batch_size = x.size(0)

    if batch_size <= 0 or x.size(1) <= 0:
        raise ValueError("batch_size 和 seq_length 必须大于 0")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    mean_loss = torch.zeros(
        (),
        dtype=torch.float64,
        device=x.device,
    )

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        is_last = end == batch_size

        sync_context = (
            nullcontext()
            if is_last
            else model.no_sync()
        )

        with sync_context:
            _, loss = model(
                x[start:end],
                y[start:end],
            )

            loss_weight = (end - start) / batch_size

            (loss * loss_weight).backward()

        mean_loss.add_(
            loss.detach().to(torch.float64) * loss_weight
        )

    stepped = optimizer.step(grad_clip=grad_clip)
    grad_norm = optimizer.last_grad_norm

    optimizer.zero_grad(set_to_none=True)

    dist.all_reduce(
        mean_loss,
        op=dist.ReduceOp.SUM,
        group=optimizer.dp_group,
    )
    mean_loss.div_(optimizer.dp_size)

    return {
        "loss": float(mean_loss.item()),
        "grad_norm": grad_norm,
        "stepped": stepped,
    }


def main():
    args = parse_args()

    required_env = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    )

    train_data_path = os.path.abspath(
        os.path.join(args.data_dir, "train.bin")
    )

    data_bytes = os.path.getsize(train_data_path)
    token_bytes = np.dtype(np.uint16).itemsize

    if data_bytes % token_bytes != 0:
        raise ValueError(
            "train.bin 的字节数必须是 uint16 大小的整数倍"
        )

    if data_bytes // token_bytes <= args.seq_length:
        raise ValueError(
            "train.bin 的 token 数必须大于 seq_length"
        )

    if any(name not in os.environ for name in required_env):
        raise RuntimeError("请通过分布式启动器运行 train_zero.py")

    local_rank = int(os.environ["LOCAL_RANK"])

    if args.device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用的 CUDA")

        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dp_group = dist.group.WORLD

        torch.manual_seed(args.seed)

        config = GPTConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            dropout=args.dropout,
            bias=not args.no_bias,
        )

        raw_model = GPT(config).to(
            device=device,
            dtype=torch.float32,
        )

        if device.type == "cuda":
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                process_group=dp_group,
            )
        else:
            model = DDP(
                raw_model,
                process_group=dp_group,
            )

        optimizer = Zero1Optimizer(
            model=raw_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            dp_group=dp_group,
        )

        torch.manual_seed(args.seed + rank)

        train_data_generator = torch.Generator(device="cpu")
        train_data_generator.manual_seed(
            args.seed + 10_000 + rank
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_numels = sum(
            param.numel()
            for param in raw_model.parameters()
            if param.requires_grad
        )

        owned_numels = optimizer.rank_numels[optimizer.dp_rank]

        print(
            f"rank={rank}/{world_size}, "
            f"model_numels={total_numels}, "
            f"owned_numels={owned_numels}, "
            f"partition={optimizer.rank_numels}",
            flush=True,
        )

        for iter_num in range(args.max_iters):
            x, y = get_zero_batch(
                data_path=train_data_path,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                device=device,
                generator=train_data_generator,
            )

            result = train_zero_step(
                model=model,
                optimizer=optimizer,
                x=x,
                y=y,
                micro_batch_size=args.micro_batch_size,
                grad_clip=args.grad_clip,
            )

            should_log = (
                iter_num % args.log_interval == 0
                or iter_num == args.max_iters - 1
            )

            if rank == 0 and should_log:
                grad_norm = result["grad_norm"]

                norm_text = (
                    f"{grad_norm:.6f}"
                    if grad_norm is not None
                    else "n/a"
                )

                print(
                    f"iter {iter_num}: "
                    f"loss={result['loss']:.6f}, "
                    f"grad_norm={norm_text}, "
                    f"stepped={result['stepped']}",
                    flush=True,
                )

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()