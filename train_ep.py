import argparse
import math
import os
from datetime import timedelta
import numpy as np

import torch
import torch.distributed as dist

from model import GPT, GPTConfig
from parallel_ep import (
    setup_ep,
    get_expert_range,
    sync_shared_parameters,
    build_ep_optimizer,
    step_ep_optimizer,
)

def positive_int(value):
    value = int(value)

    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")

    return value


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device-type",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--n-layer", type=positive_int, default=12)
    parser.add_argument("--n-head", type=positive_int, default=12)
    parser.add_argument("--n-embd", type=positive_int, default=768)
    parser.add_argument("--block-size", type=positive_int, default=1024)
    parser.add_argument("--vocab-size", type=positive_int, default=50304)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")

    parser.add_argument("--n-experts", type=positive_int, default=4)
    parser.add_argument("--moe-top-k", type=positive_int, default=1)
    parser.add_argument("--moe-aux-loss-coef", type=float, default=0.01)

    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=positive_int, default=12)
    parser.add_argument("--micro-batch-size", type=positive_int, default=4)
    parser.add_argument("--seq-length", type=positive_int, default=256)
    parser.add_argument("--max-iters", type=positive_int, default=100)
    parser.add_argument("--log-interval", type=positive_int, default=10)

    parser.add_argument("--eval-interval",type=int,default=0,)
    parser.add_argument("--eval-iters",type=positive_int,default=10,)

    args = parser.parse_args()

    if args.n_embd % args.n_head != 0:
        parser.error("n_embd 必须能被 n_head 整除")

    if args.moe_top_k > args.n_experts:
        parser.error("moe_top_k 不能超过 n_experts")

    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("dropout 必须在 [0, 1) 内")

    if args.seq_length > args.block_size:
        parser.error("seq_length 不能超过 block_size")

    if args.eval_interval < 0:
        parser.error("eval_interval 必须大于等于 0")

    for name in (
        "moe_aux_loss_coef",
        "learning_rate",
        "weight_decay",
        "grad_clip",
    ):
        value = getattr(args, name)

        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} 必须是有限的非负数")

    for name in ("beta1", "beta2"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"{name} 必须在 [0, 1) 内")

    return args

def initialize_ep(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif args.local_rank is not None:
        local_rank = args.local_rank
    else:
        raise RuntimeError("请通过分布式启动器启动 train_ep.py")

    if args.device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
            timeout=timedelta(minutes=10),
        )
    else:
        device = torch.device("cpu")

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(minutes=10),
        )

    parallel = setup_ep()

    if args.n_experts % parallel["ep_size"] != 0:
        raise ValueError(
            f"n_experts={args.n_experts} 必须能被 "
            f"ep_size={parallel['ep_size']} 整除"
        )

    return device, parallel

def get_ep_batch(
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
        raise ValueError("数据 token 数必须大于 seq_length")

    starts = torch.randint(
        len(data) - seq_length,
        (batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()

    x = torch.stack([
        torch.from_numpy(
            data[i:i + seq_length].astype(np.int64)
        )
        for i in starts
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + 1 + seq_length].astype(np.int64)
        )
        for i in starts
    ])

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

def train_ep_step(
    model,
    optimizer,
    parallel,
    token_ids,
    targets,
    micro_batch_size,
    grad_clip=0.0,
):
    if token_ids.ndim != 2 or targets.shape != token_ids.shape:
        raise ValueError("token_ids 和 targets 必须具有相同的二维形状")

    batch_size = token_ids.size(0)

    if batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("批量大小必须大于 0")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    local_loss = torch.zeros(
        (),
        dtype=torch.float64,
        device=token_ids.device,
    )

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)

        micro_x = token_ids[start:end]
        micro_y = targets[start:end]

        _, loss = model(micro_x, micro_y)

        weight = (end - start) / batch_size

        (loss * weight).backward()

        local_loss.add_(
            loss.detach().to(dtype=torch.float64) * weight
        )

    step_result = step_ep_optimizer(
        model=model,
        optimizer=optimizer,
        parallel=parallel,
        grad_clip=grad_clip,
    )

    # 所有进程共同统计用于日志的平均 loss
    dp_group = parallel["shared_dp_group"]

    dist.all_reduce(
        local_loss,
        op=dist.ReduceOp.SUM,
        group=dp_group,
    )

    local_loss.div_(
        dist.get_world_size(group=dp_group)
    )

    return {
        "loss": float(local_loss.item()),
        "grad_norm": step_result["grad_norm"],
        "stepped": step_result["stepped"],
    }

@torch.no_grad()
def evaluate_ep(
    model,
    parallel,
    data_path,
    batch_size,
    seq_length,
    eval_iters,
    device,
    generator,
):
    if eval_iters <= 0:
        raise ValueError("eval_iters 必须大于 0")

    was_training = model.training
    model.eval()

    # stats[0]：交叉熵总和
    # stats[1]：参与统计的 token 数
    stats = torch.zeros(
        2,
        dtype=torch.float64,
        device=device,
    )

    try:
        for _ in range(eval_iters):
            token_ids, targets = get_ep_batch(
                data_path=data_path,
                batch_size=batch_size,
                seq_length=seq_length,
                device=device,
                generator=generator,
            )

            _, loss = model(token_ids, targets)

            token_count = targets.numel()

            stats[0].add_(
                loss.to(dtype=torch.float64) * token_count
            )
            stats[1].add_(token_count)

        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
            group=parallel["shared_dp_group"],
        )

        return float((stats[0] / stats[1]).item())

    finally:
        model.train(was_training)

def main():
    args = parse_args()

    try:
        device, parallel = initialize_ep(args)

        config = GPTConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            bias=not args.no_bias,
            use_moe=True,
            n_experts=args.n_experts,
            moe_top_k=args.moe_top_k,
            moe_aux_loss_coef=args.moe_aux_loss_coef,
        )

        torch.manual_seed(
            args.seed + parallel["ep_rank"]
        )

        model = GPT(
            config,
            ep_group=parallel["ep_group"],
        ).to(device)

        sync_shared_parameters(model, parallel)

        optimizer = build_ep_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
        )

        optimizer.zero_grad(set_to_none=True)

        expert_start, expert_end = get_expert_range(
            args.n_experts,
            parallel["ep_size"],
            parallel["ep_rank"],
        )

        local_params = sum(
            param.numel()
            for param in model.parameters()
        )

        print(
            f"rank={parallel['world_rank']} "
            f"ep_size={parallel['ep_size']} "
            f"experts=[{expert_start}, {expert_end}) "
            f"local_params={local_params}",
            flush=True,
        )

        train_data_path = os.path.abspath(
            os.path.join(args.data_dir, "train.bin")
        )

        train_data_generator = torch.Generator(device="cpu")
        train_data_generator.manual_seed(
            args.seed + 10_000 + parallel["ep_rank"]
        )

        val_data_path = os.path.abspath(os.path.join(args.data_dir, "val.bin"))

        eval_data_generator = torch.Generator(device="cpu")
        eval_data_generator.manual_seed(args.seed + 20_000 + parallel["ep_rank"])

        for iter_num in range(args.max_iters):
            token_ids, targets = get_ep_batch(
                data_path=train_data_path,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                device=device,
                generator=train_data_generator,
            )

            result = train_ep_step(
                model=model,
                optimizer=optimizer,
                parallel=parallel,
                token_ids=token_ids,
                targets=targets,
                micro_batch_size=args.micro_batch_size,
                grad_clip=args.grad_clip,
            )

            should_log = (
                iter_num % args.log_interval == 0
                or iter_num == args.max_iters - 1
            )

            completed_iters = iter_num + 1

            should_eval = (
                args.eval_interval > 0
                and (completed_iters % args.eval_interval == 0 or completed_iters == args.max_iters)
            )

            if should_eval:
                val_loss = evaluate_ep(
                    model=model,
                    parallel=parallel,
                    data_path=val_data_path,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    eval_iters=args.eval_iters,
                    device=device,
                    generator=eval_data_generator,
                )

                if parallel["world_rank"] == 0:
                    print(
                        f"eval after {completed_iters} steps: "
                        f"val_loss={val_loss:.6f}",
                        flush=True,
                    )

            if parallel["world_rank"] == 0 and should_log:
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