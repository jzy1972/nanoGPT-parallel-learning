import argparse
import os

import math
from contextlib import nullcontext

import random

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

    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=20)

    parser.add_argument("--out-dir", type=str, default="out-zero")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)

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

    if args.eval_interval < 0:
        raise ValueError("eval_interval 必须大于等于 0")

    if args.eval_iters <= 0:
        raise ValueError("eval_iters 必须大于 0")

    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval 必须大于等于 0")

    if not args.out_dir.strip():
        raise ValueError("out_dir 不能为空")

    if args.resume is not None and not args.resume.strip():
        raise ValueError("resume 不能为空字符串")

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

@torch.no_grad()
def estimate_zero_loss(
    model,
    data_path,
    batch_size,
    seq_length,
    micro_batch_size,
    eval_iters,
    device,
    dp_group,
    seed,
):
    if batch_size <= 0 or seq_length <= 0:
        raise ValueError("batch_size、seq_length 必须大于 0")

    if micro_batch_size <= 0 or eval_iters <= 0:
        raise ValueError("micro_batch_size、eval_iters 必须大于 0")

    dp_rank = dist.get_rank(group=dp_group)

    # 评估使用独立的数据生成器
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + dp_rank)

    # 取出 DDP 内部的完整模型
    raw_model = model.module if isinstance(model, DDP) else model
    was_training = raw_model.training

    # totals[0]：loss 总和
    # totals[1]：参与评估的 token 总数
    totals = torch.zeros(
        2,
        dtype=torch.float64,
        device=device,
    )

    try:
        raw_model.eval()

        for _ in range(eval_iters):
            x, y = get_zero_batch(
                data_path=data_path,
                batch_size=batch_size,
                seq_length=seq_length,
                device=device,
                generator=generator,
            )

            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)

                micro_x = x[start:end]
                micro_y = y[start:end]

                _, loss = raw_model(micro_x, micro_y)

                token_count = micro_y.numel()

                totals[0].add_(
                    loss.detach().to(dtype=torch.float64)
                    * token_count
                )

                totals[1].add_(token_count)

        dist.all_reduce(
            totals,
            op=dist.ReduceOp.SUM,
            group=dp_group,
        )

        return float((totals[0] / totals[1]).item())

    finally:
        raw_model.train(was_training)

def capture_rng_state(device, train_data_generator):
    numpy_state = np.random.get_state()

    return {
        "device_type": device.type,
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(
                numpy_state[1].astype(np.int64),
                dtype=torch.int64,
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        ),
        "train_data_generator": train_data_generator.get_state().clone(),
    }

def atomic_torch_save(payload, path):
    path = os.fspath(path)
    temporary_path = path + ".tmp"

    try:
        with open(temporary_path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def check_checkpoint_errors(error, dp_group, phase):
    errors = [None] * dist.get_world_size(group=dp_group)

    dist.all_gather_object(
        errors,
        error,
        group=dp_group,
    )

    failures = [
        f"dp_rank={rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]

    if failures:
        raise RuntimeError(
            f"checkpoint {phase}失败：" + "; ".join(failures)
        )

def save_zero_checkpoint(
    out_dir,
    model,
    optimizer,
    next_iter,
    train_data_generator,
    device,
    train_args,
):
    dp_group = optimizer.dp_group
    dp_rank = optimizer.dp_rank
    dp_size = optimizer.dp_size

    raw_model = model.module if isinstance(model, DDP) else model

    # 所有 rank 必须保存到同一个目录、同一个迭代位置
    plan = (os.path.abspath(out_dir), next_iter)
    plans = [None] * dp_size

    dist.all_gather_object(
        plans,
        plan,
        group=dp_group,
    )

    if any(item != plans[0] for item in plans):
        raise ValueError("各 rank 的 checkpoint 路径或 next_iter 不一致")

    if any(type(item[1]) is not int or item[1] < 0 for item in plans):
        raise ValueError("next_iter 必须是非负整数")

    checkpoint_dir = os.path.join(
        plan[0],
        f"iter_{next_iter:08d}",
    )

    error = None

    try:
        if any(param.grad is not None for param in raw_model.parameters()):
            raise ValueError("请在完整训练迭代结束并清空梯度后保存")

        if dp_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=False)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(error, dp_group, "创建目录")

    error = None

    try:
        rank_payload = {
            "format_version": 1,
            "zero_stage": 1,
            "dp_size": dp_size,
            "dp_rank": dp_rank,
            "next_iter": next_iter,
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(
                device,
                train_data_generator,
            ),
        }

        atomic_torch_save(
            rank_payload,
            os.path.join(
                checkpoint_dir,
                f"rank_{dp_rank:05d}.pt",
            ),
        )

        if dp_rank == 0:
            model_payload = {
                "format_version": 1,
                "zero_stage": 1,
                "dp_size": dp_size,
                "next_iter": next_iter,
                "model_args": dict(vars(raw_model.config)),
                "train_args": dict(train_args),
                "model": raw_model.state_dict(),
            }

            atomic_torch_save(
                model_payload,
                os.path.join(checkpoint_dir, "model.pt"),
            )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(error, dp_group, "写入模型和分片")

    error = None

    try:
        if dp_rank == 0:
            manifest = {
                "format_version": 1,
                "zero_stage": 1,
                "complete": True,
                "dp_size": dp_size,
                "next_iter": next_iter,
                "files": ["model.pt"] + [
                    f"rank_{rank:05d}.pt"
                    for rank in range(dp_size)
                ],
            }

            atomic_torch_save(
                manifest,
                os.path.join(checkpoint_dir, "complete.pt"),
            )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(error, dp_group, "写入完成标记")

    return checkpoint_dir

def restore_rng_state(state, device, train_data_generator):
    if state["device_type"] != device.type:
        raise ValueError("恢复时的 device_type 必须与保存时一致")

    if train_data_generator.device.type != "cpu":
        raise ValueError("训练数据生成器必须位于 CPU")

    if device.type == "cuda" and state["torch_cuda"] is None:
        raise ValueError("checkpoint 缺少 CUDA 随机数状态")

    numpy_state = state["numpy"]

    numpy_tuple = (
        numpy_state["algorithm"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    )

    random.setstate(state["python"])
    np.random.set_state(numpy_tuple)

    torch.set_rng_state(
        state["torch_cpu"].cpu()
    )

    if device.type == "cuda":
        torch.cuda.set_rng_state(
            state["torch_cuda"].cpu(),
            device=device,
        )

    train_data_generator.set_state(
        state["train_data_generator"].cpu()
    )

def load_zero_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    train_data_generator,
    device,
    train_args,
):
    dp_group = optimizer.dp_group
    dp_rank = optimizer.dp_rank
    dp_size = optimizer.dp_size

    raw_model = model.module if isinstance(model, DDP) else model

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    paths = [None] * dp_size

    dist.all_gather_object(
        paths,
        checkpoint_dir,
        group=dp_group,
    )

    if any(path != paths[0] for path in paths):
        raise ValueError("各 rank 的恢复目录不一致")

    error = None

    try:
        if optimizer.model is not raw_model:
            raise ValueError("优化器必须引用当前模型的参数")

        manifest = torch.load(
            os.path.join(checkpoint_dir, "complete.pt"),
            map_location="cpu",
            weights_only=True,
        )

        if manifest.get("complete") is not True:
            raise ValueError("checkpoint 尚未完成保存")

        expected_files = ["model.pt"] + [
            f"rank_{rank:05d}.pt"
            for rank in range(dp_size)
        ]

        if manifest.get("files") != expected_files:
            raise ValueError("checkpoint 的文件列表或 DP 规模不匹配")

        for filename in expected_files:
            path = os.path.join(checkpoint_dir, filename)

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"checkpoint 缺少 {filename}"
                )

        model_payload = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )

        rank_payload = torch.load(
            os.path.join(
                checkpoint_dir,
                f"rank_{dp_rank:05d}.pt",
            ),
            map_location="cpu",
            weights_only=True,
        )

        for payload in (manifest, model_payload, rank_payload):
            if payload.get("format_version") != 1:
                raise ValueError("不支持的 checkpoint 版本")

            if payload.get("zero_stage") != 1:
                raise ValueError("checkpoint 不是 ZeRO-1 状态")

            if payload.get("dp_size") != dp_size:
                raise ValueError("恢复时的 DP 规模与保存时不同")

        next_iter = manifest["next_iter"]

        if type(next_iter) is not int or next_iter < 0:
            raise ValueError("checkpoint 的 next_iter 非法")

        if any(
            payload.get("next_iter") != next_iter
            for payload in (model_payload, rank_payload)
        ):
            raise ValueError("模型和分片的迭代位置不一致")

        if rank_payload.get("dp_rank") != dp_rank:
            raise ValueError("读取了其他 rank 的优化器分片")

        if model_payload["model_args"] != dict(vars(raw_model.config)):
            raise ValueError("恢复时的 GPTConfig 与保存时不同")

        optimizer_state = rank_payload["optimizer"]

        if optimizer_state["layout"] != optimizer._checkpoint_layout():
            raise ValueError("恢复时的参数分配与保存时不同")

        fixed_keys = (
            "seed",
            "data_dir",
            "batch_size",
            "seq_length",
            "micro_batch_size",
            "grad_clip",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
        )

        saved_args = model_payload["train_args"]

        for key in fixed_keys:
            if key not in saved_args or key not in train_args:
                raise ValueError(f"缺少训练参数 {key}")

            if saved_args[key] != train_args[key]:
                raise ValueError(
                    f"恢复时的训练参数 {key} 不一致"
                )

        if train_args["max_iters"] < next_iter:
            raise ValueError(
                "max_iters 不能小于 checkpoint 的 next_iter"
            )

        if rank_payload["rng"]["device_type"] != device.type:
            raise ValueError(
                "恢复时的 device_type 与保存时不同"
            )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(error, dp_group, "读取和校验")

    counters = [None] * dp_size

    dist.all_gather_object(
        counters,
        next_iter,
        group=dp_group,
    )

    if any(counter != counters[0] for counter in counters):
        raise ValueError("各 rank 的 checkpoint 迭代位置不一致")

    error = None

    try:
        raw_model.load_state_dict(
            model_payload["model"],
            strict=True,
        )

        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(
        error,
        dp_group,
        "恢复模型和优化器",
    )

    error = None

    try:
        restore_rng_state(
            rank_payload["rng"],
            device,
            train_data_generator,
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    check_checkpoint_errors(
        error,
        dp_group,
        "恢复随机数状态",
    )

    return next_iter
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

    val_data_path = None

    if args.eval_interval > 0:
        val_data_path = os.path.abspath(
        os.path.join(args.data_dir, "val.bin")
    )

        val_bytes = os.path.getsize(val_data_path)
        token_bytes = np.dtype(np.uint16).itemsize

        if val_bytes % token_bytes != 0:
            raise ValueError("val.bin 的大小不符合 uint16 格式")

        if val_bytes // token_bytes <= args.seq_length:
            raise ValueError("val.bin 的 token 数必须大于 seq_length")

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
 
        start_iter = 0

        if args.resume is not None:
            start_iter = load_zero_checkpoint(
                checkpoint_dir=args.resume,
                model=model,
                optimizer=optimizer,
                train_data_generator=train_data_generator,
                device=device,
                train_args=vars(args),
            )

            if optimizer.dp_rank == 0:
                print(
                    f"resumed checkpoint: {args.resume}, "
                    f"next_iter={start_iter}",
                    flush=True,
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

        for iter_num in range(start_iter,args.max_iters):
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

            completed_iters = iter_num + 1

            should_eval = (
                    args.eval_interval > 0
                    and (completed_iters % args.eval_interval == 0 or completed_iters == args.max_iters)
                )

            if should_eval:
                eval_losses = {}

                for split, data_path, seed_offset in (
                    ("train", train_data_path, 20_000),
                    ("val", val_data_path, 30_000),
                ):
                    eval_losses[split] = estimate_zero_loss(
                    model=model,
                    data_path=data_path,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    eval_iters=args.eval_iters,
                    device=device,
                    dp_group=dp_group,
                    seed=args.seed + seed_offset,
                    )

                if dist.get_rank() == 0:
                    print(
                        f"eval iter {iter_num}: "
                        f"train loss={eval_losses['train']:.6f}, "
                        f"val loss={eval_losses['val']:.6f}",
                        flush=True,
                    )

            completed_iters = iter_num + 1

            should_checkpoint = (
                args.checkpoint_interval > 0
                and (
                completed_iters % args.checkpoint_interval == 0
                or completed_iters == args.max_iters
                )
            )

            if should_checkpoint:
                checkpoint_dir = save_zero_checkpoint(
                    out_dir=args.out_dir,
                    model=model,
                    optimizer=optimizer,
                    next_iter=completed_iters,
                    train_data_generator=train_data_generator,
                    device=device,
                    train_args=vars(args),
                )

                if optimizer.dp_rank == 0:
                    print(
                        f"saved checkpoint: {checkpoint_dir}",
                        flush=True,
                    )

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()