import torch.distributed as dist
import argparse
import os
import torch
import numpy as np

import math

from model import GPTConfig
from model_pp import GPTStage

import torch.nn.functional as F

from parallel_utils import(
    build_pp_dp_groups,
    build_tied_weight_group,
    sync_tied_weights,
    send_forward,
    recv_forward,
    send_backward,
    recv_backward,
    get_dp_grad_weight,
    sync_dp_gradients,
    sync_tied_weight_gradients,
    global_has_nonfinite_gradients,
    clip_stage_gradients,
)

def setup_parallel(pp_size):
    assert dist.is_initialized()
    assert pp_size>0

    world_rank=dist.get_rank()
    world_size=dist.get_world_size()

    assert world_size%pp_size==0

    dp_size=world_size//pp_size
    dp_rank=world_rank//pp_size
    pp_rank=world_rank%pp_size

    pp_group,dp_group=build_pp_dp_groups(pp_size)
    tied_group = build_tied_weight_group(pp_size)

    assert pp_group is not None
    assert dp_group is not None
    assert dist.get_rank(group=pp_group)==pp_rank
    assert dist.get_rank(group=dp_group)==dp_rank

    pp_first_rank=dp_rank*pp_size
    pp_last_rank=pp_first_rank+pp_size-1

    prev_rank=world_rank-1 if pp_rank>0 else None
    next_rank=world_rank+1 if pp_rank<pp_size-1 else None

    return {
        "world_rank":world_rank,
        "world_size":world_size,
        "pp_rank":pp_rank,
        "dp_rank":dp_rank,
        "dp_size":dp_size,
        "pp_group":pp_group,
        "dp_group":dp_group,
        "tied_group":tied_group,
        "pp_first_rank":pp_first_rank,
        "pp_last_rank":pp_last_rank,
        "prev_rank":prev_rank,
        "next_rank":next_rank,
    }

def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--pp-size",type=int,required=True)
    parser.add_argument("--device-type",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--seed",type=int,default=1337)

    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")

    parser.add_argument("--data-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)

    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-iters",type=int,default=1000,)

    parser.add_argument("--log-interval",type=int,default=10)

    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=20)

    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)

    args=parser.parse_args()

    if not 1 <= args.pp_size <= args.n_layer:
        parser.error("需要满足 1 <= pp_size <= n_layer")

    return args

@torch.no_grad()
def forward_eval_microbatch(
    stage,
    token_ids,
    parallel,
    micro_batch_size,
    seq_length,
    device,
    activation_dtype=torch.float32,
):
    if stage.is_first:
        assert token_ids is not None
        assert token_ids.shape == (micro_batch_size, seq_length)

        stage_input = token_ids
    else:
        stage_input = recv_forward(
            shape=(
                micro_batch_size,
                seq_length,
                stage.config.n_embd,
            ),
            dtype=activation_dtype,
            device=device,
            src_rank=parallel["prev_rank"],
            pp_group=parallel["pp_group"],
            requires_grad=False,
        )

    stage_output = stage(stage_input)

    if stage.is_last:
        return stage_output

    assert stage_output.shape == (
        micro_batch_size,
        seq_length,
        stage.config.n_embd,
    )
    assert stage_output.dtype == activation_dtype

    send_forward(
        tensor=stage_output,
        dst_rank=parallel["next_rank"],
        pp_group=parallel["pp_group"],
    )

    return None

def forward_microbatch(
    stage,
    token_ids,
    parallel,
    micro_batch_size,
    seq_length,
    device,
    activation_dtype=torch.float32,
):
    hidden_shape = (
        micro_batch_size,
        seq_length,
        stage.config.n_embd,
    )

    # 首 stage 使用 token IDs，其余 stage 接收上游激活
    if stage.is_first:
        assert token_ids is not None
        assert token_ids.shape == (micro_batch_size, seq_length)
        assert token_ids.dtype == torch.long

        stage_input = token_ids
    else:
        stage_input = recv_forward(
            shape=hidden_shape,
            dtype=activation_dtype,
            device=device,
            src_rank=parallel["prev_rank"],
            pp_group=parallel["pp_group"],
        )

    stage_output = stage(stage_input)

    if not stage.is_last:
        assert stage_output.shape == hidden_shape
        assert stage_output.dtype == activation_dtype

        send_forward(
            tensor=stage_output,
            dst_rank=parallel["next_rank"],
            pp_group=parallel["pp_group"],
        )

    return stage_input, stage_output

def backward_microbatch(
        stage,
        stage_input,
        stage_output,
        parallel,
        targets=None,
        loss_weight=1.0,
):
    loss = None

    if stage.is_last:
        assert targets is not None
        assert targets.shape==stage_output.shape[:2]

        valid_count=(targets!=-1).sum()

        loss_sum=F.cross_entropy(
            stage_output.reshape(-1,stage_output.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )

        loss=loss_sum/valid_count.clamp_min(1)

        (loss*loss_weight).backward()

    else:
        grad_output=recv_backward(
            stage_output=stage_output,
            src_rank=parallel["next_rank"],
            pp_group=parallel["pp_group"],
        )

        stage_output.backward(grad_output)

    if not stage.is_first:
        assert stage_input.grad is not None

        send_backward(
            grad_input=stage_input.grad,
            dst_rank=parallel["prev_rank"],
            pp_group=parallel["pp_group"],
        )

    return loss.detach() if loss is not None else None


def gpipe_forward_backward(
    stage,
    token_ids,
    targets,
    parallel,
    batch_size,
    seq_length,
    micro_batch_size,
    device,
    activation_dtype=torch.float32,
):
    assert batch_size > 0
    assert micro_batch_size > 0
    assert 0 < seq_length <= stage.config.block_size

    if stage.is_first:
        assert token_ids is not None
        assert token_ids.shape == (batch_size, seq_length)

    if stage.is_last:
        assert targets is not None
        assert targets.shape == (batch_size, seq_length)

        total_valid = max(1, (targets != -1).sum().item())
        assert total_valid > 0

    cache = []
    batch_loss = None

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)

        micro_tokens = (
            token_ids[start:end] if stage.is_first else None
        )

        stage_input, stage_output = forward_microbatch(
            stage=stage,
            token_ids=micro_tokens,
            parallel=parallel,
            micro_batch_size=end - start,
            seq_length=seq_length,
            device=device,
            activation_dtype=activation_dtype,
        )

        cache.append((stage_input, stage_output, start, end))

    while cache:
        stage_input, stage_output, start, end = cache.pop()

        if stage.is_last:
            micro_targets = targets[start:end]
            micro_valid = (micro_targets != -1).sum().item()
            loss_weight = micro_valid / total_valid
        else:
            micro_targets = None
            loss_weight = 1.0

        loss = backward_microbatch(
            stage=stage,
            stage_input=stage_input,
            stage_output=stage_output,
            parallel=parallel,
            targets=micro_targets,
            loss_weight=loss_weight,
        )

        if stage.is_last:
            weighted_loss = loss * loss_weight
            batch_loss = (
                weighted_loss
                if batch_loss is None
                else batch_loss + weighted_loss
            )

        del stage_input, stage_output

    return batch_loss

def sync_stage_gradients(stage, targets, parallel, device):
    grad_weight = get_dp_grad_weight(
        stage=stage,
        targets=targets,
        parallel=parallel,
        device=device,
    )

    sync_dp_gradients(
        stage=stage,
        dp_group=parallel["dp_group"],
        dp_size=parallel["dp_size"],
        grad_weight=grad_weight,
    )

    sync_tied_weight_gradients(
        stage=stage,
        tied_group=parallel["tied_group"],
    )

    return grad_weight

def step_stage_optimizer(
        stage,
        optimizer,
        parallel,
        device,
        grad_clip,
):
    grad_clip=float(grad_clip)

    if not math.isfinite(grad_clip) or grad_clip<0:
        raise ValueError(f"grad_clip必须是有限的正数")

    grad_norm=None

    if grad_clip>0:
        grad_norm=clip_stage_gradients(
            stage=stage,
            max_norm=grad_clip,
            parallel=parallel,
            device=device,
        )

        can_step=grad_norm is not None

    else:
        can_step=not global_has_nonfinite_gradients(
            stage=stage,
            device=device
        )

    if can_step:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    return{
        "stepped": can_step,
        "grad_norm": (
            grad_norm.item()
            if grad_norm is not None
            else None
        ),
    }

def build_stage_optimizer(
    stage,
    learning_rate,
    weight_decay,
    beta1,
    beta2,
):
    params = [
        param
        for param in stage.parameters()
        if param.requires_grad
    ]

    decay_params = [
        param for param in params
        if param.dim() >= 2
    ]

    no_decay_params = [
        param for param in params
        if param.dim() < 2
    ]

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        foreach=False,
        fused=False,
    )

def get_pp_batch(
    stage,
    parallel,
    data_path,
    batch_size,
    seq_length,
    device,
    generator,
):
    token_ids = None

    if stage.is_first:
        data = np.memmap(
            data_path,
            dtype=np.uint16,
            mode="r",
        )

        indices = torch.randint(
            len(data) - seq_length,
            (batch_size,),
            generator=generator,
            device="cpu",
        ).tolist()

        token_ids = torch.stack([
            torch.from_numpy(
                data[i:i + seq_length].astype(np.int64)
            )
            for i in indices
        ])

        targets = torch.stack([
            torch.from_numpy(
                data[i + 1:i + 1 + seq_length].astype(np.int64)
            )
            for i in indices
        ])

        token_ids = token_ids.to(device)
        targets = targets.to(device)

    else:
        targets = torch.empty(
            (batch_size, seq_length),
            dtype=torch.long,
            device=device,
        )

    dist.broadcast(
        targets,
        src=parallel["pp_first_rank"],
        group=parallel["pp_group"],
    )

    return (
        token_ids,
        targets if stage.is_last else None,
    )

@torch.no_grad()
def aggregate_pp_loss(
    stage,
    local_loss,
    grad_weight,
    parallel,
    device,
):
    global_loss = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )

    if stage.is_last:
        assert local_loss is not None

        # 没有有效标签的 DP 副本贡献为 0
        if grad_weight > 0:
            global_loss.copy_(
                local_loss.detach().to(
                    device=device,
                    dtype=torch.float64,
                )
            )
            global_loss.mul_(grad_weight)

        dist.all_reduce(
            global_loss,
            op=dist.ReduceOp.SUM,
            group=parallel["dp_group"],
        )

        global_loss.div_(parallel["dp_size"])

    dist.broadcast(
        global_loss,
        src=parallel["pp_last_rank"],
        group=parallel["pp_group"],
    )

    return global_loss.item()

def train_pp_step(
    stage,
    optimizer,
    parallel,
    token_ids,
    targets,
    batch_size,
    seq_length,
    micro_batch_size,
    device,
    grad_clip,
    activation_dtype=torch.float32,
):
    local_loss = gpipe_forward_backward(
        stage=stage,
        token_ids=token_ids,
        targets=targets,
        parallel=parallel,
        batch_size=batch_size,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        device=device,
        activation_dtype=activation_dtype,
    )

    grad_weight = sync_stage_gradients(
        stage=stage,
        targets=targets,
        parallel=parallel,
        device=device,
    )

    update_result = step_stage_optimizer(
        stage=stage,
        optimizer=optimizer,
        parallel=parallel,
        device=device,
        grad_clip=grad_clip,
    )

    global_loss = aggregate_pp_loss(
        stage=stage,
        local_loss=local_loss,
        grad_weight=grad_weight,
        parallel=parallel,
        device=device,
    )

    return {
        "loss": global_loss,
        "stepped": update_result["stepped"],
        "grad_norm": update_result["grad_norm"],
    }

@torch.no_grad()
def evaluate_pp(
    stage,
    parallel,
    data_path,
    batch_size,
    seq_length,
    micro_batch_size,
    eval_iters,
    device,
    generator,
    activation_dtype=torch.float32,
):
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size 必须大于 0")

    if eval_iters <= 0:
        raise ValueError("eval_iters 必须大于 0")

    if not 1 <= seq_length <= stage.config.block_size:
        raise ValueError("seq_length 超出模型支持的范围")

    was_training = stage.training
    stage.eval()

    try:
        # stats[0]：loss 总和
        # stats[1]：有效 token 总数
        stats = torch.zeros(
            2,
            dtype=torch.float64,
            device=device,
        )

        for _ in range(eval_iters):
            token_ids, targets = get_pp_batch(
                stage=stage,
                parallel=parallel,
                data_path=data_path,
                batch_size=batch_size,
                seq_length=seq_length,
                device=device,
                generator=generator,
            )

            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)
                current_batch_size = end - start

                micro_tokens = (
                    token_ids[start:end]
                    if stage.is_first
                    else None
                )

                logits = forward_eval_microbatch(
                    stage=stage,
                    token_ids=micro_tokens,
                    parallel=parallel,
                    micro_batch_size=current_batch_size,
                    seq_length=seq_length,
                    device=device,
                    activation_dtype=activation_dtype,
                )

                if stage.is_last:
                    micro_targets = targets[start:end]

                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        micro_targets.reshape(-1),
                        ignore_index=-1,
                        reduction="sum",
                    )

                    valid_tokens = (micro_targets != -1).sum()

                    stats[0].add_(
                        loss_sum.to(dtype=torch.float64)
                    )
                    stats[1].add_(
                        valid_tokens.to(dtype=torch.float64)
                    )

        # 各条流水线的最后阶段之间汇总统计量
        if stage.is_last:
            dist.all_reduce(
                stats,
                op=dist.ReduceOp.SUM,
                group=parallel["dp_group"],
            )

        # 每条流水线内部，让所有阶段拿到汇总结果
        dist.broadcast(
            stats,
            src=parallel["pp_last_rank"],
            group=parallel["pp_group"],
        )

        if stats[1].item() == 0:
            raise ValueError("整个评估过程没有有效标签")

        return (stats[0] / stats[1]).item()

    finally:
        stage.train(was_training)

def main():
    args=parse_args()

    local_rank=int(os.environ["LOCAL_RANK"])

    if args.device_type=="cuda":
        device=torch.device("cuda",local_rank)
        torch.cuda.set_device(device)
    else:
        device=torch.device("cpu")


    config=GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        dropout=args.dropout,
        bias=not args.no_bias,
    )

    try:
        # 所有设备、所有进程都执行这些检查
        if args.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if args.micro_batch_size <= 0:
            raise ValueError("micro_batch_size 必须大于 0")

        if not 1 <= args.seq_length <= args.block_size:
            raise ValueError(
                "seq_length 必须在 1 到 block_size 之间"
            )

        if args.max_iters <= 0:
            raise ValueError("max_iters 必须大于 0")

        if args.log_interval <= 0:
            raise ValueError("log_interval 必须大于 0")

        train_data_path = os.path.abspath(
            os.path.join(args.data_dir, "train.bin")
        )

        data_bytes = os.path.getsize(train_data_path)
        token_bytes = np.dtype(np.uint16).itemsize

        if args.eval_interval < 0:
            raise ValueError("eval_interval 不能小于 0")

        if args.eval_iters <= 0:
            raise ValueError("eval_iters 必须大于 0")

        val_data_path = None

        if args.eval_interval > 0:
            val_data_path = os.path.abspath(
                os.path.join(args.data_dir, "val.bin")
            )

            val_bytes = os.path.getsize(val_data_path)

            if val_bytes % token_bytes != 0:
               raise ValueError("val.bin 的大小不符合 uint16 格式")

            if val_bytes // token_bytes <= args.seq_length:
               raise ValueError(
                "val.bin 的 token 数必须大于 seq_length"
               )

        if data_bytes % token_bytes != 0:
            raise ValueError("train.bin 的大小不符合 uint16 格式")

        if data_bytes // token_bytes <= args.seq_length:
            raise ValueError(
                "train.bin 的 token 数必须大于 seq_length"
            )

        # 通信后端根据设备选择
        if args.device_type == "cuda":
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=device,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
            )

        parallel = setup_parallel(args.pp_size)

        torch.manual_seed(args.seed+parallel["pp_rank"])


        stage=GPTStage(
            config,
            pp_size=args.pp_size,
            pp_rank=parallel["pp_rank"],
        ).to(device)

        sync_tied_weights(
            stage,
            tied_group=parallel["tied_group"],
            src_rank=parallel["pp_first_rank"],
        )

        optimizer = build_stage_optimizer(
            stage=stage,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
        )

        optimizer.zero_grad(set_to_none=True)

        train_data_generator = torch.Generator(device="cpu")
        train_data_generator.manual_seed(
            args.seed + 10_000 + parallel["dp_rank"]
        )

        eval_data_generator = torch.Generator(device="cpu")
        eval_data_generator.manual_seed(
            args.seed + 20_000 + parallel["dp_rank"]
        )

        stage.train()

        for iter_num in range(args.max_iters):
            token_ids, targets = get_pp_batch(
               stage=stage,
               parallel=parallel,
               data_path=train_data_path,
               batch_size=args.batch_size,
               seq_length=args.seq_length,
               device=device,
               generator=train_data_generator,
            )

            result = train_pp_step(
               stage=stage,
               optimizer=optimizer,
               parallel=parallel,
               token_ids=token_ids,
               targets=targets,
               batch_size=args.batch_size,
               seq_length=args.seq_length,
               micro_batch_size=args.micro_batch_size,
               device=device,
               grad_clip=args.grad_clip,
               activation_dtype=torch.float32,
            )
            should_eval = (
                args.eval_interval > 0
                and (
                    (iter_num + 1) % args.eval_interval == 0
                    or iter_num + 1 == args.max_iters
                )
            )

            if should_eval:
                val_loss = evaluate_pp(
                    stage=stage,
                    parallel=parallel,
                    data_path=val_data_path,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    eval_iters=args.eval_iters,
                    device=device,
                    generator=eval_data_generator,
                    activation_dtype=torch.float32,
                )

                if parallel["world_rank"] == 0:
                    print(
                        f"eval after iter {iter_num}: "
                        f"val_loss={val_loss:.6f}",
                        flush=True,
                    )

            should_log = (
               iter_num % args.log_interval == 0
               or iter_num == args.max_iters - 1
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

        print(
            f"rank={parallel['world_rank']} "
            f"dp_rank={parallel['dp_rank']} "
            f"pp_rank={parallel['pp_rank']} "
            f"layers=[{stage.layer_start}, {stage.layer_end}) "
            f"prev={parallel['prev_rank']} "
            f"next={parallel['next_rank']}",
            flush=True,
        )

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__=="__main__":
    main()




  