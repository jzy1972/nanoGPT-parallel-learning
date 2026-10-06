import torch.distributed as dist
import argparse
import os
import torch

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

    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)

    args=parser.parse_args()

    if not 1 <= args.pp_size <= args.n_layer:
        parser.error("需要满足 1 <= pp_size <= n_layer")

    return args

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
        if args.device_type=="cuda":
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=device
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
            )

        parallel=setup_parallel(args.pp_size)

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




  