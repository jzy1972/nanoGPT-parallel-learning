import torch.distributed as dist
import argparse
import os
import torch

from model import GPTConfig
from model_pp import GPTStage

from parallel_utils import(build_pp_dp_groups,build_tied_weight_group,sync_tied_weights)

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




  