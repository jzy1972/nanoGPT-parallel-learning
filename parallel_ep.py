import torch.distributed as dist
import torch
from torch.autograd import Function

def get_expert_range(n_experts, ep_size, ep_rank):
    if n_experts <= 0:
        raise ValueError("n_experts 必须大于 0")

    if ep_size <= 0:
        raise ValueError("ep_size 必须大于 0")

    if n_experts % ep_size != 0:
        raise ValueError("n_experts 必须能被 ep_size 整除")

    if not 0 <= ep_rank < ep_size:
        raise ValueError("ep_rank 超出范围")

    local_n_experts = n_experts // ep_size

    start = ep_rank * local_n_experts
    end = start + local_n_experts

    return start, end

def setup_ep():
    if not dist.is_initialized():
        raise RuntimeError("请先初始化分布式进程组")

    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    return {
        "world_rank": world_rank,
        "world_size": world_size,
        "ep_rank": world_rank,
        "ep_size": world_size,
        "ep_group": dist.group.WORLD,
        "shared_dp_group": dist.group.WORLD,
    }

@torch.no_grad()
def exchange_counts(send_counts, ep_group):
    ep_size = dist.get_world_size(group=ep_group)

    if send_counts.ndim != 1 or send_counts.numel() != ep_size:
        raise ValueError("send_counts 必须是一维，长度等于 ep_size")

    if send_counts.dtype != torch.long:
        raise TypeError("send_counts 必须使用 torch.long")

    recv_counts = torch.empty_like(send_counts)

    dist.all_to_all_single(
        output=recv_counts,
        input=send_counts.contiguous(),
        group=ep_group,
    )

    return recv_counts

class AllToAll(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        input_splits,
        output_splits,
        ep_group,
    ):
        ctx.input_splits = list(input_splits)
        ctx.output_splits = list(output_splits)
        ctx.ep_group = ep_group

        output = x.new_empty(
            (sum(ctx.output_splits), *x.shape[1:])
        )

        dist.all_to_all_single(
            output=output,
            input=x.contiguous(),
            output_split_sizes=ctx.output_splits,
            input_split_sizes=ctx.input_splits,
            group=ctx.ep_group,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(
            (sum(ctx.input_splits), *grad_output.shape[1:])
        )

        dist.all_to_all_single(
            output=grad_input,
            input=grad_output.contiguous(),
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.ep_group,
        )

        return grad_input, None, None, None