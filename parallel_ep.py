import torch.distributed as dist

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