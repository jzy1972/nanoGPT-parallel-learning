import torch.distributed as dist
import torch.nn as nn

from moe import Router
from parallel_ep import get_expert_range


class MoEEP(nn.Module):
    def __init__(self, config, expert_cls, ep_group):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("请先初始化分布式进程组")

        self.n_experts = config.n_experts
        self.ep_group = ep_group

        self.ep_size = dist.get_world_size(
            group=self.ep_group
        )
        self.ep_rank = dist.get_rank(
            group=self.ep_group
        )

        self.expert_start, self.expert_end = get_expert_range(
            n_experts=self.n_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
        )

        self.local_n_experts = (
            self.expert_end - self.expert_start
        )

        self.router = Router(config)

        self.experts = nn.ModuleDict({
            str(expert_id): expert_cls(config)
            for expert_id in range(
                self.expert_start,
                self.expert_end,
            )
        })