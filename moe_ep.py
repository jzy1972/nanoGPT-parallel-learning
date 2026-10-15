import torch.distributed as dist
import torch.nn as nn

from moe import Router
from parallel_ep import (
    get_expert_range,
    exchange_counts,
    AllToAll,
)

import torch
from torch.autograd import Function


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

    def forward(self, x):
        # 1. 展开 token，获得路由结果
        B, T, C = x.shape
        N = B * T
        x_flat = x.reshape(N, C)

        topk_indices, topk_weights, probs = self.router(x_flat)
        top_k = topk_indices.size(1)

        # 每一行代表一次“token -> 专家”的分配
        expert_ids = topk_indices.reshape(-1)

        token_indices = torch.arange(
            N,
            dtype=torch.long,
            device=x.device,
        ).repeat_interleave(top_k)

        owner_ranks = expert_ids // self.local_n_experts

        # 2. 按目标进程排序，打包数据
        order = torch.argsort(owner_ranks, stable=True)

        packed_token_indices = token_indices.index_select(0, order)
        packed_expert_ids = expert_ids.index_select(0, order)

        packed_weights = topk_weights.reshape(-1).index_select(
            0, order
        )

        packed_x = x_flat.index_select(
            0, packed_token_indices
        )

        # 保证训练时所有进程都有通信的反向传播路径
        if torch.is_grad_enabled() and not packed_x.requires_grad:
            packed_x.requires_grad_(True)

        # 3. 交换收发数量
        send_counts = torch.bincount(
            owner_ranks,
            minlength=self.ep_size,
        )

        recv_counts = exchange_counts(
            send_counts,
            self.ep_group,
        )

        input_splits = send_counts.tolist()
        output_splits = recv_counts.tolist()

        # 4. 发送 token，同时发送它们的全局专家编号
        received_x = AllToAll.apply(
            packed_x,
            input_splits,
            output_splits,
            self.ep_group,
        )

        received_expert_ids = packed_expert_ids.new_empty(
            (sum(output_splits),)
        )

        dist.all_to_all_single(
            output=received_expert_ids,
            input=packed_expert_ids.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.ep_group,
        )

        # 5. 本地专家计算；保留收到 0 个 token 时的梯度连接
        expert_outputs = received_x * 0.0

        for expert_id, expert in self.experts.items():
            positions = torch.where(
                received_expert_ids == int(expert_id)
            )[0]

            if positions.numel() == 0:
                continue

            expert_input = received_x.index_select(
                0, positions
            )

            expert_output = expert(expert_input)

            expert_outputs = expert_outputs.index_copy(
                0,
                positions,
                expert_output.to(dtype=expert_outputs.dtype),
            )

        # 6. 把专家输出发回 token 原来的进程
        returned_outputs = AllToAll.apply(
            expert_outputs,
            output_splits,
            input_splits,
            self.ep_group,
        )

        # 7. 在原进程加权，并累加到原 token 位置
        weighted_outputs = (
            returned_outputs * packed_weights.unsqueeze(-1)
        )

        output_flat = torch.zeros_like(x_flat)

        output_flat = output_flat.index_add(
            0,
            packed_token_indices,
            weighted_outputs.to(dtype=output_flat.dtype),
        )

        # 8. 保持原来按本地 token 计算的负载均衡损失
        counts = torch.bincount(
            expert_ids,
            minlength=self.n_experts,
        )

        fractions = counts.to(probs.dtype) / expert_ids.numel()
        mean_probs = probs.mean(dim=0)

        aux_loss = self.n_experts * (
            fractions * mean_probs
        ).sum()

        return output_flat.reshape(B, T, C), aux_loss

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