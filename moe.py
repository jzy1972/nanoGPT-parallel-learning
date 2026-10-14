import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k

        if self.n_experts <= 0:
            raise ValueError("n_experts 必须大于 0")

        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("moe_top_k 必须在 1 到 n_experts 之间")

        self.gate = nn.Linear(
            config.n_embd,
            self.n_experts,
            bias=False,
        )

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(
            probs,
            k=self.top_k,
            dim=-1,
        )

        return topk_indices, topk_weights, probs


class MoE(nn.Module):
    def __init__(self, config, expert_cls):
        super().__init__()

        self.n_experts = config.n_experts

        self.router = Router(config)

        self.experts = nn.ModuleList([
            expert_cls(config)
            for _ in range(self.n_experts)
        ])

    def forward(self, x):
        # 1. 展开 token，取得路由结果
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)

        topk_indices, topk_weights, probs = self.router(x_flat)

        output_flat = torch.zeros_like(x_flat)

        # 2. 每个专家处理分配给自己的 token
        for expert_id, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(
                topk_indices == expert_id
            )

            if token_indices.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, token_indices)
            expert_output = expert(expert_input)

            # 3. 加权后，累加回 token 原来的位置
            weights = topk_weights[
                token_indices, slot_indices
            ].unsqueeze(-1)

            weighted_output = expert_output * weights

            output_flat = output_flat.index_add(
                0,
                token_indices,
                weighted_output.to(output_flat.dtype),
            )

        # 4. 计算本次 forward 的负载均衡损失
        counts = torch.bincount(
            topk_indices.reshape(-1),
            minlength=self.n_experts,
        )

        fractions = counts.to(probs.dtype) / topk_indices.numel()
        mean_probs = probs.mean(dim=0)

        aux_loss = self.n_experts * (
            fractions * mean_probs
        ).sum()

        output = output_flat.reshape(B, T, C)

        return output, aux_loss
             