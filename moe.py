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
 