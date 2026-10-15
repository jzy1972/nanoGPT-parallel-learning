import torch.distributed as dist
import torch
from torch.autograd import Function

import math

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

def split_ep_parameters(model):
    from moe_ep import MoEEP

    expert_param_ids = set()

    for module in model.modules():
        if isinstance(module, MoEEP):
            for param in module.experts.parameters():
                expert_param_ids.add(id(param))

    shared_named_params = []
    expert_named_params = []

    for name, param in sorted(model.named_parameters()):
        if id(param) in expert_param_ids:
            expert_named_params.append((name, param))
        else:
            shared_named_params.append((name, param))

    return shared_named_params, expert_named_params

@torch.no_grad()
def sync_shared_parameters(model, parallel):
    shared_named_params, _ = split_ep_parameters(model)

    group = parallel["shared_dp_group"]
    src_rank = dist.get_global_rank(group, 0)

    for name, param in shared_named_params:
        dist.broadcast(
            param,
            src=src_rank,
            group=group,
        )

@torch.no_grad()
def sync_ep_gradients(model, parallel):
    shared_named_params, expert_named_params = (
        split_ep_parameters(model)
    )

    shared_group = parallel["shared_dp_group"]

    dp_size = dist.get_world_size(group=shared_group)
    ep_size = dist.get_world_size(
        group=parallel["ep_group"]
    )

    if dp_size != ep_size:
        raise ValueError(
            "当前版本要求一个 EP 组覆盖所有数据并行进程"
        )

    for name, param in shared_named_params:
        has_grad = torch.tensor(
            int(param.grad is not None),
            dtype=torch.int32,
            device=param.device,
        )

        dist.all_reduce(
            has_grad,
            op=dist.ReduceOp.MAX,
            group=shared_group,
        )

        if has_grad.item() == 0:
            continue

        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach()

        grad = grad.contiguous()

        dist.all_reduce(
            grad,
            op=dist.ReduceOp.SUM,
            group=shared_group,
        )

        grad.div_(dp_size)
        param.grad = grad

    for name, param in expert_named_params:
        if param.grad is not None:
            param.grad.div_(ep_size)

@torch.no_grad()
def clip_ep_gradients(model, max_norm, parallel):
    max_norm = float(max_norm)

    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm 必须是有限的非负数")

    shared_named_params, expert_named_params = (
        split_ep_parameters(model)
    )

    all_named_params = (
        shared_named_params + expert_named_params
    )

    device = all_named_params[0][1].device
    ep_group = parallel["ep_group"]
    ep_rank = parallel["ep_rank"]

    # 1. 任意进程存在非有限梯度，所有进程统一返回 None
    bad = torch.zeros(
        (),
        dtype=torch.int32,
        device=device,
    )

    for name, param in all_named_params:
        if param.grad is None:
            continue

        param_bad = (
            ~torch.isfinite(param.grad)
        ).any().to(dtype=torch.int32)

        bad = torch.maximum(bad, param_bad)

    dist.all_reduce(
        bad,
        op=dist.ReduceOp.MAX,
        group=ep_group,
    )

    if bad.item():
        return None

    # 2. 专家各自统计；共享参数只在 rank 0 统计
    norm_named_params = list(expert_named_params)

    if ep_rank == 0:
        norm_named_params.extend(shared_named_params)

    squared_norm = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )

    for name, param in norm_named_params:
        if param.grad is None:
            continue

        grad = param.grad.detach().to(dtype=torch.float64)
        squared_norm.add_(grad.square().sum())

    # 3. 汇总整个模型的梯度平方和
    dist.all_reduce(
        squared_norm,
        op=dist.ReduceOp.SUM,
        group=ep_group,
    )

    total_norm = squared_norm.sqrt()

    if not torch.isfinite(total_norm).item():
        return None

    # 4. 所有参数使用同一个裁剪系数
    # max_norm == 0 表示关闭裁剪，但仍检查并返回范数
    if max_norm > 0:
        clip_coef = (
            max_norm / (total_norm + 1e-6)
        ).clamp(max=1.0)

        for name, param in all_named_params:
            if param.grad is not None:
                param.grad.mul_(
                    clip_coef.to(dtype=param.grad.dtype)
                )

    return float(total_norm.item())

def build_ep_optimizer(
    model,
    learning_rate,
    weight_decay,
    beta1=0.9,
    beta2=0.95,
):
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    beta1 = float(beta1)
    beta2 = float(beta2)

    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate 必须是有限的非负数")

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay 必须是有限的非负数")

    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("beta1、beta2 必须在 [0, 1) 内")

    params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    if not params:
        raise ValueError("模型至少需要一个可训练参数")

    decay_params = [
        param
        for param in params
        if param.dim() >= 2
    ]

    nodecay_params = [
        param
        for param in params
        if param.dim() < 2
    ]

    param_groups = []

    if decay_params:
        param_groups.append({
            "params": decay_params,
            "weight_decay": weight_decay,
        })

    if nodecay_params:
        param_groups.append({
            "params": nodecay_params,
            "weight_decay": 0.0,
        })

    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )

@torch.no_grad()
def step_ep_optimizer(
    model,
    optimizer,
    parallel,
    grad_clip=0.0,
):
    grad_clip = float(grad_clip)

    if not math.isfinite(grad_clip) or grad_clip < 0:
        raise ValueError("grad_clip 必须是有限的非负数")

    sync_ep_gradients(model, parallel)

    grad_norm = clip_ep_gradients(
        model,
        max_norm=grad_clip,
        parallel=parallel,
    )

    stepped = grad_norm is not None

    if stepped:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    return {
        "stepped": stepped,
        "grad_norm": grad_norm,
    }