import math
import torch
import torch.distributed as dist



def build_parameter_partition(model, dp_size):
    if dp_size <= 0:
        raise ValueError("dp_size 必须大于 0")

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    owners = {}
    rank_numels = [0] * dp_size

    sorted_params = sorted(
        named_params,
        key=lambda item: (-item[1].numel(), item[0]),
    )

    for name, param in sorted_params:
        owner = min(
            range(dp_size),
            key=lambda rank: (rank_numels[rank], rank),
        )

        owners[name] = owner
        rank_numels[owner] += param.numel()

    return named_params, owners, rank_numels


class Zero1Optimizer:
    def __init__(
        self,
        model,
        learning_rate,
        weight_decay,
        beta1=0.9,
        beta2=0.95,
        dp_group=None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("请先初始化分布式进程组")

        learning_rate = float(learning_rate)
        weight_decay = float(weight_decay)
        beta1, beta2 = float(beta1), float(beta2)

        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate 必须是有限的非负数")

        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay 必须是有限的非负数")

        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("beta1、beta2 必须在 [0, 1) 内")

        self.model = model
        self.dp_group = (
            dist.group.WORLD if dp_group is None else dp_group
        )

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.dp_rank = dist.get_rank(group=self.dp_group)

        (
            self.named_params,
            self.owners,
            self.rank_numels,
        ) = build_parameter_partition(model, self.dp_size)

        self.local_named_params = [
            (name, param)
            for name, param in self.named_params
            if self.owners[name] == self.dp_rank
        ]
        decay_params = [
            param
            for name, param in self.local_named_params
            if param.dim() >= 2
        ]

        nodecay_params = [
            param
            for name, param in self.local_named_params
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

        if param_groups:
            self.local_optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=1e-8,
                weight_decay=weight_decay,
                foreach=False,
                fused=False,
            )
        else:
            self.local_optimizer = None

        if not self.named_params:
            raise ValueError("模型至少需要一个可训练参数")

        self.last_grad_norm = None

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def sync_parameters(self):
        for name, param in self.named_params:
            owner = self.owners[name]

            src_rank = dist.get_global_rank(
                self.dp_group,
                owner,
            )

            dist.broadcast(
                param,
                src=src_rank,
                group=self.dp_group,
            )

    @torch.no_grad()
    def step(self, grad_clip=0.0):
        self.last_grad_norm = self.clip_gradients(grad_clip)

        if self.last_grad_norm is None:
            return False

        if self.local_optimizer is not None:
            self.local_optimizer.step()

        self.sync_parameters()

        return True
    
    @torch.no_grad()
    def has_nonfinite_gradients(self):
        device = self.named_params[0][1].device

        flag = torch.zeros(
            (),
            dtype=torch.int32,
            device=device,
        )

        for name, param in self.named_params:
            if param.grad is None:
                continue

            param_bad = (
                ~torch.isfinite(param.grad)
            ).any().to(dtype=torch.int32)

            flag = torch.maximum(flag, param_bad)

        dist.all_reduce(
            flag,
            op=dist.ReduceOp.MAX,
            group=self.dp_group,
        )

        return bool(flag.item())

    @torch.no_grad()
    def clip_gradients(self, max_norm):
        max_norm = float(max_norm)

        if not math.isfinite(max_norm) or max_norm < 0:
            raise ValueError("max_norm 必须是有限的非负数")

        if self.has_nonfinite_gradients():
            return None

        device = self.named_params[0][1].device

        squared_norm = torch.zeros(
            (),
            dtype=torch.float64,
            device=device,
        )

        for name, param in self.named_params:
            if param.grad is None:
                continue

            grad = param.grad.detach().to(dtype=torch.float64)
            squared_norm.add_(grad.square().sum())

        total_norm = squared_norm.sqrt()

        norm_bad = (
            ~torch.isfinite(total_norm)
        ).to(dtype=torch.int32)

        dist.all_reduce(
            norm_bad,
            op=dist.ReduceOp.MAX,
            group=self.dp_group,
        )

        if norm_bad.item():
            return None

        if max_norm > 0:
            clip_coef = (
                max_norm / (total_norm + 1e-6)
            ).clamp(max=1.0)

            for name, param in self.named_params:
                if param.grad is not None:
                    param.grad.mul_(
                        clip_coef.to(dtype=param.grad.dtype)
                    )

        return float(total_norm.item())

    def _checkpoint_layout(self):
        name_by_id = {
            id(param): name
            for name, param in self.named_params
        }

        local_param_groups = []

        if self.local_optimizer is not None:
            local_param_groups = [
                [
                    name_by_id[id(param)]
                    for param in group["params"]
                ]
                for group in self.local_optimizer.param_groups
            ]

        return {
            "dp_size": self.dp_size,
            "dp_rank": self.dp_rank,
            "parameters": [
                (name, tuple(param.shape), str(param.dtype))
                for name, param in self.named_params
            ],
            "owners": dict(self.owners),
            "local_param_groups": local_param_groups,
        }

    def state_dict(self):
        local_state = None

        if self.local_optimizer is not None:
            local_state = self.local_optimizer.state_dict()

        return {
            "format_version": 1,
            "zero_stage": 1,
            "layout": self._checkpoint_layout(),
            "local_optimizer": local_state,
            "last_grad_norm": self.last_grad_norm,
        }
    
    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("优化器 checkpoint 必须是字典")

        if state.get("format_version") != 1:
            raise ValueError("不支持的优化器 checkpoint 版本")

        if state.get("zero_stage") != 1:
            raise ValueError("checkpoint 不是 ZeRO-1 状态")

        saved_layout = state.get("layout")

        if not isinstance(saved_layout, dict):
            raise ValueError("checkpoint 缺少有效的 layout")

        current_layout = self._checkpoint_layout()

        for key, expected in current_layout.items():
            if saved_layout.get(key) != expected:
                raise ValueError(
                    f"checkpoint 的 {key} 与当前优化器不一致"
                )

        if "local_optimizer" not in state:
            raise ValueError("checkpoint 缺少 local_optimizer")

        last_grad_norm = state.get("last_grad_norm")

        if last_grad_norm is not None:
            last_grad_norm = float(last_grad_norm)

            if not math.isfinite(last_grad_norm) or last_grad_norm < 0:
                raise ValueError("checkpoint 的 last_grad_norm 非法")

        local_state = state["local_optimizer"]

        if self.local_optimizer is None:
            if local_state is not None:
                raise ValueError("空分片 rank 的优化器状态应为 None")
        else:
            if not isinstance(local_state, dict):
                raise ValueError("非空分片 rank 缺少 AdamW 状态")

            self.local_optimizer.load_state_dict(local_state)

        self.last_grad_norm = last_grad_norm


