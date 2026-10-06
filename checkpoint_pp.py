import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


PP_CHECKPOINT_VERSION = 1


def _checkpoint_error_consensus(local_error, device):
    flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=device,
    )

    dist.all_reduce(flag, op=dist.ReduceOp.MAX)

    if flag.item():
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, local_error)

        details = "; ".join(
            f"rank {rank}: {message}"
            for rank, message in enumerate(messages)
            if message is not None
        )

        raise RuntimeError(f"checkpoint 操作失败：{details}")


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporary, path)


def _optimizer_parameter_names(stage, optimizer):
    names = {
        id(parameter): name
        for name, parameter in stage.named_parameters()
    }

    return [
        [names[id(parameter)] for parameter in group["params"]]
        for group in optimizer.param_groups
    ]

@torch.no_grad()
def save_pp_checkpoint(
    stage,
    optimizer,
    parallel,
    checkpoint_root,
    completed_iters,
    train_data_generator,
    eval_data_generator,
    training_config,
    device,
):
    local_error = None

    try:
        if type(completed_iters) is not int or completed_iters < 0:
            raise ValueError("completed_iters 必须是非负整数")

        if any(
            parameter.grad is not None
            for parameter in stage.parameters()
        ):
            raise RuntimeError(
                "checkpoint 必须在一次训练结束且梯度清空后保存"
            )

        step_dir = (
            Path(checkpoint_root).expanduser().resolve()
            / f"step_{completed_iters:08d}"
        )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第一阶段：rank 0 创建新的保存目录
    local_error = None

    try:
        if parallel["world_rank"] == 0:
            step_dir.parent.mkdir(parents=True, exist_ok=True)
            step_dir.mkdir(exist_ok=False)

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第二阶段：保存模型分片和各进程的随机状态
    local_error = None

    try:
        if parallel["dp_rank"] == 0:
            stage_checkpoint = {
                "format_version": PP_CHECKPOINT_VERSION,
                "completed_iters": completed_iters,
                "pp_rank": parallel["pp_rank"],
                "layer_range": [
                    stage.layer_start,
                    stage.layer_end,
                ],
                "model": stage.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_parameter_names":
                    _optimizer_parameter_names(stage, optimizer),
            }

            _atomic_torch_save(
                stage_checkpoint,
                step_dir
                / f"stage_{parallel['pp_rank']:04d}.pt",
            )

        rng_checkpoint = {
            "format_version": PP_CHECKPOINT_VERSION,
            "completed_iters": completed_iters,
            "world_rank": parallel["world_rank"],
            "pp_rank": parallel["pp_rank"],
            "dp_rank": parallel["dp_rank"],
            "device_type": device.type,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None
            ),
            "train_generator": train_data_generator.get_state(),
            "eval_generator": eval_data_generator.get_state(),
        }

        _atomic_torch_save(
            rng_checkpoint,
            step_dir
            / f"rng_rank_{parallel['world_rank']:05d}.pt",
        )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第三阶段：所有分片写完，rank 0 写入完成标记
    local_error = None

    try:
        if parallel["world_rank"] == 0:
            manifest = {
                "format_version": PP_CHECKPOINT_VERSION,
                "completed_iters": completed_iters,
                "world_size": parallel["world_size"],
                "pp_size": stage.pp_size,
                "dp_size": parallel["dp_size"],
                "device_type": device.type,
                "parameter_dtype":
                    str(next(stage.parameters()).dtype),
                "model_config": vars(stage.config).copy(),
                "training_config": dict(training_config),
                "stage_files": [
                    f"stage_{rank:04d}.pt"
                    for rank in range(stage.pp_size)
                ],
                "rng_files": [
                    f"rng_rank_{rank:05d}.pt"
                    for rank in range(parallel["world_size"])
                ],
            }

            _atomic_json_save(
                manifest,
                step_dir / "manifest.json",
            )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    return str(step_dir)

def _check_checkpoint_fields(payload, expected, label):
    for name, expected_value in expected.items():
        if name not in payload:
            raise ValueError(f"{label} 缺少字段：{name}")

        actual_value = payload[name]

        if actual_value != expected_value:
            raise ValueError(
                f"{label}.{name} 不一致："
                f"保存值={actual_value!r}，"
                f"当前值={expected_value!r}"
            )

@torch.no_grad()
def load_pp_checkpoint(
    stage,
    optimizer,
    parallel,
    checkpoint_dir,
    train_data_generator,
    eval_data_generator,
    training_config,
    device,
):
    step_dir = None
    manifest = None
    completed_iters = None
    stage_checkpoint = None
    rng_checkpoint = None

    # 第一阶段：读取 manifest，检查配置与文件是否齐全。
    local_error = None

    try:
        step_dir = Path(checkpoint_dir).expanduser().resolve()

        with (step_dir / "manifest.json").open(
            "r", encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        expected_manifest = {
            "format_version": PP_CHECKPOINT_VERSION,
            "world_size": parallel["world_size"],
            "pp_size": stage.pp_size,
            "dp_size": parallel["dp_size"],
            "device_type": device.type,
            "parameter_dtype": str(next(stage.parameters()).dtype),
            "model_config": vars(stage.config).copy(),
            "training_config": dict(training_config),
            "stage_files": [
                f"stage_{rank:04d}.pt"
                for rank in range(stage.pp_size)
            ],
            "rng_files": [
                f"rng_rank_{rank:05d}.pt"
                for rank in range(parallel["world_size"])
            ],
        }

        _check_checkpoint_fields(
            manifest,
            expected_manifest,
            "manifest",
        )

        completed_iters = manifest["completed_iters"]

        if type(completed_iters) is not int or completed_iters < 0:
            raise ValueError("completed_iters 必须是非负整数")

        for filename in (
            manifest["stage_files"] + manifest["rng_files"]
        ):
            if not (step_dir / filename).is_file():
                raise FileNotFoundError(
                    f"checkpoint 缺少文件：{filename}"
                )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第二阶段：读取本进程需要的分片，并检查内容。
    local_error = None

    try:
        stage_checkpoint = torch.load(
            step_dir / f"stage_{parallel['pp_rank']:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )

        rng_checkpoint = torch.load(
            step_dir / f"rng_rank_{parallel['world_rank']:05d}.pt",
            map_location="cpu",
            weights_only=True,
        )

        _check_checkpoint_fields(
            stage_checkpoint,
            {
                "format_version": PP_CHECKPOINT_VERSION,
                "completed_iters": completed_iters,
                "pp_rank": parallel["pp_rank"],
                "layer_range": [
                    stage.layer_start,
                    stage.layer_end,
                ],
                "optimizer_parameter_names":
                    _optimizer_parameter_names(stage, optimizer),
            },
            "stage checkpoint",
        )

        _check_checkpoint_fields(
            rng_checkpoint,
            {
                "format_version": PP_CHECKPOINT_VERSION,
                "completed_iters": completed_iters,
                "world_rank": parallel["world_rank"],
                "pp_rank": parallel["pp_rank"],
                "dp_rank": parallel["dp_rank"],
                "device_type": device.type,
            },
            "rng checkpoint",
        )

        saved_model = stage_checkpoint["model"]
        current_model = stage.state_dict()

        if set(saved_model) != set(current_model):
            raise ValueError("模型 state_dict 的键不一致")

        for name, reference in current_model.items():
            value = saved_model[name]

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} 不是 Tensor")

            if value.shape != reference.shape:
                raise ValueError(f"{name} 的形状不一致")

            if value.dtype != reference.dtype:
                raise ValueError(f"{name} 的 dtype 不一致")

        # PP=1 时，两个名称对应同一个共享参数。
        if stage.pp_size == 1:
            if not torch.equal(
                saved_model["transformer.wte.weight"],
                saved_model["lm_head.weight"],
            ):
                raise ValueError("保存的共享 embedding/head 权重不一致")

        saved_groups = stage_checkpoint["optimizer"]["param_groups"]
        current_groups = optimizer.state_dict()["param_groups"]

        if len(saved_groups) != len(current_groups):
            raise ValueError("优化器参数组数量不一致")

        optimizer_options = (
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "amsgrad",
            "maximize",
            "foreach",
            "fused",
            "capturable",
            "differentiable",
        )

        for group_index, (saved, current) in enumerate(
            zip(saved_groups, current_groups)
        ):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError(
                    f"优化器第 {group_index} 组的参数数量不一致"
                )

            for name in optimizer_options:
                if saved.get(name) != current.get(name):
                    raise ValueError(
                        f"优化器第 {group_index} 组的 {name} 不一致"
                    )

        rng_keys = [
            "torch_cpu",
            "train_generator",
            "eval_generator",
        ]

        if device.type == "cuda":
            rng_keys.append("torch_cuda")
        elif rng_checkpoint["torch_cuda"] is not None:
            raise ValueError("CPU checkpoint 的 CUDA 随机状态应为 None")

        for name in rng_keys:
            value = rng_checkpoint[name]

            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.device.type != "cpu"
                or value.ndim != 1
            ):
                raise ValueError(f"{name} 不是有效的随机状态 Tensor")

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第三阶段：加载模型与优化器，清空梯度。
    local_error = None

    try:
        stage.load_state_dict(
            stage_checkpoint["model"],
            strict=True,
            assign=False,
        )

        optimizer.load_state_dict(stage_checkpoint["optimizer"])
        optimizer.zero_grad(set_to_none=True)

        if stage.pp_size == 1:
            if stage.transformer["wte"].weight is not stage.lm_head.weight:
                raise RuntimeError("PP=1 的共享参数关系被破坏")

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    # 第四阶段：最后恢复随机状态。
    local_error = None

    try:
        torch.set_rng_state(rng_checkpoint["torch_cpu"])

        if device.type == "cuda":
            torch.cuda.set_rng_state(
                rng_checkpoint["torch_cuda"],
                device=device,
            )

        train_data_generator.set_state(
            rng_checkpoint["train_generator"]
        )

        eval_data_generator.set_state(
            rng_checkpoint["eval_generator"]
        )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    _checkpoint_error_consensus(local_error, device)

    return completed_iters