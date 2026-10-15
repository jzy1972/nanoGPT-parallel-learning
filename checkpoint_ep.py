import json
import os
from dataclasses import asdict

import torch
import torch.distributed as dist

import math


def capture_ep_rng_state(
    device,
    train_data_generator,
    eval_data_generator,
):
    state = {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
        "train_data": train_data_generator.get_state().clone(),
        "eval_data": eval_data_generator.get_state().clone(),
    }

    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device).clone()

    return state

def check_ep_checkpoint_errors(local_error, parallel, phase):
    errors = [None] * parallel["ep_size"]

    dist.all_gather_object(
        errors,
        local_error,
        group=parallel["ep_group"],
    )

    failures = [
        f"ep_rank={rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]

    if failures:
        raise RuntimeError(
            f"{phase}失败:\n" + "\n".join(failures)
        )

@torch.no_grad()
def save_ep_checkpoint(
    model,
    optimizer,
    parallel,
    checkpoint_dir,
    completed_iters,
    best_val_loss,
    train_data_generator,
    eval_data_generator,
    device,
    train_args,
):
    rank = parallel["world_rank"]

    # 1. 检查保存边界，创建目录。
    local_error = None

    try:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        manifest_path = os.path.join(
            checkpoint_dir,
            "manifest.json",
        )

        if not isinstance(completed_iters, int) or completed_iters < 0:
            raise ValueError("completed_iters 必须是非负整数")

        if parallel["ep_size"] != parallel["world_size"]:
            raise ValueError("当前格式要求一个 EP 组覆盖 WORLD")

        if any(param.grad is not None for param in model.parameters()):
            raise RuntimeError(
                "请在完整训练迭代结束、梯度清空后保存"
            )

        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

            if os.path.exists(manifest_path):
                raise FileExistsError(
                    "该目录已有完整 checkpoint，请使用新目录"
                )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "准备 checkpoint",
    )

    # 2. 每个 rank 写自己的文件。
    local_error = None

    try:
        name_by_id = {
            id(param): name
            for name, param in model.named_parameters()
        }

        optimizer_param_names = [
            [name_by_id[id(param)] for param in group["params"]]
            for group in optimizer.param_groups
        ]

        model_args = asdict(model.config)
        dtype = str(next(model.parameters()).dtype)

        checkpoint = {
            "format_version": 1,
            "parallelism": "ep_shared_dp",
            "world_size": parallel["world_size"],
            "ep_size": parallel["ep_size"],
            "world_rank": rank,
            "ep_rank": parallel["ep_rank"],
            "completed_iters": completed_iters,
            "best_val_loss": float(best_val_loss),
            "model_args": model_args,
            "train_args": dict(train_args),
            "dtype": dtype,
            "device_type": device.type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_param_names": optimizer_param_names,
            "rng": capture_ep_rng_state(
                device,
                train_data_generator,
                eval_data_generator,
            ),
        }

        rank_path = os.path.join(
            checkpoint_dir,
            f"rank_{rank:05d}.pt",
        )
        temporary_path = rank_path + ".tmp"

        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, rank_path)

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "保存 rank 文件",
    )

    # 3. 全部 rank 成功后，由 rank 0 写完整标记。
    local_error = None

    if rank == 0:
        try:
            rank_files = [
                f"rank_{r:05d}.pt"
                for r in range(parallel["world_size"])
            ]

            for filename in rank_files:
                path = os.path.join(checkpoint_dir, filename)

                if not os.path.isfile(path):
                    raise FileNotFoundError(filename)

            manifest = {
                "format_version": 1,
                "parallelism": "ep_shared_dp",
                "world_size": parallel["world_size"],
                "ep_size": parallel["ep_size"],
                "completed_iters": completed_iters,
                "model_args": model_args,
                "dtype": dtype,
                "device_type": device.type,
                "rank_files": rank_files,
            }

            temporary_path = manifest_path + ".tmp"

            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    manifest,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )

            os.replace(temporary_path, manifest_path)

        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "写入完整标记",
    )

    return checkpoint_dir

def restore_ep_rng_state(
    state,
    device,
    train_data_generator,
    eval_data_generator,
):
    torch.set_rng_state(state["torch_cpu"].cpu())

    if device.type == "cuda":
        torch.cuda.set_rng_state(
            state["torch_cuda"].cpu(),
            device=device,
        )

    train_data_generator.set_state(
        state["train_data"].cpu()
    )

    eval_data_generator.set_state(
        state["eval_data"].cpu()
    )

@torch.no_grad()
def load_ep_checkpoint(
    model,
    optimizer,
    parallel,
    checkpoint_dir,
    train_data_generator,
    eval_data_generator,
    device,
    train_args,
):
    rank = parallel["world_rank"]
    local_error = None

    try:
        checkpoint_dir = os.path.abspath(checkpoint_dir)

        manifest_path = os.path.join(
            checkpoint_dir,
            "manifest.json",
        )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if parallel["ep_size"] != parallel["world_size"]:
            raise ValueError("当前格式要求一个 EP 组覆盖 WORLD")

        expected_files = [
            f"rank_{r:05d}.pt"
            for r in range(parallel["world_size"])
        ]

        if manifest["rank_files"] != expected_files:
            raise ValueError("checkpoint 的 rank 文件列表不匹配")

        for filename in expected_files:
            path = os.path.join(checkpoint_dir, filename)

            if not os.path.isfile(path):
                raise FileNotFoundError(filename)

        rank_path = os.path.join(
            checkpoint_dir,
            f"rank_{rank:05d}.pt",
        )

        checkpoint = torch.load(
            rank_path,
            map_location="cpu",
            weights_only=True,
        )

        expected_common = {
            "format_version": 1,
            "parallelism": "ep_shared_dp",
            "world_size": parallel["world_size"],
            "ep_size": parallel["ep_size"],
            "model_args": asdict(model.config),
            "dtype": str(next(model.parameters()).dtype),
            "device_type": device.type,
        }

        for key, expected in expected_common.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"manifest 的 {key} 不匹配"
                )

            if checkpoint.get(key) != expected:
                raise ValueError(
                    f"rank 文件的 {key} 不匹配"
                )

        if checkpoint["world_rank"] != rank:
            raise ValueError("world_rank 不匹配")

        if checkpoint["ep_rank"] != parallel["ep_rank"]:
            raise ValueError("ep_rank 不匹配")

        completed_iters = manifest["completed_iters"]

        if type(completed_iters) is not int or completed_iters < 0:
            raise ValueError("completed_iters 必须是非负整数")

        if checkpoint["completed_iters"] != completed_iters:
            raise ValueError(
                "manifest 与 rank 文件的进度不一致"
            )

        if train_args["max_iters"] < completed_iters:
            raise ValueError(
                "max_iters 小于 checkpoint 中已完成的迭代数"
            )

        best_val_loss = float(
            checkpoint["best_val_loss"]
        )

        if math.isnan(best_val_loss) or best_val_loss < 0:
            raise ValueError("best_val_loss 无效")

        critical_args = (
            "seed",
            "data_dir",
            "batch_size",
            "micro_batch_size",
            "seq_length",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
            "eval_interval",
            "eval_iters",
        )

        for key in critical_args:
            if checkpoint["train_args"][key] != train_args[key]:
                raise ValueError(
                    f"恢复时的训练参数 {key} 不一致"
                )

        name_by_id = {
            id(param): name
            for name, param in model.named_parameters()
        }

        current_param_names = [
            [name_by_id[id(param)] for param in group["params"]]
            for group in optimizer.param_groups
        ]

        if checkpoint["optimizer_param_names"] != current_param_names:
            raise ValueError(
                "优化器参数的名字或顺序不匹配"
            )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "读取并检查 checkpoint",
    )

    common_states = [None] * parallel["ep_size"]

    dist.all_gather_object(
        common_states,
        (manifest, best_val_loss),
        group=parallel["ep_group"],
    )

    if any(state != common_states[0] for state in common_states):
        raise RuntimeError(
            "各 rank 的 checkpoint 公共信息或最佳 loss 不一致"
        )

    local_error = None

    try:
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        model.zero_grad(set_to_none=True)

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "恢复模型与优化器",
    )

    local_error = None

    try:
        restore_ep_rng_state(
            checkpoint["rng"],
            device,
            train_data_generator,
            eval_data_generator,
        )

    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    check_ep_checkpoint_errors(
        local_error,
        parallel,
        "恢复随机状态",
    )

    return {
        "completed_iters": completed_iters,
        "best_val_loss": best_val_loss,
    }