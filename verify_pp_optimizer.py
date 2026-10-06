#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 PP 全局裁剪、异常共识与优化器更新（CPU/Gloo，float64）。

把本文件、verify_pp_grad_sync.py、verify_pp_microbatch.py 放在 nanoGPT
根目录，运行：
    python verify_pp_optimizer.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_optimizer_report.json

直接调用项目的 GPipe、梯度同步、global_has_nonfinite_gradients、
clip_stage_gradients 和 step_stage_optimizer，不运行 train_pp.main，
不修改项目源码。验证器自己创建普通 AdamW，测试传入用户更新函数后的
行为；生产训练的优化器创建、启动流程、数据加载尚不在本测试范围内。

参考是原版 GPT 对拼接后的全局 DP batch 做训练，裁剪使用 PyTorch
clip_grad_norm_。检查所有本地参数、AdamW step/一阶/二阶矩、DP 副本
及首尾共享权重一致性。默认分别检查关闭裁剪和 clip_norm=0.01。

先用只有共享权重梯度为 [3,4] 的数据检查范数为 5、共享权重只计一次、
首尾都被裁剪、系数不放大梯度。随后连续正常更新两次，向单个 rank
注入 NaN/+Inf/-Inf，验证所有 rank 跳过更新、参数和完整优化器状态
不变、梯度清为 None。开启裁剪时还检查有限梯度导致范数溢出的共识。
最后验证恢复正常训练，以及全零梯度时仍按 AdamW 的动量/衰减更新。

默认启动 pp_size*dp_size 个 CPU worker，检查同一 world_size 的不同
分段数：默认为 (1,4)、(2,2)、(4,1)。这是验证配置，不限制算法规模。
--single-case 只检查指定分段数；--preflight-only 不导入 torch。
不测试 CUDA/NCCL、AMP、checkpoint、真实数据、生产优化器创建、
torchrun、train_pp.main 或性能。
"""

import argparse
import ast
from contextlib import redirect_stdout
import copy
from datetime import timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

try:
    import verify_pp_grad_sync as sync_checks
except ModuleNotFoundError as exc:
    if exc.name == "verify_pp_grad_sync":
        raise SystemExit("请把 verify_pp_grad_sync.py 与本文件放在同一目录。") from exc
    raise

common = sync_checks.common
EXTRA_ARGUMENTS = {
    "parallel_utils.py": {
        "global_has_nonfinite_gradients": ("stage", "device"),
        "clip_stage_gradients": ("stage", "max_norm", "parallel", "device"),
    },
    "train_pp.py": {
        "step_stage_optimizer": ("stage", "optimizer", "parallel", "device", "grad_clip"),
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-layer", type=int, default=5)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--micro-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--clip-norm", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--steps", type=int, default=2, help="异常注入前的正常更新次数，至少 2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--bias", choices=("true", "false", "both"), default="both")
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd",
                 "vocab_size", "block_size", "seq_len", "batch_size", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.steps < 2:
        parser.error("--steps 至少为 2，以检查已经存在的优化器状态")
    if args.pp_size > args.n_layer or args.n_embd % args.n_head:
        parser.error("需要 pp_size <= n_layer，且 n_embd 能被 n_head 整除")
    if args.seq_len > args.block_size or args.vocab_size < 2 or args.batch_size * args.seq_len < 2:
        parser.error("需要 seq_len <= block_size、至少两个词表项和两个标签位置")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 必须是非负的有限数")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in args.betas):
        parser.error("每个 beta 必须满足 0 <= beta < 1")
    if args.micro_batch_sizes is None:
        args.micro_batch_sizes = [1, 2, args.batch_size]
    args.micro_batch_sizes = sorted(set(args.micro_batch_sizes))
    if any(size <= 0 for size in args.micro_batch_sizes):
        parser.error("每个 micro_batch_size 必须为正整数")
    world = args.pp_size * args.dp_size
    args.cases = ([(args.pp_size, args.dp_size)] if args.single_case else
                  [(p, world // p) for p in range(1, min(args.n_layer, world) + 1) if world % p == 0])
    args.biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    return args


def preflight(project):
    expected = Path(__file__).resolve().parent / "verify_pp_grad_sync.py"
    if Path(sync_checks.__file__).resolve() != expected:
        raise RuntimeError(f"梯度验证辅助文件应来自 {expected}")
    sync_checks.preflight(project)
    for filename, functions in EXTRA_ARGUMENTS.items():
        path = project / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for name, required in functions.items():
            nodes = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
            if len(nodes) != 1:
                raise AttributeError(f"{filename} 需要定义一个顶层 {name}()")
            supplied = {arg.arg for arg in nodes[0].args.args + nodes[0].args.kwonlyargs}
            if set(required) - supplied:
                raise TypeError(f"{filename} 的 {name} 缺少参数 {sorted(set(required)-supplied)}")


def fingerprints(project):
    result = common.source_hashes(project)
    for path in (Path(common.__file__), Path(sync_checks.__file__), Path(__file__)):
        result[f"checker:{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def exact_tree(torch, actual, expected, errors, label):
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not (torch.is_tensor(actual) and torch.is_tensor(expected)):
            errors.append(f"{label}: 类型发生变化")
            return
        try:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        except (AssertionError, RuntimeError) as exc:
            errors.append(f"{label}: {exc}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            errors.append(f"{label}: 字典字段发生变化")
            return
        for key in expected:
            exact_tree(torch, actual[key], expected[key], errors, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            errors.append(f"{label}: 序列发生变化")
            return
        for index, value in enumerate(expected):
            exact_tree(torch, actual[index], value, errors, f"{label}[{index}]")
    elif actual != expected:
        errors.append(f"{label}: 应为 {expected!r}，实际为 {actual!r}")


def state_snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def check_scalar(torch, actual, expected, errors, label):
    if actual is None:
        errors.append(f"{label}: 返回值为 None")
        return
    try:
        tensor = torch.as_tensor(actual, dtype=torch.float64)
        common.check_close(torch, tensor, torch.as_tensor(expected, dtype=torch.float64), label, errors)
    except (TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"{label}: {exc}")


def check_result(torch, result, stepped, norm, errors):
    if not isinstance(result, dict):
        errors.append(f"step_stage_optimizer 应返回字典，实际为 {result!r}")
        return
    if result.get("stepped") is not stepped:
        errors.append(f"stepped 应为 {stepped}，实际为 {result.get('stepped')!r}")
    if "grad_norm" not in result:
        errors.append("返回字典缺少 grad_norm")
    elif norm is None:
        if result["grad_norm"] is not None:
            errors.append("关闭裁剪或跳过更新时 grad_norm 应为 None")
    else:
        check_scalar(torch, result["grad_norm"], norm, errors, "returned pre-clipping norm")


def cleared_gradient_errors(stage):
    return [f"更新入口返回后 {name}.grad 未清为 None"
            for name, parameter in stage.named_parameters() if parameter.grad is not None]


def make_optimizer(torch, model, args, audited=False):
    parameters = [param for param in model.parameters() if param.requires_grad]
    groups = [
        {"params": [p for p in parameters if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in parameters if p.dim() < 2], "weight_decay": 0.0},
    ]
    class AuditedAdamW(torch.optim.AdamW):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.step_calls = 0
        def step(self, *positional, **keywords):
            self.step_calls += 1
            return super().step(*positional, **keywords)
    factory = AuditedAdamW if audited else torch.optim.AdamW
    return factory(groups, lr=args.learning_rate, betas=tuple(args.betas),
                   foreach=False, fused=False)


def compare_model_and_optimizer(torch, stage, dense, optimizer, dense_optimizer, errors):
    expected_state = dense.state_dict()
    max_parameter_error, max_moment_error = 0.0, 0.0
    for name, value in stage.state_dict().items():
        common.check_close(torch, value, expected_state.get(name), f"updated model {name}", errors)
        if value.is_floating_point() and name in expected_state:
            max_parameter_error = max(max_parameter_error,
                float((value - expected_state[name]).abs().max().item()))
    reference_parameters = dict(dense.named_parameters(remove_duplicate=False))
    for name, parameter in stage.named_parameters():
        target = reference_parameters.get(name)
        if target is None:
            errors.append(f"优化器参考缺少 {name}")
            continue
        actual = optimizer.state.get(parameter, {})
        expected = dense_optimizer.state.get(target, {})
        if actual.keys() != expected.keys():
            errors.append(f"optimizer {name}: 状态字段不一致")
            continue
        for field in expected:
            common.check_close(torch, actual[field], expected[field], f"optimizer {name}.{field}", errors)
            if torch.is_tensor(actual[field]) and torch.is_tensor(expected[field]):
                max_moment_error = max(max_moment_error,
                    float((actual[field] - expected[field]).abs().max().item()))
    return max_parameter_error, max_moment_error


def replica_parameter_errors(torch, dist, stage, info, pp_size, dp_size, args):
    errors = []
    flat = torch.cat([p.detach().reshape(-1) for p in stage.parameters()]).contiguous()
    replicas = [torch.empty_like(flat) for _ in range(dp_size)]
    dist.all_gather(replicas, flat, group=info["dp_group"])
    if any(not torch.equal(flat, other) for other in replicas):
        errors.append("对应 stage 的 DP 副本参数不完全相同")
    endpoint = torch.zeros((args.vocab_size, args.n_embd), dtype=torch.float64)
    if stage.is_first:
        endpoint.copy_(stage.transformer["wte"].weight.detach())
    elif stage.is_last:
        endpoint.copy_(stage.lm_head.weight.detach())
    endpoints = [torch.empty_like(endpoint) for _ in range(dist.get_world_size())]
    dist.all_gather(endpoints, endpoint)
    for d in range(dp_size):
        if not torch.equal(endpoints[d*pp_size], endpoints[(d+1)*pp_size-1]):
            errors.append(f"DP 副本 {d} 的首尾共享权重参数不完全相同")
    return errors


def prepare_gradients(torch, trainer, stage, info, args, dp_size, event, micro_size, pattern, dense=None):
    round_args = copy.copy(args)
    round_args.seed += 10007 * event
    batches = sync_checks.make_data(torch, round_args, dp_size, micro_size, pattern)
    counts = [int((targets != -1).sum().item()) for _, targets in batches]
    assert sum(counts) > 0
    loss = None
    if dense is not None:
        global_x = torch.cat([x for x, _ in batches], dim=0)
        global_y = torch.cat([y for _, y in batches], dim=0)
        _, loss, _ = sync_checks.backward_reference(torch, dense, global_x, global_y)
    stage.zero_grad(set_to_none=True)
    x, targets = batches[info["dp_rank"]]
    sync_checks.run_gpipe(torch, trainer, stage, info, args, micro_size, x, targets)
    weight = trainer.sync_stage_gradients(
        stage=stage, targets=targets if stage.is_last else None,
        parallel=info, device=torch.device("cpu"),
    )
    errors = []
    sync_checks.check_weight(weight, dp_size * counts[info["dp_rank"]] / sum(counts), errors,
                             "sync_stage_gradients returned weight")
    if dense is not None:
        sync_checks.compare_gradients(torch, stage, dense, errors, "synchronized global gradient")
    return loss, counts, errors


def synthetic_clip_checks(torch, dist, rank, args, utils, dense, stage, info,
                          optimizer, pp_size, dp_size, label):
    errors, records = [], []
    original_state = state_snapshot(stage)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for threshold in (0.01, 10.0):
        for model in (dense, stage):
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
        dense.transformer.wte.weight.grad[0, 0] = 3.0
        dense.transformer.wte.weight.grad[1, 0] = 4.0
        if stage.is_first:
            stage.transformer["wte"].weight.grad[0, 0] = 3.0
            stage.transformer["wte"].weight.grad[1, 0] = 4.0
        elif stage.is_last:
            stage.lm_head.weight.grad[0, 0] = 3.0
            stage.lm_head.weight.grad[1, 0] = 4.0
        expected_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), threshold)
        actual_norm = utils.clip_stage_gradients(stage=stage, max_norm=threshold,
                                                parallel=info, device=torch.device("cpu"))
        check_scalar(torch, expected_norm, 5.0, errors, "synthetic dense norm")
        check_scalar(torch, actual_norm, 5.0, errors, "synthetic PP norm: tied weight counted once")
        sync_checks.compare_gradients(torch, stage, dense, errors, "both endpoints clipped")
        exact_tree(torch, stage.state_dict(), original_state, errors, "clipping leaves model unchanged")
        exact_tree(torch, optimizer.state_dict(), optimizer_state, errors, "clipping leaves optimizer unchanged")
        sync_checks.agree(torch, dist, rank, errors, f"{label}: tied-only norm=5, max_norm={threshold}")
        records.append({"threshold": threshold, "pre_clipping_norm": 5.0, "status": "passed"})
    for model in (dense, stage):
        model.zero_grad(set_to_none=True)
    return records


def finite_step(torch, dist, rank, args, trainer, utils, dense, stage, info, optimizer,
                dense_optimizer, pp_size, dp_size, grad_clip, event, micro_size, pattern, label):
    loss, counts, errors = prepare_gradients(torch, trainer, stage, info, args, dp_size,
                                            event, micro_size, pattern, dense=dense)
    sync_checks.agree(torch, dist, rank, errors, f"{label}: GPipe / synchronized gradients", quiet=True)
    if utils.global_has_nonfinite_gradients(stage=stage, device=torch.device("cpu")) is not False:
        errors.append("正常梯度的全局异常标志应为 False")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: finite consensus", quiet=True)
    expected_norm = None
    if grad_clip > 0:
        before = sync_checks.snapshot_gradients(stage)
        before_model = state_snapshot(stage)
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), grad_clip)
        actual_norm = utils.clip_stage_gradients(stage=stage, max_norm=grad_clip,
                                                parallel=info, device=torch.device("cpu"))
        check_scalar(torch, actual_norm, expected_norm, errors, "direct pre-clipping global norm")
        sync_checks.compare_gradients(torch, stage, dense, errors, "direct clipping vs PyTorch")
        exact_tree(torch, stage.state_dict(), before_model, errors, "direct clipping model state")
        exact_tree(torch, optimizer.state_dict(), before_optimizer, errors, "direct clipping optimizer state")
        sync_checks.agree(torch, dist, rank, errors, f"{label}: direct global clipping", quiet=True)
        sync_checks.restore_gradients(stage, before)
    before_calls = optimizer.step_calls
    result = trainer.step_stage_optimizer(stage=stage, optimizer=optimizer, parallel=info,
                                          device=torch.device("cpu"), grad_clip=grad_clip)
    check_result(torch, result, True, expected_norm, errors)
    if optimizer.step_calls != before_calls + 1:
        errors.append("正常更新必须恰好调用一次 optimizer.step()")
    dense_optimizer.step()
    dense_optimizer.zero_grad(set_to_none=True)
    errors.extend(cleared_gradient_errors(stage))
    param_error, moment_error = compare_model_and_optimizer(torch, stage, dense, optimizer,
                                                           dense_optimizer, errors)
    sync_checks.agree(torch, dist, rank, errors, f"{label}: AdamW parameters / moments / clearing", quiet=True)
    errors.extend(replica_parameter_errors(torch, dist, stage, info, pp_size, dp_size, args))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: clip / AdamW / DP and tied identity")
    stats = [None] * dist.get_world_size()
    dist.all_gather_object(stats, {"parameter_error": param_error, "moment_error": moment_error})
    return {"status": "passed", "event": event, "micro_batch_size": micro_size, "pattern": pattern,
            "valid_labels_per_dp_replica": counts, "global_loss": float(loss.item()),
            "pre_clipping_norm": float(expected_norm.item()) if expected_norm is not None else None,
            "max_parameter_error": max(row["parameter_error"] for row in stats),
            "max_optimizer_state_error": max(row["moment_error"] for row in stats)}


def injection_cases(pp_size, world_size, clipping):
    result = [("nan", world_size-1, "last_tied"),
              ("posinf", 0, "ordinary"),
              ("neginf", pp_size//2 if pp_size > 1 else world_size//2, "ordinary")]
    if clipping:
        result.append(("norm_overflow", 0, "ordinary"))
    return result


def inject_gradient(stage, kind, role):
    if role == "last_tied":
        parameter = stage.lm_head.weight
        name = "lm_head.weight"
    else:
        name, parameter = next((name, p) for name, p in stage.named_parameters()
                               if name not in ("transformer.wte.weight", "lm_head.weight") and p.grad is not None)
    values = {"nan": float("nan"), "posinf": float("inf"),
              "neginf": float("-inf"), "norm_overflow": 1e200}
    parameter.grad[(0,) * parameter.grad.ndim] = values[kind]
    return name


def skipped_step(torch, dist, rank, args, trainer, utils, stage, info, optimizer,
                 dp_size, grad_clip, event, micro_size, kind, injection_rank, role, label):
    _, _, errors = prepare_gradients(torch, trainer, stage, info, args, dp_size,
                                     event, micro_size, "unequal_valid")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: prepared gradients", quiet=True)
    injected_name = inject_gradient(stage, kind, role) if rank == injection_rank else None
    before_model = state_snapshot(stage)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_grad = sync_checks.snapshot_gradients(stage)
    before_calls = optimizer.step_calls
    actual_bad = utils.global_has_nonfinite_gradients(stage=stage, device=torch.device("cpu"))
    expected_bad = kind != "norm_overflow"
    if actual_bad is not expected_bad:
        errors.append(f"global_has_nonfinite_gradients 应为 {expected_bad}，实际为 {actual_bad!r}")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: world-wide anomaly consensus", quiet=True)
    # 即使训练入口关闭裁剪，也独立检查裁剪函数对异常梯度的拒绝。
    clipped_norm = utils.clip_stage_gradients(stage=stage, max_norm=args.clip_norm,
                                              parallel=info, device=torch.device("cpu"))
    if clipped_norm is not None:
        errors.append("异常梯度/范数时 clip_stage_gradients 应在所有 rank 返回 None")
    exact_tree(torch, sync_checks.snapshot_gradients(stage), before_grad, errors,
               "rejected clipping preserves gradients (including NaN)")
    exact_tree(torch, stage.state_dict(), before_model, errors, "rejected clipping model state")
    exact_tree(torch, optimizer.state_dict(), before_optimizer, errors, "rejected clipping optimizer state")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: direct clipping rejection", quiet=True)
    result = trainer.step_stage_optimizer(stage=stage, optimizer=optimizer, parallel=info,
                                          device=torch.device("cpu"), grad_clip=grad_clip)
    check_result(torch, result, False, None, errors)
    if optimizer.step_calls != before_calls:
        errors.append("异常时调用了 optimizer.step()")
    exact_tree(torch, stage.state_dict(), before_model, errors, "skipped update model state")
    exact_tree(torch, optimizer.state_dict(), before_optimizer, errors, "skipped update entire optimizer state")
    errors.extend(cleared_gradient_errors(stage))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: all ranks skip / states unchanged / clearing")
    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, injected_name)
    return {"status": "passed", "kind": kind, "injected_rank": injection_rank,
            "injected_parameter": names[injection_rank], "all_ranks_skipped": True,
            "model_and_optimizer_unchanged": True, "gradients_cleared_to_none": True}


def zero_gradient_step(torch, dist, rank, args, trainer, dense, stage, info, optimizer,
                       dense_optimizer, pp_size, dp_size, grad_clip, label):
    errors = []
    for model in (dense, stage):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    expected_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), grad_clip) if grad_clip > 0 else None
    before_calls = optimizer.step_calls
    result = trainer.step_stage_optimizer(stage=stage, optimizer=optimizer, parallel=info,
                                          device=torch.device("cpu"), grad_clip=grad_clip)
    check_result(torch, result, True, expected_norm, errors)
    if optimizer.step_calls != before_calls + 1:
        errors.append("有限的全零梯度仍应执行 AdamW 更新")
    dense_optimizer.step()
    dense_optimizer.zero_grad(set_to_none=True)
    errors.extend(cleared_gradient_errors(stage))
    compare_model_and_optimizer(torch, stage, dense, optimizer, dense_optimizer, errors)
    sync_checks.agree(torch, dist, rank, errors, f"{label}: zero gradient is an AdamW step", quiet=True)
    errors.extend(replica_parameter_errors(torch, dist, stage, info, pp_size, dp_size, args))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: zero gradients / momentum / weight decay")
    return {"status": "passed", "stepped": True, "zero_gradient_keeps_optimizer_step": True}


def verify_model(torch, dist, rank, args, trainer, utils, GPT, GPTConfig, GPTStage,
                 info, pp_size, dp_size, bias, grad_clip):
    label = f"PP={pp_size}, DP={dp_size}, bias={bias}, grad_clip={grad_clip}"
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias)
    torch.manual_seed(args.seed)
    with redirect_stdout(io.StringIO()):
        dense = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        stage = GPTStage(config, pp_size=pp_size, pp_rank=info["pp_rank"]).to(
            device="cpu", dtype=torch.float64).eval()
        stage.load_dense_state_dict(dense.state_dict())
    errors = []
    if dense.transformer.wte.weight is not dense.lm_head.weight:
        errors.append("原版 GPT 应共享 wte/lm_head Parameter")
    if pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 的 GPTStage 应共享 wte/lm_head Parameter")
    for name, value in stage.state_dict().items():
        common.check_close(torch, value, dense.state_dict().get(name), f"loaded {name}", errors)
    sync_checks.agree(torch, dist, rank, errors, f"{label}: dense loading")
    optimizer = make_optimizer(torch, stage, args, audited=True)
    dense_optimizer = make_optimizer(torch, dense, args)
    synthetic = synthetic_clip_checks(torch, dist, rank, args, utils, dense, stage, info,
                                      optimizer, pp_size, dp_size, label)
    records = []
    event = 0
    for step in range(args.steps):
        micro = args.micro_batch_sizes[event % len(args.micro_batch_sizes)]
        pattern = "one_empty_replica" if step % 2 and dp_size > 1 else "unequal_valid"
        records.append(finite_step(torch, dist, rank, args, trainer, utils, dense, stage, info,
                       optimizer, dense_optimizer, pp_size, dp_size, grad_clip, event, micro,
                       pattern, f"{label}, normal step={step}"))
        event += 1
    skipped = []
    for kind, injection_rank, role in injection_cases(pp_size, dist.get_world_size(), grad_clip > 0):
        micro = args.micro_batch_sizes[event % len(args.micro_batch_sizes)]
        skipped.append(skipped_step(torch, dist, rank, args, trainer, utils, stage, info, optimizer,
                       dp_size, grad_clip, event, micro, kind, injection_rank, role,
                       f"{label}, injected {kind} at rank={injection_rank}"))
        event += 1
    micro = args.micro_batch_sizes[-1]
    recovery = finite_step(torch, dist, rank, args, trainer, utils, dense, stage, info, optimizer,
                            dense_optimizer, pp_size, dp_size, grad_clip, event, micro, "equal_valid",
                            f"{label}, recovery step")
    zero = zero_gradient_step(torch, dist, rank, args, trainer, dense, stage, info, optimizer,
                              dense_optimizer, pp_size, dp_size, grad_clip, f"{label}, zero gradients")
    return {"status": "passed", "bias": bias, "grad_clip": grad_clip,
            "synthetic_clipping": synthetic, "normal_steps": records, "skipped_steps": skipped,
            "recovery_step": recovery, "zero_gradient_step": zero,
            "successful_optimizer_steps": optimizer.step_calls}


def worker(rank, args, temporary):
    common.configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    world_size = args.pp_size * args.dp_size
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间文件发生变化，请保存后重新运行")
    trainer, GPT, GPTConfig, GPTStage = common.load_project(args.project)
    utils = sys.modules["parallel_utils"]
    for module, filename in ((trainer, "train_pp.py"), (utils, "parallel_utils.py")):
        for name in EXTRA_ARGUMENTS[filename]:
            if not callable(getattr(module, name, None)):
                raise AttributeError(f"{filename} 缺少可调用的 {name}")
    store = dist.FileStore(str(Path(temporary) / "rendezvous"), world_size)
    store.set_timeout(timedelta(seconds=args.timeout))
    try:
        dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=args.timeout))
        for pp_size, dp_size in args.cases:
            if rank == 0:
                print(f"[CASE] PP={pp_size}, DP={dp_size}", flush=True)
            info = trainer.setup_parallel(pp_size)
            errors = common.group_errors(dist, info, rank, pp_size, dp_size)
            if "tied_group" not in info:
                errors.append("setup_parallel 缺少 tied_group")
            sync_checks.agree(torch, dist, rank, errors, f"PP={pp_size}, DP={dp_size}: groups")
            models = [verify_model(torch, dist, rank, args, trainer, utils, GPT, GPTConfig, GPTStage,
                                   info, pp_size, dp_size, bias, clip)
                      for bias in args.biases for clip in (0.0, args.clip_norm)]
            if rank == 0:
                (Path(temporary) / f"result_pp{pp_size}.json").write_text(json.dumps(
                    {"pp_size": pp_size, "dp_size": dp_size, "status": "passed", "models": models},
                    ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    common.configure_threads(args.threads)
    report = {"status": "running", "scope": "PP_global_clipping_nonfinite_consensus_and_optimizer_step",
              "requested_cases": args.cases, "completed_cases": [],
              "not_checked": ["train_pp.main", "production optimizer creation", "torchrun/env rendezvous",
                              "CUDA/NCCL", "AMP", "dataset loading", "checkpoint", "performance"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / clipping and optimizer-step signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only_passed"
            common.write_report(args.report, report)
            print("[PASS] Static preflight completed; no real communication was run.", flush=True)
            return 0
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        torch.set_num_threads(args.threads)
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("当前 PyTorch 缺少 distributed/Gloo")
        report.update(torch_version=torch.__version__, device="cpu", backend="gloo", dtype="float64",
                      seed=args.seed, threads=args.threads, timeout_seconds=args.timeout,
                      model_config={"n_layer": args.n_layer, "n_head": args.n_head,
                                    "n_embd": args.n_embd, "vocab_size": args.vocab_size,
                                    "block_size": args.block_size, "dropout": 0.0},
                      batch_size=args.batch_size, seq_len=args.seq_len, biases=args.biases,
                      micro_batch_sizes=args.micro_batch_sizes, grad_clip_modes=[0.0, args.clip_norm],
                      warmup_steps=args.steps,
                      optimizer={"name": "AdamW", "learning_rate": args.learning_rate,
                                 "weight_decay": args.weight_decay, "betas": args.betas,
                                 "foreach": False, "fused": False})
        world_size = args.pp_size * args.dp_size
        print(f"[INFO] {world_size} CPU workers; threads={args.threads}; real cases={args.cases}.", flush=True)
        print("[INFO] Checking norm, clipping, AdamW updates/moments, anomaly skips and recovery.", flush=True)
        print("[INFO] AdamW is created by the verifier; production training setup is pending.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_optimizer_") as temporary:
            try:
                mp.spawn(worker, args=(args, temporary), nprocs=world_size, join=True)
            finally:
                for pp_size, _ in args.cases:
                    path = Path(temporary) / f"result_pp{pp_size}.json"
                    if path.is_file():
                        report["completed_cases"].append(json.loads(path.read_text(encoding="utf-8")))
        if len(report["completed_cases"]) != len(args.cases):
            raise RuntimeError("worker 检查未全部完成")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        common.write_report(args.report, report)
        raise
    common.write_report(args.report, report)
    print("[PASS] PP clipping and optimizer-step numerical verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
