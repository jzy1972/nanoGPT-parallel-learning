#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 nanoGPT PP+DP 的优化器创建、数据流及训练集成。

把本文件与 verify_pp_optimizer.py、verify_pp_grad_sync.py、
verify_pp_microbatch.py 放在 nanoGPT 根目录，运行：
    python verify_pp_training.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_training_report.json

第一阶段：CPU/Gloo、float64、小模型、dropout=0；直接调用项目的
build_stage_optimizer、get_pp_batch、aggregate_pp_loss、train_pp_step。
覆盖参数分组/冻结参数/共享参数去重、连续数据采样及 RNG、最短合法
数据文件、首尾数据职责、不同有效标签数的全局 loss、关闭/开启裁剪、
连续 AdamW 参数和状态、单 rank NaN 后全部跳过并恢复。

第二阶段：启动一组全新的 CPU worker，设置启动环境和 sys.argv，
真正调用项目 train_pp.main()，默认运行指定 PP/DP 的三轮 float32
训练。只用观察用包装器和 optimizer pre-hook 记录实际数据、梯度、
初始/更新后参数及优化器状态，不替换算法或模型。由父进程组装完整
GPT，对每轮实际全局 batch 做独立参考训练，并检查日志和 RNG。
这验证 env:// 启动环境及 main；没有运行 torchrun 命令行启动器。

默认 pp_size*dp_size 个 CPU worker，第一阶段检查同一 world_size
的可用分段数：默认 (1,4)、(2,2)、(4,1)。只是测试配置，不限制算法
规模。第二阶段只检查命令行指定的 PP/DP、bias=True（--bias false
时用 False）、clip_norm。--single-case 可缩小第一阶段范围。

自动创建临时 uint16 token 文件，不修改项目源码，不保存训练 checkpoint。
依赖文件均须来自同一目录。--preflight-only 不导入 torch/NumPy。
--skip-main 只检查辅助函数集成，报告会明确标记 main 未验证。
不验证 CUDA/NCCL、AMP、非零 dropout 数值等价性、LR 调度、评估、
checkpoint/resume、真实语料、torchrun CLI 或训练性能。
"""

import argparse
import ast
from contextlib import redirect_stdout
import copy
from datetime import timedelta
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile

try:
    import verify_pp_optimizer as opt_checks
except ModuleNotFoundError as exc:
    if exc.name == "verify_pp_optimizer":
        raise SystemExit("请把 verify_pp_optimizer.py 与本文件放在同一目录。") from exc
    raise

sync_checks = opt_checks.sync_checks
common = opt_checks.common
NEW_ARGUMENTS = {
    "build_stage_optimizer": ("stage", "learning_rate", "weight_decay", "beta1", "beta2"),
    "get_pp_batch": ("stage", "parallel", "data_path", "batch_size", "seq_length", "device", "generator"),
    "aggregate_pp_loss": ("stage", "local_loss", "grad_weight", "parallel", "device"),
    "train_pp_step": ("stage", "optimizer", "parallel", "token_ids", "targets", "batch_size",
                      "seq_length", "micro_batch_size", "device", "grad_clip", "activation_dtype"),
    "main": (),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--pp-size", type=int, default=2)
    p.add_argument("--dp-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=5)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--micro-batch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--clip-norm", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=0.0006)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--steps", type=int, default=3, help="连续正常更新次数，至少 2；也用于 main")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--timeout", type=int, default=120, help="通信超时秒数")
    p.add_argument("--bias", choices=("true", "false", "both"), default="both")
    p.add_argument("--single-case", action="store_true")
    p.add_argument("--skip-main", action="store_true", help="明确跳过实际 train_pp.main 检查")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd",
                 "vocab_size", "block_size", "seq_len", "batch_size", "timeout"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.steps < 2:
        p.error("--steps 至少为 2")
    if args.pp_size > args.n_layer or args.n_embd % args.n_head:
        p.error("需要 pp_size <= n_layer，且 n_embd 能被 n_head 整除")
    if args.seq_len > args.block_size or not 2 <= args.vocab_size <= 65536:
        p.error("需要 seq_len <= block_size，uint16 数据测试的词表范围为 2..65536")
    if args.batch_size * args.seq_len < 2:
        p.error("本验证需要至少两个标签位置")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay 必须是非负的有限数")
    if any(not math.isfinite(b) or not 0 <= b < 1 for b in args.betas):
        p.error("每个 beta 必须满足 0 <= beta < 1")
    args.micro_batch_sizes = sorted(set(args.micro_batch_sizes or [1, 2, args.batch_size]))
    if any(m <= 0 for m in args.micro_batch_sizes):
        p.error("每个 micro_batch_size 必须为正整数")
    world = args.pp_size * args.dp_size
    args.cases = ([(args.pp_size, args.dp_size)] if args.single_case else
                  [(pp, world // pp) for pp in range(1, min(args.n_layer, world) + 1) if world % pp == 0])
    args.biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    return args


def preflight(project):
    expected = Path(__file__).resolve().parent / "verify_pp_optimizer.py"
    if Path(opt_checks.__file__).resolve() != expected:
        raise RuntimeError(f"优化器验证辅助文件应来自 {expected}")
    opt_checks.preflight(project)
    path = project / "train_pp.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for name, required in NEW_ARGUMENTS.items():
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise AttributeError(f"train_pp.py 需要定义一个顶层 {name}()")
        supplied = {a.arg for a in nodes[0].args.args + nodes[0].args.kwonlyargs}
        if set(required) - supplied:
            raise TypeError(f"train_pp.py 的 {name} 缺少可按关键字传入的参数 {sorted(set(required)-supplied)}")


def fingerprints(project):
    result = common.source_hashes(project)
    for module in (common, sync_checks, opt_checks):
        path = Path(module.__file__)
        result[f"checker:{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = Path(__file__)
    result[f"checker:{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def clone_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(torch, v) for v in value)
    return copy.deepcopy(value)


def optimizer_named_state(torch, stage, optimizer):
    return {name: clone_tree(torch, optimizer.state.get(param, {}))
            for name, param in stage.named_parameters() if param.requires_grad}


def group_metadata(stage, optimizer):
    names = {id(p): name for name, p in stage.named_parameters()}
    return [{"params": [names.get(id(p), "<unowned>") for p in group["params"]],
             "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
             "betas": tuple(float(b) for b in group["betas"]),
             "eps": float(group["eps"]), "amsgrad": bool(group.get("amsgrad", False)),
             "maximize": bool(group.get("maximize", False))}
            for group in optimizer.param_groups]


def optimizer_errors(torch, stage, optimizer, args):
    errors = []
    if not isinstance(optimizer, torch.optim.AdamW):
        return ["build_stage_optimizer 必须返回 AdamW"]
    owned = {id(p): (name, p) for name, p in stage.named_parameters() if p.requires_grad}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in actual]
    if len(ids) != len(set(ids)):
        errors.append("同一 Parameter 重复进入优化器（检查 PP=1 的共享权重）")
    if set(ids) != set(owned):
        errors.append("优化器参数未精确覆盖本 stage 的可训练参数，或包含冻结/其他参数")
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in owned:
                continue
            name = owned[id(p)][0]
            wd = args.weight_decay if p.dim() >= 2 else 0.0
            if not math.isclose(float(group["weight_decay"]), wd, rel_tol=0, abs_tol=1e-15):
                errors.append(f"{name}: weight_decay 应为 {wd}")
        if float(group["lr"]) != args.learning_rate or tuple(group["betas"]) != tuple(args.betas):
            errors.append("参数组 learning_rate/betas 与传入参数不同")
        if float(group["eps"]) != 1e-8 or group.get("amsgrad", False) or group.get("maximize", False):
            errors.append("初版 AdamW 应使用 eps=1e-8、amsgrad=False、maximize=False")
    return errors


def make_project_optimizer(trainer, stage, args):
    return trainer.build_stage_optimizer(stage=stage, learning_rate=args.learning_rate,
                                         weight_decay=args.weight_decay,
                                         beta1=args.betas[0], beta2=args.betas[1])


def attach_audit(torch, stage, optimizer):
    audit = {"calls": 0, "gradients": None}
    def hook(opt, positional, keywords):
        audit["calls"] += 1
        audit["gradients"] = clone_tree(torch, sync_checks.snapshot_gradients(stage))
    handle = optimizer.register_step_pre_hook(hook)
    return audit, handle


def data_generator(torch, args, dp_rank):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 10000 + dp_rank)
    return generator


def oracle_batch(torch, np, path, args, generator):
    # 独立读字节，避免使用项目 get_pp_batch 或复用其采样结果作为参考。
    data = np.fromfile(path, dtype=np.uint16)
    starts = torch.randint(len(data) - args.seq_len, (args.batch_size,),
                           generator=generator, device="cpu").tolist()
    x = torch.tensor([[int(data[i + j]) for j in range(args.seq_len)] for i in starts], dtype=torch.long)
    y = torch.tensor([[int(data[i + j + 1]) for j in range(args.seq_len)] for i in starts], dtype=torch.long)
    return x, y


def check_batch(torch, stage, x, y, expected_x, expected_y, errors):
    for name, actual, expected, should_exist in (
        ("token_ids", x, expected_x, stage.is_first), ("targets", y, expected_y, stage.is_last)
    ):
        if not should_exist:
            if actual is not None:
                errors.append(f"当前 stage 的 {name} 应为 None")
        elif not torch.is_tensor(actual):
            errors.append(f"当前 stage 的 {name} 应为 Tensor")
        elif actual.dtype != torch.long or actual.device.type != "cpu" or not torch.equal(actual, expected):
            errors.append(f"{name}: dtype/device/采样内容或标签位移不正确")


def verify_data(torch, np, dist, rank, args, trainer, stage, info):
    errors, records = [], []
    actual_gen = data_generator(torch, args, info["dp_rank"])
    expected_gen = data_generator(torch, args, info["dp_rank"])
    for event, path in enumerate([args.dataset_path] * 3 + [args.minimal_path]):
        before_gen = actual_gen.get_state().clone()
        before_global = torch.get_rng_state().clone()
        expected_x, expected_y = oracle_batch(torch, np, path, args, expected_gen)
        x, y = trainer.get_pp_batch(stage=stage, parallel=info, data_path=str(path),
                                    batch_size=args.batch_size, seq_length=args.seq_len,
                                    device=torch.device("cpu"), generator=actual_gen)
        check_batch(torch, stage, x, y, expected_x, expected_y, errors)
        expected_state = expected_gen.get_state() if stage.is_first else before_gen
        if not torch.equal(actual_gen.get_state(), expected_state):
            errors.append("数据 generator 状态错误：检查首 stage 采样、持续推进及传入 generator")
        if not torch.equal(torch.get_rng_state(), before_global):
            errors.append("数据采样改变了模型使用的全局 torch RNG")
        sync_checks.agree(torch, dist, rank, errors, f"data batch {event}", quiet=True)
        records.append({"batch": event, "minimal_file": event == 3})
    return {"status": "passed", "batches": records, "source_only_sampling": True,
            "pp_endpoint_roles": True, "independent_generator_state": True}


def verify_loss(torch, dist, rank, args, trainer, stage, info, dp_size):
    records = []
    patterns = [[1] * dp_size, [d + 1 for d in range(dp_size)]]
    if dp_size > 1:
        patterns.append([0] + [d + 1 for d in range(1, dp_size)])
    for counts in patterns:
        d = info["dp_rank"]
        losses = [1.0 + 0.25 * q for q in range(dp_size)]
        expected = sum(n * value for n, value in zip(counts, losses)) / sum(counts)
        weight = dp_size * counts[d] / sum(counts)
        local = (torch.tensor(losses[d] if counts[d] else float("nan"), dtype=torch.float64,
                              requires_grad=True) if stage.is_last else None)
        before = local.detach().clone() if local is not None else None
        actual = trainer.aggregate_pp_loss(stage=stage, local_loss=local, grad_weight=weight,
                                           parallel=info, device=torch.device("cpu"))
        errors = []
        if not isinstance(actual, float):
            errors.append("aggregate_pp_loss 应返回 Python float")
        opt_checks.check_scalar(torch, actual, expected, errors, "weighted global loss on every rank")
        if local is not None:
            opt_checks.exact_tree(torch, local.detach(), before, errors, "local loss must not be modified")
            if not local.requires_grad or local.grad is not None:
                errors.append("汇总函数改变了 local_loss 的 autograd 状态")
        sync_checks.agree(torch, dist, rank, errors, "synthetic global loss", quiet=True)
        records.append({"counts": counts, "expected_global_loss": expected})
    return {"status": "passed", "cases": records}


def call_train_step(torch, trainer, stage, optimizer, info, args, x, y, micro, clip):
    return trainer.train_pp_step(stage=stage, optimizer=optimizer, parallel=info,
                                 token_ids=x, targets=y, batch_size=args.batch_size,
                                 seq_length=args.seq_len, micro_batch_size=micro,
                                 device=torch.device("cpu"), grad_clip=clip,
                                 activation_dtype=torch.float64)


def training_batches(torch, np, args, dp_size, event, micro):
    if event == 0:
        return [oracle_batch(torch, np, args.dataset_path, args, data_generator(torch, args, d))
                for d in range(dp_size)], "sampled_uint16"
    pattern = ("unequal_valid" if event == 1 else
               "one_empty_replica" if dp_size > 1 and event % 2 == 0 else "equal_valid")
    round_args = copy.copy(args)
    round_args.seed += 10007 * event
    return sync_checks.make_data(torch, round_args, dp_size, micro, pattern), pattern


def check_hook_gradients(torch, gradients, dense, errors):
    if gradients is None:
        errors.append("未观察到 optimizer.step() 的更新前梯度")
        return
    expected = dict(dense.named_parameters(remove_duplicate=False))
    for name, value in gradients.items():
        common.check_close(torch, value, expected[name].grad, f"optimizer input gradient {name}", errors)


def finite_training_step(torch, np, dist, rank, args, trainer, dense, stage, info,
                         optimizer, reference_optimizer, audit, dp_size, clip, event, label):
    micro = args.micro_batch_sizes[event % len(args.micro_batch_sizes)]
    batches, pattern = training_batches(torch, np, args, dp_size, event, micro)
    expected_x, expected_y = batches[info["dp_rank"]]
    x = expected_x if stage.is_first else None
    y = expected_y if stage.is_last else None
    errors = []
    if event == 0:
        x, y = trainer.get_pp_batch(stage=stage, parallel=info, data_path=str(args.dataset_path),
                                    batch_size=args.batch_size, seq_length=args.seq_len,
                                    device=torch.device("cpu"),
                                    generator=data_generator(torch, args, info["dp_rank"]))
        check_batch(torch, stage, x, y, expected_x, expected_y, errors)
    global_x = torch.cat([a for a, _ in batches])
    global_y = torch.cat([b for _, b in batches])
    _, reference_loss, _ = sync_checks.backward_reference(torch, dense, global_x, global_y)
    reference_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), clip) if clip > 0 else None
    sync_checks.agree(torch, dist, rank, errors, f"{label}: reference batch", quiet=True)
    before_calls = audit["calls"]
    audit["gradients"] = None
    result = call_train_step(torch, trainer, stage, optimizer, info, args, x, y, micro, clip)
    opt_checks.check_result(torch, result, True, reference_norm, errors)
    if not isinstance(result, dict) or "loss" not in result:
        errors.append("train_pp_step 返回值缺少 loss")
    else:
        opt_checks.check_scalar(torch, result["loss"], reference_loss, errors, "returned global loss")
    if audit["calls"] != before_calls + 1:
        errors.append("一个完整 batch 应恰好调用一次 optimizer.step()")
    check_hook_gradients(torch, audit["gradients"], dense, errors)
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)
    param_error, moment_error = opt_checks.compare_model_and_optimizer(
        torch, stage, dense, optimizer, reference_optimizer, errors)
    errors.extend(opt_checks.cleared_gradient_errors(stage))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: loss / gradients / AdamW / clearing")
    errors.extend(opt_checks.replica_parameter_errors(
        torch, dist, stage, info, stage.pp_size, dp_size, args))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: replicas", quiet=True)
    return {"status": "passed", "event": event, "pattern": pattern, "micro_batch_size": micro,
            "global_loss": float(reference_loss.item()), "parameter_error": param_error,
            "optimizer_moment_error": moment_error}


def skipped_training_step(torch, np, dist, rank, args, trainer, dense, stage, info,
                          optimizer, audit, dp_size, clip, label):
    event = args.steps + 1
    micro = args.micro_batch_sizes[0]
    batches, _ = training_batches(torch, np, args, dp_size, event, micro)
    x, y = batches[info["dp_rank"]]
    _, expected_loss, _ = sync_checks.backward_reference(
        torch, dense, torch.cat([a for a, _ in batches]), torch.cat([b for _, b in batches]))
    before_model = opt_checks.state_snapshot(stage)
    before_optimizer = clone_tree(torch, optimizer.state_dict())
    before_calls = audit["calls"]
    original = trainer.sync_stage_gradients
    def injected_sync(*positional, **keywords):
        weight = original(*positional, **keywords)
        if rank == args.pp_size * args.dp_size - 1:
            opt_checks.inject_gradient(stage, "nan", "last_tied")
        return weight
    trainer.sync_stage_gradients = injected_sync
    try:
        result = call_train_step(torch, trainer, stage, optimizer, info, args,
                                 x if stage.is_first else None, y if stage.is_last else None, micro, clip)
    finally:
        trainer.sync_stage_gradients = original
    errors = []
    opt_checks.check_result(torch, result, False, None, errors)
    if not isinstance(result, dict) or "loss" not in result:
        errors.append("异常更新的 train_pp_step 返回值缺少 loss")
    else:
        opt_checks.check_scalar(torch, result["loss"], expected_loss, errors, "skipped step global loss")
    if audit["calls"] != before_calls:
        errors.append("异常梯度时调用了 optimizer.step()")
    opt_checks.exact_tree(torch, stage.state_dict(), before_model, errors, "skip preserves parameters")
    opt_checks.exact_tree(torch, optimizer.state_dict(), before_optimizer, errors, "skip preserves optimizer")
    errors.extend(opt_checks.cleared_gradient_errors(stage))
    sync_checks.agree(torch, dist, rank, errors, f"{label}: injected NaN / all skip / states preserved")
    return {"status": "passed", "injected_rank": args.pp_size * args.dp_size - 1}


def model_config(GPTConfig, args, bias):
    return GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                     vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias)


def verify_model(torch, np, dist, rank, args, trainer, GPT, GPTConfig, GPTStage, info, pp, dp, bias, clip):
    label = f"PP={pp}, DP={dp}, bias={bias}, grad_clip={clip}"
    torch.manual_seed(args.seed)
    config = model_config(GPTConfig, args, bias)
    with redirect_stdout(io.StringIO()):
        dense = GPT(config).to(dtype=torch.float64).train()
        stage = GPTStage(config, pp_size=pp, pp_rank=info["pp_rank"]).to(dtype=torch.float64).train()
        stage.load_dense_state_dict(dense.state_dict())
    initial = opt_checks.state_snapshot(stage)
    frozen = next(stage.parameters())
    frozen.requires_grad_(False)
    try:
        probe = make_project_optimizer(trainer, stage, args)
        errors = optimizer_errors(torch, stage, probe, args)
        opt_checks.exact_tree(torch, stage.state_dict(), initial, errors, "factory preserves model")
        if probe.state:
            errors.append("新建 AdamW 应尚未积累 optimizer state")
    finally:
        frozen.requires_grad_(True)
    optimizer = make_project_optimizer(trainer, stage, args)
    errors.extend(optimizer_errors(torch, stage, optimizer, args))
    if optimizer.state:
        errors.append("生产优化器创建时已有状态")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: project AdamW factory / groups / frozen / alias")
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer = opt_checks.make_optimizer(torch, dense, args)
    audit, handle = attach_audit(torch, stage, optimizer)
    records = []
    try:
        for event in range(args.steps):
            records.append(finite_training_step(torch, np, dist, rank, args, trainer, dense, stage, info,
                optimizer, reference_optimizer, audit, dp, clip, event, f"{label}, step={event}"))
        skipped = skipped_training_step(torch, np, dist, rank, args, trainer, dense, stage, info,
                                         optimizer, audit, dp, clip, label)
        recovery = finite_training_step(torch, np, dist, rank, args, trainer, dense, stage, info,
            optimizer, reference_optimizer, audit, dp, clip, args.steps + 2, f"{label}, recovery")
    finally:
        handle.remove()
    return {"status": "passed", "bias": bias, "grad_clip": clip, "steps": records,
            "nonfinite_skip": skipped, "recovery": recovery}


def bound_timeouts(dist, seconds):
    # 仅设置测试超时，不替换 rendezvous/backend/通信逻辑。
    originals = {name: getattr(dist, name) for name in ("init_process_group", "new_group")}
    for name, original in originals.items():
        def bounded(*positional, _original=original, **keywords):
            keywords.setdefault("timeout", timedelta(seconds=seconds))
            return _original(*positional, **keywords)
        setattr(dist, name, bounded)
    return originals


def restore_timeouts(dist, originals):
    for name, original in originals.items():
        setattr(dist, name, original)


def helper_worker(rank, args, temporary):
    common.configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间源码/验证器发生变化")
    trainer, GPT, GPTConfig, GPTStage = common.load_project(args.project)
    for name in NEW_ARGUMENTS:
        if not callable(getattr(trainer, name, None)):
            raise AttributeError(f"train_pp.py 缺少可调用的 {name}")
    world = args.pp_size * args.dp_size
    store = dist.FileStore(str(Path(temporary) / "helper_rendezvous"), world)
    store.set_timeout(timedelta(seconds=args.timeout))
    originals = bound_timeouts(dist, args.timeout)
    try:
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
        for pp, dp in args.cases:
            if rank == 0:
                print(f"[CASE] PP={pp}, DP={dp}", flush=True)
            info = trainer.setup_parallel(pp)
            sync_checks.agree(torch, dist, rank, common.group_errors(dist, info, rank, pp, dp),
                              f"PP={pp}, DP={dp}: groups")
            with redirect_stdout(io.StringIO()):
                stage = GPTStage(model_config(GPTConfig, args, True), pp_size=pp, pp_rank=info["pp_rank"])
            data_record = verify_data(torch, np, dist, rank, args, trainer, stage, info)
            loss_record = verify_loss(torch, dist, rank, args, trainer, stage, info, dp)
            sync_checks.agree(torch, dist, rank, [], f"PP={pp}, DP={dp}: data / RNG / endpoint roles / weighted loss")
            models = [verify_model(torch, np, dist, rank, args, trainer, GPT, GPTConfig, GPTStage,
                                   info, pp, dp, bias, clip)
                      for bias in args.biases for clip in (0.0, args.clip_norm)]
            if rank == 0:
                (Path(temporary) / f"helper_pp{pp}.json").write_text(json.dumps({
                    "status": "passed", "pp_size": pp, "dp_size": dp,
                    "data": data_record, "loss": loss_record, "models": models,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        restore_timeouts(dist, originals)


def main_argv(args, bias):
    values = {
        "--pp-size": args.pp_size, "--device-type": "cpu", "--seed": args.seed,
        "--n-layer": args.n_layer, "--n-head": args.n_head, "--n-embd": args.n_embd,
        "--vocab-size": args.vocab_size, "--block-size": args.block_size, "--dropout": 0.0,
        "--data-dir": str(Path(args.dataset_path).parent), "--batch-size": args.batch_size,
        "--seq-length": args.seq_len, "--micro-batch-size": 2,
        "--learning-rate": args.learning_rate, "--weight-decay": args.weight_decay,
        "--beta1": args.betas[0], "--beta2": args.betas[1], "--grad-clip": args.clip_norm,
        "--max-iters": args.steps, "--log-interval": 1,
    }
    argv = [str(args.project / "train_pp.py")]
    for flag, value in values.items():
        argv.extend([flag, str(value)])
    if not bias:
        argv.append("--no-bias")
    return argv


def main_worker(rank, args, temporary, port, bias):
    common.configure_threads(args.threads)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.pp_size * args.dp_size), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("main 验证开始前源码发生变化")
    trainer, _, _, _ = common.load_project(args.project)
    record = {"rank": rank, "steps": [], "data_calls": [], "factory_calls": 0}
    captured, handles = {}, []
    original_factory, original_data, original_step = (
        trainer.build_stage_optimizer, trainer.get_pp_batch, trainer.train_pp_step)
    factory_signature = inspect.signature(original_factory)
    data_signature = inspect.signature(original_data)
    step_signature = inspect.signature(original_step)

    def observed_factory(*positional, **keywords):
        bound = factory_signature.bind(*positional, **keywords)
        stage = bound.arguments["stage"]
        optimizer = original_factory(*positional, **keywords)
        errors = optimizer_errors(torch, stage, optimizer, args)
        if errors:
            raise AssertionError("main optimizer: " + "; ".join(errors))
        captured.update(stage=stage, optimizer=optimizer)
        audit, handle = attach_audit(torch, stage, optimizer)
        captured["audit"] = audit
        handles.append(handle)
        record["factory_calls"] += 1
        record["optimizer_groups"] = group_metadata(stage, optimizer)
        return optimizer

    def observed_data(*positional, **keywords):
        bound = data_signature.bind(*positional, **keywords)
        generator = bound.arguments["generator"]
        call = {"seed": generator.initial_seed(), "generator_before": generator.get_state().clone(),
                "global_rng_before": torch.get_rng_state().clone()}
        result = original_data(*positional, **keywords)
        call.update(generator_after=generator.get_state().clone(), global_rng_after=torch.get_rng_state().clone(),
                    token_ids=clone_tree(torch, result[0]), targets=clone_tree(torch, result[1]))
        record["data_calls"].append(call)
        return result

    def observed_step(*positional, **keywords):
        if len(record["steps"]) >= args.steps:
            raise AssertionError("main 超过 --max-iters 指定的训练轮数")
        bound = step_signature.bind(*positional, **keywords)
        bound.apply_defaults()
        stage, optimizer = bound.arguments["stage"], bound.arguments["optimizer"]
        if stage is not captured.get("stage") or optimizer is not captured.get("optimizer"):
            raise AssertionError("main 未使用 build_stage_optimizer 返回的优化器或对应 stage")
        if not record["steps"]:
            record["initial_model"] = clone_tree(torch, stage.state_dict())
            record["model_dtype"] = str(next(stage.parameters()).dtype)
            record["config"] = {key: getattr(stage.config, key) for key in
                                 ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "dropout", "bias")}
        audit = captured["audit"]
        audit["gradients"] = None
        event = {"token_ids": clone_tree(torch, bound.arguments["token_ids"]),
                 "targets": clone_tree(torch, bound.arguments["targets"]),
                 "calls_before": audit["calls"], "training": stage.training,
                 "batch_size": bound.arguments["batch_size"], "seq_length": bound.arguments["seq_length"],
                 "micro_batch_size": bound.arguments["micro_batch_size"], "grad_clip": bound.arguments["grad_clip"],
                 "activation_dtype": str(bound.arguments["activation_dtype"])}
        result = original_step(*positional, **keywords)
        event.update(result=clone_tree(torch, result), calls_after=audit["calls"],
                     gradients=clone_tree(torch, audit["gradients"]),
                     model=clone_tree(torch, stage.state_dict()),
                     optimizer=optimizer_named_state(torch, stage, optimizer),
                     gradients_cleared=all(p.grad is None for p in stage.parameters()))
        record["steps"].append(event)
        return result

    trainer.build_stage_optimizer = observed_factory
    trainer.get_pp_batch = observed_data
    trainer.train_pp_step = observed_step
    originals = bound_timeouts(dist, args.timeout)
    output = io.StringIO()
    previous_argv = sys.argv
    sys.argv = main_argv(args, bias)
    try:
        with redirect_stdout(output):
            try:
                result = trainer.main()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
                result = exc.code
        if result not in (None, 0):
            raise AssertionError(f"main 返回错误状态 {result!r}")
        if record["factory_calls"] != 1 or len(record["steps"]) != args.steps:
            raise AssertionError("main 的优化器创建次数或 train_pp_step 调用次数错误")
        if len(record["data_calls"]) != args.steps:
            raise AssertionError("main 的 get_pp_batch 调用次数与训练轮数不一致")
        if dist.is_initialized():
            raise AssertionError("main 返回时未销毁默认进程组")
        if fingerprints(args.project) != args.expected_hashes:
            raise AssertionError("main 修改了项目源码")
        record.update(status="passed", stdout=output.getvalue())
        torch.save(record, Path(temporary) / f"main_rank{rank}.pt")
    finally:
        (Path(temporary) / f"main_rank{rank}_stdout.txt").write_text(output.getvalue(), encoding="utf-8")
        sys.argv = previous_argv
        trainer.build_stage_optimizer, trainer.get_pp_batch, trainer.train_pp_step = (
            original_factory, original_data, original_step)
        for handle in handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
        restore_timeouts(dist, originals)


def main_close(torch, actual, expected, errors, label, kind="parameter"):
    tolerances = {"parameter": (1e-4, 2e-6), "gradient": (2e-4, 1e-8),
                  "moment": (3e-4, 1e-9), "loss": (1e-6, 2e-6), "norm": (1e-4, 1e-6), "step": (0.0, 0.0)}
    if actual is None or expected is None:
        errors.append(f"{label}: 返回张量/数值为 None")
        return
    try:
        a = torch.as_tensor(actual) if not torch.is_tensor(actual) else actual
        e = torch.as_tensor(expected) if not torch.is_tensor(expected) else expected
        # Python float 使用 float64，避免检查器自己再次舍入。
        if not torch.is_tensor(actual):
            a = torch.as_tensor(actual, dtype=torch.float64)
        if not torch.is_tensor(expected):
            e = torch.as_tensor(expected, dtype=torch.float64)
        rtol, atol = tolerances[kind]
        torch.testing.assert_close(a.to(torch.float64), e.to(torch.float64), rtol=rtol, atol=atol)
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"{label}: {exc}")


def check_main_records(torch, np, args, temporary, bias):
    world, pp, dp = args.pp_size * args.dp_size, args.pp_size, args.dp_size
    records = [torch.load(Path(temporary) / f"main_rank{r}.pt", map_location="cpu", weights_only=True)
               for r in range(world)]
    errors = []
    initial = {}
    for p in range(pp):
        initial.update(records[p]["initial_model"])
    if not torch.equal(initial["transformer.wte.weight"], initial["lm_head.weight"]):
        errors.append("实际 main 的首尾共享权重初始值不同")
    for r in range(world):
        opt_checks.exact_tree(torch, records[r]["initial_model"], records[r % pp]["initial_model"],
                              errors, f"rank {r}: initial DP replicas")
    _, GPT, GPTConfig, _ = common.load_project(args.project)
    expected_config = {"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                       "vocab_size": args.vocab_size, "block_size": args.block_size, "dropout": 0.0, "bias": bias}
    for r, record in enumerate(records):
        if record["config"] != expected_config:
            errors.append(f"rank {r}: main 未正确使用命令行模型参数")
        if record["model_dtype"] != "torch.float32":
            errors.append(f"rank {r}: 本轮 main 检查应使用 float32 模型")
    if errors:
        raise AssertionError("实际 main 初始化失败：" + "; ".join(errors[:12]))
    with redirect_stdout(io.StringIO()):
        dense = GPT(model_config(GPTConfig, args, bias)).to(dtype=torch.float32).train()
    dense.load_state_dict(initial, strict=True)
    reference_optimizer = opt_checks.make_optimizer(torch, dense, args)
    generators = [data_generator(torch, args, d) for d in range(dp)]
    reference_parameters = dict(dense.named_parameters(remove_duplicate=False))
    step_reports = []
    for event in range(args.steps):
        batches = []
        for d in range(dp):
            before = generators[d].get_state().clone()
            expected_x, expected_y = oracle_batch(torch, np, args.dataset_path, args, generators[d])
            after = generators[d].get_state().clone()
            for p in range(pp):
                r = d * pp + p
                data_record = records[r]["data_calls"][event]
                step = records[r]["steps"][event]
                flags = argparse.Namespace(is_first=p == 0, is_last=p == pp - 1)
                check_batch(torch, flags, data_record["token_ids"], data_record["targets"], expected_x, expected_y, errors)
                check_batch(torch, flags, step["token_ids"], step["targets"], expected_x, expected_y, errors)
                if p == 0 and data_record["seed"] != args.seed + 10000 + d:
                    errors.append(f"rank {r}: 实际 main 的数据 generator seed 未按 dp_rank 设置")
                expected_before = before if p == 0 else records[r]["data_calls"][0]["generator_before"]
                expected_after = after if p == 0 else expected_before
                for field, wanted in (("generator_before", expected_before), ("generator_after", expected_after)):
                    if not torch.equal(data_record[field], wanted):
                        errors.append(f"rank {r}, event {event}: {field} 不正确")
                if not torch.equal(data_record["global_rng_before"], data_record["global_rng_after"]):
                    errors.append(f"rank {r}: 实际 main 数据采样改变全局 torch RNG")
            x = records[d * pp]["steps"][event]["token_ids"]
            y = records[(d + 1) * pp - 1]["steps"][event]["targets"]
            batches.append((x, y))
        if errors:
            raise AssertionError("实际 main 数据流失败：" + "; ".join(errors[:12]))
        _, expected_loss, _ = sync_checks.backward_reference(
            torch, dense, torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches]))
        expected_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), args.clip_norm)
        for r in range(world):
            step = records[r]["steps"][event]
            result = step["result"]
            if not step["training"] or not step["gradients_cleared"]:
                errors.append(f"rank {r}: stage.train() 或轮末梯度清空不正确")
            if step["calls_before"] != event or step["calls_after"] != event + 1:
                errors.append(f"rank {r}: 每轮 optimizer.step 调用次数错误")
            if (step["batch_size"], step["seq_length"], step["micro_batch_size"], step["grad_clip"],
                step["activation_dtype"]) != (args.batch_size, args.seq_len, 2, args.clip_norm, "torch.float32"):
                errors.append(f"rank {r}: main 未正确传递训练参数")
            if not isinstance(result, dict) or result.get("stepped") is not True:
                errors.append(f"rank {r}: 正常轮次未报告 stepped=True")
                continue
            main_close(torch, result.get("loss"), expected_loss, errors, f"rank {r}: global loss", "loss")
            main_close(torch, result.get("grad_norm"), expected_norm, errors, f"rank {r}: global norm", "norm")
            gradients = step["gradients"]
            if not isinstance(gradients, dict):
                errors.append(f"rank {r}: 未记录到更新前梯度")
            else:
                for name, grad in gradients.items():
                    main_close(torch, grad, reference_parameters[name].grad, errors,
                               f"rank {r}: gradient {name}", "gradient")
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        state = dense.state_dict()
        for r in range(world):
            step = records[r]["steps"][event]
            for name, value in step["model"].items():
                main_close(torch, value, state[name], errors, f"rank {r}: parameter {name}")
            for name, actual in step["optimizer"].items():
                expected = reference_optimizer.state.get(reference_parameters[name], {})
                if actual.keys() != expected.keys():
                    errors.append(f"rank {r}: optimizer {name} 状态字段不一致")
                    continue
                for field in expected:
                    main_close(torch, actual[field], expected[field], errors, f"rank {r}: {name}.{field}",
                               "step" if field == "step" else "moment")
            opt_checks.exact_tree(torch, step["model"], records[r % pp]["steps"][event]["model"],
                                  errors, f"rank {r}: exact DP replica parameters")
        for d in range(dp):
            first = records[d * pp]["steps"][event]["model"]["transformer.wte.weight"]
            last = records[(d + 1) * pp - 1]["steps"][event]["model"]["lm_head.weight"]
            if not torch.equal(first, last):
                errors.append(f"DP={d}: 实际 main 更新后的共享权重不同")
        if errors:
            raise AssertionError(f"实际 main 第 {event} 轮失败：" + "; ".join(errors[:12]))
        step_reports.append({"iter": event, "loss": float(expected_loss.item()), "norm": float(expected_norm.item())})
    parsed_logs = []
    for r, record in enumerate(records):
        logs = re.findall(r"(?m)^\s*(?:iter|step)\s+(\d+):\s+loss=([^,\s]+)", record["stdout"])
        if r and logs:
            errors.append(f"非 rank 0 的进程 {r} 打印了训练 loss 日志")
        if r == 0:
            if [int(i) for i, _ in logs] != list(range(args.steps)):
                errors.append("rank 0 日志应按 --log-interval 1 输出每轮 iter/step 和 loss")
            else:
                for (i, loss), expected in zip(logs, step_reports):
                    main_close(torch, float(loss), expected["loss"], errors, f"printed loss iter={i}", "loss")
                parsed_logs = [line for line in record["stdout"].splitlines()
                               if re.match(r"\s*(?:iter|step)\s+\d+:", line)]
    if errors:
        raise AssertionError("实际 main 日志失败：" + "; ".join(errors[:12]))
    for line in parsed_logs:
        print(line, flush=True)
    return {"status": "passed", "pp_size": pp, "dp_size": dp, "dtype": "float32", "bias": bias,
            "grad_clip": args.clip_norm, "steps": step_reports, "actual_rank0_training_logs": parsed_logs,
            "launch": "spawn workers setting env:// environment; call actual train_pp.main",
            "torchrun_cli_checked": False, "group_destroyed": True}


def unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_datasets(np, args, temporary):
    directory = Path(temporary) / "data"
    directory.mkdir()
    rng = np.random.default_rng(args.seed + 777)
    data = rng.integers(0, args.vocab_size, size=max(4096, 4 * (args.seq_len + 1)), dtype=np.uint16)
    path = directory / "train.bin"
    data.tofile(path)
    minimal = directory / "minimal.bin"
    (np.arange(args.seq_len + 1, dtype=np.int64) % args.vocab_size).astype(np.uint16).tofile(minimal)
    args.dataset_path, args.minimal_path = path, minimal


def main(argv=None):
    args = parse_args(argv)
    common.configure_threads(args.threads)
    report = {"status": "running", "scope": "PP_DP_training_factory_data_loss_step_and_actual_main",
              "requested_cases": args.cases, "completed_cases": [], "actual_main": {"status": "not_run"},
              "not_checked": ["CUDA/NCCL", "AMP", "nonzero dropout equivalence", "LR schedule", "evaluation",
                              "checkpoint/resume", "real corpus", "torchrun CLI", "performance"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / training integration signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only_passed"
            common.write_report(args.report, report)
            print("[PASS] Static preflight completed; no real communication or training was run.", flush=True)
            return 0
        import numpy as np
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        torch.set_num_threads(args.threads)
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("当前 PyTorch 缺少 distributed/Gloo")
        world = args.pp_size * args.dp_size
        report.update(torch_version=torch.__version__, device="cpu", backend="gloo", seed=args.seed,
                      threads=args.threads, helper_dtype="float64", main_dtype="float32",
                      micro_batch_sizes=args.micro_batch_sizes, grad_clip_modes=[0.0, args.clip_norm],
                      steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
                      model_config={"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                                    "vocab_size": args.vocab_size, "block_size": args.block_size, "dropout": 0.0})
        print(f"[INFO] {world} CPU workers; threads={args.threads}; helper cases={args.cases}.", flush=True)
        print("[INFO] Project AdamW / sampling / global loss / train_pp_step vs dense GPT.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_training_") as temporary:
            make_datasets(np, args, temporary)
            try:
                mp.spawn(helper_worker, args=(args, temporary), nprocs=world, join=True)
            finally:
                for pp, _ in args.cases:
                    path = Path(temporary) / f"helper_pp{pp}.json"
                    if path.is_file():
                        report["completed_cases"].append(json.loads(path.read_text(encoding="utf-8")))
            if len(report["completed_cases"]) != len(args.cases):
                raise RuntimeError("辅助函数集成检查未全部完成")
            if args.skip_main:
                report["actual_main"] = {"status": "skipped_by_flag"}
                report["not_checked"].append("train_pp.main / env rendezvous")
            else:
                bias = args.bias != "false"
                print(f"[INFO] Fresh workers: actual train_pp.main; PP={args.pp_size}, DP={args.dp_size}, "
                      f"float32, {args.steps} iterations.", flush=True)
                try:
                    mp.spawn(main_worker, args=(args, temporary, unused_port(), bias), nprocs=world, join=True)
                    report["actual_main"] = check_main_records(torch, np, args, temporary, bias)
                except Exception:
                    report["actual_main"] = {"status": "failed", "captured_stdout": {
                        path.name: path.read_text(encoding="utf-8")[-12000:]
                        for path in Path(temporary).glob("main_rank*_stdout.txt")}}
                    raise
                print("[PASS] actual train_pp.main: iterations / data RNG / loss / gradients / AdamW / logging.", flush=True)
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码或验证器内容发生变化")
        report["status"] = "passed" if not args.skip_main else "helpers_passed_main_skipped"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        common.write_report(args.report, report)
        raise
    common.write_report(args.report, report)
    if args.skip_main:
        print("[PASS] PP training helper integration completed; actual main was skipped.", flush=True)
    else:
        print("[PASS] PP training integration verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
