#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 nanoGPT 的 PP+DP 评估与实际训练入口。

放在 nanoGPT 根目录，与以下已有文件同目录：
  verify_pp_training.py, verify_pp_optimizer.py,
  verify_pp_grad_sync.py, verify_pp_microbatch.py
运行：
  python verify_pp_eval.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_eval_report.json

第一阶段：CPU/Gloo、float64、小模型；默认检查相同 world_size 的
PP=1/2/4 等可用划分、bias=True/False、dropout=0/0.2。
直接调用项目 forward_eval_microbatch/evaluate_pp；覆盖前向通信、
可变 micro-batch、独立 uint16 数据采样、跨 DP/跨评估 batch 的
有效 token 加权 loss、空 micro-batch/副本/batch、全空标签报错、
模型模式恢复、无梯度记录、参数/已有梯度/非空 AdamW 状态保留、
全局 torch RNG 和数据 generator 状态。受控标签用测试采样器注入，
另有真实 get_pp_batch 数据检查；不替换评估或模型算法。

第二阶段：全新 CPU worker，用 env:// 环境真正调用 train_pp.main，
默认三轮训练、评估间隔 2，检查中途与最后一轮评估、val.bin 路径、
独立且持续推进的评估 generator、rank 0 日志，以及评估后训练参数、
梯度和 AdamW 状态与完整 GPT 的一致性。此阶段 dropout=0、float32。

自动创建临时 train.bin/val.bin，不修改项目源码，不写 checkpoint。
进程数和模型大小都是验证配置，可通过参数调整；不限制实现的规模。
--preflight-only 只做静态检查；--skip-main 明确标记 main 未验证。
未验证 CUDA/NCCL、AMP、非零 dropout 的训练数值等价性、checkpoint/
resume、LR 调度、真实语料、torchrun CLI 或性能。
"""

import argparse
import ast
from contextlib import redirect_stdout
from datetime import timedelta
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

try:
    import verify_pp_training as training
except ModuleNotFoundError as exc:
    if exc.name == "verify_pp_training":
        raise SystemExit("请把 verify_pp_training.py 与本文件放在同一目录。") from exc
    raise

common = training.common
sync_checks = training.sync_checks
opt_checks = training.opt_checks
EVAL_ARGUMENTS = {
    "forward_eval_microbatch": (
        "stage", "token_ids", "parallel", "micro_batch_size", "seq_length",
        "device", "activation_dtype",
    ),
    "evaluate_pp": (
        "stage", "parallel", "data_path", "batch_size", "seq_length",
        "micro_batch_size", "eval_iters", "device", "generator", "activation_dtype",
    ),
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
    p.add_argument("--eval-iters", type=int, default=3, help="每次评估 batch 数；本验证至少 2")
    p.add_argument("--eval-interval", type=int, default=2, help="实际 main 的评估间隔")
    p.add_argument("--dropout", type=float, default=0.2, help="辅助函数额外测试的非零 dropout")
    p.add_argument("--steps", type=int, default=3, help="实际 main 的训练轮数；本验证至少 3")
    p.add_argument("--clip-norm", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=0.0006)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--bias", choices=("true", "false", "both"), default="both")
    p.add_argument("--single-case", action="store_true")
    p.add_argument("--skip-main", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd", "vocab_size",
                 "block_size", "seq_len", "batch_size", "eval_interval", "timeout"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.eval_iters < 2 or args.steps < 3:
        p.error("本验证需要 --eval-iters 至少 2、--steps 至少 3")
    if args.pp_size > args.n_layer or args.n_embd % args.n_head:
        p.error("需要 pp_size <= n_layer，且 n_embd 能被 n_head 整除")
    if args.seq_len > args.block_size or not 2 <= args.vocab_size <= 65536:
        p.error("需要 seq_len <= block_size，词表范围为 2..65536")
    if args.batch_size * args.seq_len < 2:
        p.error("本验证需要至少两个标签位置")
    if not math.isfinite(args.dropout) or not 0 < args.dropout < 1:
        p.error("--dropout 必须满足 0 < dropout < 1；另外自动测试 dropout=0")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay 必须是非负的有限数")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in args.betas):
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
    expected = Path(__file__).resolve().parent / "verify_pp_training.py"
    if Path(training.__file__).resolve() != expected:
        raise RuntimeError(f"训练验证辅助文件应来自 {expected}")
    training.preflight(project)
    path = project / "train_pp.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for name, required in EVAL_ARGUMENTS.items():
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise AttributeError(f"train_pp.py 需要定义一个顶层 {name}()")
        supplied = {a.arg for a in nodes[0].args.args + nodes[0].args.kwonlyargs}
        if set(required) - supplied:
            raise TypeError(f"{name} 缺少可按关键字传入的参数 {sorted(set(required) - supplied)}")


def fingerprints(project):
    values = training.fingerprints(project)
    path = Path(__file__)
    values[f"checker:{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def eval_generator(torch, args, dp_rank):
    result = torch.Generator(device="cpu")
    result.manual_seed(args.seed + 20000 + dp_rank)
    return result


def module_modes(stage):
    return {name: module.training for name, module in stage.named_modules()}


def preservation_snapshot(torch, stage, optimizer):
    return {
        "model": training.clone_tree(torch, stage.state_dict()),
        "gradients": training.clone_tree(torch, sync_checks.snapshot_gradients(stage)),
        "gradient_ids": {name: id(p.grad) if p.grad is not None else None for name, p in stage.named_parameters()},
        "requires_grad": {name: p.requires_grad for name, p in stage.named_parameters()},
        "optimizer": training.clone_tree(torch, optimizer.state_dict()),
        "modes": module_modes(stage),
        "global_rng": torch.get_rng_state().clone(),
        "grad_enabled": torch.is_grad_enabled(),
    }


def preservation_errors(torch, stage, optimizer, before, errors):
    for name, actual in (
        ("model", stage.state_dict()),
        ("gradients", sync_checks.snapshot_gradients(stage)),
        ("optimizer", optimizer.state_dict()),
        ("modes", module_modes(stage)),
        ("requires_grad", {name: p.requires_grad for name, p in stage.named_parameters()}),
        ("gradient_ids", {name: id(p.grad) if p.grad is not None else None for name, p in stage.named_parameters()}),
        ("global_rng", torch.get_rng_state()),
        ("grad_enabled", torch.is_grad_enabled()),
    ):
        opt_checks.exact_tree(torch, actual, before[name], errors, f"evaluation preserves {name}")


class EvaluationAudit:
    """透明观察通信、模型前向和 optimizer.step；所有调用仍交给原函数。"""

    def __init__(self, torch, trainer, stage, optimizer, info):
        self.torch, self.trainer, self.stage, self.optimizer, self.info = torch, trainer, stage, optimizer, info
        self.errors, self.forwards, self.communications, self.training_calls = [], [], [], []
        self.saved, self.handles, self.step_calls = {}, [], 0

    def __enter__(self):
        for name in ("recv_forward", "send_forward", "recv_backward", "send_backward",
                     "sync_stage_gradients", "step_stage_optimizer", "train_pp_step"):
            original = getattr(self.trainer, name)
            signature = inspect.signature(original)

            def observed(*positional, _name=name, _original=original, _signature=signature, **keywords):
                bound = _signature.bind(*positional, **keywords)
                bound.apply_defaults()
                values = bound.arguments
                if _name in ("recv_forward", "send_forward"):
                    expected_rank = self.info["prev_rank" if _name == "recv_forward" else "next_rank"]
                    actual_rank = values["src_rank" if _name == "recv_forward" else "dst_rank"]
                    if actual_rank != expected_rank or values["pp_group"] is not self.info["pp_group"]:
                        self.errors.append(f"{_name}: 通信方向或 pp_group 不正确")
                    if self.torch.is_grad_enabled():
                        self.errors.append(f"{_name}: 评估通信发生在梯度记录开启时")
                    entry = {"name": _name}
                    if _name == "recv_forward":
                        entry["shape"] = tuple(values["shape"])
                        if values.get("requires_grad") is not False:
                            self.errors.append("评估 recv_forward 需要 requires_grad=False")
                    else:
                        tensor = values["tensor"]
                        entry["shape"] = tuple(tensor.shape)
                        if tensor.requires_grad or tensor.grad_fn is not None:
                            self.errors.append("评估发送的隐藏状态仍有 autograd 记录")
                    self.communications.append(entry)
                else:
                    self.training_calls.append(_name)
                result = _original(*positional, **keywords)
                if _name == "recv_forward" and (result.requires_grad or result.grad_fn is not None):
                    self.errors.append("评估接收的隐藏状态仍需要梯度")
                return result

            self.saved[name] = original
            setattr(self.trainer, name, observed)

        def forward_hook(module, inputs, output):
            if self.torch.is_grad_enabled() or output.requires_grad or output.grad_fn is not None:
                self.errors.append("评估模型前向没有关闭 autograd")
            if any(m.training for m in module.modules()):
                self.errors.append("评估时 stage 或其子模块仍处于训练模式")
            self.forwards.append({"input_shape": tuple(inputs[0].shape), "output_shape": tuple(output.shape),
                                  "input_dtype": str(inputs[0].dtype), "output_dtype": str(output.dtype)})

        def step_hook(optimizer, positional, keywords):
            self.step_calls += 1

        self.handles.append(self.stage.register_forward_hook(forward_hook))
        self.handles.append(self.optimizer.register_step_pre_hook(step_hook))
        return self

    def check(self, rows, seq_len, dtype, errors):
        errors.extend(self.errors)
        if self.step_calls or self.training_calls:
            errors.append(f"评估触发了训练操作：optimizer.step={self.step_calls}, calls={self.training_calls}")
        if len(self.forwards) != len(rows):
            errors.append(f"stage 前向次数应为 {len(rows)}，实际为 {len(self.forwards)}")
        for event, row in zip(self.forwards, rows):
            expected_input = (row, seq_len) if self.stage.is_first else (row, seq_len, self.stage.config.n_embd)
            expected_output = (row, seq_len, self.stage.config.vocab_size if self.stage.is_last else self.stage.config.n_embd)
            expected_input_dtype = "torch.int64" if self.stage.is_first else str(dtype)
            if (event["input_shape"], event["output_shape"], event["input_dtype"], event["output_dtype"]) != (
                expected_input, expected_output, expected_input_dtype, str(dtype)
            ):
                errors.append(f"评估 stage 前向形状/dtype 错误：{event}")
        expected_comms = []
        for row in rows:
            if not self.stage.is_first:
                expected_comms.append({"name": "recv_forward", "shape": (row, seq_len, self.stage.config.n_embd)})
            if not self.stage.is_last:
                expected_comms.append({"name": "send_forward", "shape": (row, seq_len, self.stage.config.n_embd)})
        if self.communications != expected_comms:
            errors.append("评估前向通信次数、顺序或 micro-batch 形状不正确")

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.saved.items():
            setattr(self.trainer, name, original)
        for handle in self.handles:
            handle.remove()


def reference_stats(torch, dense, batches):
    """完整 GPT + 逐 token CE；先合并 DP batch，最后跨 batch 计数。"""
    total_loss, total_valid = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            x = torch.cat([item[0] for item in batch], dim=0)
            y = torch.cat([item[1] for item in batch], dim=0)
            # 原版 GPT 在 targets=None 时只返回最后一个位置，故传入 targets。
            logits, _ = dense(x, y)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none")
            total_loss += float(losses.to(torch.float64).sum().item())
            total_valid += int((y != -1).sum().item())
    return total_loss, total_valid


def sampled_batches(torch, np, args, dp, generators):
    return [[training.oracle_batch(torch, np, args.val_path, args, generators[d]) for d in range(dp)]
            for _ in range(args.eval_iters)]


def controlled_batches(torch, args, dp, micro, pattern):
    total = args.batch_size * args.seq_len
    result = []
    for event in range(args.eval_iters):
        batch = []
        for d in range(dp):
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 30000 + 101 * event + 997 * d)
            x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), generator=generator)
            original = torch.randint(args.vocab_size, x.shape, generator=generator)
            y = torch.full_like(original, -1)
            count = min(total, 1 + (d + event * (dp + 1)) % total)
            positions = [i * total // count for i in range(count)]
            if pattern == "empty_microbatch":
                positions = list(range(min(micro * args.seq_len, total), total))
            elif pattern == "one_empty_replica" and d == 0:
                positions = []
            elif pattern == "one_empty_batch" and event == 0:
                positions = []
            elif pattern == "all_empty":
                positions = []
            elif pattern not in ("unequal_valid", "one_empty_replica", "one_empty_batch", "empty_microbatch", "all_empty"):
                raise ValueError(f"未知测试模式 {pattern}")
            if positions:
                y.reshape(-1)[positions] = original.reshape(-1)[positions]
            batch.append((x, y))
        result.append(batch)
    return result


def verify_direct_forward(torch, dist, rank, args, trainer, dense, stage, optimizer, info, micro):
    stage.eval()
    before = preservation_snapshot(torch, stage, optimizer)
    size = min(args.batch_size, micro)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 411)
    x = torch.randint(args.vocab_size, (size, args.seq_len), generator=g)
    with torch.no_grad():
        expected, _ = dense(x, torch.zeros_like(x))
    errors = []
    with EvaluationAudit(torch, trainer, stage, optimizer, info) as audit:
        actual = trainer.forward_eval_microbatch(stage=stage, token_ids=x if stage.is_first else None,
            parallel=info, micro_batch_size=size, seq_length=args.seq_len, device=torch.device("cpu"),
            activation_dtype=torch.float64)
    if stage.is_last:
        common.check_close(torch, actual, expected, "full logits from direct evaluation forward", errors)
    elif actual is not None:
        errors.append("非最后 stage 的 forward_eval_microbatch 应返回 None")
    audit.check([size], args.seq_len, torch.float64, errors)
    preservation_errors(torch, stage, optimizer, before, errors)
    sync_checks.agree(torch, dist, rank, errors, "direct evaluation forward", quiet=True)


def verify_eval_trial(torch, dist, rank, args, trainer, dense, stage, optimizer, info, batches,
                      micro, generator, pattern, was_training, expected_generator_state=None):
    stage.train(was_training)
    before = preservation_snapshot(torch, stage, optimizer)
    generator_before = generator.get_state().clone()
    loss_sum, valid = reference_stats(torch, dense, batches)
    expected = loss_sum / valid if valid else None
    original_data = trainer.get_pp_batch
    signature = inspect.signature(original_data)
    calls, errors = [], []
    injected_error = pattern == "injected_sampler_error"
    sampled = pattern == "sampled_uint16"

    def observed_data(*positional, **keywords):
        bound = signature.bind(*positional, **keywords)
        values = bound.arguments
        event = len(calls)
        calls.append(event)
        if (values["stage"] is not stage or values["parallel"] is not info
            or values["generator"] is not generator or Path(values["data_path"]).resolve() != args.val_path
            or values["batch_size"] != args.batch_size or values["seq_length"] != args.seq_len
            or values["device"] != torch.device("cpu")):
            errors.append("evaluate_pp 向 get_pp_batch 传入了错误的采样参数")
        if injected_error and event == 1:
            raise RuntimeError("PP_EVAL_EXPECTED_SAMPLER_ERROR")
        if event >= args.eval_iters:
            raise AssertionError("evaluate_pp 超过指定 eval_iters")
        x, y = batches[event][info["dp_rank"]]
        if sampled:
            result = original_data(*positional, **keywords)
            training.check_batch(torch, stage, result[0], result[1], x, y, errors)
            return result
        # 受控标签 fixture；评估模型、前向通信、DP 汇总均仍由项目实现。
        return (x.clone() if stage.is_first else None, y.clone() if stage.is_last else None)

    trainer.get_pp_batch = observed_data
    actual, caught = None, None
    try:
        with EvaluationAudit(torch, trainer, stage, optimizer, info) as audit:
            try:
                actual = trainer.evaluate_pp(stage=stage, parallel=info, data_path=str(args.val_path),
                    batch_size=args.batch_size, seq_length=args.seq_len, micro_batch_size=micro,
                    eval_iters=args.eval_iters, device=torch.device("cpu"), generator=generator,
                    activation_dtype=torch.float64)
            except Exception as exc:
                if injected_error and isinstance(exc, RuntimeError) and "PP_EVAL_EXPECTED_SAMPLER_ERROR" in str(exc):
                    caught = "sampler_error"
                elif valid == 0 and isinstance(exc, ValueError):
                    caught = "all_empty_value_error"
                else:
                    raise
    finally:
        trainer.get_pp_batch = original_data
    if injected_error:
        if caught != "sampler_error":
            errors.append("采样异常应传播给调用方")
    elif valid == 0:
        if caught != "all_empty_value_error":
            errors.append("全局全部标签被忽略时，应在汇总后抛出 ValueError")
    else:
        if type(actual) is not float:
            errors.append("evaluate_pp 应向每个 rank 返回 Python float")
        opt_checks.check_scalar(torch, actual, expected, errors, "token-weighted global evaluation loss")
    expected_calls = 2 if injected_error else args.eval_iters
    if len(calls) != expected_calls:
        errors.append(f"get_pp_batch 调用次数应为 {expected_calls}，实际为 {len(calls)}")
    completed_batches = 1 if injected_error else args.eval_iters
    rows = [min(micro, args.batch_size - start) for _ in range(completed_batches)
            for start in range(0, args.batch_size, micro)]
    audit.check(rows, args.seq_len, torch.float64, errors)
    preservation_errors(torch, stage, optimizer, before, errors)
    wanted_state = expected_generator_state if sampled else generator_before
    opt_checks.exact_tree(torch, generator.get_state(), wanted_state, errors, "evaluation data generator")
    sync_checks.agree(torch, dist, rank, errors, f"evaluation {pattern}, micro={micro}", quiet=True)
    return {"pattern": pattern, "micro_batch_size": micro, "was_training": was_training,
            "expected_loss": expected, "valid_tokens": valid, "expected_exception": caught,
            "data_calls": expected_calls, "status": "passed"}


def verify_model(torch, np, dist, rank, args, trainer, GPT, GPTConfig, GPTStage, info, pp, dp, bias, dropout):
    torch.manual_seed(args.seed)
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        vocab_size=args.vocab_size, block_size=args.block_size, dropout=dropout, bias=bias)
    with redirect_stdout(io.StringIO()):
        dense = GPT(config).to(dtype=torch.float64).eval()
        stage = GPTStage(config, pp_size=pp, pp_rank=info["pp_rank"]).to(dtype=torch.float64)
        stage.load_dense_state_dict(dense.state_dict())
    optimizer = training.make_project_optimizer(trainer, stage, args)
    errors = training.optimizer_errors(torch, stage, optimizer, args)
    reference_optimizer = opt_checks.make_optimizer(torch, dense, args)
    # 一次零梯度 AdamW 更新：构建真实非空 moments；共享两端仍保持相同。
    for model, opt in ((dense, reference_optimizer), (stage, optimizer)):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        opt.step()
        opt.zero_grad(set_to_none=True)
    for name, value in stage.state_dict().items():
        common.check_close(torch, value, dense.state_dict()[name], f"prepared model {name}", errors)
    for index, (_, parameter) in enumerate(stage.named_parameters()):
        parameter.grad = None if index % 4 == 0 else torch.full_like(parameter, 0.03125 * (index + 1))
    if not optimizer.state:
        errors.append("验证准备阶段未生成 AdamW 状态")
    sync_checks.agree(torch, dist, rank, errors, "prepared model / optimizer / sentinel gradients", quiet=True)
    records = []
    actual_generator = eval_generator(torch, args, info["dp_rank"])
    oracle_generators = [eval_generator(torch, args, d) for d in range(dp)]
    for index, micro in enumerate(args.micro_batch_sizes):
        verify_direct_forward(torch, dist, rank, args, trainer, dense, stage, optimizer, info, micro)
        batches = sampled_batches(torch, np, args, dp, oracle_generators)
        wanted = (oracle_generators[info["dp_rank"]].get_state().clone() if stage.is_first
                  else actual_generator.get_state().clone())
        records.append(verify_eval_trial(torch, dist, rank, args, trainer, dense, stage, optimizer, info,
            batches, micro, actual_generator, "sampled_uint16", index % 2 == 0, wanted))
    micro = next((m for m in args.micro_batch_sizes if 1 < m < args.batch_size), args.micro_batch_sizes[0])
    patterns = ["unequal_valid", "one_empty_batch", "all_empty", "injected_sampler_error"]
    if micro < args.batch_size:
        patterns.insert(1, "empty_microbatch")
    if dp > 1:
        patterns.insert(1, "one_empty_replica")
    for index, pattern in enumerate(patterns):
        batches = controlled_batches(torch, args, dp, micro,
                                     "unequal_valid" if pattern == "injected_sampler_error" else pattern)
        records.append(verify_eval_trial(torch, dist, rank, args, trainer, dense, stage, optimizer, info,
            batches, micro, eval_generator(torch, args, info["dp_rank"]), pattern, index % 2 == 0))
    sync_checks.agree(torch, dist, rank, [],
        f"PP={pp}, DP={dp}, bias={bias}, dropout={dropout}: logits / weighted eval / modes / state / RNG")
    return {"status": "passed", "bias": bias, "dropout": dropout, "trials": records}


def helper_worker(rank, args, temporary):
    common.configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间源码或验证器发生变化")
    trainer, GPT, GPTConfig, GPTStage = common.load_project(args.project)
    world = args.pp_size * args.dp_size
    store = dist.FileStore(str(Path(temporary) / "eval_helper_rendezvous"), world)
    store.set_timeout(timedelta(seconds=args.timeout))
    originals = training.bound_timeouts(dist, args.timeout)
    try:
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
        for pp, dp in args.cases:
            if rank == 0:
                print(f"[CASE] PP={pp}, DP={dp}", flush=True)
            info = trainer.setup_parallel(pp)
            sync_checks.agree(torch, dist, rank, common.group_errors(dist, info, rank, pp, dp),
                              f"PP={pp}, DP={dp}: groups")
            record = {"status": "running", "pp_size": pp, "dp_size": dp, "models": []}
            for bias in args.biases:
                for dropout in (0.0, args.dropout):
                    record["models"].append(verify_model(torch, np, dist, rank, args, trainer, GPT,
                        GPTConfig, GPTStage, info, pp, dp, bias, dropout))
                    if rank == 0:
                        (Path(temporary) / f"eval_pp{pp}.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            record["status"] = "passed"
            if rank == 0:
                (Path(temporary) / f"eval_pp{pp}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        training.restore_timeouts(dist, originals)


def main_argv(args, bias):
    return training.main_argv(args, bias) + ["--eval-interval", str(args.eval_interval),
                                           "--eval-iters", str(args.eval_iters)]


def expected_eval_iterations(args):
    return [i for i in range(args.steps) if (i + 1) % args.eval_interval == 0 or i + 1 == args.steps]


def main_worker(rank, args, temporary, port, bias):
    common.configure_threads(args.threads)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.pp_size * args.dp_size), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("main 开始前源码或验证器发生变化")
    trainer, _, _, _ = common.load_project(args.project)
    names = ("build_stage_optimizer", "get_pp_batch", "train_pp_step", "evaluate_pp")
    originals = {name: getattr(trainer, name) for name in names}
    signatures = {name: inspect.signature(value) for name, value in originals.items()}
    record = {"rank": rank, "steps": [], "data_calls": [], "evaluations": [], "factory_calls": 0}
    captured, handles = {"current_eval": None}, []

    def observed_factory(*positional, **keywords):
        bound = signatures["build_stage_optimizer"].bind(*positional, **keywords)
        stage = bound.arguments["stage"]
        optimizer = originals["build_stage_optimizer"](*positional, **keywords)
        errors = training.optimizer_errors(torch, stage, optimizer, args)
        if errors:
            raise AssertionError("main optimizer: " + "; ".join(errors))
        captured.update(stage=stage, optimizer=optimizer)
        audit, handle = training.attach_audit(torch, stage, optimizer)
        captured["audit"] = audit
        handles.append(handle)
        record["factory_calls"] += 1
        record["optimizer_groups"] = training.group_metadata(stage, optimizer)
        return optimizer

    def observed_data(*positional, **keywords):
        bound = signatures["get_pp_batch"].bind(*positional, **keywords)
        values = bound.arguments
        generator = values["generator"]
        purpose = "eval" if captured["current_eval"] is not None else "train"
        key = f"{purpose}_generator"
        previous = captured.setdefault(key, generator)
        if previous is not generator:
            raise AssertionError(f"main 的 {purpose} generator 在循环内重新创建或被替换")
        if purpose == "eval" and generator is captured.get("train_generator"):
            raise AssertionError("训练与评估共用了同一个数据 generator")
        call = {"seed": generator.initial_seed(), "generator_before": generator.get_state().clone(),
                "global_rng_before": torch.get_rng_state().clone(),
                "data_path": str(Path(values["data_path"]).resolve())}
        result = originals["get_pp_batch"](*positional, **keywords)
        call.update(generator_after=generator.get_state().clone(), global_rng_after=torch.get_rng_state().clone(),
                    token_ids=training.clone_tree(torch, result[0]), targets=training.clone_tree(torch, result[1]))
        destination = captured["current_eval"]["data_calls"] if purpose == "eval" else record["data_calls"]
        destination.append(call)
        return result

    def observed_step(*positional, **keywords):
        bound = signatures["train_pp_step"].bind(*positional, **keywords)
        bound.apply_defaults()
        values = bound.arguments
        stage, optimizer = values["stage"], values["optimizer"]
        if stage is not captured.get("stage") or optimizer is not captured.get("optimizer"):
            raise AssertionError("main 未使用生产工厂创建的 stage/optimizer")
        if len(record["steps"]) >= args.steps:
            raise AssertionError("main 超过指定训练轮数")
        if not record["steps"]:
            record["initial_model"] = training.clone_tree(torch, stage.state_dict())
            record["model_dtype"] = str(next(stage.parameters()).dtype)
            record["config"] = {key: getattr(stage.config, key) for key in
                                 ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "dropout", "bias")}
        audit = captured["audit"]
        audit["gradients"] = None
        event = {"token_ids": training.clone_tree(torch, values["token_ids"]),
                 "targets": training.clone_tree(torch, values["targets"]),
                 "calls_before": audit["calls"], "training": stage.training,
                 "batch_size": values["batch_size"], "seq_length": values["seq_length"],
                 "micro_batch_size": values["micro_batch_size"], "grad_clip": values["grad_clip"],
                 "activation_dtype": str(values["activation_dtype"])}
        result = originals["train_pp_step"](*positional, **keywords)
        event.update(result=training.clone_tree(torch, result), calls_after=audit["calls"],
                     gradients=training.clone_tree(torch, audit["gradients"]),
                     model=training.clone_tree(torch, stage.state_dict()),
                     optimizer=training.optimizer_named_state(torch, stage, optimizer),
                     gradients_cleared=all(p.grad is None for p in stage.parameters()))
        record["steps"].append(event)
        return result

    def observed_eval(*positional, **keywords):
        bound = signatures["evaluate_pp"].bind(*positional, **keywords)
        bound.apply_defaults()
        values = bound.arguments
        stage, generator = values["stage"], values["generator"]
        optimizer = captured["optimizer"]
        if stage is not captured["stage"]:
            raise AssertionError("main 评估了另一个 stage")
        if (Path(values["data_path"]).resolve() != args.val_path
            or values["batch_size"] != args.batch_size or values["seq_length"] != args.seq_len
            or values["micro_batch_size"] != 2 or values["eval_iters"] != args.eval_iters
            or values["device"] != torch.device("cpu") or values["activation_dtype"] != torch.float32):
            raise AssertionError("main 未正确传入评估路径或参数")
        before = preservation_snapshot(torch, stage, optimizer)
        train_generator = captured.get("train_generator")
        if train_generator is None or train_generator is generator:
            raise AssertionError("评估应位于训练更新之后，并使用独立 generator")
        train_state = train_generator.get_state().clone()
        if captured["current_eval"] is not None:
            raise AssertionError("evaluate_pp 出现递归调用")
        event = {"after_iter": len(record["steps"]) - 1, "data_calls": [],
                 "model": before["model"], "seed": generator.initial_seed(),
                 "generator_before": generator.get_state().clone()}
        captured["current_eval"] = event
        try:
            with EvaluationAudit(torch, trainer, stage, optimizer, values["parallel"]) as audit:
                result = originals["evaluate_pp"](*positional, **keywords)
        finally:
            captured["current_eval"] = None
        errors = []
        rows = [min(2, args.batch_size - start) for _ in range(args.eval_iters)
                for start in range(0, args.batch_size, 2)]
        audit.check(rows, args.seq_len, torch.float32, errors)
        preservation_errors(torch, stage, optimizer, before, errors)
        opt_checks.exact_tree(torch, train_generator.get_state(), train_state, errors,
                              "main evaluation preserves train data generator")
        if type(result) is not float or not math.isfinite(result):
            errors.append("main 评估应返回有限 Python float")
        if len(event["data_calls"]) != args.eval_iters:
            errors.append("main 评估采样次数与 eval_iters 不同")
        if errors:
            raise AssertionError("main evaluation: " + "; ".join(errors[:12]))
        event.update(loss=result, generator_after=generator.get_state().clone())
        record["evaluations"].append(event)
        return result

    trainer.build_stage_optimizer, trainer.get_pp_batch, trainer.train_pp_step, trainer.evaluate_pp = (
        observed_factory, observed_data, observed_step, observed_eval)
    timeout_originals = training.bound_timeouts(dist, args.timeout)
    output, previous_argv = io.StringIO(), sys.argv
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
        expected_iters = expected_eval_iterations(args)
        if [event["after_iter"] for event in record["evaluations"]] != expected_iters:
            raise AssertionError(f"main 评估应在训练迭代 {expected_iters} 结束后执行")
        if record["factory_calls"] != 1 or len(record["steps"]) != args.steps or len(record["data_calls"]) != args.steps:
            raise AssertionError("main 的训练循环或优化器创建次数错误")
        if dist.is_initialized():
            raise AssertionError("main 返回时未销毁默认进程组")
        if fingerprints(args.project) != args.expected_hashes:
            raise AssertionError("main 修改了项目或验证源码")
        record.update(status="passed", stdout=output.getvalue())
        torch.save(record, Path(temporary) / f"main_rank{rank}.pt")
    finally:
        (Path(temporary) / f"main_rank{rank}_stdout.txt").write_text(output.getvalue(), encoding="utf-8")
        sys.argv = previous_argv
        for name, original in originals.items():
            setattr(trainer, name, original)
        for handle in handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
        training.restore_timeouts(dist, timeout_originals)


def check_main_records(torch, np, args, temporary, bias):
    # 已有完整 GPT 参考：证明插入评估后的训练数据/更新/moments 仍正确。
    training_report = training.check_main_records(torch, np, args, temporary, bias)
    world, pp, dp = args.pp_size * args.dp_size, args.pp_size, args.dp_size
    records = [torch.load(Path(temporary) / f"main_rank{r}.pt", map_location="cpu", weights_only=True)
               for r in range(world)]
    _, GPT, GPTConfig, _ = common.load_project(args.project)
    with redirect_stdout(io.StringIO()):
        dense = GPT(training.model_config(GPTConfig, args, bias)).to(dtype=torch.float32).eval()
    generators = [eval_generator(torch, args, d) for d in range(dp)]
    errors, evaluations = [], []
    for index, after_iter in enumerate(expected_eval_iterations(args)):
        initial = {}
        for p in range(pp):
            initial.update(records[p]["evaluations"][index]["model"])
        if not torch.equal(initial["transformer.wte.weight"], initial["lm_head.weight"]):
            errors.append("main 评估时首尾共享权重不同")
        dense.load_state_dict(initial, strict=True)
        batches = []
        for event in range(args.eval_iters):
            batch = []
            for d in range(dp):
                expected_before = generators[d].get_state().clone()
                x, y = training.oracle_batch(torch, np, args.val_path, args, generators[d])
                expected_after = generators[d].get_state().clone()
                batch.append((x, y))
                for p in range(pp):
                    r = d * pp + p
                    evaluation = records[r]["evaluations"][index]
                    call = evaluation["data_calls"][event]
                    flags = argparse.Namespace(is_first=p == 0, is_last=p == pp - 1)
                    training.check_batch(torch, flags, call["token_ids"], call["targets"], x, y, errors)
                    if call["data_path"] != str(args.val_path):
                        errors.append(f"rank {r}: 评估未使用 val.bin")
                    if p == 0:
                        if call["seed"] != args.seed + 20000 + d:
                            errors.append(f"rank {r}: 评估 generator seed 未按 dp_rank 设置")
                        wanted_before, wanted_after = expected_before, expected_after
                    else:
                        wanted_before = wanted_after = records[r]["evaluations"][0]["generator_before"]
                    for field, wanted in (("generator_before", wanted_before), ("generator_after", wanted_after)):
                        opt_checks.exact_tree(torch, call[field], wanted, errors, f"rank {r}: eval {field}")
                    opt_checks.exact_tree(torch, call["global_rng_after"], call["global_rng_before"],
                                          errors, f"rank {r}: val sampler global RNG")
            batches.append(batch)
        loss_sum, count = reference_stats(torch, dense, batches)
        expected_loss = loss_sum / count
        for r in range(world):
            evaluation = records[r]["evaluations"][index]
            if evaluation["after_iter"] != after_iter:
                errors.append(f"rank {r}: 评估轮次不正确")
            training.main_close(torch, evaluation["loss"], expected_loss, errors, f"rank {r}: val loss", "loss")
            opt_checks.exact_tree(torch, evaluation["model"], records[r]["steps"][after_iter]["model"],
                                  errors, f"rank {r}: evaluation uses updated training model")
            opt_checks.exact_tree(torch, evaluation["model"], records[r % pp]["evaluations"][index]["model"],
                                  errors, f"rank {r}: evaluation DP replica model")
            first_call, last_call = evaluation["data_calls"][0], evaluation["data_calls"][-1]
            opt_checks.exact_tree(torch, evaluation["generator_before"], first_call["generator_before"],
                                  errors, f"rank {r}: evaluation starting generator state")
            opt_checks.exact_tree(torch, evaluation["generator_after"], last_call["generator_after"],
                                  errors, f"rank {r}: evaluation ending generator state")
        if errors:
            raise AssertionError("main 评估数值或数据流失败：" + "; ".join(errors[:12]))
        evaluations.append({"after_iter": after_iter, "loss": expected_loss, "valid_tokens": count})
    parsed_logs = []
    for r, record in enumerate(records):
        logs = re.findall(r"(?m)^\s*eval after iter (\d+):\s*val_loss=([^,\s]+)", record["stdout"])
        if r and logs:
            errors.append(f"非 rank 0 的进程 {r} 打印了评估 loss")
        if r == 0:
            if [int(i) for i, _ in logs] != expected_eval_iterations(args):
                errors.append("rank 0 评估日志轮次缺失或错误")
            else:
                for (i, value), expected in zip(logs, evaluations):
                    training.main_close(torch, float(value), expected["loss"], errors,
                                        f"printed val loss after iter={i}", "loss")
                parsed_logs = [line for line in record["stdout"].splitlines()
                               if re.match(r"\s*eval after iter \d+:", line)]
    if errors:
        raise AssertionError("main 评估日志失败：" + "; ".join(errors[:12]))
    for line in parsed_logs:
        print(line, flush=True)
    return {"status": "passed", "training": training_report, "evaluations": evaluations,
            "actual_rank0_eval_logs": parsed_logs, "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters, "eval_and_train_generators_independent": True,
            "parameters_gradients_optimizer_and_rng_preserved": True}


def make_datasets(np, args, temporary):
    training.make_datasets(np, args, temporary)
    rng = np.random.default_rng(args.seed + 888)
    data = rng.integers(0, args.vocab_size, size=max(4096, 4 * (args.seq_len + 1)), dtype=np.uint16)
    args.val_path = Path(args.dataset_path).parent / "val.bin"
    data.tofile(args.val_path)


def main(argv=None):
    args = parse_args(argv)
    common.configure_threads(args.threads)
    report = {"status": "running", "scope": "PP_DP_evaluation_helpers_and_actual_training_main",
              "requested_cases": args.cases, "completed_cases": [], "actual_main": {"status": "not_run"},
              "not_checked": ["CUDA/NCCL", "AMP", "nonzero-dropout training equivalence", "checkpoint/resume",
                              "LR schedule", "real corpus", "torchrun CLI", "performance"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / evaluation signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only_passed"
            common.write_report(args.report, report)
            print("[PASS] Static preflight completed; distributed evaluation was not run.", flush=True)
            return 0
        import numpy as np
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        torch.set_num_threads(args.threads)
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("当前 PyTorch 缺少 distributed/Gloo")
        world = args.pp_size * args.dp_size
        report.update(torch_version=torch.__version__, backend="gloo", device="cpu", threads=args.threads,
                      helper_dtype="float64", main_dtype="float32", helper_dropouts=[0.0, args.dropout],
                      main_dropout=0.0, micro_batch_sizes=args.micro_batch_sizes,
                      batch_size=args.batch_size, seq_len=args.seq_len, eval_iters=args.eval_iters,
                      eval_interval=args.eval_interval, steps=args.steps)
        print(f"[INFO] {world} CPU workers; threads={args.threads}; helper cases={args.cases}.", flush=True)
        print("[INFO] Weighted eval / dropout / mode restoration / preserved gradients, AdamW and RNG.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_eval_") as temporary:
            make_datasets(np, args, temporary)
            try:
                mp.spawn(helper_worker, args=(args, temporary), nprocs=world, join=True)
            finally:
                for pp, _ in args.cases:
                    path = Path(temporary) / f"eval_pp{pp}.json"
                    if path.is_file():
                        report["completed_cases"].append(json.loads(path.read_text(encoding="utf-8")))
            if len(report["completed_cases"]) != len(args.cases) or any(
                case["status"] != "passed" for case in report["completed_cases"]
            ):
                raise RuntimeError("评估辅助函数检查未全部完成")
            if args.skip_main:
                report["actual_main"] = {"status": "skipped_by_flag"}
                report["not_checked"].append("actual train_pp.main evaluation integration")
            else:
                bias = args.bias != "false"
                print(f"[INFO] Fresh workers: actual train_pp.main; PP={args.pp_size}, DP={args.dp_size}; "
                      f"eval after iterations {expected_eval_iterations(args)}.", flush=True)
                try:
                    mp.spawn(main_worker, args=(args, temporary, training.unused_port(), bias), nprocs=world, join=True)
                    report["actual_main"] = check_main_records(torch, np, args, temporary, bias)
                except Exception:
                    report["actual_main"] = {"status": "failed", "captured_stdout": {
                        path.name: path.read_text(encoding="utf-8")[-12000:]
                        for path in Path(temporary).glob("main_rank*_stdout.txt")}}
                    raise
                print("[PASS] actual main: evaluation cadence / validation RNG / weighted loss / training recovery.", flush=True)
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码或验证器发生变化")
        report["status"] = "helpers_passed_main_skipped" if args.skip_main else "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        common.write_report(args.report, report)
        raise
    common.write_report(args.report, report)
    if args.skip_main:
        print("[PASS] PP evaluation helper verification completed; actual main was skipped.", flush=True)
    else:
        print("[PASS] PP evaluation verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
