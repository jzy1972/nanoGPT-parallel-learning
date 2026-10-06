#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 nanoGPT PP+DP checkpoint 保存、恢复与实际 main 接入。

与以下已有验证文件一起放在 nanoGPT 根目录：
  verify_pp_training.py, verify_pp_optimizer.py,
  verify_pp_grad_sync.py, verify_pp_microbatch.py
运行：
  python verify_pp_checkpoint.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_checkpoint_report.json

直接调用项目 checkpoint_pp.save_pp_checkpoint/load_pp_checkpoint；
CPU/Gloo、float64、小模型，默认覆盖同一 world_size 下的 PP=1/2/4
等合法划分、bias=True/False、dropout=0/0.2、非整除 micro-batch。
检查分片布局、manifest、模型/AdamW/RNG roundtrip、Parameter 身份和
共享参数、两个 generator、保存不改变运行状态，以及连续训练与恢复
训练的逐步精确一致性。dropout=0 的对照训练另与完整 GPT 比较。
每个拓扑还检查重名保存、未清梯度、缺失/损坏文件、配置/参数顺序
不一致、单 rank 写失败时的全局错误共识及未完成目录拒绝加载。

随后用三批全新 worker 真正调用 train_pp.main：连续运行、在成功保存
的边界停止、从该 checkpoint 恢复。观察器在保存成功后标记停止，
等当前迭代的日志等收尾完成，在下一批训练数据采样前抛出测试异常；
不消费下一批训练数据，不改训练/评估/保存/恢复算法。比较每轮采样、
loss、梯度、参数、AdamW、评估、RNG、迭代日志和最终 checkpoint。
三次启动保持同一 max_iters，避免分段运行额外触发“最后一次评估”。
此阶段使用 float32 和非零 dropout，验证同一 PP+DP 程序的恢复等价。

自动创建并清理临时 uint16 数据和 checkpoint，不修改项目源码，也
不读取你的真实训练语料。默认四个 CPU worker 仅是验证配置；可调整
--pp-size/--dp-size，不限制实现支持的规模。导入 torch/NumPy 前限制
BLAS/OpenMP 线程，Windows rendezvous 设 USE_LIBUV=0。
--preflight-only 仅静态检查；--skip-main 在报告中明确标记入口未验证。
未验证 CUDA/NCCL、AMP、不同 PP/DP 规模恢复、LR 调度、torchrun CLI、
磁盘断电一致性、真实语料或性能。数值检查必须在安装 PyTorch 后运行。
"""

import argparse
import ast
from contextlib import redirect_stdout
import copy
from datetime import timedelta
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
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
CHECKPOINT_ARGUMENTS = {
    "save_pp_checkpoint": (
        "stage", "optimizer", "parallel", "checkpoint_root", "completed_iters",
        "train_data_generator", "eval_data_generator", "training_config", "device",
    ),
    "load_pp_checkpoint": (
        "stage", "optimizer", "parallel", "checkpoint_dir", "train_data_generator",
        "eval_data_generator", "training_config", "device",
    ),
}
EVALUATION_ARGUMENTS = {
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
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=2, help="保存前的训练次数，至少 2")
    p.add_argument("--resume-steps", type=int, default=2, help="保存后的训练次数，至少 2")
    p.add_argument("--eval-interval", type=int, default=2)
    p.add_argument("--eval-iters", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2, help="非零 dropout；另自动测试 dropout=0")
    p.add_argument("--clip-norm", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=0.0006)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--bias", choices=("true", "false", "both"), default="both")
    p.add_argument("--single-case", action="store_true", help="辅助函数只检查指定的 PP/DP")
    p.add_argument("--skip-main", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd", "vocab_size",
                 "block_size", "seq_len", "batch_size", "micro_batch_size", "eval_interval", "eval_iters", "timeout"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.warmup_steps < 2 or args.resume_steps < 2:
        p.error("本验证需要 --warmup-steps 和 --resume-steps 至少为 2")
    if args.pp_size > args.n_layer or args.n_embd % args.n_head:
        p.error("需要 pp_size <= n_layer，且 n_embd 能被 n_head 整除")
    if args.seq_len > args.block_size or not 2 <= args.vocab_size <= 65536:
        p.error("需要 seq_len <= block_size，uint16 测试的词表范围为 2..65536")
    if not math.isfinite(args.dropout) or not 0 < args.dropout < 1:
        p.error("--dropout 必须满足 0 < dropout < 1")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay 必须是非负的有限数")
    if any(not math.isfinite(b) or not 0 <= b < 1 for b in args.betas):
        p.error("每个 beta 必须满足 0 <= beta < 1")
    world = args.pp_size * args.dp_size
    args.cases = ([(args.pp_size, args.dp_size)] if args.single_case else
                  [(pp, world // pp) for pp in range(1, min(args.n_layer, world) + 1) if world % pp == 0])
    args.biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    args.steps = args.warmup_steps + args.resume_steps
    return args


def check_functions(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    compile(tree, str(path), "exec")
    for name, arguments in required.items():
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise AttributeError(f"{path.name} 需要定义一个顶层 {name}()")
        supplied = {a.arg for a in nodes[0].args.args + nodes[0].args.kwonlyargs}
        missing = set(arguments) - supplied
        if missing:
            raise TypeError(f"{name} 缺少可按关键字传入的参数 {sorted(missing)}")


def preflight(project):
    if Path(training.__file__).resolve() != Path(__file__).resolve().parent / "verify_pp_training.py":
        raise RuntimeError("verify_pp_training.py 必须来自本验证文件所在目录")
    training.preflight(project)
    check_functions(project / "checkpoint_pp.py", CHECKPOINT_ARGUMENTS)
    check_functions(project / "train_pp.py", EVALUATION_ARGUMENTS)
    check_functions(project / "parallel_utils.py", {"sync_tied_weights": ("stage", "tied_group", "src_rank")})


def fingerprints(project):
    result = training.fingerprints(project)
    for label, path in (("checkpoint_pp.py", project / "checkpoint_pp.py"),
                        (f"checker:{Path(__file__).name}", Path(__file__))):
        result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_project(project):
    trainer, GPT, GPTConfig, GPTStage = common.load_project(project)
    checkpoint = importlib.import_module("checkpoint_pp")
    if Path(checkpoint.__file__).resolve() != project / "checkpoint_pp.py":
        raise RuntimeError("导入了其他目录的 checkpoint_pp.py")
    for name, required in CHECKPOINT_ARGUMENTS.items():
        function = getattr(checkpoint, name, None)
        if not callable(function):
            raise AttributeError(f"checkpoint_pp.py 缺少 {name}")
        parameters = inspect.signature(function).parameters
        if set(required) - set(parameters):
            raise TypeError(f"{name} 的运行时签名不匹配")
    return trainer, checkpoint, GPT, GPTConfig, GPTStage


def optimizer_names(stage, optimizer):
    names = {id(parameter): name for name, parameter in stage.named_parameters()}
    return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]


def parameter_ids(stage, optimizer):
    return {
        "model": {name: id(p) for name, p in stage.named_parameters(remove_duplicate=False)},
        "optimizer": [[id(p) for p in group["params"]] for group in optimizer.param_groups],
    }


def state_bundle(torch, stage, optimizer, train_generator, eval_generator):
    return {
        "model": training.clone_tree(torch, stage.state_dict()),
        "optimizer": training.clone_tree(torch, optimizer.state_dict()),
        "optimizer_names": optimizer_names(stage, optimizer),
        "torch_cpu": torch.get_rng_state().clone(),
        "train_generator": train_generator.get_state().clone(),
        "eval_generator": eval_generator.get_state().clone(),
        "gradients": training.clone_tree(torch, sync_checks.snapshot_gradients(stage)),
        "modes": {name: module.training for name, module in stage.named_modules()},
    }


def identity_errors(stage, optimizer, before):
    errors = []
    if parameter_ids(stage, optimizer) != before:
        errors.append("恢复替换了 Parameter 对象或优化器的参数引用（检查 assign=False）")
    if stage.pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 恢复后 embedding/head 不再共享同一个 Parameter")
    owned = {id(p) for p in stage.parameters() if p.requires_grad}
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if set(actual) != owned or len(actual) != len(set(actual)):
        errors.append("恢复后优化器参数覆盖或共享参数去重错误")
    errors.extend(opt_checks.cleared_gradient_errors(stage))
    return errors


def run_settings(args):
    return {
        "batch_size": args.batch_size, "seq_length": args.seq_len,
        "micro_batch_size": args.micro_batch_size, "grad_clip": args.clip_norm,
        "eval_interval": args.eval_interval, "eval_iters": args.eval_iters,
    }


def generators(torch, args, dp_rank):
    train = training.data_generator(torch, args, dp_rank)
    evaluation = torch.Generator(device="cpu")
    evaluation.manual_seed(args.seed + 20000 + dp_rank)
    return train, evaluation


def make_stage(torch, trainer, GPTConfig, GPTStage, args, info, pp, bias, dropout, salt=0):
    torch.manual_seed(args.seed + salt + info["pp_rank"])
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size, dropout=dropout, bias=bias)
    with redirect_stdout(io.StringIO()):
        stage = GPTStage(config, pp_size=pp, pp_rank=info["pp_rank"]).to(dtype=torch.float64).train()
    utils = importlib.import_module("parallel_utils")
    utils.sync_tied_weights(stage=stage, tied_group=info["tied_group"], src_rank=info["pp_first_rank"])
    optimizer = training.make_project_optimizer(trainer, stage, args)
    optimizer.zero_grad(set_to_none=True)
    return stage, optimizer


def save_arguments(torch, stage, optimizer, info, root, completed, train, evaluation, args):
    return dict(stage=stage, optimizer=optimizer, parallel=info, checkpoint_root=root,
                completed_iters=completed, train_data_generator=train, eval_data_generator=evaluation,
                training_config=run_settings(args), device=torch.device("cpu"))


def load_arguments(torch, stage, optimizer, info, directory, train, evaluation, args):
    return dict(stage=stage, optimizer=optimizer, parallel=info, checkpoint_dir=directory,
                train_data_generator=train, eval_data_generator=evaluation,
                training_config=run_settings(args), device=torch.device("cpu"))


def load_file(torch, path):
    return torch.load(path, map_location="cpu", weights_only=True)


def inspect_layout(torch, args, stage, info, directory, boundary, completed, errors):
    expected_stage_files = [f"stage_{p:04d}.pt" for p in range(stage.pp_size)]
    expected_rng_files = [f"rng_rank_{r:05d}.pt" for r in range(info["world_size"])]
    expected_files = set(expected_stage_files + expected_rng_files + ["manifest.json"])
    actual_files = {path.name for path in directory.iterdir()}
    if actual_files != expected_files:
        errors.append(f"checkpoint 文件布局不一致：{sorted(actual_files ^ expected_files)}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "format_version": 1, "completed_iters": completed, "world_size": info["world_size"],
        "pp_size": stage.pp_size, "dp_size": info["dp_size"], "device_type": "cpu",
        "parameter_dtype": str(next(stage.parameters()).dtype), "model_config": vars(stage.config).copy(),
        "training_config": run_settings(args), "stage_files": expected_stage_files, "rng_files": expected_rng_files,
    }
    opt_checks.exact_tree(torch, manifest, expected_manifest, errors, "manifest")
    payload = load_file(torch, directory / expected_stage_files[info["pp_rank"]])
    expected_payload = {
        "format_version": 1, "completed_iters": completed, "pp_rank": info["pp_rank"],
        "layer_range": [stage.layer_start, stage.layer_end], "model": boundary["model"],
        "optimizer": boundary["optimizer"], "optimizer_parameter_names": boundary["optimizer_names"],
    }
    opt_checks.exact_tree(torch, payload, expected_payload, errors, "saved stage payload")
    rng = load_file(torch, directory / expected_rng_files[info["world_rank"]])
    expected_rng = {
        "format_version": 1, "completed_iters": completed, "world_rank": info["world_rank"],
        "pp_rank": info["pp_rank"], "dp_rank": info["dp_rank"], "device_type": "cpu", "torch_cuda": None,
        **{key: boundary[key] for key in ("torch_cpu", "train_generator", "eval_generator")},
    }
    opt_checks.exact_tree(torch, rng, expected_rng, errors, "saved rank RNG payload")


def train_event(torch, trainer, args, stage, optimizer, info, train, evaluation, completed):
    before = state_bundle(torch, stage, optimizer, train, evaluation)
    x, y = trainer.get_pp_batch(stage=stage, parallel=info, data_path=args.dataset_path,
                               batch_size=args.batch_size, seq_length=args.seq_len,
                               device=torch.device("cpu"), generator=train)
    result = trainer.train_pp_step(stage=stage, optimizer=optimizer, parallel=info,
                                   token_ids=x, targets=y, batch_size=args.batch_size,
                                   seq_length=args.seq_len, micro_batch_size=args.micro_batch_size,
                                   device=torch.device("cpu"), grad_clip=args.clip_norm,
                                   activation_dtype=torch.float64)
    validation_loss = None
    if completed % args.eval_interval == 0:
        validation_loss = trainer.evaluate_pp(stage=stage, parallel=info, data_path=args.val_path,
            batch_size=args.batch_size, seq_length=args.seq_len, micro_batch_size=args.micro_batch_size,
            eval_iters=args.eval_iters, device=torch.device("cpu"), generator=evaluation,
            activation_dtype=torch.float64)
    event = {"completed_iters": completed, "before": before, "token_ids": training.clone_tree(torch, x),
             "targets": training.clone_tree(torch, y), "result": training.clone_tree(torch, result),
             "validation_loss": validation_loss, "after": state_bundle(torch, stage, optimizer, train, evaluation)}
    return event


def normal_event_errors(torch, event):
    errors = []
    result = event["result"]
    if not isinstance(result, dict) or result.get("stepped") is not True:
        errors.append("正常训练应返回 stepped=True")
    else:
        for key in ("loss", "grad_norm"):
            value = result.get(key)
            if value is None or not math.isfinite(float(value)):
                errors.append(f"正常训练的 {key} 应是有限数")
    if event["validation_loss"] is not None and not math.isfinite(float(event["validation_loss"])):
        errors.append("评估 loss 不是有限数")
    if any(value is not None for value in event["after"]["gradients"].values()):
        errors.append("训练边界仍有未清空梯度")
    for name, value in event["after"]["model"].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            errors.append(f"正常训练参数 {name} 出现非有限值")
    if not event["after"]["optimizer"]["state"]:
        errors.append("正常更新后 AdamW 状态仍为空")
    return errors


def make_dense_reference(torch, dist, stage, GPT, args, info):
    parts = [None] * info["world_size"]
    dist.all_gather_object(parts, training.clone_tree(torch, stage.state_dict()))
    full_state = {}
    for p in range(stage.pp_size):
        full_state.update(parts[p])
    with redirect_stdout(io.StringIO()):
        dense = GPT(copy.deepcopy(stage.config)).to(dtype=torch.float64).train()
    dense.load_state_dict(full_state, strict=True)
    return dense, opt_checks.make_optimizer(torch, dense, args)


def check_dense_update(torch, dist, args, stage, optimizer, dense, dense_optimizer, info, event, errors):
    pieces = [None] * info["world_size"]
    dist.all_gather_object(pieces, (event["token_ids"], event["targets"]))
    x = torch.cat([pieces[d * stage.pp_size][0] for d in range(info["dp_size"])])
    y = torch.cat([pieces[(d + 1) * stage.pp_size - 1][1] for d in range(info["dp_size"])])
    _, loss, _ = sync_checks.backward_reference(torch, dense, x, y)
    norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), args.clip_norm)
    dense_optimizer.step()
    dense_optimizer.zero_grad(set_to_none=True)
    opt_checks.check_scalar(torch, event["result"].get("loss"), loss, errors, "global dense loss")
    opt_checks.check_scalar(torch, event["result"].get("grad_norm"), norm, errors, "global dense norm")
    opt_checks.compare_model_and_optimizer(torch, stage, dense, optimizer, dense_optimizer, errors)


def compare_bundles(torch, actual, expected, errors, label):
    opt_checks.exact_tree(torch, actual, expected, errors, label)


def rejected_call(torch, dist, rank, function, label):
    message = None
    try:
        function()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
    messages = [None] * dist.get_world_size()
    dist.all_gather_object(messages, message)
    errors = []
    if any(m is None for m in messages):
        errors.append("错误输入没有让所有 rank 一起抛出异常")
    elif len(set(messages)) != 1:
        errors.append("各 rank 未收到相同的全局错误详情")
    sync_checks.agree(torch, dist, rank, errors, label)
    return {"status": "passed", "case": label, "all_ranks_rejected": True}


def directory_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def mutate_fixture(torch, directory, kind, world):
    manifest_path = directory / "manifest.json"
    if kind == "missing_manifest":
        manifest_path.unlink()
    elif kind == "missing_rng_file":
        (directory / f"rng_rank_{world - 1:05d}.pt").unlink()
    elif kind.startswith("manifest_"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if kind == "manifest_topology":
            manifest["world_size"] += 1
        elif kind == "manifest_training_config":
            manifest["training_config"]["micro_batch_size"] += 1
        elif kind == "manifest_model_config":
            manifest["model_config"]["n_layer"] += 1
        elif kind == "manifest_completed_iters":
            manifest["completed_iters"] = -1
        elif kind == "manifest_version":
            manifest["format_version"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif kind.startswith("stage_"):
        path = directory / "stage_0000.pt"
        payload = load_file(torch, path)
        if kind == "stage_rank":
            payload["pp_rank"] += 1
        elif kind == "stage_optimizer_order":
            group = next(names for names in payload["optimizer_parameter_names"] if len(names) >= 2)
            group[0], group[1] = group[1], group[0]
        elif kind == "stage_optimizer_lr":
            payload["optimizer"]["param_groups"][0]["lr"] *= 2
        elif kind == "stage_model_shape":
            key = "transformer.wte.weight"
            payload["model"][key] = payload["model"][key][:-1].clone()
        elif kind == "stage_model_dtype":
            key = "transformer.wte.weight"
            payload["model"][key] = payload["model"][key].to(dtype=torch.float32)
        torch.save(payload, path)
    else:
        path = directory / f"rng_rank_{world - 1:05d}.pt"
        if kind == "rng_broken_file":
            path.write_bytes(b"intentionally broken checkpoint\n")
        else:
            payload = load_file(torch, path)
            if kind == "rng_rank":
                payload["world_rank"] += 1
            elif kind == "rng_dtype":
                payload["train_generator"] = payload["train_generator"].to(dtype=torch.int32)
            elif kind == "rng_invalid_length":
                payload["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
            torch.save(payload, path)


def verify_failures(torch, dist, rank, args, checkpoint, stage, optimizer, info,
                    train, evaluation, valid_dir, root, label, boundary):
    records = []
    common_args = save_arguments(torch, stage, optimizer, info, root / "failures",
                                 args.warmup_steps, train, evaluation, args)
    before = state_bundle(torch, stage, optimizer, train, evaluation)
    hashes = directory_hashes(valid_dir)
    records.append(rejected_call(torch, dist, rank,
        lambda: checkpoint.save_pp_checkpoint(**{**common_args, "checkpoint_root": valid_dir.parent}),
        f"{label}: reject duplicate step directory"))
    errors = []
    if directory_hashes(valid_dir) != hashes:
        errors.append("重名保存改变了已有 checkpoint")
    compare_bundles(torch, state_bundle(torch, stage, optimizer, train, evaluation), before,
                    errors, "duplicate save preserves live state")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: existing checkpoint preserved")

    if rank == info["world_size"] - 1:
        p = next(stage.parameters())
        p.grad = torch.zeros_like(p)
    try:
        records.append(rejected_call(torch, dist, rank, lambda: checkpoint.save_pp_checkpoint(**common_args),
                                     f"{label}: reject live gradients on one rank"))
    finally:
        optimizer.zero_grad(set_to_none=True)
    errors = []
    if common_args["checkpoint_root"].exists():
        errors.append("梯度检查失败后仍创建了 checkpoint 目录")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: reject before creating files")

    kinds = ["missing_manifest", "missing_rng_file", "manifest_topology", "manifest_training_config",
             "manifest_model_config", "manifest_completed_iters", "manifest_version", "stage_rank",
             "stage_optimizer_order", "stage_optimizer_lr", "stage_model_shape", "stage_model_dtype",
             "rng_rank", "rng_dtype", "rng_broken_file", "rng_invalid_length"]
    for kind in kinds:
        directory = root / f"invalid_{kind}"
        errors = []
        if rank == 0:
            try:
                shutil.copytree(valid_dir, directory)
                mutate_fixture(torch, directory, kind, info["world_size"])
            except Exception as exc:
                errors.append(f"无法构造测试文件：{type(exc).__name__}: {exc}")
        sync_checks.agree(torch, dist, rank, errors, f"{label}: prepare {kind}", quiet=True)
        records.append(rejected_call(torch, dist, rank,
            lambda: checkpoint.load_pp_checkpoint(**load_arguments(
                torch, stage, optimizer, info, directory, train, evaluation, args)),
            f"{label}: all-rank rejection / {kind}"))

    # 只在一个 rank 的保存 I/O 点注入 OSError，模型/通信算法保持原样。
    target = info["world_size"] - 1
    original_save = torch.save
    injected = False
    def failing_save(payload, destination, *positional, **keywords):
        nonlocal injected
        name = Path(destination).name if isinstance(destination, (str, os.PathLike)) else ""
        if rank == target and name.startswith(f"rng_rank_{target:05d}.pt"):
            injected = True
            raise OSError("injected single-rank checkpoint write failure")
        return original_save(payload, destination, *positional, **keywords)
    failed_root = root / "io_failure"
    try:
        torch.save = failing_save
        records.append(rejected_call(torch, dist, rank,
            lambda: checkpoint.save_pp_checkpoint(**{**common_args, "checkpoint_root": failed_root}),
            f"{label}: single-rank write failure consensus"))
    finally:
        torch.save = original_save
    flags = [None] * info["world_size"]
    dist.all_gather_object(flags, injected)
    failed_dir = failed_root / f"step_{args.warmup_steps:08d}"
    errors = []
    if flags != [r == target for r in range(info["world_size"])]:
        errors.append("单 rank I/O 注入没有在预期位置触发")
    if (failed_dir / "manifest.json").exists():
        errors.append("文件未全部写完却发布了 manifest.json")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: incomplete directory has no manifest")
    records.append(rejected_call(torch, dist, rank,
        lambda: checkpoint.load_pp_checkpoint(**load_arguments(
            torch, stage, optimizer, info, failed_dir, train, evaluation, args)),
        f"{label}: reject incomplete save"))
    # 验证错误检查之后，仍能正常加载原来的有效 checkpoint。
    restored = checkpoint.load_pp_checkpoint(**load_arguments(
        torch, stage, optimizer, info, valid_dir, train, evaluation, args))
    errors = [] if restored == args.warmup_steps else ["错误测试之后有效 checkpoint 恢复失败"]
    compare_bundles(torch, state_bundle(torch, stage, optimizer, train, evaluation), boundary,
                    errors, "recovery after rejected checkpoints")
    sync_checks.agree(torch, dist, rank, errors,
        f"{label}: valid load after rejected checkpoints")
    return records


def verify_roundtrip(torch, dist, rank, args, trainer, checkpoint, GPT, GPTConfig, GPTStage,
                     info, pp, dp, bias, dropout, root, failures=False):
    label = f"PP={pp}, DP={dp}, bias={bias}, dropout={dropout}"
    stage, optimizer = make_stage(torch, trainer, GPTConfig, GPTStage, args, info, pp, bias, dropout)
    train, evaluation = generators(torch, args, info["dp_rank"])
    dense, dense_optimizer = (make_dense_reference(torch, dist, stage, GPT, args, info)
                              if dropout == 0 else (None, None))
    for completed in range(1, args.warmup_steps + 1):
        event = train_event(torch, trainer, args, stage, optimizer, info, train, evaluation, completed)
        errors = normal_event_errors(torch, event)
        if dense is not None:
            check_dense_update(torch, dist, args, stage, optimizer, dense, dense_optimizer, info, event, errors)
        sync_checks.agree(torch, dist, rank, errors, f"{label}: warmup {completed}", quiet=True)
    boundary = state_bundle(torch, stage, optimizer, train, evaluation)
    directory = root / "checkpoints" / f"step_{args.warmup_steps:08d}"
    result = checkpoint.save_pp_checkpoint(**save_arguments(
        torch, stage, optimizer, info, directory.parent, args.warmup_steps, train, evaluation, args))
    errors = []
    if not isinstance(result, str) or Path(result).resolve() != directory.resolve():
        errors.append("save_pp_checkpoint 未返回预期 step 目录的字符串路径")
    compare_bundles(torch, state_bundle(torch, stage, optimizer, train, evaluation), boundary,
                    errors, "save preserves live model / optimizer / RNG / generators")
    try:
        inspect_layout(torch, args, stage, info, directory, boundary, args.warmup_steps, errors)
    except Exception as exc:
        errors.append(f"读取保存结果失败：{type(exc).__name__}: {exc}")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: shard files / manifest / nonempty AdamW / RNG")

    expected_events = []
    for completed in range(args.warmup_steps + 1, args.steps + 1):
        event = train_event(torch, trainer, args, stage, optimizer, info, train, evaluation, completed)
        errors = normal_event_errors(torch, event)
        if dense is not None:
            check_dense_update(torch, dist, args, stage, optimizer, dense, dense_optimizer, info, event, errors)
        sync_checks.agree(torch, dist, rank, errors, f"{label}: continuous reference {completed}", quiet=True)
        expected_events.append(event)

    # 创建新的模型和优化器并故意改变三个随机状态，避免“未恢复却碰巧相同”。
    restored_stage, restored_optimizer = make_stage(
        torch, trainer, GPTConfig, GPTStage, args, info, pp, bias, dropout, salt=98765)
    restored_train, restored_eval = generators(torch, args, info["dp_rank"])
    restored_train.manual_seed(args.seed + 333333 + rank)
    restored_eval.manual_seed(args.seed + 444444 + rank)
    torch.rand(19 + rank)
    identities = parameter_ids(restored_stage, restored_optimizer)
    completed = checkpoint.load_pp_checkpoint(**load_arguments(
        torch, restored_stage, restored_optimizer, info, directory, restored_train, restored_eval, args))
    errors = identity_errors(restored_stage, restored_optimizer, identities)
    if type(completed) is not int or completed != args.warmup_steps:
        errors.append("load_pp_checkpoint 未返回正确的 completed_iters 整数")
    compare_bundles(torch, state_bundle(torch, restored_stage, restored_optimizer, restored_train, restored_eval),
                    boundary, errors, "loaded boundary")
    sync_checks.agree(torch, dist, rank, errors, f"{label}: exact roundtrip / Parameter identity / clean boundary")

    for expected in expected_events:
        actual = train_event(torch, trainer, args, restored_stage, restored_optimizer, info,
                             restored_train, restored_eval, expected["completed_iters"])
        errors = normal_event_errors(torch, actual)
        compare_bundles(torch, actual, expected, errors, "resumed vs continuous event")
        sync_checks.agree(torch, dist, rank, errors,
                          f"{label}: resumed step {expected['completed_iters']} exactly matches continuous")
    rejected = (verify_failures(torch, dist, rank, args, checkpoint, restored_stage, restored_optimizer,
                info, restored_train, restored_eval, directory, root, label, boundary) if failures else [])
    return {"status": "passed", "bias": bias, "dropout": dropout, "warmup_steps": args.warmup_steps,
            "resume_steps": args.resume_steps, "dense_reference": dropout == 0,
            "exact_resume": True, "failure_cases": rejected}


def helper_worker(rank, args, temporary):
    common.configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间源码发生变化")
    trainer, checkpoint, GPT, GPTConfig, GPTStage = load_project(args.project)
    world = args.pp_size * args.dp_size
    store = dist.FileStore(str(Path(temporary) / "helper_rendezvous"), world)
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
            for index, (bias, dropout) in enumerate((b, d) for b in args.biases for d in (0.0, args.dropout)):
                root = Path(temporary) / f"helper_pp{pp}_model{index}"
                record["models"].append(verify_roundtrip(torch, dist, rank, args, trainer, checkpoint,
                    GPT, GPTConfig, GPTStage, info, pp, dp, bias, dropout, root, failures=index == 0))
                if rank == 0:
                    (Path(temporary) / f"checkpoint_pp{pp}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                dist.barrier()
            record["status"] = "passed"
            if rank == 0:
                (Path(temporary) / f"checkpoint_pp{pp}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        training.restore_timeouts(dist, originals)


class CheckpointBoundaryStop(Exception):
    """测试用停止信号：保存迭代收尾完成后，在下一批训练采样前停止。"""


def main_argv(args, bias, phase, root):
    values = {
        "--pp-size": args.pp_size, "--device-type": "cpu", "--seed": args.seed,
        "--n-layer": args.n_layer, "--n-head": args.n_head, "--n-embd": args.n_embd,
        "--vocab-size": args.vocab_size, "--block-size": args.block_size, "--dropout": args.dropout,
        "--data-dir": str(Path(args.dataset_path).parent), "--batch-size": args.batch_size,
        "--seq-length": args.seq_len, "--micro-batch-size": args.micro_batch_size,
        "--learning-rate": args.learning_rate, "--weight-decay": args.weight_decay,
        "--beta1": args.betas[0], "--beta2": args.betas[1], "--grad-clip": args.clip_norm,
        "--max-iters": args.steps, "--log-interval": 1, "--eval-interval": args.eval_interval,
        "--eval-iters": args.eval_iters, "--save-interval": args.warmup_steps, "--out-dir": str(root),
    }
    if phase == "resumed":
        values["--resume"] = str(root / f"step_{args.warmup_steps:08d}")
    argv = [str(args.project / "train_pp.py")]
    for flag, value in values.items():
        argv.extend([flag, str(value)])
    if not bias:
        argv.append("--no-bias")
    return argv


def phase_range(args, phase):
    return (args.warmup_steps, args.steps) if phase == "resumed" else (
        0, args.warmup_steps if phase == "interrupted" else args.steps)


def expected_evaluations(args, phase):
    first, last = phase_range(args, phase)
    return [n for n in range(first + 1, last + 1) if n % args.eval_interval == 0 or n == args.steps]


def expected_saves(args, phase):
    first, last = phase_range(args, phase)
    return [n for n in range(first + 1, last + 1) if n % args.warmup_steps == 0 or n == args.steps]


def check_bound_generators(captured, train, evaluation):
    for key, generator in (("train_generator", train), ("eval_generator", evaluation)):
        if key in captured and captured[key] is not generator:
            raise AssertionError(f"main 更换了 {key} 对象")
        captured[key] = generator
    if train is evaluation:
        raise AssertionError("训练和评估共用了同一个 generator")


def main_worker(rank, args, temporary, port, bias, phase):
    common.configure_threads(args.threads)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.pp_size * args.dp_size), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    if fingerprints(args.project) != args.expected_hashes:
        raise RuntimeError("main 验证开始前源码发生变化")
    trainer, checkpoint, _, _, _ = load_project(args.project)
    names = ("build_stage_optimizer", "get_pp_batch", "train_pp_step", "evaluate_pp",
             "save_pp_checkpoint", "load_pp_checkpoint")
    originals = {name: getattr(trainer, name) for name in names}
    if originals["save_pp_checkpoint"] is not checkpoint.save_pp_checkpoint or originals["load_pp_checkpoint"] is not checkpoint.load_pp_checkpoint:
        raise AssertionError("main 应导入并使用 checkpoint_pp.py 的保存与恢复函数")
    signatures = {name: inspect.signature(value) for name, value in originals.items()}
    record = {"rank": rank, "phase": phase, "steps": [], "data_calls": [], "evaluations": [],
              "saves": [], "loads": [], "factory_calls": 0}
    captured, handles = {"current_eval": None, "stop_after_checkpoint": False}, []
    first, last = phase_range(args, phase)
    root = Path(temporary) / ("main_continuous" if phase == "continuous" else "main_interrupted")

    def bound_values(name, positional, keywords):
        bound = signatures[name].bind(*positional, **keywords)
        bound.apply_defaults()
        return bound.arguments

    def live_state():
        return state_bundle(torch, captured["stage"], captured["optimizer"],
                            captured["train_generator"], captured["eval_generator"])

    def observed_factory(*positional, **keywords):
        values = bound_values("build_stage_optimizer", positional, keywords)
        stage = values["stage"]
        optimizer = originals["build_stage_optimizer"](*positional, **keywords)
        errors = training.optimizer_errors(torch, stage, optimizer, args)
        if errors:
            raise AssertionError("main optimizer: " + "; ".join(errors))
        captured.update(stage=stage, optimizer=optimizer)
        audit, handle = training.attach_audit(torch, stage, optimizer)
        captured["audit"] = audit
        handles.append(handle)
        record["factory_calls"] += 1
        record["config"] = vars(stage.config).copy()
        record["model_dtype"] = str(next(stage.parameters()).dtype)
        record["factory_model"] = training.clone_tree(torch, stage.state_dict())
        return optimizer

    def observed_data(*positional, **keywords):
        if phase == "interrupted" and captured["stop_after_checkpoint"] and captured["current_eval"] is None:
            raise CheckpointBoundaryStop("test stop before next training batch after successful checkpoint")
        values = bound_values("get_pp_batch", positional, keywords)
        purpose = "eval" if captured["current_eval"] is not None else "train"
        key = f"{purpose}_generator"
        generator = values["generator"]
        if key in captured and captured[key] is not generator:
            raise AssertionError(f"main 更换了 {key}")
        captured[key] = generator
        if purpose == "eval" and generator is captured.get("train_generator"):
            raise AssertionError("评估共用了训练 generator")
        call = {"data_path": str(Path(values["data_path"]).resolve()),
                "generator_before": generator.get_state().clone(), "torch_before": torch.get_rng_state().clone()}
        result = originals["get_pp_batch"](*positional, **keywords)
        call.update(generator_after=generator.get_state().clone(), torch_after=torch.get_rng_state().clone(),
                    token_ids=training.clone_tree(torch, result[0]), targets=training.clone_tree(torch, result[1]))
        destination = captured["current_eval"]["data_calls"] if purpose == "eval" else record["data_calls"]
        destination.append(call)
        return result

    def observed_step(*positional, **keywords):
        values = bound_values("train_pp_step", positional, keywords)
        stage, optimizer = values["stage"], values["optimizer"]
        if stage is not captured["stage"] or optimizer is not captured["optimizer"]:
            raise AssertionError("main 使用了其他 stage/optimizer")
        if len(record["steps"]) >= last - first:
            raise AssertionError("main 的恢复训练轮数错误：检查 range(start_iter, max_iters)")
        if phase == "resumed" and len(record["loads"]) != 1:
            raise AssertionError("恢复训练开始前没有完成一次 checkpoint 加载")
        event = {"iter_num": first + len(record["steps"]), "token_ids": training.clone_tree(torch, values["token_ids"]),
                 "targets": training.clone_tree(torch, values["targets"]), "torch_before": torch.get_rng_state().clone(),
                 "training": stage.training}
        audit = captured["audit"]
        previous_calls = audit["calls"]
        audit["gradients"] = None
        result = originals["train_pp_step"](*positional, **keywords)
        if audit["calls"] != previous_calls + 1:
            raise AssertionError("正常训练每轮应执行一次 AdamW 更新")
        event.update(result=training.clone_tree(torch, result), gradients=training.clone_tree(torch, audit["gradients"]),
                     model=training.clone_tree(torch, stage.state_dict()),
                     optimizer=training.clone_tree(torch, optimizer.state_dict()),
                     torch_after=torch.get_rng_state().clone(), gradients_cleared=all(p.grad is None for p in stage.parameters()))
        if not event["training"] or not event["gradients_cleared"] or result.get("stepped") is not True:
            raise AssertionError("正常训练模式、更新或梯度清空错误")
        if not math.isfinite(float(result["loss"])) or not math.isfinite(float(result["grad_norm"])):
            raise AssertionError("正常训练的 loss/norm 不是有限数")
        record["steps"].append(event)
        return result

    def observed_eval(*positional, **keywords):
        values = bound_values("evaluate_pp", positional, keywords)
        generator = values["generator"]
        check_bound_generators(captured, captured["train_generator"], generator)
        before = live_state()
        event = {"completed_iters": first + len(record["steps"]), "data_calls": [],
                 "generator_before": generator.get_state().clone(), "torch_before": torch.get_rng_state().clone()}
        captured["current_eval"] = event
        try:
            result = originals["evaluate_pp"](*positional, **keywords)
        finally:
            captured["current_eval"] = None
        after = live_state()
        errors = []
        for key in before:
            if key != "eval_generator":
                compare_bundles(torch, after[key], before[key], errors, f"evaluation preserves {key}")
        if errors:
            raise AssertionError("main evaluation: " + "; ".join(errors[:8]))
        if not math.isfinite(float(result)) or len(event["data_calls"]) != args.eval_iters:
            raise AssertionError("main 评估次数或返回 loss 错误")
        event.update(loss=float(result), generator_after=generator.get_state().clone(), torch_after=torch.get_rng_state().clone())
        record["evaluations"].append(event)
        return result

    def observe_checkpoint(name, positional, keywords):
        values = bound_values(name, positional, keywords)
        if values["stage"] is not captured["stage"] or values["optimizer"] is not captured["optimizer"]:
            raise AssertionError("main 保存/恢复了其他 stage/optimizer")
        if values["training_config"] != run_settings(args):
            raise AssertionError("main 的 training_config 字段与实际训练设置不同")
        check_bound_generators(captured, values["train_data_generator"], values["eval_data_generator"])
        if values["device"] != torch.device("cpu"):
            raise AssertionError("本轮 main 验证应使用 CPU")
        return values

    def observed_save(*positional, **keywords):
        values = observe_checkpoint("save_pp_checkpoint", positional, keywords)
        completed = first + len(record["steps"])
        if values["completed_iters"] != completed:
            raise AssertionError("保存的 completed_iters 与已执行训练循环次数不同")
        before = live_state()
        result = originals["save_pp_checkpoint"](*positional, **keywords)
        errors = []
        compare_bundles(torch, live_state(), before, errors, "main save preserves state")
        wanted = root / f"step_{completed:08d}"
        if not isinstance(result, str) or Path(result).resolve() != wanted.resolve():
            errors.append("main 保存到其他路径或返回值错误")
        try:
            inspect_layout(torch, args, captured["stage"], values["parallel"], wanted, before, completed, errors)
        except Exception as exc:
            errors.append(f"main 保存结果读取失败：{type(exc).__name__}: {exc}")
        if errors:
            raise AssertionError("main save: " + "; ".join(errors[:8]))
        record["saves"].append({"completed_iters": completed, "state": before})
        if phase == "interrupted" and completed == args.warmup_steps:
            captured["stop_after_checkpoint"] = True
        return result

    def observed_load(*positional, **keywords):
        if phase != "resumed" or record["steps"] or record["data_calls"] or record["loads"]:
            raise AssertionError("main 在错误时机调用恢复，或恢复前已经采样训练数据")
        values = observe_checkpoint("load_pp_checkpoint", positional, keywords)
        wanted = root / f"step_{args.warmup_steps:08d}"
        if Path(values["checkpoint_dir"]).resolve() != wanted.resolve():
            raise AssertionError("main 未按 --resume 选择正确的 step 目录")
        identities = parameter_ids(captured["stage"], captured["optimizer"])
        result = originals["load_pp_checkpoint"](*positional, **keywords)
        errors = identity_errors(captured["stage"], captured["optimizer"], identities)
        if type(result) is not int or result != args.warmup_steps:
            errors.append("main 恢复函数返回的 completed_iters 错误")
        if errors:
            raise AssertionError("main load: " + "; ".join(errors))
        record["loads"].append({"completed_iters": result, "state": live_state()})
        return result

    replacements = dict(build_stage_optimizer=observed_factory, get_pp_batch=observed_data,
                        train_pp_step=observed_step, evaluate_pp=observed_eval,
                        save_pp_checkpoint=observed_save, load_pp_checkpoint=observed_load)
    for name, replacement in replacements.items():
        setattr(trainer, name, replacement)
    timeout_originals = training.bound_timeouts(dist, args.timeout)
    previous_argv, output = sys.argv, io.StringIO()
    sys.argv = main_argv(args, bias, phase, root)
    stopped = False
    try:
        with redirect_stdout(output):
            try:
                result = trainer.main()
            except CheckpointBoundaryStop:
                if phase != "interrupted":
                    raise
                stopped = True
                result = None
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
                result = exc.code
        if result not in (None, 0) or stopped != (phase == "interrupted"):
            raise AssertionError("main 返回状态或测试停止边界错误")
        if record["factory_calls"] != 1 or len(record["steps"]) != last - first or len(record["data_calls"]) != last - first:
            raise AssertionError("main 优化器创建次数、训练次数或采样次数错误")
        if [e["completed_iters"] for e in record["evaluations"]] != expected_evaluations(args, phase):
            raise AssertionError("main 的评估时机与迭代数不匹配")
        if [e["completed_iters"] for e in record["saves"]] != expected_saves(args, phase):
            raise AssertionError("main 的保存时机与迭代数不匹配")
        if len(record["loads"]) != int(phase == "resumed"):
            raise AssertionError("main 的恢复调用次数错误")
        if dist.is_initialized():
            raise AssertionError("main 退出时未销毁进程组")
        record.update(status="passed", stdout=output.getvalue(), final_state=live_state())
        torch.save(record, Path(temporary) / f"{phase}_rank{rank}.pt")
    finally:
        (Path(temporary) / f"{phase}_rank{rank}_stdout.txt").write_text(output.getvalue(), encoding="utf-8")
        sys.argv = previous_argv
        for name, original in originals.items():
            setattr(trainer, name, original)
        for handle in handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
        training.restore_timeouts(dist, timeout_originals)


def compare_main_records(torch, np, args, temporary, bias):
    world = args.pp_size * args.dp_size
    phases = {phase: [load_file(torch, Path(temporary) / f"{phase}_rank{r}.pt") for r in range(world)]
              for phase in ("continuous", "interrupted", "resumed")}
    errors = []
    expected_config = {"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                       "vocab_size": args.vocab_size, "block_size": args.block_size,
                       "dropout": args.dropout, "bias": bias}
    for r in range(world):
        full, stopped, resumed = (phases[name][r] for name in ("continuous", "interrupted", "resumed"))
        for phase, record in (("continuous", full), ("interrupted", stopped), ("resumed", resumed)):
            if record["config"] != expected_config or record["model_dtype"] != "torch.float32":
                errors.append(f"rank {r}, {phase}: main 未正确使用模型 CLI 配置或 dtype")
        compare_bundles(torch, stopped["steps"], full["steps"][:args.warmup_steps], errors, f"rank {r}: interrupted prefix steps")
        compare_bundles(torch, resumed["steps"], full["steps"][args.warmup_steps:], errors, f"rank {r}: resumed suffix steps")
        compare_bundles(torch, stopped["data_calls"], full["data_calls"][:args.warmup_steps], errors, f"rank {r}: prefix sampling")
        compare_bundles(torch, resumed["data_calls"], full["data_calls"][args.warmup_steps:], errors, f"rank {r}: resumed sampling")
        compare_bundles(torch, stopped["evaluations"] + resumed["evaluations"], full["evaluations"], errors, f"rank {r}: evaluations / eval RNG")
        compare_bundles(torch, stopped["saves"] + resumed["saves"], full["saves"], errors, f"rank {r}: saved boundaries")
        compare_bundles(torch, resumed["loads"][0]["state"], stopped["saves"][-1]["state"], errors, f"rank {r}: actual main loaded boundary")
        compare_bundles(torch, resumed["final_state"], full["final_state"], errors, f"rank {r}: final model / AdamW / RNG")
        for phase in phases:
            record = phases[phase][r]
            first, last = phase_range(args, phase)
            logs = re.findall(r"(?m)^\s*(?:iter|step)\s+(\d+):\s+loss=([^,\s]+)", record["stdout"])
            eval_logs = re.findall(r"(?m)^\s*eval after iter\s+(\d+):\s+val_loss=([^,\s]+)", record["stdout"])
            if r == 0:
                logged_iters, expected_iters = [int(i) for i, _ in logs], list(range(first, last))
                if logged_iters != expected_iters:
                    errors.append(f"{phase}: rank 0 的迭代日志错误：实际 {logged_iters}，应为 {expected_iters}")
                for (_, text), step in zip(logs, record["steps"]):
                    if not math.isclose(float(text), float(step["result"]["loss"]), rel_tol=0, abs_tol=5.1e-7):
                        errors.append(f"{phase}: 训练日志 loss 与真实返回值不同")
                if [int(i) + 1 for i, _ in eval_logs] != expected_evaluations(args, phase):
                    errors.append(f"{phase}: rank 0 的评估日志时机错误")
            elif logs or eval_logs:
                errors.append(f"rank {r}: 非主进程打印了训练或评估 loss")

    # 独立重放实际 main 的采样，验证两个 generator 的种子和持续推进。
    for d in range(args.dp_size):
        train, evaluation = generators(torch, args, d)
        eval_by_completed = {e["completed_iters"]: index for index, e in enumerate(phases["continuous"][d * args.pp_size]["evaluations"])}
        for iteration in range(args.steps):
            before = train.get_state().clone()
            x, y = training.oracle_batch(torch, np, args.dataset_path, args, train)
            after = train.get_state().clone()
            for p in range(args.pp_size):
                r = d * args.pp_size + p
                calls = phases["continuous"][r]["data_calls"]
                call = calls[iteration]
                flags = argparse.Namespace(is_first=p == 0, is_last=p == args.pp_size - 1)
                training.check_batch(torch, flags, call["token_ids"], call["targets"], x, y, errors)
                wanted_before = before if p == 0 else calls[0]["generator_before"]
                wanted_after = after if p == 0 else wanted_before
                compare_bundles(torch, call["generator_before"], wanted_before, errors, f"rank {r}: train sampler before")
                compare_bundles(torch, call["generator_after"], wanted_after, errors, f"rank {r}: train sampler after")
                compare_bundles(torch, call["torch_before"], call["torch_after"], errors, f"rank {r}: sampling preserves global RNG")
                if Path(call["data_path"]) != args.dataset_path:
                    errors.append(f"rank {r}: 训练数据路径错误")
            completed = iteration + 1
            if completed not in eval_by_completed:
                continue
            eval_index = eval_by_completed[completed]
            for batch in range(args.eval_iters):
                before = evaluation.get_state().clone()
                x, y = training.oracle_batch(torch, np, args.val_path, args, evaluation)
                after = evaluation.get_state().clone()
                for p in range(args.pp_size):
                    r = d * args.pp_size + p
                    evals = phases["continuous"][r]["evaluations"]
                    call = evals[eval_index]["data_calls"][batch]
                    flags = argparse.Namespace(is_first=p == 0, is_last=p == args.pp_size - 1)
                    training.check_batch(torch, flags, call["token_ids"], call["targets"], x, y, errors)
                    initial = evals[0]["data_calls"][0]["generator_before"]
                    wanted_before = before if p == 0 else initial
                    wanted_after = after if p == 0 else initial
                    compare_bundles(torch, call["generator_before"], wanted_before, errors, f"rank {r}: eval sampler before")
                    compare_bundles(torch, call["generator_after"], wanted_after, errors, f"rank {r}: eval sampler after")
                    if Path(call["data_path"]) != args.val_path:
                        errors.append(f"rank {r}: 评估数据路径错误")

    for step in range(args.steps):
        for r in range(world):
            compare_bundles(torch, phases["continuous"][r]["steps"][step]["model"],
                phases["continuous"][r % args.pp_size]["steps"][step]["model"], errors, f"rank {r}: DP model replicas")
        for d in range(args.dp_size):
            wte = phases["continuous"][d * args.pp_size]["steps"][step]["model"]["transformer.wte.weight"]
            head = phases["continuous"][(d + 1) * args.pp_size - 1]["steps"][step]["model"]["lm_head.weight"]
            compare_bundles(torch, wte, head, errors, f"DP {d}: tied endpoint weights")
    full_dir = Path(temporary) / "main_continuous" / f"step_{args.steps:08d}"
    resumed_dir = Path(temporary) / "main_interrupted" / f"step_{args.steps:08d}"
    for filename in [f"stage_{p:04d}.pt" for p in range(args.pp_size)] + [f"rng_rank_{r:05d}.pt" for r in range(world)]:
        compare_bundles(torch, load_file(torch, resumed_dir / filename), load_file(torch, full_dir / filename), errors,
                        f"final serialized checkpoint: {filename}")
    compare_bundles(torch, json.loads((resumed_dir / "manifest.json").read_text(encoding="utf-8")),
                    json.loads((full_dir / "manifest.json").read_text(encoding="utf-8")), errors, "final manifest")
    if errors:
        raise AssertionError("实际 main 保存/恢复验证失败：" + "; ".join(errors[:16]))
    return {"status": "passed", "pp_size": args.pp_size, "dp_size": args.dp_size, "bias": bias,
            "dropout": args.dropout, "dtype": "float32", "total_iters": args.steps,
            "checkpoint_boundary": args.warmup_steps, "exact_resume": True,
            "continuous_iters": list(range(args.steps)), "resumed_iters": list(range(args.warmup_steps, args.steps)),
            "eval_completed_iters": expected_evaluations(args, "continuous"),
            "saved_completed_iters": expected_saves(args, "continuous")}


def make_datasets(np, args, temporary):
    training.make_datasets(np, args, temporary)
    args.val_path = args.dataset_path.parent / "val.bin"
    rng = np.random.default_rng(args.seed + 888)
    rng.integers(0, args.vocab_size, size=max(4096, 4 * (args.seq_len + 1)), dtype=np.uint16).tofile(args.val_path)


def main(argv=None):
    args = parse_args(argv)
    common.configure_threads(args.threads)
    report = {"status": "running", "scope": "PP_DP_checkpoint_save_load_and_actual_main_resume",
              "requested_cases": args.cases, "completed_cases": [], "actual_main": {"status": "not_run"},
              "not_checked": ["CUDA/NCCL", "AMP", "different PP/DP topology resume", "LR schedule",
                              "torchrun CLI", "power-loss durability", "real corpus", "performance"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / checkpoint and evaluation signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only_passed"
            common.write_report(args.report, report)
            print("[PASS] Static preflight completed; checkpoint numerical verification was not run.", flush=True)
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
                      helper_dtype="float64", main_dtype="float32", dropouts=[0.0, args.dropout],
                      warmup_steps=args.warmup_steps, resume_steps=args.resume_steps,
                      batch_size=args.batch_size, seq_len=args.seq_len, micro_batch_size=args.micro_batch_size,
                      eval_interval=args.eval_interval, eval_iters=args.eval_iters, seed=args.seed,
                      learning_rate=args.learning_rate, weight_decay=args.weight_decay, betas=list(args.betas),
                      grad_clip=args.clip_norm, timeout=args.timeout,
                      model_dimensions={"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                                        "vocab_size": args.vocab_size, "block_size": args.block_size})
        print(f"[INFO] {world} CPU workers; threads={args.threads}; helper cases={args.cases}.", flush=True)
        print("[INFO] Real temporary checkpoint files; model / AdamW / generators / RNG / dropout resume.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_checkpoint_") as temporary:
            make_datasets(np, args, temporary)
            try:
                mp.spawn(helper_worker, args=(args, temporary), nprocs=world, join=True)
            finally:
                for pp, _ in args.cases:
                    path = Path(temporary) / f"checkpoint_pp{pp}.json"
                    if path.is_file():
                        report["completed_cases"].append(json.loads(path.read_text(encoding="utf-8")))
            if len(report["completed_cases"]) != len(args.cases) or any(c["status"] != "passed" for c in report["completed_cases"]):
                raise RuntimeError("checkpoint 辅助函数检查未全部完成")
            if args.skip_main:
                report["actual_main"] = {"status": "skipped_by_flag"}
                report["not_checked"].append("actual train_pp.main checkpoint/resume integration")
            else:
                bias = args.bias != "false"
                try:
                    for phase in ("continuous", "interrupted", "resumed"):
                        print(f"[INFO] Fresh workers: actual main / {phase}; PP={args.pp_size}, DP={args.dp_size}; "
                              f"dropout={args.dropout}.", flush=True)
                        mp.spawn(main_worker, args=(args, temporary, training.unused_port(), bias, phase), nprocs=world, join=True)
                    report["actual_main"] = compare_main_records(torch, np, args, temporary, bias)
                except Exception:
                    report["actual_main"] = {"status": "failed", "captured_stdout": {
                        path.name: path.read_text(encoding="utf-8")[-10000:]
                        for phase in ("continuous", "interrupted", "resumed")
                        for path in Path(temporary).glob(f"{phase}_rank*_stdout.txt")}}
                    raise
                print("[PASS] actual main: fresh-process resume / iterations / sampling / gradients / AdamW / eval / RNG.", flush=True)
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码或验证器发生变化")
        report["status"] = "helpers_passed_main_skipped" if args.skip_main else "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        common.write_report(args.report, report)
        raise
    common.write_report(args.report, report)
    if args.skip_main:
        print("[PASS] PP checkpoint helper verification completed; actual main was skipped.", flush=True)
    else:
        print("[PASS] PP checkpoint verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
