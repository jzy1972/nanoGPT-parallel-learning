#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证 nanoGPT 手写 ZeRO-1 的 checkpoint 与实际 main 恢复。

放在 model.py、zero1.py、train_zero.py 同级，运行：
  python verify_zero1_checkpoint.py --project . --device cpu --dp-size 4 --threads 1 --report zero1_checkpoint_report.json

只依赖标准库、NumPy、PyTorch；不依赖旧验证脚本，不修改项目源码。
自动在临时目录生成 uint16 数据和 checkpoint，结束时清理。

函数检查使用 float64、非零 Dropout、WORLD/单 rank/奇偶子组，检查
参数归属、AdamW 分片序列化、空分片、共享 Parameter 引用、RNG、
恢复后的下一次更新，以及不完整/不兼容 checkpoint 的协调拒绝。
通过注入普通写入异常，检查失败时没有 complete.pt，并能继续保存。

实际 main 使用 FP32、非零 Dropout、真实 env://、DDP 和 ZeRO-1。
每个配置分别启动新进程：连续训练；训练到中途后正常保存退出；
重新启动并恢复至同一总迭代数。比较每步数据、loss、参数、AdamW
一阶/二阶矩、step、Python/NumPy/PyTorch RNG、数据 generator，
并在 main 正常结束后检查完整日志、评估/保存顺序与迭代起点。
另检查恢复到已经完成的迭代数时不重复训练或保存。

数据、RNG、加载瞬间的参数/状态要求精确相同；重新计算的训练轨迹
按 float64/FP32 容差比较，不要求 CUDA 内核跨进程位级一致。
这不是 AdamW 数学或性能测试；请同时保留之前的优化器/训练/评估验证。
恢复范围：相同代码、数据、DP 规模、参数布局和设备类型。
不覆盖改变 DP 规模、AMP、torch.compile、多机或进程硬崩溃恢复。

CPU 使用 Gloo；CUDA 使用 NCCL，单机每 rank 一个设备。
--dp-size 仅为验证进程数，不限定实现规模。Windows 使用线程限制、
127.0.0.1 与 USE_LIBUV=0。--preflight-only 不导入 PyTorch。
"""

import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
from functools import wraps
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import sys
import tempfile
import traceback
from types import SimpleNamespace
from contextlib import contextmanager, redirect_stderr, redirect_stdout


METHODS = {
    "__init__": ("self", "model", "learning_rate", "weight_decay", "beta1", "beta2", "dp_group"),
    "zero_grad": ("self", "set_to_none"),
    "sync_parameters": ("self",), "step": ("self", "grad_clip"),
    "has_nonfinite_gradients": ("self",), "clip_gradients": ("self", "max_norm"),
    "_checkpoint_layout": ("self",), "state_dict": ("self",),
    "load_state_dict": ("self", "state"),
}
TRAINER_SIGNATURES = {
    "parse_args": (), "main": (),
    "get_zero_batch": ("data_path", "batch_size", "seq_length", "device", "generator"),
    "train_zero_step": ("model", "optimizer", "x", "y", "micro_batch_size", "grad_clip"),
    "estimate_zero_loss": ("model", "data_path", "batch_size", "seq_length", "micro_batch_size", "eval_iters", "device", "dp_group", "seed"),
    "capture_rng_state": ("device", "train_data_generator"),
    "restore_rng_state": ("state", "device", "train_data_generator"),
    "atomic_torch_save": ("payload", "path"),
    "check_checkpoint_errors": ("error", "dp_group", "phase"),
    "save_zero_checkpoint": ("out_dir", "model", "optimizer", "next_iter", "train_data_generator", "device", "train_args"),
    "load_zero_checkpoint": ("checkpoint_dir", "model", "optimizer", "train_data_generator", "device", "train_args"),
}
FIXED_KEYS = ("seed", "data_dir", "batch_size", "seq_length", "micro_batch_size",
              "grad_clip", "learning_rate", "weight_decay", "beta1", "beta2")
LOG_PATTERN = re.compile(r"^iter\s+(\d+):\s+loss=(\S+),\s+grad_norm=(\S+),\s+stepped=(True|False)\s*$")
EVAL_LOG_PATTERN = re.compile(r"^eval iter\s+(\d+):\s+train loss=(\S+),\s+val loss=(\S+)\s*$")

def configure_threads(count):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(count)
    os.environ["USE_LIBUV"] = "0"


def group_layouts(world):
    result = [("WORLD", [list(range(world))])]
    if world > 1:
        result.append(("singleton", [[rank] for rank in range(world)]))
    if world > 2:
        result.append(("strided", [list(range(0, world, 2)), list(range(1, world, 2))]))
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def clone_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_tree(torch, item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(torch, item) for item in value)
    return copy.deepcopy(value)


def agree(dist, rank, errors, label, record):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors[:10])
    if any(gathered):
        if rank == 0:
            for other_rank, failures in enumerate(gathered):
                for failure in failures:
                    print(f"[FAIL rank={other_rank}] {label}: {failure}", flush=True)
        raise AssertionError(f"验证失败：{label}")
    record["checks"].append(label)
    if rank == 0:
        print(f"[PASS] {label}", flush=True)


def alias_map(raw):
    canonical = {id(param): name for name, param in raw.named_parameters()}
    return {name: canonical[id(param)] for name, param in raw.named_parameters(remove_duplicate=False)}


def fingerprints(project):
    paths = [project / name for name in ("model.py", "zero1.py", "train_zero.py")]
    paths.append(Path(__file__).resolve())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def ast_signature(node, expected, label):
    names = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    missing = [name for name in expected if name not in names]
    if missing:
        raise TypeError(f"{label} 缺少参数：{', '.join(missing)}")


def preflight(project):
    trees = {}
    for filename in ("model.py", "zero1.py", "train_zero.py"):
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"项目根目录需要 {path}")
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
        trees[filename] = ast.parse(source, filename=str(path))
    for name, required in TRAINER_SIGNATURES.items():
        found = [node for node in trees["train_zero.py"].body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(found) != 1:
            raise AttributeError(f"train_zero.py 需要唯一的顶层 {name}()")
        ast_signature(found[0], required, f"train_zero.{name}")
    top = trees["zero1.py"].body
    functions = [node for node in top if isinstance(node, ast.FunctionDef) and node.name == "build_parameter_partition"]
    classes = [node for node in top if isinstance(node, ast.ClassDef) and node.name == "Zero1Optimizer"]
    if len(functions) != 1 or len(classes) != 1:
        raise AttributeError("zero1.py 需要唯一的 build_parameter_partition() 和 Zero1Optimizer")
    ast_signature(functions[0], ("model", "dp_size"), "build_parameter_partition")
    for name, required in METHODS.items():
        found = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(found) != 1:
            raise AttributeError(f"Zero1Optimizer 需要唯一的 {name}() 方法")
        ast_signature(found[0], required, f"Zero1Optimizer.{name}")
    declared = {node.name for node in trees["model.py"].body if isinstance(node, ast.ClassDef)}
    if not {"GPT", "GPTConfig"}.issubset(declared):
        raise AttributeError("model.py 需要原版 GPT 和 GPTConfig")
    return fingerprints(project)


def load_project(project):
    sys.path.insert(0, str(project))
    for name in ("train_zero", "zero1", "model"):
        sys.modules.pop(name, None)
    modules = {name: importlib.import_module(name) for name in ("model", "zero1", "train_zero")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != project / f"{name}.py":
            raise ImportError(f"{name} 导入了其他目录的文件：{module.__file__}")
    for name, expected in TRAINER_SIGNATURES.items():
        if any(arg not in inspect.signature(getattr(modules["train_zero"], name)).parameters for arg in expected):
            raise TypeError(f"train_zero.{name} 的运行时参数与静态检查不符")
    return modules["model"].GPT, modules["model"].GPTConfig, modules["zero1"], modules["train_zero"]


@contextmanager
def patched_attributes(changes):
    original = []
    try:
        for obj, name, value in changes:
            previous = getattr(obj, name)
            original.append((obj, name, previous))
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, previous in reversed(original):
            setattr(obj, name, previous)


@contextmanager
def command_line(argv):
    previous = sys.argv
    sys.argv = ["train_zero.py"] + list(argv)
    try:
        yield
    finally:
        sys.argv = previous


def bind_call(function, positional, keyword):
    bound = inspect.signature(function).bind(*positional, **keyword)
    bound.apply_defaults()
    return bound.arguments


def oracle_batch(torch, np, data, count, length, generator, device):
    starts = torch.randint(len(data) - length, (count,), generator=generator, device="cpu").tolist()
    x = torch.from_numpy(np.stack([data[start:start + length] for start in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[start + 1:start + 1 + length] for start in starts]).astype(np.int64))
    return x.to(device), y.to(device), starts


def setup_worker(torch, args, rank):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def collect_workers(directory, world):
    result = []
    for rank in range(world):
        path = directory / f"rank_{rank}.json"
        if path.is_file():
            result.append(json.loads(path.read_text(encoding="utf-8")))
    return result


def require_all_workers(workers, world, label):
    if len(workers) != world or any(item["status"] != "passed" for item in workers):
        raise AssertionError(f"{label}: 部分 worker 没有完整通过")


def check_tree(torch, actual, expected, errors, label, args, exact=False):
    """结构/dtype 始终相同；浮点训练值可使用数值容差。"""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            errors.append(f"{label}: 应为 Tensor")
            return
        try:
            torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                rtol=0 if exact else args.rtol, atol=0 if exact else args.atol,
                equal_nan=False)
        except (AssertionError, TypeError, RuntimeError) as exc:
            errors.append(f"{label}: {str(exc)[:700]}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            errors.append(f"{label}: 字典键不同")
            return
        for key in expected:
            check_tree(torch, actual[key], expected[key], errors, f"{label}.{key}", args, exact)
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            errors.append(f"{label}: 序列不同")
            return
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            check_tree(torch, got, wanted, errors, f"{label}[{index}]", args, exact)
    elif isinstance(expected, float) and not exact:
        if type(actual) is not float or not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(
                actual, expected, rel_tol=args.rtol, abs_tol=args.atol):
            errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")
    elif isinstance(actual, torch.Tensor) or type(actual) is not type(expected) or actual != expected:
        errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")


def rng_snapshot(torch, np, device, generator):
    """独立读取 RNG，不调用被测 capture_rng_state。"""
    state = np.random.get_state()
    return {"device_type": device.type, "python": random.getstate(),
        "numpy": {"algorithm": state[0], "keys": torch.tensor(state[1].astype(np.int64), dtype=torch.int64),
                  "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None,
        "train_data_generator": generator.get_state().clone()}


def full_snapshot(torch, np, raw, optimizer, device, generator):
    return {"model": clone_tree(torch, raw.state_dict()),
            "optimizer": clone_tree(torch, optimizer.state_dict()),
            "rng": rng_snapshot(torch, np, device, generator),
            "training": bool(raw.training), "aliases": alias_map(raw),
            "gradients": {name: None if param.grad is None else param.grad.detach().cpu().clone()
                          for name, param in raw.named_parameters()}}


def compare_snapshot(torch, actual, expected, errors, label, args, exact=False):
    if not isinstance(actual, dict) or not isinstance(expected, dict) or actual.keys() != expected.keys():
        errors.append(f"{label}: 快照字段不完整")
        return
    for key in expected:
        check_tree(torch, actual[key], expected[key], errors, f"{label}.{key}", args,
                   exact=exact or key in ("rng", "training", "aliases", "gradients"))


def reference_layout(raw, size, rank):
    params = [(name, param) for name, param in raw.named_parameters() if param.requires_grad]
    loads, owners = [0] * size, {}
    for name, param in sorted(params, key=lambda item: (-item[1].numel(), item[0])):
        owner = min(range(size), key=lambda item: (loads[item], item))
        owners[name], loads[owner] = owner, loads[owner] + param.numel()
    groups = []
    for decayed in (True, False):
        names = [name for name, param in params if owners[name] == rank and (param.dim() >= 2) == decayed]
        if names:
            groups.append(names)
    return {"dp_size": size, "dp_rank": rank,
            "parameters": [(name, tuple(param.shape), str(param.dtype)) for name, param in params],
            "owners": owners, "local_param_groups": groups}, loads


def optimizer_errors(torch, raw, optimizer, errors, args, initialized):
    layout, loads = reference_layout(raw, optimizer.dp_size, optimizer.dp_rank)
    check_tree(torch, optimizer._checkpoint_layout(), layout, errors, "checkpoint layout", args, exact=True)
    if optimizer.rank_numels != loads or optimizer.owners != layout["owners"]:
        errors.append("参数归属/负载与独立分配参考不符")
    state = optimizer.state_dict()
    if not isinstance(state, dict) or set(state) != {"format_version", "zero_stage", "layout", "local_optimizer", "last_grad_norm"}:
        errors.append("Zero1Optimizer.state_dict 字段错误")
        return
    for key, value in (("format_version", 1), ("zero_stage", 1), ("layout", layout), ("last_grad_norm", optimizer.last_grad_norm)):
        check_tree(torch, state[key], value, errors, f"optimizer.{key}", args, exact=True)
    current = dict(raw.named_parameters())
    for name, param in optimizer.named_params:
        if current.get(name) is not param:
            errors.append(f"优化器参数引用错误：{name}")
    owned = {name: param for name, param in current.items() if param.requires_grad and layout["owners"][name] == optimizer.dp_rank}
    local = optimizer.local_optimizer
    if not owned:
        if local is not None or state["local_optimizer"] is not None:
            errors.append("没有归属参数的 rank 应保存 local_optimizer=None")
        return
    if not isinstance(local, torch.optim.AdamW):
        errors.append("本地优化器应为 AdamW")
        return
    actual_params = [param for group in local.param_groups for param in group["params"]]
    if len(actual_params) != len(owned) or {id(param) for param in actual_params} != {id(param) for param in owned.values()}:
        errors.append("本地优化器参数集合不是恰好归属参数")
    check_tree(torch, state["local_optimizer"], local.state_dict(), errors, "native optimizer serialization", args, exact=True)
    expected_ids = {id(param) for param in owned.values()} if initialized else set()
    if {id(param) for param in local.state} != expected_ids:
        errors.append("AdamW 状态没有恰好按归属参数分片")
    if initialized:
        for name, param in owned.items():
            moments = local.state.get(param, {})
            if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                errors.append(f"AdamW.{name}: 状态字段错误")
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if moments[key].shape != param.shape or moments[key].dtype != param.dtype or moments[key].device != param.device:
                    errors.append(f"AdamW.{name}.{key}: 形状/dtype/设备错误")


def reference_checks(raw, optimizer, identities, aliases, errors):
    if {name: id(param) for name, param in raw.named_parameters()} != identities:
        errors.append("保存/恢复替换了模型 Parameter 对象")
    if alias_map(raw) != aliases:
        errors.append("保存/恢复改变了共享参数引用")
    current = dict(raw.named_parameters())
    for name, param in optimizer.named_params:
        if current.get(name) is not param:
            errors.append(f"保存/恢复使优化器引用失效：{name}")
    if any(param.grad is not None for param in raw.parameters()):
        errors.append("迭代边界或恢复后没有清空完整模型梯度")


def inspect_checkpoint(torch, directory, size, rank, next_iter, raw, train_args, expected, args, errors):
    directory = Path(directory)
    expected_files = ["model.pt"] + [f"rank_{item:05d}.pt" for item in range(size)]
    actual_files = sorted(path.name for path in directory.iterdir())
    if actual_files != sorted(expected_files + ["complete.pt"]):
        errors.append("checkpoint 文件集合错误，或有残留 .tmp 文件")
    # 三个文件都以 weights_only=True 读取，避免依赖 NumPy 对象反序列化。
    manifest = torch.load(directory / "complete.pt", map_location="cpu", weights_only=True)
    common = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    shard = torch.load(directory / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
    for label, payload in (("manifest", manifest), ("model", common), ("rank", shard)):
        for key, value in (("format_version", 1), ("zero_stage", 1), ("dp_size", size), ("next_iter", next_iter)):
            check_tree(torch, payload.get(key), value, errors, f"{label}.{key}", args, exact=True)
    if manifest.get("complete") is not True or manifest.get("files") != expected_files:
        errors.append("complete.pt 完成标记或文件列表错误")
    check_tree(torch, shard.get("dp_rank"), rank, errors, "rank.dp_rank", args, exact=True)
    check_tree(torch, common.get("model_args"), dict(vars(raw.config)), errors, "saved GPTConfig", args, exact=True)
    check_tree(torch, common.get("train_args"), dict(train_args), errors, "saved train_args", args, exact=True)
    check_tree(torch, common.get("model"), expected["model"], errors, "saved model", args, exact=True)
    check_tree(torch, shard.get("optimizer"), expected["optimizer"], errors, "saved optimizer shard", args, exact=True)
    check_tree(torch, shard.get("rng"), expected["rng"], errors, "saved independent RNG", args, exact=True)
    return manifest, common, shard


def make_train_args(args, data_dir, bias, clip, maximum=20):
    return {"seed": args.seed, "data_dir": str(data_dir), "batch_size": args.batch_size,
        "seq_length": args.seq_len, "micro_batch_size": args.micro_batch_size, "grad_clip": clip,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "beta1": args.betas[0], "beta2": args.betas[1], "max_iters": maximum,
        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
        "vocab_size": args.vocab_size, "block_size": args.block_size,
        "dropout": args.dropout, "no_bias": not bias, "device_type": args.device,
        "log_interval": 2, "eval_interval": 2, "eval_iters": args.eval_iters,
        "checkpoint_interval": 2, "out_dir": "unused", "resume": None}


def optimizer_rejection_checks(torch, np, optimizer, raw, generator, device, args, errors):
    good = clone_tree(torch, optimizer.state_dict())
    mutations = [("version", lambda value: value.update(format_version=99)),
                 ("stage", lambda value: value.update(zero_stage=2)),
                 ("rank", lambda value: value["layout"].update(dp_rank=optimizer.dp_size)),
                 ("size", lambda value: value["layout"].update(dp_size=optimizer.dp_size + 1)),
                 ("layout", lambda value: value["layout"].update(parameters=[])),
                 ("owners", lambda value: value["layout"].update(owners={})),
                 ("groups", lambda value: value["layout"].update(local_param_groups=[["wrong"]])),
                 ("missing state", lambda value: value.pop("local_optimizer")),
                 ("negative norm", lambda value: value.update(last_grad_norm=-1.0)),
                 ("nonfinite norm", lambda value: value.update(last_grad_norm=float("inf"))),
                 ("local state", lambda value: value.update(local_optimizer={} if optimizer.local_optimizer is None else None))]
    for label, mutate in mutations:
        value = clone_tree(torch, good)
        mutate(value)
        before = full_snapshot(torch, np, raw, optimizer, device, generator)
        try:
            optimizer.load_state_dict(value)
        except (ValueError, TypeError, KeyError, RuntimeError):
            pass
        else:
            errors.append(f"优化器没有拒绝非法 {label}")
        after = full_snapshot(torch, np, raw, optimizer, device, generator)
        compare_snapshot(torch, after, before, errors, f"rejected optimizer {label}: unchanged", args, exact=True)


def construct_model(torch, GPT, GPTConfig, zero, args, device, group, bias, kind):
    if kind == "empty_owner_probe":
        class OneParameter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(kind="one_parameter")
                self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float64))
                self.tied_weight = self.weight
        raw = OneParameter().to(device)
        model = raw
    else:
        raw = GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            vocab_size=args.vocab_size, block_size=args.block_size, dropout=args.dropout, bias=bias))
        raw = raw.to(device=device, dtype=torch.float64)
        model = torch.nn.parallel.DistributedDataParallel(raw, process_group=group,
            **({"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}))
    optimizer = zero.Zero1Optimizer(model=raw, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, beta1=args.betas[0], beta2=args.betas[1], dp_group=group)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    return model, raw, optimizer


def perform_step(torch, trainer, model, raw, optimizer, generator, device, data_path, args, kind, event, clip):
    if kind == "empty_owner_probe":
        optimizer.zero_grad(set_to_none=True)
        # 所有副本使用同一完整梯度；这里只测试空归属 rank 的状态恢复。
        raw.weight.grad = torch.tensor([[0.2 + event * 0.03, -0.1]], dtype=torch.float64, device=device)
        stepped = optimizer.step(grad_clip=clip)
        result = {"loss": 0.0, "grad_norm": optimizer.last_grad_norm, "stepped": stepped}
        optimizer.zero_grad(set_to_none=True)
        # 覆盖不同 RNG 流的恢复；这些消费仅属于函数检查。
        random.random()
        __import__("numpy").random.normal(size=3)
        torch.rand(7, device=device)
        torch.randint(1000, (5,), generator=generator)
        return {"x": None, "y": None, "result": result}
    x, y = trainer.get_zero_batch(data_path=str(data_path), batch_size=args.batch_size,
        seq_length=args.seq_len, device=device, generator=generator)
    result = trainer.train_zero_step(model=model, optimizer=optimizer, x=x, y=y,
        micro_batch_size=args.micro_batch_size, grad_clip=clip)
    return {"x": clone_tree(torch, x), "y": clone_tree(torch, y), "result": clone_tree(torch, result)}


def expect_rejection(torch, np, dist, rank, call, raw, optimizer, generator, device, args, label, record):
    errors = []
    before = full_snapshot(torch, np, raw, optimizer, device, generator)
    exception = None
    try:
        call()
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    if exception is None:
        errors.append("没有拒绝非法/不完整 checkpoint")
    after = full_snapshot(torch, np, raw, optimizer, device, generator)
    compare_snapshot(torch, after, before, errors, f"{label}: metadata rejection before mutation", args, exact=True)
    agree(dist, rank, errors, label, record)
    return exception


def mutate_saved(torch, directory, size, case):
    """只改临时 checkpoint；被测源码和用户 checkpoint 不参与。"""
    manifest_path = directory / "complete.pt"
    model_path = directory / "model.pt"
    shard_path = directory / f"rank_{size - 1:05d}.pt"
    if case == "missing complete":
        manifest_path.unlink()
        return
    if case == "missing shard":
        shard_path.unlink()
        return
    filename = model_path if case in ("version", "config", "fixed args") else (
        shard_path if case in ("rank", "counter", "layout", "device type") else manifest_path)
    value = torch.load(filename, map_location="cpu", weights_only=True)
    if case == "version":
        value["format_version"] = 99
    elif case == "stage":
        value["zero_stage"] = 2
    elif case == "size":
        value["dp_size"] = size + 1
    elif case == "rank":
        value["dp_rank"] = size
    elif case == "counter":
        value["next_iter"] += 1
    elif case == "negative counter":
        value["next_iter"] = -1
    elif case == "incomplete":
        value["complete"] = False
    elif case == "file list":
        value["files"] = ["model.pt"]
    elif case == "config":
        value["model_args"]["n_embd"] += 1
    elif case == "layout":
        value["optimizer"]["layout"]["owners"] = {}
    elif case == "fixed args":
        value["train_args"]["batch_size"] += 1
    elif case == "device type":
        value["rng"]["device_type"] = "cuda" if value["rng"]["device_type"] == "cpu" else "cpu"
    else:
        raise ValueError(case)
    torch.save(value, filename)


def failure_checks(torch, np, dist, rank, trainer, model, raw, optimizer, generator, device,
                   args, train_args, directory, root, record):
    saved_next = 2
    cases = ("missing complete", "missing shard", "incomplete", "file list", "version", "stage", "size",
             "rank", "counter", "negative counter", "config", "layout", "fixed args", "device type")
    for index, case in enumerate(cases):
        damaged = root / f"damaged_{index:02d}"
        if rank == 0:
            shutil.copytree(directory, damaged)
            mutate_saved(torch, damaged, args.dp_size, case)
        dist.barrier()
        expect_rejection(torch, np, dist, rank,
            lambda: trainer.load_zero_checkpoint(checkpoint_dir=str(damaged), model=model, optimizer=optimizer,
                train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, f"load rejection: {case} (all ranks)", record)
        dist.barrier()
    too_short = dict(train_args, max_iters=saved_next - 1)
    expect_rejection(torch, np, dist, rank,
        lambda: trainer.load_zero_checkpoint(checkpoint_dir=str(directory), model=model, optimizer=optimizer,
            train_data_generator=generator, device=device, train_args=too_short),
        raw, optimizer, generator, device, args, "load rejection: max_iters before next_iter", record)
    for key in FIXED_KEYS:
        changed_args = dict(train_args)
        value = changed_args[key]
        changed_args[key] = value + "_different" if isinstance(value, str) else value + 1
        expect_rejection(torch, np, dist, rank,
            lambda options=changed_args: trainer.load_zero_checkpoint(checkpoint_dir=str(directory), model=model,
                optimizer=optimizer, train_data_generator=generator, device=device, train_args=options),
            raw, optimizer, generator, device, args, f"load rejection: changed fixed argument {key}", record)
    old_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
    expect_rejection(torch, np, dist, rank,
        lambda: trainer.save_zero_checkpoint(out_dir=str(directory.parent), model=model, optimizer=optimizer,
            next_iter=saved_next, train_data_generator=generator, device=device, train_args=train_args),
        raw, optimizer, generator, device, args, "save rejection: existing checkpoint", record)
    errors = []
    if old_hashes != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}:
        errors.append("拒绝重复保存时修改了既有 checkpoint")
    agree(dist, rank, errors, "existing checkpoint preserved", record)
    if args.dp_size > 1:
        expect_rejection(torch, np, dist, rank,
            lambda: trainer.load_zero_checkpoint(checkpoint_dir=str(directory if rank == 0 else root / "other_path"),
                model=model, optimizer=optimizer, train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, "load rejection: different paths across ranks", record)
        expect_rejection(torch, np, dist, rank,
            lambda: trainer.save_zero_checkpoint(out_dir=str(root / "different_counter"), model=model, optimizer=optimizer,
                next_iter=4 if rank == 0 else 5, train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, "save rejection: different counters across ranks", record)
        expect_rejection(torch, np, dist, rank,
            lambda: trainer.save_zero_checkpoint(out_dir=str(root / f"different_path_{rank}"), model=model,
                optimizer=optimizer, next_iter=5, train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, "save rejection: different paths across ranks", record)
    for bad_counter in (-1, True, 1.5):
        expect_rejection(torch, np, dist, rank,
            lambda value=bad_counter: trainer.save_zero_checkpoint(out_dir=str(root / "bad_counter"), model=model,
                optimizer=optimizer, next_iter=value, train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, f"save rejection: next_iter={bad_counter!r}", record)
    if rank == args.dp_size - 1:
        next(raw.parameters()).grad = torch.ones_like(next(raw.parameters()))
    expect_rejection(torch, np, dist, rank,
        lambda: trainer.save_zero_checkpoint(out_dir=str(root / "pending_gradient"), model=model, optimizer=optimizer,
            next_iter=6, train_data_generator=generator, device=device, train_args=train_args),
        raw, optimizer, generator, device, args, "save rejection: pending gradient on one rank", record)
    errors = []
    if (root / "pending_gradient" / "iter_00000006" / "complete.pt").exists():
        errors.append("存在待处理梯度的保存尝试不应写出 complete.pt")
    agree(dist, rank, errors, "pending-gradient save has no completion marker", record)
    optimizer.zero_grad(set_to_none=True)
    original_atomic = trainer.atomic_torch_save
    def failing_atomic(payload, path):
        if rank == args.dp_size - 1 and Path(path).name == f"rank_{rank:05d}.pt":
            raise OSError("verification injected shard write error")
        return original_atomic(payload, path)
    failed_root = root / "injected_write"
    with patched_attributes([(trainer, "atomic_torch_save", failing_atomic)]):
        expect_rejection(torch, np, dist, rank,
            lambda: trainer.save_zero_checkpoint(out_dir=str(failed_root), model=model, optimizer=optimizer,
                next_iter=7, train_data_generator=generator, device=device, train_args=train_args),
            raw, optimizer, generator, device, args, "save rejection: ordinary shard write exception coordinated", record)
    errors = []
    if (failed_root / "iter_00000007" / "complete.pt").exists():
        errors.append("某 rank 写入失败后仍写出了 complete.pt")
    agree(dist, rank, errors, "incomplete write has no completion marker", record)
    before = full_snapshot(torch, np, raw, optimizer, device, generator)
    recovered = trainer.save_zero_checkpoint(out_dir=str(root / "save_recovery"), model=model, optimizer=optimizer,
        next_iter=8, train_data_generator=generator, device=device, train_args=train_args)
    errors = []
    inspect_checkpoint(torch, recovered, args.dp_size, rank, 8, raw, train_args, before, args, errors)
    agree(dist, rank, errors, "healthy save after rejected write", record)


def direct_case(torch, np, dist, rank, GPT, GPTConfig, zero, trainer, group, members,
                layout_name, bias, kind, args, device, temporary, record, with_failures=False):
    label = f"functions {layout_name}, members={members}, {kind}, bias={bias}"
    group_rank = dist.get_rank(group=group)
    root = Path(temporary) / f"{layout_name}_{members[0]}_{kind}_{int(bias)}"
    data_path = Path(temporary) / "data" / "train.bin"
    torch.manual_seed(args.seed + 110 + members[0])
    model, raw, optimizer = construct_model(torch, GPT, GPTConfig, zero, args, device, group, bias, kind)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000 + rank)
    random.seed(args.seed + 30_000 + rank)
    np.random.seed(args.seed + 40_000 + rank)
    torch.manual_seed(args.seed + 20_000 + rank)
    errors = []
    optimizer_errors(torch, raw, optimizer, errors, args, initialized=False)
    optimizer_rejection_checks(torch, np, optimizer, raw, generator, device, args, errors)
    agree(dist, rank, errors, f"{label}: layout / empty-or-uninitialized state / invalid optimizer metadata", record)
    clip = args.clip_norm if bias else 0.0
    train_args = make_train_args(args, data_path.parent, bias, clip)
    for event in range(2):
        perform_step(torch, trainer, model, raw, optimizer, generator, device, data_path, args, kind, event, clip)
    before = full_snapshot(torch, np, raw, optimizer, device, generator)
    identities = {name: id(param) for name, param in raw.named_parameters()}
    aliases = alias_map(raw)
    directory = trainer.save_zero_checkpoint(out_dir=str(root / "saved"), model=model, optimizer=optimizer,
        next_iter=2, train_data_generator=generator, device=device, train_args=train_args)
    errors = []
    optimizer_errors(torch, raw, optimizer, errors, args, initialized=True)
    reference_checks(raw, optimizer, identities, aliases, errors)
    compare_snapshot(torch, full_snapshot(torch, np, raw, optimizer, device, generator), before,
                     errors, "save leaves live state unchanged", args, exact=True)
    expected_directory = (root / "saved" / "iter_00000002").resolve()
    if Path(directory).resolve() != expected_directory:
        errors.append("save_zero_checkpoint 返回了错误目录")
    inspect_checkpoint(torch, directory, len(members), group_rank, 2, raw, train_args, before, args, errors)
    agree(dist, rank, errors, f"{label}: files / weights_only load / model / owned moments / independent RNG", record)
    continuous_data = perform_step(torch, trainer, model, raw, optimizer, generator, device,
        data_path, args, kind, 2, clip)
    continuous = full_snapshot(torch, np, raw, optimizer, device, generator)
    # 重建模型和优化器，验证 load 对新 Parameter 引用和原生 AdamW 绑定有效。
    torch.manual_seed(args.seed + 999)
    restored_model, restored_raw, restored_optimizer = construct_model(torch, GPT, GPTConfig, zero,
        args, device, group, bias, kind)
    restored_generator = torch.Generator(device="cpu").manual_seed(args.seed + 888)
    random.seed(args.seed + 777)
    np.random.seed(args.seed + 666)
    torch.manual_seed(args.seed + 555)
    restored_ids = {name: id(param) for name, param in restored_raw.named_parameters()}
    restored_aliases = alias_map(restored_raw)
    for param in restored_raw.parameters():
        param.grad = torch.ones_like(param)
    load_args = dict(train_args, max_iters=30, out_dir="changed_output", log_interval=7,
                     checkpoint_interval=0, eval_interval=0, resume=str(directory))
    next_iter = trainer.load_zero_checkpoint(checkpoint_dir=directory, model=restored_model,
        optimizer=restored_optimizer, train_data_generator=restored_generator, device=device, train_args=load_args)
    errors = []
    if type(next_iter) is not int or next_iter != 2:
        errors.append("load 应返回绝对 next_iter=2")
    loaded = full_snapshot(torch, np, restored_raw, restored_optimizer, device, restored_generator)
    compare_snapshot(torch, loaded, before, errors, "load exact live state", args, exact=True)
    reference_checks(restored_raw, restored_optimizer, restored_ids, restored_aliases, errors)
    optimizer_errors(torch, restored_raw, restored_optimizer, errors, args, initialized=True)
    agree(dist, rank, errors, f"{label}: exact load / new optimizer references / aliases / all RNG", record)
    resumed_data = perform_step(torch, trainer, restored_model, restored_raw, restored_optimizer, restored_generator,
        device, data_path, args, kind, 2, clip)
    resumed = full_snapshot(torch, np, restored_raw, restored_optimizer, device, restored_generator)
    errors = []
    for key in ("x", "y"):
        check_tree(torch, resumed_data[key], continuous_data[key], errors, f"next batch.{key}", args, exact=True)
    check_tree(torch, resumed_data["result"], continuous_data["result"], errors, "next step result", args)
    compare_snapshot(torch, resumed, continuous, errors, "next update vs uninterrupted", args)
    reference_checks(restored_raw, restored_optimizer, restored_ids, restored_aliases, errors)
    agree(dist, rank, errors, f"{label}: next data / Dropout / AdamW update and moments", record)
    if with_failures:
        failure_checks(torch, np, dist, rank, trainer, restored_model, restored_raw, restored_optimizer,
            restored_generator, device, args, train_args, Path(directory), root, record)


def function_worker(rank, args, temporary):
    configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "status": "running", "checks": []}
    path = Path(temporary) / f"rank_{rank}.json"
    try:
        GPT, GPTConfig, zero, trainer = load_project(args.project)
        store = dist.FileStore(str(Path(temporary) / "function_store"), args.dp_size)
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", store=store,
            rank=rank, world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        for layout_name, parts in group_layouts(args.dp_size):
            group, members = None, None
            for ranks in parts:
                candidate = dist.new_group(ranks=ranks, timeout=timedelta(seconds=args.timeout))
                if rank in ranks:
                    group, members = candidate, ranks
            for bias in (True, False):
                direct_case(torch, np, dist, rank, GPT, GPTConfig, zero, trainer, group, members,
                    layout_name, bias, "GPT", args, device, temporary, record,
                    with_failures=(layout_name == "WORLD" and bias))
            direct_case(torch, np, dist, rank, GPT, GPTConfig, zero, trainer, group, members,
                layout_name, True, "empty_owner_probe", args, device, temporary, record)
        errors = []
        if fingerprints(args.project) != args.expected_hashes:
            errors.append("函数验证期间项目源码发生变化")
        agree(dist, rank, errors, "function checks leave project source unchanged", record)
        record["status"] = "passed"
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def event_schedule(start, maximum, interval):
    if interval <= 0:
        return []
    return [event for event in range(start, maximum)
            if (event + 1) % interval == 0 or event + 1 == maximum]


def main_specs(args):
    plans = []
    for index, bias in enumerate((True, False)):
        plans.append({"name": f"bias_{bias}", "bias": bias, "seed": args.seed + (index + 1) * 101,
            "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
            "vocab_size": args.vocab_size, "block_size": args.block_size,
            "dropout": args.dropout, "seq_length": args.seq_len, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size, "grad_clip": args.clip_norm if bias else 0.0,
            "max_iters": args.main_steps + index, "split_step": args.split_step,
            "log_interval": 2, "eval_interval": 2, "eval_iters": args.eval_iters,
            "checkpoint_interval": 2, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "beta1": args.betas[0], "beta2": args.betas[1]})
    return plans


def main_argv(spec, data_dir, out_dir, resume, device):
    result = ["--device-type", device, "--data-dir", str(data_dir), "--out-dir", str(out_dir)]
    for name in ("seed", "n_layer", "n_head", "n_embd", "vocab_size", "block_size", "dropout",
                 "seq_length", "batch_size", "micro_batch_size", "grad_clip", "max_iters", "log_interval",
                 "learning_rate", "weight_decay", "beta1", "beta2", "eval_interval", "eval_iters", "checkpoint_interval"):
        result += ["--" + name.replace("_", "-"), str(spec[name])]
    if not spec["bias"]:
        result.append("--no-bias")
    if resume is not None:
        result += ["--resume", str(resume)]
    return result


def compare_main_logs(text, rank, start, spec, trace, args, errors):
    steps = {item["event"]: item for item in trace["steps"]}
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("iter ")]
    matches = [LOG_PATTERN.fullmatch(line) for line in lines]
    expected = [event for event in range(start, spec["max_iters"])
                if event % spec["log_interval"] == 0 or event == spec["max_iters"] - 1] if rank == 0 else []
    if any(match is None for match in matches):
        errors.append("main 的迭代日志格式错误")
    else:
        indices = [int(match.group(1)) for match in matches]
        if indices != expected:
            errors.append(f"rank {rank} 完整迭代日志错误：预期 {expected}，实际 {indices}；检查恢复循环起点")
        for match in matches:
            item = steps.get(int(match.group(1)))
            if item is None:
                errors.append("日志对应没有实际执行的训练迭代")
                continue
            for label, text_value, value in (("loss", match.group(2), item["result"]["loss"]),
                                             ("grad_norm", match.group(3), item["result"]["grad_norm"])):
                if value is None:
                    if text_value != "n/a":
                        errors.append(f"日志 {label} 应为 n/a")
                    continue
                try:
                    if not math.isclose(float(text_value), float(value), rel_tol=args.rtol, abs_tol=max(args.atol, 5.1e-7)):
                        errors.append(f"日志 {label} 与实际返回值不同")
                except ValueError:
                    errors.append(f"日志 {label} 不是数值")
            if (match.group(4) == "True") is not item["result"]["stepped"]:
                errors.append("日志 stepped 与实际返回值不同")
    eval_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("eval iter ")]
    eval_matches = [EVAL_LOG_PATTERN.fullmatch(line) for line in eval_lines]
    expected_eval = event_schedule(start, spec["max_iters"], spec["eval_interval"]) if rank == 0 else []
    if any(match is None for match in eval_matches) or [int(match.group(1)) for match in eval_matches if match] != expected_eval:
        errors.append("完整评估日志的迭代位置/次数错误")
    else:
        evaluation = {(item["event"], item["split"]): item["loss"] for item in trace["evaluations"]}
        for match in eval_matches:
            for split, text_value in (("train", match.group(2)), ("val", match.group(3))):
                try:
                    wanted = evaluation.get((int(match.group(1)), split))
                    if wanted is None or not math.isclose(float(text_value), wanted, rel_tol=args.rtol, abs_tol=max(args.atol, 5.1e-7)):
                        errors.append("评估日志与实际返回值不同")
                except ValueError:
                    errors.append("评估日志不是数值")
    saved_lines = [line.strip()[len("saved checkpoint: "):] for line in text.splitlines()
                   if line.strip().startswith("saved checkpoint: ")]
    expected_saved = [item["directory"] for item in trace["saves"]] if rank == 0 else []
    if [os.path.abspath(value) for value in saved_lines] != [os.path.abspath(value) for value in expected_saved]:
        errors.append("完整保存日志与实际保存调用不一致")
    resumed_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("resumed checkpoint: ")]
    expected_resume = [f"resumed checkpoint: {trace['resume']}, next_iter={start}"] if rank == 0 and trace["resume"] else []
    if resumed_lines != expected_resume:
        errors.append("恢复日志的次数、目录或 next_iter 错误")


def main_worker(rank, args, temporary, case_root, spec, phase, port):
    configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "phase": phase, "status": "running", "checks": [], "steps": [], "saves": [], "evaluations": []}
    path = Path(temporary) / f"rank_{rank}.json"
    saved_init, saved_destroy = dist.init_process_group, dist.destroy_process_group
    start = spec["split_step"] if phase in ("resumed", "already_complete") else 0
    resume = (Path(case_root) / "interrupted" / "out" / f"iter_{spec['split_step']:08d}") if start else None
    out_dir = Path(temporary) / "out"
    data_dir = Path(case_root) / "data"
    trace = {"phase": phase, "start": start, "spec": spec, "resume": None if resume is None else str(resume),
             "steps": [], "evaluations": [], "saves": [], "loaded": None, "final": None}
    tolerance = copy.copy(args)
    tolerance.rtol, tolerance.atol = args.fp32_rtol, args.fp32_atol
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("实际 main 启动前源码发生变化")
        GPT, GPTConfig, zero, trainer = load_project(args.project)
        train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
        train_data = np.fromfile(train_path, dtype=np.uint16)
        data_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, val_path)}
        observed = {"ddp": [], "optimizers": [], "batches": [], "eval_depth": 0,
                    "init_calls": 0, "destroy_calls": 0, "load_calls": 0, "generator": None}
        original_ddp, original_zero = torch.nn.parallel.DistributedDataParallel, zero.Zero1Optimizer
        original_batch, original_train, original_eval = trainer.get_zero_batch, trainer.train_zero_step, trainer.estimate_zero_loss
        original_save, original_load = trainer.save_zero_checkpoint, trainer.load_zero_checkpoint
        expected_eval = [(event, split) for event in event_schedule(start, spec["max_iters"], spec["eval_interval"])
                         for split in ("train", "val")]
        expected_save = [event + 1 for event in event_schedule(start, spec["max_iters"], spec["checkpoint_interval"])]

        def observed_init(*positional, **keyword):
            observed["init_calls"] += 1
            bound = inspect.signature(saved_init).bind(*positional, **keyword)
            bound.apply_defaults()
            bound.arguments["timeout"] = timedelta(seconds=args.timeout)
            return saved_init(*bound.args, **bound.kwargs)

        def observed_destroy(*positional, **keyword):
            observed["destroy_calls"] += 1
            return saved_destroy(*positional, **keyword)

        class TracedDDP(original_ddp):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                observed["ddp"].append(self)
                raw, errors = self.module, []
                for name in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "dropout", "bias"):
                    if getattr(raw.config, name) != spec[name]:
                        errors.append(f"main GPTConfig.{name} 与命令行不同")
                if dist.get_world_size(group=self.process_group) != args.dp_size:
                    errors.append("main DDP 组大小错误")
                if any(param.dtype != torch.float32 or param.device != device for param in raw.parameters()):
                    errors.append("main 模型 dtype/设备错误")
                if raw.transformer.wte.weight is not raw.lm_head.weight:
                    errors.append("main 丢失 nanoGPT 的共享 embedding/lm_head")
                observed.update(model=self, raw=raw,
                    identities={name: id(param) for name, param in raw.named_parameters()}, aliases=alias_map(raw))
                agree(dist, rank, errors, f"actual main {phase}: DDP / config / shared weights", record)

        class TracedZero(original_zero):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                observed["optimizers"].append(self)
                errors = []
                if self.model is not observed.get("raw") or self.dp_size != args.dp_size or self.dp_rank != rank:
                    errors.append("main ZeRO-1 引用或组映射错误")
                optimizer_errors(torch, observed["raw"], self, errors, tolerance, initialized=False)
                agree(dist, rank, errors, f"actual main {phase}: real ZeRO-1 initial layout", record)

        @wraps(original_batch)
        def observed_batch(*positional, **keyword):
            bound = bind_call(original_batch, positional, keyword)
            if observed["eval_depth"]:
                return original_batch(*positional, **keyword)
            generator = bound["generator"]
            errors = []
            if generator.device.type != "cpu" or os.path.abspath(bound["data_path"]) != str(train_path.resolve()):
                errors.append("main 训练 generator 设备或数据路径错误")
            if bound["batch_size"] != spec["batch_size"] or bound["seq_length"] != spec["seq_length"] or bound["device"] != device:
                errors.append("main 训练 batch 尺寸或设备错误")
            if observed["generator"] is not None and generator is not observed["generator"]:
                errors.append("main 在迭代间更换训练 generator")
            if observed["generator"] is None and start == 0 and generator.initial_seed() != spec["seed"] + 10_000 + rank:
                errors.append("main 训练数据 seed 错误")
            observed["generator"] = generator
            expected_generator = torch.Generator(device="cpu")
            expected_generator.set_state(generator.get_state())
            expected_x, expected_y, starts = oracle_batch(torch, np, train_data, spec["batch_size"], spec["seq_length"], expected_generator, device)
            x, y = original_batch(*positional, **keyword)
            check_tree(torch, x, expected_x, errors, "actual main data.x", tolerance, exact=True)
            check_tree(torch, y, expected_y, errors, "actual main data.y", tolerance, exact=True)
            check_tree(torch, generator.get_state(), expected_generator.get_state(), errors, "data generator consumption", tolerance, exact=True)
            observed["batches"].append({"x": clone_tree(torch, x), "y": clone_tree(torch, y), "starts": starts})
            agree(dist, rank, errors, f"actual main {phase}: training batch {len(observed['batches']) - 1 + start}", record)
            return x, y

        @wraps(original_train)
        def observed_train(*positional, **keyword):
            bound = bind_call(original_train, positional, keyword)
            event, errors = start + len(trace["steps"]), []
            optimizer, raw = observed["optimizers"][0], observed["raw"]
            if bound["model"] is not observed["model"] or bound["optimizer"] is not optimizer:
                errors.append("main 传给 train_zero_step 的对象错误")
            if bound["micro_batch_size"] != spec["micro_batch_size"] or bound["grad_clip"] != spec["grad_clip"]:
                errors.append("main 没有传递训练 micro_batch_size/grad_clip")
            if event >= spec["max_iters"] or len(observed["batches"]) != len(trace["steps"]) + 1:
                errors.append("main 的实际训练迭代/数据调用次数错误")
            agree(dist, rank, errors, f"actual main {phase}, iter={event}: train wiring", record)
            result = original_train(*positional, **keyword)
            errors = []
            if not isinstance(result, dict) or set(result) != {"loss", "grad_norm", "stepped"}:
                errors.append("train_zero_step 返回值字段错误")
            elif type(result["stepped"]) is not bool or result["stepped"] is not True:
                errors.append("有限梯度的验证训练应成功更新")
            else:
                if not math.isfinite(float(result["loss"])) or result["grad_norm"] is None or not math.isfinite(float(result["grad_norm"])):
                    errors.append("正常训练 loss/grad_norm 应为有限值")
            reference_checks(raw, optimizer, observed["identities"], observed["aliases"], errors)
            optimizer_errors(torch, raw, optimizer, errors, tolerance, initialized=True)
            agree(dist, rank, errors, f"actual main {phase}, iter={event}: live optimizer / cleared gradients", record)
            item = {"event": event, "batch": observed["batches"][-1], "result": clone_tree(torch, result),
                    "state": full_snapshot(torch, np, raw, optimizer, device, observed["generator"])}
            trace["steps"].append(item)
            record["steps"].append({"event": event, **result})
            return result

        @wraps(original_eval)
        def observed_eval(*positional, **keyword):
            bound = bind_call(original_eval, positional, keyword)
            index, event, errors = len(trace["evaluations"]), start + len(trace["steps"]) - 1, []
            wanted = expected_eval[index] if index < len(expected_eval) else None
            if wanted is None or event != wanted[0]:
                errors.append("main 的评估顺序或迭代位置错误")
            split = wanted[1] if wanted else "train"
            if os.path.abspath(bound["data_path"]) != str((train_path if split == "train" else val_path).resolve()):
                errors.append("main 评估数据 split 错误")
            optimizer, raw = observed["optimizers"][0], observed["raw"]
            if bound["model"] is not observed["model"] or bound["dp_group"] is not optimizer.dp_group:
                errors.append("main 的评估模型/组错误")
            for name in ("batch_size", "seq_length", "micro_batch_size", "eval_iters"):
                if bound[name] != spec[name]:
                    errors.append(f"main 没有传递评估 {name}")
            offset = 20_000 if split == "train" else 30_000
            if bound["seed"] != spec["seed"] + offset or bound["device"] != device:
                errors.append("main 评估 seed 或设备错误")
            agree(dist, rank, errors, f"actual main {phase}, iter={event}: {split} eval wiring", record)
            before = full_snapshot(torch, np, raw, optimizer, device, observed["generator"])
            observed["eval_depth"] += 1
            try:
                loss = original_eval(*positional, **keyword)
            finally:
                observed["eval_depth"] -= 1
            errors = []
            compare_snapshot(torch, full_snapshot(torch, np, raw, optimizer, device, observed["generator"]),
                before, errors, "evaluation preserves live state and RNG", tolerance, exact=True)
            if type(loss) is not float or not math.isfinite(loss):
                errors.append("评估返回值不是有限 float")
            agree(dist, rank, errors, f"actual main {phase}, iter={event}: {split} eval preserves state", record)
            entry = {"event": event, "split": split, "loss": loss}
            trace["evaluations"].append(entry)
            record["evaluations"].append(entry)
            return loss

        @wraps(original_save)
        def observed_save(*positional, **keyword):
            bound = bind_call(original_save, positional, keyword)
            index, errors = len(trace["saves"]), []
            completed = start + len(trace["steps"])
            optimizer, raw = observed["optimizers"][0], observed["raw"]
            if index >= len(expected_save) or bound["next_iter"] != expected_save[index] or bound["next_iter"] != completed:
                errors.append("main 的保存 next_iter/次数错误；应记录已完成的迭代数")
            if bound["model"] is not observed["model"] or bound["optimizer"] is not optimizer or bound["train_data_generator"] is not observed["generator"]:
                errors.append("main 传给保存函数的模型/优化器/generator 错误")
            if os.path.abspath(bound["out_dir"]) != str(out_dir.resolve()) or bound["device"] != device:
                errors.append("main 保存目录或设备错误")
            due_eval = [(event, split) for event, split in expected_eval if event <= completed - 1]
            if [(item["event"], item["split"]) for item in trace["evaluations"]] != due_eval:
                errors.append("main 应在本轮评估结束后保存 checkpoint")
            agree(dist, rank, errors, f"actual main {phase}, next_iter={completed}: save order / wiring", record)
            before = full_snapshot(torch, np, raw, optimizer, device, observed["generator"])
            directory = original_save(*positional, **keyword)
            errors = []
            inspect_checkpoint(torch, directory, args.dp_size, rank, completed, raw, bound["train_args"], before, tolerance, errors)
            compare_snapshot(torch, full_snapshot(torch, np, raw, optimizer, device, observed["generator"]),
                before, errors, "main save unchanged", tolerance, exact=True)
            agree(dist, rank, errors, f"actual main {phase}, next_iter={completed}: persisted shard / RNG / complete", record)
            entry = {"next_iter": completed, "directory": str(directory)}
            trace["saves"].append(entry)
            record["saves"].append(entry)
            return directory

        @wraps(original_load)
        def observed_load(*positional, **keyword):
            bound = bind_call(original_load, positional, keyword)
            observed["load_calls"] += 1
            optimizer, raw, errors = observed["optimizers"][0], observed["raw"], []
            if resume is None or observed["load_calls"] != 1 or trace["steps"] or observed["batches"]:
                errors.append("main 应仅在恢复启动时、取 batch 前调用一次 load")
            if resume is None or os.path.abspath(bound["checkpoint_dir"]) != str(resume.resolve()):
                errors.append("main 传给 load 的恢复路径错误")
            if bound["model"] is not observed["model"] or bound["optimizer"] is not optimizer or bound["device"] != device:
                errors.append("main 传给 load 的模型/优化器/设备错误")
            observed["generator"] = bound["train_data_generator"]
            agree(dist, rank, errors, f"actual main {phase}: load before training", record)
            next_iter = original_load(*positional, **keyword)
            errors = []
            if type(next_iter) is not int or next_iter != start:
                errors.append("main 恢复 next_iter 不正确")
            reference_checks(raw, optimizer, observed["identities"], observed["aliases"], errors)
            loaded = full_snapshot(torch, np, raw, optimizer, device, observed["generator"])
            common = torch.load(resume / "model.pt", map_location="cpu", weights_only=True)
            shard = torch.load(resume / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
            check_tree(torch, loaded["model"], common["model"], errors, "main exact loaded model", tolerance, exact=True)
            check_tree(torch, loaded["optimizer"], shard["optimizer"], errors, "main exact loaded optimizer", tolerance, exact=True)
            check_tree(torch, loaded["rng"], shard["rng"], errors, "main exact loaded RNG", tolerance, exact=True)
            agree(dist, rank, errors, f"actual main {phase}: loaded weights / moments / RNG / preserved references", record)
            trace["loaded"] = loaded
            return next_iter

        # Python/NumPy 未在训练 main 中使用，但其保存/恢复也必须正确。
        # 恢复进程故意使用不同的初始值，成功恢复后应回到保存时状态。
        random.seed(spec["seed"] + 50_000 + rank + (99_000 if start else 0))
        np.random.seed(spec["seed"] + 60_000 + rank + (99_000 if start else 0))
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(args.dp_size),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
        changes = [(dist, "init_process_group", observed_init), (dist, "destroy_process_group", observed_destroy),
                   (torch.nn.parallel, "DistributedDataParallel", TracedDDP), (zero, "Zero1Optimizer", TracedZero),
                   (trainer, "get_zero_batch", observed_batch), (trainer, "train_zero_step", observed_train),
                   (trainer, "estimate_zero_loss", observed_eval), (trainer, "save_zero_checkpoint", observed_save),
                   (trainer, "load_zero_checkpoint", observed_load)]
        for name, value in (("DDP", TracedDDP), ("Zero1Optimizer", TracedZero)):
            if hasattr(trainer, name):
                changes.append((trainer, name, value))
        buffer = io.StringIO()
        try:
            with patched_attributes(changes), command_line(main_argv(spec, data_dir, out_dir, resume, args.device)), redirect_stdout(buffer):
                trainer.main()
        finally:
            captured = buffer.getvalue()
            record["stdout"] = captured
            if rank == 0 or "[FAIL" in captured:
                print(captured, end="", flush=True)
        # 完整日志仅在真实 main 正常返回以后验证，避免漏掉末次打印。
        errors = []
        record["main_destroyed_group"] = not dist.is_initialized()
        if dist.is_initialized():
            errors.append("main 正常返回后没有销毁进程组")
            saved_destroy()
        if observed["init_calls"] != 1 or observed["destroy_calls"] != 1:
            errors.append("main 初始化/销毁次数错误")
        if len(observed["ddp"]) != 1 or len(observed["optimizers"]) != 1:
            errors.append("main 应创建一组真实 DDP 和 ZeRO-1")
        if observed["load_calls"] != (1 if resume else 0):
            errors.append("main 的恢复调用次数错误")
        if [item["event"] for item in trace["steps"]] != list(range(start, spec["max_iters"])):
            errors.append("main 实际训练的绝对迭代范围错误")
        if len(observed["batches"]) != spec["max_iters"] - start:
            errors.append("main 实际取训练 batch 次数错误")
        if [(item["event"], item["split"]) for item in trace["evaluations"]] != expected_eval:
            errors.append("main 完整评估调用序列错误")
        if [item["next_iter"] for item in trace["saves"]] != expected_save:
            errors.append("main 完整 checkpoint 调用序列错误")
        if observed["generator"] is not None and observed["optimizers"]:
            trace["final"] = full_snapshot(torch, np, observed["raw"], observed["optimizers"][0], device, observed["generator"])
        else:
            errors.append("main 未建立用于训练/恢复的 generator")
        for filename, digest in data_hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                errors.append("main 修改了验证数据文件")
        compare_main_logs(captured, rank, start, spec, trace, tolerance, errors)
        if fingerprints(args.project) != args.expected_hashes:
            errors.append("实际 main 验证期间项目源码发生变化")
        store = dist.FileStore(str(Path(temporary) / "post_main_store"), args.dp_size)
        saved_init("nccl" if args.device == "cuda" else "gloo", store=store, rank=rank,
            world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        agree(dist, rank, errors, f"actual main {phase}: full logs / absolute iterations / save and eval schedule / cleanup", record)
        torch.save(trace, Path(temporary) / f"trace_{rank:05d}.pt")
        record.update(status="passed", init_calls=observed["init_calls"], destroy_calls=observed["destroy_calls"])
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            saved_destroy()


def compare_main_runs(torch, case_root, spec, args, include_noop):
    errors = []
    tolerance = copy.copy(args)
    tolerance.rtol, tolerance.atol = args.fp32_rtol, args.fp32_atol
    for rank in range(args.dp_size):
        traces = {phase: torch.load(case_root / phase / f"trace_{rank:05d}.pt", map_location="cpu", weights_only=True)
                  for phase in ("continuous", "interrupted", "resumed")}
        continuous, interrupted, resumed = (traces[phase] for phase in ("continuous", "interrupted", "resumed"))
        all_steps = {item["event"]: item for item in continuous["steps"]}
        if list(all_steps) != list(range(spec["max_iters"])):
            errors.append(f"rank {rank}: 连续训练的轨迹不完整")
            continue
        if [item["event"] for item in interrupted["steps"]] != list(range(spec["split_step"])):
            errors.append(f"rank {rank}: 中途训练的轨迹不完整")
        if [item["event"] for item in resumed["steps"]] != list(range(spec["split_step"], spec["max_iters"])):
            errors.append(f"rank {rank}: 恢复训练的轨迹不完整")
        for phase in ("interrupted", "resumed"):
            for step in traces[phase]["steps"]:
                expected = all_steps.get(step["event"])
                if expected is None:
                    errors.append(f"rank {rank}: {phase} 轨迹出现无效迭代")
                    continue
                label = f"rank {rank}, {phase}, iter={step['event']}"
                check_tree(torch, step["batch"], expected["batch"], errors, label + ".data", tolerance, exact=True)
                check_tree(torch, step["result"], expected["result"], errors, label + ".loss/norm/step", tolerance)
                compare_snapshot(torch, step["state"], expected["state"], errors, label + ".state", tolerance)
        compare_snapshot(torch, resumed["loaded"], interrupted["final"], errors,
                         f"rank {rank}: restart exact saved boundary", tolerance, exact=True)
        compare_snapshot(torch, resumed["final"], continuous["final"], errors,
                         f"rank {rank}: final model / moments / RNG", tolerance)
        if include_noop:
            noop = torch.load(case_root / "already_complete" / f"trace_{rank:05d}.pt", map_location="cpu", weights_only=True)
            if noop["steps"] or noop["saves"] or noop["evaluations"]:
                errors.append(f"rank {rank}: 已完成恢复重复执行了训练/评估/保存")
            compare_snapshot(torch, noop["final"], interrupted["final"], errors,
                             f"rank {rank}: already-complete load", tolerance, exact=True)
        # 实际 main 写出的最终文件，不能只比较内存观察值。
        for phase in ("continuous", "resumed"):
            final_dir = case_root / phase / "out" / f"iter_{spec['max_iters']:08d}"
            common = torch.load(final_dir / "model.pt", map_location="cpu", weights_only=True)
            shard = torch.load(final_dir / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
            expected = traces[phase]["final"]
            for key, actual in (("model", common["model"]), ("optimizer", shard["optimizer"]), ("rng", shard["rng"])):
                check_tree(torch, actual, expected[key], errors, f"rank {rank}: {phase} final file.{key}", tolerance, exact=True)
        eval_continuous = {(item["event"], item["split"]): item["loss"] for item in continuous["evaluations"]}
        for item in resumed["evaluations"]:
            label = (item["event"], item["split"])
            if label not in eval_continuous:
                errors.append(f"rank {rank}: 恢复后的评估没有连续参考")
            else:
                check_tree(torch, item["loss"], eval_continuous[label], errors, f"rank {rank}: resumed eval {label}", tolerance)
    if errors:
        for error in errors[:24]:
            print(f"[FAIL] {error}", flush=True)
        raise AssertionError("实际 main 连续训练与新进程恢复结果不同：" + "; ".join(errors[:4]))
    print(f"[PASS] actual main {spec['name']}: fresh-process resume vs uninterrupted; data / losses / parameters / moments / RNG", flush=True)
    return {"status": "passed", "bias": spec["bias"], "dropout": spec["dropout"],
            "total_iterations": spec["max_iters"], "split_next_iter": spec["split_step"],
            "already_complete_checked": include_noop,
            "comparison": "exact serialized reload/data/RNG; tolerance for recomputed FP32 trajectory"}


def serial_checks(torch, trainer, args, temporary):
    directory = Path(temporary)
    payload = {"version": 1, "tensor": torch.arange(5), "nested": [True, None, ("x", 2)]}
    target = directory / "atomic.pt"
    trainer.atomic_torch_save(payload, target)
    errors = []
    check_tree(torch, torch.load(target, map_location="cpu", weights_only=True), payload, errors, "atomic initial file", args, exact=True)
    previous = target.read_bytes()
    def broken_save(value, handle, *positional, **keyword):
        handle.write(b"verification partial write")
        raise OSError("verification injected torch.save failure")
    with patched_attributes([(torch, "save", broken_save)]):
        try:
            trainer.atomic_torch_save({"replacement": True}, target)
        except OSError:
            pass
        else:
            errors.append("atomic_torch_save 未抛出注入的写入异常")
    if target.read_bytes() != previous or Path(str(target) + ".tmp").exists():
        errors.append("原子写入失败后破坏旧文件或留下 .tmp")
    replacement = {"replacement": torch.tensor([9, 8])}
    trainer.atomic_torch_save(replacement, target)
    check_tree(torch, torch.load(target, map_location="cpu", weights_only=True), replacement, errors, "atomic healthy replacement", args, exact=True)
    with command_line([]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        defaults = trainer.parse_args()
    if not getattr(defaults, "out_dir", "").strip() or getattr(defaults, "checkpoint_interval", None) != 0 or getattr(defaults, "resume", "missing") is not None:
        errors.append("parse_args 需要非空 out_dir、checkpoint_interval=0、resume=None")
    for flags in (["--checkpoint-interval", "-1"], ["--out-dir", ""], ["--resume", ""]):
        try:
            with command_line(flags), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                trainer.parse_args()
        except ValueError:
            pass
        except SystemExit as exc:
            if exc.code != 2:
                errors.append(f"CLI 非法参数返回了非预期 exit code: {exc.code}")
        else:
            errors.append(f"CLI 没有拒绝 {flags}")
    if errors:
        raise AssertionError("串行检查失败：" + "; ".join(errors[:8]))
    print("[PASS] atomic save / failed-write cleanup / replacement / checkpoint CLI guards", flush=True)
    return {"status": "passed", "atomic_save_checked": True, "checkpoint_cli_checked": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--clip-norm", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--fp32-rtol", type=float, default=3e-4)
    parser.add_argument("--fp32-atol", type=float, default=3e-6)
    parser.add_argument("--main-steps", type=int, default=5)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("dp_size", "threads", "timeout", "n_layer", "n_head", "n_embd", "vocab_size", "block_size",
                 "seq_len", "batch_size", "micro_batch_size", "eval_iters", "main_steps", "split_step"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not 1 <= args.split_step < args.main_steps or args.main_steps < 3:
        parser.error("需要 main_steps>=3 且 1<=split_step<main_steps")
    if not 2 <= args.vocab_size <= 65536 or args.n_embd % args.n_head or args.seq_len > args.block_size:
        parser.error("需要 2<=vocab_size<=65536、n_embd 能被 n_head 整除、seq_len<=block_size")
    if not math.isfinite(args.dropout) or not 0 < args.dropout < 1:
        parser.error("--dropout 必须在 (0,1) 内，才能检验训练 RNG 恢复")
    for name in ("learning_rate", "clip_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    for name in ("weight_decay", "rtol", "atol", "fp32_rtol", "fp32_atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负的有限数")
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in args.betas):
        parser.error("betas 必须在 [0,1) 内")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(args.project), "device": args.device, "world_size": args.dp_size, "threads": args.threads,
        "function_dtype": "float64", "main_dtype": "float32", "dropout": args.dropout,
        "group_layouts": group_layouts(args.dp_size), "function_workers": [], "actual_main": [],
        "scope": "ZeRO-1 checkpoint with fixed DP/layout/data/device type; actual fresh-process main resume; no AMP/compile/multinode/performance/hard-crash guarantee",
        "numeric_policy": "exact data/RNG/serialized reload; float64 and FP32 tolerances for recomputed training"}
    try:
        args.expected_hashes = preflight(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / checkpoint and optimizer signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only"
            print("[INFO] Static checks only; numerical verification was NOT run.", flush=True)
            return 0
        import numpy as np
        import torch
        import torch.multiprocessing as mp
        report.update(torch_version=str(torch.__version__), numpy_version=str(np.__version__))
        if not torch.distributed.is_available():
            raise RuntimeError("当前 PyTorch 没有分布式支持")
        if args.device == "cpu" and not torch.distributed.is_gloo_available():
            raise RuntimeError("CPU 验证需要 Gloo")
        if args.device == "cuda" and (not torch.cuda.is_available() or not torch.distributed.is_nccl_available() or torch.cuda.device_count() < args.dp_size):
            raise RuntimeError("CUDA 验证需要 NCCL，且单机设备数至少为 --dp-size；也可使用 --device cpu")
        torch.set_num_threads(args.threads)
        _, _, _, trainer = load_project(args.project)
        print(f"[INFO] {args.dp_size} {args.device} workers per run; threads={args.threads}; dropout={args.dropout}.", flush=True)
        print("[INFO] Checkpoint functions + fresh-process actual main; generating temporary data automatically.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_zero1_checkpoint_") as temporary:
            root = Path(temporary)
            report["serial"] = serial_checks(torch, trainer, args, root)
            functions = root / "functions"
            (functions / "data").mkdir(parents=True)
            rng = np.random.default_rng(args.seed)
            tokens = rng.integers(0, args.vocab_size, size=max(257, args.seq_len + 101), dtype=np.uint16)
            tokens.tofile(functions / "data" / "train.bin")
            try:
                mp.spawn(function_worker, args=(args, str(functions)), nprocs=args.dp_size, join=True)
            finally:
                report["function_workers"] = collect_workers(functions, args.dp_size)
            require_all_workers(report["function_workers"], args.dp_size, "checkpoint functions")
            for index, spec in enumerate(main_specs(args)):
                case_root = root / spec["name"]
                data_dir = case_root / "data"
                data_dir.mkdir(parents=True)
                rng = np.random.default_rng(args.seed + 70_000 + index)
                for name in ("train.bin", "val.bin"):
                    rng.integers(0, spec["vocab_size"], size=max(311, args.seq_len + 103), dtype=np.uint16).tofile(data_dir / name)
                entry = {"case": spec, "phases": []}
                report["actual_main"].append(entry)
                phases = ["continuous", "interrupted", "resumed"] + (["already_complete"] if index == 0 else [])
                for phase in phases:
                    phase_dir = case_root / phase
                    phase_dir.mkdir()
                    phase_spec = dict(spec)
                    if phase in ("interrupted", "already_complete"):
                        phase_spec["max_iters"] = spec["split_step"]
                    phase_record = {"phase": phase, "workers": []}
                    entry["phases"].append(phase_record)
                    print(f"[CASE] actual main {spec['name']}: {phase}, max_iters={phase_spec['max_iters']}", flush=True)
                    try:
                        mp.spawn(main_worker, args=(args, str(phase_dir), str(case_root), phase_spec, phase, unused_port()),
                            nprocs=args.dp_size, join=True)
                    finally:
                        phase_record["workers"] = collect_workers(phase_dir, args.dp_size)
                    require_all_workers(phase_record["workers"], args.dp_size, f"actual main {phase}")
                entry["comparison"] = compare_main_runs(torch, case_root, spec, args, include_noop=(index == 0))
        report["status"] = "passed"
        print(f"[PASS] ZeRO-1 checkpoint verification completed ({args.device}, dp_size={args.dp_size}).", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if args.report is not None:
            write_json(args.report, report)
            print(f"[INFO] Report saved: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
