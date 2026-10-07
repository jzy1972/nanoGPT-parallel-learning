#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证 nanoGPT 手写 ZeRO-1 的评估函数及启用评估的实际 main。

放在 model.py、zero1.py、train_zero.py 同级，运行：
  python verify_zero1_eval.py --project . --device cpu --dp-size 4 --threads 1 --report zero1_eval_report.json

只依赖标准库、NumPy、PyTorch；不需要旧验证脚本，不修改项目源码。
临时生成 uint16 train.bin/val.bin。实际调用 estimate_zero_loss()、
get_zero_batch()、train_zero_step()、parse_args() 和 main()。

函数检查：float64；GPT bias=True/False、非零 Dropout，以及具有可解析
token loss 的小模型。使用 WORLD、单 rank 子组和奇偶子组，覆盖最后一个
较小的 micro-batch、不同本地 token 数、重复评估、裸模型/DDP 包装，
检查无梯度、eval 模式、token 加权汇总、组内种子、真实数据抽样、模式
恢复、已有梯度/参数/AdamW 状态/全局 RNG 不变，以及异常后的模式恢复。

实际 main：FP32、dropout=0；使用真实 env:// 初始化和真实 DDP/ZeRO-1。
每个配置全新进程，检查关闭评估、周期评估、最后一轮评估及不重复调用，
rank 0 日志、训练数据 generator 状态及训练更新。等待 main 正常结束后
核对完整日志。FP32 训练先独立核对全局 dense 前向/梯度，普通 AdamW
参考再使用已核对的同步梯度，避免近零梯度误差经 eps 放大。

CPU 使用 Gloo；CUDA 使用 NCCL、单机每 rank 一个设备。--dp-size 是验证
进程数，不限定实现规模。使用线程限制、127.0.0.1、USE_LIBUV=0 兼容
Windows CPU 环境。--preflight-only 不导入 PyTorch，仅做静态检查。
不覆盖 checkpoint/恢复、AMP、torch.compile、多机启动和性能测量。
"""

import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
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
import socket
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout


METHODS = {
    "__init__": ("self", "model", "learning_rate", "weight_decay", "beta1", "beta2", "dp_group"),
    "zero_grad": ("self", "set_to_none"),
    "sync_parameters": ("self",),
    "step": ("self", "grad_clip"),
    "has_nonfinite_gradients": ("self",),
    "clip_gradients": ("self", "max_norm"),
}
TRAINER_SIGNATURES = {
    "parse_args": (),
    "get_zero_batch": ("data_path", "batch_size", "seq_length", "device", "generator"),
    "train_zero_step": ("model", "optimizer", "x", "y", "micro_batch_size", "grad_clip"),
    "estimate_zero_loss": ("model", "data_path", "batch_size", "seq_length", "micro_batch_size", "eval_iters", "device", "dp_group", "seed"),
    "main": (),
}
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


def check_tensor(torch, actual, expected, errors, label, args, exact=False):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            errors.append(f"{label}: 梯度/张量是否为 None 不一致")
        return 0.0
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        errors.append(f"{label}: 应为 Tensor")
        return 0.0
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    try:
        torch.testing.assert_close(actual, expected, rtol=0 if exact else args.rtol,
                                   atol=0 if exact else args.atol, equal_nan=True)
    except AssertionError as exc:
        errors.append(f"{label}: {str(exc)[:1000]}")
    if actual.shape == expected.shape and actual.numel() and actual.is_floating_point() and expected.is_floating_point():
        difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        if torch.isfinite(difference).all():
            return float(difference.max().item())
    return 0.0


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


def snapshot(torch, raw, optimizer):
    return {"model": clone_tree(torch, raw.state_dict()),
            "optimizer": None if optimizer.local_optimizer is None else clone_tree(torch, optimizer.local_optimizer.state_dict())}


def alias_map(raw):
    canonical = {id(param): name for name, param in raw.named_parameters()}
    return {name: canonical[id(param)] for name, param in raw.named_parameters(remove_duplicate=False)}


def model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors):
    state, expected = raw.state_dict(), dense.state_dict()
    largest = 0.0
    if state.keys() != expected.keys():
        errors.append("模型 state_dict 键不同")
    else:
        for name in state:
            largest = max(largest, check_tensor(torch, state[name], expected[name], errors, f"model.{name}", args))
    current = dict(raw.named_parameters())
    if {name: id(param) for name, param in current.items()} != identities:
        errors.append("更新替换了 Parameter 对象")
    all_params = dict(raw.named_parameters(remove_duplicate=False))
    if alias_map(raw) != aliases or any(all_params[name] is not all_params[canonical] for name, canonical in aliases.items()):
        errors.append("共享参数引用关系改变")
    for name, param in optimizer.named_params:
        if current.get(name) is not param:
            errors.append(f"优化器没有引用模型原参数：{name}")
    return largest


def make_reference_optimizer(torch, dense, args):
    groups = []
    for decayed in (True, False):
        params = [param for param in dense.parameters() if param.requires_grad and (param.dim() >= 2) == decayed]
        if params:
            groups.append({"params": params, "weight_decay": args.weight_decay if decayed else 0.0})
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=tuple(args.betas), eps=1e-8,
                             weight_decay=args.weight_decay, foreach=False, fused=False)


def state_errors(torch, raw, dense, optimizer, reference, args, errors, initialized=True):
    names = {id(param): name for name, param in raw.named_parameters()}
    wanted = {name: param for name, param in raw.named_parameters() if param.requires_grad and optimizer.owners[name] == optimizer.dp_rank}
    actual = optimizer.local_optimizer
    if not wanted:
        if actual is not None:
            errors.append("空分片 rank 的 local_optimizer 应为 None")
        return 0
    if not isinstance(actual, torch.optim.AdamW):
        errors.append("本地优化器应为 AdamW")
        return 0
    parameters = [param for group in actual.param_groups for param in group["params"]]
    if len(parameters) != len({id(param) for param in parameters}) or {id(param) for param in parameters} != {id(param) for param in wanted.values()}:
        errors.append("本地 AdamW 参数集合包含重复/非归属参数，或遗漏归属参数")
    for group in actual.param_groups:
        for key, value in {"lr": args.learning_rate, "betas": tuple(args.betas), "eps": 1e-8,
                           "foreach": False, "fused": False, "amsgrad": False}.items():
            if group.get(key) != value:
                errors.append(f"AdamW 参数组 {key} 不符合参考配置")
        for param in group["params"]:
            if group["weight_decay"] != (args.weight_decay if param.dim() >= 2 else 0.0):
                errors.append("weight_decay 分组不同于原版 nanoGPT")
    expected_ids = {id(param) for param in wanted.values()} if initialized else set()
    if {id(param) for param in actual.state} != expected_ids:
        errors.append("AdamW 状态没有恰好按参数归属分片")
    dense_params = dict(dense.named_parameters())
    moments = 0
    for param, state in actual.state.items():
        name = names.get(id(param))
        if name not in wanted:
            errors.append("本地状态属于非归属 Parameter")
            continue
        expected_state = reference.state.get(dense_params[name], {})
        check_tree(torch, state, expected_state, errors, f"AdamW.{name}", args)
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or state[key].shape != param.shape or state[key].dtype != param.dtype:
                errors.append(f"{name}.{key} 形状/dtype 错误")
            else:
                moments += state[key].numel()
    if initialized and moments != 2 * sum(param.numel() for param in wanted.values()):
        errors.append("本地一阶/二阶矩数量不符合分片大小")
    return moments


def partition_checks(torch, dist, rank, raw, zero, optimizer, reference, dense, group, members, args, label, record):
    errors = []
    named, owners, numels = zero.build_parameter_partition(raw, len(members))
    canonical = [(name, param) for name, param in raw.named_parameters() if param.requires_grad]
    for entries, description in ((named, "分配函数"), (optimizer.named_params, "优化器")):
        if len(entries) != len(canonical) or any(name != wanted_name or param is not wanted for (name, param), (wanted_name, wanted) in zip(entries, canonical)):
            errors.append(f"{description}没有按 canonical 顺序引用全部独立可训练参数")
    if owners.keys() != dict(canonical).keys() or any(not isinstance(owner, int) or not 0 <= owner < len(members) for owner in owners.values()):
        errors.append("owners 键或组内 rank 非法")
    else:
        expected_numels = [sum(param.numel() for name, param in canonical if owners[name] == owner) for owner in range(len(members))]
        if numels != expected_numels:
            errors.append("rank_numels 与实际归属不符")
    if optimizer.owners != owners or optimizer.rank_numels != numels or optimizer.dp_rank != members.index(rank) or optimizer.dp_size != len(members):
        errors.append("优化器分配表或 DP rank/size 不一致")
    expected_local = [(name, param) for name, param in canonical if owners.get(name) == members.index(rank)]
    if len(optimizer.local_named_params) != len(expected_local) or any(name != wanted_name or param is not wanted for (name, param), (wanted_name, wanted) in zip(optimizer.local_named_params, expected_local)):
        errors.append("local_named_params 的参数集合、顺序或引用错误")
    signature = {"names": [(name, tuple(param.shape), str(param.dtype)) for name, param in named],
                 "owners": owners, "rank_numels": numels}
    signatures = [None] * len(members)
    dist.all_gather_object(signatures, signature, group=group)
    if any(item != signatures[0] for item in signatures):
        errors.append("DP 组内各 rank 的参数顺序、形状或归属不同")
    state_errors(torch, raw, dense, optimizer, reference, args, errors, initialized=False)
    agree(dist, rank, errors, f"{label}: partition / aliases / local AdamW", record)
    return {"members": members, "dp_rank": members.index(rank), "rank_numels": numels,
            "local_parameter_names": [name for name, _ in expected_local],
            "local_parameter_numels": sum(param.numel() for _, param in expected_local),
            "empty_optimizer": optimizer.local_optimizer is None}


def reference_norm(torch, dense):
    gradients = [param.grad.detach().to(torch.float64).reshape(-1) for param in dense.parameters() if param.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(gradients)).item()) if gradients else 0.0


def check_number(actual, expected, errors, label, args):
    try:
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=args.rtol, abs_tol=args.atol):
            errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")
    except (TypeError, ValueError):
        errors.append(f"{label}: 返回值不可转为数值")


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


def check_tree(torch, actual, expected, errors, label, args, exact=False):
    if isinstance(expected, torch.Tensor):
        check_tensor(torch, actual, expected, errors, label, args, exact)
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
    elif isinstance(actual, torch.Tensor) or type(actual) is not type(expected) or actual != expected:
        errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")


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


def captured_gradients(torch, raw):
    return {name: None if param.grad is None else param.grad.detach().cpu().clone()
            for name, param in raw.named_parameters()}


def compare_gradients(torch, captured, dense, args, errors, label):
    wanted = dict(dense.named_parameters())
    if captured.keys() != wanted.keys():
        errors.append(f"{label}: 参数名称不同")
        return
    for name, param in wanted.items():
        check_tensor(torch, captured[name], param.grad, errors, f"{label}.{name}", args)


def number_for_report(value):
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None if value is None else repr(value)
    return converted if math.isfinite(converted) else str(converted)


def execute_step(torch, dist, rank, trainer_function, model, raw, dense, optimizer, reference,
                 local_x, local_y, expected_x, expected_y, full_x, full_y, micro, clip,
                 kind, group, members, affected, anomaly, identities, aliases, args, label, record):
    errors = []
    check_tensor(torch, local_x, expected_x, errors, "local data.x", args, exact=True)
    check_tensor(torch, local_y, expected_y, errors, "local data.y", args, exact=True)
    dense.train()
    dense.zero_grad(set_to_none=True)
    dense_output, dense_loss = dense(full_x, full_y)
    dense_loss.backward()
    expected_norm = reference_norm(torch, dense)
    before = snapshot(torch, raw, optimizer)
    # 人为留下旧梯度，检查实际训练函数在入口清理它们。
    for param in raw.parameters():
        param.grad = torch.full_like(param, 37.0)
    model.eval()
    forward_events, backward_flags, step_events, zero_events = [], [], [], []
    original_step, original_zero = optimizer.step, optimizer.zero_grad
    expected_skip = bool(anomaly is not None and affected)

    def forward_hook(module, inputs, output):
        forward_events.append({"synchronize": bool(module.require_backward_grad_sync),
                               "x": clone_tree(torch, inputs[0]), "y": clone_tree(torch, inputs[1]),
                               "output": clone_tree(torch, output[0]), "loss": clone_tree(torch, output[1])})

    def backward_hook(gradient):
        backward_flags.append(bool(model.require_backward_grad_sync))
        return gradient

    def observed_zero(set_to_none=True):
        zero_events.append(bool(set_to_none))
        return original_zero(set_to_none=set_to_none)

    def observed_step(grad_clip=0.0):
        item = {"grad_clip": float(grad_clip), "before_gradients": captured_gradients(torch, raw)}
        if expected_skip and rank == members[-1]:
            candidates = [(name, param) for name, param in optimizer.named_params
                          if optimizer.owners[name] != optimizer.dp_rank]
            target = (candidates or optimizer.named_params)[0][1]
            target.grad.reshape(-1)[0] = {"nan": float("nan"), "+inf": float("inf"),
                                         "-inf": -float("inf"), "norm_overflow": 1e200}[anomaly]
        item["returned"] = original_step(grad_clip=grad_clip)
        item["norm"] = optimizer.last_grad_norm
        item["after_gradients"] = captured_gradients(torch, raw)
        step_events.append(item)
        return item["returned"]

    first_parameter = next(param for param in raw.parameters() if param.requires_grad)
    handles = [model.register_forward_hook(forward_hook), first_parameter.register_hook(backward_hook)]
    try:
        with patched_attributes([(optimizer, "step", observed_step), (optimizer, "zero_grad", observed_zero)]):
            result = trainer_function(model=model, optimizer=optimizer, x=local_x, y=local_y,
                                      micro_batch_size=micro, grad_clip=clip)
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(result, dict):
        errors.append("train_zero_step 必须返回 dict")
        result = {}
    check_number(result.get("loss"), dense_loss.item(), errors, "global loss", args)
    count = expected_x.size(0)
    chunks = [(start, min(start + micro, count)) for start in range(0, count, micro)]
    expected_flags = [False] * (len(chunks) - 1) + [True]
    if [item["synchronize"] for item in forward_events] != expected_flags:
        errors.append("forward 的 no_sync 标志/次数错误；前向也必须在上下文内")
    if backward_flags != expected_flags:
        errors.append("backward 的 no_sync 标志/次数错误")
    base = members.index(rank) * count
    for item, (start, end) in zip(forward_events, chunks):
        check_tensor(torch, item["x"], expected_x[start:end], errors, "micro input.x", args, exact=True)
        check_tensor(torch, item["y"], expected_y[start:end], errors, "micro input.y", args, exact=True)
        expected_output = dense_output.detach()[base + start:base + end]
        check_tensor(torch, item["output"], expected_output, errors, "micro forward", args)
        expected_loss = (torch.nn.functional.mse_loss(expected_output, expected_y[start:end]) if kind == "toy"
                         else torch.nn.functional.cross_entropy(expected_output.reshape(-1, expected_output.size(-1)),
                                                                expected_y[start:end].reshape(-1)))
        check_tensor(torch, item["loss"], expected_loss, errors, "micro unweighted loss", args)
    if len(step_events) != 1:
        errors.append(f"每个本地 batch 应调用一次 step，实际 {len(step_events)} 次")
    if zero_events != [True, True]:
        errors.append(f"入口及更新/跳过后应清理完整梯度，实际 zero_grad 调用 {zero_events}")
    if not model.training or not raw.training:
        errors.append("train_zero_step 没有切换到训练模式")
    if any(param.grad is not None for param in raw.parameters()):
        errors.append("train_zero_step 返回时完整模型的梯度没有清空")
    if step_events:
        event = step_events[0]
        check_number(event["grad_clip"], clip, errors, "step grad_clip", args)
        compare_gradients(torch, event["before_gradients"], dense, args, errors, "DDP accumulated gradients")
        if event["returned"] is not (not expected_skip) or result.get("stepped") is not (not expected_skip):
            errors.append("更新/跳过返回值错误，或 DP 子组没有隔离异常")
        if expected_skip:
            if event["norm"] is not None or result.get("grad_norm") is not None:
                errors.append("异常跳过的 grad_norm 应为 None")
            check_tree(torch, snapshot(torch, raw, optimizer), before, errors, "skipped parameters/AdamW unchanged", args, exact=True)
        else:
            check_number(event["norm"], expected_norm, errors, "optimizer pre-clip norm", args)
            check_number(result.get("grad_norm"), expected_norm, errors, "returned pre-clip norm", args)
            if getattr(args, "reference_from_validated_gradients", False):
                # 前面已与独立全局 dense 梯度核对；FP32 优化器对照使用这些同步梯度。
                # FP64 函数测试不进入此分支，保留完整的独立 dense AdamW 轨迹。
                for name, param in dense.named_parameters():
                    gradient = event["before_gradients"][name]
                    param.grad = None if gradient is None else gradient.to(device=param.device, dtype=param.dtype)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(dense.parameters(), clip, foreach=False)
            compare_gradients(torch, event["after_gradients"], dense, args, errors, "post-clip gradients")
            reference.step()
    difference = model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors)
    moments = state_errors(torch, raw, dense, optimizer, reference, args, errors)
    agree(dist, rank, errors, f"{label}: data / micro-batches / DDP / loss / AdamW / cleanup", record)
    details = {"loss": number_for_report(result.get("loss")), "grad_norm": number_for_report(result.get("grad_norm")),
               "stepped": result.get("stepped"), "micro_batch_size": micro, "grad_clip": clip,
               "anomaly": anomaly, "affected_group": affected, "max_parameter_error": difference,
               "local_moment_numels": moments, "forward_sync_flags": expected_flags}
    return details, result


def setup_worker(torch, args, rank):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def compare_logs(text, rank, spec, steps, args, errors):
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("iter ")]
    parsed = [LOG_PATTERN.fullmatch(line) for line in lines]
    if any(match is None for match in parsed):
        errors.append("main 的迭代日志格式错误")
        return
    expected = [event for event in range(spec["max_iters"])
                if event % spec["log_interval"] == 0 or event == spec["max_iters"] - 1] if rank == 0 else []
    if [int(match.group(1)) for match in parsed] != expected:
        errors.append(f"rank {rank} 的完整迭代日志错误，应为 {expected}")
    for match in parsed:
        event = int(match.group(1))
        if event >= len(steps):
            errors.append("日志引用没有实际训练的迭代")
            continue
        step = steps[event]
        for label, text_value, value in (("loss", match.group(2), step["loss"]),
                                          ("grad_norm", match.group(3), step["grad_norm"])):
            try:
                converted = float(text_value)
                if value is None or not math.isclose(converted, float(value), rel_tol=args.rtol,
                                                    abs_tol=max(args.atol, 5.1e-7)):
                    errors.append(f"iter {event} 日志 {label} 与实际返回值不同")
            except (TypeError, ValueError):
                errors.append(f"iter {event} 日志 {label} 不是正常数值")
        if (match.group(4) == "True") is not step["stepped"]:
            errors.append(f"iter {event} 日志 stepped 错误")


def unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
    parser.add_argument("--eval-iters", type=int, default=3)
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("dp_size", "threads", "timeout", "n_layer", "n_head", "n_embd", "vocab_size", "block_size",
                 "seq_len", "batch_size", "micro_batch_size", "eval_iters", "main_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not 2 <= args.vocab_size <= 65536 or args.n_embd % args.n_head or args.seq_len > args.block_size:
        parser.error("需要 2<=vocab_size<=65536、n_embd 能被 n_head 整除、seq_len<=block_size")
    if not math.isfinite(args.dropout) or not 0 < args.dropout < 1:
        parser.error("--dropout 必须在 (0,1) 内，以测试 eval 是否关闭 Dropout")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    for name in ("weight_decay", "rtol", "atol", "fp32_rtol", "fp32_atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负的有限数")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in args.betas):
        parser.error("betas 必须在 [0,1) 内")
    return args


@contextmanager
def environment(values):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def evaluation_events(spec):
    if spec["eval_interval"] == 0:
        return []
    return [event for event in range(spec["max_iters"])
            if (event + 1) % spec["eval_interval"] == 0 or event + 1 == spec["max_iters"]]


def main_specs(args):
    plans = [("disabled_missing_val", 0, 3, True),
             ("periodic_and_final", 2, max(3, args.main_steps) | 1, True),
             ("final_on_interval", 2, 4, False),
             ("final_before_interval", 100, 3, False)]
    result = []
    for index, (name, interval, steps, bias) in enumerate(plans):
        result.append({"index": index, "name": name, "bias": bias, "seed": args.seed + (index + 1) * 101,
                       "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                       "vocab_size": min(args.vocab_size + index, 65536), "block_size": args.block_size,
                       "seq_length": args.seq_len, "batch_size": args.batch_size,
                       "micro_batch_size": args.micro_batch_size, "grad_clip": 0.0 if bias else args.clip_norm,
                       "max_iters": steps, "log_interval": 2, "eval_interval": interval, "eval_iters": args.eval_iters,
                       "learning_rate": args.learning_rate * 1.25, "weight_decay": args.weight_decay + 0.03,
                       "beta1": 0.85, "beta2": 0.97})
    return result


def main_argv(spec, directory, device):
    result = ["--device-type", device, "--data-dir", str(directory), "--dropout", "0"]
    for name in ("seed", "n_layer", "n_head", "n_embd", "vocab_size", "block_size", "seq_length", "batch_size",
                 "micro_batch_size", "grad_clip", "max_iters", "log_interval", "learning_rate", "weight_decay", "beta1", "beta2",
                 "eval_interval", "eval_iters"):
        result += ["--" + name.replace("_", "-"), str(spec[name])]
    if not spec["bias"]:
        result.append("--no-bias")
    return result


def evaluation_state(torch, np, model, raw, optimizer, training_generator=None):
    return {
        "state": snapshot(torch, raw, optimizer), "gradients": captured_gradients(torch, raw),
        "modes": {name: module.training for name, module in model.named_modules()},
        "identities": {name: id(param) for name, param in raw.named_parameters()}, "aliases": alias_map(raw),
        "ddp_flags": {name: getattr(model, name) for name in ("require_backward_grad_sync", "require_forward_param_sync")
                      if hasattr(model, name)},
        "last_grad_norm": optimizer.last_grad_norm,
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(next(raw.parameters()).device).clone()
                    if next(raw.parameters()).device.type == "cuda" else None,
        "python_rng": random.getstate(), "numpy_rng": copy.deepcopy(np.random.get_state()),
        "training_rng": None if training_generator is None else training_generator.get_state().clone(),
    }


def unchanged_evaluation_state(torch, np, actual, expected, errors, args):
    for name in ("state", "gradients", "modes", "identities", "aliases", "ddp_flags", "last_grad_norm",
                 "cpu_rng", "cuda_rng", "training_rng", "python_rng"):
        check_tree(torch, actual[name], expected[name], errors, f"eval preserves {name}", args, exact=True)
    a, b = actual["numpy_rng"], expected["numpy_rng"]
    if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
        errors.append("评估改变了 NumPy 全局 RNG")


def reference_evaluation(torch, np, dense, kwargs, kind="GPT"):
    generator = torch.Generator(device="cpu").manual_seed(kwargs["seed"] + kwargs["dp_rank"])
    data = np.fromfile(kwargs["data_path"], dtype=np.uint16)
    totals = torch.zeros(2, dtype=torch.float64, device=kwargs["device"])
    batches = []
    old_mode = dense.training
    dense.eval()
    try:
        with torch.no_grad():
            for _ in range(kwargs["eval_iters"]):
                x, y, starts = oracle_batch(torch, np, data, kwargs["batch_size"], kwargs["seq_length"], generator, kwargs["device"])
                output, loss = dense(x, y)
                if kind == "probe":
                    value = float(np.square(y.cpu().numpy().astype(np.float64)).sum())
                    totals[0].add_(value)
                else:
                    totals[0].add_(loss.to(dtype=torch.float64) * y.numel())
                totals[1].add_(y.numel())
                batches.append({"x": clone_tree(torch, x), "y": clone_tree(torch, y), "output": clone_tree(torch, output), "starts": starts})
    finally:
        dense.train(old_mode)
    return totals, batches, generator.get_state().clone()


def verify_evaluation(torch, np, dist, rank, trainer, eval_function, model, raw, dense, optimizer,
                      kwargs, members, args, label, record, kind="GPT", training_generator=None):
    expected_args = dict(kwargs, dp_rank=members.index(rank))
    expected_totals, expected_batches, expected_rng = reference_evaluation(torch, np, dense, expected_args, kind)
    dist.all_reduce(expected_totals, op=dist.ReduceOp.SUM, group=kwargs["dp_group"])
    expected_loss = float((expected_totals[0] / expected_totals[1]).item())
    data_digest = hashlib.sha256(Path(kwargs["data_path"]).read_bytes()).hexdigest()
    before = evaluation_state(torch, np, model, raw, optimizer, training_generator)
    errors, loads, forwards = [], [], []
    original_loader = trainer.get_zero_batch

    def traced_loader(*positional, **keyword):
        bound = bind_call(original_loader, positional, keyword)
        result = original_loader(*positional, **keyword)
        loads.append({"arguments": bound, "x": clone_tree(torch, result[0]), "y": clone_tree(torch, result[1])})
        return result

    def forbid_ddp_forward(module, inputs):
        raise AssertionError("评估应调用 DDP 内部 raw_model，以避免额外前向同步")

    def capture_forward(module, inputs, output):
        forwards.append({"x": clone_tree(torch, inputs[0]), "y": clone_tree(torch, inputs[1]),
                         "output": clone_tree(torch, output[0]), "loss": clone_tree(torch, output[1]),
                         "grad_enabled": torch.is_grad_enabled(), "requires_grad": bool(output[0].requires_grad or output[1].requires_grad),
                         "training": any(child.training for child in raw.modules())})

    handles = [raw.register_forward_hook(capture_forward)]
    if model is not raw:
        handles.append(model.register_forward_pre_hook(forbid_ddp_forward))
    try:
        with patched_attributes([(trainer, "get_zero_batch", traced_loader)]):
            actual_loss = eval_function(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    after = evaluation_state(torch, np, model, raw, optimizer, training_generator)
    unchanged_evaluation_state(torch, np, after, before, errors, args)
    if hashlib.sha256(Path(kwargs["data_path"]).read_bytes()).hexdigest() != data_digest:
        errors.append("评估修改了数据文件")
    if not isinstance(actual_loss, float):
        errors.append("estimate_zero_loss 应返回 Python float")
    check_number(actual_loss, expected_loss, errors, "global token-weighted eval loss", args)
    if len(loads) != kwargs["eval_iters"]:
        errors.append("评估取 batch 次数与 eval_iters 不同")
    chunks = [(start, min(start + kwargs["micro_batch_size"], kwargs["batch_size"]))
              for start in range(0, kwargs["batch_size"], kwargs["micro_batch_size"])]
    if len(forwards) != kwargs["eval_iters"] * len(chunks):
        errors.append("评估 micro-batch 前向次数错误")
    first_generator = None
    for index, (item, expected) in enumerate(zip(loads, expected_batches)):
        bound, generator = item["arguments"], item["arguments"]["generator"]
        if os.path.abspath(bound["data_path"]) != os.path.abspath(kwargs["data_path"]):
            errors.append("评估读取了错误的数据文件")
        if bound["batch_size"] != kwargs["batch_size"] or bound["seq_length"] != kwargs["seq_length"] or bound["device"] != kwargs["device"]:
            errors.append("评估传给 get_zero_batch 的尺寸或设备错误")
        if generator.device.type != "cpu" or generator.initial_seed() != kwargs["seed"] + members.index(rank):
            errors.append("评估 generator 设备/种子没有使用组内 dp_rank")
        if first_generator is not None and generator is not first_generator:
            errors.append("同一次评估在 batch 间重新创建 generator")
        first_generator = generator
        check_tensor(torch, item["x"], expected["x"], errors, f"eval batch {index}.x", args, exact=True)
        check_tensor(torch, item["y"], expected["y"], errors, f"eval batch {index}.y", args, exact=True)
    if first_generator is not None and not torch.equal(first_generator.get_state(), expected_rng):
        errors.append("评估数据 generator 的末状态不符合实际抽样次数")
    for index, item in enumerate(forwards):
        if index >= len(expected_batches) * len(chunks):
            break
        expected = expected_batches[index // len(chunks)]
        start, end = chunks[index % len(chunks)]
        check_tensor(torch, item["x"], expected["x"][start:end], errors, "eval micro input.x", args, exact=True)
        check_tensor(torch, item["y"], expected["y"][start:end], errors, "eval micro input.y", args, exact=True)
        check_tensor(torch, item["output"], expected["output"][start:end], errors, "eval micro output", args)
        target = expected["y"][start:end]
        expected_micro_loss = target.to(torch.float64).square().mean() if kind == "probe" else torch.nn.functional.cross_entropy(
            expected["output"][start:end].reshape(-1, expected["output"].size(-1)), target.reshape(-1))
        check_tensor(torch, item["loss"], expected_micro_loss, errors, "eval micro loss", args)
        if item["training"] or item["grad_enabled"] or item["requires_grad"]:
            errors.append("评估前向没有同时启用 eval 模式和关闭梯度记录")
    agree(dist, rank, errors, f"{label}: token mean / data / eval mode / no-grad / state and RNG", record)
    details = {"loss": number_for_report(actual_loss), "expected_loss": expected_loss,
               "global_tokens": int(expected_totals[1].item()), "local_tokens": kwargs["batch_size"] * kwargs["seq_length"] * kwargs["eval_iters"],
               "batch_size": kwargs["batch_size"], "seq_length": kwargs["seq_length"], "micro_batch_size": kwargs["micro_batch_size"],
               "eval_iters": kwargs["eval_iters"], "seed": kwargs["seed"], "members": members,
               "sample_starts": [item["starts"] for item in expected_batches]}
    return details, actual_loss


class EvaluationProbeError(RuntimeError):
    pass


class InitReached(RuntimeError):
    pass


def serial_checks(torch, np, trainer, args, temporary):
    result = {"cli": [], "main_file_validation": []}
    spec = main_specs(args)[0]
    directory = Path(temporary) / "serial"
    directory.mkdir()
    data = (np.arange(max(args.seq_len + 3, 31), dtype=np.int64) % args.vocab_size).astype(np.uint16)
    data.tofile(directory / "train.bin")
    base = main_argv(spec, directory, "cpu")
    with command_line(base):
        parsed = trainer.parse_args()
    if parsed.eval_interval != 0 or parsed.eval_iters != args.eval_iters:
        raise AssertionError("parse_args 没有读取 eval_interval/eval_iters")
    with command_line(base + ["--eval-interval", "2", "--eval-iters", "3"]):
        parsed = trainer.parse_args()
    if parsed.eval_interval != 2 or parsed.eval_iters != 3:
        raise AssertionError("parse_args 没有采用评估参数覆盖值")
    result["cli"].append("zero/positive interval and eval_iters")
    for flag, value in (("--eval-interval", "-1"), ("--eval-iters", "0"), ("--eval-iters", "-1")):
        with command_line(base + [flag, value]), redirect_stderr(io.StringIO()):
            try:
                trainer.parse_args()
            except (ValueError, SystemExit) as exc:
                if isinstance(exc, SystemExit) and exc.code != 2:
                    raise AssertionError(f"{flag}={value} 没有正常拒绝") from exc
            else:
                raise AssertionError(f"parse_args 没有拒绝 {flag}={value}")
        result["cli"].append(f"rejected {flag}={value}")

    import torch.distributed as dist
    calls = []

    def stop_before_init(*positional, **keyword):
        calls.append((positional, keyword))
        raise InitReached("仅检查初始化前的文件验证")

    val_path = directory / "val.bin"
    cases = [("disabled_missing_val", 0, None, True), ("enabled_missing_val", 1, None, False),
             ("enabled_empty_val", 1, b"", False), ("enabled_odd_val", 1, b"abc", False),
             ("enabled_short_val", 1, np.arange(args.seq_len, dtype=np.uint16).tobytes(), False),
             ("enabled_minimum_valid_val", 1, np.arange(args.seq_len + 1, dtype=np.uint16).tobytes(), True)]
    if dist.is_initialized():
        raise RuntimeError("串行文件检查开始前不应存在进程组")
    with patched_attributes([(dist, "init_process_group", stop_before_init)]), environment(
            {"RANK": 0, "WORLD_SIZE": 1, "LOCAL_RANK": 0, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": 29500, "USE_LIBUV": 0}):
        for label, interval, payload, should_reach in cases:
            if payload is None:
                val_path.unlink(missing_ok=True)
            else:
                val_path.write_bytes(payload)
            count_before = len(calls)
            reached, rejected = False, False
            with command_line(base + ["--eval-interval", str(interval)]), redirect_stdout(io.StringIO()):
                try:
                    trainer.main()
                except InitReached:
                    reached = True
                except (FileNotFoundError, ValueError):
                    rejected = True
            if reached != should_reach or (not should_reach and not rejected) or len(calls) - count_before != int(should_reach):
                raise AssertionError(f"实际 main 的初始化前文件检查错误：{label}")
            if dist.is_initialized():
                raise AssertionError("串行检查不应创建实际进程组")
            result["main_file_validation"].append({"case": label, "reached_init": reached, "rejected": rejected})
    print("[PASS] actual CLI / disabled missing-val / pre-init val-file validation (CPU).", flush=True)
    return result


def make_loss_probe(torch):
    class TokenLossProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, x, targets):
            values = targets.to(dtype=self.anchor.dtype) + self.anchor * 0.0
            return values.unsqueeze(-1), values.square().mean()
    return TokenLossProbe()


def failure_restore_checks(torch, np, dist, rank, trainer, model, raw, optimizer, kwargs, args, record, label):
    def fail_loader(*positional, **keyword):
        raise EvaluationProbeError("检查 finally 恢复模式")

    results = []
    for mode in (True, False):
        model.train(mode)
        before = evaluation_state(torch, np, model, raw, optimizer)
        caught, errors = False, []
        with patched_attributes([(trainer, "get_zero_batch", fail_loader)]):
            try:
                trainer.estimate_zero_loss(**kwargs)
            except EvaluationProbeError:
                caught = True
        if not caught:
            errors.append("评估吞掉了数据加载异常")
        unchanged_evaluation_state(torch, np, evaluation_state(torch, np, model, raw, optimizer), before, errors, args)
        agree(dist, rank, errors, f"{label}: exception restores mode={mode} and state", record)
        results.append({"initial_training": mode, "exception_propagated": caught})
    for name in ("batch_size", "seq_length", "micro_batch_size", "eval_iters"):
        invalid = dict(kwargs, **{name: 0})
        before = evaluation_state(torch, np, model, raw, optimizer)
        caught, errors = False, []
        try:
            trainer.estimate_zero_loss(**invalid)
        except ValueError:
            caught = True
        if not caught:
            errors.append(f"评估没有拒绝 {name}=0")
        unchanged_evaluation_state(torch, np, evaluation_state(torch, np, model, raw, optimizer), before, errors, args)
        agree(dist, rank, errors, f"{label}: invalid {name} rejected", record)
    return results


def direct_model(torch, np, dist, rank, GPT, GPTConfig, zero, trainer, group, members, layout, kind, bias, args, device, record, data_path):
    label = f"eval function {layout}, {kind}" + (f", bias={bias}" if kind == "GPT" else "")
    if rank == 0:
        print(f"[CASE] {label}", flush=True)
    torch.manual_seed(args.seed + rank)
    raw = make_loss_probe(torch) if kind == "probe" else GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, vocab_size=args.vocab_size, block_size=args.block_size, dropout=args.dropout, bias=bias))
    raw = raw.to(device=device, dtype=torch.float64)
    options = {"process_group": group}
    if device.type == "cuda":
        options.update(device_ids=[device.index], output_device=device.index)
    ddp = torch.nn.parallel.DistributedDataParallel(raw, **options)
    optimizer = zero.Zero1Optimizer(raw, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                    beta1=args.betas[0], beta2=args.betas[1], dp_group=group)
    # 先为真实本地 AdamW 建立非空矩状态，再检查评估不改变这些状态。
    for index, param in enumerate(raw.parameters()):
        if param.requires_grad:
            param.grad = torch.full_like(param, 0.125 + 0.001 * index)
    stepped = optimizer.step(grad_clip=0.0)
    agree(dist, rank, [] if stepped else ["评估状态准备的有限梯度更新被跳过"], f"{label}: prepare real AdamW moments", record)
    optimizer.zero_grad(set_to_none=True)
    for index, param in enumerate(raw.parameters()):
        param.grad = torch.full_like(param, 0.37 + index * 0.01) if index % 2 == 0 else None
    dense = copy.deepcopy(raw)
    group_rank = members.index(rank)
    base = dict(model=ddp, data_path=str(data_path), batch_size=args.batch_size, seq_length=args.seq_len,
                micro_batch_size=args.micro_batch_size, eval_iters=args.eval_iters, device=device, dp_group=group, seed=args.seed + 20_000)
    results = []
    plans = [("unequal_micro", args.micro_batch_size, True, False, False, 2),
             ("micro_one", 1, False, False, False, 1),
             ("uneven_rank_tokens", args.batch_size + 3, True, True, False, 1),
             ("bare_model", args.micro_batch_size, False, True, True, 1)]
    for name, micro, mode, uneven, bare, repetitions in plans:
        ddp.train(mode)
        candidate = raw if bare else ddp
        kwargs = dict(base, model=candidate, micro_batch_size=micro)
        if uneven:
            kwargs.update(batch_size=args.batch_size + group_rank % 2, eval_iters=args.eval_iters + group_rank % 2,
                          seq_length=max(1, args.seq_len - group_rank % 2))
        previous_loss = None
        for repetition in range(repetitions):
            details, loss = verify_evaluation(torch, np, dist, rank, trainer, trainer.estimate_zero_loss,
                candidate, raw, dense, optimizer, kwargs, members, args,
                f"{label}, {name}, repeat={repetition}", record, kind=kind)
            if previous_loss is not None:
                errors = []
                check_number(loss, previous_loss, errors, "repeated eval same seed", args)
                agree(dist, rank, errors, f"{label}: repeated eval reproducible", record)
            previous_loss = loss
            details.update(case=name, initial_training=mode, bare_model=bare, repetition=repetition)
            results.append(details)
    failures = failure_restore_checks(torch, np, dist, rank, trainer, ddp, raw, optimizer, base, args, record, label)
    return {"label": label, "status": "passed", "cases": results, "failure_restore": failures, "members": members}


def function_worker(rank, args, temporary):
    configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "status": "running", "checks": [], "models": []}
    path = Path(temporary) / f"rank_{rank}.json"
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("函数 worker 启动前源码发生变化")
        GPT, GPTConfig, zero, trainer = load_project(args.project)
        store = dist.FileStore(str(Path(temporary) / "rendezvous"), args.dp_size)
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", store=store,
                                rank=rank, world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        for layout, partitions in group_layouts(args.dp_size):
            group, members = None, None
            for ranks in partitions:
                candidate = dist.group.WORLD if layout == "WORLD" else dist.new_group(ranks=ranks, timeout=timedelta(seconds=args.timeout))
                if rank in ranks:
                    group, members = candidate, ranks
            for kind, bias in (("GPT", True), ("GPT", False), ("probe", True)):
                record["models"].append(direct_model(torch, np, dist, rank, GPT, GPTConfig, zero, trainer, group,
                    members, layout, kind, bias, args, device, record, Path(temporary) / "eval.bin"))
                write_json(path, record)
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("评估函数验证期间源码发生变化")
        record["status"] = "passed"
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def compare_evaluation_logs(text, rank, spec, evaluations, args, errors):
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("eval iter ")]
    parsed = [EVAL_LOG_PATTERN.fullmatch(line) for line in lines]
    if any(match is None for match in parsed):
        errors.append("main 的评估日志格式错误")
        return
    expected = evaluation_events(spec) if rank == 0 else []
    if [int(match.group(1)) for match in parsed] != expected:
        errors.append(f"rank {rank} 完整评估日志的迭代编号错误，应为 {expected}")
    for match in parsed:
        event = int(match.group(1))
        pair = {item["split"]: item for item in evaluations if item["event"] == event}
        if pair.keys() != {"train", "val"}:
            errors.append("评估日志缺少实际 train/val 评估调用")
            continue
        for split, value in (("train", match.group(2)), ("val", match.group(3))):
            try:
                if not math.isclose(float(value), float(pair[split]["loss"]), rel_tol=args.rtol, abs_tol=max(args.atol, 5.1e-7)):
                    errors.append(f"eval iter {event} 日志 {split} loss 与实际返回值不同")
            except (TypeError, ValueError):
                errors.append(f"eval iter {event} 日志 {split} loss 不是正常数值")


def main_worker(rank, args, temporary, spec, port):
    configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "status": "running", "checks": [], "steps": [], "evaluations": [], "case": spec}
    path = Path(temporary) / f"rank_{rank}.json"
    saved_init, saved_destroy = dist.init_process_group, dist.destroy_process_group
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("实际 main 启动前源码发生变化")
        GPT, GPTConfig, zero, trainer = load_project(args.project)
        observed = {"ddp": [], "optimizers": [], "train_batch_calls": [], "eval_batch_calls": [],
                    "init_calls": 0, "destroy_calls": 0, "eval_depth": 0}
        data_dir = Path(temporary) / "data"
        train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
        train_data = np.fromfile(train_path, dtype=np.uint16)
        data_digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, val_path) if p.is_file()}
        expected_generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 10_000 + rank)
        tolerance = copy.copy(args)
        tolerance.rtol, tolerance.atol = args.fp32_rtol, args.fp32_atol
        tolerance.learning_rate, tolerance.weight_decay = spec["learning_rate"], spec["weight_decay"]
        tolerance.betas = (spec["beta1"], spec["beta2"])
        tolerance.reference_from_validated_gradients = True
        original_ddp, original_zero = torch.nn.parallel.DistributedDataParallel, zero.Zero1Optimizer
        original_batch, original_train, original_eval = trainer.get_zero_batch, trainer.train_zero_step, trainer.estimate_zero_loss
        expected_evaluations = [(event, split, offset) for event in evaluation_events(spec)
                                for split, offset in (("train", 20_000), ("val", 30_000))]

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
                if dist.get_world_size(group=self.process_group) != args.dp_size:
                    errors.append("main 的 DDP 组大小错误")
                for name in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "bias"):
                    if getattr(raw.config, name) != spec[name]:
                        errors.append(f"main 的 GPTConfig.{name} 没有采用命令行参数")
                if raw.config.dropout != 0.0:
                    errors.append("main 没有采用 dropout=0 的验证配置")
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(spec["seed"])
                    dense = GPT(GPTConfig(n_layer=spec["n_layer"], n_head=spec["n_head"], n_embd=spec["n_embd"],
                        vocab_size=spec["vocab_size"], block_size=spec["block_size"], dropout=0.0, bias=spec["bias"]))
                    dense = dense.to(device=device, dtype=torch.float32)
                check_tree(torch, raw.state_dict(), dense.state_dict(), errors, "main initial seed/weights", tolerance, exact=True)
                if any(param.device != device or param.dtype != torch.float32 for param in raw.parameters()):
                    errors.append("main 参数应在指定设备且为 float32")
                observed.update(raw=raw, model=self, dense=dense, reference=make_reference_optimizer(torch, dense, tolerance),
                                identities={name: id(param) for name, param in raw.named_parameters()}, aliases=alias_map(raw))
                agree(dist, rank, errors, "actual main: real DDP / config / seed", record)

        class TracedZero(original_zero):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                observed["optimizers"].append(self)
                errors = []
                if self.model is not observed.get("raw") or self.dp_size != args.dp_size or self.dp_rank != rank:
                    errors.append("main 的 ZeRO-1 模型引用或 DP 组错误")
                agree(dist, rank, errors, "actual main: real ZeRO-1 references / group", record)
                record["partition"] = partition_checks(torch, dist, rank, observed["raw"], zero, self, observed["reference"],
                    observed["dense"], self.dp_group, list(range(args.dp_size)), tolerance, "actual main", record)

        def observed_batch(*positional, **keyword):
            bound = bind_call(original_batch, positional, keyword)
            item = {"path": os.path.abspath(bound["data_path"]), "batch_size": bound["batch_size"],
                    "seq_length": bound["seq_length"], "generator": bound["generator"]}
            observed["eval_batch_calls" if observed["eval_depth"] else "train_batch_calls"].append(item)
            return original_batch(*positional, **keyword)

        # 评估观察器会再次检查 loader 参数；保留原函数的可检查签名。
        observed_batch.__signature__ = inspect.signature(original_batch)

        def observed_train(*positional, **keyword):
            bound = bind_call(original_train, positional, keyword)
            event, errors = len(record["steps"]), []
            if event >= spec["max_iters"]:
                errors.append("main 超过了 max_iters")
            if bound["model"] is not observed.get("model") or bound["optimizer"] not in observed["optimizers"]:
                errors.append("main 传入训练函数的模型/优化器错误")
            if bound["micro_batch_size"] != spec["micro_batch_size"] or bound["grad_clip"] != spec["grad_clip"]:
                errors.append("main 没有传递训练 micro_batch_size/grad_clip")
            for name in ("x", "y"):
                if bound[name].device != device or bound[name].dtype != torch.long:
                    errors.append(f"main 的 {name} 设备/dtype 错误")
            agree(dist, rank, errors, f"actual main, iter={event}: training call", record)
            expected_x, expected_y, starts = oracle_batch(torch, np, train_data, spec["batch_size"], spec["seq_length"], expected_generator, device)
            group = bound["optimizer"].dp_group
            all_x, all_y = [torch.empty_like(expected_x) for _ in range(args.dp_size)], [torch.empty_like(expected_y) for _ in range(args.dp_size)]
            dist.all_gather(all_x, expected_x, group=group)
            dist.all_gather(all_y, expected_y, group=group)
            details, result = execute_step(torch, dist, rank, original_train, observed["model"], observed["raw"], observed["dense"],
                bound["optimizer"], observed["reference"], bound["x"], bound["y"], expected_x, expected_y, torch.cat(all_x), torch.cat(all_y),
                spec["micro_batch_size"], spec["grad_clip"], "GPT", group, list(range(args.dp_size)), True, None,
                observed["identities"], observed["aliases"], tolerance, f"actual main, iter={event}", record)
            details["sample_starts"] = starts
            record["steps"].append(details)
            write_json(path, record)
            return result

        def observed_eval(*positional, **keyword):
            bound = dict(bind_call(original_eval, positional, keyword))
            count, event, errors = len(record["evaluations"]), len(record["steps"]) - 1, []
            if count >= len(expected_evaluations):
                errors.append("main 在关闭评估时调用评估，或调用次数过多")
                agree(dist, rank, errors, "actual main: unexpected eval call", record)
            expected_event, split, offset = expected_evaluations[count]
            optimizer = observed["optimizers"][0]
            if event != expected_event:
                errors.append(f"评估应在 iter {expected_event} 更新尝试结束后调用，实际 iter {event}")
            if bound["model"] is not observed["model"] or bound["dp_group"] is not optimizer.dp_group:
                errors.append("main 传入评估的模型或 DP 组错误")
            expected_path = train_path if split == "train" else val_path
            if os.path.abspath(bound["data_path"]) != os.path.abspath(expected_path):
                errors.append("main 的评估 split 顺序或数据路径错误")
            for name in ("batch_size", "seq_length", "micro_batch_size", "eval_iters"):
                if bound[name] != spec[name]:
                    errors.append(f"main 未传递评估参数 {name}")
            if bound["device"] != device or bound["seed"] != spec["seed"] + offset:
                errors.append("main 的评估设备或独立 seed 偏移错误")
            if not observed["train_batch_calls"]:
                errors.append("main 应在完成训练迭代后评估")
            agree(dist, rank, errors, f"actual main, iter={event}, {split}: eval scheduling and wiring", record)
            training_generator = observed["train_batch_calls"][-1]["generator"]
            observed["eval_depth"] += 1
            try:
                details, result = verify_evaluation(torch, np, dist, rank, trainer, original_eval,
                    observed["model"], observed["raw"], observed["dense"], optimizer, bound, list(range(args.dp_size)),
                    tolerance, f"actual main, iter={event}, {split}", record, training_generator=training_generator)
            finally:
                observed["eval_depth"] -= 1
            details.update(event=event, split=split)
            record["evaluations"].append(details)
            write_json(path, record)
            return result

        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(args.dp_size),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
        changes = [(dist, "init_process_group", observed_init), (dist, "destroy_process_group", observed_destroy),
                   (torch.nn.parallel, "DistributedDataParallel", TracedDDP), (zero, "Zero1Optimizer", TracedZero),
                   (trainer, "get_zero_batch", observed_batch), (trainer, "train_zero_step", observed_train),
                   (trainer, "estimate_zero_loss", observed_eval)]
        for name, value in (("DDP", TracedDDP), ("Zero1Optimizer", TracedZero)):
            if hasattr(trainer, name):
                changes.append((trainer, name, value))
        buffer = io.StringIO()
        try:
            with patched_attributes(changes), command_line(main_argv(spec, data_dir, args.device)), redirect_stdout(buffer):
                trainer.main()
        finally:
            captured = buffer.getvalue()
            record["stdout"] = captured
            if rank == 0 or "[FAIL" in captured:
                print(captured, end="", flush=True)
        errors = []
        record["main_destroyed_group"] = not dist.is_initialized()
        if dist.is_initialized():
            errors.append("main 正常返回后没有销毁进程组")
            saved_destroy()
        if observed["init_calls"] != 1 or observed["destroy_calls"] != 1:
            errors.append("main 的实际初始化/销毁次数错误")
        if len(observed["ddp"]) != 1 or len(observed["optimizers"]) != 1:
            errors.append("main 应创建一组 DDP 模型与 ZeRO-1 优化器")
        if len(record["steps"]) != spec["max_iters"] or len(observed["train_batch_calls"]) != spec["max_iters"]:
            errors.append("main 的训练/取训练 batch 次数错误")
        if [(item["event"], item["split"]) for item in record["evaluations"]] != [(event, split) for event, split, _ in expected_evaluations]:
            errors.append("main 的完整 train/val 评估调用序列错误")
        if len(observed["eval_batch_calls"]) != len(expected_evaluations) * spec["eval_iters"]:
            errors.append("main 的取评估 batch 次数错误")
        previous = None
        for item in observed["train_batch_calls"]:
            generator = item["generator"]
            if item["path"] != os.path.abspath(train_path) or item["batch_size"] != spec["batch_size"] or item["seq_length"] != spec["seq_length"]:
                errors.append("main 的训练 batch 路径/尺寸错误")
            if generator.device.type != "cpu" or generator.initial_seed() != spec["seed"] + 10_000 + rank:
                errors.append("main 的训练 generator 设备/种子错误")
            if previous is not None and generator is not previous:
                errors.append("main 在迭代间重新创建训练 generator")
            previous = generator
        if previous is not None and not torch.equal(previous.get_state(), expected_generator.get_state()):
            errors.append("main 的训练数据随机序列被评估改变")
        for filename, digest in data_digests.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                errors.append("main 修改了验证数据文件")
        if spec["eval_interval"] == 0 and val_path.exists():
            errors.append("关闭评估的 main 不应创建 val.bin")
        compare_logs(captured, rank, spec, record["steps"], tolerance, errors)
        compare_evaluation_logs(captured, rank, spec, record["evaluations"], tolerance, errors)
        if fingerprints(args.project) != args.expected_hashes:
            errors.append("实际 main 验证期间源码发生变化")
        # 实际 main 已结束且清理状态已记录，新组只用于汇总结果。
        store = dist.FileStore(str(Path(temporary) / "post_main_results"), args.dp_size)
        saved_init("nccl" if args.device == "cuda" else "gloo", store=store, rank=rank,
                   world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        agree(dist, rank, errors, "actual main: full logs / schedule / training RNG / cleanup", record)
        record.update(status="passed", init_calls=observed["init_calls"], destroy_calls=observed["destroy_calls"])
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            saved_destroy()


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


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "project": str(args.project), "device": args.device, "world_size": args.dp_size, "threads": args.threads,
              "function_dtype": "float64", "function_dropout": args.dropout, "main_dtype": "float32", "main_dropout": 0.0,
              "main_reference_policy": "independent global dense forward/gradient checks; ordinary AdamW from numerically verified synchronized gradients",
              "group_layouts": group_layouts(args.dp_size), "function_workers": [], "actual_main": [],
              "scope": "ZeRO-1 evaluation function and actual main; checkpoint/AMP/compile/multinode/performance pending"}
    try:
        args.expected_hashes = preflight(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / evaluation and optimizer signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only"
            print("[INFO] Static checks only; numerical verification was NOT run.", flush=True)
            return 0
        import numpy as np
        import torch
        import torch.multiprocessing as mp
        report["torch_version"], report["numpy_version"] = str(torch.__version__), str(np.__version__)
        if not torch.distributed.is_available():
            raise RuntimeError("当前 PyTorch 没有分布式支持")
        if args.device == "cpu" and not torch.distributed.is_gloo_available():
            raise RuntimeError("CPU 验证需要 Gloo")
        if args.device == "cuda" and (not torch.cuda.is_available() or not torch.distributed.is_nccl_available() or torch.cuda.device_count() < args.dp_size):
            raise RuntimeError("CUDA 验证需要 NCCL，且设备数至少为 --dp-size；也可使用 --device cpu")
        torch.set_num_threads(args.threads)
        _, _, _, trainer = load_project(args.project)
        print(f"[INFO] {args.dp_size} {args.device} workers; threads={args.threads}; temporary train/val data.", flush=True)
        print("[INFO] Function checks use float64 and nonzero Dropout; main checks retain FP32 with dropout=0.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_zero1_eval_") as temporary:
            root = Path(temporary)
            report["serial_checks"] = serial_checks(torch, np, trainer, args, root)
            functions = root / "functions"
            functions.mkdir()
            np.random.default_rng(args.seed + 91).integers(0, args.vocab_size, max(257, args.seq_len * 5 + 3), dtype=np.uint16).tofile(functions / "eval.bin")
            try:
                mp.spawn(function_worker, args=(args, str(functions)), nprocs=args.dp_size, join=True)
            finally:
                report["function_workers"] = collect_workers(functions, args.dp_size)
            require_all_workers(report["function_workers"], args.dp_size, "eval function")
            for spec in main_specs(args):
                directory = root / f"main_{spec['index']}"
                data_dir = directory / "data"
                data_dir.mkdir(parents=True)
                for split, offset in (("train", 91), ("val", 191)):
                    if split == "val" and spec["eval_interval"] == 0:
                        continue
                    np.random.default_rng(spec["seed"] + offset).integers(0, spec["vocab_size"], max(257, spec["seq_length"] * 5 + 3), dtype=np.uint16).tofile(data_dir / f"{split}.bin")
                print(f"[CASE] actual main: {spec['name']}, eval_interval={spec['eval_interval']}, max_iters={spec['max_iters']}", flush=True)
                try:
                    mp.spawn(main_worker, args=(args, str(directory), spec, unused_port()), nprocs=args.dp_size, join=True)
                finally:
                    workers = collect_workers(directory, args.dp_size)
                    report["actual_main"].append({"case": spec, "workers": workers})
                require_all_workers(workers, args.dp_size, f"actual main {spec['name']}")
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码发生变化")
        report["status"] = "passed"
        print(f"[PASS] ZeRO-1 evaluation verification completed ({args.device}, world_size={args.dp_size}).", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if args.report is not None:
            write_json(args.report.resolve(), report)
            print(f"[INFO] Report saved: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
