#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证手写 nanoGPT ZeRO-1 优化器；不修改项目源码。

把本文件放在含 model.py、zero1.py 的 nanoGPT 根目录，运行：
  python verify_zero1.py --project . --device cpu --dp-size 4 --threads 1 --report zero1_report.json

只依赖 Python 标准库和 PyTorch，不依赖之前的 TP/PP 验证文件。
直接调用项目 build_parameter_partition 和 Zero1Optimizer 的真实实现。
使用真实 DDP + Gloo/NCCL，float64、小模型、dropout=0；参考为普通
AdamW 对整个 DP 组的拼接 batch 训练。验证前向、loss、带不等大小
micro-batch 的梯度累积、完整梯度、范数、裁剪、所有模型参数和 AdamW
step/一阶/二阶矩，以及优化器只持有本 rank 的参数和状态。

默认测试 WORLD、单进程子组和奇偶分组，检查组内/全局 rank 转换。
测试 GPT 的 bias=True/False，另用带共享及冻结参数的小模型检查去重、
冻结参数排除，以及 dp_size>独立参数数目时的空优化器 rank。
检查范数=5 的合成梯度、关闭裁剪、零梯度、单 rank 注入 NaN/+Inf/-Inf、
有限梯度导致范数溢出、整组跳过时参数和优化器状态不变、子组隔离与恢复。

导入 torch 前限制 BLAS/OpenMP 线程；CPU 使用临时 FileStore。
--dp-size 是验证 worker 数，不限制实现支持的规模；--dp-size 1 只验证
单 rank 路径。CUDA 模式为单机、每个 worker 一个设备，需要 NCCL。
不运行生产训练入口，不验证真实数据加载、AMP、checkpoint、torchrun、
多机启动、内存峰值或性能。--preflight-only 只做静态检查，不算数值通过。
"""

import argparse
import ast
from contextlib import nullcontext
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback


METHODS = {
    "__init__": ("self", "model", "learning_rate", "weight_decay", "beta1", "beta2", "dp_group"),
    "zero_grad": ("self", "set_to_none"),
    "sync_parameters": ("self",),
    "step": ("self", "grad_clip"),
    "has_nonfinite_gradients": ("self",),
    "clip_gradients": ("self", "max_norm"),
}


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
    parser.add_argument("--clip-norm", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("dp_size", "threads", "timeout", "n_layer", "n_head", "n_embd", "vocab_size",
                 "block_size", "seq_len", "batch_size", "micro_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.vocab_size < 2 or args.n_embd % args.n_head or args.seq_len > args.block_size:
        parser.error("需要 vocab_size>=2、n_embd 能被 n_head 整除、seq_len<=block_size")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    for name in ("weight_decay", "rtol", "atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负的有限数")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in args.betas):
        parser.error("betas 必须在 [0,1) 内")
    return args


def configure_threads(count):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(count)
    os.environ["USE_LIBUV"] = "0"


def fingerprints(project):
    paths = [project / "model.py", project / "zero1.py", Path(__file__).resolve()]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def check_signature(node, expected, label):
    names = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    missing = [name for name in expected if name not in names]
    if missing:
        raise TypeError(f"{label} 缺少参数：{', '.join(missing)}")


def preflight(project):
    trees = {}
    for filename in ("model.py", "zero1.py"):
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"项目根目录需要 {path}")
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
        trees[filename] = ast.parse(source, filename=str(path))
    top = trees["zero1.py"].body
    functions = [node for node in top if isinstance(node, ast.FunctionDef) and node.name == "build_parameter_partition"]
    classes = [node for node in top if isinstance(node, ast.ClassDef) and node.name == "Zero1Optimizer"]
    if len(functions) != 1 or len(classes) != 1:
        raise AttributeError("zero1.py 需要唯一的顶层 build_parameter_partition() 和 Zero1Optimizer 类")
    check_signature(functions[0], ("model", "dp_size"), "build_parameter_partition")
    for name, expected in METHODS.items():
        candidates = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(candidates) != 1:
            raise AttributeError(f"Zero1Optimizer 需要唯一的 {name}() 方法")
        check_signature(candidates[0], expected, f"Zero1Optimizer.{name}")
    declared = {node.name for node in trees["model.py"].body if isinstance(node, ast.ClassDef)}
    if not {"GPT", "GPTConfig"}.issubset(declared):
        raise AttributeError("model.py 需要原版 GPT 和 GPTConfig 类")
    return fingerprints(project)


def load_project(project):
    sys.path.insert(0, str(project))
    modules = []
    for name in ("model", "zero1"):
        sys.modules.pop(name, None)
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != project / f"{name}.py":
            raise ImportError(f"{name} 导入了其他目录的文件：{module.__file__}")
        modules.append(module)
    for name, expected in METHODS.items():
        signature = inspect.signature(getattr(modules[1].Zero1Optimizer, name))
        if any(arg not in signature.parameters for arg in expected):
            raise TypeError(f"运行时 Zero1Optimizer.{name} 的参数与静态检查不同")
    return modules[0].GPT, modules[0].GPTConfig, modules[1]


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
    elif actual != expected:
        errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")


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


def make_toy(torch):
    class TiedToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.tied_weight = self.weight
            self.bias = torch.nn.Parameter(torch.randn(4))
            self.frozen = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)

        def forward(self, x, targets):
            output = x @ self.weight + x @ self.tied_weight + self.bias + self.frozen
            return output, torch.nn.functional.mse_loss(output, targets)
    return TiedToy()


def batch(torch, kind, args, dp_size, event, first_rank, device):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000 + 97 * event + 1009 * first_rank)
    count = args.batch_size * dp_size
    if kind == "toy":
        x = torch.randn(count, 4, dtype=torch.float64, generator=generator)
        y = torch.randn(count, 4, dtype=torch.float64, generator=generator)
    else:
        x = torch.randint(args.vocab_size, (count, args.seq_len), generator=generator)
        y = torch.randint(args.vocab_size, (count, args.seq_len), generator=generator)
    return x.to(device), y.to(device)


def reference_norm(torch, dense):
    gradients = [param.grad.detach().to(torch.float64).reshape(-1) for param in dense.parameters() if param.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(gradients)).item()) if gradients else 0.0


def check_number(actual, expected, errors, label, args):
    try:
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=args.rtol, abs_tol=args.atol):
            errors.append(f"{label}: 实际 {actual!r}，预期 {expected!r}")
    except (TypeError, ValueError):
        errors.append(f"{label}: 返回值不可转为数值")


def gradients_equal(torch, raw, dense, args, errors, label):
    dense_params = dict(dense.named_parameters())
    for name, param in raw.named_parameters():
        check_tensor(torch, param.grad, dense_params[name].grad, errors, f"{label}.{name}", args)


def normal_step(torch, dist, rank, kind, args, raw, ddp, dense, optimizer, reference,
                group, members, identities, aliases, device, event, clip, label, record):
    errors = []
    optimizer.zero_grad(set_to_none=True)
    dense.zero_grad(set_to_none=True)
    full_x, full_y = batch(torch, kind, args, len(members), event, members[0], device)
    dense_output, dense_loss = dense(full_x, full_y)
    dense_loss.backward()
    dp_rank = members.index(rank)
    start = dp_rank * args.batch_size
    local_x = full_x[start:start + args.batch_size]
    local_y = full_y[start:start + args.batch_size]
    micro_sizes = [args.micro_batch_size, 1, args.batch_size]
    micro = micro_sizes[event % len(micro_sizes)]
    local_loss = torch.zeros((), dtype=torch.float64, device=device)
    for offset in range(0, args.batch_size, micro):
        end = min(offset + micro, args.batch_size)
        with (ddp.no_sync() if end < args.batch_size else nullcontext()):
            output, loss = ddp(local_x[offset:end], local_y[offset:end])
            expected_output = dense_output.detach()[start + offset:start + end]
            check_tensor(torch, output, expected_output, errors, "forward", args)
            if kind == "toy":
                expected_loss = torch.nn.functional.mse_loss(expected_output, local_y[offset:end])
            else:
                expected_loss = torch.nn.functional.cross_entropy(expected_output.reshape(-1, args.vocab_size), local_y[offset:end].reshape(-1))
            check_tensor(torch, loss, expected_loss, errors, "micro-batch loss", args)
            weight = (end - offset) / args.batch_size
            (loss * weight).backward()
            local_loss.add_(loss.detach().to(torch.float64) * weight)
    dist.all_reduce(local_loss, group=group)
    local_loss.div_(len(members))
    check_number(local_loss.item(), dense_loss.item(), errors, "global DP loss", args)
    gradients_equal(torch, raw, dense, args, errors, "DDP accumulated gradients")
    agree(dist, rank, errors, f"{label}: forward / loss / accumulated gradients", record)
    errors = []
    norm = reference_norm(torch, dense)
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(dense.parameters(), clip, foreach=False)
    stepped = optimizer.step(grad_clip=clip)
    if stepped is not True:
        errors.append("正常 step() 应返回 True")
    check_number(optimizer.last_grad_norm, norm, errors, "pre-clip norm", args)
    gradients_equal(torch, raw, dense, args, errors, "post-clip gradients")
    reference.step()
    difference = model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors)
    moments = state_errors(torch, raw, dense, optimizer, reference, args, errors)
    optimizer.zero_grad(set_to_none=True)
    if any(param.grad is not None for param in raw.parameters()):
        errors.append("zero_grad(True) 未清空完整模型梯度")
    agree(dist, rank, errors, f"{label}: norm / clipping / parameters / sharded AdamW moments", record)
    return {"event": event, "micro_batch_size": micro, "grad_clip": clip, "loss": dense_loss.item(),
            "grad_norm": norm, "max_parameter_error": difference, "local_moment_numels": moments}


def synthetic_clip_checks(torch, dist, rank, raw, optimizer, args, label, record):
    initial = snapshot(torch, raw, optimizer)
    for threshold in (0.0, args.clip_norm, 10.0):
        errors = []
        for _, param in optimizer.named_params:
            param.grad = torch.zeros_like(param)
        param = optimizer.named_params[0][1]
        param.grad.reshape(-1)[:2] = torch.tensor([3.0, 4.0], dtype=param.dtype, device=param.device)
        actual_norm = optimizer.clip_gradients(threshold)
        check_number(actual_norm, 5.0, errors, "synthetic norm=5", args)
        coefficient = 1.0 if threshold == 0 else min(1.0, threshold / (5.0 + 1e-6))
        for _, param in optimizer.named_params:
            expected = torch.zeros_like(param)
            if param is optimizer.named_params[0][1]:
                expected.reshape(-1)[:2] = torch.tensor([3.0 * coefficient, 4.0 * coefficient], dtype=param.dtype, device=param.device)
            check_tensor(torch, param.grad, expected, errors, "synthetic clipped gradient", args)
        check_tree(torch, snapshot(torch, raw, optimizer), initial, errors, "clipping preserves weights/AdamW", args, exact=True)
        optimizer.zero_grad(set_to_none=False)
        if any(param.grad is None or torch.count_nonzero(param.grad).item() for _, param in optimizer.named_params):
            errors.append("zero_grad(False) 未将完整可训练参数的梯度清零")
        optimizer.zero_grad(set_to_none=True)
        if any(param.grad is not None for param in raw.parameters()):
            errors.append("zero_grad(True) 未将完整模型梯度设为 None")
        agree(dist, rank, errors, f"{label}: synthetic norm=5, max_norm={threshold}", record)


def anomaly_step(torch, dist, rank, kind, raw, dense, optimizer, reference, members,
                 affected, args, identities, aliases, label, record):
    errors = []
    optimizer.zero_grad(set_to_none=True)
    dense.zero_grad(set_to_none=True)
    for param in raw.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param)
    for param in dense.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param)
    before = snapshot(torch, raw, optimizer)
    # 仅第一个子组受影响；优先在注入 rank 不负责的参数上注入。
    if affected and rank == members[-1]:
        candidates = [(name, param) for name, param in optimizer.named_params if optimizer.owners[name] != optimizer.dp_rank]
        target = (candidates or optimizer.named_params)[0][1]
        value = {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf"), "norm_overflow": 1e200}[kind]
        target.grad.reshape(-1)[0] = value
    detected = optimizer.has_nonfinite_gradients()
    expected_bad = affected and kind != "norm_overflow"
    if detected is not expected_bad:
        errors.append(f"异常检测/子组隔离错误：{detected!r}，应为 {expected_bad}")
    stepped = optimizer.step(grad_clip=args.clip_norm)
    if stepped is not (not affected):
        errors.append("异常组应统一 False，未受影响组应正常 True")
    if affected:
        if optimizer.last_grad_norm is not None:
            errors.append("异常跳过时 last_grad_norm 应为 None")
        check_tree(torch, snapshot(torch, raw, optimizer), before, errors, "skipped state unchanged", args, exact=True)
    else:
        check_number(optimizer.last_grad_norm, 0.0, errors, "zero-gradient norm", args)
        reference.step()
    model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors)
    state_errors(torch, raw, dense, optimizer, reference, args, errors)
    optimizer.zero_grad(set_to_none=True)
    if any(param.grad is not None for param in raw.parameters()):
        errors.append("异常后完整梯度未清空")
    agree(dist, rank, errors, f"{label}: {kind} consensus / unchanged skip / subgroup isolation", record)


def zero_gradient_step(torch, dist, rank, raw, dense, optimizer, reference, identities, aliases, args, label, record):
    errors = []
    for model in (raw, dense):
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
    if optimizer.step(grad_clip=0.0) is not True:
        errors.append("零梯度范数合法，应正常执行 AdamW")
    check_number(optimizer.last_grad_norm, 0.0, errors, "zero norm", args)
    reference.step()
    model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors)
    state_errors(torch, raw, dense, optimizer, reference, args, errors)
    optimizer.zero_grad(set_to_none=True)
    agree(dist, rank, errors, f"{label}: zero gradients / AdamW momentum and decay", record)


def verify_model(torch, dist, rank, GPT, GPTConfig, zero, group, members, affected, layout, kind, bias, args, device, record):
    label = f"{layout}, {kind}" + (f", bias={bias}" if kind == "GPT" else "")
    if rank == 0:
        print(f"[CASE] {label}", flush=True)
    torch.manual_seed(args.seed + rank)
    if kind == "toy":
        raw = make_toy(torch)
    else:
        raw = GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                            vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias))
    raw = raw.to(device=device, dtype=torch.float64).train()
    ddp_options = {"process_group": group}
    if device.type == "cuda":
        ddp_options.update(device_ids=[device.index], output_device=device.index)
    ddp = torch.nn.parallel.DistributedDataParallel(raw, **ddp_options)
    dense = copy.deepcopy(raw).train()
    optimizer = zero.Zero1Optimizer(raw, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                    beta1=args.betas[0], beta2=args.betas[1],
                                    dp_group=None if layout == "WORLD" else group)
    reference = make_reference_optimizer(torch, dense, args)
    identities = {name: id(param) for name, param in raw.named_parameters()}
    aliases = alias_map(raw)
    result = {"label": label, "kind": kind, "bias": bias if kind == "GPT" else None}
    result.update(partition_checks(torch, dist, rank, raw, zero, optimizer, reference, dense,
                                  group, members, args, label, record))
    errors = []
    if kind == "GPT" and raw.transformer.wte.weight is not raw.lm_head.weight:
        errors.append("GPT 词嵌入与输出层未共享 Parameter")
    if kind == "toy" and (raw.weight is not raw.tied_weight or raw.frozen.requires_grad):
        errors.append("验证小模型的共享/冻结设置异常")
    model_errors(torch, raw, dense, optimizer, identities, aliases, args, errors)
    agree(dist, rank, errors, f"{label}: initial identity / dense reference", record)
    synthetic_clip_checks(torch, dist, rank, raw, optimizer, args, label, record)
    results = []
    for event, clip in enumerate((0.0, 0.0, args.clip_norm, args.clip_norm)):
        results.append(normal_step(torch, dist, rank, kind, args, raw, ddp, dense, optimizer, reference,
                                   group, members, identities, aliases, device, event, clip,
                                   f"{label}, normal step={event}", record))
    zero_gradient_step(torch, dist, rank, raw, dense, optimizer, reference, identities, aliases, args, label, record)
    for index, anomaly in enumerate(("nan", "+inf", "-inf", "norm_overflow")):
        anomaly_step(torch, dist, rank, anomaly, raw, dense, optimizer, reference, members, affected,
                     args, identities, aliases, label, record)
        results.append(normal_step(torch, dist, rank, kind, args, raw, ddp, dense, optimizer, reference,
                                   group, members, identities, aliases, device, 4 + index, args.clip_norm,
                                   f"{label}, recovery after {anomaly}", record))
    result.update(status="passed", steps=results,
                  max_parameter_error=max(item["max_parameter_error"] for item in results),
                  local_moment_numels=results[-1]["local_moment_numels"],
                  shared_aliases={name: canonical for name, canonical in aliases.items() if name != canonical},
                  skipped_anomalies=4 if affected else 0)
    del ddp
    return result


def worker(rank, args, temporary):
    configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    record = {"rank": rank, "status": "running", "checks": [], "models": []}
    path = Path(temporary) / f"rank_{rank}.json"
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("worker 启动前源码发生变化")
        GPT, GPTConfig, zero = load_project(args.project)
        store = dist.FileStore(str(Path(temporary) / "rendezvous"), args.dp_size)
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", store=store,
                                rank=rank, world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        for layout, partitions in group_layouts(args.dp_size):
            group, members, affected = None, None, None
            for index, ranks in enumerate(partitions):
                candidate = dist.group.WORLD if layout == "WORLD" else dist.new_group(ranks=ranks, timeout=timedelta(seconds=args.timeout))
                if rank in ranks:
                    group, members, affected = candidate, ranks, index == 0
            for kind, bias in (("GPT", True), ("GPT", False), ("toy", True)):
                record["models"].append(verify_model(torch, dist, rank, GPT, GPTConfig, zero, group, members,
                                                      affected, layout, kind, bias, args, device, record))
                write_json(path, record)
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码发生变化")
        record["status"] = "passed"
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "project": str(args.project), "device": args.device, "world_size": args.dp_size,
              "threads": args.threads, "dtype": "float64", "dropout": 0.0,
              "group_layouts": group_layouts(args.dp_size),
              "scope": "ZeRO-1 core + real DDP numerical verification; production train/eval/data/checkpoint/AMP pending",
              "workers": []}
    try:
        args.expected_hashes = preflight(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / Zero1Optimizer signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only"
            print("[INFO] Static checks only; numerical verification was NOT run.", flush=True)
            return 0
        import torch
        import torch.multiprocessing as mp
        report["torch_version"] = str(torch.__version__)
        if not torch.distributed.is_available():
            raise RuntimeError("当前 PyTorch 没有分布式支持")
        if args.device == "cpu" and not torch.distributed.is_gloo_available():
            raise RuntimeError("CPU 验证需要 Gloo")
        if args.device == "cuda" and (not torch.cuda.is_available() or not torch.distributed.is_nccl_available() or torch.cuda.device_count() < args.dp_size):
            raise RuntimeError("CUDA 验证需要 NCCL，且可用设备数至少为 --dp-size；也可以使用 --device cpu")
        print(f"[INFO] {args.dp_size} {args.device} workers; threads={args.threads}; layouts={group_layouts(args.dp_size)}.", flush=True)
        print("[INFO] Checking ZeRO-1 core with real DDP; production training entry/checkpoint/AMP are pending.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_zero1_verification_") as temporary:
            try:
                mp.spawn(worker, args=(args, temporary), nprocs=args.dp_size, join=True)
            finally:
                for rank in range(args.dp_size):
                    path = Path(temporary) / f"rank_{rank}.json"
                    if path.is_file():
                        report["workers"].append(json.loads(path.read_text(encoding="utf-8")))
        if len(report["workers"]) != args.dp_size or any(item["status"] != "passed" for item in report["workers"]):
            raise AssertionError("部分 worker 未完整通过验证")
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码发生变化")
        report["status"] = "passed"
        print(f"[PASS] ZeRO-1 core verification completed ({args.device}, world_size={args.dp_size}).", flush=True)
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
