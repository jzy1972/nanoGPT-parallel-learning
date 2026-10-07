#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证手写 nanoGPT ZeRO-1 训练函数及实际 main；不修改项目源码。

把本文件放在含 model.py、zero1.py、train_zero.py 的项目根目录，运行：
  python verify_zero1_training.py --project . --device cpu --dp-size 4 --threads 1 --report zero1_training_report.json

只依赖标准库、NumPy 和 PyTorch，不需要旧 TP/PP 或 verify_zero1.py 文件。
真实调用项目 get_zero_batch()、train_zero_step()、parse_args() 和 main()。
使用真实 DDP、真实手写 Zero1Optimizer；float64 参考为普通 AdamW
训练拼接的全局 batch。记录器不替代被测函数的训练数学逻辑。

训练函数：float64、dropout=0，测试 WORLD、单进程子组及奇偶子组；
GPT bias=True/False，以及共享/冻结参数和空优化器 rank 的小模型。
检查前向、loss、累积梯度、forward/backward 的 no_sync 标志、更新次数、
范数/裁剪、完整参数、AdamW step/一阶/二阶矩与梯度清理。检查最后一个
micro-batch 较小、单 micro-batch、NaN/+Inf/-Inf/范数溢出跳过和恢复。

实际 main：每个配置启动全新 worker，设置真实 env:// 启动环境，保留
实际进程组初始化/销毁。自动创建临时 uint16 train.bin，测试 FP32 训练，
两种 bias 配置、零/正裁剪、非默认超参数、数据生成器、全局 loss、
参数/矩状态及完整迭代日志；等待 main 正常结束后核对日志。FP32 先
独立核对全局 dense 前向/梯度，AdamW 参考再使用已核对的同步梯度，
隔离近零梯度误差经 eps 放大的影响；参数更新与矩状态仍独立计算。

CPU 使用 Gloo；函数检查使用临时 FileStore，main 使用 127.0.0.1 的
TCP rendezvous、USE_LIBUV=0。CUDA 单机每 worker 一个设备，需要 NCCL。
--dp-size 是验证进程数，不限制实现规模；设为 1 仅覆盖单 rank 路径。
不验证 eval、checkpoint/恢复、AMP、torch.compile、多机启动、性能或
内存峰值。--preflight-only 只做静态检查，不代表数值验证通过。
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


TRAINER_SIGNATURES = {
    "parse_args": (),
    "get_zero_batch": ("data_path", "batch_size", "seq_length", "device", "generator"),
    "train_zero_step": ("model", "optimizer", "x", "y", "micro_batch_size", "grad_clip"),
    "main": (),
}
LOG_PATTERN = re.compile(
    r"^iter\s+(\d+):\s+loss=(\S+),\s+grad_norm=(\S+),\s+stepped=(True|False)\s*$"
)


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
    parser.add_argument("--fp32-rtol", type=float, default=3e-4)
    parser.add_argument("--fp32-atol", type=float, default=3e-6)
    parser.add_argument("--main-steps", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("dp_size", "threads", "timeout", "n_layer", "n_head", "n_embd", "vocab_size", "block_size",
                 "seq_len", "batch_size", "micro_batch_size", "main_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not 2 <= args.vocab_size <= 65536 or args.n_embd % args.n_head or args.seq_len > args.block_size:
        parser.error("需要 2<=vocab_size<=65536、n_embd 能被 n_head 整除、seq_len<=block_size")
    for name in ("clip_norm", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")
    for name in ("weight_decay", "rtol", "atol", "fp32_rtol", "fp32_atol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负的有限数")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in args.betas):
        parser.error("betas 必须在 [0,1) 内")
    return args


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


def serial_checks(torch, np, trainer, args, temporary):
    report = {"argument_cases": 0, "loader_cases": []}
    base = ["--device-type", "cpu", "--n-layer", "2", "--n-head", "2", "--n-embd", "16",
            "--vocab-size", "32", "--block-size", "8", "--seq-length", "6", "--batch-size", "5",
            "--micro-batch-size", "2", "--max-iters", "4", "--log-interval", "2"]
    with command_line(base + ["--grad-clip", "0", "--no-bias"]):
        parsed = trainer.parse_args()
    wanted = {"device_type": "cpu", "n_layer": 2, "n_head": 2, "n_embd": 16, "vocab_size": 32,
              "block_size": 8, "seq_length": 6, "batch_size": 5, "micro_batch_size": 2,
              "max_iters": 4, "log_interval": 2, "grad_clip": 0.0, "no_bias": True}
    for name, value in wanted.items():
        if getattr(parsed, name, None) != value:
            raise AssertionError(f"parse_args: {name} 应为 {value!r}")
    invalid = [("--batch-size", "0"), ("--micro-batch-size", "0"), ("--max-iters", "0"),
               ("--log-interval", "0"), ("--grad-clip", "-1"), ("--grad-clip", "nan"),
               ("--grad-clip", "inf"), ("--seq-length", "0"), ("--seq-length", "9"),
               ("--n-layer", "0"), ("--n-head", "0"), ("--n-embd", "15")]
    for flag, value in invalid:
        with command_line(base + [flag, value]), redirect_stderr(io.StringIO()):
            try:
                trainer.parse_args()
            except (ValueError, SystemExit) as exc:
                if isinstance(exc, SystemExit) and exc.code != 2:
                    raise AssertionError(f"{flag}={value} 应由参数检查拒绝") from exc
            else:
                raise AssertionError(f"parse_args 没有拒绝 {flag}={value}")
        report["argument_cases"] += 1
    directory = Path(temporary) / "serial_data"
    directory.mkdir()
    for count, length, samples in ((79, 6, 5), (7, 6, 5)):
        path = directory / "train.bin"
        data = np.random.default_rng(args.seed + count).integers(0, 32, count, dtype=np.uint16)
        data.tofile(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_generator = torch.Generator(device="cpu").manual_seed(args.seed + 7)
        expected_generator = torch.Generator(device="cpu").manual_seed(args.seed + 7)
        for event in range(3):
            expected_x, expected_y, starts = oracle_batch(torch, np, data, samples, length, expected_generator, torch.device("cpu"))
            actual_x, actual_y = trainer.get_zero_batch(data_path=str(path), batch_size=samples,
                                                       seq_length=length, device=torch.device("cpu"), generator=actual_generator)
            torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
            torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
            if actual_x.dtype != torch.long or actual_y.dtype != torch.long or actual_x.device.type != "cpu" or actual_y.device.type != "cpu":
                raise AssertionError("get_zero_batch 应返回指定设备上的 long 张量")
            if not torch.equal(actual_generator.get_state(), expected_generator.get_state()):
                raise AssertionError("get_zero_batch 没有正确推进传入的 CPU generator")
            report["loader_cases"].append({"tokens": count, "length": length, "event": event, "starts": starts})
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError("get_zero_batch 修改了 train.bin")
    for label, payload, count, length in (("too_short", np.arange(6, dtype=np.uint16).tobytes(), 5, 6),
                                          ("empty", b"", 5, 6), ("odd_bytes", b"abc", 5, 1),
                                          ("batch_zero", np.arange(9, dtype=np.uint16).tobytes(), 0, 6),
                                          ("seq_zero", np.arange(9, dtype=np.uint16).tobytes(), 5, 0)):
        path = directory / f"{label}.bin"
        path.write_bytes(payload)
        try:
            trainer.get_zero_batch(data_path=str(path), batch_size=count, seq_length=length,
                                   device=torch.device("cpu"), generator=torch.Generator(device="cpu"))
        except (ValueError, RuntimeError):
            report["loader_cases"].append({"rejected": label})
        else:
            raise AssertionError(f"get_zero_batch 没有拒绝 {label}")
    print("[PASS] actual parse_args / CPU data loading / boundary and generator checks.", flush=True)
    return report


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


def direct_model(torch, dist, rank, GPT, GPTConfig, zero, trainer, group, members, affected, layout, kind, bias, args, device, record):
    label = f"function {layout}, {kind}" + (f", bias={bias}" if kind == "GPT" else "")
    if rank == 0:
        print(f"[CASE] {label}", flush=True)
    torch.manual_seed(args.seed + rank)
    raw = (make_toy(torch) if kind == "toy" else GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head,
           n_embd=args.n_embd, vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias)))
    raw = raw.to(device=device, dtype=torch.float64)
    options = {"process_group": group}
    if device.type == "cuda":
        options.update(device_ids=[device.index], output_device=device.index)
    ddp = torch.nn.parallel.DistributedDataParallel(raw, **options)
    dense = copy.deepcopy(raw)
    optimizer = zero.Zero1Optimizer(raw, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                    beta1=args.betas[0], beta2=args.betas[1], dp_group=group)
    reference = make_reference_optimizer(torch, dense, args)
    identities, aliases = {name: id(param) for name, param in raw.named_parameters()}, alias_map(raw)
    result = {"label": label}
    result.update(partition_checks(torch, dist, rank, raw, zero, optimizer, reference, dense,
                                  group, members, args, label, record))
    steps = []
    plans = [(args.micro_batch_size, 0.0, None), (1, 0.0, None),
             (args.batch_size + 1, args.clip_norm, None), (args.micro_batch_size, args.clip_norm, None)]
    for anomaly in ("nan", "+inf", "-inf", "norm_overflow"):
        plans += [(args.micro_batch_size, args.clip_norm, anomaly), (args.micro_batch_size, args.clip_norm, None)]
    for event, (micro, clip, anomaly) in enumerate(plans):
        full_x, full_y = batch(torch, kind, args, len(members), event, members[0], device)
        start = members.index(rank) * args.batch_size
        local_x, local_y = full_x[start:start + args.batch_size], full_y[start:start + args.batch_size]
        details, _ = execute_step(torch, dist, rank, trainer.train_zero_step, ddp, raw, dense, optimizer, reference,
                                  local_x, local_y, local_x, local_y, full_x, full_y, micro, clip, kind, group,
                                  members, affected, anomaly, identities, aliases, args, f"{label}, step={event}", record)
        steps.append(details)
    result.update(status="passed", steps=steps, max_parameter_error=max(step["max_parameter_error"] for step in steps))
    return result


def setup_worker(torch, args, rank):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def function_worker(rank, args, temporary):
    configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "status": "running", "checks": [], "models": []}
    path = Path(temporary) / f"rank_{rank}.json"
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("worker 启动前源码发生变化")
        GPT, GPTConfig, zero, trainer = load_project(args.project)
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
                record["models"].append(direct_model(torch, dist, rank, GPT, GPTConfig, zero, trainer, group, members,
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


def main_specs(args):
    result = []
    for index, bias in enumerate((True, False)):
        result.append({"index": index, "bias": bias, "seed": args.seed + (index + 1) * 101,
                       "n_layer": args.n_layer + index, "n_head": args.n_head, "n_embd": args.n_embd,
                       "vocab_size": min(args.vocab_size + index, 65536), "block_size": args.block_size,
                       "seq_length": args.seq_len, "batch_size": args.batch_size,
                       "micro_batch_size": args.micro_batch_size, "grad_clip": 0.0 if bias else args.clip_norm,
                       "max_iters": args.main_steps, "log_interval": 2,
                       "learning_rate": args.learning_rate * 1.25, "weight_decay": args.weight_decay + 0.03,
                       "beta1": 0.85, "beta2": 0.97})
    return result


def main_argv(spec, data_dir, device):
    result = ["--device-type", device, "--data-dir", str(data_dir)]
    for name in ("seed", "n_layer", "n_head", "n_embd", "vocab_size", "block_size", "seq_length", "batch_size",
                 "micro_batch_size", "grad_clip", "max_iters", "log_interval", "learning_rate", "weight_decay", "beta1", "beta2"):
        result += ["--" + name.replace("_", "-"), str(spec[name])]
    result += ["--dropout", "0"]
    if not spec["bias"]:
        result.append("--no-bias")
    return result


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


def main_worker(rank, args, temporary, spec, port):
    configure_threads(args.threads)
    import numpy as np
    import torch
    import torch.distributed as dist
    device = setup_worker(torch, args, rank)
    record = {"rank": rank, "status": "running", "checks": [], "steps": [], "case": spec}
    path = Path(temporary) / f"rank_{rank}.json"
    saved_init, saved_destroy = dist.init_process_group, dist.destroy_process_group
    try:
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("main worker 启动前源码发生变化")
        GPT, GPTConfig, zero, trainer = load_project(args.project)
        observed = {"ddp": [], "optimizers": [], "batch_calls": [], "init_calls": 0, "destroy_calls": 0}
        data_path = Path(temporary) / "data" / "train.bin"
        data = np.fromfile(data_path, dtype=np.uint16)
        data_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        expected_generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 10_000 + rank)
        tolerance = copy.copy(args)
        tolerance.rtol, tolerance.atol = args.fp32_rtol, args.fp32_atol
        tolerance.learning_rate, tolerance.weight_decay = spec["learning_rate"], spec["weight_decay"]
        tolerance.betas = (spec["beta1"], spec["beta2"])
        tolerance.reference_from_validated_gradients = True
        original_ddp, original_zero = torch.nn.parallel.DistributedDataParallel, zero.Zero1Optimizer
        original_batch, original_train = trainer.get_zero_batch, trainer.train_zero_step

        def observed_init(*positional, **keyword):
            observed["init_calls"] += 1
            # 只缩短验证超时；后端、env://、设备、rank/world_size 都保留实际 main 的参数。
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
                raw = self.module
                errors = []
                if dist.get_world_size(group=self.process_group) != args.dp_size:
                    errors.append("main 的 DDP 组大小错误")
                config = raw.config
                for name in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "bias"):
                    if getattr(config, name) != spec[name]:
                        errors.append(f"main 的 GPTConfig.{name} 没有采用命令行参数")
                if config.dropout != 0.0:
                    errors.append("main 未采用 dropout=0 的验证配置")
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(spec["seed"])
                    dense = GPT(GPTConfig(n_layer=spec["n_layer"], n_head=spec["n_head"], n_embd=spec["n_embd"],
                                vocab_size=spec["vocab_size"], block_size=spec["block_size"], dropout=0.0, bias=spec["bias"]))
                    dense = dense.to(device=device, dtype=torch.float32)
                check_tree(torch, raw.state_dict(), dense.state_dict(), errors, "main initial seed/weights", tolerance, exact=True)
                if any(param.device != device or param.dtype != torch.float32 for param in raw.parameters()):
                    errors.append("main 应使用指定设备上的 float32 参数")
                observed.update(raw=raw, model=self, dense=dense, reference=make_reference_optimizer(torch, dense, tolerance),
                                identities={name: id(param) for name, param in raw.named_parameters()}, aliases=alias_map(raw))
                agree(dist, rank, errors, "actual main: DDP / CLI config / initial seed and weights", record)

        class TracedZero(original_zero):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                observed["optimizers"].append(self)
                errors = []
                if "raw" not in observed or self.model is not observed["raw"]:
                    errors.append("main 的 Zero1Optimizer 没有引用 DDP 对应的 raw_model")
                if self.dp_size != args.dp_size or self.dp_rank != rank:
                    errors.append("main 的优化器 DP rank/size 错误")
                agree(dist, rank, errors, "actual main: optimizer references / DP group", record)
                record["partition"] = partition_checks(torch, dist, rank, observed["raw"], zero, self, observed["reference"],
                                                        observed["dense"], self.dp_group, list(range(args.dp_size)), tolerance,
                                                        "actual main", record)

        def observed_batch(*positional, **keyword):
            bound = bind_call(original_batch, positional, keyword)
            observed["batch_calls"].append({"path": os.path.abspath(bound["data_path"]), "batch_size": bound["batch_size"],
                                            "seq_length": bound["seq_length"], "generator": bound["generator"]})
            return original_batch(*positional, **keyword)

        def observed_train(*positional, **keyword):
            bound = bind_call(original_train, positional, keyword)
            errors = []
            event = len(record["steps"])
            if event >= spec["max_iters"]:
                errors.append("main 执行了超过 --max-iters 的训练次数")
            if bound["model"] is not observed.get("model") or bound["optimizer"] not in observed["optimizers"]:
                errors.append("main 传入训练函数的模型/优化器对象错误")
            if bound["micro_batch_size"] != spec["micro_batch_size"] or bound["grad_clip"] != spec["grad_clip"]:
                errors.append("main 没有传递 micro_batch_size 或 grad_clip 命令行参数")
            for name in ("x", "y"):
                if bound[name].device != device or bound[name].dtype != torch.long:
                    errors.append(f"main 的 {name} 设备/dtype 错误")
            agree(dist, rank, errors, f"actual main, iter={event}: training call wiring", record)
            expected_x, expected_y, starts = oracle_batch(torch, np, data, spec["batch_size"], spec["seq_length"],
                                                        expected_generator, device)
            group = bound["optimizer"].dp_group
            all_x, all_y = [torch.empty_like(expected_x) for _ in range(args.dp_size)], [torch.empty_like(expected_y) for _ in range(args.dp_size)]
            dist.all_gather(all_x, expected_x, group=group)
            dist.all_gather(all_y, expected_y, group=group)
            result, original_result = execute_step(torch, dist, rank, original_train, observed["model"], observed["raw"], observed["dense"],
                                  bound["optimizer"], observed["reference"], bound["x"], bound["y"], expected_x, expected_y,
                                  torch.cat(all_x), torch.cat(all_y), spec["micro_batch_size"], spec["grad_clip"], "GPT", group,
                                  list(range(args.dp_size)), True, None, observed["identities"], observed["aliases"], tolerance,
                                  f"actual main, iter={event}", record)
            result["sample_starts"] = starts
            record["steps"].append(result)
            write_json(path, record)
            return original_result

        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(args.dp_size),
                          MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
        changes = [(dist, "init_process_group", observed_init), (dist, "destroy_process_group", observed_destroy),
                   (torch.nn.parallel, "DistributedDataParallel", TracedDDP), (zero, "Zero1Optimizer", TracedZero),
                   (trainer, "get_zero_batch", observed_batch), (trainer, "train_zero_step", observed_train)]
        for name, value in (("DDP", TracedDDP), ("Zero1Optimizer", TracedZero)):
            if hasattr(trainer, name):
                changes.append((trainer, name, value))
        buffer = io.StringIO()
        try:
            with patched_attributes(changes), command_line(main_argv(spec, data_path.parent, args.device)), redirect_stdout(buffer):
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
            errors.append(f"main 的初始化/销毁次数错误：{observed['init_calls']}/{observed['destroy_calls']}")
        if len(observed["ddp"]) != 1 or len(observed["optimizers"]) != 1:
            errors.append("main 应创建一组训练模型与优化器")
        if len(record["steps"]) != spec["max_iters"] or len(observed["batch_calls"]) != spec["max_iters"]:
            errors.append("main 的更新尝试次数或取 batch 次数与 --max-iters 不一致")
        previous_generator = None
        for item in observed["batch_calls"]:
            if item["path"] != os.path.abspath(data_path) or item["batch_size"] != spec["batch_size"] or item["seq_length"] != spec["seq_length"]:
                errors.append("main 传入 get_zero_batch 的路径或尺寸错误")
            generator = item["generator"]
            if generator.device.type != "cpu" or generator.initial_seed() != spec["seed"] + 10_000 + rank:
                errors.append("main 的训练数据生成器设备/种子错误")
            if previous_generator is not None and generator is not previous_generator:
                errors.append("main 在迭代间重新创建数据生成器")
            previous_generator = generator
        if previous_generator is not None and not torch.equal(previous_generator.get_state(), expected_generator.get_state()):
            errors.append("main 的训练数据 generator 末状态不符合真实调用次数")
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != data_digest:
            errors.append("main 修改了验证 train.bin")
        compare_logs(captured, rank, spec, record["steps"], tolerance, errors)
        if fingerprints(args.project) != args.expected_hashes:
            errors.append("实际 main 验证期间项目源码发生变化")
        # main 的实际组已销毁；新的组仅用于汇总验证结果，不能掩盖清理缺失。
        store = dist.FileStore(str(Path(temporary) / "post_main_results"), args.dp_size)
        saved_init("nccl" if args.device == "cuda" else "gloo", store=store, rank=rank,
                   world_size=args.dp_size, timeout=timedelta(seconds=args.timeout))
        agree(dist, rank, errors, "actual main: complete logs / data RNG / iteration count / cleanup", record)
        record.update(status="passed", init_calls=observed["init_calls"], destroy_calls=observed["destroy_calls"])
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    finally:
        if dist.is_initialized():
            saved_destroy()


def unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_workers(mp, worker, args, directory, extra=()):
    directory.mkdir(parents=True)
    workers = []
    try:
        mp.spawn(worker, args=(args, str(directory), *extra), nprocs=args.dp_size, join=True)
    finally:
        for rank in range(args.dp_size):
            path = directory / f"rank_{rank}.json"
            if path.is_file():
                workers.append(json.loads(path.read_text(encoding="utf-8")))
        args.collected_workers = workers
    if len(workers) != args.dp_size or any(item["status"] != "passed" for item in workers):
        raise AssertionError("部分 worker 没有完整通过验证")
    return workers


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "project": str(args.project),
              "device": args.device, "world_size": args.dp_size, "threads": args.threads,
              "function_dtype": "float64", "main_dtype": "float32", "dropout": 0.0,
              "main_reference_policy": "independent global dense forward/gradient check; ordinary AdamW from numerically verified synchronized gradients",
              "group_layouts": group_layouts(args.dp_size), "function_workers": [], "actual_main": [],
              "scope": "ZeRO-1 train_zero_step + data/CLI + actual main; eval/checkpoint/AMP/compile/multinode/performance pending"}
    try:
        args.expected_hashes = preflight(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / training and optimizer signatures (static only).", flush=True)
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
            raise RuntimeError("CUDA 验证需要 NCCL，且设备数至少为 --dp-size；也可以用 --device cpu")
        torch.set_num_threads(args.threads)
        _, _, _, trainer = load_project(args.project)
        print(f"[INFO] {args.dp_size} {args.device} workers; threads={args.threads}; tiny synthetic data; dropout=0.", flush=True)
        print("[INFO] Function tests use float64; actual main tests retain float32 and real env:// init.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_zero1_training_") as temporary:
            root = Path(temporary)
            report["serial_checks"] = serial_checks(torch, np, trainer, args, root)
            try:
                report["function_workers"] = run_workers(mp, function_worker, args, root / "functions")
            except BaseException:
                report["function_workers"] = getattr(args, "collected_workers", [])
                raise
            for spec in main_specs(args):
                directory = root / f"main_{spec['index']}"
                data_dir = directory / "data"
                data_dir.mkdir(parents=True)
                np.random.default_rng(spec["seed"] + 91).integers(0, spec["vocab_size"], max(257, spec["seq_length"] * 5 + 3), dtype=np.uint16).tofile(data_dir / "train.bin")
                if spec["index"] == 0:
                    print("[CASE] actual main, bias=True, grad_clip=0", flush=True)
                else:
                    print(f"[CASE] actual main, bias=False, grad_clip={spec['grad_clip']}", flush=True)
                # directory 已含 data；worker 日志目录复用它。
                workers = []
                try:
                    mp.spawn(main_worker, args=(args, str(directory), spec, unused_port()), nprocs=args.dp_size, join=True)
                finally:
                    for rank in range(args.dp_size):
                        path = directory / f"rank_{rank}.json"
                        if path.is_file():
                            workers.append(json.loads(path.read_text(encoding="utf-8")))
                    report["actual_main"].append({"case": spec, "workers": workers})
                if len(workers) != args.dp_size or any(item["status"] != "passed" for item in workers):
                    raise AssertionError("实际 main 的部分 worker 未完整通过")
        if fingerprints(args.project) != args.expected_hashes:
            raise RuntimeError("验证期间源码发生变化")
        report["status"] = "passed"
        print(f"[PASS] ZeRO-1 training verification completed ({args.device}, world_size={args.dp_size}).", flush=True)
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
