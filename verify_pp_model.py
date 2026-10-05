#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 nanoGPT 的 GPTStage 切分、完整权重加载和 eval 前向结果。

在项目根目录运行：
    python verify_pp_model.py --project . --report pp_model_report.json

项目需要：model.py（原版完整 GPT）、model_pp.py（GPTStage）、
parallel_utils.py（get_stage_layer_range）。脚本不会修改这些文件。

所有 stage 均放在一个进程里顺序执行，不初始化 distributed、不启动 worker。
pp_size 是逻辑分段数，与当前电脑的显卡数量无关。默认检查 5 层小模型的
所有分段数 1..5，并覆盖 bias=True/False；含不能整除的分层情况。

这里只检查模型结构、加载和 eval 前向/损失，不检查跨进程通信、反向调度、
DP 梯度同步或首尾共享权重的梯度同步，也不表示完整 PP 训练已经验证。
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0 等设备")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-layer", type=int, default=5)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pp-sizes", type=int, nargs="+", default=None,
                        help="要检查的逻辑分段数；默认检查 1..n_layer")
    parser.add_argument("--bias", choices=("true", "false", "both"), default="both")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("threads", "n_layer", "n_embd", "n_head", "vocab_size",
                 "block_size", "seq_len", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.n_embd % args.n_head:
        parser.error("n_embd 必须能被 n_head 整除")
    if args.seq_len > args.block_size:
        parser.error("seq_len 不能超过 block_size")
    if args.pp_sizes is None:
        args.pp_sizes = list(range(1, args.n_layer + 1))
    args.pp_sizes = sorted(set(args.pp_sizes))
    if any(p < 1 or p > args.n_layer for p in args.pp_sizes):
        parser.error("每个 pp_size 都必须满足 1 <= pp_size <= n_layer")
    return args


def configure_native_threads(threads):
    # 必须在 torch/NumPy 及项目模块导入之前设置，避免过多原生线程。
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)


def stage_keys_from_dense(dense_state, n_layer, pp_size, pp_rank):
    """独立按全局名字计算预期键；不调用项目里的分层函数。"""
    start = n_layer * pp_rank // pp_size
    end = n_layer * (pp_rank + 1) // pp_size
    selected = set()
    for name in dense_state:
        if name.startswith("transformer.h."):
            layer = int(name.split(".")[2])
            if start <= layer < end:
                selected.add(name)
        elif name.startswith(("transformer.wte.", "transformer.wpe.")):
            if pp_rank == 0:
                selected.add(name)
        elif name.startswith(("transformer.ln_f.", "lm_head.")):
            if pp_rank == pp_size - 1:
                selected.add(name)
        else:
            raise AssertionError(f"原版 GPT 出现未预期的状态键：{name}")
    return start, end, selected


def write_report(path, report):
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"[INFO] Report saved: {path}", flush=True)


def load_project(project):
    project = project.expanduser().resolve()
    for name in ("model.py", "model_pp.py", "parallel_utils.py"):
        if not (project / name).is_file():
            raise FileNotFoundError(f"找不到 {project / name}，请检查 --project 和当前分支")
    sys.path.insert(0, str(project))
    dense_module = importlib.import_module("model")
    stage_module = importlib.import_module("model_pp")
    for module, filename in ((dense_module, "model.py"), (stage_module, "model_pp.py")):
        if Path(module.__file__).resolve() != project / filename:
            raise RuntimeError(f"导入了错误的模块路径：{module.__file__}")
    return project, dense_module.GPT, dense_module.GPTConfig, stage_module.GPTStage


def verify_case(torch, args, dense_state, GPTStage, config, tokens,
                targets, ref_logits, ref_loss, pp_size):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    stages = []
    ranges = []

    for pp_rank in range(pp_size):
        stage = GPTStage(config, pp_size, pp_rank).to(device=device, dtype=dtype)
        stage.eval()
        start, end, expected_keys = stage_keys_from_dense(
            dense_state, config.n_layer, pp_size, pp_rank)
        if (stage.layer_start, stage.layer_end) != (start, end):
            raise AssertionError(f"pp_rank={pp_rank} 的层范围应为 [{start}, {end})")
        if (stage.is_first, stage.is_last) != (pp_rank == 0, pp_rank == pp_size - 1):
            raise AssertionError(f"pp_rank={pp_rank} 的首尾标记错误")

        actual_keys = set(stage.state_dict())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise AssertionError(f"pp_rank={pp_rank} 参数归属错误：missing={missing}, extra={extra}")
        loader = getattr(stage, "load_dense_state_dict", None)
        if not callable(loader):
            raise AttributeError("请先在 GPTStage 类里添加 load_dense_state_dict 方法")
        loader(dense_state)

        for name, actual in stage.state_dict().items():
            try:
                torch.testing.assert_close(actual, dense_state[name], rtol=0, atol=0)
            except AssertionError as exc:
                raise AssertionError(f"pp_rank={pp_rank} 权重加载错误：{name}") from exc
        stages.append(stage)
        ranges.append([start, end])

    if pp_size == 1:
        if stages[0].transformer["wte"].weight is not stages[0].lm_head.weight:
            raise AssertionError("pp_size=1 时 wte 和 lm_head 必须共享同一个 Parameter")
    else:
        torch.testing.assert_close(stages[0].transformer["wte"].weight,
                                   stages[-1].lm_head.weight, rtol=0, atol=0)
    label = f"bias={config.bias}, pp_size={pp_size}"
    print(f"[PASS] {label}: layer ownership / dense loading / initial tied weights", flush=True)

    with torch.no_grad():
        output = tokens
        for pp_rank, stage in enumerate(stages):
            output = stage(output)
            if not isinstance(output, torch.Tensor):
                raise AssertionError(f"pp_rank={pp_rank} 的 forward 必须返回 Tensor，实际为 {type(output).__name__}")
            last_dim = config.vocab_size if pp_rank == pp_size - 1 else config.n_embd
            expected_shape = (args.batch_size, args.seq_len, last_dim)
            if tuple(output.shape) != expected_shape:
                raise AssertionError(f"pp_rank={pp_rank} 输出形状应为 {expected_shape}，实际为 {tuple(output.shape)}")

        rtol, atol = (1e-9, 1e-10) if args.dtype == "float64" else (1e-5, 1e-6)
        torch.testing.assert_close(output, ref_logits, rtol=rtol, atol=atol)
        stage_loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, config.vocab_size), targets.reshape(-1), ignore_index=-1)
        torch.testing.assert_close(stage_loss, ref_loss, rtol=rtol, atol=atol)
        max_logits_error = (output - ref_logits).abs().max().item()
        loss_error = (stage_loss - ref_loss).abs().item()
    print(f"[PASS] {label}: full logits / loss vs dense GPT; max logits error={max_logits_error:.3e}",
          flush=True)
    return {"bias": config.bias, "pp_size": pp_size, "stage_layer_ranges": ranges,
            "max_logits_abs_error": max_logits_error, "loss_abs_error": loss_error,
            "rtol": rtol, "atol": atol, "status": "passed"}


def main(argv=None):
    args = parse_args(argv)
    configure_native_threads(args.threads)
    import torch  # 只在原生线程限制设置好之后导入。
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    project, GPT, GPTConfig, GPTStage = load_project(args.project)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 环境没有可用 CUDA")
    dtype = getattr(torch, args.dtype)
    print(f"[INFO] Single process; logical pp_sizes={args.pp_sizes}; device={device}; "
          f"dtype={args.dtype}; threads={args.threads}.", flush=True)
    print("[INFO] Checking model split/loading/eval only; distributed training is not tested.", flush=True)
    biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    report = {"status": "running", "scope": "single_process_stage_model_eval",
              "torch_version": torch.__version__, "device": str(device), "dtype": args.dtype,
              "config": {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head,
                         "vocab_size": args.vocab_size, "block_size": args.block_size,
                         "seq_len": args.seq_len, "batch_size": args.batch_size,
                         "dropout": 0.1, "mode": "eval"},
              "source_sha256": {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                                for name in ("model.py", "model_pp.py", "parallel_utils.py")},
              "not_checked": ["distributed communication", "pipeline backward schedule",
                              "data parallel synchronization", "cross-stage tied gradient synchronization"],
              "cases": []}
    try:
        for bias in biases:
            torch.manual_seed(1234)
            config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                               vocab_size=args.vocab_size, block_size=args.block_size,
                               dropout=0.1, bias=bias)
            dense = GPT(config).to(device=device, dtype=dtype)
            dense.eval()
            if dense.transformer["wte"].weight is not dense.lm_head.weight:
                raise AssertionError("作为基准的 model.py 没有原版 nanoGPT 的 wte/lm_head 权重共享")
            dense_state = dense.state_dict()
            tokens = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), device=device)
            targets = torch.randint(config.vocab_size, tokens.shape, device=device)
            if targets.numel() > 1:
                targets[0, -1] = -1
            with torch.no_grad():
                # 必须传 targets：原版 GPT 不传 targets 时只返回最后一个位置的 logits。
                ref_logits, ref_loss = dense(tokens, targets)
            for pp_size in args.pp_sizes:
                report["cases"].append(verify_case(
                    torch, args, dense_state, GPTStage, config, tokens,
                    targets, ref_logits, ref_loss, pp_size))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(args.report, report)
        raise
    report["status"] = "passed"
    write_report(args.report, report)
    print("[PASS] PP stage model verification completed (single-process eval).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
