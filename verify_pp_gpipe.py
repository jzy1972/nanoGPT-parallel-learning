#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 GPipe 多 microbatch 调度（CPU/Gloo，float64，小模型）。

将本文件与 verify_pp_microbatch.py 一起放到 nanoGPT 根目录，运行：
    python verify_pp_gpipe.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_gpipe_report.json

项目需要 train_pp.py 中的 gpipe_forward_backward，以及之前的前向、反向
通信函数。验证器直接调用用户函数，不执行 train_pp.main，不修改源码。

默认固定 4 个 CPU worker，检查 (PP,DP)=(1,4),(2,2),(4,1)。这是逻辑
测试配置，不限制实现的设备数。默认每条流水线 batch=5，micro_batch_size
为 1、2、5，覆盖一块、多块、最后一块较小及不能整除的模型分层。

检查本实现约定的“全部前向，再逆序反向”、每块损失权重、标签切片、
完整 logits/边界激活、激活梯度、整批 loss、所有本地参数梯度。每组数据
连续调用调度两次，中间不清零，检查跨调用累积。还覆盖某一 microbatch
全是忽略标签，以及各块有效标签数不相等。不同 DP 流水线使用不同数据。

参考为项目 model.py 的原版 GPT。参考副本另外拆开首尾共享 Parameter，
检查两处独立梯度；验证器收集首尾贡献求和，对照共享 GPT 的总梯度。
这不替用户实现训练中的 DP 或 tied-weight 梯度同步。

不检查 optimizer、DP 梯度平均、训练时首尾梯度同步、AMP、CUDA/NCCL、
数据加载、checkpoint、torchrun 启动或 train_pp.main；不是速度测试。
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
import sys
import tempfile

try:
    import verify_pp_microbatch as common
except ModuleNotFoundError as exc:
    if exc.name == "verify_pp_microbatch":
        raise SystemExit("请把 verify_pp_microbatch.py 与本文件放在同一目录。") from exc
    raise


GPIPE_ARGUMENTS = ("stage", "token_ids", "targets", "parallel", "batch_size",
                   "seq_length", "micro_batch_size", "device", "activation_dtype")


def microbatch_bounds(batch_size, micro_batch_size):
    return [(start, min(start + micro_batch_size, batch_size))
            for start in range(0, batch_size, micro_batch_size)]


def expected_events(count):
    return [("F", m) for m in range(count)] + [("B", m) for m in reversed(range(count))]


def trace_errors(events, count):
    expected = expected_events(count)
    return [] if events == expected else [f"GPipe 调用顺序应为 {expected}，实际为 {events}"]


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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--bias", choices=("true", "false", "both"), default="both")
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="只检查源码语法/函数签名，不导入 torch、不启动 worker")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd",
                 "vocab_size", "block_size", "seq_len", "batch_size", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.pp_size > args.n_layer or args.n_embd % args.n_head:
        parser.error("需要 pp_size <= n_layer，且 n_embd 能被 n_head 整除")
    if args.seq_len > args.block_size:
        parser.error("seq_len 不能超过 block_size")
    if args.vocab_size < 2 or args.batch_size * args.seq_len < 2:
        parser.error("本测试需要至少两个词表项、两个标签位置")
    if args.micro_batch_sizes is None:
        args.micro_batch_sizes = [1, 2, args.batch_size]
    args.micro_batch_sizes = sorted(set(args.micro_batch_sizes))
    if any(size <= 0 for size in args.micro_batch_sizes):
        parser.error("每个 micro_batch_size 必须为正整数")
    world_size = args.pp_size * args.dp_size
    args.cases = ([(args.pp_size, args.dp_size)] if args.single_case else
                  [(p, world_size // p) for p in range(1, min(args.n_layer, world_size) + 1)
                   if world_size % p == 0])
    args.biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    return args


def fingerprints(project):
    result = common.source_hashes(project)
    result["checker:verify_pp_microbatch.py"] = hashlib.sha256(
        Path(common.__file__).read_bytes()).hexdigest()
    result["checker:verify_pp_gpipe.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def preflight(project):
    expected_dependency = Path(__file__).resolve().parent / "verify_pp_microbatch.py"
    if Path(common.__file__).resolve() != expected_dependency:
        raise RuntimeError(f"验证辅助文件应来自 {expected_dependency}，实际为 {common.__file__}")
    common.preflight(project)
    path = project / "train_pp.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "gpipe_forward_backward"]
    if len(nodes) != 1:
        raise AttributeError("train_pp.py 需要定义一个顶层 gpipe_forward_backward()")
    node = nodes[0]
    supplied = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
    missing = set(GPIPE_ARGUMENTS) - supplied
    if missing:
        raise TypeError(f"gpipe_forward_backward 缺少参数 {sorted(missing)}")


def agree(torch, dist, rank, errors, label, quiet=False):
    flag = torch.tensor(int(bool(errors)), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        details = [None] * dist.get_world_size()
        dist.all_gather_object(details, errors[:12])
        if rank == 0:
            for r, messages in enumerate(details):
                for message in messages:
                    print(f"[FAIL rank={r}] {label}: {message}", file=sys.stderr, flush=True)
        raise AssertionError(f"验证失败：{label}")
    if rank == 0 and not quiet:
        print(f"[PASS] {label}", flush=True)


def make_targets(torch, args, dp_rank, call_index, micro_size, pattern):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 10000 + 997 * dp_rank + 37 * call_index)
    x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), generator=generator)
    targets = torch.randint(args.vocab_size, x.shape, generator=generator)
    last_label = int(targets[-1, -1].item())
    if pattern == "unequal_valid":
        targets[0, 0] = -1
        if args.batch_size > 1:
            targets[1, :min(3, args.seq_len)] = -1
    elif pattern == "empty_microbatch":
        targets[:micro_size] = -1
    else:
        raise ValueError(f"未知测试 pattern：{pattern}")
    # 整批至少保留一个有效标签；允许其中一整块没有有效标签。
    targets[-1, -1] = last_label
    return x, targets


def reference_round(torch, shared, reference, x, targets, layer_start, layer_end,
                    pp_rank, pp_size, errors):
    captures, handles = {}, []
    def capture(key):
        def hook(module, inputs, output):
            output.retain_grad()
            captures[key] = output
        return hook
    if pp_rank > 0:
        handles.append(reference.transformer.h[layer_start - 1].register_forward_hook(capture("input")))
    if pp_rank < pp_size - 1:
        handles.append(reference.transformer.h[layer_end - 1].register_forward_hook(capture("output")))
    try:
        shared_logits, shared_loss = shared(x, targets)
        ref_logits, ref_loss = reference(x, targets)
        if pp_rank == pp_size - 1:
            ref_logits.retain_grad()
            captures["output"] = ref_logits
        shared_loss.backward()
        ref_loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    common.check_close(torch, ref_logits, shared_logits, "reference logits", errors)
    common.check_close(torch, ref_loss, shared_loss, "reference loss", errors)
    common.check_close(torch, reference.transformer.wte.weight.grad + reference.lm_head.weight.grad,
                       shared.transformer.wte.weight.grad, "reference tied-gradient sum", errors)
    return captures, ref_loss.detach()


def observe_schedule(torch, trainer, stage, info, args, micro_size, x, targets,
                     captures, ref_loss, errors, call_label):
    bounds = microbatch_bounds(args.batch_size, micro_size)
    events, output_ids, seen_gradients = [], {}, set()
    next_index = 0
    total_valid = int((targets != -1).sum().item())
    original_forward = trainer.forward_microbatch
    original_backward = trainer.backward_microbatch
    forward_signature = inspect.signature(original_forward)
    backward_signature = inspect.signature(original_backward)

    def gradient_hook(key, expected):
        def hook(gradient):
            seen_gradients.add(key)
            common.check_close(torch, gradient, expected, f"{call_label}: {key} gradient", errors)
            return None
        return hook

    def observed_forward(*positional, **keywords):
        nonlocal next_index
        m = next_index
        next_index += 1
        events.append(("F", m))
        arguments = forward_signature.bind(*positional, **keywords)
        arguments.apply_defaults()
        if m >= len(bounds):
            errors.append("调用的 forward_microbatch 数量超过预期")
            return original_forward(*positional, **keywords)
        start, end = bounds[m]
        if arguments.arguments["micro_batch_size"] != end - start:
            errors.append(f"mb={m}: 接收形状应使用实际大小 {end - start}")
        stage_input, stage_output = original_forward(*positional, **keywords)
        output_ids[id(stage_output)] = m
        expected_input = x[start:end] if stage.is_first else captures["input"][start:end]
        common.check_close(torch, stage_input, expected_input, f"mb={m}: forward input", errors)
        common.check_close(torch, stage_output, captures["output"][start:end],
                           f"mb={m}: forward output / logits", errors)
        if stage_output.requires_grad:
            stage_output.register_hook(gradient_hook(("output", m),
                captures["output"].grad[start:end].detach()))
        else:
            errors.append(f"mb={m}: stage_output 丢失计算图")
        if not stage.is_first:
            if stage_input.is_leaf and stage_input.requires_grad:
                stage_input.register_hook(gradient_hook(("input", m),
                    captures["input"].grad[start:end].detach()))
            else:
                errors.append(f"mb={m}: 接收激活不是可求导叶子张量")
        return stage_input, stage_output

    def observed_backward(*positional, **keywords):
        arguments = backward_signature.bind(*positional, **keywords)
        arguments.apply_defaults()
        m = output_ids.get(id(arguments.arguments["stage_output"]), -1)
        events.append(("B", m))
        if m < 0:
            errors.append("反向传播使用了不属于本次前向缓存的 stage_output")
        elif stage.is_last:
            start, end = bounds[m]
            micro_targets = targets[start:end]
            common.check_close(torch, arguments.arguments["targets"], micro_targets,
                               f"mb={m}: target slice", errors)
            expected_weight = int((micro_targets != -1).sum().item()) / total_valid
            actual_weight = float(arguments.arguments["loss_weight"])
            if not math.isclose(actual_weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12):
                errors.append(f"mb={m}: loss_weight 应为 {expected_weight}，实际为 {actual_weight}")
        return original_backward(*positional, **keywords)

    trainer.forward_microbatch = observed_forward
    trainer.backward_microbatch = observed_backward
    try:
        actual_loss = trainer.gpipe_forward_backward(
            stage=stage, token_ids=x if stage.is_first else None,
            targets=targets if stage.is_last else None, parallel=info,
            batch_size=args.batch_size, seq_length=args.seq_len, micro_batch_size=micro_size,
            device=torch.device("cpu"), activation_dtype=torch.float64)
    finally:
        trainer.forward_microbatch = original_forward
        trainer.backward_microbatch = original_backward
    errors.extend(trace_errors(events, len(bounds)))
    expected_gradients = {("output", m) for m in range(len(bounds))}
    if not stage.is_first:
        expected_gradients |= {("input", m) for m in range(len(bounds))}
    if seen_gradients != expected_gradients:
        errors.append(f"缺少激活梯度：{sorted(expected_gradients - seen_gradients)}")
    if stage.is_last:
        common.check_close(torch, actual_loss, ref_loss, "returned whole-batch mean loss", errors)
        if actual_loss is not None and actual_loss.requires_grad:
            errors.append("返回的 batch loss 应已 detach")
    elif actual_loss is not None:
        errors.append("非尾 stage 应返回 None loss")
    return {"reference_loss": float(ref_loss.item()),
            "returned_loss": (float(actual_loss.item()) if actual_loss is not None else None),
            "events": events}


def verify_trial(torch, dist, rank, args, trainer, shared, reference, stage, info,
                 initial_state, pp_size, dp_size, bias, micro_size, pattern):
    p, d = rank % pp_size, rank // pp_size
    label = f"PP={pp_size}, DP={dp_size}, bias={bias}, micro_batch={micro_size}, {pattern}"
    for model in (shared, reference, stage):
        model.zero_grad(set_to_none=True)
    losses, errors = [], []
    # 两次调度之间不清零，检查调度函数保留外层累积的梯度。
    for call_index in range(2):
        x, targets = make_targets(torch, args, d, call_index, micro_size, pattern)
        captures, ref_loss = reference_round(torch, shared, reference, x, targets,
            stage.layer_start, stage.layer_end, p, pp_size, errors)
        agree(torch, dist, rank, errors, f"{label}: reference graph", quiet=True)
        losses.append(observe_schedule(torch, trainer, stage, info, args, micro_size, x, targets,
                                       captures, ref_loss, errors, f"call={call_index}"))
        expected_parameters = dict((shared if pp_size == 1 else reference).named_parameters())
        for name, parameter in stage.named_parameters():
            expected = expected_parameters.get(name)
            if expected is None:
                errors.append(f"参考模型缺少 {name}")
            else:
                common.check_close(torch, parameter.grad, expected.grad,
                                   f"call={call_index}: accumulated gradient {name}", errors)
        agree(torch, dist, rank, errors, f"{label}: GPipe order / loss / gradients", quiet=True)

    max_gradient_error = 0.0
    for name, parameter in stage.named_parameters():
        max_gradient_error = max(max_gradient_error,
            float((parameter.grad - expected_parameters[name].grad).abs().max().item()))
    for name, value in stage.state_dict().items():
        if not torch.equal(value, initial_state[name]):
            errors.append(f"只做 forward/backward 时修改了参数或 buffer：{name}")
    contribution = torch.zeros((args.vocab_size, args.n_embd), dtype=torch.float64)
    if p == 0:
        contribution.copy_(stage.transformer["wte"].weight.grad)
    elif p == pp_size - 1:
        contribution.copy_(stage.lm_head.weight.grad)
    contributions = [torch.empty_like(contribution) for _ in range(dist.get_world_size())]
    dist.all_gather(contributions, contribution)
    summed = contributions[d * pp_size]
    if pp_size > 1:
        summed = summed + contributions[(d + 1) * pp_size - 1]
    common.check_close(torch, summed, shared.transformer.wte.weight.grad,
                       "accumulated endpoint contributions vs tied GPT", errors)
    agree(torch, dist, rank, errors, f"{label}: order / whole-batch loss / accumulated gradients")
    all_stats = [None] * dist.get_world_size()
    dist.all_gather_object(all_stats, {"world_rank": rank, "dp_rank": d, "pp_rank": p,
        "max_gradient_error": max_gradient_error, "losses": losses if stage.is_last else None})
    return {"micro_batch_size": micro_size, "microbatch_sizes": [end-start for start, end in
            microbatch_bounds(args.batch_size, micro_size)], "pattern": pattern,
            "scheduled_calls_without_zeroing": 2, "status": "passed",
            "max_gradient_error": max(row["max_gradient_error"] for row in all_stats),
            "pipeline_losses": [row for row in all_stats if row["losses"] is not None]}


def verify_model(torch, dist, rank, args, trainer, GPT, GPTConfig, GPTStage,
                 info, pp_size, dp_size, bias):
    p = rank % pp_size
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias)
    torch.manual_seed(args.seed)
    with redirect_stdout(io.StringIO()):
        shared = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        reference = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        reference.load_state_dict(shared.state_dict(), strict=True)
        reference.lm_head.weight = torch.nn.Parameter(reference.lm_head.weight.detach().clone())
        stage = GPTStage(config, pp_size=pp_size, pp_rank=p).to(
            device="cpu", dtype=torch.float64).eval()
        stage.load_dense_state_dict(shared.state_dict())
    initial_state = {name: value.detach().clone() for name, value in stage.state_dict().items()}
    errors = []
    if shared.transformer.wte.weight is not shared.lm_head.weight:
        errors.append("原版 GPT 应共享 wte/lm_head Parameter")
    if pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 的 GPTStage 应共享 wte/lm_head Parameter")
    if (stage.layer_start, stage.layer_end) != (
            args.n_layer * p // pp_size, args.n_layer * (p + 1) // pp_size):
        errors.append("stage 的层范围不正确")
    for name, tensor in initial_state.items():
        common.check_close(torch, tensor, shared.state_dict().get(name), f"loaded {name}", errors)
    agree(torch, dist, rank, errors, f"PP={pp_size}, DP={dp_size}, bias={bias}: dense loading")
    trials = []
    for micro_size in args.micro_batch_sizes:
        patterns = ["unequal_valid"]
        if micro_size < args.batch_size:
            patterns.append("empty_microbatch")
        for pattern in patterns:
            trials.append(verify_trial(torch, dist, rank, args, trainer, shared, reference, stage,
                          info, initial_state, pp_size, dp_size, bias, micro_size, pattern))
    return {"bias": bias, "status": "passed", "trials": trials}


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
    if not callable(getattr(trainer, "gpipe_forward_backward", None)):
        raise AttributeError("train_pp.py 缺少 gpipe_forward_backward")
    store = dist.FileStore(str(Path(temporary) / "rendezvous"), world_size)
    store.set_timeout(timedelta(seconds=args.timeout))
    try:
        # 默认组只初始化一次；不同测试分段数通过不同的子组实现。
        dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=args.timeout))
        for pp_size, dp_size in args.cases:
            if rank == 0:
                print(f"[CASE] PP={pp_size}, DP={dp_size}", flush=True)
            info = trainer.setup_parallel(pp_size)
            agree(torch, dist, rank, common.group_errors(dist, info, rank, pp_size, dp_size),
                  f"PP={pp_size}, DP={dp_size}: groups / neighbour ranks")
            models = [verify_model(torch, dist, rank, args, trainer, GPT, GPTConfig, GPTStage,
                                   info, pp_size, dp_size, bias) for bias in args.biases]
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
    report = {"status": "running", "scope": "GPipe_schedule_and_local_accumulated_gradients",
        "requested_cases": args.cases, "completed_cases": [],
        "not_checked": ["train_pp.main", "CUDA/NCCL", "torchrun/env rendezvous", "AMP",
                        "DP gradient averaging", "training tied-weight gradient synchronization",
                        "optimizer updates", "gradient clipping", "dataset loading", "checkpoint"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / GPipe helper signatures (static only).", flush=True)
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
            model_config={"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
                          "vocab_size": args.vocab_size, "block_size": args.block_size, "dropout": 0.0},
            batch_size=args.batch_size, seq_len=args.seq_len, biases=args.biases,
            micro_batch_sizes=args.micro_batch_sizes)
        world_size = args.pp_size * args.dp_size
        print(f"[INFO] {world_size} CPU workers; threads={args.threads}; real cases={args.cases}.", flush=True)
        print(f"[INFO] batch={args.batch_size}, micro_batch_sizes={args.micro_batch_sizes}; two accumulated calls.",
              flush=True)
        print("[INFO] Checking GPipe/local gradients; DP/tied gradient sync and optimizer are pending.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_gpipe_") as temporary:
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
    print("[PASS] GPipe multi-microbatch numerical verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
