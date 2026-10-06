#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 PP+DP 的梯度同步（CPU/Gloo，微型模型，float64）。

把本文件与 verify_pp_microbatch.py 放在 nanoGPT 根目录，运行：
    python verify_pp_grad_sync.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_grad_sync_report.json

调用项目的 gpipe_forward_backward、get_dp_grad_weight、sync_dp_gradients、
sync_tied_weight_gradients 和 sync_stage_gradients；不调用 train_pp.main，
不修改项目源码。直接验证三个同步辅助函数，然后恢复本地梯度，验证训练
入口 sync_stage_gradients 是否得到相同的完整梯度。

以原版 GPT 对全部 DP 数据拼接后的全局 batch 做前后向为参考。额外使用
首尾 Parameter 分开的参考模型，独立核对 DP 平均后的两处梯度贡献，再
核对首尾 SUM 后的共享参数总梯度。不同 DP 副本使用不同的数据。

覆盖 bias=True/False、一个/多个 microbatch、最后一块较小、有效标签数
相等/不等、一整个 microbatch 被忽略、一个 DP 副本整批被忽略、全局
标签全被忽略时所有 rank 一致抛出 ValueError。每个 trial 开始清零梯度；
一个完整 GPipe batch 对应一次同步，不测试多 batch 跨调用的同步累计。

启动的 CPU worker 数为 pp_size * dp_size，默认检查同一 world_size 可用
的分段数，即 (1,4)、(2,2)、(4,1)。这些是验证配置，不限制算法规模。
--single-case 只检查指定的 PP/DP；--preflight-only 不导入 torch。

不验证优化器更新、全局梯度裁剪、非有限梯度共识、AMP、CUDA/NCCL、
数据加载、checkpoint、torchrun 或 train_pp.main，也不是性能测试。
"""

import argparse
import ast
from contextlib import redirect_stdout
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
    import verify_pp_microbatch as common
except ModuleNotFoundError as exc:
    if exc.name == "verify_pp_microbatch":
        raise SystemExit("请把 verify_pp_microbatch.py 与本文件放在同一目录。") from exc
    raise


EXTRA_ARGUMENTS = {
    "parallel_utils.py": {
        "get_dp_grad_weight": ("stage", "targets", "parallel", "device"),
        "sync_dp_gradients": ("stage", "dp_group", "dp_size", "grad_weight"),
        "sync_tied_weight_gradients": ("stage", "tied_group"),
    },
    "train_pp.py": {
        "gpipe_forward_backward": ("stage", "token_ids", "targets", "parallel",
                                  "batch_size", "seq_length", "micro_batch_size",
                                  "device", "activation_dtype"),
        "sync_stage_gradients": ("stage", "targets", "parallel", "device"),
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
    for path in (Path(common.__file__), Path(__file__)):
        result[f"checker:{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preflight(project):
    expected = Path(__file__).resolve().parent / "verify_pp_microbatch.py"
    if Path(common.__file__).resolve() != expected:
        raise RuntimeError(f"验证辅助文件应来自 {expected}，实际为 {common.__file__}")
    common.preflight(project)
    for filename, functions in EXTRA_ARGUMENTS.items():
        path = project / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for name, required in functions.items():
            nodes = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
            if len(nodes) != 1:
                raise AttributeError(f"{filename} 需要定义一个顶层 {name}()")
            node = nodes[0]
            supplied = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            missing = set(required) - supplied
            if missing:
                raise TypeError(f"{filename} 的 {name} 缺少可按关键字传入的参数 {sorted(missing)}")


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


def patterns_for(args, dp_size, micro_size):
    patterns = ["equal_valid", "unequal_valid"]
    if micro_size < args.batch_size:
        patterns.append("empty_microbatch")
    if dp_size > 1:
        patterns.append("one_empty_replica")
    return patterns


def valid_positions(total, dp_rank, dp_size, micro_rows, seq_len, pattern):
    if pattern == "equal_valid":
        return list(range(1, total))
    if pattern == "unequal_valid":
        count = min(total, 1 + dp_rank * total // dp_size)
        return [i * total // count for i in range(count)]
    if pattern == "empty_microbatch":
        return list(range(min(micro_rows * seq_len, total), total))
    if pattern == "one_empty_replica":
        return [] if dp_rank == 0 else list(range(total))
    if pattern == "all_empty":
        return []
    raise ValueError(f"未知 pattern：{pattern}")


def make_data(torch, args, dp_size, micro_size, pattern):
    batches = []
    total = args.batch_size * args.seq_len
    for d in range(dp_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 10000 + 997 * d)
        x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), generator=generator)
        original = torch.randint(args.vocab_size, x.shape, generator=generator)
        targets = torch.full_like(original, -1)
        positions = valid_positions(total, d, dp_size, micro_size, args.seq_len, pattern)
        if positions:
            targets.reshape(-1)[positions] = original.reshape(-1)[positions]
        batches.append((x, targets))
    return batches


def backward_reference(torch, model, x, targets):
    model.zero_grad(set_to_none=True)
    # 传入 targets，确保原版 GPT 输出完整序列的 logits。
    # 全被忽略时原版 mean loss 为 NaN，下面独立使用 sum / max(1,count)。
    logits, original_loss = model(x, targets)
    count = (targets != -1).sum()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=-1, reduction="sum",
    ) / count.clamp_min(1)
    loss.backward()
    return logits.detach(), loss.detach(), original_loss.detach()


def compare_gradients(torch, stage, reference, errors, label):
    # remove_duplicate=False 保留共享参考模型的 lm_head.weight 别名，
    # 使末尾 stage 的独立 Parameter 能对照原版共享权重的总梯度。
    expected = dict(reference.named_parameters(remove_duplicate=False))
    max_error = 0.0
    for name, parameter in stage.named_parameters():
        target = expected.get(name)
        if target is None:
            errors.append(f"{label}: 参考模型缺少 {name}")
            continue
        common.check_close(torch, parameter.grad, target.grad, f"{label}: {name}", errors)
        if parameter.grad is not None and target.grad is not None:
            max_error = max(max_error, float((parameter.grad - target.grad).abs().max().item()))
    return max_error


def snapshot_gradients(stage):
    return {name: None if param.grad is None else param.grad.detach().clone()
            for name, param in stage.named_parameters()}


def compare_snapshot(torch, stage, snapshot, errors, label):
    for name, parameter in stage.named_parameters():
        common.check_close(torch, parameter.grad, snapshot[name], f"{label}: {name}", errors)


def restore_gradients(stage, snapshot):
    for name, parameter in stage.named_parameters():
        grad = snapshot[name]
        parameter.grad = None if grad is None else grad.clone()


def check_weight(actual, expected, errors, label):
    try:
        value = float(actual)
    except (TypeError, ValueError):
        errors.append(f"{label}: 应返回梯度权重 {expected}，实际为 {actual!r}")
        return
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        errors.append(f"{label}: 应为 {expected:.16g}，实际为 {value:.16g}")


def state_errors(torch, stage, initial_state):
    return [f"只同步梯度时修改了参数或 buffer：{name}"
            for name, value in stage.state_dict().items()
            if not torch.equal(value, initial_state[name])]


def replica_gradient_errors(torch, dist, stage, info, pp_size, dp_size, args):
    errors = []
    flat = torch.cat([param.grad.reshape(-1) for param in stage.parameters()
                      if param.requires_grad]).contiguous()
    replicas = [torch.empty_like(flat) for _ in range(dp_size)]
    dist.all_gather(replicas, flat, group=info["dp_group"])
    if any(not torch.equal(flat, other) for other in replicas):
        errors.append("同步后相同 stage 的 DP 副本梯度不完全相同")
    endpoint = torch.zeros((args.vocab_size, args.n_embd), dtype=torch.float64)
    if stage.is_first:
        endpoint.copy_(stage.transformer["wte"].weight.grad)
    elif stage.is_last:
        endpoint.copy_(stage.lm_head.weight.grad)
    endpoints = [torch.empty_like(endpoint) for _ in range(dist.get_world_size())]
    dist.all_gather(endpoints, endpoint)
    for d in range(dp_size):
        if not torch.equal(endpoints[d * pp_size], endpoints[(d + 1) * pp_size - 1]):
            errors.append(f"DP 副本 {d} 的首尾共享权重梯度不完全相同")
    return errors


def run_gpipe(torch, trainer, stage, info, args, micro_size, x, targets):
    return trainer.gpipe_forward_backward(
        stage=stage,
        token_ids=x if stage.is_first else None,
        targets=targets if stage.is_last else None,
        parallel=info,
        batch_size=args.batch_size,
        seq_length=args.seq_len,
        micro_batch_size=micro_size,
        device=torch.device("cpu"),
        activation_dtype=torch.float64,
    )


def verify_trial(torch, dist, rank, args, trainer, utils, dense, untied, local,
                 stage, info, initial_state, pp_size, dp_size, bias, micro_size, pattern):
    d = info["dp_rank"]
    label = f"PP={pp_size}, DP={dp_size}, bias={bias}, micro_batch={micro_size}, {pattern}"
    batches = make_data(torch, args, dp_size, micro_size, pattern)
    counts = [int((y != -1).sum().item()) for _, y in batches]
    assert sum(counts) > 0
    expected_weight = dp_size * counts[d] / sum(counts)
    x, targets = batches[d]
    global_x = torch.cat([tokens for tokens, _ in batches], dim=0)
    global_targets = torch.cat([labels for _, labels in batches], dim=0)
    errors = []

    logits, global_loss, original_loss = backward_reference(torch, dense, global_x, global_targets)
    untied_logits, untied_loss, _ = backward_reference(torch, untied, global_x, global_targets)
    _, local_loss, _ = backward_reference(torch, local, x, targets)
    common.check_close(torch, logits, untied_logits, "global reference logits", errors)
    common.check_close(torch, global_loss, original_loss, "global reference loss normalization", errors)
    common.check_close(torch, untied_loss, global_loss, "untied reference loss", errors)
    common.check_close(torch, untied.transformer.wte.weight.grad + untied.lm_head.weight.grad,
                       dense.transformer.wte.weight.grad, "global reference endpoint sum", errors)
    agree(torch, dist, rank, errors, f"{label}: reference graphs", quiet=True)

    stage.zero_grad(set_to_none=True)
    returned_loss = run_gpipe(torch, trainer, stage, info, args, micro_size, x, targets)
    if stage.is_last:
        common.check_close(torch, returned_loss, local_loss, "returned local batch loss", errors)
        if returned_loss is not None and returned_loss.requires_grad:
            errors.append("返回的 loss 应已 detach")
    elif returned_loss is not None:
        errors.append("非末尾 stage 应返回 None loss")
    compare_gradients(torch, stage, local, errors, "before synchronization / local gradient")
    agree(torch, dist, rank, errors, f"{label}: local GPipe gradients", quiet=True)
    before_sync = snapshot_gradients(stage)

    # 第一次：直接调用三个实际辅助函数，逐步核对中间结果。
    weight = utils.get_dp_grad_weight(
        stage=stage, targets=targets if stage.is_last else None,
        parallel=info, device=torch.device("cpu"),
    )
    check_weight(weight, expected_weight, errors, "get_dp_grad_weight")
    compare_snapshot(torch, stage, before_sync, errors, "metadata must not alter gradients")
    agree(torch, dist, rank, errors, f"{label}: global valid-count weighting", quiet=True)

    utils.sync_dp_gradients(stage=stage, dp_group=info["dp_group"], dp_size=dp_size,
                            grad_weight=weight)
    compare_gradients(torch, stage, dense if pp_size == 1 else untied, errors,
                      "after DP averaging / separate endpoint contributions")
    agree(torch, dist, rank, errors, f"{label}: weighted DP averaging", quiet=True)

    utils.sync_tied_weight_gradients(stage=stage, tied_group=info["tied_group"])
    direct_error = compare_gradients(torch, stage, dense, errors, "after tied SUM / global dense gradient")
    errors.extend(replica_gradient_errors(torch, dist, stage, info, pp_size, dp_size, args))
    errors.extend(state_errors(torch, stage, initial_state))
    agree(torch, dist, rank, errors, f"{label}: tied gradient SUM", quiet=True)
    direct = snapshot_gradients(stage)

    # 第二次：恢复原始本地梯度，再调用用户训练入口，避免重复同步同一梯度。
    restore_gradients(stage, before_sync)
    wrapper_weight = trainer.sync_stage_gradients(
        stage=stage, targets=targets if stage.is_last else None,
        parallel=info, device=torch.device("cpu"),
    )
    check_weight(wrapper_weight, expected_weight, errors, "sync_stage_gradients returned weight")
    wrapper_error = compare_gradients(torch, stage, dense, errors, "wrapper / global dense gradient")
    compare_snapshot(torch, stage, direct, errors, "wrapper vs direct helper calls")
    errors.extend(replica_gradient_errors(torch, dist, stage, info, pp_size, dp_size, args))
    errors.extend(state_errors(torch, stage, initial_state))
    agree(torch, dist, rank, errors, f"{label}: local GPipe / DP average / tied SUM / wrapper")
    stats = [None] * dist.get_world_size()
    dist.all_gather_object(stats, {
        "world_rank": rank, "dp_rank": d, "pp_rank": info["pp_rank"],
        "max_gradient_error": max(direct_error, wrapper_error),
        "returned_local_loss": float(returned_loss.item()) if stage.is_last else None,
    })
    return {"status": "passed", "pattern": pattern, "micro_batch_size": micro_size,
            "microbatch_sizes": [min(micro_size, args.batch_size - start)
                                 for start in range(0, args.batch_size, micro_size)],
            "valid_labels_per_dp_replica": counts,
            "expected_grad_weights": [dp_size * count / sum(counts) for count in counts],
            "global_reference_loss": float(global_loss.item()),
            "max_gradient_error": max(row["max_gradient_error"] for row in stats),
            "pipeline_losses": [row for row in stats if row["returned_local_loss"] is not None]}


def verify_all_empty(torch, dist, rank, args, trainer, utils, stage, info,
                     initial_state, pp_size, dp_size, bias):
    label = f"PP={pp_size}, DP={dp_size}, bias={bias}: globally ignored batch"
    micro_size = args.micro_batch_sizes[0]
    x, targets = make_data(torch, args, dp_size, micro_size, "all_empty")[info["dp_rank"]]
    stage.zero_grad(set_to_none=True)
    loss = run_gpipe(torch, trainer, stage, info, args, micro_size, x, targets)
    errors = []
    if stage.is_last:
        common.check_close(torch, loss, torch.zeros((), dtype=torch.float64), "empty batch loss", errors)
    elif loss is not None:
        errors.append("非末尾 stage 的空 batch loss 应为 None")
    for name, parameter in stage.named_parameters():
        common.check_close(torch, parameter.grad, torch.zeros_like(parameter),
                           f"empty batch zero gradient {name}", errors)
    agree(torch, dist, rank, errors, f"{label}: GPipe zero-gradient communication", quiet=True)

    outcomes = []
    for name, function in (("get_dp_grad_weight", utils.get_dp_grad_weight),
                           ("sync_stage_gradients", trainer.sync_stage_gradients)):
        try:
            function(stage=stage, targets=targets if stage.is_last else None,
                     parallel=info, device=torch.device("cpu"))
        except ValueError:
            outcomes.append(name)
        except Exception as exc:
            errors.append(f"{name}: 应抛出 ValueError，实际为 {type(exc).__name__}: {exc}")
        else:
            errors.append(f"{name}: 全局无有效标签时未拒绝更新")
        agree(torch, dist, rank, errors, f"{label}: {name} rejection on every rank", quiet=True)
    for name, parameter in stage.named_parameters():
        common.check_close(torch, parameter.grad, torch.zeros_like(parameter),
                           f"rejected batch keeps zero gradient {name}", errors)
    errors.extend(state_errors(torch, stage, initial_state))
    agree(torch, dist, rank, errors, f"{label}: zero communication / consistent ValueError")
    return {"status": "passed", "micro_batch_size": micro_size,
            "expected_exception": "ValueError", "rejected_by": outcomes,
            "all_ranks_agreed": True}


def verify_model(torch, dist, rank, args, trainer, utils, GPT, GPTConfig, GPTStage,
                 info, pp_size, dp_size, bias):
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size, dropout=0.0, bias=bias)
    torch.manual_seed(args.seed)
    with redirect_stdout(io.StringIO()):
        dense = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        untied = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        untied.load_state_dict(dense.state_dict(), strict=True)
        untied.lm_head.weight = torch.nn.Parameter(untied.lm_head.weight.detach().clone())
        local = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        local.load_state_dict(dense.state_dict(), strict=True)
        if pp_size > 1:
            local.lm_head.weight = torch.nn.Parameter(local.lm_head.weight.detach().clone())
        stage = GPTStage(config, pp_size=pp_size, pp_rank=info["pp_rank"]).to(
            device="cpu", dtype=torch.float64).eval()
        stage.load_dense_state_dict(dense.state_dict())
    initial_state = {name: value.detach().clone() for name, value in stage.state_dict().items()}
    errors = []
    if dense.transformer.wte.weight is not dense.lm_head.weight:
        errors.append("model.py 的原版 GPT 应共享 wte/lm_head Parameter")
    if pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 的 GPTStage 应共享 wte/lm_head Parameter")
    p = info["pp_rank"]
    if (stage.layer_start, stage.layer_end) != (
            args.n_layer * p // pp_size, args.n_layer * (p + 1) // pp_size):
        errors.append("stage 的层范围不正确")
    for name, tensor in initial_state.items():
        common.check_close(torch, tensor, dense.state_dict().get(name), f"loaded {name}", errors)
    agree(torch, dist, rank, errors, f"PP={pp_size}, DP={dp_size}, bias={bias}: dense loading")
    trials = []
    for micro_size in args.micro_batch_sizes:
        for pattern in patterns_for(args, dp_size, micro_size):
            trials.append(verify_trial(torch, dist, rank, args, trainer, utils, dense, untied, local,
                          stage, info, initial_state, pp_size, dp_size, bias, micro_size, pattern))
    empty = verify_all_empty(torch, dist, rank, args, trainer, utils, stage, info,
                            initial_state, pp_size, dp_size, bias)
    return {"bias": bias, "status": "passed", "trials": trials, "globally_ignored_batch": empty}


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
            p, d = rank % pp_size, rank // pp_size
            tied = info.get("tied_group")
            if pp_size > 1 and p in (0, pp_size - 1):
                members = [d * pp_size, (d + 1) * pp_size - 1]
                if tied is None or dist.get_process_group_ranks(tied) != members:
                    errors.append("首尾 tied_group 成员不正确")
            elif tied is not None:
                errors.append("PP=1 或中间 stage 的 tied_group 应为 None")
            agree(torch, dist, rank, errors, f"PP={pp_size}, DP={dp_size}: PP/DP/tied groups")
            models = [verify_model(torch, dist, rank, args, trainer, utils, GPT, GPTConfig, GPTStage,
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
    report = {"status": "running", "scope": "PP_DP_valid_token_weighting_and_tied_gradient_sync",
              "requested_cases": args.cases, "completed_cases": [],
              "not_checked": ["train_pp.main", "torchrun/env rendezvous", "CUDA/NCCL", "AMP",
                              "optimizer updates", "gradient clipping", "nonfinite consensus",
                              "dataset loading", "checkpoint", "multiple batches per sync"]}
    try:
        preflight(args.project)
        args.expected_hashes = fingerprints(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / GPipe and gradient-sync signatures (static only).", flush=True)
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
                      micro_batch_sizes=args.micro_batch_sizes)
        world_size = args.pp_size * args.dp_size
        print(f"[INFO] {world_size} CPU workers; threads={args.threads}; real cases={args.cases}.", flush=True)
        print("[INFO] Reference: dense GPT on concatenated global DP batch; dropout=0, float64.", flush=True)
        print("[INFO] Checking local GPipe, weighted DP average, tied SUM, wrapper and all-empty rejection.",
              flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_grad_sync_") as temporary:
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
    print("[PASS] PP+DP gradient synchronization numerical verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
