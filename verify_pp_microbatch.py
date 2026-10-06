#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证手写 PP 的单 microbatch 前向通信和反向通信（CPU/Gloo）。

放到 nanoGPT 根目录后运行：
    python verify_pp_microbatch.py --project . --pp-size 2 --dp-size 2 \
        --threads 1 --report pp_microbatch_report.json

直接调用项目里的 setup_parallel、forward_microbatch、backward_microbatch，
不会执行 train_pp.main，不修改项目源码或模型文件。

默认固定启动 PP*DP 个 CPU worker，在同一默认通信组内检查该 world_size
能支持的不同分段数。默认 4 个 worker、5 层微型模型，检查 (PP,DP) 为
(1,4)、(2,2)、(4,1)；这只是测试配置，不限制实现的并行规模。

用原版 GPT 的完整图作参考。PP>1 的参考副本另外把首尾权重拆成两个
初始值相同的 Parameter，以检查两处分别产生的梯度。验证器收集首尾
梯度并求和，与原版共享 Parameter 的梯度比较；该检查不会替用户实现
训练过程中的 tied-weight 梯度同步。

覆盖 bias=True/False、loss_weight=1 和 0.37、忽略标签 -1、边界激活和
激活梯度、每个 stage 的参数梯度、不同 DP 流水线使用不同数据。
不检查 GPipe 多 microbatch 调度、DP 梯度平均、优化器、裁剪、AMP、
CUDA/NCCL、数据加载、checkpoint 或 train_pp.main。
"""

import argparse
import ast
from contextlib import redirect_stdout
from datetime import timedelta
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile


SOURCE_FILES = ("model.py", "model_pp.py", "parallel_utils.py", "train_pp.py")
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
FUNCTION_ARGUMENTS = {
    "parallel_utils.py": {
        "send_forward": ("tensor", "dst_rank", "pp_group"),
        "recv_forward": ("shape", "dtype", "device", "src_rank", "pp_group"),
        "send_backward": ("grad_input", "dst_rank", "pp_group"),
        "recv_backward": ("stage_output", "src_rank", "pp_group"),
    },
    "train_pp.py": {
        "setup_parallel": ("pp_size",),
        "forward_microbatch": ("stage", "token_ids", "parallel", "micro_batch_size",
                               "seq_length", "device", "activation_dtype"),
        "backward_microbatch": ("stage", "stage_input", "stage_output", "parallel",
                                "targets", "loss_weight"),
    },
}


def configure_threads(threads):
    # 父进程和 spawn worker 都必须在导入 torch/NumPy 前执行。
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)


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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--bias", choices=("true", "false", "both"), default="both")
    parser.add_argument("--single-case", action="store_true",
                        help="只验证指定的 PP/DP，不检查同一 world_size 的其他分段数")
    parser.add_argument("--preflight-only", action="store_true",
                        help="只检查源码语法和函数参数，不导入 torch、不启动 worker")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd",
                 "vocab_size", "block_size", "seq_len", "batch_size", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.pp_size > args.n_layer:
        parser.error("pp_size 不能超过 n_layer")
    if args.n_embd % args.n_head:
        parser.error("n_embd 必须能被 n_head 整除")
    if args.seq_len > args.block_size:
        parser.error("seq_len 不能超过 block_size")
    if args.vocab_size < 2 or args.batch_size * args.seq_len < 2:
        parser.error("本验证需要至少两个词表项、两个标签位置")
    world_size = args.pp_size * args.dp_size
    args.cases = ([(args.pp_size, args.dp_size)] if args.single_case else
                  [(p, world_size // p) for p in range(1, min(args.n_layer, world_size) + 1)
                   if world_size % p == 0])
    args.biases = [True, False] if args.bias == "both" else [args.bias == "true"]
    return args


def source_hashes(project):
    return {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def preflight(project):
    for filename in SOURCE_FILES:
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}")
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name, required in FUNCTION_ARGUMENTS.get(filename, {}).items():
            if name not in functions:
                raise AttributeError(f"{filename} 缺少顶层函数 {name}")
            node = functions[name]
            supplied = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
            missing = set(required) - supplied
            if missing:
                raise TypeError(f"{filename} 的 {name} 缺少参数 {sorted(missing)}")
        if filename == "train_pp.py":
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name) and func.id == "main":
                        raise ValueError("main() 需要放在 if __name__ == '__main__': 内")


def load_project(project):
    sys.path.insert(0, str(project))
    modules = {name: importlib.import_module(name) for name in
               ("model", "model_pp", "parallel_utils", "train_pp")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != project / f"{name}.py":
            raise RuntimeError(f"导入了错误文件：{module.__file__}")
    trainer = modules["train_pp"]
    for name in FUNCTION_ARGUMENTS["train_pp.py"]:
        if not callable(getattr(trainer, name, None)):
            raise AttributeError(f"train_pp.py 缺少可调用的 {name}")
    return trainer, modules["model"].GPT, modules["model"].GPTConfig, modules["model_pp"].GPTStage


def finish_checks(torch, dist, rank, errors, label):
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
    if rank == 0:
        print(f"[PASS] {label}", flush=True)


def check_close(torch, actual, expected, name, errors):
    if actual is None or expected is None:
        errors.append(f"{name}: 梯度或张量为 None")
        return
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    except (AssertionError, RuntimeError) as exc:
        errors.append(f"{name}: {exc}")


def group_errors(dist, info, rank, pp_size, dp_size):
    p, d = rank % pp_size, rank // pp_size
    expected = {"world_rank": rank, "world_size": pp_size * dp_size,
                "pp_rank": p, "dp_rank": d, "dp_size": dp_size,
                "pp_first_rank": d * pp_size, "pp_last_rank": (d + 1) * pp_size - 1,
                "prev_rank": rank - 1 if p > 0 else None,
                "next_rank": rank + 1 if p < pp_size - 1 else None}
    errors = [f"{key}: 应为 {value}，实际为 {info.get(key)}"
              for key, value in expected.items() if info.get(key) != value]
    for name, members in (("pp_group", list(range(d * pp_size, (d + 1) * pp_size))),
                          ("dp_group", [q * pp_size + p for q in range(dp_size)])):
        if info.get(name) is None:
            errors.append(f"缺少 {name}")
        elif dist.get_process_group_ranks(info[name]) != members:
            errors.append(f"{name} 成员不正确")
    return errors


def verify_model_case(torch, dist, rank, args, trainer, GPT, GPTConfig, GPTStage,
                      info, pp_size, dp_size, bias):
    p, d = rank % pp_size, rank // pp_size
    label = f"PP={pp_size}, DP={dp_size}, bias={bias}"
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size,
                       dropout=0.0, bias=bias)
    torch.manual_seed(args.seed)
    with redirect_stdout(io.StringIO()):
        shared = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        reference = GPT(config).to(device="cpu", dtype=torch.float64).eval()
        reference.load_state_dict(shared.state_dict(), strict=True)
        # 只修改验证器的参考副本，用来独立检查 embedding/head 两处梯度。
        reference.lm_head.weight = torch.nn.Parameter(reference.lm_head.weight.detach().clone())
        stage = GPTStage(config, pp_size=pp_size, pp_rank=p).to(
            device="cpu", dtype=torch.float64).eval()
        stage.load_dense_state_dict(shared.state_dict())

    errors = []
    if shared.transformer.wte.weight is not shared.lm_head.weight:
        errors.append("model.py 的原版 GPT 应共享 wte/lm_head Parameter")
    if pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 的 GPTStage 应共享 wte/lm_head Parameter")
    expected_range = (args.n_layer * p // pp_size, args.n_layer * (p + 1) // pp_size)
    if (stage.layer_start, stage.layer_end) != expected_range:
        errors.append("stage 层范围不正确")
    initial_state = {name: value.detach().clone() for name, value in stage.state_dict().items()}
    for name, value in initial_state.items():
        check_close(torch, value, shared.state_dict().get(name), f"loaded {name}", errors)
    finish_checks(torch, dist, rank, errors, f"{label}: dense loading / endpoint identity")

    records = []
    for trial, loss_weight in enumerate((1.0, 0.37)):
        trial_label = f"{label}, loss_weight={loss_weight}"
        # 同一流水线各 rank 的参考数据一致，不同 DP 流水线数据不同。
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 10000 + 997 * d + 31 * trial)
        x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), generator=generator)
        targets = torch.randint(args.vocab_size, x.shape, generator=generator)
        targets[0, 0] = -1
        for model in (shared, reference, stage):
            model.zero_grad(set_to_none=True)

        captures, hooks = {}, []
        def capture(key):
            def hook(module, inputs, output):
                output.retain_grad()
                captures[key] = output
            return hook

        if p > 0:
            hooks.append(reference.transformer.h[expected_range[0] - 1].register_forward_hook(
                capture("input")))
        if p < pp_size - 1:
            hooks.append(reference.transformer.h[expected_range[1] - 1].register_forward_hook(
                capture("output")))
        shared_logits, shared_loss = shared(x, targets)
        ref_logits, ref_loss = reference(x, targets)
        if p == pp_size - 1:
            ref_logits.retain_grad()
            captures["output"] = ref_logits
        (shared_loss * loss_weight).backward()
        (ref_loss * loss_weight).backward()
        for hook in hooks:
            hook.remove()
        errors = []
        check_close(torch, ref_logits, shared_logits, "reference logits", errors)
        check_close(torch, ref_loss, shared_loss, "reference loss", errors)
        check_close(torch, reference.transformer.wte.weight.grad + reference.lm_head.weight.grad,
                    shared.transformer.wte.weight.grad, "reference tied-gradient sum", errors)
        finish_checks(torch, dist, rank, errors, f"{trial_label}: reference graph")

        stage_input, stage_output = trainer.forward_microbatch(
            stage=stage, token_ids=x if p == 0 else None, parallel=info,
            micro_batch_size=args.batch_size, seq_length=args.seq_len,
            device=torch.device("cpu"), activation_dtype=torch.float64)
        errors = []
        check_close(torch, stage_input, x if p == 0 else captures["input"], "stage input", errors)
        check_close(torch, stage_output, captures["output"], "stage output / full logits", errors)
        if not stage_output.requires_grad:
            errors.append("stage_output 被 detach 或在 no_grad 下计算，丢失本地计算图")
        if p > 0 and not (stage_input.is_leaf and stage_input.requires_grad):
            errors.append("接收激活应为 requires_grad=True 的叶子张量")
        finish_checks(torch, dist, rank, errors, f"{trial_label}: forward communication / activations")
        # 验证器保留非叶输出的 .grad，用来检查收到的下游梯度。
        stage_output.retain_grad()
        actual_loss = trainer.backward_microbatch(
            stage=stage, stage_input=stage_input, stage_output=stage_output,
            parallel=info, targets=targets if p == pp_size - 1 else None,
            loss_weight=loss_weight)

        errors = []
        if p == pp_size - 1:
            check_close(torch, actual_loss, ref_loss.detach(), "returned unweighted loss", errors)
            if actual_loss is not None and actual_loss.requires_grad:
                errors.append("返回的日志 loss 应 detach")
        elif actual_loss is not None:
            errors.append("非尾 stage 应返回 None loss")
        check_close(torch, stage_output.grad, captures["output"].grad, "output activation gradient", errors)
        if p > 0:
            check_close(torch, stage_input.grad, captures["input"].grad, "input activation gradient", errors)
        expected_parameters = dict((shared if pp_size == 1 else reference).named_parameters())
        for name, param in stage.named_parameters():
            expected = expected_parameters.get(name)
            if expected is None:
                errors.append(f"参考模型缺少参数 {name}")
            else:
                check_close(torch, param.grad, expected.grad, f"parameter gradient {name}", errors)
        for name, value in stage.state_dict().items():
            if not torch.equal(value, initial_state[name]):
                errors.append(f"只做 forward/backward 时权重或 buffer 被修改：{name}")
        finish_checks(torch, dist, rank, errors, f"{trial_label}: backward / all local parameter gradients")

        # 测试收集并相加；不改动 stage 参数梯度，也不执行训练梯度同步。
        contribution = torch.zeros((args.vocab_size, args.n_embd), dtype=torch.float64)
        if p == 0:
            contribution.copy_(stage.transformer["wte"].weight.grad)
        elif p == pp_size - 1:
            contribution.copy_(stage.lm_head.weight.grad)
        all_contributions = [torch.empty_like(contribution) for _ in range(dist.get_world_size())]
        dist.all_gather(all_contributions, contribution)
        summed = all_contributions[d * pp_size]
        if pp_size > 1:
            summed = summed + all_contributions[(d + 1) * pp_size - 1]
        errors = []
        check_close(torch, summed, shared.transformer.wte.weight.grad,
                    "sum of this pipeline's endpoint contributions", errors)
        finish_checks(torch, dist, rank, errors, f"{trial_label}: endpoint gradient contributions vs tied GPT")
        records.append({"loss_weight": loss_weight, "status": "passed"})
    return {"bias": bias, "checks": records, "status": "passed"}


def worker(rank, args, temporary):
    configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    world_size = args.pp_size * args.dp_size
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    if source_hashes(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间项目代码发生变化，请保存后重新运行")
    trainer, GPT, GPTConfig, GPTStage = load_project(args.project)
    store = dist.FileStore(str(Path(temporary) / "rendezvous"), world_size)
    store.set_timeout(timedelta(seconds=args.timeout))
    try:
        dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=args.timeout))
        for pp_size, dp_size in args.cases:
            if rank == 0:
                print(f"[CASE] PP={pp_size}, DP={dp_size}", flush=True)
            info = trainer.setup_parallel(pp_size)
            finish_checks(torch, dist, rank, group_errors(dist, info, rank, pp_size, dp_size),
                          f"PP={pp_size}, DP={dp_size}: groups / neighbour ranks")
            results = [verify_model_case(torch, dist, rank, args, trainer, GPT, GPTConfig,
                                         GPTStage, info, pp_size, dp_size, bias)
                       for bias in args.biases]
            if rank == 0:
                (Path(temporary) / f"result_pp{pp_size}.json").write_text(json.dumps(
                    {"pp_size": pp_size, "dp_size": dp_size, "status": "passed", "models": results},
                    ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def write_report(path, report):
    if path is not None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Report saved: {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "scope": "one_microbatch_PP_forward_backward_communication",
              "requested_cases": args.cases, "completed_cases": [],
              "not_checked": ["train_pp.main", "torchrun/env rendezvous", "CUDA/NCCL",
                              "multiple-microbatch GPipe schedule", "DP gradient averaging",
                              "training tied-weight gradient synchronization", "optimizer updates",
                              "AMP", "gradient clipping", "dataset loading", "checkpoint"]}
    try:
        preflight(args.project)
        args.expected_hashes = source_hashes(args.project)
        report["source_sha256"] = args.expected_hashes
        print("[PASS] source syntax / communication helper signatures (static only).", flush=True)
        if args.preflight_only:
            report["status"] = "preflight_only_passed"
            write_report(args.report, report)
            print("[PASS] Static preflight completed; no real communication was run.", flush=True)
            return 0
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        torch.set_num_threads(args.threads)
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("当前 PyTorch 缺少 distributed/Gloo")
        report.update(torch_version=torch.__version__, device="cpu", backend="gloo", dtype="float64",
                      model_config={"n_layer": args.n_layer, "n_head": args.n_head,
                                    "n_embd": args.n_embd, "vocab_size": args.vocab_size,
                                    "block_size": args.block_size, "dropout": 0.0},
                      biases=args.biases, loss_weights=[1.0, 0.37])
        world_size = args.pp_size * args.dp_size
        print(f"[INFO] {world_size} CPU workers; threads={args.threads}; real cases={args.cases}.", flush=True)
        print("[INFO] Checking single-microbatch communication; DP/tied gradient sync and optimizer are pending.",
              flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_microbatch_") as temporary:
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
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print("[PASS] PP single-microbatch forward/backward verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
