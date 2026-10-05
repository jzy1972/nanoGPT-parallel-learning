#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 nanoGPT 的 PP/DP 初始化和首尾权重同步，不修改项目文件。

在项目根目录运行：
    python verify_pp_init.py --project . --pp-size 2 --dp-size 2 --threads 1 --report pp_init_report.json

需要 parallel_utils.py 中的 build_pp_dp_groups、build_tied_weight_group、
sync_tied_weights，以及 train_pp.py 中的 setup_parallel。

先用模拟 rank 验证 24 种逻辑拓扑和所有进程的建组顺序，不启动 worker。
然后启动 pp_size * dp_size 个 CPU worker。默认在同一批 worker 中依次检查
所有满足 pp_size <= n_layer 的分段数；例如 4 个 worker、5 层时检查
PP1+DP4、PP2+DP2、PP4+DP1。--single-case 只检查指定的一组。

真实通信调用项目里的 setup_parallel 和 sync_tied_weights，使用 CPU/Gloo
和临时 FileStore；小模型由验证启动器创建。不会执行 train_pp.main。
训练入口只检查是否存在同步调用，不验证其完整控制流。未检查 torchrun/env
启动、CUDA/NCCL、激活传递、反向调度、DP/首尾梯度同步、优化器更新。

--mapping-only 仅做逻辑分组和建组顺序检查，不导入 PyTorch。
"""

import argparse
import ast
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile


THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
SOURCE_FILES = ("parallel_utils.py", "train_pp.py", "model.py", "model_pp.py")


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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("pp_size", "dp_size", "threads", "n_layer", "n_head", "n_embd",
                 "vocab_size", "block_size", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    if args.pp_size > args.n_layer:
        parser.error("pp_size 不能超过 n_layer")
    if args.n_embd % args.n_head:
        parser.error("n_embd 必须能被 n_head 整除")
    args.project = args.project.expanduser().resolve()
    world_size = args.pp_size * args.dp_size
    args.cases = [(args.pp_size, args.dp_size)] if args.single_case else [
        (p, world_size // p) for p in range(1, min(world_size, args.n_layer) + 1)
        if world_size % p == 0
    ]
    return args


def configure_threads(threads):
    # parent 和 Windows spawn worker 都在 torch/NumPy 导入之前设置。
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)


def source_hashes(project):
    return {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def read_sources(project):
    trees = {}
    for name in SOURCE_FILES:
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path}，请检查项目目录和 feature/pp 分支")
        trees[name] = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return trees


def top_level_function(tree, name, filename):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise ValueError(f"{filename} 需要且只能有一个顶层函数 {name}；请检查名称和缩进")
    return found[0]


def check_main_hook(trees):
    main = top_level_function(trees["train_pp.py"], "main", "train_pp.py")
    calls = [node.func for node in ast.walk(main) if isinstance(node, ast.Call)]
    if not any((isinstance(fn, ast.Name) and fn.id == "sync_tied_weights") or
               (isinstance(fn, ast.Attribute) and fn.attr == "sync_tied_weights") for fn in calls):
        raise ValueError("train_pp.main 中缺少 sync_tied_weights 调用，请放在创建 stage 并 .to(device) 之后")
    for node in trees["train_pp.py"].body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id == "main":
                raise ValueError("main() 必须放在 if __name__ == '__main__': 下面，不能在导入时执行")


class FakeGroup:
    def __init__(self, members):
        self.members = tuple(members)


class FakeDist:
    """只模拟分组元数据；不把模拟检查当作真实通信检查。"""
    GroupMember = SimpleNamespace(NON_GROUP_MEMBER=-100)

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.trace = []

    def is_initialized(self):
        return True

    def get_rank(self, group=None):
        if group is None:
            return self.rank
        if group == self.GroupMember.NON_GROUP_MEMBER:
            return -1
        return group.members.index(self.rank) if self.rank in group.members else -1

    def get_world_size(self, group=None):
        if group is None:
            return self.world_size
        if group == self.GroupMember.NON_GROUP_MEMBER or self.rank not in group.members:
            return -1
        return len(group.members)

    def get_process_group_ranks(self, group):
        return list(range(self.world_size)) if group is None else list(group.members)

    def new_group(self, ranks=None, *args, **kwargs):
        members = list(range(self.world_size)) if ranks is None else list(ranks)
        if not members or len(set(members)) != len(members):
            raise AssertionError(f"非法成员列表：{members}")
        if any(not 0 <= r < self.world_size for r in members):
            raise AssertionError(f"成员超出 world_size={self.world_size}：{members}")
        if kwargs.get("sort_ranks", True):
            members.sort()
        self.trace.append(tuple(members))
        return FakeGroup(members) if self.rank in members else self.GroupMember.NON_GROUP_MEMBER


def validate_parallel_info(info, rank, pp_size, dp_size, dist):
    p, d = rank % pp_size, rank // pp_size
    expected = {
        "world_rank": rank, "world_size": pp_size * dp_size,
        "pp_rank": p, "dp_rank": d, "dp_size": dp_size,
        "pp_first_rank": d * pp_size, "pp_last_rank": (d + 1) * pp_size - 1,
        "prev_rank": rank - 1 if p > 0 else None,
        "next_rank": rank + 1 if p + 1 < pp_size else None,
    }
    errors = []
    for key, value in expected.items():
        if key not in info or info[key] != value:
            errors.append(f"{key}: 应为 {value}，实际为 {info.get(key, '<missing>')}")
    member_specs = {
        "pp_group": [d * pp_size + q for q in range(pp_size)],
        "dp_group": [q * pp_size + p for q in range(dp_size)],
    }
    for key, members in member_specs.items():
        if key not in info or info[key] is None:
            errors.append(f"缺少有效的 {key}")
            continue
        try:
            actual = dist.get_process_group_ranks(info[key])
            if actual != members:
                errors.append(f"{key}: 应为 {members}，实际为 {actual}")
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    if "tied_group" not in info:
        errors.append("setup_parallel 返回字典中缺少 tied_group")
    elif pp_size == 1 or p not in (0, pp_size - 1):
        if info["tied_group"] is not None:
            errors.append("单 stage 或中间 stage 的 tied_group 应为 None")
    elif info["tied_group"] is None:
        errors.append("首尾 stage 缺少 tied_group")
    else:
        members = [d * pp_size, (d + 1) * pp_size - 1]
        try:
            actual = dist.get_process_group_ranks(info["tied_group"])
            if actual != members:
                errors.append(f"tied_group: 应为 {members}，实际为 {actual}")
        except Exception as exc:
            errors.append(f"tied_group: {exc}")
    return errors


def verify_logical_mapping(project, trees):
    ns = {"__builtins__": __builtins__}
    targets = [("parallel_utils.py", "build_pp_dp_groups"),
               ("parallel_utils.py", "build_tied_weight_group"),
               ("train_pp.py", "setup_parallel")]
    for filename, name in targets:
        node = top_level_function(trees[filename], name, filename)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        unit = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        exec(compile(unit, str(project / filename), "exec"), ns)
    topologies, rank_layouts = 0, 0
    for pp_size in range(1, 7):
        for dp_size in range(1, 5):
            reference_trace = None
            for rank in range(pp_size * dp_size):
                fake = FakeDist(rank, pp_size * dp_size)
                ns["dist"] = fake
                try:
                    info = ns["setup_parallel"](pp_size)
                    errors = validate_parallel_info(info, rank, pp_size, dp_size, fake)
                    if errors:
                        raise AssertionError("; ".join(errors))
                    if reference_trace is None:
                        reference_trace = fake.trace
                    elif fake.trace != reference_trace:
                        raise AssertionError("不同 rank 的 new_group 调用序列不一致；应由所有进程按相同顺序建组")
                except Exception as exc:
                    raise AssertionError(f"逻辑映射 PP={pp_size}, DP={dp_size}, rank={rank}: {exc}") from exc
                rank_layouts += 1
            topologies += 1
    print(f"[PASS] logical mapping / group creation order: {topologies} simulated topologies, "
          f"{rank_layouts} rank layouts (no workers).", flush=True)
    return {"topologies": topologies, "rank_layouts": rank_layouts, "mode": "simulated", "status": "passed"}


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


def state_digest(torch, state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(value.shape)}|{value.dtype}".encode("utf-8"))
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def endpoint_frame(torch, dist, stage, vocab_size, n_embd):
    own = torch.zeros((vocab_size, n_embd), dtype=torch.float64)
    if stage.is_first:
        own.copy_(stage.transformer["wte"].weight.detach())
    elif stage.is_last:
        own.copy_(stage.lm_head.weight.detach())
    all_weights = [torch.empty_like(own) for _ in range(dist.get_world_size())]
    dist.all_gather(all_weights, own)
    return all_weights


def verify_one_case(torch, dist, rank, args, trainer, GPTConfig, GPTStage, pp_size, dp_size):
    label = f"PP={pp_size}, DP={dp_size}"
    if rank == 0:
        print(f"[CASE] {label}", flush=True)
    info = trainer.setup_parallel(pp_size)
    finish_checks(torch, dist, rank,
                  validate_parallel_info(info, rank, pp_size, dp_size, dist),
                  f"{label}: real groups / neighbour ranks")

    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       vocab_size=args.vocab_size, block_size=args.block_size,
                       dropout=0.0, bias=True)
    p, d = rank % pp_size, rank // pp_size
    torch.manual_seed(args.seed + p)
    stage = GPTStage(config, pp_size=pp_size, pp_rank=p).to(dtype=torch.float64, device="cpu")
    before = {name: value.detach().clone() for name, value in stage.state_dict().items()}
    trainable = {name: value.requires_grad for name, value in stage.named_parameters()}
    digests = [None] * dist.get_world_size()
    dist.all_gather_object(digests, state_digest(torch, before))
    errors = []
    replicas = [digests[q * pp_size + p] for q in range(dp_size)]
    if len(set(replicas)) != 1:
        errors.append("同一 stage 的 DP 副本初始参数不一致")
    if (stage.layer_start, stage.layer_end) != (
            args.n_layer * p // pp_size, args.n_layer * (p + 1) // pp_size):
        errors.append("stage 层范围不正确")
    if (stage.is_first, stage.is_last) != (p == 0, p == pp_size - 1):
        errors.append("stage 首尾标记不正确")
    finish_checks(torch, dist, rank, errors, f"{label}: stage replicas / layer ownership")

    original_sources = endpoint_frame(torch, dist, stage, args.vocab_size, args.n_embd)
    trainer.sync_tied_weights(stage, tied_group=info["tied_group"], src_rank=info["pp_first_rank"])
    errors = []
    allowed = {"lm_head.weight"} if pp_size > 1 and stage.is_last else set()
    for name, actual in stage.state_dict().items():
        if name not in allowed and not torch.equal(actual, before[name]):
            errors.append(f"初始化同步意外修改了 {name}")
    if {name: value.requires_grad for name, value in stage.named_parameters()} != trainable:
        errors.append("同步改变了参数的 requires_grad 或参数注册")
    if pp_size == 1 and stage.transformer["wte"].weight is not stage.lm_head.weight:
        errors.append("PP=1 时 wte/lm_head 没有共享同一个 Parameter")
    expected_source = original_sources[d * pp_size]
    if stage.is_first and not torch.equal(stage.transformer["wte"].weight, expected_source):
        errors.append("首 stage 的源权重被修改")
    if stage.is_last and not torch.equal(stage.lm_head.weight, expected_source):
        errors.append("尾 stage 权重未匹配当前流水线首 stage 的原始权重")
    finish_checks(torch, dist, rank, errors, f"{label}: initial tied weights / other parameters unchanged")

    # 不同 DP 副本使用不同测试数值，防止相同初始化恰好掩盖串组/错误源 rank。
    expected = torch.arange(args.vocab_size * args.n_embd, dtype=torch.float64).reshape(
        args.vocab_size, args.n_embd) / 16.0 + 100.0 * (d + 1)
    with torch.no_grad():
        if stage.is_first:
            stage.transformer["wte"].weight.copy_(expected)
        elif stage.is_last:
            stage.lm_head.weight.fill_(-123.0 - d)
    unchanged = {name: value.detach().clone() for name, value in stage.state_dict().items()}
    trainer.sync_tied_weights(stage, tied_group=info["tied_group"], src_rank=info["pp_first_rank"])
    errors = []
    for name, actual in stage.state_dict().items():
        if name not in allowed and not torch.equal(actual, unchanged[name]):
            errors.append(f"第二次同步意外修改了 {name}")
    if stage.is_first and not torch.equal(stage.transformer["wte"].weight, expected):
        errors.append("当前流水线的源权重被其他 DP 副本覆盖")
    if stage.is_last and not torch.equal(stage.lm_head.weight, expected):
        errors.append("尾 stage 没有收到自己所属流水线的测试权重")
    finish_checks(torch, dist, rank, errors, f"{label}: per-pipeline tied broadcast isolation")

    local_record = {"world_rank": rank, "pp_rank": p, "dp_rank": d,
                    "pp_members": dist.get_process_group_ranks(info["pp_group"]),
                    "dp_members": dist.get_process_group_ranks(info["dp_group"]),
                    "tied_members": (dist.get_process_group_ranks(info["tied_group"])
                                     if info["tied_group"] is not None else None),
                    "layer_range": [stage.layer_start, stage.layer_end],
                    "prev_rank": info["prev_rank"], "next_rank": info["next_rank"]}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local_record)
    return {"pp_size": pp_size, "dp_size": dp_size, "status": "passed", "ranks": records}


def load_runtime_project(project):
    sys.path.insert(0, str(project))
    trainer = importlib.import_module("train_pp")
    dense_module = importlib.import_module("model")
    stage_module = importlib.import_module("model_pp")
    for module, name in ((trainer, "train_pp.py"), (dense_module, "model.py"),
                         (stage_module, "model_pp.py")):
        if Path(module.__file__).resolve() != project / name:
            raise RuntimeError(f"导入了错误路径：{module.__file__}")
    if not callable(getattr(trainer, "sync_tied_weights", None)):
        raise AttributeError("train_pp.py 需要导入 sync_tied_weights")
    return trainer, dense_module.GPTConfig, stage_module.GPTStage


def worker(rank, args, temporary):
    configure_threads(args.threads)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.pp_size * args.dp_size), LOCAL_RANK=str(rank))
    if source_hashes(args.project) != args.expected_hashes:
        raise RuntimeError("验证期间项目代码发生变化，请保存全部文件后重新运行")
    trainer, GPTConfig, GPTStage = load_runtime_project(args.project)
    world_size = args.pp_size * args.dp_size
    # 默认通信组只初始化一次；各测试拓扑创建各自的子组，最后统一清理。
    store = dist.FileStore(str(Path(temporary) / "rendezvous"), world_size)
    store.set_timeout(timedelta(seconds=args.timeout))
    try:
        dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=args.timeout))
        for pp_size, dp_size in args.cases:
            result = verify_one_case(torch, dist, rank, args, trainer, GPTConfig, GPTStage,
                                     pp_size, dp_size)
            if rank == 0:
                (Path(temporary) / f"result_pp{pp_size}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def write_report(path, report):
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Report saved: {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    report = {"status": "running", "scope": "PP_DP_and_tied_weight_initialization",
              "real_cases": [], "not_checked": ["train_pp.main execution", "torchrun/env rendezvous",
                "CUDA/NCCL", "activation transfer and pipeline backward schedule",
                "DP and tied-weight gradient synchronization", "optimizer updates"]}
    try:
        trees = read_sources(args.project)
        args.expected_hashes = source_hashes(args.project)
        report["source_sha256"] = args.expected_hashes
        check_main_hook(trees)
        report["main_sync_call_static_check"] = "present; control flow not executed"
        report["logical_mapping"] = verify_logical_mapping(args.project, trees)
        if args.mapping_only:
            report["status"] = "mapping_only_passed"
            write_report(args.report, report)
            print("[PASS] Logical mapping checks completed; real communication was not run.", flush=True)
            return 0

        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        torch.set_num_threads(args.threads)
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("当前 PyTorch 没有可用的 distributed/Gloo")
        report.update(torch_version=torch.__version__, device="cpu", backend="gloo", dtype="float64",
                      model_config={"n_layer": args.n_layer, "n_head": args.n_head,
                        "n_embd": args.n_embd, "vocab_size": args.vocab_size,
                        "block_size": args.block_size, "dropout": 0.0, "bias": True})
        world_size = args.pp_size * args.dp_size
        print(f"[INFO] {world_size} CPU workers; thread limit={args.threads}; real cases={args.cases}.", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanogpt_pp_init_") as temporary:
            try:
                mp.spawn(worker, args=(args, temporary), nprocs=world_size, join=True)
            finally:
                for pp_size, _ in args.cases:
                    result_path = Path(temporary) / f"result_pp{pp_size}.json"
                    if result_path.is_file():
                        report["real_cases"].append(json.loads(result_path.read_text(encoding="utf-8")))
        if len(report["real_cases"]) != len(args.cases):
            raise RuntimeError("worker 检查未全部完成")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print("[PASS] PP initialization verification completed (CPU/Gloo).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
