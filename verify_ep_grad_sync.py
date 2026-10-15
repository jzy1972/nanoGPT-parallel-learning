#!/usr/bin/env python3
"""Verify EP parameter partition, broadcast, and averaged accumulated gradients.

Put this file AND verify_ep_forward.py beside parallel_ep.py and moe_ep.py:
    python verify_ep_grad_sync.py --project . --device cpu --ep-size 2 --threads 1 --report ep_grad_sync_report.json

New project functions under test, all in parallel_ep.py:
    split_ep_parameters(model) -> (shared_named_params, expert_named_params)
    sync_shared_parameters(model, parallel)
    sync_ep_gradients(model, parallel)

The previous verifier supplies independent dense-MoE reference calculations,
thread configuration, group checks, and diagnostics. No project source is
edited. The training entry is not run and no dataset is needed.

A small fixture has TWO nested MoEEP layers, shared input/output projections,
a tied shared Parameter, a sometimes-used shared Parameter, a globally unused
Parameter, and a frozen shared Parameter. It tests both small toy experts and
the project's actual MLP. Its dense counterpart evaluates ALL experts and
uses routing masks, rather than the project's MoE.forward/Router.forward.

The objective is mean_over_ranks(sum_over_microbatches(weight * local_loss)),
where microbatch weights use local sample counts. Auxiliary loss is computed
on each local microbatch and then weighted; it is not recomputed on a global
concatenated batch. The tested topology is one EP group covering WORLD.

Tests include exact parameter partition/deduplication, broadcast of all shared
parameters including frozen ones, preservation of expert weights, asymmetric
grad=None, globally unused parameters, noncontiguous gradients, and experts
used only in earlier microbatches. Input gradients remain LOCAL contributions;
only parameter gradients are normalized to the global mean objective.

No optimizer, clipping/overflow, production GPT integration, AMP, compile,
checkpoint, multi-node, or independent expert-replica DP groups are tested.
CPU uses direct FileStore/Gloo; CUDA uses one NCCL worker per visible device.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EP/shared-DP parameter and accumulated-gradient verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)
    for name in ("ep_size", "threads", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and nonnegative")
    args.project = args.project.resolve()
    return args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_base():
    path = Path(__file__).resolve().with_name("verify_ep_forward.py")
    if not path.is_file():
        raise FileNotFoundError("Place verify_ep_forward.py in the SAME directory as verify_ep_grad_sync.py.")
    spec = importlib.util.spec_from_file_location("_ep_forward_reference_for_grad", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("preflight", "load_project", "Checks", "configure_threads", "check_groups",
                 "controlled_input", "independent_reference", "objective", "toy_expert_class"):
        ensure(hasattr(module, name), f"verify_ep_forward.py is missing {name}; use the supplied version.")
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(project, base):
    hashes = base.preflight(project)
    path = project / "parallel_ep.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    needed = {"split_ep_parameters", "sync_shared_parameters", "sync_ep_gradients"}
    ensure(needed <= functions, f"parallel_ep.py needs top-level functions: {sorted(needed - functions)}")
    return hashes


def expert_key(name):
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "blocks" and parts[2] == "experts":
        return parts[1], int(parts[3])
    return None


def cases(size):
    experts = 2 * size
    return [{"family": family, "bias": bias, "k": k, "microsteps": microsteps}
            for family in ("toy", "MLP") for bias in (True, False)
            for k in sorted({1, min(2, experts), experts}) for microsteps in (1, 3)]


def micro_sizes(rank, count):
    return [2 + rank % 2] if count == 1 else [1 + rank % 2, 3, 2 + rank % 2]


def make_probe(torch, modules, config, info, expert_cls, sharded):
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            width = config.n_embd
            self.pre = torch.nn.Linear(width, width, bias=config.bias)
            self.blocks = torch.nn.ModuleDict()
            for layer in ("0", "1"):
                if sharded:
                    block = modules["moe_ep"].MoEEP(config, expert_cls=expert_cls, ep_group=info["ep_group"])
                else:
                    block = modules["moe"].MoE(config, expert_cls=expert_cls)
                self.blocks[layer] = block
            self.post = torch.nn.Linear(width, width, bias=config.bias)
            self.offset = torch.nn.Parameter(torch.zeros(width))
            self.offset_alias = self.offset
            self.optional = torch.nn.Parameter(torch.zeros(width))
            self.unused = torch.nn.Parameter(torch.zeros(width))
            self.frozen = torch.nn.Parameter(torch.ones(width), requires_grad=False)

        def forward(self, x, use_optional):
            h = self.pre(x)
            auxiliary = torch.zeros((), dtype=torch.float32, device=x.device)
            for block in self.blocks.values():
                y, layer_auxiliary = block(h)
                h = h + y
                auxiliary = auxiliary + layer_auxiliary
            output = self.post(h) + self.offset + self.offset_alias + 0.01 * self.frozen
            if use_optional:
                output = output + self.optional
            return output, auxiliary
    return Probe()


def build_models(checks, base, modules, info, case):
    torch = checks.torch
    experts = 2 * checks.args.ep_size
    config = SimpleNamespace(n_embd=max(experts + 2, 8), n_experts=experts,
                             moe_top_k=case["k"], bias=case["bias"], dropout=0.0,
                             n_layer=2, block_size=8, experts_per_rank=2,
                             ep_size=checks.args.ep_size)
    expert_cls = base.toy_expert_class(torch) if case["family"] == "toy" else modules["model"].MLP
    torch.manual_seed(checks.args.seed)
    reference = make_probe(torch, modules, config, info, expert_cls, False)
    stage = make_probe(torch, modules, config, info, expert_cls, True)
    dtype = getattr(torch, checks.args.dtype)
    reference = reference.to(device=checks.device, dtype=dtype)
    stage = stage.to(device=checks.device, dtype=dtype)
    # Make this fixture's intended tie explicit after any device conversion.
    reference.offset_alias = reference.offset
    stage.offset_alias = stage.offset
    with torch.no_grad():
        reference.pre.weight.copy_(torch.eye(config.n_embd, device=checks.device, dtype=dtype))
        if reference.pre.bias is not None:
            reference.pre.bias.zero_()
        for block in reference.blocks.values():
            block.router.gate.weight.zero_()
            block.router.gate.weight[:, :experts].copy_(torch.eye(experts, device=checks.device, dtype=dtype))
    first, end = checks.rank * 2, (checks.rank + 1) * 2
    expected = {name for name in reference.state_dict()
                if expert_key(name) is None or first <= expert_key(name)[1] < end}
    ensure(set(stage.state_dict()) == expected, "Fixture EP state keys must retain GLOBAL expert IDs.")
    stage.load_state_dict({name: value for name, value in reference.state_dict().items() if name in expected}, strict=True)
    return config, stage, reference


def expected_partition(stage):
    expert_ids = {id(param) for block in stage.blocks.values() for param in block.experts.parameters()}
    params = dict(stage.named_parameters())
    shared = {name: param for name, param in params.items() if id(param) not in expert_ids}
    experts = {name: param for name, param in params.items() if id(param) in expert_ids}
    return shared, experts


def check_partition(checks, utils, stage):
    checks.phase("nested parameter partition / ties / frozen parameters")
    shared, experts = expected_partition(stage)
    errors, actual_shared_names = [], []
    try:
        got_shared, got_experts = utils.split_ep_parameters(stage)
        got_shared, got_experts = list(got_shared), list(got_experts)
        all_pairs = got_shared + got_experts
        ensure(len(all_pairs) == len(dict(stage.named_parameters())), "Partition omitted or duplicated a Parameter")
        ensure(len({id(param) for _, param in all_pairs}) == len(all_pairs), "Tied Parameters must appear only once")
        for got, expected, label in ((got_shared, shared, "shared"), (got_experts, experts, "expert")):
            ensure({name for name, _ in got} == set(expected), f"Incorrect {label} parameter names")
            ensure(all(expected[name] is param for name, param in got), f"Incorrect {label} Parameter objects")
        ensure("frozen" in shared and "unused" in shared and "optional" in shared, "Shared classification is incomplete")
        ensure(stage.offset is stage.offset_alias, "Fixture tie was lost")
        actual_shared_names = [name for name, _ in got_shared]
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, "nested parameter partition / ties / frozen parameters")
    names_on_ranks = [None] * checks.args.ep_size
    checks.dist.all_gather_object(names_on_ranks, actual_shared_names)
    errors = []
    if not all(names == names_on_ranks[0] for names in names_on_ranks):
        errors.append("Shared parameters must be communicated in the same order on every rank")
    checks.agree(errors, "shared collective order")
    checks.passed("nested parameter partition / deduplicated ties / shared collective order")


def check_broadcast(checks, utils, stage, reference, info):
    checks.phase("broadcast shared parameters and preserve local experts")
    torch = checks.torch
    shared, experts = expected_partition(stage)
    with torch.no_grad():
        for index, (_, param) in enumerate(sorted(shared.items())):
            if checks.rank != 0:
                param.add_(0.125 * checks.rank + 0.01 * (index + 1))
        for index, (_, param) in enumerate(sorted(experts.items())):
            param.add_(0.03125 * (checks.rank + 1) + 0.001 * (index + 1))
    before_experts = {name: param.detach().clone() for name, param in experts.items()}
    utils.sync_shared_parameters(stage, info)
    full_params = dict(reference.named_parameters())
    errors = []
    for name, param in shared.items():
        checks.close(param, full_params[name], name + " broadcast from rank 0", errors, exact=True)
    for name, param in experts.items():
        checks.close(param, before_experts[name], name + " local expert preserved", errors, exact=True)
    if stage.offset is not stage.offset_alias or stage.frozen.requires_grad:
        errors.append("Broadcast changed a shared tie or requires_grad flag")
    checks.agree(errors, "shared broadcast, including frozen parameters; expert preservation")
    checks.passed("shared broadcast, including frozen parameters; expert preservation")


def shared_synthetic_value(name, rank):
    if name in ("unused", "frozen"):
        return None
    if name == "optional":
        return 7.0 if rank == 0 else None
    if name == "offset":
        return float(rank + 2)
    index = sum(name.encode("utf-8")) % 11
    return None if (rank + index) % 2 else 0.25 * (rank + 1) * (index + 1)


def check_synthetic_gradients(checks, utils, stage, info):
    checks.phase("synthetic asymmetric None / unused / noncontiguous gradients")
    torch, size = checks.torch, checks.args.ep_size
    shared, experts = expected_partition(stage)
    stage.zero_grad(set_to_none=True)
    before_params = {name: param.detach().clone() for name, param in stage.named_parameters()}
    expected = {}
    for name, param in shared.items():
        basis = (torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape_as(param) + 1) / 8
        local = shared_synthetic_value(name, checks.rank)
        param.grad = None if local is None else basis * local
        contributors = [shared_synthetic_value(name, rank) for rank in range(size)]
        present = [value for value in contributors if value is not None]
        expected[name] = None if not present else basis * (sum(present) / size)
    for name, param in experts.items():
        basis = (torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape_as(param) + 1) / 8
        value = None if sum(name.encode("utf-8")) % 3 == 0 else float(2 + checks.rank + sum(name.encode("utf-8")) % 5)
        param.grad = None if value is None else basis * value
        expected[name] = None if value is None else basis * (value / size)
    for _, param in stage.named_parameters():
        if param.grad is not None and param.grad.ndim == 2:
            param.grad = param.grad.t().contiguous().t()
    utils.sync_ep_gradients(stage, info)
    errors = []
    for name, param in stage.named_parameters():
        if expected[name] is None:
            if param.grad is not None:
                errors.append(f"{name}: expected grad=None")
        else:
            checks.close(param.grad, expected[name], name + " synchronized gradient", errors)
        checks.close(param, before_params[name], name + " parameter unchanged", errors, exact=True)
    checks.agree(errors, "asymmetric None / global None / expert scaling / noncontiguous gradients")
    checks.passed("asymmetric None / global None / expert scaling / noncontiguous gradients")
    stage.zero_grad(set_to_none=True)


def reference_forward(torch, base, reference, x, config, use_optional):
    h = reference.pre(x)
    auxiliary = torch.zeros((), dtype=torch.float32, device=x.device)
    layer_counts = []
    for block in reference.blocks.values():
        y, layer_auxiliary, counts = base.independent_reference(torch, block, h, config)
        h = h + y
        auxiliary = auxiliary + layer_auxiliary
        layer_counts.append(counts)
    output = reference.post(h) + reference.offset + reference.offset_alias + 0.01 * reference.frozen
    if use_optional:
        output = output + reference.optional
    return output, auxiliary, torch.stack(layer_counts)


def micro_input(checks, base, config, step, count, samples):
    torch = checks.torch
    values = base.controlled_input(torch, config, checks.rank,
                                   "rotating" if count == 1 else "one_owner",
                                   checks.device, getattr(torch, checks.args.dtype))
    repeats = math.ceil(samples / values.shape[1])
    values = values.repeat(1, repeats, 1)[:, :samples].contiguous()
    if count > 1 and step > 0:
        # Earlier microbatch -> first owner. Later microbatches -> last owner.
        # For top-k <= 2 and EP>1, first-owner experts become early-only.
        values[:, :, :config.n_experts] = torch.roll(values[:, :, :config.n_experts],
                                                    shifts=config.n_experts - config.experts_per_rank, dims=-1)
    return values


def check_accumulation(checks, base, modules, info, case):
    torch, dist = checks.torch, checks.dist
    label = f"{case['family']} bias={case['bias']} k={case['k']} microbatches={case['microsteps']}"
    checks.phase(label + " construction/loading")
    errors, built = [], None
    try:
        built = build_models(checks, base, modules, info, case)
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + " construction/loading")
    config, stage, reference = built
    shared, _ = expected_partition(stage)
    sizes = micro_sizes(checks.rank, case["microsteps"])
    total = sum(sizes)
    count_history = []
    inputs = []
    local_loss = 0.0
    stage.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    for step, samples in enumerate(sizes):
        checks.phase(label + f" microbatch {step}: forward/backward")
        values = micro_input(checks, base, config, step, len(sizes), samples)
        x = values.clone().requires_grad_(True)
        reference_x = values.clone().requires_grad_(True)
        use_optional = (checks.rank + step) % checks.args.ep_size == 0
        expected_output, expected_aux, counts = reference_forward(torch, base, reference, reference_x, config, use_optional)
        output, auxiliary = stage(x, use_optional)
        errors = []
        checks.close(output, expected_output, f"microbatch {step} output", errors)
        checks.close(auxiliary, expected_aux, f"microbatch {step} local auxiliary", errors)
        checks.agree(errors, label + f" microbatch {step} forward")
        weight = samples / total
        loss = weight * base.objective(torch, output, auxiliary, checks.rank + 11 * step, "mixed")
        expected_loss = weight * base.objective(torch, expected_output, expected_aux, checks.rank + 11 * step, "mixed")
        loss.backward()
        expected_loss.backward()
        local_loss += float(loss.detach().item())
        count_history.append(counts)
        inputs.append((x, reference_x))

    checks.phase(label + " ONE gradient synchronization after all microbatches")
    before_params = {name: param.detach().clone() for name, param in stage.named_parameters()}
    modules["parallel_ep"].sync_ep_gradients(stage, info)
    global_history = torch.stack(count_history)
    dist.all_reduce(global_history, op=dist.ReduceOp.SUM, group=info["ep_group"])
    global_counts = global_history.sum(dim=0)
    early_only = (global_history[0] > 0) & (global_history[1:].sum(dim=0) == 0) if len(sizes) > 1 else None
    actual_params = dict(stage.named_parameters())
    errors = []
    if len(sizes) > 1 and checks.args.ep_size > 1 and case["k"] <= config.experts_per_rank:
        if not early_only[0].any().item():
            errors.append("The first MoE layer did not exercise experts used only in earlier microbatches")
    for name, param in reference.named_parameters():
        key = expert_key(name)
        globally_unrouted = key is not None and global_counts[int(key[0]), key[1]].item() == 0
        flag = torch.tensor(int(param.grad is not None), dtype=torch.int32, device=checks.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=info["shared_dp_group"])
        expected_grad = None
        if flag.item():
            expected_grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
            dist.all_reduce(expected_grad, op=dist.ReduceOp.SUM, group=info["shared_dp_group"])
            expected_grad.div_(checks.args.ep_size)
        if name not in actual_params:
            continue
        actual = actual_params[name]
        if expected_grad is None or globally_unrouted:
            if actual.grad is not None:
                errors.append(f"{name}: globally unused/frozen parameter must keep grad=None")
        else:
            checks.close(actual.grad, expected_grad, name + " GLOBAL MEAN parameter gradient", errors)
        checks.close(actual, before_params[name], name + " parameter unchanged", errors, exact=True)
    for index, (x, expected_x) in enumerate(inputs):
        checks.close(x.grad, expected_x.grad, f"microbatch {index} input keeps LOCAL gradient", errors)
    if stage.offset is not stage.offset_alias:
        errors.append("Gradient synchronization broke a tied Parameter")
    checks.agree(errors, label + " averaged accumulated gradients / unused experts / input gradients")
    checks.passed(label + ": global-mean accumulated gradients", {
        "local_microbatch_sizes": sizes, "local_microbatch_weights": [n / total for n in sizes],
        "local_loss": local_loss, "global_routes_per_layer": global_counts.tolist(),
        "early_only_experts_per_layer": [torch.where(row)[0].tolist() for row in early_only] if early_only is not None else [],
        "shared_parameter_count": len(shared)})


def worker(rank, args, workdir):
    base, _ = load_base()
    base.configure_threads(args.threads)
    record = {"rank": rank, "status": "running", "passed": [], "active_phase": "import/initialization"}
    rank_path = Path(workdir) / f"rank_{rank}.json"
    write_json(rank_path, record)
    dist = None
    try:
        import torch
        import torch.distributed as dist
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if args.device == "cuda":
            torch.cuda.set_device(rank)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        store = dist.FileStore(str(Path(workdir) / "rendezvous.store"), args.ep_size)
        store.set_timeout(timedelta(seconds=args.timeout))
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo", store=store,
                                rank=rank, world_size=args.ep_size, timeout=timedelta(seconds=args.timeout))
        record["torch_version"], record["device"] = torch.__version__, str(device)
        modules = base.load_project(args.project)
        utils = modules["parallel_ep"]
        checks = base.Checks(torch, dist, args, rank, device, record, rank_path)
        info = utils.setup_ep()
        base.check_groups(checks, utils, info)
        checks.phase("nested fixture construction")
        errors, built = [], None
        try:
            built = build_models(checks, base, modules, info, {"family": "MLP", "bias": True, "k": 1, "microsteps": 1})
        except Exception as exc:
            errors.append(str(exc))
        checks.agree(errors, "nested fixture construction")
        _, stage, reference = built
        check_partition(checks, utils, stage)
        check_broadcast(checks, utils, stage, reference, info)
        check_synthetic_gradients(checks, utils, stage, info)
        for case in cases(args.ep_size):
            check_accumulation(checks, base, modules, info, case)
        dist.barrier()
        record["status"], record["active_phase"] = "passed", "complete"
    except BaseException:
        record["status"] = "failed"
        record["exception"] = traceback.format_exc()
        print(f"[FAIL rank={rank}] phase: {record['active_phase']}", flush=True)
        raise
    finally:
        write_json(rank_path, record)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "scope": {"parameter_partition_and_broadcast": True, "shared_DP_gradient_average": True,
                        "expert_gradient_scaling": True, "weighted_microbatch_accumulation": True,
                        "one_EP_group_covers_WORLD": True,
                        "objective": "mean_ranks(sum_microbatches(local_sample_weight * local_loss))",
                        "auxiliary": "weighted local-microbatch auxiliary loss, not global concatenated statistics",
                        "input_gradients": "LOCAL; parameter normalization must not rescale input gradients",
                        "optimizer_clipping_production_GPT_checkpoint": False,
                        "AMP_compile_multinode_expert_replicas": False}, "workers": []}
    workdir, exit_code = None, 1
    try:
        base, report["reference_script_sha256"] = load_base()
        base.configure_threads(args.threads)
        report["source_sha256"] = preflight(args.project, base)
        print("[PASS] source syntax / EP gradient helper definitions (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; use the Python environment used for nanoGPT.") from exc
        if args.device == "cuda":
            ensure(torch.cuda.is_available(), "CUDA is not available")
            ensure(torch.cuda.device_count() >= args.ep_size, "CUDA verification requires one visible device per worker")
        report["torch_version"] = torch.__version__
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}; accumulation cases={len(cases(args.ep_size))}.", flush=True)
        print("[INFO] Checking nested partition, broadcast, None gradients and global-mean accumulation; optimizer is pending.", flush=True)
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_grad_")
        mp.spawn(worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        records = [json.loads((Path(workdir) / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(args.ep_size)]
        ensure(all(record["status"] == "passed" for record in records), "A worker did not finish verification")
        report["checks_per_worker"] = len(records[0]["passed"])
        report["status"], exit_code = "passed", 0
        print("[PASS] EP+shared-DP gradient synchronization verification completed.", flush=True)
    except BaseException:
        report["status"], report["exception"] = "failed", traceback.format_exc()
        traceback.print_exc()
    finally:
        if workdir is not None:
            for rank in range(args.ep_size):
                path = Path(workdir) / f"rank_{rank}.json"
                report["workers"].append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                                         else {"rank": rank, "status": "missing_worker_report"})
            if args.keep_workdir:
                report["workdir"] = workdir
                print(f"[INFO] Temporary reports kept: {workdir}", flush=True)
            else:
                shutil.rmtree(workdir, ignore_errors=True)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if args.report is not None:
            write_json(args.report, report)
            print(f"[INFO] Report saved: {args.report.resolve()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
