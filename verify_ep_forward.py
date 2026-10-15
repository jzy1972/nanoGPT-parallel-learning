#!/usr/bin/env python3
"""Verify EP routing and raw backward contributions, without editing project code.

Run from nanoGPT (one standalone verifier; no dataset or sibling verifier):
    python verify_ep_forward.py --project . --device cpu --ep-size 2 --threads 1 --report ep_forward_report.json

Expected project interface:
  parallel_ep.py: get_expert_range(), setup_ep(), exchange_counts(), AllToAll
  moe_ep.py: MoEEP(config, expert_cls, ep_group), forward -> (output, aux_loss)
  moe.py: MoE(config, expert_cls), with router.gate and experts
  model.py: MLP(config)

The current topology has one EP group covering WORLD. Nonexpert DP gradient
averaging is NOT invoked: Router/input gradients match this rank's local loss,
while expert gradients match the SUM of local-loss contributions over ranks.
The reference evaluates all experts and applies an independent routing mask;
it never calls the project's Router.forward or MoE.forward.

Coverage: uneven and zero communication counts, inverse communication,
AllToAll autograd, global expert ownership/loading, non-renormalized top-k
weights, local auxiliary loss, input/Router/expert gradients, unused experts,
an empty expert owner, inputs without requires_grad, actual project MLP,
and no-grad evaluation. All model comparisons use dropout=0 and eager mode.

Not covered: nonexpert DP synchronization, expert gradient scaling, optimizer,
clipping/overflow, GPT integration, AMP, torch.compile, checkpoint, multi-node.
CPU uses Gloo with direct FileStore to avoid TCPStore/libuv rendezvous on
Windows. CUDA uses NCCL with one local worker per visible device. EP size is
configurable. Importing this script does not import PyTorch or start workers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
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
    parser = argparse.ArgumentParser(description="EP forward/raw-backward numerical verification.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=int, default=90, help="Collective timeout in seconds.")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true", help="Keep temporary per-rank reports.")
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


def configure_threads(threads):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def preflight(project):
    contracts = {
        "parallel_ep.py": {"get_expert_range", "setup_ep", "exchange_counts", "AllToAll"},
        "moe_ep.py": {"MoEEP"},
        "moe.py": {"MoE", "Router"},
        "model.py": {"MLP"},
    }
    hashes = {}
    for filename, required in contracts.items():
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing project file: {path}")
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        names = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        ensure(required <= names, f"{filename}: missing top-level definitions {sorted(required - names)}")
        if filename == "moe_ep.py":
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MoEEP")
            ensure(any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in cls.body),
                   "MoEEP must define forward(self, x); its constructor alone is not enough.")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_project(project):
    sys.path.insert(0, str(project))
    importlib.invalidate_caches()
    names = ("parallel_ep", "moe", "moe_ep", "model")
    for name in names:
        sys.modules.pop(name, None)
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        ensure(Path(module.__file__).resolve() == (project / (name + ".py")).resolve(),
               f"Imported {name} from the wrong project: {module.__file__}")
        modules[name] = module
    return modules


class Checks:
    def __init__(self, torch, dist, args, rank, device, record, rank_path):
        self.torch, self.dist = torch, dist
        self.args, self.rank, self.device = args, rank, device
        self.record, self.rank_path = record, rank_path
        self.rtol = args.rtol if args.rtol is not None else (3e-5 if args.dtype == "float64" else 3e-4)
        self.atol = args.atol if args.atol is not None else (3e-8 if args.dtype == "float64" else 3e-5)

    def phase(self, label):
        self.record["active_phase"] = label
        write_json(self.rank_path, self.record)

    def close(self, actual, expected, label, errors, exact=False):
        try:
            ensure(actual is not None and expected is not None, f"{label}: tensor/gradient is None")
            self.torch.testing.assert_close(actual, expected, rtol=0 if exact else self.rtol,
                                           atol=0 if exact else self.atol)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    def agree(self, errors, label):
        flag = self.torch.tensor(int(bool(errors)), dtype=self.torch.int32, device=self.device)
        self.dist.all_reduce(flag, op=self.dist.ReduceOp.MAX)
        if flag.item():
            for message in errors[:8]:
                print(f"[FAIL rank={self.rank}] {label}: {message}", flush=True)
            self.record["local_errors"] = errors
            raise AssertionError(f"Verification failed: {label}; inspect the per-rank errors.")

    def passed(self, label, details=None):
        self.record["passed"].append({"label": label, "details": details or {}})
        if self.rank == 0:
            print(f"[PASS] {label}", flush=True)


def check_groups(checks, utils, info):
    checks.phase("group mapping and expert ranges")
    errors = []
    try:
        for key, expected in (("world_rank", checks.rank), ("ep_rank", checks.rank),
                              ("world_size", checks.args.ep_size), ("ep_size", checks.args.ep_size)):
            ensure(info[key] == expected, f"{key}: expected {expected}, got {info[key]}")
        for key in ("ep_group", "shared_dp_group"):
            group = info[key]
            ensure(checks.dist.get_rank(group=group) == checks.rank, f"Wrong rank in {key}")
            ensure(checks.dist.get_world_size(group=group) == checks.args.ep_size, f"Wrong size of {key}")
        for size in (1, 2, 3, 4, checks.args.ep_size):
            for local in (1, 2, 3):
                for rank in range(size):
                    actual = utils.get_expert_range(size * local, size, rank)
                    ensure(tuple(actual) == (rank * local, (rank + 1) * local),
                           f"Wrong expert range E={size * local}, EP={size}, rank={rank}: {actual}")
        for arguments in ((0, 1, 0), (4, 0, 0), (5, 2, 0), (4, 2, -1), (4, 2, 2)):
            try:
                utils.get_expert_range(*arguments)
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError(f"Invalid expert range was accepted: {arguments}")
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, "WORLD EP / shared-DP mapping and expert ownership")
    checks.passed("WORLD EP / shared-DP mapping and expert ownership")


def count_cases(size):
    return [
        ("self_only", [[src + 1 if src == dst else 0 for dst in range(size)] for src in range(size)]),
        ("uneven", [[(src + 2 * dst) % 3 for dst in range(size)] for src in range(size)]),
        ("one_owner", [[src + 1 if dst == 0 else 0 for dst in range(size)] for src in range(size)]),
        ("empty_source", [[0 if src == 0 else (src + dst) % 3 for dst in range(size)] for src in range(size)]),
        ("all_empty", [[0 for _ in range(size)] for _ in range(size)]),
    ]


def check_collectives(checks, utils, info):
    torch, rank, device = checks.torch, checks.rank, checks.device
    size, group = checks.args.ep_size, info["ep_group"]
    width = 3
    for name, matrix in count_cases(size):
        label = f"AllToAll {name}: counts / values / inverse / backward"
        checks.phase(label)
        sends = matrix[rank]
        receives = [matrix[src][rank] for src in range(size)]
        errors = []
        actual_counts = utils.exchange_counts(torch.tensor(sends, dtype=torch.long, device=device), group)
        checks.close(actual_counts, torch.tensor(receives, dtype=torch.long, device=device), "receive counts", errors, exact=True)
        checks.agree(errors, label + " counts")

        def input_for(src):
            rows = sum(matrix[src])
            return torch.arange(rows * width, dtype=torch.float64, device=device).reshape(rows, width) + src * 1000

        x = input_for(rank).t().contiguous().t().requires_grad_(True)
        y = utils.AllToAll.apply(x, sends, receives, group)
        pieces = []
        for src in range(size):
            start = sum(matrix[src][:rank])
            pieces.append(input_for(src)[start:start + matrix[src][rank]])
        expected_y = torch.cat(pieces, dim=0)
        errors = []
        checks.close(y, expected_y, "received rows/source order", errors, exact=True)
        checks.agree(errors, label + " forward")
        returned = utils.AllToAll.apply(y.detach().requires_grad_(True), receives, sends, group)
        errors = []
        checks.close(returned, x.detach(), "inverse routing", errors, exact=True)
        checks.agree(errors, label + " inverse")

        def cotangent(dst):
            rows = sum(matrix[src][dst] for src in range(size))
            # Dyadic values keep the communication-only adjoint check exact,
            # including on heterogeneous CUDA devices.
            return (torch.arange(rows * width, dtype=torch.float64, device=device) + dst * 0.125).reshape(rows, width)

        (y * cotangent(rank)).sum().backward()
        grad_pieces = []
        for dst in range(size):
            start = sum(matrix[src][dst] for src in range(rank))
            grad_pieces.append(cotangent(dst)[start:start + matrix[rank][dst]])
        errors = []
        checks.close(x.grad, torch.cat(grad_pieces, dim=0), "input adjoint", errors, exact=True)
        checks.agree(errors, label + " backward")
        checks.passed(label)


def toy_expert_class(torch):
    class ToyExpert(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.fc = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        def forward(self, x):
            return torch.tanh(self.fc(x))
    return ToyExpert


def model_cases(size):
    cases = []
    experts = 2 * size
    for family in ("toy", "MLP"):
        for bias in (True, False):
            for k in sorted({1, min(2, experts), experts}):
                variants = [("rotating", True, "mixed"), ("one_owner", False, "main")]
                if family == "toy":
                    variants += [("cross_rank", True, "main"), ("rank_disjoint", True, "aux")]
                for pattern, input_grad, objective in variants:
                    cases.append({"family": family, "bias": bias, "k": k,
                                  "experts_per_rank": 2, "pattern": pattern,
                                  "input_grad": input_grad, "objective": objective})
    cases.append({"family": "toy", "bias": False, "k": 1, "experts_per_rank": 1,
                  "pattern": "one_owner", "input_grad": False, "objective": "mixed"})
    return cases


def controlled_input(torch, config, rank, pattern, device, dtype):
    n = 2 + rank % 3
    width, experts, k = config.n_embd, config.n_experts, config.moe_top_k
    values = torch.arange(n * width, device=device, dtype=dtype).reshape(n, width)
    x = 0.1 * torch.sin(values + rank)
    local = config.experts_per_rank
    for token in range(n):
        logits = -2.0 - 0.03 * torch.arange(experts, device=device, dtype=dtype)
        if pattern == "one_owner":
            base = 0
        elif pattern == "cross_rank":
            base = ((rank + 1) % config.ep_size) * local
        elif pattern == "rank_disjoint":
            base = rank * local
        else:
            base = (rank + token) * local
        for slot in range(k):
            logits[(base + slot) % experts] = 3.0 - 0.25 * slot + 0.001 * token
        x[token, :experts] = logits
    return x.reshape(1, n, width)


def independent_reference(torch, model, x, config):
    flat = x.reshape(-1, config.n_embd)
    logits = torch.nn.functional.linear(flat, model.router.gate.weight)
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    indices = torch.argsort(probs, dim=-1, descending=True, stable=True)[:, :config.moe_top_k]
    weights = torch.zeros_like(probs).scatter(1, indices, probs.gather(1, indices))
    all_outputs = torch.stack([expert(flat) for expert in model.experts], dim=1)
    output = (all_outputs * weights.unsqueeze(-1)).sum(dim=1).to(dtype=x.dtype).reshape_as(x)
    selected = torch.zeros_like(probs, dtype=torch.bool).scatter(1, indices, True)
    counts = selected.sum(dim=0)
    fractions = counts.to(probs.dtype) / (flat.shape[0] * config.moe_top_k)
    auxiliary = config.n_experts * (fractions * probs.mean(dim=0)).sum()
    return output, auxiliary, counts


def objective(torch, output, auxiliary, rank, mode):
    if mode == "aux":
        return 0.07 * auxiliary
    target = torch.cos(torch.arange(output.numel(), device=output.device, dtype=output.dtype) * 0.17 + rank).reshape_as(output)
    main = (output * target).sum() / output.shape[1] + 0.031 * output.square().mean()
    return main + 0.07 * auxiliary if mode == "mixed" else main


def build_models(checks, modules, info, case):
    torch = checks.torch
    experts = case["experts_per_rank"] * checks.args.ep_size
    config = SimpleNamespace(n_embd=max(experts + 2, 8), n_experts=experts,
                             moe_top_k=case["k"], dropout=0.0, bias=case["bias"],
                             n_layer=2, block_size=8, experts_per_rank=case["experts_per_rank"],
                             ep_size=checks.args.ep_size)
    expert_cls = toy_expert_class(torch) if case["family"] == "toy" else modules["model"].MLP
    torch.manual_seed(checks.args.seed)
    reference = modules["moe"].MoE(config, expert_cls=expert_cls)
    stage = modules["moe_ep"].MoEEP(config, expert_cls=expert_cls, ep_group=info["ep_group"])
    dtype = getattr(torch, checks.args.dtype)
    reference = reference.to(device=checks.device, dtype=dtype)
    stage = stage.to(device=checks.device, dtype=dtype)
    with torch.no_grad():
        reference.router.gate.weight.zero_()
        reference.router.gate.weight[:, :experts].copy_(torch.eye(experts, device=checks.device, dtype=dtype))
    first = checks.rank * case["experts_per_rank"]
    end = first + case["experts_per_rank"]
    expected_names = {name for name in reference.state_dict()
                      if not name.startswith("experts.") or first <= int(name.split(".")[1]) < end}
    ensure(set(stage.state_dict()) == expected_names,
           "EP state_dict keys must contain the full Router and only globally named local experts; "
           f"missing={sorted(expected_names - set(stage.state_dict()))}, extra={sorted(set(stage.state_dict()) - expected_names)}")
    expected_parameters = {name for name, _ in reference.named_parameters()
                           if not name.startswith("experts.") or first <= int(name.split(".")[1]) < end}
    ensure(set(dict(stage.named_parameters())) == expected_parameters,
           "Registered local parameters are missing/aliased or contain other ranks' experts")
    ensure(set(stage.experts.keys()) == {str(i) for i in range(first, end)},
           "Use a ModuleDict with GLOBAL expert IDs, rather than renumbering each rank from zero.")
    for name, expected in (("n_experts", experts), ("local_n_experts", case["experts_per_rank"]),
                           ("ep_size", checks.args.ep_size), ("ep_rank", checks.rank),
                           ("expert_start", first), ("expert_end", end)):
        ensure(getattr(stage, name) == expected, f"MoEEP.{name} should be {expected}")
    stage.load_state_dict({name: value for name, value in reference.state_dict().items() if name in expected_names}, strict=True)
    return config, stage, reference


def check_model(checks, modules, info, case):
    torch, dist = checks.torch, checks.dist
    label = (f"{case['family']} bias={case['bias']} k={case['k']} "
             f"local_experts={case['experts_per_rank']} {case['pattern']} "
             f"input_grad={case['input_grad']} objective={case['objective']}")
    checks.phase(label + " construction/loading")
    errors, built = [], None
    try:
        built = build_models(checks, modules, info, case)
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + " ownership/loading")
    config, stage, reference = built
    dtype = getattr(torch, checks.args.dtype)
    values = controlled_input(torch, config, checks.rank, case["pattern"], checks.device, dtype)
    x = values.clone().requires_grad_(case["input_grad"])
    reference_x = values.clone().requires_grad_(case["input_grad"])
    expected_output, expected_aux, local_counts = independent_reference(torch, reference, reference_x, config)
    expected_loss = objective(torch, expected_output, expected_aux, checks.rank, case["objective"])
    checks.phase(label + " forward")
    output, auxiliary = stage(x)
    loss = objective(torch, output, auxiliary, checks.rank, case["objective"])
    errors = []
    checks.close(output, expected_output, "output", errors)
    checks.close(auxiliary, expected_aux, "local auxiliary loss", errors)
    checks.close(loss, expected_loss, "objective", errors)
    try:
        ensure(auxiliary.dtype == torch.float32, "Auxiliary loss should stay float32")
        ensure(output.shape == x.shape and output.dtype == x.dtype, "Output shape/dtype must match x")
    except Exception as exc:
        errors.append(str(exc))
    checks.agree(errors, label + " forward")

    checks.phase(label + " backward (all ranks must participate, including empty owners)")
    loss.backward()
    expected_loss.backward()
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=info["ep_group"])
    actual_params = dict(stage.named_parameters())
    errors = []
    for name, param in reference.named_parameters():
        if name.startswith("experts."):
            expected_grad = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
            # Every reference replica contributes its LOCAL objective. SUM is
            # intentional: raw EP owner gradients have not been DP-normalized.
            dist.all_reduce(expected_grad, op=dist.ReduceOp.SUM, group=info["ep_group"])
            if name not in actual_params:
                continue
            expert = int(name.split(".")[1])
            should_be_unused = case["objective"] == "aux" or global_counts[expert].item() == 0
            if should_be_unused:
                if actual_params[name].grad is not None:
                    errors.append(f"{name}: globally unused expert must keep grad=None")
            else:
                checks.close(actual_params[name].grad, expected_grad, name + " global SUM gradient", errors)
        else:
            checks.close(actual_params[name].grad, param.grad, name + " LOCAL gradient", errors)
    if case["input_grad"]:
        checks.close(x.grad, reference_x.grad, "LOCAL input gradient", errors)
    elif x.grad is not None or x.requires_grad:
        errors.append("Original x without requires_grad must remain unchanged; use a separate packed tensor for the communication graph")
    checks.agree(errors, label + " Router/input/expert gradients")
    checks.passed(label, {"local_routes": local_counts.tolist(), "global_routes": global_counts.tolist(),
                          "loss": float(loss.detach().item()), "auxiliary": float(auxiliary.detach().item())})
    return config, stage, reference, values


def check_eval(checks, config, stage, reference, values):
    checks.phase("MoEEP no-grad evaluation")
    stage.eval()
    reference.eval()
    with checks.torch.no_grad():
        output, auxiliary = stage(values)
        expected_output, expected_auxiliary, _ = independent_reference(checks.torch, reference, values, config)
    errors = []
    checks.close(output, expected_output, "eval output", errors)
    checks.close(auxiliary, expected_auxiliary, "eval auxiliary loss", errors)
    if output.requires_grad or auxiliary.requires_grad:
        errors.append("no_grad evaluation must not build a backward graph")
    checks.agree(errors, "MoEEP no-grad evaluation")
    checks.passed("MoEEP no-grad evaluation")


def worker(rank, args, workdir):
    configure_threads(args.threads)
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
        record["torch_version"] = torch.__version__
        record["device"] = str(device)
        modules = load_project(args.project)
        checks = Checks(torch, dist, args, rank, device, record, rank_path)
        info = modules["parallel_ep"].setup_ep()
        check_groups(checks, modules["parallel_ep"], info)
        check_collectives(checks, modules["parallel_ep"], info)
        last = None
        for case in model_cases(args.ep_size):
            last = check_model(checks, modules, info, case)
        check_eval(checks, *last)
        dist.barrier()
        record["status"] = "passed"
        record["active_phase"] = "complete"
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
              "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
              "scope": {"EP_forward_and_raw_backward": True, "one_EP_group_covers_WORLD": True,
                        "nonexpert_DP_gradient_averaging": False, "expert_gradient_scaling": False,
                        "optimizer_training_checkpoint": False, "AMP_compile_multinode": False,
                        "expert_gradient_reference": "SUM of per-rank local objectives",
                        "Router_and_input_gradient_reference": "this rank's LOCAL objective",
                        "auxiliary_definition": "E * sum((local_route_counts / (N*k)) * local_mean_probs)"},
              "workers": []}
    workdir = None
    exit_code = 1
    try:
        configure_threads(args.threads)
        report["source_sha256"] = preflight(args.project)
        print("[PASS] source syntax / EP interface definitions (static only).", flush=True)
        try:
            import torch
            import torch.multiprocessing as mp
        except ImportError as exc:
            raise RuntimeError("PyTorch is required. Run this verifier with the Python environment used for nanoGPT.") from exc
        if args.device == "cuda":
            ensure(torch.cuda.is_available(), "CUDA is not available in this Python environment")
            ensure(torch.cuda.device_count() >= args.ep_size,
                   f"CUDA verification needs one visible device per worker: EP={args.ep_size}, visible={torch.cuda.device_count()}")
        report["torch_version"] = torch.__version__
        workdir = tempfile.mkdtemp(prefix="nanogpt_ep_verify_")
        print(f"[INFO] {args.ep_size} {args.device} workers; threads={args.threads}; dtype={args.dtype}; "
              f"MoEEP cases={len(model_cases(args.ep_size))}.", flush=True)
        print("[INFO] Checking EP communication and raw gradients; DP averaging/scaling and optimizer are pending.", flush=True)
        mp.spawn(worker, args=(args, workdir), nprocs=args.ep_size, join=True)
        records = [json.loads((Path(workdir) / f"rank_{rank}.json").read_text(encoding="utf-8"))
                   for rank in range(args.ep_size)]
        ensure(all(record["status"] == "passed" for record in records), "A worker did not complete verification")
        report["status"] = "passed"
        report["checks_per_worker"] = len(records[0]["passed"])
        exit_code = 0
        print("[PASS] EP forward/raw-backward numerical verification completed.", flush=True)
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if workdir is not None:
            for rank in range(args.ep_size):
                path = Path(workdir) / f"rank_{rank}.json"
                if path.is_file():
                    report["workers"].append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    report["workers"].append({"rank": rank, "status": "missing_worker_report"})
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
