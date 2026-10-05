"""Helpers for the verification scripts, not replacements for training code."""
from __future__ import annotations

import ast
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent = str(path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec.loader.exec_module(module)
    return module


def load_project_function(project: Path, name: str, candidate: ModuleType):
    """Compile only a named function; never execute train.py's module body."""
    import torch
    import torch.distributed as dist

    for filename in ("parallel_utils.py", "train.py"):
        path = project / filename
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        node = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
            None,
        )
        if node is None:
            continue
        namespace = {
            "torch": torch,
            "dist": dist,
            "nn": torch.nn,
            "ColumnParallelLinear": candidate.ColumnParallelLinear,
            "RowParallelLinear": candidate.RowParallelLinear,
        }
        isolated = ast.Module(body=[node], type_ignores=[])
        exec(compile(isolated, str(path), "exec"), namespace)
        return namespace[name]
    raise RuntimeError(
        f"Cannot find {name} in {project}/parallel_utils.py or train.py. "
        "Keep your existing function in one of these files."
    )


class CpuZeroDropoutTracker:
    """Test-only adapter. Valid exclusively when the verification dropout is 0."""

    def __init__(self):
        self.states = {}

    def fork(self, name):
        return nullcontext()


def setup_tracker(candidate: ModuleType, device, rank: int) -> None:
    if not hasattr(candidate, "tp_rng_tracker"):
        raise RuntimeError("model.py must expose tp_rng_tracker")
    if device.type == "cpu":
        candidate.tp_rng_tracker = CpuZeroDropoutTracker()
    else:
        candidate.tp_rng_tracker.states.clear()
        candidate.tp_rng_tracker.add("attention", 9000 + rank)


def oracle_shard(name, full, candidate_model, candidate_module, tp_rank, tp_size):
    """Independent index-based oracle; never calls the user's slicing function."""
    import torch

    parent_name, leaf = name.rsplit(".", 1)
    parent = candidate_model.get_submodule(parent_name)
    if isinstance(parent, candidate_module.ColumnParallelLinear):
        if parent_name.endswith("attn.c_attn"):
            width = full.shape[0] // 3
            if full.shape[0] != 3 * width or width % tp_size:
                raise AssertionError(f"Invalid full QKV dimensions: {name}")
            start = tp_rank * (width // tp_size)
            indices = torch.arange(start, start + width // tp_size, device=full.device)
            indices = torch.cat((indices, indices + width, indices + 2 * width))
            return full.index_select(0, indices)
        if full.shape[0] % tp_size:
            raise AssertionError(f"Column dimension not divisible: {name}")
        width = full.shape[0] // tp_size
        return full.narrow(0, tp_rank * width, width)
    if isinstance(parent, candidate_module.RowParallelLinear) and leaf == "weight":
        if full.shape[1] % tp_size:
            raise AssertionError(f"Row dimension not divisible: {name}")
        width = full.shape[1] // tp_size
        return full.narrow(1, tp_rank * width, width)
    return full


def math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


class CollectiveChecks:
    """Defer local assertion failures to a shared stage boundary."""

    def __init__(self, device, rank, atol, rtol):
        self.device = device
        self.rank = rank
        self.atol = atol
        self.rtol = rtol
        self.errors: list[str] = []
        self.completed: list[str] = []

    def condition(self, condition, message):
        if not condition and len(self.errors) < 12:
            self.errors.append(message)

    def close(self, actual, expected, label, *, exact=False):
        import torch

        try:
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.0 if exact else self.atol,
                rtol=0.0 if exact else self.rtol,
                check_device=False,
            )
        except AssertionError as exc:
            if len(self.errors) < 12:
                self.errors.append(f"{label}: {str(exc)[:1500]}")

    def finish(self, stage):
        import torch
        import torch.distributed as dist

        flag = torch.tensor(int(bool(self.errors)), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            for error in self.errors:
                print(f"[FAIL rank={self.rank}] {stage}: {error}", flush=True)
            dist.barrier()
            raise AssertionError(f"Verification failed at: {stage}; see rank messages")
        if self.rank == 0:
            print(f"[PASS] {stage}", flush=True)
        self.completed.append(stage)


def compare_trees(actual, expected, label, *, atol=0.0, rtol=0.0):
    """Compare checkpoint data after tensors have been mapped to CPU."""
    import torch

    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol, check_device=False,
            msg=lambda message: f"{label}: {message}",
        )
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: dictionary keys differ")
        for key in actual:
            compare_trees(actual[key], expected[key], f"{label}.{key}", atol=atol, rtol=rtol)
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{label}: sequences differ")
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_trees(left, right, f"{label}[{i}]", atol=atol, rtol=rtol)
    elif isinstance(actual, float) and isinstance(expected, float):
        import math

        if not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
