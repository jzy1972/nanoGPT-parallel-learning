#!/usr/bin/env python3
"""Verify the nanoGPT MoE model without running train.py or editing the project.

Example (run from the nanoGPT directory):
    python verify_moe_model.py --project . --device cpu --threads 1 --report moe_model_report.json

Contract under test:
* All experts are local. Selected weights retain full-softmax probabilities.
* No capacity limit / dropped tokens. Auxiliary loss is E * sum(f_e * P_e),
  where f_e is the fraction of the N*k selected routes assigned to expert e.
* Every transformer block uses MoE when use_moe=True.
* Training loss includes the sum of layer auxiliary losses; eval loss is CE.
* Dropout=0 and eager execution are used for numerical comparisons.

References use dense expert evaluation plus routing masks, and a functional
GPT with manually computed causal attention. They do not call MoE.forward,
Block.forward, or GPT.forward. Both implementations use PyTorch autograd.

This script tests one process only. Distributed DP/EP, AMP, torch.compile,
fused CUDA optimizer kernels, and the actual train.py loop/resume are pending.
Only the JSON report (if requested) is written; project source is read-only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import io
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

torch = None  # Imported after the BLAS/OpenMP thread limit is configured.


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-process numerical verification of nanoGPT MoE."
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be greater than zero")
    for key in ("rtol", "atol"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{key} must be finite and nonnegative")
    return args


def configure_threads(threads):
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(threads)


def preflight(project):
    sources = {}
    for filename in ("model.py", "moe.py", "train.py"):
        path = project / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required project file is missing: {path}")
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, filename=str(path))
        compile(tree, str(path), "exec")
        sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def load_project(project):
    sys.path.insert(0, str(project))
    importlib.invalidate_caches()
    for name in ("model", "moe"):
        sys.modules.pop(name, None)
    moe_module = importlib.import_module("moe")
    model_module = importlib.import_module("model")
    for module, filename in ((moe_module, "moe.py"), (model_module, "model.py")):
        actual = Path(module.__file__).resolve()
        ensure(actual == (project / filename).resolve(), f"Imported wrong module: {actual}")
    return (
        model_module.GPT, model_module.GPTConfig, model_module.MLP,
        moe_module.Router, moe_module.MoE,
    )


class Verifier:
    def __init__(self, args, report):
        self.args = args
        self.report = report
        self.device = torch.device(args.device)
        self.dtype = getattr(torch, args.dtype)
        # The router softmax is float32 even when model parameters are float64.
        self.rtol = args.rtol if args.rtol is not None else (
            3e-5 if args.dtype == "float64" else 3e-4
        )
        self.atol = args.atol if args.atol is not None else (
            3e-8 if args.dtype == "float64" else 3e-5
        )
        self.comparisons = 0
        self.max_abs_error = 0.0

    def close(self, actual, expected, label):
        ensure(actual is not None, f"{label}: actual value/gradient is None")
        ensure(expected is not None, f"{label}: reference value/gradient is None")
        ensure(actual.shape == expected.shape, f"{label}: shape mismatch")
        try:
            torch.testing.assert_close(
                actual, expected, rtol=self.rtol, atol=self.atol,
            )
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from exc
        self.comparisons += 1
        if actual.numel():
            error = (actual.detach().to(torch.float64) -
                     expected.detach().to(torch.float64)).abs().max().item()
            self.max_abs_error = max(self.max_abs_error, float(error))

    def gradients(self, actual_named, reference_named, label):
        actual_named = dict(actual_named)
        reference_named = dict(reference_named)
        ensure(actual_named.keys() == reference_named.keys(), f"{label}: parameter names differ")
        for name, parameter in actual_named.items():
            actual = parameter.grad
            expected = reference_named[name].grad
            if actual is None and expected is None:
                continue
            # Dense masked references give zero gradients to unselected experts.
            if actual is None:
                actual = torch.zeros_like(expected)
            if expected is None:
                expected = torch.zeros_like(actual)
            self.close(actual, expected, f"{label}: {name}")

    def run(self, label, function):
        try:
            detail = function() or {}
        except Exception:
            self.report["checks"].append({"name": label, "status": "failed"})
            print(f"[FAIL] {label}", flush=True)
            raise
        self.report["checks"].append({"name": label, "status": "passed", **detail})
        print(f"[PASS] {label}", flush=True)


def tiny_config(GPTConfig, *, bias=True, use_moe=True, experts=4, top_k=1, coef=0.07):
    return GPTConfig(
        n_layer=2, n_head=3, n_embd=12, block_size=7, vocab_size=23,
        dropout=0.0, bias=bias, use_moe=use_moe,
        n_experts=experts, moe_top_k=top_k, moe_aux_loss_coef=coef,
    )


def sorted_routes(logits, top_k):
    # Independent selection implementation: sort rather than torch.topk.
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    indices = torch.argsort(probabilities, dim=-1, descending=True)[:, :top_k]
    mask = torch.nn.functional.one_hot(
        indices, num_classes=probabilities.shape[-1]
    ).sum(dim=1).to(probabilities.dtype)
    fractions = mask.mean(dim=0) / top_k
    auxiliary = probabilities.shape[-1] * (
        fractions * probabilities.mean(dim=0)
    ).sum()
    return probabilities, indices, mask, auxiliary


def controlled_input(v, experts, pattern="mixed", rotation=0):
    x = torch.linspace(
        -0.9, 1.1, 6 * 12, dtype=v.dtype, device=v.device
    ).reshape(2, 3, 12)
    flat = x.reshape(6, 12)
    base = torch.arange(experts, 0, -1, dtype=v.dtype, device=v.device)
    for row in range(6):
        shift = rotation + (row if pattern == "mixed" else 0)
        flat[row, :experts] = base.roll(shift)
    return x


def identity_gate(gate):
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[:, :gate.weight.shape[0]].copy_(
            torch.eye(
                gate.weight.shape[0], dtype=gate.weight.dtype, device=gate.weight.device
            )
        )


def verify_router(v, GPTConfig, Router, top_k):
    cfg = tiny_config(GPTConfig, top_k=top_k)
    candidate = Router(cfg).to(device=v.device, dtype=v.dtype)
    identity_gate(candidate.gate)
    reference = copy.deepcopy(candidate)
    x = controlled_input(v, cfg.n_experts).reshape(-1, cfg.n_embd).requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    indices, weights, probabilities = candidate(x)
    ref_logits = torch.nn.functional.linear(xr, reference.gate.weight)
    ref_probs, ref_indices, _, _ = sorted_routes(ref_logits, top_k)
    ensure(probabilities.dtype == torch.float32, "Router probabilities must be float32")
    ensure(indices.dtype == torch.long, "Router indices must have dtype torch.long")
    ensure(indices.shape == (6, top_k), "Router returned wrong index shape")
    ensure(torch.equal(indices, ref_indices), "Router top-k indices differ")
    v.close(probabilities, ref_probs, "Router full softmax")
    ref_weights = ref_probs.gather(1, ref_indices)
    v.close(weights, ref_weights, "Router selected full-softmax weights")
    v.close(probabilities.sum(-1), torch.ones(6, device=v.device), "Router probabilities sum")
    if top_k < cfg.n_experts:
        ensure(bool((weights.sum(-1) < 0.999).all()), "Selected weights were renormalized")
    probe = torch.linspace(
        -0.6, 0.8, weights.numel(), dtype=torch.float32, device=v.device
    ).reshape_as(weights)
    (weights * probe).sum().backward()
    (ref_weights * probe).sum().backward()
    v.close(x.grad, xr.grad, "Router input gradient")
    v.gradients(candidate.named_parameters(), reference.named_parameters(), "Router gradients")
    ensure(candidate.gate.bias is None, "Router gate must not have a bias")
    ensure(bool((candidate.gate.weight.grad != 0).any()), "Router main-task gradient is zero")


def verify_router_validation(GPTConfig, Router):
    for experts, top_k in ((0, 1), (-1, 1), (4, 0), (4, 5)):
        cfg = tiny_config(GPTConfig, experts=experts, top_k=top_k)
        try:
            Router(cfg)
        except ValueError:
            continue
        raise AssertionError(f"Router accepted invalid experts={experts}, top_k={top_k}")


def make_toy_expert():
    class ToyExpert(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        def forward(self, x):
            return torch.tanh(self.proj(x))
    return ToyExpert


def dense_toy_moe(reference, x):
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.nn.functional.linear(flat, reference.router.gate.weight)
    probabilities, indices, mask, auxiliary = sorted_routes(logits, reference.router.top_k)
    all_outputs = torch.stack([
        torch.tanh(torch.nn.functional.linear(
            flat, expert.proj.weight, expert.proj.bias,
        ))
        for expert in reference.experts
    ], dim=1)
    route_weights = (probabilities * mask).to(all_outputs.dtype).unsqueeze(-1)
    output = (all_outputs * route_weights).sum(dim=1).reshape_as(x)
    counts = mask.sum(dim=0).to(torch.long)
    return output, auxiliary, counts


def compare_optimizer_states(v, actual, reference, actual_named, reference_named):
    for name, parameter in actual_named:
        ref_parameter = reference_named[name]
        actual_state = actual.state.get(parameter, {})
        reference_state = reference.state.get(ref_parameter, {})
        ensure(actual_state.keys() == reference_state.keys(), f"AdamW state keys differ: {name}")
        for key, value in actual_state.items():
            other = reference_state[key]
            if isinstance(value, torch.Tensor):
                v.close(value, other, f"AdamW state {name}/{key}")
            else:
                ensure(value == other, f"AdamW state differs: {name}/{key}")


def verify_dispatch(v, GPTConfig, MoE, *, bias, top_k, pattern):
    torch.manual_seed(v.args.seed + 31 + top_k + int(bias))
    cfg = tiny_config(GPTConfig, bias=bias, top_k=top_k)
    candidate = MoE(cfg, expert_cls=make_toy_expert()).to(device=v.device, dtype=v.dtype)
    identity_gate(candidate.router.gate)
    reference = copy.deepcopy(candidate)
    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=0.002, weight_decay=0.02,
        betas=(0.9, 0.95), foreach=False, fused=False,
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=0.002, weight_decay=0.02,
        betas=(0.9, 0.95), foreach=False, fused=False,
    )
    unused_events = 0
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        x = controlled_input(v, cfg.n_experts, pattern, step).requires_grad_(True)
        xr = x.detach().clone().requires_grad_(True)
        calls = [[] for _ in candidate.experts]
        handles = []
        for expert_id, expert in enumerate(candidate.experts):
            def capture(_module, inputs, expert_id=expert_id):
                calls[expert_id].append(int(inputs[0].shape[0]))
            handles.append(expert.register_forward_pre_hook(capture))
        try:
            output, auxiliary = candidate(x)
        finally:
            for handle in handles:
                handle.remove()
        ref_output, ref_auxiliary, counts = dense_toy_moe(reference, xr)
        v.close(output, ref_output, "Sparse dispatch vs dense masked experts")
        v.close(auxiliary, ref_auxiliary, "Load balance loss")
        ensure(auxiliary.requires_grad, "Auxiliary loss is detached from autograd")
        for expert_id, count in enumerate(counts.tolist()):
            expected_calls = [count] if count else []
            ensure(calls[expert_id] == expected_calls,
                   f"Expert {expert_id}: processed {calls[expert_id]}, expected {expected_calls}")
        probe = torch.linspace(
            -0.7, 0.9, x.numel(), dtype=v.dtype, device=v.device
        ).reshape_as(x)
        ((output * probe).mean() + 0.37 * auxiliary).backward()
        ((ref_output * probe).mean() + 0.37 * ref_auxiliary).backward()
        v.close(x.grad, xr.grad, "MoE input gradient")
        v.gradients(candidate.named_parameters(), reference.named_parameters(), "MoE gradients")
        for expert_id, count in enumerate(counts.tolist()):
            if count == 0:
                unused_events += 1
                for parameter in candidate.experts[expert_id].parameters():
                    ensure(parameter.grad is None, f"Unused expert {expert_id} has a gradient")
                # Dense masks produce zeros; sparse execution produces None.
                # Clear those reference zeros so AdamW has the same sparse-step
                # semantics (including weight decay and lazy moment creation).
                for parameter in reference.experts[expert_id].parameters():
                    parameter.grad = None
        optimizer.step()
        reference_optimizer.step()
        reference_named = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            v.close(parameter, reference_named[name], f"AdamW parameter {name}")
        compare_optimizer_states(
            v, optimizer, reference_optimizer, list(candidate.named_parameters()), reference_named
        )
    if pattern == "sparse":
        ensure(unused_events > 0, "Sparse case did not exercise unselected experts")
    return {"optimizer_steps": 2, "unused_expert_events": unused_events}


def verify_auxiliary_gradient(v, GPTConfig, MoE):
    cfg = tiny_config(GPTConfig, top_k=1)
    candidate = MoE(cfg, expert_cls=make_toy_expert()).to(device=v.device, dtype=v.dtype)
    identity_gate(candidate.router.gate)
    reference = copy.deepcopy(candidate)
    x = controlled_input(v, cfg.n_experts, "sparse")
    _, auxiliary = candidate(x)
    _, ref_auxiliary, _ = dense_toy_moe(reference, x)
    v.close(auxiliary, ref_auxiliary, "Imbalanced auxiliary value")
    ensure(auxiliary.item() > 1.0, "Imbalanced routing did not produce auxiliary loss > 1")
    auxiliary.backward()
    ref_auxiliary.backward()
    v.close(candidate.router.gate.weight.grad, reference.router.gate.weight.grad, "Auxiliary-only Router gradient")
    ensure(bool((candidate.router.gate.weight.grad != 0).any()), "Auxiliary-only Router gradient is zero")
    for expert in candidate.experts:
        ensure(all(parameter.grad is None for parameter in expert.parameters()),
               "Auxiliary-only backward should not update expert parameters")
    with torch.no_grad():
        candidate.router.gate.weight.zero_()
        _, uniform_auxiliary = candidate(x)
    v.close(uniform_auxiliary, torch.ones((), device=v.device), "Uniform-probability auxiliary baseline")


def verify_single_expert(v, GPTConfig, MLP, MoE):
    cfg = tiny_config(GPTConfig, experts=1, top_k=1)
    candidate = MoE(cfg, expert_cls=MLP).to(device=v.device, dtype=v.dtype)
    reference = copy.deepcopy(candidate.experts[0])
    x = controlled_input(v, 1).requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    output, auxiliary = candidate(x)
    reference_output = reference(xr)
    v.close(output, reference_output, "One-expert MoE vs the existing MLP")
    v.close(auxiliary, torch.ones((), device=v.device), "One-expert auxiliary baseline")
    (output.square().mean() + 0.2 * auxiliary).backward()
    reference_output.square().mean().backward()
    v.close(x.grad, xr.grad, "One-expert input gradient")
    v.gradients(candidate.experts[0].named_parameters(), reference.named_parameters(),
                "One-expert MLP gradients")
    v.close(candidate.router.gate.weight.grad, torch.zeros_like(candidate.router.gate.weight),
            "One-expert Router gradient")


def parameter_reference(model):
    parameters = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in model.named_parameters()
    }
    # Keep the reference embedding/head tied to one independent leaf tensor.
    parameters["lm_head.weight"] = parameters["transformer.wte.weight"]
    return parameters


def linear(parameters, prefix, x):
    return torch.nn.functional.linear(
        x, parameters[prefix + ".weight"], parameters.get(prefix + ".bias"),
    )


def layer_norm(parameters, prefix, x, width):
    return torch.nn.functional.layer_norm(
        x, (width,), parameters[prefix + ".weight"],
        parameters.get(prefix + ".bias"), 1e-5,
    )


def dense_mlp(parameters, prefix, x):
    h = torch.nn.functional.gelu(linear(parameters, prefix + ".c_fc", x))
    return linear(parameters, prefix + ".c_proj", h)


def functional_gpt(parameters, cfg, token_ids, targets=None):
    B, T = token_ids.shape
    C = cfg.n_embd
    positions = torch.arange(T, device=token_ids.device)
    x = (
        torch.nn.functional.embedding(token_ids, parameters["transformer.wte.weight"])
        + torch.nn.functional.embedding(positions, parameters["transformer.wpe.weight"])
    )
    auxiliary_sum = x.new_zeros((), dtype=torch.float32)
    layer_counts = []
    causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    for layer in range(cfg.n_layer):
        prefix = f"transformer.h.{layer}"
        h = layer_norm(parameters, prefix + ".ln_1", x, C)
        q, k, value = linear(parameters, prefix + ".attn.c_attn", h).split(C, dim=-1)
        q = q.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        k = k.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        value = value.reshape(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(C // cfg.n_head)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        h = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(B, T, C)
        x = x + linear(parameters, prefix + ".attn.c_proj", h)
        h = layer_norm(parameters, prefix + ".ln_2", x, C)
        if cfg.use_moe:
            flat = h.reshape(-1, C)
            logits = linear(parameters, prefix + ".mlp.router.gate", flat)
            probabilities, _, mask, auxiliary = sorted_routes(logits, cfg.moe_top_k)
            experts = torch.stack([
                dense_mlp(parameters, prefix + f".mlp.experts.{expert}", flat)
                for expert in range(cfg.n_experts)
            ], dim=1)
            weighted = (probabilities * mask).to(experts.dtype).unsqueeze(-1)
            mlp_output = (experts * weighted).sum(dim=1).reshape(B, T, C)
            auxiliary_sum = auxiliary_sum + auxiliary
            layer_counts.append(mask.sum(dim=0).to(torch.long))
        else:
            mlp_output = dense_mlp(parameters, prefix + ".mlp", h)
        x = x + mlp_output
    x = layer_norm(parameters, "transformer.ln_f", x, C)
    logits = linear(parameters, "lm_head", x)
    ce = None
    if targets is not None:
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
        )
    return logits, ce, auxiliary_sum, layer_counts


def model_batch(v, cfg):
    ids = torch.arange(10, dtype=torch.long, device=v.device).reshape(2, 5) % cfg.vocab_size
    targets = (ids + 3) % cfg.vocab_size
    targets[0, 0] = -1  # Verify the existing CE ignore_index contract as well.
    return ids, targets


def verify_gpt(v, GPT, GPTConfig, MLP, MoE, *, bias, use_moe, top_k=1, coef=0.07):
    torch.manual_seed(v.args.seed + 103 + top_k + int(bias))
    cfg = tiny_config(
        GPTConfig, bias=bias, use_moe=use_moe, experts=3, top_k=top_k, coef=coef
    )
    candidate = GPT(cfg).to(device=v.device, dtype=v.dtype)
    ensure(candidate.transformer.wte.weight is candidate.lm_head.weight, "GPT head/embedding are not tied")
    if use_moe:
        for block in candidate.transformer.h:
            ensure(isinstance(block.mlp, MoE), "A GPT block did not use MoE")
            ensure(len(block.mlp.experts) == cfg.n_experts, "Wrong expert count in GPT")
            ensure(all(isinstance(expert, MLP) for expert in block.mlp.experts),
                   "GPT experts must use the existing MLP")
    reference = parameter_reference(candidate)
    ids, targets = model_batch(v, cfg)
    candidate.train()
    logits, training_loss = candidate(ids, targets)
    ref_logits, ref_ce, ref_auxiliary, layer_counts = functional_gpt(reference, cfg, ids, targets)
    expected_training = ref_ce + cfg.moe_aux_loss_coef * ref_auxiliary if use_moe else ref_ce
    v.close(logits, ref_logits, "GPT full training logits")
    v.close(training_loss, expected_training, "GPT training loss CE + layer auxiliary sum")
    ensure(training_loss.requires_grad, "GPT training loss is detached")
    training_loss.backward()
    expected_training.backward()
    reference_unique = {
        name: reference[name] for name, _ in candidate.named_parameters()
    }
    v.gradients(candidate.named_parameters(), reference_unique, "GPT all parameter gradients")
    for layer, counts in enumerate(layer_counts):
        for expert, count in enumerate(counts.tolist()):
            if count == 0:
                for parameter in candidate.transformer.h[layer].mlp.experts[expert].parameters():
                    ensure(parameter.grad is None, "Unselected GPT expert has a gradient")
    candidate.eval()
    with torch.no_grad():
        eval_logits, eval_loss = candidate(ids, targets)
        inference_logits, inference_loss = candidate(ids)
    v.close(eval_logits, ref_logits.detach(), "GPT full evaluation logits")
    v.close(eval_loss, ref_ce.detach(), "GPT evaluation loss is CE")
    v.close(inference_logits, ref_logits.detach()[:, [-1], :], "GPT inference last-token logits")
    ensure(inference_loss is None, "GPT inference without targets must return loss=None")
    return {"parameters": sum(p.numel() for p in candidate.parameters()),
            "training_auxiliary_sum": float(ref_auxiliary.detach().item())}


def build_unfused_optimizer(model):
    decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.dim() >= 2]
    nodecay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=0.002, betas=(0.9, 0.95), foreach=False, fused=False,
    )


def verify_checkpoint(v, GPT, GPTConfig):
    cfg = tiny_config(GPTConfig, bias=True, experts=3, top_k=2, coef=0.11)
    model = GPT(cfg).to(device=v.device, dtype=v.dtype)
    optimizer = build_unfused_optimizer(model)
    ids, targets = model_batch(v, cfg)
    model.train()
    _, loss = model(ids, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    stream = io.BytesIO()
    torch.save({
        "model_args": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, stream)
    stream.seek(0)
    loaded = torch.load(stream, map_location=v.device, weights_only=True)
    restored = GPT(GPTConfig(**loaded["model_args"])).to(device=v.device, dtype=v.dtype)
    restored.load_state_dict(loaded["model"], strict=True)
    ensure(restored.config == model.config, "MoE checkpoint configuration did not round-trip")
    ensure(restored.transformer.wte.weight is restored.lm_head.weight, "Restored weights are not tied")
    restored_optimizer = build_unfused_optimizer(restored)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    compare_optimizer_states(v, optimizer, restored_optimizer,
                             list(model.named_parameters()), dict(restored.named_parameters()))
    model.train()
    restored.train()
    _, original_loss = model(ids, targets)
    _, restored_loss = restored(ids, targets)
    v.close(restored_loss, original_loss, "Restored next training loss")
    original_loss.backward()
    restored_loss.backward()
    v.gradients(restored.named_parameters(), dict(model.named_parameters()), "Restored next gradients")
    optimizer.step()
    restored_optimizer.step()
    original_parameters = dict(model.named_parameters())
    for name, parameter in restored.named_parameters():
        v.close(parameter, original_parameters[name], f"Restored next AdamW update: {name}")
    compare_optimizer_states(v, optimizer, restored_optimizer,
                             list(model.named_parameters()), dict(restored.named_parameters()))
    return {"scope": "in-memory model/unfused AdamW round-trip; actual train.py resume is pending"}


def main(argv=None):
    global torch
    args = parse_args(argv)
    args.project = args.project.expanduser().resolve()
    report = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(args.project),
        "scope": "single-process eager MoE model math",
        "not_tested": ["distributed DP", "EP", "AMP", "torch.compile",
                       "fused CUDA optimizer", "actual train.py loop/resume"],
        "checks": [],
    }
    success = False
    try:
        configure_threads(args.threads)
        report["source_sha256"] = preflight(args.project)
        print("[PASS] model.py / moe.py / train.py syntax (train.py is not executed).", flush=True)
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is not installed in this Python environment.") from exc
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        if torch.device(args.device).type not in ("cpu", "cuda"):
            raise ValueError("--device must be cpu, cuda, or cuda:N")
        if torch.device(args.device).type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this verifier also supports --device cpu.")
            if torch.device(args.device).index is None:
                args.device = "cuda:0"
            torch.cuda.set_device(torch.device(args.device))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        GPT, GPTConfig, MLP, Router, MoE = load_project(args.project)
        v = Verifier(args, report)
        report.update(torch_version=torch.__version__, device=str(v.device),
                      dtype=args.dtype, threads=args.threads, rtol=v.rtol, atol=v.atol)
        print(f"[INFO] Single process; device={v.device}; dtype={args.dtype}; threads={args.threads}.", flush=True)
        print("[INFO] Model math / gradients / unfused AdamW are checked; distributed training is pending.", flush=True)
        v.run("Router invalid-configuration checks", lambda: verify_router_validation(GPTConfig, Router))
        for top_k in (1, 2, 4):
            v.run(f"Router top_k={top_k}: probabilities / indices / main-task gradients",
                  lambda top_k=top_k: verify_router(v, GPTConfig, Router, top_k))
        for bias in (True, False):
            for top_k, pattern in ((1, "mixed"), (2, "mixed"), (4, "mixed"),
                                   (1, "sparse"), (2, "sparse")):
                v.run(f"MoE bias={bias}, top_k={top_k}, {pattern}: dispatch / gradients / 2 AdamW steps",
                      lambda bias=bias, top_k=top_k, pattern=pattern:
                      verify_dispatch(v, GPTConfig, MoE, bias=bias, top_k=top_k, pattern=pattern))
        v.run("Auxiliary loss: uniform baseline / imbalanced value / Router-only gradients",
              lambda: verify_auxiliary_gradient(v, GPTConfig, MoE))
        v.run("One-expert boundary: existing MLP equivalence / constant auxiliary / zero Router gradient",
              lambda: verify_single_expert(v, GPTConfig, MLP, MoE))
        for bias in (True, False):
            v.run(f"Dense GPT bias={bias}: functional reference / gradients / eval / inference",
                  lambda bias=bias: verify_gpt(
                      v, GPT, GPTConfig, MLP, MoE, bias=bias, use_moe=False))
            for top_k in (1, 2, 3):
                v.run(f"MoE GPT bias={bias}, top_k={top_k}: functional reference / total loss / all gradients / eval",
                      lambda bias=bias, top_k=top_k: verify_gpt(
                          v, GPT, GPTConfig, MLP, MoE, bias=bias, use_moe=True, top_k=top_k))
        v.run("MoE GPT auxiliary coefficient=0: CE-only objective",
              lambda: verify_gpt(v, GPT, GPTConfig, MLP, MoE,
                                bias=True, use_moe=True, top_k=1, coef=0.0))
        v.run("MoE model / AdamW checkpoint round-trip and next update",
              lambda: verify_checkpoint(v, GPT, GPTConfig))
        report.update(status="passed", tensor_comparisons=v.comparisons,
                      max_abs_error=v.max_abs_error)
        success = True
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        print(f"[FAIL] {report['error']}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if args.report is not None:
            report_path = args.report.expanduser().resolve()
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[INFO] Report saved: {report_path}", flush=True)
            except OSError as exc:
                print(f"[FAIL] Cannot write report: {exc}", file=sys.stderr, flush=True)
                success = False
    if success:
        print("[PASS] MoE model numerical verification completed (single process).", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
