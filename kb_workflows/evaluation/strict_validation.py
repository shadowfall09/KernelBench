"""Opt-in reviewed contracts, public development cases and held-out final cases.

Additional checks do not replace the existing correctness or timing protocol.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import torch

from src.validation import clone_tensors, compare_outputs, preserve_backend_settings, synchronize

from ..numerics.numerical import PrecisionContract, precision_gate, signal_audit, to_oracle
from ..numerics.specs import contract_for
from .evidence import check
from .state_contract import state_checks


def _backend_flags():
    flags = [(torch.backends.cudnn, "benchmark"), (torch.backends.cudnn, "deterministic")]
    flags += [(torch.backends.cuda.matmul, name) for name in (
        "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
        if hasattr(torch.backends.cuda.matmul, name)]
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        flags += [(torch.backends, "fp32_precision"), (torch.backends.cuda.matmul, "fp32_precision"),
                  (torch.backends.cudnn, "fp32_precision")]
    else:
        flags += [(torch.backends.cuda.matmul, "allow_tf32"), (torch.backends.cudnn, "allow_tf32")]
    return [(obj, name, getattr(obj, name)) for obj, name in flags]


@contextmanager
def environment_guard(changes):
    """Record names only, restore process state, and never serialize env values."""
    env, dtype, threads = dict(os.environ), torch.get_default_dtype(), torch.get_num_threads()
    grad = torch.is_grad_enabled()
    flags = _backend_flags()
    try:
        with preserve_backend_settings():
            try:
                yield
            finally:
                changes.extend(f"backend.{name}" for obj, name, value in flags if getattr(obj, name) != value)
    finally:
        if torch.is_grad_enabled() != grad:
            changes.append("grad_enabled")
            torch.set_grad_enabled(grad)
        changes.extend(f"env.{name}" for name in env.keys() | os.environ.keys() if env.get(name) != os.environ.get(name))
        if torch.get_default_dtype() != dtype:
            changes.append("default_dtype")
            torch.set_default_dtype(dtype)
        if torch.get_num_threads() != threads:
            changes.append("cpu_threads")
            torch.set_num_threads(threads)
        for name in set(os.environ) - env.keys():
            del os.environ[name]
        os.environ.update(env)


def _strided(value):
    backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device=value.device)
    result = backing[..., ::2]
    result.copy_(value)
    return result


def cases(contract, *, hidden, seed, device, dtype):
    """Small legal cases; no original input allocation or global resizing guesses."""
    generator = torch.Generator(device=device).manual_seed(seed)
    family = contract["family"]
    shapes = ((17, 17), (35, 35)) if family == "square_matmul" else ((3, 17, 19), (5, 31, 23))
    if family not in {"square_matmul", "matmul", "batched_matmul"}:
        shapes = ((3, 17, 19), (5, 23, 31)) if family == "reduction" else ((3, 17), (5, 31))
    if not hidden:
        shapes = shapes[:1]
    for index, dimensions in enumerate(shapes):
        def rand(shape):
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)
        if family == "square_matmul":
            n = dimensions[0]
            args = [rand((n, n)), rand((n, n))]
        elif family == "matmul":
            m, k, n = dimensions
            args = [rand((m, k)), rand((k, n))]
        elif family == "batched_matmul":
            m, k, n = dimensions
            args = [rand((2, m, k)), rand((2, k, n))]
        else:
            args = [rand(dimensions)]
        yield f"shape_{index}/signed", args
        for name in ("near_constant", "wide_range", "border", "gelu_discriminator"):
            if name == "gelu_discriminator" and contract.get("operator") != "gelu":
                continue
            fresh = clone_tensors(args)
            if name == "near_constant":
                fresh[0].fill_(128)
                fresh[0].reshape(-1)[::2] += 1
            elif name == "wide_range":
                fresh[0].mul_(32)
                fresh[0].reshape(-1)[::2] *= 0.0001
            elif name == "border":
                fresh[0].zero_()
                fresh[0].reshape(-1)[-1] = -7 if contract.get("operator") == "argmin" else 7
            else:
                fresh[0].fill_(2.699)
                fresh[0].reshape(-1)[::2] *= -1
            yield f"shape_{index}/{name}", fresh
        if hidden:
            yield f"shape_{index}/noncontiguous", [_strided(value) for value in args]


def run_checks(reference, candidate, contract, *, hidden, seed, device, dtype, set_seed, oracle_on_cpu=False):
    numerical, signals, sensitivity, all_cases = [], [], [], []
    policy = PrecisionContract(**contract["precision"][str(dtype).removeprefix("torch.")])
    changes, exceptions = [], []
    mode_record = None
    with torch.no_grad():
        for label, args in cases(contract, hidden=hidden, seed=seed, device=device, dtype=dtype):
            try:
                ref, cand = copy.deepcopy(reference), copy.deepcopy(candidate)
                oracle = copy.deepcopy(reference).to(dtype=torch.float64, **({"device": "cpu"} if oracle_on_cpu else {}))
                set_seed(seed)
                oracle_args = [value.detach().cpu() if isinstance(value, torch.Tensor) else value for value in args] if oracle_on_cpu else args
                truth = clone_tensors(oracle(*to_oracle(oracle_args)))
                set_seed(seed)
                expected = clone_tensors(ref(*clone_tensors(args)))
                set_seed(seed)
                with environment_guard(changes):
                    actual = clone_tensors(cand(*clone_tensors(args)))
                    synchronize(device)
                if oracle_on_cpu:
                    from ..hardware.neuron_worker import map_tensors

                    expected = map_tensors(expected, lambda value: value.detach().cpu())
                    actual = map_tensors(actual, lambda value: value.detach().cpu())
                numerical.append({"case": label, **precision_gate(expected, actual, truth, policy)})
                signals.append({"case": label, **signal_audit(expected, absolute_tolerance=policy.element_atol)})
                # A same-object change catches identity caches. Reference sensitivity decides applicability.
                mutable = clone_tensors(args)
                ref, cand = copy.deepcopy(reference), copy.deepcopy(candidate)
                set_seed(seed)
                first_ref = clone_tensors(ref(*clone_tensors(mutable)))
                set_seed(seed)
                with environment_guard(changes):
                    first_cand = clone_tensors(cand(*mutable))
                mutable[0].add_(0.25)
                set_seed(seed)
                next_ref = clone_tensors(ref(*clone_tensors(mutable)))
                set_seed(seed)
                with environment_guard(changes):
                    next_cand = clone_tensors(cand(*mutable))
                    synchronize(device)
                reference_changed = compare_outputs(first_ref, next_ref, 0) is not None
                cached = reference_changed and compare_outputs(first_cand, next_cand, 0) is None
                sensitivity.append({"case": label, "reference_changed": reference_changed, "cached": cached})
                if mode_record is None:
                    with environment_guard(changes):
                        mode_record = state_checks(reference, candidate, args, seed=seed, set_seed=set_seed,
                                                   tolerance=policy.element_atol + policy.element_rtol, device=device)
                all_cases.append(label)
            except Exception as exc:
                exceptions.append({"case": label, "error": type(exc).__name__, "message": str(exc)[:1000]})
    complete = bool(all_cases) and not exceptions
    informative = any(row["informative"] for row in signals)
    checks = {
        "precision_contract": check("passed" if complete and all(row["passed"] for row in numerical) else "failed",
                                    reason="FP64 oracle and every-element plus RMS bounds", values=numerical),
        "reference_signal": check("passed" if complete and informative else "failed",
                                  reason="At least one legal input case must contain informative reference signal", values=signals),
        "input_families": check("passed" if complete else "failed", reason="All declared legal cases must execute",
                                values={"completed": all_cases, "errors": exceptions, "hidden_shapes": hidden}),
        "input_sensitivity": check("passed" if complete and not any(row["cached"] for row in sensitivity) else "failed",
                                   reason="Same input objects updated between calls; only reference-sensitive changes count", values=sensitivity),
        "environment_integrity": check("failed" if changes else "passed", reason="Restore and audit process settings around candidate execution",
                                       values={"changed_names": sorted(set(changes))}),
        "mode_state": mode_record or check("not_measured", reason="No complete case available for mode/state validation"),
    }
    return checks


def evaluate_contract(payload, reference_source, build):
    from src.eval import get_torch_dtype_from_string, load_custom_model, load_original_model_and_inputs, set_seed

    contract = contract_for(payload["level"], payload["problem_id"], reference_source)
    if contract is None:
        return {name: check("not_measured", reason="Task has no reviewed strict contract") for name in
                ("precision_contract", "reference_signal", "input_families", "input_sensitivity", "environment_integrity", "mode_state")}
    settings = payload["eval_config"]
    dtype = get_torch_dtype_from_string(settings["precision"])
    device = torch.device("cuda:0")
    Model, init, _ = load_original_model_and_inputs(reference_source, {})
    seed = settings.get("evaluation_seed", 42)
    set_seed(seed)
    reference = Model(*init()).to(device=device, dtype=dtype)
    changes = []
    with environment_guard(changes):
        Candidate = load_custom_model(payload["kernel_src"], {}, str(build))
        set_seed(seed)
        candidate = Candidate(*init()).to(device=device, dtype=dtype)
    checks = run_checks(reference, candidate, contract, hidden=settings.get("use_hidden_tests", False),
                        seed=seed, device=device, dtype=dtype, set_seed=set_seed)
    if changes:
        checks["environment_integrity"] = check("failed", reason="Candidate import/initialization changed process settings",
                                                 values={"changed_names": sorted(set(changes))})
    return checks
