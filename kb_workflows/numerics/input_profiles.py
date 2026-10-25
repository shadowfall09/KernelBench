"""Reviewed, source-bound nominal input distributions; no SDK imports at import."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

PROTOCOL = "kernelbench_pro_nominal_input_profiles_v1"
ACTIVE = ContextVar("kernelbench_nominal_input_profile", default=None)


def recipes(pid, level=1):
    if level in {2, 3}:
        from .model_input_profiles import policy

        return tuple(policy(level, pid)["recipes"])
    if level != 1 or type(pid) is not int or not 1 <= pid <= 100:
        raise ValueError("Nominal input distribution adapters currently cover original L1 only")
    if pid == 98:
        return ("identity", "probability_peaks")
    extra = ("bounded_product_walk",) if pid == 90 else ("separated_triplets",) if pid == 99 else ()
    extrema = ("normal",) if pid in {41, 42, 43, 49, 51, 52, 53} else ()
    return ("identity", *extra, "signed", "positive", *extrema)


def selection_policy(level, pid, source=None):
    if level == 1:
        minimum = .4 if pid in {14, 15, 19} else .9
        family = "reviewed_operator"
    else:
        from .model_input_profiles import policy

        model = policy(level, pid, source)
        minimum, family = model["minimum_fraction"], model["family"]
    return {"level": level, "problem_id": pid, "family": family, "recipes": list(recipes(pid, level)),
            "minimum_fraction": minimum, "forward_seed": 42,
            "output_variation": "at_least_two_distinct_bounded_output_probes_at_fixed_forward_seed",
            "selection": "first_reference_qualified_recipe; no_candidate_or_timing_feedback"}


def transform(inputs, pid, recipe, *, level=1):
    """Change values only; keep shapes, strides, discrete roles and parameters."""
    if level in {2, 3}:
        from .model_input_profiles import transform as model_transform

        return model_transform(inputs, level, pid, recipe)
    if level != 1 or recipe not in recipes(pid):
        raise ValueError("Unreviewed nominal input distribution")
    import torch

    from src.validation import clone_tensors

    result = clone_tensors(inputs)
    if recipe == "identity":
        return result
    if not isinstance(result, list) or not result or any(isinstance(value, (tuple, list, dict)) for value in result):
        raise ValueError("Reviewed L1 profiles require the declared flat argument list")
    if recipe == "bounded_product_walk":
        value = result[0]
        if value.ndim < 2 or not value.is_floating_point():
            raise ValueError("Product profile requires a floating input and scan axis 1")
        if value.shape[1] == 0:
            return result
        # Every exact prefix is a signed power of two in [2^-3, 2^3].
        # This keeps long products informative, including FP16/BF16 storage,
        # without shrinking dimensions or fixing a deterministic answer.
        exponents = torch.randint(-3, 4, value.shape, device=value.device)
        previous = torch.cat((torch.zeros_like(exponents.narrow(1, 0, 1)), exponents.narrow(1, 0, value.shape[1] - 1)), 1)
        signs = torch.randint(0, 2, value.shape, dtype=torch.int8, device=value.device) * 2 - 1
        previous_signs = torch.cat((torch.ones_like(signs.narrow(1, 0, 1)), signs.narrow(1, 0, value.shape[1] - 1)), 1)
        factors = torch.pow(2.0, (exponents - previous).float()) * (signs * previous_signs)
        value.copy_(factors)
        return result
    if recipe == "probability_peaks":
        for index, value in enumerate(result):
            if not isinstance(value, torch.Tensor) or value.ndim < 2 or not value.is_floating_point() or not value.shape[-1]:
                raise ValueError("KL profile requires floating probability rows")
            mass = 0.05 if index == 0 else 0.9
            positions = torch.randint(value.shape[-1], (*value.shape[:-1], 1), device=value.device)
            value.fill_((1 - mass) / value.shape[-1])
            value.scatter_add_(-1, positions, torch.full(positions.shape, mass, dtype=value.dtype, device=value.device))
        return result
    if recipe == "separated_triplets":
        if len(result) != 3 or any(value.shape != result[0].shape for value in result):
            raise ValueError("Triplet profile requires equal embedding shapes")
        result[1].copy_(result[0] + 2)
        result[2].copy_(result[0] + result[2] * 0.01)
        return result
    for index, value in enumerate(result):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or (pid == 100 and index == 1):
            continue
        updated = torch.randn_like(value) if recipe == "normal" else value * 2 - 1 if recipe == "signed" else value.abs() + 0.5
        if pid == 14:
            updated = updated.triu()
        elif pid == 15:
            updated = updated.tril()
        value.copy_(updated)
    return result


def wrap_factory(source, factory):
    active = ACTIVE.get()
    if active is None:
        return factory
    if hashlib.sha256(source.encode()).hexdigest() != active["source_sha256"]:
        raise ValueError("Nominal input policy cannot apply to another reference source")

    @wraps(factory)
    def make():
        from ..evaluation.measurement import input_signature

        values = factory()
        before = input_signature(values)
        output = transform(values, active["problem_id"], active["recipe"], level=active["level"])
        if input_signature(output) != before:
            raise ValueError("Calibrated input distribution changed shapes, strides or storage roles")
        active["factory_calls"] += 1
        if active["input_signature"] is None:
            active["input_signature"] = before
        elif active["input_signature"] != before:
            raise ValueError("Nominal input signature changed after calibration")
        return output
    return make


@contextmanager
def context(source, pid, recipe, *, level=1):
    if recipe not in recipes(pid, level):
        raise ValueError("Unknown calibrated input distribution")
    if level in {2, 3}:
        from .model_input_profiles import policy

        policy(level, pid, source)
    state = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "problem_id": pid, "level": level,
             "recipe": recipe, "factory_calls": 0, "input_signature": None}
    token = ACTIVE.set(state)
    try:
        yield state
    finally:
        ACTIVE.reset(token)
