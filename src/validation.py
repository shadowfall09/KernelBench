"""Shared output comparison and tensor snapshots for correctness evaluation."""

import math

import torch


def compare_outputs(expected, actual, tolerance, path="output"):
    """Return an actionable mismatch, or None. Discrete values are always exact."""
    if isinstance(expected, torch.Tensor):
        if type(actual) is not torch.Tensor:
            return f"{path}: expected a materialized torch.Tensor"
        if expected.shape != actual.shape:
            return f"{path}: shape {tuple(expected.shape)} != {tuple(actual.shape)}"
        if expected.dtype != actual.dtype:
            return f"{path}: dtype {expected.dtype} != {actual.dtype}"
        if not (expected.is_floating_point() or expected.is_complex()):
            return None if torch.equal(expected, actual) else f"{path}: discrete values differ"
        if expected.is_complex():
            return compare_outputs(torch.view_as_real(expected), torch.view_as_real(actual), tolerance, path)
        if torch.isfinite(expected).all() and torch.isfinite(actual).all():
            return None if torch.allclose(expected, actual, atol=tolerance, rtol=tolerance) else f"{path}: finite values differ"
        for name, mask in (("NaN", torch.isnan), ("+Inf", torch.isposinf), ("-Inf", torch.isneginf)):
            if not torch.equal(mask(expected), mask(actual)):
                return f"{path}: {name} locations differ"
        finite = torch.isfinite(expected)
        if not torch.allclose(expected[finite], actual[finite], atol=tolerance, rtol=tolerance):
            return f"{path}: finite values differ"
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            return f"{path}: dictionary keys differ"
        pairs = [(expected[k], actual[k], f"{path}.{k}") for k in expected]
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            return f"{path}: sequence structure differs"
        pairs = [(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(expected, actual))]
    else:
        if isinstance(expected, float) and isinstance(actual, float):
            same = (math.isnan(expected) and math.isnan(actual)) or math.isclose(
                expected, actual, abs_tol=tolerance, rel_tol=tolerance
            )
            return None if same else f"{path}: scalar differs"
        return None if type(expected) is type(actual) and expected == actual else f"{path}: scalar differs"
    for a, b, label in pairs:
        issue = compare_outputs(a, b, tolerance, label)
        if issue:
            return issue
    return None


def clone_tensors(value):
    """Prevent in-place inputs and reused output buffers from contaminating checks."""
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous() and all(stride > 0 for stride in value.stride()):
            # clone() can compact sliced tensors, hiding stride-dependent bugs.
            output = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
            return output.copy_(value.detach())
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tensors(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tensors(v) for v in value)
    return value


def synchronize(device):
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device=device)
