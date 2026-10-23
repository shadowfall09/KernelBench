"""All-element and normwise precision contracts against a trusted FP64 oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionContract:
    norm_rtol: float
    norm_atol: float
    element_rtol: float
    element_atol: float
    element_scale_rtol: float = 0.0

    def __post_init__(self):
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in vars(self).values()) or not (self.norm_rtol or self.norm_atol):
            raise ValueError("Precision thresholds must be finite and nonnegative with a nonzero norm budget")


def to_oracle(value):
    if isinstance(value, torch.Tensor):
        dtype = torch.complex128 if value.is_complex() else (torch.float64 if value.is_floating_point() else value.dtype)
        return value.detach().to(dtype=dtype).clone()
    if isinstance(value, dict):
        return {key: to_oracle(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_oracle(item) for item in value)
    return value


def _rms(values):
    scale = values.abs().max()
    if not torch.isfinite(scale) or scale == 0:
        return scale
    return scale * torch.sqrt(torch.mean((values / scale).square()))


def precision_gate(reference, candidate, oracle, contract, *, _self_check=False):
    """Reference must meet the same contract; do not bless an uncalibrated oracle."""
    rows, issues = [], []

    def visit(ref, cand, truth, path):
        if isinstance(ref, torch.Tensor):
            if type(cand) is not torch.Tensor or type(truth) is not torch.Tensor:
                issues.append(f"{path}: output is not a materialized tensor")
                return
            if cand.shape != ref.shape or truth.shape != ref.shape or cand.dtype != ref.dtype:
                issues.append(f"{path}: shape or output dtype mismatch")
                return
            if not (ref.is_floating_point() or ref.is_complex()):
                if truth.dtype != ref.dtype or not torch.equal(ref, cand) or not torch.equal(ref, truth):
                    issues.append(f"{path}: discrete outputs differ")
                return
            if truth.dtype != (torch.complex128 if ref.is_complex() else torch.float64):
                issues.append(f"{path}: oracle must use float64 or complex128")
                return
            if ref.is_complex():
                visit(torch.view_as_real(ref), torch.view_as_real(cand), torch.view_as_real(truth), path + ".complex")
                return
            masks = (torch.isnan, torch.isposinf, torch.isneginf)
            if any(not torch.equal(mask(ref), mask(truth)) or not torch.equal(mask(cand), mask(truth)) for mask in masks):
                issues.append(f"{path}: non-finite positions differ from oracle")
                return
            finite = torch.isfinite(truth)
            truth = truth[finite].double()
            if not truth.numel():
                issues.append(f"{path}: no finite oracle values; numerical contract not informative")
                return
            scale = _rms(truth)
            limit = contract.norm_atol + contract.norm_rtol * scale
            element_limit = contract.element_atol + contract.element_rtol * truth.abs() + contract.element_scale_rtol * scale
            row = {"path": path, "elements": truth.numel(), "oracle_rms": scale.item(), "norm_limit": limit.item()}
            for name, value in (("reference", ref), ("candidate", cand)):
                delta = (value[finite].double() - truth).abs()
                rms = _rms(delta)
                bad = int((delta > element_limit).sum().item())
                row[name] = {"rms_error": rms.item(), "max_abs_error": delta.max().item(), "elements_failed": bad,
                             "passed": bool(rms <= limit) and bad == 0}
                if not row[name]["passed"]:
                    issues.append(f"{path}: {name} violates precision contract")
            rows.append(row)
        elif isinstance(ref, dict):
            if not isinstance(cand, dict) or not isinstance(truth, dict) or ref.keys() != cand.keys() or ref.keys() != truth.keys():
                issues.append(f"{path}: dictionary structure mismatch")
            else:
                for key in ref:
                    visit(ref[key], cand[key], truth[key], f"{path}.{key}")
        elif isinstance(ref, (list, tuple)):
            if type(ref) is not type(cand) or type(ref) is not type(truth) or len(ref) != len(cand) or len(ref) != len(truth):
                issues.append(f"{path}: sequence structure mismatch")
            else:
                for index, (a, b, c) in enumerate(zip(ref, cand, truth)):
                    visit(a, b, c, f"{path}[{index}]")
        elif type(ref) is not type(cand) or ref != cand or ref != truth:
            issues.append(f"{path}: scalar mismatch")

    visit(reference, candidate, oracle, "output")
    # Structural errors may prevent the joint walk from visiting the reference.
    # Qualify it independently on failures; an empty leaf list is not evidence
    # that a mismatched oracle shape or discrete dtype is acceptable.
    qualification = precision_gate(reference, reference, oracle, contract, _self_check=True) if issues and not _self_check else None
    reference_qualified = qualification["passed"] if qualification is not None else not issues
    return {"passed": not issues, "issues": issues, "leaves": rows, "all_elements_required": True,
            "oracle_dtype": "float64/complex128", "reference_qualified": reference_qualified,
            "reference_issues": qualification["issues"] if qualification is not None else issues if _self_check else []}


def signal_audit(value, *, absolute_tolerance):
    """Diagnose weak tests. Constant outputs are warnings, not semantic errors."""
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("Invalid diagnostic tolerance")
    rows = []

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            if not (item.is_floating_point() or item.is_complex()):
                rows.append({"path": path, "informative": True, "kind": "discrete_exact"})
                return
            flat = item.detach().abs().double().reshape(-1)
            finite = flat[torch.isfinite(flat)]
            meaningful = bool(finite.numel()) and bool((finite > absolute_tolerance).any())
            rows.append({"path": path, "elements": flat.numel(), "finite_elements": finite.numel(),
                         "max_abs": finite.max().item() if finite.numel() else None,
                         "rms": _rms(finite).item() if finite.numel() else None,
                         "fraction_above_atol": (finite > absolute_tolerance).double().mean().item() if finite.numel() else 0,
                         "constant": bool(torch.equal(item, torch.full_like(item, item.flatten()[0].item()))) if item.numel() else True,
                         "informative": meaningful and finite.numel() == flat.numel()})
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "output")
    return {"informative": bool(rows) and all(row["informative"] for row in rows), "leaves": rows,
            "purpose": "test_signal_diagnostic; a constant output can be semantically valid"}
