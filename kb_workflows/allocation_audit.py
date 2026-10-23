"""Separate candidate-allocation poisoning controls; SDK-light receipt replay.

This observes the current thread's ATen allocation dispatch. It does not claim
coverage of raw CUDA allocators, cached Graph buffers, or arbitrary threads.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

from .common import fingerprint
from .evidence import check

PROTOCOL = "kernelbench_pro_allocation_poison_v1"
PATTERNS = ("nan_max", "finite_min")
STAGES = {pattern: f"allocation_{pattern}" for pattern in PATTERNS}
ENVIRONMENT = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1", "CUDA_LAUNCH_BLOCKING": "1", "PYTORCH_ALLOC_CONF": "backend:native"}
EMPTY_OPS = {"aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty", "aten::new_empty_strided",
             "aten::empty_permuted", "aten::_empty_affine_quantized", "aten::_empty_per_channel_affine_quantized",
             "aten::empty_quantized"}
RESIZE_OPS = {"aten::resize_", "aten::resize_as_"}
ACTIVE = ContextVar("kernelbench_allocation_audit", default=None)


def validate_inline(payload):
    settings = payload["eval_config"]
    enabled = settings.get("allocation_validation", False)
    if type(enabled) is not bool or enabled and settings.get("device_backend", "cuda") != "cuda":
        raise ValueError("Allocation validation requires a boolean and the CUDA backend")
    return enabled


class Collector:
    def __init__(self, payload, pattern):
        if (not validate_inline(payload) or pattern not in PATTERNS or payload.get("task", "evaluate") != "evaluate"
                or payload["eval_config"].get("measure_performance") is not False
                or payload["eval_config"].get("diagnostic_stage") != STAGES.get(pattern)):
            raise ValueError("Allocation poisoning requires a known pattern and untimed correctness request")
        self.payload, self.pattern, self.windows = payload, pattern, []
        self.environment = {key: os.environ.get(key) == value for key, value in ENVIRONMENT.items()}
        self.environment["legacy_allocator_config_absent"] = "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        import torch

        self.cuda_initialized = torch.cuda.is_initialized()
        self.torch_version = str(torch.__version__)

    @contextmanager
    def window(self, phase, *, label=None, seed=None, order=None):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

        from .strict_validation import environment_guard

        groups, changes = {}, []
        row = {"phase": phase, "label": label, "seed": seed, "order": order,
               "allocations": [], "environment_changes": [], "mode_present_after": False}
        self.windows.append(row)
        pattern = self.pattern

        def record(op, tensor, status, elements=None):
            key = op, str(tensor.dtype), str(tensor.device), status
            group = groups.setdefault(key, {"op": op, "dtype": str(tensor.dtype), "device": str(tensor.device),
                                            "status": status, "allocations": 0, "elements": 0})
            group["allocations"] += 1
            if status == "metadata_only":
                group["elements"] = None  # Symbolic sizes are not measured device storage.
            elif group["elements"] is not None:
                group["elements"] += tensor.numel() if elements is None else elements

        class Poison(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                op = func._schema.name
                before = args[0].numel() if op in RESIZE_OPS else None
                contiguous_before = args[0].is_contiguous() if op in RESIZE_OPS else None
                output = func(*args, **(kwargs or {}))
                if op not in EMPTY_OPS | RESIZE_OPS or not isinstance(output, torch.Tensor):
                    return output
                if output.device.type == "meta" or hasattr(output, "fake_mode"):
                    record(op, output, "metadata_only")
                elif op in RESIZE_OPS:
                    if output.numel() > before:
                        if contiguous_before and output.is_contiguous():
                            # Preserve the initialized prefix; only new logical
                            # elements have unspecified contents after growth.
                            self.fill(op, output.reshape(-1)[before:], elements=output.numel() - before)
                        else:
                            record(op, output, "unsupported_resize_growth")
                elif type(output) is not torch.Tensor or output.layout != torch.strided or output.is_quantized:
                    record(op, output, "unsupported_tensor")
                else:
                    self.fill(op, output)
                return output

            def fill(self, op, output, elements=None):
                try:
                    if output.dtype == torch.bool:
                        value = pattern == "nan_max"
                    elif output.is_complex():
                        value = complex(float("nan"), float("nan")) if pattern == "nan_max" else complex(-0.5, 0.5)
                    elif output.is_floating_point():
                        value = float("nan") if pattern == "nan_max" else -0.5
                    else:
                        info = torch.iinfo(output.dtype)
                        value = info.max if pattern == "nan_max" else info.min
                    with torch.no_grad():
                        output.fill_(value)
                    record(op, output, "poisoned", elements)
                except (RuntimeError, TypeError, ValueError):
                    record(op, output, "unsupported_fill", elements)

        mode = Poison()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            with environment_guard(changes), mode:
                try:
                    yield
                finally:
                    row["mode_present_after"] = mode in _get_current_dispatch_mode_stack()
        finally:
            if (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.utils.deterministic.fill_uninitialized_memory) != (deterministic, warn_only, fill):
                changes.append("deterministic_policy")
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.utils.deterministic.fill_uninitialized_memory = fill
            row["allocations"] = list(groups.values())
            row["environment_changes"] = sorted(set(changes))

    def document(self):
        settings = self.payload["eval_config"]
        return {"protocol": PROTOCOL, "pattern": self.pattern, "request_sha256": fingerprint(self.payload),
                "precision": settings["precision"], "environment": self.environment, "windows": self.windows,
                "cuda_initialized_before_audit": self.cuda_initialized, "torch_version": self.torch_version,
                "algorithm_policy_forced": False, "reference_allocations_poisoned": False,
                "scope": "candidate_import_constructor_and_native_paired_calls; current_thread_ATen_dispatch_only",
                "timing_samples_collected": False, "complete_device_memory_coverage": False}


@contextmanager
def audit(payload, pattern):
    collector = Collector(payload, pattern)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


@contextmanager
def candidate_scope(phase, *, label=None, seed=None, order=None):
    collector = ACTIVE.get()
    if collector is None:
        yield
    else:
        with collector.window(phase, label=label, seed=seed, order=order):
            yield


def verdict(document, payload, cases, *, pattern):
    """Recompute coverage; a poisoned pass is bounded functional evidence only."""
    def missing(reason):
        return check("not_measured", reason=reason)

    if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL or document.get("pattern") != pattern
            or document.get("request_sha256") != fingerprint(payload)
            or document.get("precision") != payload["eval_config"]["precision"]
            or document.get("cuda_initialized_before_audit") is not False
            or not isinstance(document.get("torch_version"), str) or document["torch_version"].split("+")[0] != "2.12.0"
            or document.get("environment") != {**dict.fromkeys(ENVIRONMENT, True), "legacy_allocator_config_absent": True}
            or document.get("algorithm_policy_forced") is not False or document.get("reference_allocations_poisoned") is not False
            or document.get("timing_samples_collected") is not False or document.get("complete_device_memory_coverage") is not False
            or document.get("scope") != "candidate_import_constructor_and_native_paired_calls; current_thread_ATen_dispatch_only"
            or not isinstance(cases, list) or not cases):
        return missing("No matching allocation pattern, fresh-worker environment or native case evidence")
    windows = document.get("windows")
    if not isinstance(windows, list) or any(not isinstance(row, dict) for row in windows):
        return missing("Allocation windows are missing")
    phases = [row.get("phase") for row in windows]
    if "import" not in phases or "constructor" not in phases or any(phase not in {"import", "constructor", "forward"} for phase in phases):
        return missing("Allocation audit omitted candidate import or construction")
    forwards = [row for row in windows if row["phase"] == "forward"]
    expected = [(row.get("label"), row.get("seed"), order) for row in cases
                for order in (["candidate_first", "reference_first"] if row.get("passed") is True else row.get("orders", []) or ["candidate_first"])]
    if [(row.get("label"), row.get("seed"), row.get("order")) for row in forwards] != expected:
        return missing("Allocation audit omitted or changed native calls, cases or seeds")
    allocations = elements = 0
    for row in windows:
        if row.get("mode_present_after") is not True or not isinstance(row.get("environment_changes"), list):
            return missing("Allocation dispatch mode was removed or its integrity is unknown")
        if row["environment_changes"]:
            return check("failed", reason="Candidate changed process settings during allocation control")
        groups = row.get("allocations")
        if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
            return missing("Allocation counters are malformed")
        for group in groups:
            if (group.get("op") not in EMPTY_OPS | RESIZE_OPS or not isinstance(group.get("dtype"), str)
                    or not isinstance(group.get("device"), str) or type(group.get("allocations")) is not int
                    or group["allocations"] <= 0
                    or (group.get("status") == "metadata_only" and group.get("elements") is not None)
                    or (group.get("status") != "metadata_only" and (type(group.get("elements")) is not int or group["elements"] < 0))):
                return missing("Allocation counter identities are incomplete")
            if group.get("status") not in {"poisoned", "metadata_only"}:
                return missing("Observed allocation/resize could not be safely poisoned")
            if group["status"] == "poisoned":
                allocations += group["allocations"]
                elements += group["elements"]
    if any(row.get("passed") is not True for row in cases):
        return check("failed", reason="Candidate failed native correctness under allocation poisoning")
    if any(type(row.get("seed")) is not int or row.get("orders") != ["candidate_first", "reference_first"] for row in cases):
        return missing("Poisoned native comparison did not complete both call orders")
    return check("passed", reason="Candidate passed native correctness under the declared allocation poison pattern",
                 values={"pattern": pattern, "case_count": len(cases), "candidate_calls": len(forwards),
                         "poisoned_allocations": allocations, "poisoned_elements": elements,
                         "complete_device_memory_coverage": False})
