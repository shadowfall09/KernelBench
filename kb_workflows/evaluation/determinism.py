"""Scoped repeat checks with reviewed atomics policies; imports do not load Torch."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import struct
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from ..common import fingerprint, read_json
from .evidence import check

POLICY_PROTOCOL = "kernelbench_pro_determinism_policies_v1"
PROTOCOL = "kernelbench_pro_repeat_consistency_v1"
ACTIVE = ContextVar("kernelbench_repeat_auditor", default=None)


def read_policies(path):
    document = read_json(Path(path))
    if (not isinstance(document, dict) or document.get("protocol") != POLICY_PROTOCOL
            or not isinstance(document.get("tasks"), dict)):
        raise ValueError("Missing determinism policy manifest")
    for key, row in document["tasks"].items():
        if (not re.fullmatch(r"[1235]/[1-9][0-9]*", key) or not isinstance(row, dict)
                or set(row) - {"reference_sha256", "mode", "atol", "rtol", "review", "atomics_allowed"}
                or not re.fullmatch("[0-9a-f]{64}", row.get("reference_sha256", ""))
                or row.get("mode") not in {"bitwise", "bounded_atomics"}):
            raise ValueError("Invalid task determinism declaration")
        if row["mode"] == "bounded_atomics":
            bounds = [row.get("atol"), row.get("rtol")]
            review = row.get("review", {})
            if (row.get("atomics_allowed") is not True or not isinstance(review, dict)
                    or not review.get("reviewer") or not review.get("rationale")
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in bounds)
                    or not any(bounds)):
                raise ValueError("Bounded atomics require a reviewed positive numerical bound")
        elif row.get("atomics_allowed", False) or row.get("atol", 0) or row.get("rtol", 0):
            raise ValueError("Bitwise policy cannot declare a numerical exemption")
    return document


def policy_for(path, level, pid, source):
    row = read_policies(path)["tasks"].get(f"{level}/{pid}") if path else None
    if row and row["reference_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Determinism policy reference changed; review atomics and error bounds")
    return copy.deepcopy(row) if row else {"mode": "bitwise", "atomics_allowed": False,
        "reference_sha256": hashlib.sha256(source.encode()).hexdigest(), "policy_source": "default_exact_repeat_contract"}


def validate_inline(policy, source):
    """Workers receive a single frozen policy, never permission to loosen accuracy."""
    if not isinstance(policy, dict) or policy.get("reference_sha256") != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Repeat policy is missing or bound to another reference")
    if policy.get("mode") == "bitwise":
        if policy.get("atomics_allowed", False) or policy.get("atol", 0) or policy.get("rtol", 0):
            raise ValueError("Bitwise repeat policy cannot grant an error exemption")
    elif policy.get("mode") == "bounded_atomics":
        review, bounds = policy.get("review", {}), (policy.get("atol"), policy.get("rtol"))
        if (policy.get("atomics_allowed") is not True or not isinstance(review, dict)
                or not review.get("reviewer") or not review.get("rationale")
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in bounds)
                or not any(bounds)):
            raise ValueError("Worker atomics policy needs a reviewed finite numerical bound")
    else:
        raise ValueError("Unknown repeat policy mode")


def signature(value, *, typed=False):
    import torch

    if type(value) is torch.Tensor:
        if value.layout != torch.strided:
            raise ValueError("Repeat audit requires dense materialized tensors")
        digest = hashlib.sha256()

        def blocks(tensor):
            # Bound host copies even for large, strided nominal inputs. Slice in
            # logical row-major order so this retains the old canonical digest.
            if tensor.numel() * tensor.element_size() <= 1024**2:
                yield tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                return
            axis = next(index for index, size in enumerate(tensor.shape) if size > 1)
            row_bytes = tensor.numel() // tensor.shape[axis] * tensor.element_size()
            step = max(1, 1024**2 // row_bytes)
            for start in range(0, tensor.shape[axis], step):
                yield from blocks(tensor.narrow(axis, start, min(step, tensor.shape[axis] - start)))

        for raw in blocks(value):
            digest.update(raw)
        return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes_sha256": digest.hexdigest()}
    if isinstance(value, dict):
        if typed:
            return {"type": "dict", "items": [[signature(key, typed=True), signature(item, typed=True)]
                                              for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))]}
        return {key: signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [signature(item, typed=typed) for item in value]}
    if type(value) is float:
        return {"type": "float", "bits": struct.pack(">d", value).hex()}
    if type(value) is complex:
        return {"type": "complex", "real": signature(value.real), "imag": signature(value.imag)}
    return {"type": type(value).__name__, "value": value}


def repeat_issue(first, current, policy, path="output"):
    import torch

    if isinstance(first, torch.Tensor):
        if type(current) is not torch.Tensor or first.shape != current.shape or first.dtype != current.dtype:
            return f"{path}: repeat output structure, shape or dtype changed"
        if policy["mode"] == "bitwise" or not (first.is_floating_point() or first.is_complex()):
            return None if signature(first) == signature(current) else f"{path}: repeated output bits differ"
        a, b = first.detach().cpu(), current.detach().cpu()
        if a.is_complex():
            return repeat_issue(torch.view_as_real(a), torch.view_as_real(b), policy, path)
        if any(not torch.equal(mask(a), mask(b)) for mask in (torch.isnan, torch.isposinf, torch.isneginf)):
            return f"{path}: non-finite locations/signs differ between repeats"
        finite = torch.isfinite(a)
        return None if torch.allclose(a[finite], b[finite], atol=policy["atol"], rtol=policy["rtol"]) else f"{path}: declared atomic variation bound exceeded"
    if isinstance(first, dict):
        if not isinstance(current, dict) or first.keys() != current.keys():
            return f"{path}: repeat mapping changed"
        rows = [(first[key], current[key], f"{path}.{key}") for key in first]
    elif isinstance(first, (tuple, list)):
        if type(first) is not type(current) or len(first) != len(current):
            return f"{path}: repeat sequence changed"
        rows = [(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(first, current))]
    else:
        return None if signature(first) == signature(current) else f"{path}: repeat scalar changed"
    return next((issue for a, b, label in rows if (issue := repeat_issue(a, b, policy, label))), None)


@contextmanager
def preserve_rng(device):
    import torch

    cpu, python = torch.get_rng_state(), random.getstate()
    backend = str(device).split(":")[0]
    module = torch.cuda if backend == "cuda" else (torch.neuron if backend == "neuron" else None)
    state = module.get_rng_state(device) if module else None
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        random.setstate(python)
        if module:
            module.set_rng_state(state, device)
        if numpy:
            numpy.random.set_state(numpy_state)


class Collector:
    def __init__(self, policy, repeats):
        if type(repeats) is not int or not 3 <= repeats <= 100:
            raise ValueError("Repeat consistency requires 3 to 100 declared repeats")
        self.policy, self.repeats, self.rows = policy, repeats, []

    @staticmethod
    def snapshot(model):
        cloned = copy.deepcopy(model)
        if cloned is model:
            raise ValueError("Model deepcopy returned the original object")
        original = [*model.parameters(), *model.buffers()]
        copied = [*cloned.parameters(), *cloned.buffers()]
        storage = {(str(value.device), value.untyped_storage().data_ptr()) for value in original if value.numel()}
        if any((str(value.device), value.untyped_storage().data_ptr()) in storage for value in copied if value.numel()):
            raise ValueError("Model snapshot retained original parameter/buffer storage")
        return cloned

    def observe(self, reference, candidate, inputs, *, seed, device, set_seed, tolerance, label):
        import torch

        from src.input_groups import copy_inputs
        from src.validation import clone_tensors, compare_outputs, synchronize

        from .measurement import consume_output, input_signature
        from .strict_validation import environment_guard

        row = {"label": label, "seed": seed, "input_signature": input_signature(inputs), "status": "not_measured",
               "snapshot_policy": "fresh_deepcopy_before_each_repeat; original_models_not_advanced",
               "candidate_calls": 0, "reference_calls": 0, "output_sha256": [], "issues": []}
        self.rows.append(row)
        changes, first = [], None
        try:
            with torch.no_grad(), preserve_rng(device), environment_guard(changes):
                template_ref, template_cand = self.snapshot(reference), self.snapshot(candidate)
        except Exception as exc:
            row["reason"] = f"State snapshot unavailable: {type(exc).__name__}"
            return
        try:
            with torch.no_grad(), preserve_rng(device):
                for _ in range(self.repeats):
                    with environment_guard(changes):
                        ref, cand = self.snapshot(template_ref), self.snapshot(template_cand)
                    set_seed(seed)
                    random.seed(seed)
                    if numpy := sys.modules.get("numpy"):
                        numpy.random.seed(seed % (1 << 32))
                    expected = clone_tensors(ref(*copy_inputs(inputs)))
                    synchronize(device)
                    row["reference_calls"] += 1
                    set_seed(seed)
                    random.seed(seed)
                    if numpy:
                        numpy.random.seed(seed % (1 << 32))
                    with environment_guard(changes):
                        actual = cand(*copy_inputs(inputs))
                        consumed = consume_output(actual)
                        synchronize(device)
                    row["candidate_calls"] += 1
                    actual = clone_tensors(actual)
                    if issue := compare_outputs(expected, actual, tolerance):
                        row["issues"].append(f"Native correctness on repeated output: {issue}")
                    if first is not None and (issue := repeat_issue(first, actual, self.policy)):
                        row["issues"].append(issue)
                    if first is None:
                        first = clone_tensors(actual)
                    row["output_sha256"].append(fingerprint(signature(actual)))
                    del actual, expected, consumed, ref, cand
                if changes:
                    row["issues"].append("Candidate changed process settings during repeated execution")
            row["status"] = "failed" if row["issues"] else "passed"
        except Exception as exc:
            row["status"] = "failed"
            row["issues"].append(f"Repeat execution failed: {type(exc).__name__}: {exc}")

    def document(self):
        status = "not_measured" if not self.rows or any(row["status"] == "not_measured" for row in self.rows) else "passed"
        if any(row["status"] == "failed" for row in self.rows):
            status = "failed"
        return {"protocol": PROTOCOL, "policy": self.policy, "repeats": self.repeats, "cases": self.rows,
                "check": check(status, reason="Controlled seeded repeats with state snapshots and unchanged native correctness gates",
                               values={"case_count": len(self.rows), "mode": self.policy["mode"]}),
                "race_freedom_proven": False, "timing_samples_collected": False}


@contextmanager
def audit(policy, repeats):
    collector = Collector(policy, repeats)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


def observe(*args, **kwargs):
    if collector := ACTIVE.get():
        collector.observe(*args, **kwargs)
