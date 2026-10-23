"""Independent full-input numerical probes; import-time SDK dependencies are absent."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

from .common import fingerprint
from .evidence import check
from .operator_specs import spec_for

PROTOCOL = "kernelbench_pro_native_numerical_audit_v1"
ACTIVE = ContextVar("kernelbench_numerical_auditor", default=None)


def _spec(payload):
    settings = payload["eval_config"]
    if type(settings.get("oracle_memory_mb")) is not int or not 1 <= settings["oracle_memory_mb"] <= 1048576:
        raise ValueError("Invalid independent oracle host-memory envelope")
    spec = spec_for(payload["level"], payload["problem_id"], payload["ref_arch_src"])
    if settings.get("oracle_spec_sha256") != fingerprint(spec):
        raise ValueError("Independent oracle specification is missing or changed")
    precision = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[settings["precision"]]
    from .input_calibration import validate_inline

    profile = validate_inline(payload)
    return spec, profile["precision_contract"] if profile is not None else spec["precision"][precision]


def _signal_valid(signal, absolute_tolerance):
    if (not isinstance(signal, dict) or not isinstance(signal.get("leaves"), list) or not signal["leaves"]
            or type(signal.get("informative")) is not bool):
        return False
    flags = []
    for leaf in signal["leaves"]:
        if not isinstance(leaf, dict) or not isinstance(leaf.get("path"), str):
            return False
        if leaf.get("kind") == "discrete_exact":
            flags.append(True)
        else:
            if (any(type(leaf.get(key)) is not int or leaf[key] < 0 for key in ("elements", "finite_elements"))
                    or leaf["finite_elements"] > leaf["elements"]
                    or type(leaf.get("fraction_above_atol")) not in (int, float)
                    or not math.isfinite(leaf["fraction_above_atol"]) or not 0 <= leaf["fraction_above_atol"] <= 1
                    or type(leaf.get("constant")) is not bool):
                return False
            if leaf["finite_elements"]:
                if any(type(leaf.get(key)) not in (int, float) or not math.isfinite(leaf[key]) or leaf[key] < 0
                       for key in ("max_abs", "rms")):
                    return False
                informative = (leaf["finite_elements"] == leaf["elements"] and leaf["max_abs"] > absolute_tolerance
                               and leaf["fraction_above_atol"] > 0)
            else:
                informative = False
            flags.append(informative)
        if type(leaf.get("informative")) is not bool or leaf["informative"] != flags[-1]:
            return False
    return signal["informative"] == all(flags)


def verdicts(document, cases, payload):
    spec, policy = _spec(payload)
    settings = payload["eval_config"]
    def missing(reason):
        return {name: check("not_measured", reason=reason) for name in ("precision_contract", "reference_signal")}
    if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
            or document.get("oracle_spec_sha256") != fingerprint(spec) or document.get("policy") != policy
            or document.get("oracle_memory_mb") != settings["oracle_memory_mb"]
            or not isinstance(cases, list) or not cases or any(not isinstance(row, dict) for row in cases)):
        return missing("No matching independent numerical audit and frozen memory/error policy")
    rows = document.get("cases", [])
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [(row.get("label"), row.get("seed")) for row in rows]
            != [(row.get("label"), row.get("seed")) for row in cases]
            or any(not isinstance(row.get("input_signature"), list) for row in rows)):
        return missing("Independent oracle omitted or changed native correctness cases/seeds")
    if any(row.get("status") not in {"passed", "failed", "not_measured"} for row in rows):
        return missing("Independent oracle execution has no recognized status")
    if any(row["status"] == "not_measured" for row in rows):
        return missing("Independent oracle exceeded admission limits or could not execute every actual input")
    if any(row["status"] == "failed" for row in rows):
        return {"precision_contract": check("failed", reason="Independent numerical comparison or its execution failed"),
                "reference_signal": check("not_measured", reason="Numerical qualification failed")}
    informative = False
    for row in rows:
        gate, signal = row.get("precision_gate", {}), row.get("reference_signal", {})
        if (not isinstance(gate, dict) or not _signal_valid(signal, policy["element_atol"])
                or type(row.get("estimated_host_bytes")) is not int
                or not 0 < row["estimated_host_bytes"] <= settings["oracle_memory_mb"] * 1024**2
                or row.get("errors") != [] or gate.get("passed") is not True
                or gate.get("reference_qualified") is not True or gate.get("reference_issues") != []
                or gate.get("issues") != [] or gate.get("all_elements_required") is not True
                or gate.get("oracle_dtype") != "float64/complex128" or not isinstance(gate.get("leaves"), list)):
            return missing("Incomplete or inconsistent independent numerical/signal evidence")
        floating_paths = {leaf["path"] for leaf in signal["leaves"] if leaf.get("kind") != "discrete_exact"}
        if (any(not isinstance(leaf, dict) or not isinstance(leaf.get("path"), str) for leaf in gate["leaves"])
                or {leaf["path"].removesuffix(".complex") for leaf in gate["leaves"]} != floating_paths):
            return missing("Numerical and signal leaves have inconsistent output coverage")
        for leaf in gate["leaves"]:
            if leaf.get("kind") in {"empty_tensor", "nonfinite_only"}:
                count = leaf.get("elements")
                valid_count = count == 0 if leaf["kind"] == "empty_tensor" else type(count) is int and count > 0
                matching_signal = next((item for item in signal["leaves"] if item["path"] == leaf["path"].removesuffix(".complex")), {})
                if (not valid_count or type(count) is not int or not isinstance(leaf.get("shape"), list)
                        or any(type(size) is not int or size < 0 for size in leaf["shape"])
                        or math.prod(leaf["shape"]) != count or not isinstance(leaf.get("dtype"), str)
                        or (leaf["kind"] == "nonfinite_only" and (matching_signal.get("finite_elements") != 0 or matching_signal.get("informative") is not False))
                        or leaf.get("oracle_rms") is not None or leaf.get("norm_limit") is not None):
                    return missing("Non-numeric leaf has no matching shape/signal and null measurement contract")
                for kind in ("reference", "candidate"):
                    if leaf.get(kind) != {"passed": True, "rms_error": None, "max_abs_error": None, "elements_failed": 0}:
                        return missing("Empty/non-finite-only output cannot invent numerical error measurements")
                continue
            if (not isinstance(leaf, dict) or type(leaf.get("elements")) is not int or leaf["elements"] <= 0
                    or type(leaf.get("oracle_rms")) not in (int, float) or not math.isfinite(leaf["oracle_rms"])
                    or leaf["oracle_rms"] < 0
                    or leaf.get("norm_limit") != policy["norm_atol"] + policy["norm_rtol"] * leaf["oracle_rms"]):
                return missing("Numerical norm budget differs from the frozen policy")
            for kind in ("reference", "candidate"):
                values = leaf.get(kind, {})
                if (not isinstance(values, dict) or values.get("passed") is not True
                        or type(values.get("elements_failed")) is not int or values["elements_failed"] != 0
                        or any(type(values.get(key)) not in (int, float) or not math.isfinite(values[key]) or values[key] < 0
                               for key in ("rms_error", "max_abs_error"))
                        or values["rms_error"] > leaf["norm_limit"]):
                    return missing("Numerical error counters contradict a passed summary")
        informative |= signal["informative"]
    return {"precision_contract": check("passed", reason="Independent CPU FP64 formulas qualify both reference and candidate on every actual native input",
                values={"case_count": len(rows), "oracle_spec_sha256": fingerprint(spec), "dimensions_changed": False}),
            "reference_signal": check("passed" if informative else "failed",
                reason="At least one actual input must contain informative reference output; no automatic input correction",
                values={"case_count": len(rows), "informative_case_count": sum(row["reference_signal"]["informative"] for row in rows)})}


class Collector:
    def __init__(self, payload):
        self.payload = payload
        self.spec, self.policy = _spec(payload)
        self.rows = []

    def observe(self, reference, candidate, inputs, *, seed, device, set_seed, tolerance, label):
        import torch

        from src.validation import clone_tensors, synchronize

        from .determinism import preserve_rng
        from .measurement import input_signature
        from .numerical import PrecisionContract, precision_gate, signal_audit
        from .operator_oracles import estimate_bytes, execute
        from .state_contract import _isolated_pair
        from .strict_validation import environment_guard

        row = {"label": label, "seed": seed, "input_signature": input_signature(inputs),
               "status": "not_measured", "errors": []}
        self.rows.append(row)
        changes = []
        try:
            with torch.no_grad(), preserve_rng(device), environment_guard(changes):
                estimate = estimate_bytes(self.payload["problem_id"], reference, inputs)
                row["estimated_host_bytes"] = estimate
                if estimate > self.payload["eval_config"]["oracle_memory_mb"] * 1024**2:
                    row["errors"].append("Full-size oracle exceeds the declared host-memory envelope; no resizing performed")
                    return
                # Compute truth before candidate activity and do not advance the
                # original models that the native paired protocol will execute.
                truth = execute(self.payload["problem_id"], reference, inputs)
                ref, cand = _isolated_pair(reference, candidate)
                set_seed(seed)
                expected = clone_tensors(ref(*clone_tensors(inputs)))
                synchronize(device)
                set_seed(seed)
                actual = clone_tensors(cand(*clone_tensors(inputs)))
                synchronize(device)
                # Preserve output storage dtypes when bringing measured values to
                # CPU. Only the independent arithmetic truth uses FP64.
                def host_output(value):
                    if isinstance(value, torch.Tensor):
                        return value.detach().cpu().clone()
                    if isinstance(value, dict):
                        return {key: host_output(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return type(value)(host_output(item) for item in value)
                    return value

                expected, actual = host_output(expected), host_output(actual)
                row["precision_gate"] = precision_gate(expected, actual, truth, PrecisionContract(**self.policy))
                row["reference_signal"] = signal_audit(expected, absolute_tolerance=self.policy["element_atol"])
                row["status"] = "passed" if row["precision_gate"]["passed"] else "failed"
                del expected, actual, truth, ref, cand
        except (MemoryError, torch.OutOfMemoryError) as exc:
            row["errors"].append(f"Independent oracle memory unavailable: {type(exc).__name__}")
        except Exception as exc:
            row["status"] = "failed"
            row["errors"].append(f"Independent oracle execution failed: {type(exc).__name__}: {str(exc)[:1000]}")
        finally:
            if changes:
                row["status"] = "failed"
                row["errors"].extend("Process setting changed: " + name for name in sorted(set(changes)))

    def document(self):
        return {"protocol": PROTOCOL, "oracle_spec_sha256": fingerprint(self.spec), "policy": self.policy,
                "oracle_memory_mb": self.payload["eval_config"]["oracle_memory_mb"], "cases": self.rows,
                "known_reference_issue": self.spec["known_reference_issue"],
                "dimensions_changed": False, "timing_samples_collected": False, "security_isolation_certified": False}


@contextmanager
def audit(payload):
    collector = Collector(payload)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


def observe(*args, **kwargs):
    if (collector := ACTIVE.get()) is not None:
        collector.observe(*args, **kwargs)
