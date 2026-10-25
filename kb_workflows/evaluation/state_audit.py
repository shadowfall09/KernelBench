"""Scoped native-case state auditing and receipt replay; no SDK imports at import."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar

from .evidence import check

PROTOCOL = "kernelbench_pro_native_state_audit_v1"
STATE_PROTOCOL = "kernelbench_pro_state_contract_v1"
MODE_SEQUENCE = (True, True, False, True, False)
ACTIVE = ContextVar("kernelbench_state_auditor", default=None)


def _check_valid(record):
    """Recompute structural coverage rather than trusting a passed summary."""
    if not isinstance(record, dict) or record.get("status") not in {"passed", "failed"}:
        return False
    values = record.get("values", {})
    if (not isinstance(values, dict) or values.get("protocol") != STATE_PROTOCOL
            or type(values.get("parameter_count")) is not int or values["parameter_count"] < 0
            or not isinstance(values.get("issues"), list)):
        return False
    if record["status"] == "failed":
        return bool(values["issues"])
    if values["issues"] != []:
        return False
    count = values["parameter_count"]
    phases = ["modes"] + (["parameters/training=True", "parameters/training=False"] if count else [])
    bindings = values.get("bindings", [])
    if (not isinstance(bindings, list) or any(not isinstance(row, dict) for row in bindings)
            or [row.get("phase") for row in bindings] != phases):
        return False
    if len({binding.get("sha256") for binding in bindings}) != 1:
        return False
    for binding in bindings:
        states = binding.get("states")
        if not isinstance(states, list) or any(not isinstance(row, dict) for row in states):
            return False
        if (binding.get("sha256") != hashlib.sha256(json.dumps(states, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                or any(not isinstance(row.get("reference"), str) or row.get("kind") not in {"parameter", "buffer", "tensor"}
                       or not isinstance(row.get("target"), str) or not isinstance(row.get("shape"), list)
                       or any(type(size) is not int or size < 0 for size in row["shape"])
                       or type(row.get("alias_group")) is not int or row["alias_group"] < 0
                       or not isinstance(row.get("reference_owner"), str) or not isinstance(row.get("target_owner"), str)
                       or not isinstance(row.get("dtype"), str) for row in states)
                or len({row["reference"] for row in states}) != len(states)):
            return False
        if any(row["kind"] == "tensor" and (row.get("replacement") != "reference_matched"
                or row.get("target_kind") not in {"tensor", "buffer"}) for row in states):
            return False
        groups = {row["alias_group"]: row for row in states if row["kind"] == "parameter"}
        if count != len(groups):
            return False
        updated_count = sum(math.prod(row["shape"]) > 0 for row in groups.values())
    mode_pairs = list(dict.fromkeys([("", ""), *((state["reference_owner"], state["target_owner"]) for state in bindings[0]["states"])]))
    events = [(f"mode={mode}/call={index}", mode) for index, mode in enumerate(MODE_SEQUENCE)]
    if count:
        for mode in (True, False):
            events.append((f"parameter warmup/training={mode}", mode))
            events.extend((f"after parameter update {step}/training={mode}", mode) for step in range(2))
    rows = values.get("events", [])
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [(row.get("case"), row.get("training")) for row in rows] != events
            or any(type(row.get("seed")) is not int for row in rows)):
        return False
    for row in rows:
        modes = row.get("module_modes_before")
        if (not isinstance(modes, list) or not modes or row.get("module_modes_after") != modes or row.get("no_grad_after") is not True
                or any(not isinstance(mode, dict) for mode in modes)
                or modes[0] != {"reference": "", "target": "", "expected": row["training"], "actual": row["training"]}
                or [(mode.get("reference"), mode.get("target")) for mode in modes] != mode_pairs
                or any(not isinstance(mode, dict) or not isinstance(mode.get("reference"), str)
                       or not isinstance(mode.get("target"), str) or type(mode.get("expected")) is not bool
                       or mode.get("actual") is not mode["expected"] for mode in modes)
                or len({(mode["reference"], mode["target"]) for mode in modes}) != len(modes)):
            return False
        if row["case"].startswith("after parameter update"):
            if (type(row.get("parameters_updated")) is not int or row["parameters_updated"] != updated_count
                    or type(row.get("reference_output_changed")) is not bool):
                return False
    return True


def verdict(document, cases, *, expected_tolerance=None, expected_policy=None):
    if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
            or not isinstance(cases, list) or not cases or any(not isinstance(row, dict) for row in cases)
            or type(document.get("tolerance")) not in (int, float)
            or not math.isfinite(document["tolerance"]) or document["tolerance"] < 0):
        return check("not_measured", reason="Missing native state audit or numerical comparison policy")
    if expected_tolerance is not None and document["tolerance"] != expected_tolerance:
        return check("not_measured", reason="Native state audit changed the frozen comparison tolerance")
    if document.get("policy") != expected_policy:
        return check("not_measured", reason="Native state audit changed its source-bound task policy")
    rows = document.get("cases", [])
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [(row.get("label"), row.get("seed")) for row in rows]
            != [(row.get("label"), row.get("seed")) for row in cases]
            or any(not isinstance(row.get("input_signature"), (list, dict)) for row in rows)):
        return check("not_measured", reason="State audit omitted or changed native correctness cases/seeds")
    if any(not _check_valid(row.get("check")) for row in rows):
        return check("not_measured", reason="State audit lacks complete bindings, transitions or parameter probes")
    for row in rows:
        expected_ignored = ([{"reference": name, **rule} for name, rule in expected_policy["ordinary"].items()
                             if rule["role"] == "unused_initialization"] if expected_policy else [])
        for binding in row["check"]["values"]["bindings"]:
            if binding.get("ignored_ordinary") != expected_ignored:
                return check("not_measured", reason="State audit changed ignored reference initialization roles")
            if expected_policy is not None:
                required = {name for name, rule in expected_policy["ordinary"].items() if rule["role"] == "persistent"}
                if {state["reference"] for state in binding["states"] if state["kind"] == "tensor"} != required:
                    return check("not_measured", reason="State audit omitted or added required ordinary reference state")
        if any(event.get("seed") != row["seed"] for event in row["check"]["values"].get("events", [])):
            return check("not_measured", reason="State audit changed a native case RNG seed")
    status = "failed" if any(row["check"]["status"] == "failed" for row in rows) else "passed"
    return check(status, reason="Every native case audits mapped registered state, persistent mode transitions and parameter updates",
                 values={"protocol": PROTOCOL, "case_count": len(rows), "timing_samples_collected": False,
                         "representation_scope": "registered_views_and_source_bound_ordinary_tensors",
                         "security_isolation_certified": False})


class Collector:
    def __init__(self, payload=None):
        from .state_policies import validate_inline

        self.policy = validate_inline(payload) if payload is not None else None
        self.rows, self.tolerance = [], None

    def observe(self, reference, candidate, inputs, *, seed, device, set_seed, tolerance, label):
        from .determinism import preserve_rng
        from .measurement import input_signature
        from .state_contract import state_checks
        from .strict_validation import environment_guard

        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("State comparison tolerance must be finite and nonnegative")
        if self.tolerance is not None and self.tolerance != tolerance:
            raise ValueError("Native state audit cannot change comparison policy between cases")
        self.tolerance = tolerance
        changes = []
        with preserve_rng(device), environment_guard(changes):
            record = state_checks(reference, candidate, inputs, seed=seed, set_seed=set_seed,
                                  tolerance=tolerance, device=device, policy=self.policy)
        if changes:
            record["status"] = "failed"
            record["values"]["issues"].extend("Candidate changed process setting " + name for name in sorted(set(changes)))
        self.rows.append({"label": label, "seed": seed, "input_signature": input_signature(inputs), "check": record})

    def document(self):
        document = {"protocol": PROTOCOL, "cases": self.rows, "tolerance": self.tolerance,
                    "policy": self.policy,
                    "timing_samples_collected": False, "security_isolation_certified": False}
        document["check"] = verdict(document, self.rows, expected_policy=self.policy)
        return document


@contextmanager
def audit(payload=None):
    collector = Collector(payload)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


def observe(*args, **kwargs):
    if (collector := ACTIVE.get()) is not None:
        collector.observe(*args, **kwargs)
