"""Explicit evidence levels. Missing diagnostics never become passing checks."""

from __future__ import annotations

LEVELS = {
    "current": ("correctness",),
    "L0": ("correctness", "precision_contract", "reference_signal", "input_families",
           "input_sensitivity", "environment_integrity", "mode_state"),
    "L1": ("paired_timing", "noise_floor", "kernel_origin", "device_calibration", "bound_certificate"),
    "L2": ("hardware_counters", "memory_safety", "algorithm_certificate"),
}
STATES = {"passed", "failed", "not_measured", "not_applicable"}


def check(status, *, reason, values=None):
    if status not in STATES or not isinstance(reason, str) or not reason:
        raise ValueError("Checks require an explicit state and reason")
    return {"status": status, "reason": reason, "values": values}


def accepted(item):
    return (isinstance(item, dict) and item.get("status") in {"passed", "not_applicable"}
            and bool(item.get("reason")))


def receipt(checks, *, requested="current", scope="standard"):
    if requested not in LEVELS or scope not in {"standard", "hidden"}:
        raise ValueError("Unknown evidence level or scope")
    if any(not isinstance(item, dict) or item.get("status") not in STATES or not item.get("reason")
           for item in checks.values()):
        raise ValueError("Invalid evidence check")
    required, achieved = [], None
    missing = {}
    for level, additions in LEVELS.items():
        required.extend(additions)
        missing[level] = [name for name in required if not accepted(checks.get(name))]
        if not missing[level]:
            achieved = level
    all_checks = {name: checks.get(name, check("not_measured", reason="No evidence was collected"))
                  for additions in LEVELS.values() for name in additions}
    return {"protocol": "kernelbench-pro-evidence-v1", "scope": scope, "requested_level": requested,
            "achieved_level": achieved, "requested_level_met": not missing[requested],
            "missing_required_checks": missing[requested], "checks": {**all_checks, **checks},
            "security_isolation_certified": False}


def from_result(result, settings, *, scope="standard", additional=None):
    checks = {"correctness": check("passed" if result.get("correctness") is True else "failed",
                                   reason="Native correctness verdict for this phase")}
    checks.update(additional or {})
    return receipt(checks, requested=settings.get("evidence_level", "current"), scope=scope)


def score_issue(result, settings, scope):
    required = settings.get("strict_validation") or settings.get("evidence_level", "current") != "current"
    if not required:
        return None
    record = result.get("metadata", {}).get("evidence_receipt", {})
    requested = settings.get("evidence_level", "current")
    try:
        expected = receipt(record["checks"], requested=requested, scope=scope)
    except (KeyError, TypeError, ValueError):
        return "missing_or_invalid_evidence_receipt"
    if (record.get("protocol") != expected["protocol"] or record.get("scope") != scope
            or record.get("requested_level") != requested or record.get("requested_level_met") is not True
            or record.get("achieved_level") != expected["achieved_level"]
            or not expected["requested_level_met"]):
        return "requested_evidence_level_not_met"
    if settings.get("strict_validation"):
        if any(not accepted(record["checks"].get(name)) for name in LEVELS["L0"]):
            return "strict_validation_evidence_incomplete"
    return None
