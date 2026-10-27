"""Explicit evidence levels. Missing diagnostics never become passing checks."""

from __future__ import annotations

LEVELS = {
    "current": ("correctness",),
    "L0": ("correctness", "precision_contract", "reference_signal", "input_families",
           "input_sensitivity", "environment_integrity", "mode_state"),
    "L1": ("paired_timing", "noise_floor", "kernel_origin", "device_calibration", "bound_certificate", "launch_attribution"),
    "L2": ("hardware_counters", "memory_safety", "repeat_consistency", "algorithm_certificate", "exclusive_device"),
}
STATES = {"passed", "failed", "not_measured", "not_applicable"}
MEASUREMENT_CHECKS = {"paired_timing", "noise_floor", "device_calibration", "bound_certificate", "launch_attribution"}


def check(status, *, reason, values=None):
    if status not in STATES or not isinstance(reason, str) or not reason:
        raise ValueError("Checks require an explicit state and reason")
    return {"status": status, "reason": reason, "values": values}


def accepted(item):
    return (isinstance(item, dict) and item.get("status") in {"passed", "not_applicable"}
            and bool(item.get("reason")))


def origin_feedback(result, settings):
    """Compact trusted-worker feedback for selection; final scoring recomputes it."""
    if settings.get("evidence_level", "current") not in {"L1", "L2"}:
        return None
    track = settings.get("kernel_track", "custom")
    fraction = settings.get("origin_minimum_custom_fraction", 0.95)
    try:
        item = result["metadata"]["evidence_receipt"]["checks"]["kernel_origin"]
        values = item.get("values") or {}
        if (item.get("status") not in STATES or not isinstance(item.get("reason"), str) or not item["reason"]
                or item["status"] == "passed" and (values.get("track") != track
                    or values.get("minimum_custom_fraction") != fraction)):
            raise ValueError("Missing or mismatched development origin policy")
        return {"status": item["status"], "reason": item["reason"][:500], "track": track,
                "minimum_custom_fraction": fraction}
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"status": "not_measured", "reason": "Missing or mismatched development origin feedback",
                "track": track, "minimum_custom_fraction": fraction}


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
    if settings.get("evidence_level") == "L2" and "native_call_receipt" in result.get("metadata", {}):
        from .tool_runner import exclusive_verdict

        checks["exclusive_device"] = exclusive_verdict(result)
    return receipt(checks, requested=settings.get("evidence_level", "current"), scope=scope)


def measurement_checks(config, pid, sid, baseline, *, variant=None, paired=None):
    """Assemble public timing evidence from independently replayed artifacts."""
    if config.evidence_level not in {"L1", "L2"}:
        return {}
    from pathlib import Path

    from ..analysis.calibration import validate
    from ..analysis.headroom import certificate
    from ..common import evaluator_digest, fingerprint, read_json
    from .paired_task import validate_for_score

    checks = {name: check("not_measured", reason="Verified public measurement evidence is missing") for name in MEASUREMENT_CHECKS}
    if config.evidence_level == "L2":
        from ..provenance.counter_task import score_checks

        checks.update(score_checks(config, pid, sid))
    if not config.paired_timing:
        return checks
    if paired is None:
        if variant is None:
            from ..engine import paired_variant

            try:
                variant = paired_variant(config, pid)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                checks.update({name: check("not_measured", reason=str(exc)) for name in MEASUREMENT_CHECKS})
                return checks
        selected = baseline.get("variants", {}).get(variant, baseline)
        issue, paired = validate_for_score(config, pid, sid, variant, selected)
        if issue:
            checks.update({name: check("not_measured", reason=issue) for name in MEASUREMENT_CHECKS})
            return checks
    bindings = paired.get("measurement_bindings", [])
    if not bindings:
        return checks
    sources = [row["result_sha256"] for row in bindings]
    checks["paired_timing"] = check("passed", reason="Current public paired artifacts replayed with verified source and timing mode",
                                    values={"result_sha256": sources, "pairs_sha256": fingerprint({"pairs": paired["pairs"]})})
    checks["noise_floor"] = check("passed", reason="Public reference/control noise estimated; gain must separately clear this floor",
                                  values={"reference_control_ci95": paired["reference_control_ci95"], "result_sha256": sources})
    try:
        calibration = read_json(Path(config.device_calibration)) if config.device_calibration else {}
        for row in bindings:
            validate(calibration, row["device_fingerprint"], digest=evaluator_digest())
        checks["device_calibration"] = check("passed", reason="Calibration matches every actual paired timing device",
            values={"calibration_sha256": calibration["receipt_sha256"], "result_sha256": sources})
        specs = read_json(Path(config.problem_specs)) if config.problem_specs else {}
        model = specs.get(f"level{config.level}", {}).get(str(pid), {}).get("headroom_certificate", {})
        for row in bindings:
            certificate(model, calibration, row, source_sha256=row["reference_source_sha256"], digest=evaluator_digest())
        checks["bound_certificate"] = check("passed", reason="Reviewed bound matches every actual paired timing geometry, precision and calibration",
            values={"certificate_sha256": fingerprint(model), "result_sha256": sources})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        missing = "bound_certificate" if checks["device_calibration"]["status"] == "passed" else "device_calibration"
        checks[missing] = check("not_measured", reason=str(exc)[:500] or "Measurement evidence is invalid")
    if checks["device_calibration"]["status"] == "passed":
        from ..analysis.performance import paired_attribution

        try:
            attribution = paired_attribution(paired, calibration)
            status = "failed" if attribution["launch_only"] else (
                "not_measured" if attribution["mixed_repeat_classification"] else "passed")
            checks["launch_attribution"] = check(status, reason="Actual paired device activity and calibrated launch-cost classification", values=attribution)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            checks["launch_attribution"] = check("not_measured", reason=str(exc)[:500] or "Launch attribution is invalid")
    return checks


def score_issue(result, settings, scope, *, additional=None):
    required = settings.get("strict_validation") or settings.get("evidence_level", "current") != "current"
    if not required:
        return None
    record = result.get("metadata", {}).get("evidence_receipt", {})
    requested = settings.get("evidence_level", "current")
    allowed = MEASUREMENT_CHECKS
    if requested == "L2":
        from ..provenance.counter_task import SCORE_CHECKS

        allowed = allowed | SCORE_CHECKS
    try:
        original = receipt(record["checks"], requested=requested, scope=scope)
        if additional is not None and set(additional) - allowed:
            return "invalid_external_measurement_checks"
        expected = receipt({**record["checks"], **(additional or {})}, requested=requested, scope=scope)
    except (KeyError, TypeError, ValueError):
        return "missing_or_invalid_evidence_receipt"
    if (record.get("protocol") != expected["protocol"] or record.get("scope") != scope
            or record.get("requested_level") != requested
            or record.get("requested_level_met") != original["requested_level_met"]
            or record.get("achieved_level") != original["achieved_level"]
            or not expected["requested_level_met"]):
        return "requested_evidence_level_not_met"
    if settings.get("strict_validation"):
        if any(not accepted(record["checks"].get(name)) for name in LEVELS["L0"]):
            return "strict_validation_evidence_incomplete"
    if requested in {"L1", "L2"}:
        from ..provenance.origin_task import origin_check, validate_scored

        try:
            request, _ = validate_scored(result)
        except (ValueError, OSError):
            return "missing_or_invalid_scored_call_receipt"
        if (request["eval_config"].get("evidence_level") != requested
                or bool(request["eval_config"].get("use_hidden_tests")) != (scope == "hidden")):
            return "scored_call_scope_or_level_mismatch"
        if (request["eval_config"].get("kernel_track", "custom") != settings.get("kernel_track", "custom")
                or request["eval_config"].get("origin_minimum_custom_fraction", 0.95)
                != settings.get("origin_minimum_custom_fraction", 0.95)):
            return "scored_execution_track_mismatch"
        origin = origin_check(result)
        if not accepted(origin) or record["checks"].get("kernel_origin") != origin:
            return "actual_scored_origin_missing_or_invalid"
        if set(additional or {}) != allowed:
            return "validated_measurement_evidence_missing"
        if requested == "L2":
            try:
                from pathlib import Path

                from ..common import fingerprint, read_json
                from ..config import RunConfig
                from ..provenance.counter_task import score_checks

                values = additional["hardware_counters"]["values"]
                directory = Path(values["run_dir"])
                config = RunConfig(**read_json(directory / "run_config.json"))
                pid, sid = request["problem_id"], request.get("sample_id", 0)
                current = score_checks(config, pid, sid)
                if (config.run_dir.resolve() != directory.resolve() or config.level != request["level"]
                        or config.evidence_level != "L2" or values.get("simulated") is not False
                        or fingerprint(result) not in values["actual_result_sha256"]
                        or any(additional[name] != item or not accepted(item) for name, item in current.items())):
                    return "counter_evidence_missing_or_invalid"
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return "counter_evidence_missing_or_invalid"
            try:
                from .tool_checks import TOOLS, select_tools, validate

                native = result["metadata"]["native_call_receipt"]["native_request"]
                policy = native["eval_config"]
                if set(select_tools(policy.get("sanitizer_tools", "none"))) != set(TOOLS) or not policy.get("determinism_checks"):
                    return "l2_safety_and_repeat_policy_missing"
                tools = validate(result, native)
                if any(not accepted(tools.get(name)) or record["checks"].get(name) != tools[name]
                       for name in ("memory_safety", "repeat_consistency")):
                    return "l2_safety_and_repeat_evidence_invalid"
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return "l2_safety_and_repeat_evidence_invalid"
    return None
