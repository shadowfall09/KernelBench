"""Source-bound sanitizer/repeat receipts; independent of vendor SDK imports."""

from __future__ import annotations

import copy
import hashlib
import re

from .common import evaluator_digest, fingerprint
from .determinism import PROTOCOL as REPEAT_PROTOCOL
from .evidence import check, from_result

PROTOCOL = "kernelbench_pro_tool_checks_v1"
TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
ERROR_EXIT = 86
MAX_LOG_BYTES = 16 * 1024**2
DIAGNOSTIC_FIELDS = ("sanitizer_tools", "sanitizer_command", "sanitizer_timeout", "determinism_checks", "determinism_repeats",
                     "state_validation", "numerical_validation", "allocation_validation", "oracle_memory_mb", "input_family_memory_mb",
                     "throughput_validation", "throughput_calls", "throughput_repeats", "throughput_memory_mb",
                     "cache_validation", "cache_memory_mb")


def diagnostic_options(config):
    """One policy surface for payloads, frozen repeats, studies and comparisons."""
    return {name: getattr(config, name) for name in DIAGNOSTIC_FIELDS}


def select_tools(value):
    if not isinstance(value, str):
        raise ValueError("Sanitizer tools must be none, all or a comma-separated selection")
    names = [part.strip() for part in value.split(",")]
    if names == ["none"]:
        return ()
    if names == ["all"]:
        return TOOLS
    if not names or any(name not in TOOLS for name in names) or len(set(names)) != len(names):
        raise ValueError("Invalid or duplicated sanitizer tools")
    return tuple(tool for tool in TOOLS if tool in names)


def enabled(payload):
    from .state_policies import validate_inline as validate_state

    validate_state(payload)
    from .allocation_audit import validate_inline

    validate_inline(payload)
    from .throughput_audit import validate_inline as throughput_enabled

    throughput = throughput_enabled(payload)
    from .cache_audit import validate_inline as cache_enabled

    cache = cache_enabled(payload)
    settings = payload["eval_config"]
    return payload.get("task", "evaluate") in {"evaluate", "paired_timing"} and bool(
        select_tools(settings.get("sanitizer_tools", "none")) or settings.get("determinism_checks")
        or settings.get("state_validation") or settings.get("numerical_validation") or settings.get("allocation_validation") or throughput or cache)


def child_payload(payload, nonce, stage):
    result = copy.deepcopy(payload)
    result["task"] = "evaluate"
    result["eval_config"].update(measure_performance=False, diagnostic_nonce=nonce, diagnostic_stage=stage)
    result.pop("request_id", None)
    result["request_id"] = "kb-" + fingerprint(result)
    return result


def parse_log(tool, text):
    if tool not in TOOLS or not isinstance(text, str):
        raise ValueError("Unknown sanitizer tool or non-text log")
    if tool == "racecheck":
        matches = re.findall(r"^=+\s*RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*\((\d+) errors?,\s*(\d+) warnings?\)\s*$", text, re.MULTILINE)
        # Zero-hazard runs use a different documented completion line.
        zero = re.findall(r"^=+\s*RACECHECK SUMMARY:\s*0 hazards? displayed\s*$", text, re.MULTILINE)
        counts = [max(int(hazards), int(a) + int(b)) for hazards, a, b in matches] + [0] * len(zero)
    else:
        counts = list(map(int, re.findall(r"^=+\s*ERROR SUMMARY:\s*(\d+) errors?(?: \([^\n]*\))?\s*$", text, re.MULTILINE)))
    if "COMPUTE-SANITIZER" not in text or not counts:
        return {"status": "not_measured", "reason": "No recognized tool completion summary", "error_count": None}
    errors = sum(counts)
    return {"status": "failed" if errors else "passed", "error_count": errors,
            "reason": "Tool reported errors or warnings" if errors else "Recognized completed tool report with no reported errors/warnings"}


def coverage(result, *, paired=False):
    if not isinstance(result, dict) or result.get("compiled") is not True or result.get("correctness") is not True:
        return None
    if paired:
        timing = result.get("metadata", {}).get("paired_timing", {})
        rows = timing.get("pairs", [])
        if (timing.get("protocol") != "balanced_paired_wall_v1" or not isinstance(rows, list) or len(rows) < 10
                or len(rows) != timing.get("num_pairs")
                or not all(timing.get(key) is True for key in ("all_outputs_verified", "output_consumed", "varied_inputs"))
                or any(not isinstance(row, dict) or type(row.get("seed")) is not int
                       or row.get("order") not in {"RCB", "CRB", "BCR", "BRC", "CBR", "RBC"} for row in rows)):
            return None
        return {"candidate_invocations": len(rows), "correctness_cases": len(rows),
                "scope": "original_uninstrumented_verified_paired_calls; diagnostics_use_separate_correctness_workers"}
    rows = result.get("metadata", {}).get("reliability_checks", [])
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or row.get("passed") is not True
                   or type(row.get("seed")) is not int or not isinstance(row.get("label"), str)
                   or row.get("orders") != ["candidate_first", "reference_first"] for row in rows)):
        return None
    return {"candidate_invocations": 2 * len(rows), "correctness_cases": len(rows),
            "scope": "verifier_observed_calls_to_the_frozen_candidate; kernel_authorship_requires_separate_origin_evidence"}


def tool_verdict(tool, record, expected_payload, device):
    expected = fingerprint(expected_payload)
    child = record.get("result", {})
    metadata = child.get("metadata", {})
    logs = record.get("logs", {})
    if record.get("payload_sha256") != expected or record.get("error"):
        return check("not_measured", reason="Sanitizer target identity or complete log collection is unavailable")
    if not isinstance(logs, dict):
        return check("not_measured", reason="Missing sanitizer log bundle")
    parsed = []
    for name, item in logs.items():
        if (not re.fullmatch(r"[a-z]+\.[0-9]+\.log", name) or not name.startswith(tool + ".")
                or not isinstance(item, dict) or not isinstance(item.get("text"), str)
                or len(item["text"].encode()) > MAX_LOG_BYTES
                or hashlib.sha256(item["text"].encode()).hexdigest() != item.get("sha256")):
            return check("not_measured", reason="Sanitizer raw log identity is missing or changed")
        parsed.append(parse_log(tool, item["text"]))
    failures = [row for row in parsed if row["status"] == "failed"]
    if failures:
        return check("failed", reason="Sanitizer reported errors or warnings", values={"reported_error_count": sum(row["error_count"] for row in failures)})
    if (record.get("payload_sha256") != expected or record.get("timed_out") is not False or record.get("exit_code") != 0
            or not parsed or any(row["status"] != "passed" for row in parsed) or not coverage(child)
            or metadata.get("device_fingerprint") != device
            or metadata.get("reference_sha256") != fingerprint({"source": expected_payload["ref_arch_src"]})
            or metadata.get("kernel_sha256") != fingerprint({"source": expected_payload["kernel_src"]})
            or record.get("coverage") != coverage(child)):
        return check("not_measured", reason="Incomplete, timed-out or mismatched sanitized target execution")
    return check("passed", reason="All retained process logs completed cleanly and the frozen candidate was exercised",
                 values={"logs": len(parsed), **coverage(child), "memory_race_freedom_proven": False})


def repeat_verdict(document, policy, repeats, cases):
    if (not isinstance(document, dict) or document.get("protocol") != REPEAT_PROTOCOL
            or document.get("policy") != policy or document.get("repeats") != repeats):
        return check("not_measured", reason="Repeat audit has no matching task policy or repeat budget")
    rows = document.get("cases", [])
    if (not isinstance(rows, list) or not rows or not cases
            or [(row.get("label"), row.get("seed")) for row in rows]
            != [(row.get("label"), row.get("seed")) for row in cases]):
        return check("not_measured", reason="Repeat audit does not cover every native correctness case")
    if any(row.get("status") == "failed" for row in rows):
        return check("failed", reason="Candidate violated repeated native correctness or declared consistency policy", values={"case_count": len(rows)})
    if (any(row.get("status") != "passed" or row.get("candidate_calls") != repeats or row.get("reference_calls") != repeats
            or len(row.get("output_sha256", [])) != repeats or row.get("issues") != [] for row in rows)
            or any(not re.fullmatch("[0-9a-f]{64}", sha) for row in rows for sha in row["output_sha256"])):
        return check("not_measured", reason="Repeat audit did not execute all declared snapshots")
    if policy["mode"] == "bitwise" and any(len(set(row["output_sha256"])) != 1 for row in rows):
        return check("failed", reason="Output bit fingerprints differ across exact-policy repeats")
    return check("passed", reason="All declared repeated snapshots satisfy native correctness and the reviewed consistency policy",
                 values={"case_count": len(rows), "mode": policy["mode"], "race_freedom_proven": False})


def derive(document, payload):
    settings = payload["eval_config"]
    tools = select_tools(settings.get("sanitizer_tools", "none"))
    base = document["native_result"]
    device = base.get("metadata", {}).get("device_fingerprint", {})
    identity = device.get("available") is True and bool(device.get("uuid")) and bool(device.get("driver_version"))
    checks = {}
    for tool in tools:
        verdict = tool_verdict(tool, document.get("tools", {}).get(tool, {}),
            child_payload(payload, document["nonce"], tool), device)
        if (not identity or document.get("binary", {}).get("status") != "measured") and verdict["status"] == "passed":
            verdict = check("not_measured", reason="Tool run lacks physical-device/software identity")
        checks[f"sanitizer_{tool}"] = verdict
    control = document.get("control", {})
    expected_control = child_payload(payload, document["nonce"], "control")
    child = control.get("result", {})
    control_ok = (identity and not control.get("error") and control.get("exit_code") == 0 and control.get("timed_out") is False
        and control.get("payload_sha256") == fingerprint(expected_control) and coverage(child)
        and child.get("metadata", {}).get("device_fingerprint") == device
        and child.get("metadata", {}).get("reference_sha256") == fingerprint({"source": payload["ref_arch_src"]})
        and child.get("metadata", {}).get("kernel_sha256") == fingerprint({"source": payload["kernel_src"]}))
    def cases(result):
        return [(row["label"], row["seed"]) for row in result.get("metadata", {}).get("reliability_checks", [])]

    if control_ok and payload.get("task", "evaluate") == "evaluate" and cases(base) != cases(child):
        control_ok = False
    if settings.get("allocation_validation"):
        from .allocation_audit import STAGES, verdict

        for pattern, stage in STAGES.items():
            record = document.get("allocation_controls", {}).get(pattern, {})
            expected = child_payload(payload, document["nonce"], stage)
            poisoned = record.get("result", {})
            metadata = poisoned.get("metadata", {})
            valid = (identity and record.get("exit_code") == 0 and record.get("timed_out") is False and not record.get("error")
                     and record.get("payload_sha256") == fingerprint(expected) and poisoned.get("compiled") is True
                     and metadata.get("device_fingerprint") == device
                     and metadata.get("reference_sha256") == fingerprint({"source": payload["ref_arch_src"]})
                     and metadata.get("kernel_sha256") == fingerprint({"source": payload["kernel_src"]}))
            if poisoned.get("correctness") is True:
                valid = valid and bool(coverage(poisoned)) and record.get("coverage") == coverage(poisoned)
                if payload.get("task", "evaluate") == "evaluate" and cases(base) != cases(poisoned):
                    valid = False
                if not control_ok or cases(child) != cases(poisoned):
                    valid = False
            value = verdict(metadata.get("allocation_audit"), expected,
                metadata.get("reliability_checks"), pattern=pattern) if valid else check(
                "not_measured", reason="No complete source/device-bound fresh allocation-control worker")
            if value["status"] == "passed" and poisoned.get("correctness") is not True:
                value = check("not_measured", reason="Poisoned worker did not complete native correctness")
            checks[stage] = value
    selected = []
    for tool in tools:
        name = f"sanitizer_{tool}"
        record = document["tools"][tool]
        if checks[name]["status"] == "passed" and control_ok and cases(record["result"]) != cases(child):
            checks[name] = check("not_measured", reason="Sanitized execution omitted or changed native correctness cases/seeds")
        selected.append(checks[name])
    if tools:
        status = "failed" if any(row["status"] == "failed" for row in selected) else "not_measured"
        if tools == TOOLS and control_ok and all(row["status"] == "passed" for row in selected):
            status = "passed"
        checks["memory_safety"] = check(status, reason="Complete four-tool tested-process suite; partial or missing tools do not certify full coverage",
            values={"requested_tools": list(tools), "scope": "tested_worker_and_children; tool_limitations_apply", "race_freedom_proven": False})
    checks["post_tool_correctness"] = check("passed" if control_ok else "not_measured",
        reason="Fresh uninstrumented correctness run after diagnostics; its times never replace scored samples")
    if settings.get("determinism_checks"):
        checks["repeat_consistency"] = repeat_verdict(child.get("metadata", {}).get("repeat_audit"),
            settings["determinism_policy"], settings["determinism_repeats"],
            child.get("metadata", {}).get("reliability_checks")) if control_ok else check("not_measured", reason="No complete direct repeat-control run")
    if settings.get("state_validation"):
        from .state_audit import verdict

        state_document = child.get("metadata", {}).get("state_audit")
        tolerance = settings["fp32_tolerance"] if settings.get("precision") == "fp32" or settings.get("device_backend") == "neuron" else 1e-2
        checks["mode_state"] = verdict(state_document, child.get("metadata", {}).get("reliability_checks"),
            expected_tolerance=tolerance, expected_policy=settings.get("state_policy")) if control_ok else check(
            "not_measured", reason="No complete native state-control run")
    if settings.get("numerical_validation"):
        from .numerical_audit import verdicts

        if control_ok:
            checks.update(verdicts(child.get("metadata", {}).get("numerical_audit"),
                                   child.get("metadata", {}).get("reliability_checks"), payload))
        else:
            checks.update({name: check("not_measured", reason="No complete independent numerical-control run")
                           for name in ("precision_contract", "reference_signal")})
    from .throughput_audit import STAGES as THROUGHPUT_STAGES
    from .throughput_audit import validate_inline as throughput_enabled
    from .throughput_audit import verdict as throughput_verdict

    if throughput_enabled(payload):
        requests = {mode: child_payload(payload, document["nonce"], stage) for mode, stage in THROUGHPUT_STAGES.items()}
        documents, native_cases, valid = {}, {}, bool(identity and control_ok)
        for mode, expected in requests.items():
            record = document.get("throughput_controls", {}).get(mode, {})
            result = record.get("result", {})
            metadata = result.get("metadata", {})
            valid = valid and (record.get("exit_code") == 0 and record.get("timed_out") is False and not record.get("error")
                and record.get("payload_sha256") == fingerprint(expected) and result.get("compiled") is True
                and metadata.get("device_fingerprint") == device
                and metadata.get("reference_sha256") == fingerprint({"source": payload["ref_arch_src"]})
                and metadata.get("kernel_sha256") == fingerprint({"source": payload["kernel_src"]}))
            documents[mode], native_cases[mode] = metadata.get("throughput_audit"), metadata.get("reliability_checks")
            if (isinstance(documents[mode], dict) and isinstance(documents[mode].get("case"), dict)
                    and documents[mode]["case"].get("status") == "passed"):
                valid = valid and (result.get("correctness") is True and bool(coverage(result))
                    and record.get("coverage") == coverage(result) and cases(child) == cases(result))
        checks["throughput_contrast"] = throughput_verdict(documents, requests, native_cases,
            allow_simulated=bool(document.get("simulated_execution"))) if valid else check(
            "not_measured", reason="No complete source/device-bound fresh throughput workers and direct control")
    from .cache_audit import STAGE as CACHE_STAGE
    from .cache_audit import validate_inline as cache_enabled
    from .cache_audit import verdict as cache_verdict

    if cache_enabled(payload):
        record = document.get("input_cache_control", {})
        expected = child_payload(payload, document["nonce"], CACHE_STAGE)
        result = record.get("result", {})
        metadata = result.get("metadata", {})
        valid = (identity and control_ok and record.get("exit_code") == 0 and record.get("timed_out") is False
            and not record.get("error") and record.get("payload_sha256") == fingerprint(expected)
            and result.get("compiled") is True and metadata.get("device_fingerprint") == device
            and metadata.get("reference_sha256") == fingerprint({"source": payload["ref_arch_src"]})
            and metadata.get("kernel_sha256") == fingerprint({"source": payload["kernel_src"]}))
        cache_doc = metadata.get("input_cache_audit")
        if isinstance(cache_doc, dict) and cache_doc.get("cases") and all(row.get("status") == "passed" for row in cache_doc["cases"]):
            valid = valid and (result.get("correctness") is True and bool(coverage(result))
                and record.get("coverage") == coverage(result) and cases(child) == cases(result))
        checks["same_object_inputs"] = cache_verdict(cache_doc, expected, metadata.get("reliability_checks"),
            allow_simulated=bool(document.get("simulated_execution"))) if valid else check(
            "not_measured", reason="No complete source/device-bound fresh input-cache worker and direct control")
    return checks


def finish(document, payload):
    checks = derive(document, payload)
    base = copy.deepcopy(document["native_result"])
    gates = [row for key, row in checks.items() if key != "memory_safety"]
    base["correctness"] = base.get("correctness") is True and all(row["status"] == "passed" for row in gates)
    old = base.setdefault("metadata", {}).get("evidence_receipt", {}).get("checks", {})
    additional = {key: value for key, value in old.items() if key != "correctness"}
    base["metadata"]["evidence_receipt"] = from_result(base, payload["eval_config"],
        scope="hidden" if payload["eval_config"].get("use_hidden_tests") else "standard", additional={**additional, **checks})
    document["checks"] = checks
    document["receipt_sha256"] = fingerprint({key: value for key, value in document.items() if key != "receipt_sha256"})
    base["metadata"]["tool_checks"] = document
    return base


def validate(result, payload, *, allow_simulated=False):
    document = result.get("metadata", {}).get("tool_checks", {})
    settings = payload["eval_config"]
    if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
            or document.get("evaluator_sha256") != evaluator_digest() or document.get("payload_sha256") != fingerprint(payload)
            or not re.fullmatch("[0-9a-f]{32}", document.get("nonce", ""))
            or document.get("receipt_sha256") != fingerprint({key: value for key, value in document.items() if key != "receipt_sha256"})):
        raise ValueError("Missing, changed or stale tool-check receipt")
    if document.get("simulated_execution") and not allow_simulated:
        raise ValueError("Simulated diagnostics cannot establish device/tool qualification")
    native, process = document.get("native_result", {}), document.get("native_process", {})
    if (process.get("exit_code") != 0 or process.get("timed_out") is not False or process.get("error")
            or process.get("payload_sha256") != fingerprint(payload) or not coverage(native, paired=payload.get("task") == "paired_timing")
            or process.get("coverage") != coverage(native, paired=payload.get("task") == "paired_timing")):
        raise ValueError("Original uninstrumented worker did not establish its declared correctness coverage")
    tools = select_tools(settings.get("sanitizer_tools", "none"))
    if set(document.get("tools", {})) != set(tools):
        raise ValueError("Tool-check receipt lost or added a declared tool")
    from .allocation_audit import PATTERNS

    if set(document.get("allocation_controls", {})) != (set(PATTERNS) if settings.get("allocation_validation") else set()):
        raise ValueError("Allocation-control receipt lost or added a declared pattern")
    from .throughput_audit import MODES
    from .throughput_audit import validate_inline as throughput_enabled

    if set(document.get("throughput_controls", {})) != (set(MODES) if throughput_enabled(payload) else set()):
        raise ValueError("Throughput-control receipt lost or added a declared mode")
    from .cache_audit import validate_inline as cache_enabled

    cache_record = document.get("input_cache_control", {})
    if not isinstance(cache_record, dict) or bool(cache_record) != bool(cache_enabled(payload)):
        raise ValueError("Input-cache receipt lost or added its declared worker")
    if tools and document.get("binary", {}).get("status") == "measured":
        binary = document.get("binary", {})
        if (not re.fullmatch("[0-9a-f]{64}", binary.get("sha256", "")) or not binary.get("path")
                or not binary.get("version") or binary.get("version_sha256") != hashlib.sha256(binary["version"].encode()).hexdigest()):
            raise ValueError("Sanitizer executable/version identity is missing")
    elif tools and (document.get("binary", {}).get("status") != "not_measured" or not document["binary"].get("reason")):
        raise ValueError("Sanitizer executable discovery lacks an explicit unknown status")
    expected = finish(copy.deepcopy(document), payload)
    recorded = copy.deepcopy(result)
    if "sample_id" not in expected:
        recorded.pop("sample_id", None)  # Batch index is appended by the parent, never tool evidence.
    for field in ("workflow_signature", "evaluation_config", "device_lease_receipt", "execution_envelope", "restricted_worker"):
        recorded["metadata"].pop(field, None)
    if recorded != expected or document.get("checks") != derive(document, payload):
        raise ValueError("Tool-check verdict differs from its raw evidence")
    return document["checks"]


def reusable(result, payload):
    """A matching negative outcome can resume; a missing/changed receipt cannot."""
    if result.get("metadata", {}).get("workflow_signature") != fingerprint(payload):
        return False
    from .input_calibration import issue as input_issue
    from .isolation import issue as isolation_issue
    from .operator_input_policy import issue as family_issue
    from .reference_revisions import issue as revision_issue
    from .timed_numerics import issue as timed_issue

    if isolation_issue(result, payload) or revision_issue(result, payload) or input_issue(result, payload) or family_issue(result, payload) or timed_issue(result, payload):
        return False
    if not enabled(payload) or (not result.get("metadata", {}).get("tool_checks") and result.get("correctness") is not True):
        return True
    try:
        validate(result, payload)
    except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError):
        return False
    return True


def issue(result, payload):
    from .timed_numerics import issue as timed_issue

    if reason := timed_issue(result, payload):
        return reason
    from .input_calibration import issue as input_issue
    from .operator_input_policy import issue as family_issue
    from .reference_revisions import issue as revision_issue

    if reason := family_issue(result, payload):
        return reason
    if reason := input_issue(result, payload):
        return reason
    if reason := revision_issue(result, payload):
        return reason
    if not enabled(payload):
        return None
    if result.get("compiled") is not True or (result.get("correctness") is not True and not result.get("metadata", {}).get("tool_checks")):
        return None
    try:
        checks = validate(result, payload)
    except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError):
        return "tool_checks_missing_stale_or_invalid"
    if any(row["status"] != "passed" for name, row in checks.items() if name != "memory_safety"):
        return "configured_tool_or_repeat_check_not_passed"
    return None
