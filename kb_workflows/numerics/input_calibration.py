"""Frozen reference-only distribution selection and runtime cohort bindings."""

from __future__ import annotations

import hashlib
import json
import math
from functools import wraps
from pathlib import Path

from ..analysis.calibration import identity
from ..common import evaluator_digest, fingerprint
from .input_profiles import recipes, selection_policy
from .specs import spec_for

PROTOCOL = "kernelbench_pro_reference_input_calibration_v1"
BUNDLE = "kernelbench_pro_input_calibration_bundle_v1"
SEEDS = (42, 123, 456)


def precision_policy(spec, precision):
    dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[precision]
    result = dict(spec["precision"][dtype])
    # Tighten the published budget. Never enlarge it to make a reference pass.
    result.update(norm_atol=0.0, element_atol=0.0)
    return result


def _qualified(row, policy, minimum_fraction):
    from .numerical_audit import _signal_valid

    if row.get("status") != "passed" or row.get("errors") != []:
        return False
    gate, signal = row.get("reference_gate", {}), row.get("signal", {})
    if (not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("reference_qualified") is not True
            or gate.get("issues") != [] or gate.get("reference_issues") != []
            or gate.get("all_elements_required") is not True or gate.get("oracle_dtype") != "float64/complex128"
            or not isinstance(gate.get("leaves"), list)
            or not _signal_valid(signal, 0.0) or signal["informative"] is not True):
        return False
    floating_paths = {leaf["path"] for leaf in signal["leaves"] if leaf.get("kind") != "discrete_exact"}
    if (any(not isinstance(leaf, dict) or not isinstance(leaf.get("path"), str) for leaf in gate["leaves"])
            or {leaf["path"].removesuffix(".complex") for leaf in gate["leaves"]} != floating_paths):
        return False
    for leaf in signal["leaves"]:
        if leaf.get("kind") != "discrete_exact" and leaf.get("fraction_above_atol", 0) < minimum_fraction:
            return False
    for leaf in gate["leaves"]:
        if (not isinstance(leaf, dict) or type(leaf.get("elements")) is not int or leaf["elements"] <= 0
                or type(leaf.get("oracle_rms")) not in (int, float) or not math.isfinite(leaf["oracle_rms"])
                or leaf["oracle_rms"] <= 0
                or leaf.get("norm_limit") != policy["norm_rtol"] * leaf["oracle_rms"]):
            return False
        for kind in ("reference", "candidate"):
            metrics = leaf.get(kind, {})
            if (not isinstance(metrics, dict) or metrics.get("passed") is not True or type(metrics.get("elements_failed")) is not int
                    or metrics["elements_failed"] != 0
                    or type(metrics.get("rms_error")) not in (int, float)
                    or not math.isfinite(metrics["rms_error"]) or not 0 <= metrics["rms_error"] <= leaf["norm_limit"]
                    or type(metrics.get("max_abs_error")) not in (int, float) or not math.isfinite(metrics["max_abs_error"])
                    or metrics["max_abs_error"] < metrics["rms_error"]):
                return False
    return True


def selected_attempt(record):
    """Select the first complete qualifying distribution, never the fastest one."""
    policy = selection_policy(record["level"], record["problem_id"])
    minimum = policy["minimum_fraction"]
    allowed = recipes(record["problem_id"], record["level"])
    if record.get("selection_policy") != policy:
        raise ValueError("Input calibration changed its reviewed selection policy")
    attempts = record.get("attempts", [])
    if (not isinstance(attempts, list) or not attempts
            or any(not isinstance(item, dict) for item in attempts)
            or [item.get("recipe") for item in attempts] != list(allowed[:len(attempts)])):
        raise ValueError("Calibration must retain the declared distribution order")
    for index, attempt in enumerate(attempts):
        rows = attempt.get("cases", [])
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or any(type(row.get("seed")) is not int for row in rows)
                or [row.get("seed") for row in rows] != list(SEEDS)
                or any(type(row.get("forward_seed")) is not int or row["forward_seed"] != policy["forward_seed"] for row in rows)):
            raise ValueError("Input calibration omitted or changed reference seeds")
        signatures = [row.get("input_signature_sha256") for row in rows]
        raw_signatures = [row.get("raw_input_signature_sha256") for row in rows]
        if (any(not isinstance(value, str) or len(value) != 64 for value in signatures + raw_signatures)
                or len(set(signatures)) != 1 or len(set(raw_signatures)) != 1
                or raw_signatures[0] != attempts[0]["cases"][0].get("raw_input_signature_sha256")):
            raise ValueError("Calibration changed nominal input dimensions, strides or roles")
        if all(_qualified(row, record["precision_contract"], minimum) for row in rows):
            if any(type(row.get("estimated_host_bytes")) is not int or not 0 < row["estimated_host_bytes"] <= record["oracle_memory_mb"] * 1024**2 for row in rows):
                raise ValueError("Calibration crossed its declared host-memory estimate envelope")
            samples = [row.get("input_probe_sha256") for row in rows]
            if any(not isinstance(value, str) or len(value) != 64 for value in samples) or len(set(samples)) != len(SEEDS):
                continue
            outputs = [row.get("output_probe_sha256") for row in rows]
            if any(not isinstance(value, str) or len(value) != 64 for value in outputs) or len(set(outputs)) < 2:
                continue
            spec = record.get("oracle_spec", {})
            if spec.get("family") in {"composite", "model_block"}:
                from .specs import validate_composite_receipt as composite_receipt
                from .specs import validate_model_receipt as model_receipt

                replay = model_receipt if spec["family"] == "model_block" else composite_receipt
                if any(not isinstance(row.get("source_device"), str) or not replay(row.get("oracle_receipt"), spec,
                        precision=record["precision"], source_device=row["source_device"]) for row in rows):
                    raise ValueError("Calibration lacks independent graph/model stochastic evidence")
            if index != len(attempts) - 1:
                raise ValueError("Calibration ran extra distributions after selecting a qualifying one")
            return attempt
    return None


def validate_record(record, source, *, allow_simulated=False, require_qualified=True):
    if (not isinstance(record, dict) or record.get("protocol") != PROTOCOL
            or record.get("receipt_sha256") != fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})
            or type(record.get("level")) is not int or record["level"] not in {1, 2, 3} or record.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or record.get("evaluator_sha256") != evaluator_digest()
            or record.get("device_identity_sha256") != identity(record.get("device", {}))
            or record.get("selection_basis") != "reference_precision_and_signal_only"
            or record.get("environment_changes") != []
            or type(record.get("oracle_memory_mb")) is not int or not 1 <= record["oracle_memory_mb"] <= 1048576
            or record.get("candidate_executed") is not False or record.get("timings_collected") is not False):
        raise ValueError("Input calibration is missing, changed, stale or bound to another task/device")
    if record.get("simulated_execution") is not False and not allow_simulated:
        raise ValueError("Simulated input calibration cannot define a measured cohort")
    if (type(record.get("problem_id")) is not int or not 1 <= record["problem_id"] <= (50 if record["level"] == 3 else 100)
            or record.get("precision") not in {"fp32", "fp16", "bf16"}):
        raise ValueError("Invalid input calibration task or precision")
    spec = spec_for(record["level"], record["problem_id"], source)
    if record.get("oracle_spec") != spec or record.get("oracle_spec_sha256") != fingerprint(spec):
        raise ValueError("Input calibration changed its independent oracle specification")
    if record.get("selection_policy") != selection_policy(record["level"], record["problem_id"], source):
        raise ValueError("Input calibration changed its reviewed nominal input roles")
    if record.get("precision_contract") != precision_policy(spec, record["precision"]):
        raise ValueError("Input calibration cannot widen the declared numerical budget")
    winner = selected_attempt(record)
    expected = winner["recipe"] if winner else None
    if record.get("selected_recipe") != expected or record.get("qualified") is not bool(winner):
        raise ValueError("Input calibration summary differs from its raw reference cases")
    if require_qualified and not winner:
        raise ValueError("No input distribution qualified; automatic fallback is not allowed")
    return record


def read_bundle(path):
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or data.get("protocol") != BUNDLE or not isinstance(data.get("records"), dict)
            or not data["records"]):
        raise ValueError("Missing reference input calibration bundle")
    return data


def record_for(path, level, pid, precision, source, *, allow_simulated=False):
    row = read_bundle(path)["records"].get(f"{level}/{pid}/{precision}")
    if row is None:
        raise ValueError(f"Input calibration unavailable for {level}/{pid}/{precision}")
    if not isinstance(row, dict) or (row.get("level"), row.get("problem_id"), row.get("precision")) != (level, pid, precision):
        raise ValueError("Input calibration task key differs from its record")
    return validate_record(row, source, allow_simulated=allow_simulated)


def subset(path, level, pid, precision, source, *, population=None):
    record = record_for(path, level, pid, precision, source)
    key = f"{level}/{pid}/{precision}"
    result = {"protocol": BUNDLE, "records": {key: record}}
    population = population or read_bundle(path).get("population_manifest")
    if population is not None:
        result.update(population_manifest=population, delegated_task=key)
    return result


def validate_inline(payload, *, allow_simulated=False):
    row = payload["eval_config"].get("input_profile")
    from ..benchmarks.reference_population import validate_inline as validate_population

    validate_population(payload, row)
    if row is None:
        return None
    if not isinstance(row, dict) or (row.get("level"), row.get("problem_id"), row.get("precision")) != (
            payload["level"], payload["problem_id"], payload["eval_config"]["precision"]):
        raise ValueError("Input distribution belongs to another task or precision")
    if payload["eval_config"].get("numerical_validation") is not True:
        raise ValueError("Calibrated cohorts require independent native-input numerical validation")
    return validate_record(row, payload["ref_arch_src"], allow_simulated=allow_simulated)


def comparison(policy):
    def compare(expected, actual, path):
        import torch

        from .numerical import PrecisionContract, precision_gate

        a, b = expected.detach().cpu(), actual.detach().cpu()
        truth = a.to(dtype=torch.complex128 if a.is_complex() else torch.float64)
        report = precision_gate(a, b, truth, PrecisionContract(**policy))
        return None if report["passed"] else f"{path}: calibrated native-output budget violated"
    return compare


def prepare_output(reference, inputs, *, scope="timing"):
    """Freeze truth before a reference forward, without advancing its state/RNG."""
    from ..evaluation.timed_numerics import active_state

    state = active_state()
    if state is None or "payload" not in state:
        return
    import torch

    from ..evaluation.determinism import preserve_rng
    from ..evaluation.measurement import input_signature
    from .specs import prepare_oracle as prepare

    payload = state["payload"]
    spec = spec_for(payload["level"], payload["problem_id"], payload["ref_arch_src"])
    device = next(value.device for value in inputs if isinstance(value, torch.Tensor))
    with torch.no_grad(), preserve_rng(device):
        prepared = prepare(reference, inputs, spec=spec, problem_id=payload["problem_id"])
        if prepared.estimated_host_bytes > payload["eval_config"]["oracle_memory_mb"] * 1024**2:
            raise ValueError("Full-size timed-output oracle exceeds the host-memory estimate envelope; no resizing")
        truth = prepared.execute()
    return {"truth": truth, "input_signature": input_signature(inputs), "reference_id": id(reference),
            "level": payload["level"], "problem_id": payload["problem_id"], "oracle_spec_sha256": fingerprint(spec),
            "pre_reference_forward": True, "scope": scope, "source_device": str(device),
            "estimated_host_bytes": prepared.estimated_host_bytes, "oracle_receipt": getattr(prepared, "receipt", lambda: None)()}


def verify_output(reference, inputs, expected, actual, *, prepared=None):
    """Independent formula checks outside timing; inert without a frozen profile."""
    from ..evaluation.timed_numerics import active_state

    state = active_state()
    if state is None or "payload" not in state:
        return
    import torch

    from ..evaluation.measurement import input_signature
    from .numerical import PrecisionContract, precision_gate

    payload = state["payload"]
    if prepared is None:
        if payload["level"] != 1:
            raise ValueError("Model timed-output verification needs a pre-forward independent oracle ticket")
        prepared = prepare_output(reference, inputs)
    if (not isinstance(prepared, dict) or prepared.get("reference_id") != id(reference)
            or prepared.get("input_signature") != input_signature(inputs)
            or prepared.get("pre_reference_forward") is not True
            or (prepared.get("level"), prepared.get("problem_id")) != (payload["level"], payload["problem_id"])
            or prepared.get("oracle_spec_sha256") != fingerprint(spec_for(payload["level"], payload["problem_id"], payload["ref_arch_src"]))):
        raise ValueError("Timed-output oracle ticket belongs to another reference/input plan")
    truth = prepared["truth"]

    def host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (tuple, list)):
            return type(value)(host(child) for child in value)
        if isinstance(value, dict):
            return {key: host(child) for key, child in value.items()}
        return value

    policy = state.get("policy") or payload["eval_config"]["input_profile"]["precision_contract"]
    expected, actual = host(expected), host(actual)
    report = precision_gate(expected, actual, truth, PrecisionContract(**policy))
    from ..evaluation.timed_numerics import observe

    observe(reference, inputs, expected, actual, prepared, report)
    if not report["passed"] or not report["reference_qualified"]:
        raise ValueError("Calibrated timed output or reference failed its independent FP64 budget")
    state["independent_outputs_verified"] += 1
    from .input_profiles import ACTIVE as NOMINAL

    nominal = NOMINAL.get()
    if nominal is not None and nominal is not state and "payload" in nominal:
        nominal["independent_outputs_verified"] += 1


def task_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        if payload.get("task") == "calibrate_precision":
            # The reference-only fitter applies the selected recipe exactly once.
            return function(payload, *args, **kwargs)
        row = validate_inline(payload)
        if row is None:
            return function(payload, *args, **kwargs)
        from src.validation import output_comparison_policy

        from ..evaluation.measurement import device_fingerprint
        from .input_profiles import context

        backend = payload["eval_config"].get("device_backend", "cuda")
        observed = identity(device_fingerprint(f"{backend}:0"))
        with context(payload["ref_arch_src"], payload["problem_id"], row["selected_recipe"], level=payload["level"]) as state:
            state.update(payload=payload, independent_outputs_verified=0)
            with output_comparison_policy(comparison(row["precision_contract"])):
                result = function(payload, *args, **kwargs)
        signature = fingerprint({"signature": state["input_signature"]}) if state["input_signature"] is not None else None
        if state["factory_calls"] and signature != selected_attempt(row)["cases"][0]["raw_input_signature_sha256"]:
            raise ValueError("Nominal input signature differs from the calibrated original-size cohort")
        # Value distributions transfer across devices. Actual-device FP64 checks
        # remain compulsory; pilot observations are never new-device evidence.
        result.setdefault("metadata", {})["input_calibration"] = {
            "record_sha256": row["receipt_sha256"], "recipe": row["selected_recipe"],
            "calibration_device_identity_sha256": row["device_identity_sha256"],
            "execution_device_identity_sha256": observed, "factory_calls": state["factory_calls"],
            "independent_outputs_verified": state["independent_outputs_verified"],
            "nominal_input_signature_sha256": signature, "shapes_resized": False,
            "transfer_scope": "input_distribution_only; current_device_numerical_qualification_required",
            "comparison_reference": "native_output_plus_independent_FP64_actual_inputs"}
        from ..benchmarks.reference_population import validate_inline as validate_population

        result["metadata"]["input_calibration"]["reference_population_sha256"] = validate_population(payload, row)
        return result
    return wrapped


def issue(result, payload):
    if payload["eval_config"].get("input_profile") is None:
        return None
    try:
        row = validate_inline(payload)
        from ..benchmarks.reference_population import validate_inline as validate_population

        population_sha = validate_population(payload, row)
        receipt = result.get("metadata", {}).get("input_calibration", {})
        positive = result.get("correctness") is True or result.get("baseline_correctness") is True
        if not positive:
            return None
        settings, phase = payload["eval_config"], payload.get("task", "evaluate")
        nominal = phase != "qualify_small" and not settings.get("use_hidden_tests")
        verified = receipt.get("independent_outputs_verified")
        required = 0
        if phase == "baseline":
            required = settings["num_perf_trials"] + 4  # First output, warmup, measured calls.
        elif phase == "paired_timing":
            required = (settings.get("paired_trials", 30) + 3) * 3
        elif phase == "evaluate" and settings.get("measure_performance", not settings.get("use_hidden_tests")):
            required = settings["num_perf_trials"] + 3 + (settings.get("device_backend", "cuda") == "cuda")
        if (receipt.get("record_sha256") != row["receipt_sha256"] or receipt.get("recipe") != row["selected_recipe"]
                or receipt.get("reference_population_sha256") != population_sha
                or receipt.get("shapes_resized") is not False
                or receipt.get("calibration_device_identity_sha256") != row["device_identity_sha256"]
                or receipt.get("execution_device_identity_sha256") != identity(result.get("device_fingerprint", result.get("metadata", {}).get("device_fingerprint", {})))
                or type(receipt.get("factory_calls")) is not int or receipt["factory_calls"] < int(nominal)
                or (receipt["factory_calls"] > 0 and receipt.get("nominal_input_signature_sha256") != selected_attempt(row)["cases"][0]["raw_input_signature_sha256"])
                or type(verified) is not int or verified < required):
            return "calibrated_input_execution_evidence_missing"
    except (ValueError, KeyError, TypeError, OSError, AttributeError):
        return "input_calibration_missing_stale_or_changed"
    return None


def baseline_issue(config, baseline, pid):
    if not config.input_calibration:
        return None
    from ..engine import task_payload

    base = task_payload(config, pid, task="baseline")
    if not config.baseline_suite:
        return issue(baseline, base)
    for variant, result in baseline.get("variants", {}).items():
        if result.get("baseline_correctness") is True:
            payload = {**base, "eval_config": {**base["eval_config"], "baseline_variant": variant}}
            if reason := issue(result, payload):
                return reason
    return issue(baseline, {**base, "eval_config": {**base["eval_config"], "baseline_variant": baseline.get("baseline_variant")}})
