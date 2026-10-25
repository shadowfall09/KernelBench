"""Reference-only, device-bound precision fits inside immutable published caps."""

from __future__ import annotations

import hashlib
import json
import math
from functools import wraps
from pathlib import Path

from ..analysis.calibration import IDENTITY_FIELDS, identity
from ..common import atomic_json, evaluator_digest, fingerprint, read_json
from .operator_specs import spec_for

PROTOCOL = "kernelbench_pro_reference_precision_calibration_v1"
BUNDLE = "kernelbench_pro_precision_calibration_bundle_v1"
FIT_SEEDS = (811, 1291, 2027)
CHECK_SEEDS = (1729, 2718, 3141)
FIT = {"margin": 4.0, "minimum_factor": 1e-6, "maximum_factor": 1.0,
       "method": "uniform_scale_of_all_five_fixed_budget_terms; no_widening; no_holdout_refit",
       "initialization_seed": 42, "forward_seed": 42}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def numeric_class(device):
    identity(device)
    return fingerprint({key: device[key] for key in IDENTITY_FIELDS if key != "uuid"})


def base_policy(spec, precision, profile=None):
    if profile is not None:
        return dict(profile["precision_contract"])
    return dict(spec["precision"][{"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[precision]])


def fitted_policy(base, factor):
    if type(factor) not in (int, float) or not math.isfinite(factor) or not FIT["minimum_factor"] <= factor <= 1:
        raise ValueError("Fitted precision factor must stay inside its published cap")
    return {key: value * factor for key, value in base.items()}


def _utilization(row, policy):
    from .numerical_audit import _signal_valid

    if (not isinstance(row, dict) or row.get("status") != "passed" or row.get("errors") != []
            or not _signal_valid(row.get("signal"), policy["element_atol"]) or not row["signal"]["informative"]
            or not isinstance(row.get("utilization"), dict)):
        raise ValueError("Reference precision observation is missing, failed or uninformative")
    utilization = row["utilization"]
    gate = row.get("reference_gate", {})
    if (not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("reference_qualified") is not True
            or gate.get("issues") != [] or gate.get("reference_issues") != []
            or gate.get("all_elements_required") is not True or gate.get("oracle_dtype") != "float64/complex128"
            or not isinstance(gate.get("leaves"), list)):
        raise ValueError("Precision fit lacks a complete all-element reference self-check")
    leaves = utilization.get("leaves")
    if (not isinstance(leaves, list) or any(not isinstance(leaf, dict) for leaf in leaves)
            or len(gate["leaves"]) != len(leaves) or any(not isinstance(leaf, dict) for leaf in gate["leaves"])):
        raise ValueError("Missing raw precision-utilization leaves")
    paths = set()
    for leaf in leaves:
        if (not isinstance(leaf.get("path"), str) or leaf["path"] in paths
                or type(leaf.get("elements")) is not int or leaf["elements"] <= 0
                or any(type(leaf.get(k)) not in (int, float) or not math.isfinite(leaf[k]) or leaf[k] < 0
                    for k in ("oracle_rms", "rms_error", "max_abs_error", "norm_limit", "norm_utilization", "element_utilization"))
                or leaf["norm_limit"] != policy["norm_atol"] + policy["norm_rtol"] * leaf["oracle_rms"]
                or leaf["norm_utilization"] != (leaf["rms_error"] / leaf["norm_limit"] if leaf["norm_limit"] else 0.0)
                or leaf["norm_utilization"] > 1 or leaf["element_utilization"] > 1):
            raise ValueError("Precision observations contradict the fixed error cap")
        matching = [item for item in gate["leaves"] if isinstance(item, dict) and item.get("path") == leaf["path"]]
        if len(matching) != 1 or any(matching[0].get(k) != leaf[k] for k in ("elements", "oracle_rms", "norm_limit")):
            raise ValueError("Precision-fit errors do not match their raw reference gate")
        for kind in ("reference", "candidate"):
            metrics = matching[0].get(kind, {})
            if (not isinstance(metrics, dict) or metrics.get("passed") is not True or metrics.get("elements_failed") != 0
                    or metrics.get("rms_error") != leaf["rms_error"] or metrics.get("max_abs_error") != leaf["max_abs_error"]
                    or leaf["max_abs_error"] < leaf["rms_error"]):
                raise ValueError("Reference error counters contradict the threshold fit")
        paths.add(leaf["path"])
    if (utilization.get("floating_elements") != sum(leaf["elements"] for leaf in leaves)
            or type(utilization.get("floating_elements")) is not int
            or utilization.get("norm_max") != max((leaf["norm_utilization"] for leaf in leaves), default=0.0)
            or utilization.get("element_max") != max((leaf["element_utilization"] for leaf in leaves), default=0.0)):
        raise ValueError("Precision-utilization summary differs from its observations")
    floating_paths = {leaf["path"] for leaf in row["signal"]["leaves"] if leaf.get("kind") != "discrete_exact"}
    if {path.removesuffix(".complex") for path in paths} != floating_paths:
        raise ValueError("Precision-fit leaves differ from the materialized reference outputs")
    return max(utilization["norm_max"], utilization["element_max"])


def fit_factor(rows, policy):
    probes = [row.get("input_probe_sha256") for row in rows]
    if (len(rows) != len(FIT_SEEDS) or any(not _sha(v) for v in probes)
            or len(set(probes)) != len(rows)):
        raise ValueError("Precision fitting needs distinct seeded input probes, not repeated bytes")
    values = [_utilization(row, policy) for row in rows]
    if all(row["utilization"]["floating_elements"] == 0 for row in rows):
        return 1.0  # Discrete outputs stay exact; do not invent floating observations.
    return min(1.0, max(FIT["minimum_factor"], FIT["margin"] * max(values)))


def heldout_passed(row, base, factor):
    if _utilization(row, base) > factor:
        return False
    gate, policy = row.get("fitted_gate", {}), fitted_policy(base, factor)
    if (not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("reference_qualified") is not True
            or gate.get("issues") != [] or gate.get("reference_issues") != []
            or gate.get("all_elements_required") is not True or gate.get("oracle_dtype") != "float64/complex128"
            or not isinstance(gate.get("leaves"), list) or len(gate["leaves"]) != len(row["utilization"]["leaves"])
            or any(not isinstance(item, dict) for item in gate["leaves"])):
        return False
    leaves = row["utilization"]["leaves"]
    if [item.get("path") for item in gate["leaves"] if isinstance(item, dict)] != [item["path"] for item in leaves]:
        return False
    for raw, fitted in zip(leaves, gate["leaves"]):
        if (fitted.get("elements") != raw["elements"] or fitted.get("oracle_rms") != raw["oracle_rms"]
                or fitted.get("norm_limit") != policy["norm_atol"] + policy["norm_rtol"] * raw["oracle_rms"]):
            return False
        for name in ("reference", "candidate"):
            metrics = fitted.get(name, {})
            if (not isinstance(metrics, dict) or metrics.get("passed") is not True or metrics.get("elements_failed") != 0
                    or metrics.get("rms_error") != raw["rms_error"] or metrics.get("max_abs_error") != raw["max_abs_error"]
                    or metrics["rms_error"] > fitted["norm_limit"]):
                return False
    return True


def validate_record(record, source, precision, profile=None, *, allow_simulated=False, require_qualified=True):
    if (not isinstance(record, dict) or type(record.get("level")) is not int or record["level"] not in {1, 2, 3}
            or type(record.get("problem_id")) is not int or precision not in {"fp32", "fp16", "bf16"}):
        raise ValueError("Precision calibration requires an original task/dtype identity")
    spec = spec_for(record["level"], record["problem_id"], source)
    if (not isinstance(record, dict) or record.get("protocol") != PROTOCOL
            or record.get("receipt_sha256") != fingerprint({k: v for k, v in record.items() if k != "receipt_sha256"})
            or record.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or record.get("evaluator_sha256") != evaluator_digest() or record.get("oracle_spec") != spec
            or record.get("oracle_spec_sha256") != fingerprint(spec) or record.get("precision") != precision
            or record.get("device_identity_sha256") != identity(record.get("device", {}))
            or record.get("fit_method") != FIT or record.get("candidate_executed") is not False
            or record.get("timings_collected") is not False or record.get("environment_changes") != []
            or record.get("input_profile_sha256") != (profile or {}).get("receipt_sha256")
            or record.get("base_policy") != base_policy(spec, precision, profile)
            or type(record.get("oracle_memory_mb")) is not int or not 1 <= record["oracle_memory_mb"] <= 1048576):
        raise ValueError("Precision calibration is missing, stale or bound to another task/policy")
    if record.get("simulated_execution") is not False and not allow_simulated:
        raise ValueError("Simulated precision calibration cannot establish device thresholds")
    if (type(record.get("level")) is not int or record["level"] not in {1, 2, 3}
            or type(record.get("problem_id")) is not int):
        raise ValueError("Precision calibration requires an original task identity")
    stages = []
    for name, seeds in (("fit", FIT_SEEDS), ("check", CHECK_SEEDS)):
        rows = record.get(name)
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or [row.get("seed") for row in rows] != list(seeds)
                or any(row.get("forward_seed") != FIT["forward_seed"] for row in rows)):
            raise ValueError("Precision calibration omitted or changed fit/check seeds")
        stages.append(rows)
    signatures = [row.get("input_signature_sha256") for rows in stages for row in rows]
    raw_signatures = [row.get("raw_input_signature_sha256") for rows in stages for row in rows]
    if (any(not _sha(v) for v in signatures + raw_signatures)
            or len(set(signatures)) != 1 or len(set(raw_signatures)) != 1):
        raise ValueError("Precision calibration changed native dimensions or storage roles")
    for rows in stages:
        for row in rows:
            if row.get("status") == "passed" and (not isinstance(row.get("input_signature"), list)
                    or row["input_signature_sha256"] != fingerprint({"signature": row["input_signature"]})):
                raise ValueError("Precision fit input signature differs from its raw geometry")
            if (type(row.get("estimated_host_bytes")) is not int
                    or not 0 < row["estimated_host_bytes"] <= record["oracle_memory_mb"] * 1024**2):
                raise ValueError("Precision calibration has unresolved original-size memory admission")
            if row.get("status") == "passed" and spec["family"] in {"composite", "model_block"}:
                from .composite_specs import receipt_valid as composite_receipt
                from .model_specs import receipt_valid as model_receipt

                check = model_receipt if spec["family"] == "model_block" else composite_receipt
                if not isinstance(row.get("source_device"), str) or not check(row.get("oracle_receipt"), spec,
                        precision=precision, source_device=row["source_device"]):
                    raise ValueError("Precision fit lacks frozen program/native-mask evidence")
    qualified = False
    try:
        factor = fit_factor(stages[0], record["base_policy"])
    except ValueError:
        factor = None
    if factor is not None:
        try:
            probes = [row.get("input_probe_sha256") for rows in stages for row in rows]
            qualified = (all(_sha(v) for v in probes) and len(set(probes)) == len(probes)
                         and all(heldout_passed(row, record["base_policy"], factor) for row in stages[1]))
        except ValueError:
            qualified = False
    expected = fitted_policy(record["base_policy"], factor) if factor is not None else None
    kind = ("discrete_exact_only; no_float_threshold_fit" if factor is not None and all(row["utilization"]["floating_elements"] == 0 for row in stages[0])
            else "floating_budget_fit" if factor is not None else None)
    if ((record.get("factor") is not None and type(record["factor"]) not in (int, float))
            or record.get("factor") != factor or record.get("policy") != expected or record.get("qualified") is not qualified
            or record.get("fit_kind") != kind):
        raise ValueError("Fitted policy or held-out qualification differs from raw reference observations")
    if require_qualified and not qualified:
        raise ValueError("Reference failed its frozen precision fit/holdout; no widening or automatic fallback")
    return record


def read_bundle(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("protocol") != BUNDLE or not isinstance(value.get("records"), dict) or not value["records"]:
        raise ValueError("Missing reference precision-calibration bundle")
    return value


def record_for(path, level, pid, precision, source, profile=None, *, allow_simulated=False):
    row = read_bundle(path)["records"].get(f"{level}/{pid}/{precision}")
    if not isinstance(row, dict) or (row.get("level"), row.get("problem_id"), row.get("precision")) != (level, pid, precision):
        raise ValueError("Precision calibration is missing for the declared task/dtype")
    return validate_record(row, source, precision, profile, allow_simulated=allow_simulated)


def subset(path, level, pid, precision, source, profile=None):
    row = record_for(path, level, pid, precision, source, profile)
    return {"protocol": BUNDLE, "records": {f"{level}/{pid}/{precision}": row}}


def for_config(config, pid, source):
    from .input_calibration import record_for as input_record

    profile = input_record(config.input_calibration, config.level, pid, config.precision, source) if config.input_calibration else None
    return record_for(config.precision_calibration, config.level, pid, config.precision, source, profile)


def subset_for_config(config, pid, source):
    row = for_config(config, pid, source)
    return {"protocol": BUNDLE, "records": {f"{config.level}/{pid}/{config.precision}": row}}


def freeze_run(config, selected=None):
    """A fit is a preregistered run policy; regrading uses a new output run."""
    path = config.run_dir / "precision_calibration_manifest.json"
    expected = None
    if config.precision_calibration:
        if selected is None:
            from ..common import select_problems

            selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track)
        classes = {numeric_class(for_config(config, pid, source.read_text())["device"]) for pid, source in selected.items()}
        if len(classes) != 1:
            raise ValueError("Precision-fit runs require one hardware/software class; split device configurations")
        expected = {"protocol": PROTOCOL, "bundle_sha256": fingerprint(read_bundle(config.precision_calibration)),
                    "numeric_class_sha256": next(iter(classes)),
                    "evaluator_sha256": evaluator_digest(),
                    "contract": {key: getattr(config, key) for key in (
                        "level", "precision", "start", "end", "suite", "suite_role", "task_track", "num_samples", "gpus")}}
    if path.exists() or path.is_symlink():
        if path.is_symlink() or expected is None or read_json(path) != expected:
            raise ValueError("Frozen precision calibration changed; use a new run or artifact regrading")
    elif expected is not None and not config.dry_run:
        if any(config.run_dir.glob("level_*_kernel.py")) or any((config.run_dir / name).exists() for name in (
                "work", "generation_summary.json", "generation_attempts.json", "eval_results.json", "eval_results_hidden.json",
                "final_manifest.json", "paired_results.json", "paper_analysis.json", "harbor_export.json")):
            raise ValueError("Precision thresholds must freeze before candidates; use a new run or artifact regrading")
        atomic_json(path, expected)
    return expected


def validate_inline(payload):
    row = payload["eval_config"].get("precision_profile")
    if row is None:
        return None
    if not isinstance(row, dict) or payload["eval_config"].get("numerical_validation") is not True or (row.get("level"), row.get("problem_id")) != (payload["level"], payload["problem_id"]):
        raise ValueError("Precision calibration requires matching independent native numerical validation")
    return validate_record(row, payload["ref_arch_src"], payload["eval_config"]["precision"], payload["eval_config"].get("input_profile"))


def task_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        row = validate_inline(payload)
        if row is None:
            return function(payload, *args, **kwargs)
        from ..evaluation.measurement import device_fingerprint

        device = device_fingerprint(f"{payload['eval_config'].get('device_backend', 'cuda')}:0")
        observed = identity(device)
        if numeric_class(device) != numeric_class(row["device"]):
            raise ValueError("Fitted precision thresholds cannot transfer to a different hardware/software class")
        result = function(payload, *args, **kwargs)
        result.setdefault("metadata", {})["precision_calibration"] = {
            "record_sha256": row["receipt_sha256"], "policy_sha256": fingerprint(row["policy"]),
            "calibration_device_identity_sha256": row["device_identity_sha256"],
            "execution_device_identity_sha256": observed,
            "numeric_class_sha256": numeric_class(device),
            "transfer_scope": "numerical_policy_only; current_device_FP64_self_qualification_required"}
        return result
    return wrapped


def issue(result, payload):
    if payload["eval_config"].get("precision_profile") is None:
        return None
    if result.get("correctness") is not True and result.get("baseline_correctness") is not True:
        return None
    try:
        row = validate_inline(payload)
        receipt = result.get("metadata", {}).get("precision_calibration", {})
        device = result.get("device_fingerprint", result.get("metadata", {}).get("device_fingerprint", {}))
        observed = identity(device)
        if receipt != {"record_sha256": row["receipt_sha256"], "policy_sha256": fingerprint(row["policy"]),
                       "calibration_device_identity_sha256": row["device_identity_sha256"],
                       "execution_device_identity_sha256": observed, "numeric_class_sha256": numeric_class(device),
                       "transfer_scope": "numerical_policy_only; current_device_FP64_self_qualification_required"} or numeric_class(device) != numeric_class(row["device"]):
            return "precision_calibration_device_or_execution_evidence_missing"
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        return "precision_calibration_missing_stale_or_changed"
    return None
