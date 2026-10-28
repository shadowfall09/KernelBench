"""Source-bound hidden input plans and coverage replay; SDK-light on import."""

from __future__ import annotations

import math
import re
from contextvars import ContextVar
from functools import wraps

from ..common import fingerprint
from .operator_cases import NONFINITE_TASKS, PROTOCOL, cases, labels
from .specs import spec_for

ACTIVE = ContextVar("kernelbench_operator_inputs", default=None)


def plan(level, pid, source, memory_mb):
    if level not in {1, 2, 3}:
        return None
    if type(memory_mb) is not int or not 1 <= memory_mb <= 1048576:
        raise ValueError("Invalid reviewed input-family memory envelope")
    if level in {2, 3}:
        from .model_input_cases import plan as model_plan

        return model_plan(level, pid, source, memory_mb)
    spec = spec_for(level, pid, source)
    return {"protocol": PROTOCOL, "reference_sha256": spec["reference_sha256"], "family": spec["family"],
            "labels": list(labels(pid)), "memory_mb": memory_mb,
            "nonfinite_domain_extension": pid in NONFINITE_TASKS,
            "scope": "additional_hidden_correctness_only; actual_reference_initialization; nominal_timing_unchanged"}


def validate_inline(payload):
    settings = payload["eval_config"]
    expected = plan(payload["level"], payload["problem_id"], payload["ref_arch_src"], settings.get("input_family_memory_mb"))
    if settings.get("use_hidden_tests"):
        seeds = settings.get("hidden_seeds", (42, 123, 456))
        if (not isinstance(seeds, (list, tuple)) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("Operator hidden seeds must be distinct nonnegative integers")
    if settings.get("operator_inputs") != expected:
        raise ValueError("Operator input-family plan is missing, changed or bound to another reference")
    return expected


def task_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        spec = validate_inline(payload)
        enabled = spec is not None and payload["eval_config"].get("use_hidden_tests") and payload.get("task", "evaluate") == "evaluate"
        state = {"spec": spec, "pid": payload["problem_id"], "records": []} if enabled else None
        if state is not None and payload["level"] in {2, 3}:
            from .model_input_cases import reviewed_spec

            state["model_family"] = reviewed_spec(payload["level"], payload["problem_id"], payload["ref_arch_src"])
        token = ACTIVE.set(state)
        try:
            result = function(payload, *args, **kwargs)
        finally:
            ACTIVE.reset(token)
        if state is not None:
            result.setdefault("metadata", {})["operator_inputs"] = {
                "protocol": PROTOCOL, "plan_sha256": fingerprint(spec), "cases": state["records"],
                "nominal_timing_changed": False, "reference_parameters_replaced": False}
        return result
    return wrapped


def extra_cases(reference, seed):
    state = ACTIVE.get()
    if state is None:
        return
    if "model_family" in state:
        from .model_input_cases import cases as model_cases

        iterator = model_cases(state["model_family"], seed=seed, memory_mb=state["spec"]["memory_mb"], reference=reference)
    else:
        iterator = cases(state["pid"], reference, seed=seed, memory_mb=state["spec"]["memory_mb"])
    for row in iterator:
        yield {**row, "seed": seed, "case_label": f"operator/{seed}/{row['label']}"}


def completed(row, inputs, passed):
    from src.input_groups import layout

    from ..evaluation.measurement import input_signature
    from .input_layouts import LABELS as LAYOUT_LABELS
    from .input_layouts import expected

    state = ACTIVE.get()
    if state is None:
        raise ValueError("Operator input case executed outside its frozen plan")
    signature = input_signature(inputs)
    actual_layout = layout(inputs)
    if row["label"] in LAYOUT_LABELS and actual_layout != expected(row["label"], signature):
        raise ValueError("Actual backend did not deliver the required input offset/storage aliases")
    state["records"].append({"label": row["case_label"], "family_label": row["label"], "seed": row["seed"],
        "passed": passed, "input_signature": signature, "input_layout": actual_layout,
        "estimated_input_bytes": row["estimated_input_bytes"], "stress_details": row.get("stress_details", {})})


def _valid_signature(signature):
    if not isinstance(signature, list) or not signature:
        return False
    for item in signature:
        if not isinstance(item, dict):
            return False
        if "shape" in item:
            shape, stride = item.get("shape"), item.get("stride")
            if (not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape)
                    or not isinstance(stride, list) or len(stride) != len(shape)
                    or any(type(n) is not int or n <= 0 for n in stride)
                    or item.get("dtype") not in {"torch.float32", "torch.float16", "torch.bfloat16", "torch.int64", "torch.bool"}):
                return False
        elif item.get("type") != "float" or type(item.get("value")) is not float or not math.isfinite(item["value"]):
            return False
    return True


def issue(result, payload):
    from .input_layouts import LABELS as LAYOUT_LABELS
    from .input_layouts import estimate, matches
    from .input_layouts import expected as expected_layout
    settings = payload["eval_config"]
    if payload["level"] not in {1, 2, 3} or not settings.get("use_hidden_tests") or result.get("correctness") is not True:
        return None
    try:
        spec = validate_inline(payload)
        model_family = None
        if payload["level"] in {2, 3}:
            from .model_input_cases import reviewed_spec, signature_matches

            model_family = reviewed_spec(payload["level"], payload["problem_id"], payload["ref_arch_src"])
        document = result.get("metadata", {}).get("operator_inputs", {})
        seeds = settings.get("hidden_seeds", (42, 123, 456))
        expected = [(f"operator/{seed}/{label}", label, seed) for seed in seeds for label in spec["labels"]]
        records = document.get("cases", [])
        if (document.get("protocol") != PROTOCOL or document.get("plan_sha256") != fingerprint(spec)
                or document.get("nominal_timing_changed") is not False or document.get("reference_parameters_replaced") is not False
                or not isinstance(records, list) or any(not isinstance(row, dict) for row in records)
                or [(row.get("label"), row.get("family_label"), row.get("seed")) for row in records] != expected):
            return "operator_input_family_coverage_missing_or_changed"
        actual_rows = result.get("metadata", {}).get("reliability_checks", [])
        if (not isinstance(actual_rows, list) or any(not isinstance(row, dict) for row in actual_rows)
                or len({row.get("label") for row in actual_rows}) != len(actual_rows)):
            return "operator_input_family_execution_evidence_incomplete"
        actual = {row.get("label"): row for row in actual_rows}
        for row in records:
            if (row.get("passed") is not True or type(row.get("seed")) is not int or not _valid_signature(row.get("input_signature"))
                    or type(row.get("estimated_input_bytes")) is not int
                    or not 0 < row["estimated_input_bytes"] <= spec["memory_mb"] * 1024**2
                    or row["estimated_input_bytes"] != estimate(row["family_label"], [item["shape"] for item in row["input_signature"] if "shape" in item])
                    or actual.get(row["label"], {}).get("passed") is not True
                    or actual[row["label"]].get("seed") != row["seed"]
                    or actual[row["label"]].get("orders") != ["candidate_first", "reference_first"]
                    or actual[row["label"]].get("input_signature") != row["input_signature"]
                    or (row["family_label"] == "strided" and any(item.get("stride", [2])[-1] != 2 for item in row["input_signature"]))):
                return "operator_input_family_execution_evidence_incomplete"
            if row["family_label"] in LAYOUT_LABELS:
                target = expected_layout(row["family_label"], row["input_signature"])
                native = actual[row["label"]]
                expected_calls = [(order, name) for order, names in (("candidate_first", ("candidate", "reference")),
                                  ("reference_first", ("reference", "candidate"))) for name in names]
                copies = native.get("copied_layouts", [])
                if (not matches(row.get("input_layout"), target) or not matches(native.get("input_layout"), target)
                        or not isinstance(copies, list) or any(not isinstance(item, dict) for item in copies)
                        or [(item.get("order"), item.get("model")) for item in copies] != expected_calls
                        or any(not matches(item.get("layout"), target) for item in copies)):
                    return "operator_input_family_alias_delivery_missing_or_changed"
            if model_family is not None and not signature_matches(
                    model_family, row["family_label"], row["input_signature"], settings.get("precision")):
                return "operator_input_family_geometry_or_precision_changed"
            if row["family_label"] == "weight_cancellation":
                details = row.get("stress_details", {})
                if (details.get("value_recipe") != "weight_cancellation" or details.get("reference_parameters_replaced") is not False
                        or details.get("coefficient_source") != "actual_reference_weight"
                        or not isinstance(details.get("coefficient_sha256"), str)
                        or not re.fullmatch("[0-9a-f]{64}", details["coefficient_sha256"])
                        or type(details.get("two_nonzero_products")) is not bool
                        or not isinstance(details.get("weight_positions"), list) or not 2 <= len(details["weight_positions"]) <= 3
                        or not isinstance(details.get("input_positions"), list) or len(details["input_positions"]) != len(details["weight_positions"])):
                    return "operator_input_family_weight_counterexample_missing_or_changed"
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
        return "operator_input_family_contract_invalid"
    return None
