"""Independent native-output gates outside every scored wall-clock interval."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

from ..common import evaluator_digest, fingerprint

PROTOCOL = "kernelbench_pro_timed_numerical_audit_v1"
ACTIVE = ContextVar("kernelbench_timed_numerics", default=None)


def active_state():
    state = ACTIVE.get()
    if state is not None:
        return state
    # Preserve the standalone nominal-profile helper API. Formal workers use
    # this module's outer context, independently of input-distribution changes.
    from ..numerics.input_profiles import ACTIVE as NOMINAL

    state = NOMINAL.get()
    return state if state is not None and "payload" in state else None


def requirements(payload):
    settings, phase = payload["eval_config"], payload.get("task", "evaluate")
    if phase == "baseline":
        return {"first": 1, "timing": settings["num_perf_trials"] + 3}
    if phase == "paired_timing":
        return {"timing": (settings.get("paired_trials", 30) + 3) * 3}
    if phase == "evaluate" and settings.get("measure_performance", not settings.get("use_hidden_tests")):
        return {"first": 1, "timing": settings["num_perf_trials"] + 3,
                **({"memory": 1} if settings.get("device_backend", "cuda") == "cuda" else {})}
    return {}


def pair_hooks(reference, inputs, *, seed, device, set_seed, first=False):
    """Gate actual native correctness/cold-call outputs, in both call orders."""
    if ACTIVE.get() is None:
        return {}
    ticket = None

    def prepare_pair():
        nonlocal ticket
        from ..numerics.input_calibration import prepare_output
        from .determinism import preserve_rng

        with preserve_rng(device):
            set_seed(seed)
            ticket = prepare_output(reference, inputs, scope="first" if first else "control")

    def verify_pair(values, outputs):
        from ..numerics.input_calibration import verify_output

        for output in outputs.values():
            verify_output(reference, values, outputs["reference"], output, prepared=ticket)

    return {"prepare_pair": prepare_pair, "verify_pair": verify_pair}


def observe(reference, inputs, expected, actual, prepared, report):
    state = ACTIVE.get()
    if state is None:
        return
    from ..numerics.numerical import signal_audit
    from .measurement import input_signature

    scope = prepared["scope"]
    if scope not in {"first", "timing", "memory", "control"}:
        raise ValueError("Unknown independent numerical output scope")
    record = state["payload"]["eval_config"].get("precision_profile")
    if record is not None and scope in {"first", "timing", "memory"}:
        if fingerprint({"signature": input_signature(inputs)}) != record["fit"][0]["input_signature_sha256"]:
            raise ValueError("Nominal dimensions/dtype differ from the device-fitted precision contract")
    state["rows"].append({"label": f"output/{len(state['rows'])}", "seed": None,
        "scope": scope, "input_signature": input_signature(inputs), "status": "passed" if report["passed"] else "failed",
        "errors": [], "estimated_host_bytes": prepared["estimated_host_bytes"], "precision_gate": report,
        "reference_signal": signal_audit(expected, absolute_tolerance=state["policy"]["element_atol"]),
        "source_device": prepared["source_device"], "composite_receipt": prepared["oracle_receipt"],
        "model_receipt": prepared["oracle_receipt"], "pre_reference_forward": True})


def _document(state):
    from ..numerics.numerical_audit import PROTOCOL as NUMERICAL

    payload, settings = state["payload"], state["payload"]["eval_config"]
    audit = {"protocol": NUMERICAL, "oracle_spec_sha256": fingerprint(state["spec"]), "policy": state["policy"],
             "oracle_memory_mb": settings["oracle_memory_mb"], "cases": state["rows"]}
    document = {"protocol": PROTOCOL, "evaluator_sha256": evaluator_digest(), "task": payload.get("task", "evaluate"),
        "payload_sha256": fingerprint(payload),
        "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
        "precision": settings["precision"], "requirements": requirements(payload), "audit": audit,
        "input_profile_sha256": (settings.get("input_profile") or {}).get("receipt_sha256"),
        "oracle_timing_scope": "outside_scored_intervals; actual_pre_reference_state_and_rng",
        "seed_scope": "actual_rng_replay; per_output_seed_not_inferred_from_ordinal"}
    document["receipt_sha256"] = fingerprint(document)
    return document


def issue(result, payload):
    from ..numerics.precision_calibration import issue as precision_issue

    if reason := precision_issue(result, payload):
        return reason
    if not payload["eval_config"].get("numerical_validation"):
        return None
    if result.get("correctness") is not True and result.get("baseline_correctness") is not True:
        return None
    try:
        from ..numerics.numerical_audit import _spec, verdicts

        spec, policy = _spec(payload)
        document = result.get("metadata", {}).get("timed_numerical_audit", {})
        if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
                or document.get("receipt_sha256") != fingerprint({k: v for k, v in document.items() if k != "receipt_sha256"})
                or document.get("evaluator_sha256") != evaluator_digest()
                or document.get("payload_sha256") != fingerprint(payload)
                or document.get("task") != payload.get("task", "evaluate")
                or document.get("reference_sha256") != fingerprint({"source": payload["ref_arch_src"]})
                or document.get("precision") != payload["eval_config"]["precision"]
                or document.get("input_profile_sha256") != (payload["eval_config"].get("input_profile") or {}).get("receipt_sha256")
                or document.get("requirements") != requirements(payload)
                or document.get("oracle_timing_scope") != "outside_scored_intervals; actual_pre_reference_state_and_rng"
                or document.get("seed_scope") != "actual_rng_replay; per_output_seed_not_inferred_from_ordinal"):
            return "timed_numerical_receipt_missing_or_changed"
        audit = document.get("audit", {})
        rows = audit.get("cases") if isinstance(audit, dict) else None
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or [row.get("label") for row in rows] != [f"output/{i}" for i in range(len(rows))]
                or any(row.get("scope") not in {"first", "timing", "memory", "control"} or row.get("seed") is not None
                    or row.get("pre_reference_forward") is not True for row in rows)
                or audit.get("oracle_spec_sha256") != fingerprint(spec) or audit.get("policy") != policy
                or audit.get("oracle_memory_mb") != payload["eval_config"]["oracle_memory_mb"]):
            return "timed_numerical_cases_missing_or_changed"
        for scope, minimum in requirements(payload).items():
            if sum(row["scope"] == scope for row in rows) < minimum:
                return "timed_numerical_output_coverage_missing"
        if rows:
            cases = [{"label": row["label"], "seed": None} for row in rows]
            checks = verdicts(audit, cases, payload)
            if checks["precision_contract"]["status"] != "passed":
                return "timed_numerical_budget_or_reference_failed"
            if requirements(payload) and checks["reference_signal"]["status"] != "passed":
                return "timed_reference_signal_uninformative"
    except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError):
        return "timed_numerical_contract_missing_or_stale"
    return None


def task_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        if not payload["eval_config"].get("numerical_validation") or payload.get("task") in {"calibrate_inputs", "calibrate_precision", "origin_audit"}:
            return function(payload, *args, **kwargs)
        from ..numerics.numerical_audit import _spec

        spec, policy = _spec(payload)
        state = {"payload": payload, "spec": spec, "policy": policy, "rows": [], "independent_outputs_verified": 0}
        token = ACTIVE.set(state)
        try:
            result = function(payload, *args, **kwargs)
        finally:
            ACTIVE.reset(token)
        result.setdefault("metadata", {})["timed_numerical_audit"] = _document(state)
        if reason := issue(result, payload):
            if "correctness" in result:
                result["correctness"] = False
            if "baseline_correctness" in result:
                result["baseline_correctness"] = False
            result["metadata"]["timed_numerical_error"] = reason
        return result
    return wrapped


def baseline_issue(config, baseline, pid):
    if not config.numerical_validation:
        return None
    from ..engine import task_payload

    base = task_payload(config, pid, task="baseline")
    if config.baseline_suite:
        for variant, result in baseline.get("variants", {}).items():
            expected = {**base, "eval_config": {**base["eval_config"], "baseline_variant": variant}}
            if reason := issue(result, expected):
                return reason
        base = {**base, "eval_config": {**base["eval_config"], "baseline_variant": baseline.get("winning_baseline")}}
    return issue(baseline, base)
