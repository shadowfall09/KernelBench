"""Explicit reference-only threshold fitting; never called by preview/import."""

from __future__ import annotations

import copy
import hashlib

from ..common import evaluator_digest, fingerprint, json_safe
from .operator_specs import spec_for
from .precision_calibration import (
    CHECK_SEEDS,
    FIT,
    FIT_SEEDS,
    PROTOCOL,
    _utilization,
    base_policy,
    fit_factor,
    fitted_policy,
    heldout_passed,
)


def calibrate(payload, *, adapter=None, loader=None):
    import torch

    from src.eval import (
        _process_input_group,
        _process_input_tensor,
        get_torch_dtype_from_string,
        load_original_model_and_inputs,
        move_model,
        set_seed,
    )
    from src.input_groups import copy_inputs
    from src.validation import clone_tensors, synchronize

    from ..analysis.calibration import identity
    from ..evaluation.determinism import preserve_rng
    from ..evaluation.evaluate_task import validate_sources
    from ..evaluation.measurement import input_signature
    from ..hardware.devices import adapter as device_adapter
    from .input_calibration import validate_inline as input_profile
    from .input_calibration_task import precision_environment, probe
    from .input_profiles import transform
    from .numerical import PrecisionContract, precision_gate, reference_utilization, signal_audit
    from .reference_oracles import prepare

    settings = payload["eval_config"]
    if "kernel_src" in payload or payload["level"] not in {1, 2, 3} or settings.get("precision_profile"):
        raise ValueError("Precision calibration accepts original references, with no candidate or previous precision fit")
    source = validate_sources(payload)
    profile = input_profile(payload)
    spec = spec_for(payload["level"], payload["problem_id"], source)
    policy = base_policy(spec, settings["precision"], profile)
    simulated = adapter is not None or loader is not None
    adapter = adapter or device_adapter(settings.get("device_backend", "cuda"), "0")
    device, caps = adapter.device, adapter.caps()
    record = {"protocol": PROTOCOL, "level": payload["level"], "problem_id": payload["problem_id"], "precision": settings["precision"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "evaluator_sha256": evaluator_digest(),
        "oracle_spec": spec, "oracle_spec_sha256": fingerprint(spec), "base_policy": policy, "fit_method": dict(FIT),
        "device": caps, "device_identity_sha256": identity(caps), "simulated_execution": simulated or bool(getattr(adapter, "simulated", False)),
        "input_profile_sha256": (profile or {}).get("receipt_sha256"), "oracle_memory_mb": settings["oracle_memory_mb"],
        "candidate_executed": False, "timings_collected": False, "fit": [], "check": [], "factor": None, "policy": None,
        "qualified": False, "fit_kind": None, "input_probe_scope": "up_to_256_deterministic_positions_per_tensor; not_a_full_content_hash"}
    dtype = get_torch_dtype_from_string(settings["precision"])
    Model, init, inputs = (loader or load_original_model_and_inputs)(source, {})

    def seed(value):
        set_seed(value)
        if str(device).split(":")[0] == "neuron":
            torch.neuron.manual_seed(value)

    def host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (tuple, list)):
            return type(value)(host(item) for item in value)
        if isinstance(value, dict):
            return {key: host(item) for key, item in value.items()}
        return value

    changes = []
    with torch.no_grad(), preserve_rng(device), precision_environment(changes):
        seed(FIT["initialization_seed"])
        initial = [_process_input_tensor(value, device, dtype) for value in init()]
        model = move_model(Model(*initial), device, dtype)
        for stage, seeds in (("fit", FIT_SEEDS), ("check", CHECK_SEEDS)):
            for value in seeds:
                row = {"seed": value, "forward_seed": FIT["forward_seed"], "status": "not_measured", "errors": []}
                record[stage].append(row)
                try:
                    seed(value)
                    raw = inputs()
                    raw_signature = input_signature(raw)
                    row["raw_input_signature_sha256"] = fingerprint({"signature": raw_signature})
                    prepared = prepare(model, raw, spec=spec, problem_id=payload["problem_id"])
                    row["estimated_host_bytes"] = prepared.estimated_host_bytes
                    if prepared.estimated_host_bytes > settings["oracle_memory_mb"] * 1024**2:
                        raise ValueError("Original-size precision fit exceeds the host-memory estimate envelope; no resizing")
                    values = transform(raw, payload["problem_id"], profile["selected_recipe"], level=payload["level"]) if profile else raw
                    if input_signature(values) != raw_signature:
                        raise ValueError("Precision calibration changed nominal dimensions or storage roles")
                    args = _process_input_group(values, device, dtype)
                    row["input_signature"] = input_signature(args)
                    row["input_signature_sha256"] = fingerprint({"signature": row["input_signature"]})
                    row["input_probe_sha256"] = fingerprint({"probe": probe(args)})
                    reference = copy.deepcopy(model)
                    prepared = prepare(reference, args, spec=spec, problem_id=payload["problem_id"])
                    row["estimated_host_bytes"] = prepared.estimated_host_bytes
                    if prepared.estimated_host_bytes > settings["oracle_memory_mb"] * 1024**2:
                        raise ValueError("Actual stored precision exceeds the full-size oracle memory estimate")
                    seed(FIT["forward_seed"])
                    truth = prepared.execute()
                    row.update(oracle_receipt=prepared.receipt(), source_device=str(device))
                    seed(FIT["forward_seed"])
                    expected = clone_tensors(reference(*copy_inputs(args)))
                    synchronize(device)
                    expected = host(expected)
                    row["output_probe_sha256"] = fingerprint({"probe": probe(expected)})
                    row["reference_gate"] = precision_gate(expected, expected, truth, PrecisionContract(**policy))
                    row["signal"] = signal_audit(expected, absolute_tolerance=policy["element_atol"])
                    row["utilization"] = reference_utilization(expected, truth, PrecisionContract(**policy))
                    if stage == "check" and record["policy"] is not None:
                        row["fitted_gate"] = precision_gate(expected, expected, truth, PrecisionContract(**record["policy"]))
                        if not row["fitted_gate"]["passed"] or not row["fitted_gate"]["reference_qualified"]:
                            raise ValueError("Held-out reference failed its frozen fitted precision budget; no refit")
                    row["status"] = "passed"
                    _utilization(row, policy)
                    del raw, values, args, reference, prepared, expected, truth
                except Exception as exc:
                    row["status"] = "failed"
                    row["errors"].append(f"{type(exc).__name__}: {str(exc)[:1000]}")
                    row.setdefault("input_signature_sha256", row.get("raw_input_signature_sha256", fingerprint({"unavailable": True})))
            if stage == "fit":
                try:
                    record["factor"] = fit_factor(record["fit"], policy)
                    record["policy"] = fitted_policy(policy, record["factor"])
                    record["fit_kind"] = ("discrete_exact_only; no_float_threshold_fit" if all(row["utilization"]["floating_elements"] == 0 for row in record["fit"])
                                          else "floating_budget_fit")
                except ValueError:
                    pass
        if record["factor"] is not None:
            try:
                probes = [row.get("input_probe_sha256") for stage in ("fit", "check") for row in record[stage]]
                signatures = [row.get("input_signature_sha256") for stage in ("fit", "check") for row in record[stage]]
                raw_signatures = [row.get("raw_input_signature_sha256") for stage in ("fit", "check") for row in record[stage]]
                record["qualified"] = (len(set(signatures)) == 1 and len(set(raw_signatures)) == 1
                    and all(isinstance(v, str) and len(v) == 64 for v in probes) and len(set(probes)) == len(probes)
                    and all(heldout_passed(row, policy, record["factor"]) for row in record["check"]))
            except ValueError:
                pass
    record["environment_changes"] = sorted(set(changes))
    if changes:
        record["qualified"] = False
    record = json_safe(record)
    record["receipt_sha256"] = fingerprint(record)
    return {"status": "ok", "purpose": "reference_precision_calibration_only", "record": record,
            "candidate_executed": False, "timings_collected": False, "experiments_started": True}
