"""Reference-only input calibration; execution is an explicit device task."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager

from .common import evaluator_digest, fingerprint, json_safe
from .input_calibration import PROTOCOL, SEEDS, precision_policy, selected_attempt
from .input_profiles import recipes, transform
from .operator_specs import spec_for


def probe(value):
    """Hash bounded deterministic probes, not a claimed full input checksum."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel():
            count = min(256, value.numel())
            positions = torch.arange(count, dtype=torch.int64, device=value.device) * (value.numel() - 1) // max(1, count - 1)
            sample = torch.take(value, positions).detach().cpu()
            if sample.is_complex():
                sample = torch.view_as_real(sample)
            raw = sample.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        else:
            raw = b""
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sample_sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, (tuple, list)):
        return [probe(item) for item in value]
    if isinstance(value, dict):
        return {key: probe(item) for key, item in value.items()}
    return value


@contextmanager
def precision_environment(changes):
    import torch

    from src.validation import preserve_backend_settings

    from .evaluate_task import configure_tf32
    from .strict_validation import environment_guard

    with preserve_backend_settings():
        configure_tf32(torch, False)
        with environment_guard(changes):
            yield


def calibrate(payload, *, adapter=None, loader=None):
    import torch

    from src.eval import (
        _process_input_tensor,
        get_torch_dtype_from_string,
        load_original_model_and_inputs,
        move_model,
        set_seed,
    )
    from src.validation import clone_tensors, synchronize

    from .calibration import identity
    from .determinism import preserve_rng
    from .devices import adapter as device_adapter
    from .evaluate_task import validate_sources
    from .measurement import input_signature
    from .numerical import PrecisionContract, precision_gate, signal_audit
    from .operator_oracles import estimate_bytes, execute

    if "kernel_src" in payload or payload["level"] != 1 or payload["eval_config"].get("input_profile"):
        raise ValueError("Input calibration accepts original L1 references only, with no candidate or existing profile")
    source = validate_sources(payload)
    spec = spec_for(payload["level"], payload["problem_id"], source)
    settings = payload["eval_config"]
    simulated = adapter is not None or loader is not None
    adapter = adapter or device_adapter(settings.get("device_backend", "cuda"), "0")
    device, caps = adapter.device, adapter.caps()
    policy = precision_policy(spec, settings["precision"])
    record = {"protocol": PROTOCOL, "level": 1, "problem_id": payload["problem_id"], "precision": settings["precision"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "evaluator_sha256": evaluator_digest(),
        "device": caps, "device_identity_sha256": identity(caps), "precision_contract": policy,
        "simulated_execution": simulated or bool(getattr(adapter, "simulated", False)),
        "selection_basis": "reference_precision_and_signal_only", "candidate_executed": False, "timings_collected": False,
        "attempts": [], "selected_recipe": None, "qualified": False,
        "input_probe_scope": "up_to_256_deterministic_positions_per_tensor; not_a_full_content_hash",
        "oracle_memory_mb": settings["oracle_memory_mb"]}
    dtype = get_torch_dtype_from_string(settings["precision"])
    Model, init, inputs = (loader or load_original_model_and_inputs)(source, {})

    def seed(value):
        set_seed(value)
        if str(device).split(":")[0] == "neuron":
            torch.neuron.manual_seed(value)

    changes = []
    with torch.no_grad(), preserve_rng(device), precision_environment(changes):
        seed(settings.get("evaluation_seed", 42))
        initial = [_process_input_tensor(value, device, dtype) for value in init()]
        model = move_model(Model(*initial), device, dtype)
        for recipe in recipes(payload["problem_id"]):
            attempt = {"recipe": recipe, "cases": []}
            record["attempts"].append(attempt)
            for value in SEEDS:
                row = {"seed": value, "status": "not_measured", "errors": []}
                attempt["cases"].append(row)
                try:
                    seed(value)
                    raw = inputs()
                    raw_signature = input_signature(raw)
                    row["raw_input_signature_sha256"] = fingerprint({"signature": raw_signature})
                    estimate = estimate_bytes(payload["problem_id"], model, raw)
                    row["estimated_host_bytes"] = estimate
                    if estimate > settings["oracle_memory_mb"] * 1024**2:
                        row["input_signature_sha256"] = row["raw_input_signature_sha256"]
                        row["errors"].append("Original-size calibration exceeds the host-memory estimate envelope; no resizing")
                        continue
                    values = transform(raw, payload["problem_id"], recipe)
                    if input_signature(values) != raw_signature:
                        raise ValueError("Input calibration changed nominal shape, stride or storage roles")
                    args = [_process_input_tensor(item, device, dtype) for item in values]
                    row["input_signature_sha256"] = fingerprint({"signature": input_signature(args)})
                    row["input_probe_sha256"] = fingerprint({"probe": probe(args)})
                    truth = execute(payload["problem_id"], model, args)
                    reference = copy.deepcopy(model)
                    seed(value)
                    expected = clone_tensors(reference(*clone_tensors(args)))
                    synchronize(device)
                    def host(item):
                        if isinstance(item, torch.Tensor):
                            return item.detach().cpu().clone()
                        if isinstance(item, (tuple, list)):
                            return type(item)(host(child) for child in item)
                        if isinstance(item, dict):
                            return {key: host(child) for key, child in item.items()}
                        return item
                    expected = host(expected)
                    row["reference_gate"] = precision_gate(expected, clone_tensors(expected), truth, PrecisionContract(**policy))
                    row["signal"] = signal_audit(expected, absolute_tolerance=0.0)
                    row["status"] = "passed" if row["reference_gate"]["passed"] else "failed"
                    del raw, values, args, truth, expected, reference
                except Exception as exc:
                    row.setdefault("raw_input_signature_sha256", fingerprint({"unavailable": True}))
                    row.setdefault("input_signature_sha256", row["raw_input_signature_sha256"])
                    row["status"] = "failed"
                    row["errors"].append(f"{type(exc).__name__}: {str(exc)[:1000]}")
            winner = selected_attempt(record)
            if winner is not None:
                record.update(qualified=True, selected_recipe=recipe)
                break
    record["environment_changes"] = sorted(set(changes))
    if changes:
        record.update(qualified=False, selected_recipe=None)
    record = json_safe(record)
    record["receipt_sha256"] = fingerprint(record)
    return {"status": "ok", "purpose": "reference_input_calibration_only", "record": record,
            "experiments_started": True, "candidate_executed": False}
