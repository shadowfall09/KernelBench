"""Reviewed, source-bound work models and device-specific headroom scores."""

from __future__ import annotations

import hashlib
import math
import re
from statistics import mean

from ..common import ROOT, evaluator_digest, fingerprint, read_json, select_problems
from .calibration import identity, positive, validate

PROTOCOL = "reviewed_headroom_certificate_v1"


def validate_witness(witness, *, digest):
    from ..evaluation.final_verification import PROTOCOL as FINAL_PROTOCOL

    if not isinstance(witness, dict):
        raise ValueError("Malformed reachability witness")
    groups = witness.get("verification", [])
    if (len(groups) < 2 or witness.get("verification_receipt_sha256") != fingerprint({"groups": groups})
            or any(not re.fullmatch("[0-9a-f]{64}", witness.get(name, ""))
                   for name in ("source_sha256", "reference_sha256", "source_fingerprint", "reference_fingerprint"))):
        raise ValueError("Reachability requires source-bound independent final verification records")
    nonces, means = [], []
    for group in groups:
        for phase in ("standard", "hidden"):
            result = group.get(phase, {})
            options = result.get("metadata", {}).get("evaluation_config", {})
            if (result.get("compiled") is not True or result.get("correctness") is not True
                    or options.get("evaluator_sha256") != digest or options.get("final_protocol") != FINAL_PROTOCOL
                    or options.get("use_hidden_tests") != (phase == "hidden") or not options.get("final_nonce")
                    or result.get("metadata", {}).get("kernel_sha256") != witness.get("source_fingerprint")
                    or result.get("metadata", {}).get("reference_sha256") != witness.get("reference_fingerprint")
                    or result.get("metadata", {}).get("precision") != witness.get("precision")
                    or identity(result.get("metadata", {}).get("device_fingerprint", {})) != witness.get("device_identity_sha256")):
                raise ValueError("Reachability verification is missing a passing phase or frozen final identity")
            if phase == "standard":
                if result["metadata"].get("input_signature") != witness.get("input_signature"):
                    raise ValueError("Witness input signatures differ")
                samples = result.get("runtime_stats", {}).get("samples_ms", [])
                if len(samples) < 2 or not all(map(positive, samples)) or result.get("runtime") != mean(samples):
                    raise ValueError("Witness timings must match every independent repeat")
                nonces.append(options["final_nonce"])
                means.extend(samples)
            elif options["final_nonce"] != nonces[-1]:
                raise ValueError("Witness phases must belong to the same independent repeat")
    if len(set(nonces)) != len(nonces) or witness.get("runtime_ms") != mean(means):
        raise ValueError("Witness repeats are duplicated or their mean was changed")


def witness_from_run(config, pid, sid, calibration):
    """Export already-completed verifier results; never execute a workload."""
    from ..evaluation.final_verification import validate as validate_final

    if not config.hidden_tests or config.final_repeats < 2:
        raise ValueError("Reachability requires hidden tests and at least two independent final repeats")
    reference = select_problems(config.level, problem_ids=[pid])[pid]
    baselines = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {})
    baseline = baselines.get(f"level{config.level}", {}).get(reference.name, {})
    issue, records = validate_final(config, pid, sid, baseline)
    if issue:
        raise ValueError(f"Reachability final verification failed: {issue}")
    validate(calibration, records[0]["metadata"]["device_fingerprint"], digest=evaluator_digest())
    manifest = read_json(config.run_dir / "final_manifest.json")
    slot = manifest["slots"][f"{pid}_{sid}"]
    results = read_json(config.run_dir / "final_results.json")
    groups = [{phase: results[f"{pid}_{sid}:{repeat}:{phase}"] for phase in ("standard", "hidden")}
              for repeat in range(config.final_repeats)]
    witness = {"protocol": "independent_reachability_v1", "correctness": True, "hidden_correctness": True,
               "source_sha256": slot["source_sha256"], "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
               "source_fingerprint": records[0]["metadata"]["kernel_sha256"],
               "reference_fingerprint": records[0]["metadata"]["reference_sha256"],
               "device_identity_sha256": identity(calibration["device"]), "calibration_sha256": calibration["receipt_sha256"],
               "input_signature": records[0]["metadata"]["input_signature"], "precision": config.precision,
               "runtime_ms": mean(sample for record in records for sample in record["runtime_stats"]["samples_ms"]),
               "verification": groups, "verification_receipt_sha256": fingerprint({"groups": groups}),
               "provenance": "trusted_verifier_export; content_hashes_are_not_cryptographic_authentication"}
    validate_witness(witness, digest=evaluator_digest())
    return witness


def sol(work, calibration):
    """Resource demands may overlap; use their maximum, never their sum."""
    if not isinstance(work, dict) or not work:
        raise ValueError("A bound needs explicit work by resource")
    components = {}
    for resource, quantity in work.items():
        if resource == "critical_path_ms":
            if not positive(quantity):
                raise ValueError("Critical path must be positive")
            components[resource] = quantity
            continue
        probe = calibration.get("probes", {}).get(resource)
        if not positive(quantity) or not probe or not positive(probe.get("rate")):
            raise ValueError(f"Missing measured rate or positive work: {resource}")
        if probe.get("unit") != ("bytes/s" if resource == "memory_bytes" else "operations/s"):
            raise ValueError("Work and rate units do not match")
        components[resource] = quantity / probe["rate"] * 1000
    return max(components.values()), components


def certificate(record, calibration, metadata, *, source_sha256, digest=None):
    """Trust resides in the verifier-owned reviewed spec, not a candidate declaration."""
    digest = digest or evaluator_digest()
    if not isinstance(record, dict) or not isinstance(metadata, dict) or not re.fullmatch("[0-9a-f]{64}", source_sha256):
        raise ValueError("Malformed bound certificate or source digest")
    validate(calibration, metadata.get("device_fingerprint", {}), digest=digest)
    review = record.get("review", {})
    if (record.get("protocol") != PROTOCOL or record.get("layer") not in {"I", "F", "D"}
            or not review.get("reviewer") or not review.get("rationale")
            or record.get("reference_sha256") != source_sha256
            or record.get("precision") != metadata.get("precision")
            or record.get("input_signature") != metadata.get("input_signature")
            or record.get("calibration_sha256") != calibration.get("receipt_sha256")):
        raise ValueError("Missing review or certificate source/input/precision/calibration mismatch")
    reference_sol, reference_components = sol(record.get("reference_work"), calibration)
    algorithm_sol, algorithm_components = sol(record.get("algorithm_work"), calibration)
    if algorithm_sol > reference_sol * (1 + 1e-9):
        raise ValueError("Best-known algorithm model exceeds the reference model")
    design = None
    if record["layer"] == "D":
        design = record.get("design", {})
        if not isinstance(design, dict):
            raise ValueError("Malformed design certificate")
        witness = design.get("reachability", {})
        if not design.get("algorithm") or not design.get("comparison") or not design.get("numerical_contract"):
            raise ValueError("Design label requires algorithmic and numerical justification")
        # Witness is on an explicitly identified reference device; it may be a different device.
        witness_calibration = design.get("witness_calibration", calibration)
        validate(witness_calibration, witness_calibration.get("device", {}), digest=digest)
        witness_sol, _ = sol(record["algorithm_work"], witness_calibration)
        validate_witness(witness, digest=digest)
        if (witness.get("protocol") != "independent_reachability_v1"
                or witness.get("correctness") is not True or witness.get("hidden_correctness") is not True
                or witness.get("device_identity_sha256") != identity(witness_calibration["device"])
                or witness.get("calibration_sha256") != witness_calibration["receipt_sha256"]
                or witness.get("reference_sha256") != source_sha256
                or witness.get("input_signature") != record["input_signature"]
                or witness.get("precision") != record["precision"]
                or not witness.get("source_sha256") or not witness.get("verification_receipt_sha256")
                or not positive(witness.get("runtime_ms"))
                or not positive(design.get("maximum_witness_gap")) or design["maximum_witness_gap"] < 1
                or not 1 <= witness["runtime_ms"] / witness_sol <= design["maximum_witness_gap"]):
            raise ValueError("Design reachability witness is missing or does not reach the declared bound")
    return {"certificate_sha256": fingerprint(record), "calibration_sha256": calibration["receipt_sha256"],
            "layer": record["layer"], "sol_reference_ms": reference_sol, "sol_algorithm_ms": algorithm_sol,
            "sol_target_ms": algorithm_sol if record["layer"] == "D" else reference_sol,
            "reference_components_ms": reference_components, "algorithm_components_ms": algorithm_components,
            "design_reference_device_witnessed": design is not None,
            "target_device_witnessed": bool(design and design["reachability"]["device_identity_sha256"] == identity(calibration["device"])),
            "bound_kind": "reviewed_model_using_attained_rates", "physical_lower_bound_proven": False}


def score(reference, candidate, target):
    if not all(map(positive, (reference, candidate, target))):
        raise ValueError("Headroom scoring requires positive finite times")
    return max(0, min(1, math.log(reference / candidate) / max(math.log(reference / target), math.log(1.25))))


def evaluate(record, calibration, metadata, *, source_sha256, reference_ms, candidate_ms, digest=None):
    result = certificate(record, calibration, metadata, source_sha256=source_sha256, digest=digest)
    result.update(score=score(reference_ms, candidate_ms, result["sol_target_ms"]),
                  below_model_estimate=candidate_ms < result["sol_target_ms"],
                  below_model_policy="investigate_model; not_a_correctness_or_misconduct_verdict")
    return result
