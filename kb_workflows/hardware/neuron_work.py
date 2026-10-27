"""Reviewed Neuron instruction-demand models, distinct from hardware counters.

The reviewer supplies resource rules. Report/capture binding is either explicit
manual review or replay of a verifier-owned offline Explorer processing record.
Neither path establishes kernel ownership or equivalence to native timing.
"""

from __future__ import annotations

import copy
from pathlib import PurePosixPath

from ..analysis.calibration import identity, positive
from ..common import evaluator_digest, fingerprint
from ..evaluation.state_paths import INVOCATION_SCOPE
from ..provenance.work_counters import sha
from .neuron_tables import integer

CAPTURE = "kernelbench_pro_neuron_verified_capture_v2"
REVIEW = "kernelbench_pro_neuron_work_review_v1"
MODEL = "kernelbench_pro_neuron_instruction_model_v1"
SCOPE = "continuous_state; full_candidate_state_and_input_aliases_before_and_after_profile"
MEMORY = "instruction_hbm_annotations_only; DMA_packets_not_added"
DTYPES = ("src1_dtype", "src2_dtype", "dst_dtype")
RESOURCES = {"fp32_flops", "fp16_flops", "bf16_flops"}
HASHES = (
    "source_sha256",
    "reference_sha256",
    "evaluation_contract_sha256",
    "input_signature_sha256",
    "input_content_sha256",
    "input_layout_sha256",
    "output_content_sha256",
    "model_state_before_sha256",
    "model_state_after_sha256",
    "preceding_cases_sha256",
    "policy_sha256",
)


def seal(value):
    value["receipt_sha256"] = fingerprint({key: item for key, item in value.items() if key != "receipt_sha256"})
    return value


def capture_record(payload, result):
    """Called after the worker has checked its frozen compiler-source files."""
    metadata, settings = result["metadata"], payload["eval_config"]
    call, sources = metadata["profile_call"], metadata["compiler_source_sha256"]
    binding = {
        key: call[key] for key in ("input_content_sha256", "input_layout_sha256", "output_content_sha256", "preceding_cases_sha256")
    }
    binding.update({f"model_state_{side}_sha256": call[f"model_state_{side}"].get("sha256") for side in ("before", "after")})
    binding.update(
        source_sha256=sources["candidate.py"],
        reference_sha256=sources["reference.py"],
        evaluation_contract_sha256=fingerprint({"settings": settings}),
        precision=settings["precision"],
        input_signature_sha256=fingerprint({"signature": call["input_signature"]}),
        policy_sha256=fingerprint(call["precision_policy"]),
    )
    return seal(
        copy.deepcopy(
            {
                "protocol": CAPTURE,
                "evaluator_sha256": evaluator_digest(),
                "binding": binding,
                "device": metadata["device_fingerprint"],
                "neuron_core_allocation": metadata.get("neuron_core_allocation"),
                "call": call,
                "compiler_source_sha256": sources,
                "artifact_files": result.get("artifact_files", {}),
                "artifact_file_bytes": result.get("artifact_file_bytes"),
                "output_verified": result.get("output_verified") is True,
                "exporter": result.get("exporter"),
                "simulated": metadata["simulated_execution"],
                "state_scope": SCOPE,
                "scored_wall_times_replaced": False,
                "formal_origin_qualified": False,
            }
        )
    )


def validate_capture(value, *, allow_simulated=False):
    if (
        not isinstance(value, dict)
        or value.get("protocol") != CAPTURE
        or value.get("evaluator_sha256") != evaluator_digest()
        or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
        or (value.get("simulated") is not False and not allow_simulated)
        or value.get("output_verified") is not True
        or value.get("exporter") != "NeuronProfiler.export_trace"
        or value.get("state_scope") != SCOPE
        or value.get("scored_wall_times_replaced") is not False
        or value.get("formal_origin_qualified") is not False
    ):
        raise ValueError("Neuron capture is missing, stale, changed, unverified or simulated")
    device, binding, call = value.get("device"), value.get("binding"), value.get("call")
    identity(device)
    physical = device.get("physical_device")
    if (
        device.get("device_backend") != "neuron"
        or device.get("cpu_fallback_allowed") is not False
        or not isinstance(physical, dict)
        or not integer(physical.get("physical_core"))
        or not integer(physical.get("neuron_device"))
        or not integer(physical.get("core_index_within_device"))
        or not isinstance(physical.get("serial_number"), str)
        or not physical["serial_number"]
        or not isinstance(physical.get("driver_version"), str)
        or not physical["driver_version"]
        or device.get("uuid") != "neuron:" + fingerprint(physical)
        or device.get("name") != physical.get("name")
        or not isinstance(device.get("software_versions"), dict)
        or device.get("driver_version")
        != fingerprint({"loaded_driver": physical["driver_version"], "software": device["software_versions"]})
        or not isinstance(binding, dict)
        or any(not sha(binding.get(key)) for key in HASHES)
        or binding.get("precision") not in {"fp32", "fp16", "bf16"}
        or not isinstance(call, dict)
        or not integer(call.get("seed"))
        or call["seed"] >= 2**32
        or type(call.get("training")) is not bool
        or call.get("no_grad") is not True
        or call.get("output_verified") is not True
        or call.get("state_scope") != SCOPE
        or not isinstance(call.get("input_signature"), list)
        or not isinstance(call.get("precision_policy"), dict)
        or binding["policy_sha256"] != fingerprint(call["precision_policy"])
        or binding["input_signature_sha256"] != fingerprint({"signature": call["input_signature"]})
        or any(
            call.get(key) != binding[key]
            for key in HASHES
            if key in {"input_content_sha256", "input_layout_sha256", "output_content_sha256", "preceding_cases_sha256"}
        )
        or any(
            not isinstance(state := call.get(f"model_state_{side}"), dict)
            or state.get("status") != "observed"
            or state.get("scope") != INVOCATION_SCOPE
            or not integer(state.get("tensor_count"))
            or state.get("sha256") != binding[f"model_state_{side}_sha256"]
            for side in ("before", "after")
        )
        or value.get("compiler_source_sha256")
        != {"candidate.py": binding["source_sha256"], "reference.py": binding["reference_sha256"]}
    ):
        raise ValueError("Neuron capture lacks consistent native source/input/state/policy/device bindings")
    files = value.get("artifact_files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError("Neuron capture lacks its complete native artifact inventory")
    basenames = {}
    for name, checksum in files.items():
        if not isinstance(name, str) or not name or not sha(checksum):
            raise ValueError("Invalid Neuron native artifact identity")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("Neuron artifact path is outside its capture")
        if path.suffix == ".neff":
            if path.name in basenames:
                raise ValueError("Ambiguous NEFF basenames cannot associate executions")
            basenames[path.name] = checksum
    if not basenames or not any(PurePosixPath(name).suffix == ".ntff" for name in files):
        raise ValueError("Neuron capture has no native NEFF/NTFF artifacts")
    sizes = value.get("artifact_file_bytes")
    if sizes is not None and (
        not isinstance(sizes, dict) or set(sizes) != set(files) or any(not integer(v) for v in sizes.values())
    ):
        raise ValueError("Neuron capture has an inconsistent artifact byte inventory")
    return value


def review_valid(review, report, capture, processing=None):
    from ..numerics.precision_calibration import numeric_class

    pairing = (
        {"processing_receipt_sha256": processing["receipt_sha256"]}
        if processing is not None
        else {
            "capture_receipt_sha256": capture["receipt_sha256"],
            "report_receipt_sha256": report["receipt_sha256"],
            "physical_device_sha256": fingerprint(capture["device"]["physical_device"]),
            **{key: report["metadata"].get(key) for key in ("neuron_device_idx", "neuroncore_idx")},
        }
    )
    if (
        not isinstance(review, dict)
        or review.get("protocol") != REVIEW
        or review.get("coverage") != "reviewed_complete_workload"
        or any(not isinstance(review.get(k), str) or not review[k].strip() for k in ("reviewer", "rationale"))
        or review.get("memory_accounting") != MEMORY
        or review.get("numeric_class_sha256") != numeric_class(capture["device"])
        or review.get("workload_contract") != capture["binding"]
        or review.get("pairing") != pairing
        or any(not integer(report["metadata"].get(key)) for key in ("neuron_device_idx", "neuroncore_idx"))
    ):
        raise ValueError("Neuron work needs reviewed report/capture pairing, workload, device class and memory policy")
    rules = review.get("rules")
    if not isinstance(rules, list) or not 1 <= len(rules) <= 256:
        raise ValueError("Neuron work review has no bounded instruction mapping")
    selectors = {}
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"engine", "opcode", "dtypes", "arithmetic_resource"}
            or any(not isinstance(rule[k], str) or not rule[k] for k in ("engine", "opcode"))
            or not isinstance(rule["dtypes"], dict)
            or set(rule["dtypes"]) != set(DTYPES)
            or any(v is not None and (not isinstance(v, str) or not v) for v in rule["dtypes"].values())
            or rule["arithmetic_resource"] not in RESOURCES | {None}
        ):
            raise ValueError("Neuron work mapping requires exact engine/opcode/dtypes and raw-FLOP resource")
        key = rule["engine"], rule["opcode"], *(rule["dtypes"][key] for key in DTYPES)
        if key in selectors:
            raise ValueError("Neuron instruction has duplicate or conflicting work rules")
        selectors[key] = rule["arithmetic_resource"]
    return selectors


def execution_model(report, capture, review, calibration, *, processing=None):
    from ..analysis.calibration import validate as validate_calibration
    from ..analysis.headroom import sol
    from .neuron_tables import validate as validate_report

    validate_report(report)
    validate_capture(capture)
    if processing is not None:
        from .neuron_processing import validate as validate_processing

        validate_processing(processing, report, capture)
    validate_calibration(calibration, capture["device"])
    if any(
        calibration["device"].get(key) != capture["device"].get(key)
        for key in ("device_backend", "physical_device", "software_versions")
    ):
        raise ValueError("Neuron calibration topology/software differs from the verified capture")
    selectors = review_valid(review, report, capture, processing)
    if (
        report["capture_declares_hardware"] is not True
        or report["unassigned_instruction_count"]
        or report["quality_rows"] != {"Error": [], "Warning": []}
    ):
        raise ValueError(
            "Neuron model requires hardware-declared, completely assigned work and explicit empty quality tables"
        )
    neffs = {PurePosixPath(name).name for name in capture["artifact_files"] if PurePosixPath(name).suffix == ".neff"}
    if any(row.get("neff_name") not in neffs for row in report["tables"]["ExecutionInfo"]):
        raise ValueError("Observed execution has no corresponding captured native executable")
    work, engines = {"memory_bytes": 0}, {}
    for row in report["tables"]["Instruction"]:
        selector = row["engine"], row["opcode"], *(row.get(key) for key in DTYPES)
        if selector not in selectors:
            raise ValueError("Neuron instruction has no unique reviewed work mapping")
        if any(not integer(row.get(key)) for key in ("raw_flops", "hbm_read_bytes", "hbm_write_bytes")):
            raise ValueError("Null or missing instruction work cannot become zero demand")
        resource = selectors[selector]
        if row["raw_flops"] and (resource is None or any(row.get(key) is None for key in DTYPES)):
            raise ValueError("Positive Neuron FLOPs need explicit storage dtypes and calibrated arithmetic resource")
        engine = engines.setdefault(row["engine"], {"memory_bytes": 0})
        memory = row["hbm_read_bytes"] + row["hbm_write_bytes"]
        engine["memory_bytes"] += memory
        work["memory_bytes"] += memory
        if resource is not None:
            engine[resource] = engine.get(resource, 0) + row["raw_flops"]
            work[resource] = work.get(resource, 0) + row["raw_flops"]
    demands = {key: value for key, value in work.items() if value > 0}
    for name in demands:
        unit = "bytes/s" if name == "memory_bytes" else "operations/s"
        if calibration.get("probes", {}).get(name, {}).get("unit") != unit:
            raise ValueError("Neuron instruction work has no matching calibrated resource units")
    latency, components = sol(demands, calibration)
    if not positive(latency):
        raise ValueError("Neuron instruction model has no positive calibrated demand")
    return seal(
        copy.deepcopy(
            {
                "protocol": MODEL,
                "report": report,
                "capture": capture,
                "review": review,
                "calibration": calibration,
                "processing": processing,
                "work": work,
                "per_engine_work": engines,
                "sol_execution_ms": latency,
                "components_ms": components,
                "work_evidence_kind": "reviewed_neuron_instruction_annotations",
                "hardware_counters_collected": False,
                "automatic_capture_report_binding": processing is not None,
                "association": (
                    "verified_Explorer_view_of_complete_capture_snapshot; not_source_ownership_or_native_equivalence_proof"
                    if processing is not None
                    else "trusted_reviewer_pairs_complete_report_and_verifier_capture; not_automatic_cross_capture_proof"
                ),
                "formal_origin_qualified": False,
                "scored_wall_times_replaced": False,
                "interpretation": "reviewed_resource_demand_over_attained_rates; not_a_physical_lower_bound",
            }
        )
    )


def validate_model(value):
    if (
        not isinstance(value, dict)
        or value.get("protocol") != MODEL
        or value.get("receipt_sha256")
        != fingerprint({key: item for key, item in value.items() if key != "receipt_sha256"})
    ):
        raise ValueError("Neuron execution model is missing or changed")
    expected = execution_model(
        value.get("report"),
        value.get("capture"),
        value.get("review"),
        value.get("calibration"),
        processing=value.get("processing"),
    )
    if expected != value:
        raise ValueError("Neuron execution work or SOL differs from its complete evidence")
    return expected
