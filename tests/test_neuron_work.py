"""Unscored serialization/Parquet/CPU fixtures; no Neuron or profiler execution."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import pytest
from test_input_calibration import CAPS
from test_neuron_tables import tables, write

from kb_workflows import calibration, neuron_tables, neuron_work, performance
from kb_workflows.common import ROOT, evaluator_digest, fingerprint
from kb_workflows.precision_calibration import numeric_class
from scripts.analyze_neuron_work import main


def caps():
    physical = {
        "name": "Trainium2",
        "neuron_device": 3,
        "physical_core": 12,
        "core_index_within_device": 0,
        "serial_number": "SIMULATED serialization fixture",
        "driver_version": "not an installed driver",
    }
    software = {"torch_neuronx": "schema fixture", "nki": "schema fixture", "neuronx_cc": "schema fixture"}
    return {
        **copy.deepcopy(CAPS),
        "name": "Trainium2",
        "device_backend": "neuron",
        "physical_device": physical,
        "software_versions": software,
        "uuid": "neuron:" + fingerprint(physical),
        "cpu_fallback_allowed": False,
        "driver_version": fingerprint({"loaded_driver": physical["driver_version"], "software": software}),
    }


def profile_result():
    signature = [{"shape": [2, 3], "dtype": "torch.bfloat16"}]
    call = {
        "seed": 42,
        "input_signature": signature,
        "training": True,
        "no_grad": True,
        "output_verified": True,
        "input_probe_sha256": "1" * 64,
        "registered_state_probe_sha256": "2" * 64,
        "preceding_cases_sha256": "3" * 64,
        "state_scope": neuron_work.SCOPE,
        "precision_policy": {
            "native_tolerance": 0.001,
            "production_contract": None,
            "independent_numerical": False,
            "oracle_spec_sha256": None,
        },
    }
    payload = {"task": "profile", "eval_config": {"precision": "bf16"}}
    result = {
        "metadata": {
            "profile_call": call,
            "device_fingerprint": caps(),
            "simulated_execution": True,
            "compiler_source_sha256": {
                "candidate.py": hashlib.sha256(b"unscored candidate source").hexdigest(),
                "reference.py": hashlib.sha256(b"unscored reference source").hexdigest(),
            },
        },
        "output_verified": True,
        "exporter": "NeuronProfiler.export_trace",
        "artifact_files": {"a.neff": "a" * 64, "b.neff": "b" * 64, "fixture.ntff": "c" * 64},
    }
    return payload, result


def device_calibration(device):
    probes = {
        "memory_bytes": calibration.probe_record(100_000, [100] * 5, unit="bytes/s", parameters={}),
        "bf16_flops": calibration.probe_record(20_000, [10] * 5, unit="operations/s", parameters={}),
    }
    return calibration.seal(
        copy.deepcopy(device), probes, [0.01] * 5, settings={"basis": "unscored schema fixture only"}
    )


def review(report, capture):
    return {
        "protocol": neuron_work.REVIEW,
        "coverage": "reviewed_complete_workload",
        "reviewer": "unscored fixture; not an actual hardware review",
        "rationale": "Explicit fixture pairing and raw arithmetic/HBM accounting only",
        "memory_accounting": neuron_work.MEMORY,
        "numeric_class_sha256": numeric_class(capture["device"]),
        "workload_contract": copy.deepcopy(capture["binding"]),
        "pairing": {
            "capture_receipt_sha256": capture["receipt_sha256"],
            "report_receipt_sha256": report["receipt_sha256"],
            "physical_device_sha256": fingerprint(capture["device"]["physical_device"]),
            "neuron_device_idx": 3,
            "neuroncore_idx": 0,
        },
        "rules": [
            {
                "engine": engine,
                "opcode": opcode,
                "dtypes": dict.fromkeys(neuron_work.DTYPES, dtype),
                "arithmetic_resource": resource,
            }
            for engine, opcode, dtype, resource in [
                ("tensor", "MATMUL", "BF16", "bf16_flops"),
                ("vector", "ADD", None, None),
            ]
        ],
    }


@pytest.fixture
def artifacts(tmp_path):
    data = tables()
    data["Metadata"][0].update(neuron_device_idx=3, neuroncore_idx=0)
    for i, row in enumerate(data["Instruction"]):
        row.update(
            raw_flops=[64, 0, 128][i],
            adjusted_flops=[64, 0, 128][i],
            hbm_read_bytes=[64, 0, 0][i],
            hbm_write_bytes=[64, 0, 0][i],
            **dict.fromkeys(neuron_work.DTYPES, "BF16" if i != 1 else None),
        )
    directory = write(tmp_path / "profile", data)
    report = neuron_tables.load(directory)
    payload, result = profile_result()
    capture = neuron_work.capture_record(payload, result)
    neuron_work.validate_capture(capture, allow_simulated=True)
    # Serialization fixture only. Never counted as a device experiment or saved as benchmark evidence.
    capture["simulated"] = False
    neuron_work.seal(capture)
    return directory, report, capture, review(report, capture), device_calibration(capture["device"])


def test_model_replays_complete_parquet_work_without_promoting_annotations_to_hardware_counters(artifacts):
    _, report, capture, policy, rates = artifacts
    model = neuron_work.execution_model(report, capture, policy, rates)
    assert model["work"] == {"memory_bytes": 128, "bf16_flops": 192}
    assert model["per_engine_work"] == {
        "tensor": {"memory_bytes": 128, "bf16_flops": 192},
        "vector": {"memory_bytes": 0},
    }
    assert model["components_ms"] == {"memory_bytes": 0.128, "bf16_flops": 0.096}
    assert model["sol_execution_ms"] == 0.128
    assert not model["hardware_counters_collected"] and not model["formal_origin_qualified"]
    assert "not_automatic_cross_capture_proof" in model["association"]
    assert neuron_work.validate_model(json.loads(json.dumps(model))) == model
    factors = performance.decomposition(1, 0.8, 0.001, counters=model)
    assert factors["A"] == 128 and factors["E"] == 6.25 and factors["AE"] == 800 and factors["O"] == 1.25
    assert factors["execution_model_replayed"] and not factors["hardware_counters_collected"]
    assert factors["work_evidence_kind"] == "reviewed_neuron_instruction_annotations"


@pytest.mark.parametrize(
    "damage",
    [
        "simulated",
        "output",
        "grad",
        "source",
        "input_probe",
        "seed",
        "policy",
        "uuid",
        "physical",
        "driver",
        "ntff",
        "neff",
        "duplicate_neff",
        "artifact_path",
    ],
)
def test_resealed_capture_cannot_borrow_unverified_calls_or_other_source_and_device_bindings(artifacts, damage):
    _, _, capture, _, _ = artifacts
    if damage == "simulated":
        capture["simulated"] = True
    elif damage == "output":
        capture["output_verified"] = False
    elif damage == "grad":
        capture["call"]["no_grad"] = False
    elif damage == "source":
        capture["binding"]["source_sha256"] = "0" * 64
    elif damage == "input_probe":
        capture["binding"]["input_probe_sha256"] = "0" * 64
    elif damage == "seed":
        capture["call"]["seed"] = True
    elif damage == "policy":
        capture["call"]["precision_policy"]["native_tolerance"] = 1
    elif damage == "uuid":
        capture["device"]["uuid"] = "borrowed"
    elif damage == "physical":
        capture["device"]["physical_device"]["physical_core"] += 1
    elif damage == "driver":
        capture["device"]["software_versions"]["nki"] = "changed"
    elif damage == "ntff":
        del capture["artifact_files"]["fixture.ntff"]
    elif damage == "neff":
        capture["artifact_files"] = {"fixture.ntff": "c" * 64}
    elif damage == "duplicate_neff":
        capture["artifact_files"]["other/a.neff"] = "d" * 64
    elif damage == "artifact_path":
        capture["artifact_files"]["../other.neff"] = "d" * 64
    neuron_work.seal(capture)
    with pytest.raises(ValueError):
        neuron_work.validate_capture(capture)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_work",
        "positive_unknown_dtype",
        "missing_rule",
        "duplicate_rule",
        "no_flop_resource",
        "partial",
        "source",
        "policy",
        "device_class",
        "pairing",
        "coordinates",
        "memory_policy",
        "absent_quality",
        "warnings",
        "errors",
        "simulator",
        "unknown_execution",
        "missing_neff",
        "calibration_device",
        "calibration_topology",
        "calibration_units",
        "calibration_samples",
    ],
)
def test_incomplete_or_repaired_review_does_not_enable_a_neuron_execution_model(artifacts, damage):
    _, report, capture, policy, rates = artifacts
    data = report["tables"]
    if damage == "missing_work":
        data["Instruction"][0]["raw_flops"] = None
    elif damage == "positive_unknown_dtype":
        data["Instruction"][0]["src1_dtype"] = None
        policy["rules"][0]["dtypes"]["src1_dtype"] = None
    elif damage == "missing_rule":
        policy["rules"].pop()
    elif damage == "duplicate_rule":
        policy["rules"].append(copy.deepcopy(policy["rules"][0]))
    elif damage == "no_flop_resource":
        policy["rules"][0]["arithmetic_resource"] = None
    elif damage == "partial":
        policy["coverage"] = "reported_resources_only"
    elif damage == "source":
        policy["workload_contract"]["source_sha256"] = "0" * 64
    elif damage == "policy":
        policy["workload_contract"]["policy_sha256"] = "0" * 64
    elif damage == "device_class":
        policy["numeric_class_sha256"] = "0" * 64
    elif damage == "pairing":
        policy["pairing"]["capture_receipt_sha256"] = "0" * 64
    elif damage == "coordinates":
        data["Metadata"][0]["neuroncore_idx"] = None
    elif damage == "memory_policy":
        policy["memory_accounting"] = "add all DMA traffic to HBM"
    elif damage == "absent_quality":
        del data["Warning"]
        del report["files"]["Warning.parquet"]
    elif damage in {"warnings", "errors"}:
        name = "Warning" if damage == "warnings" else "Error"
        data[name] = [{"message": "incomplete profile"}]
        report["files"][name + ".parquet"].update(rows=1, columns=["message"])
    elif damage == "simulator":
        data["Metadata"][0]["capture_type"] = "perf-sim"
    elif damage == "unknown_execution":
        data["ExecutionInfo"][1]["execution_start_ts"] = 15
    elif damage == "missing_neff":
        data["ExecutionInfo"][1]["neff_name"] = "not_captured.neff"
        data["NkiKernelCallSites"][1]["neff_name"] = "not_captured.neff"
    elif damage == "calibration_device":
        rates["device"]["uuid"] = "other core"
        rates["device_identity_sha256"] = calibration.identity(rates["device"])
        neuron_work.seal(rates)
    elif damage == "calibration_topology":
        rates["device"]["physical_device"]["physical_core"] += 1
        neuron_work.seal(rates)
    elif damage == "calibration_units":
        rates["probes"]["memory_bytes"]["unit"] = "operations/s"
        neuron_work.seal(rates)
    elif damage == "calibration_samples":
        rates["probes"]["bf16_flops"]["samples_ms"][0] = 1
        rates["probes"]["bf16_flops"]["samples_ms"][1] = 1
        rates["probes"]["bf16_flops"]["samples_ms"][2] = 1
        neuron_work.seal(rates)
    report.update(neuron_tables.analyze(data))
    neuron_work.seal(report)
    policy["pairing"]["report_receipt_sha256"] = report["receipt_sha256"]
    with pytest.raises(ValueError):
        neuron_work.execution_model(report, capture, policy, rates)


@pytest.mark.parametrize(
    "field", ["work", "per_engine_work", "sol_execution_ms", "components_ms", "hardware_counters_collected"]
)
def test_resealed_models_cannot_invent_execution_work_or_promote_counter_status(artifacts, field):
    _, report, capture, policy, rates = artifacts
    model = neuron_work.execution_model(report, capture, policy, rates)
    model[field] = True if field == "hardware_counters_collected" else 123
    neuron_work.seal(model)
    with pytest.raises(ValueError):
        neuron_work.validate_model(model)
    factors = performance.decomposition(1, 0.8, 0.001, counters=model)
    assert factors["A"] is None and factors["E"] is None and factors["counter_issue"]
    assert not factors["hardware_counters_collected"] and not factors["execution_model_replayed"]


def test_optional_cli_builds_only_from_existing_reviewed_files_and_never_launches_a_tool(
    artifacts, tmp_path, monkeypatch
):
    directory, report, capture, policy, rates = artifacts
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("offline model launched a process"))
    argv = [str(directory), "--output", str(tmp_path / "model.json")]
    for key, value in (("capture", {"capture_receipt": capture}), ("review", policy), ("calibration", rates)):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        argv += ["--" + key, str(path)]
    assert main(argv) == 0
    result = json.loads((tmp_path / "model.json").read_text())
    assert neuron_work.validate_model(result)["work"] == {"memory_bytes": 128, "bf16_flops": 192}
    assert (
        main(
            [str(directory), "--output", str(tmp_path / "no-output.json"), "--capture", str(tmp_path / "capture.json")]
        )
        == 2
    )
    assert not (tmp_path / "no-output.json").exists()


@pytest.mark.parametrize("failure", ["missing_review", "partial_review", "simulated_capture"])
def test_failed_model_keeps_the_complete_raw_report_without_launching_a_tool(artifacts, tmp_path, monkeypatch, failure):
    directory, report, capture, policy, rates = artifacts
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("offline failure launched a process"))
    if failure == "partial_review":
        policy["coverage"] = "reported_resources_only"
    elif failure == "simulated_capture":
        capture["simulated"] = True
        neuron_work.seal(capture)
    argv = [str(directory), "--output", str(tmp_path / "failure.json")]
    for name, value in (("capture", capture), ("review", policy), ("calibration", rates)):
        path = tmp_path / (name + ".json")
        if failure != "missing_review" or name != "review":
            path.write_text(json.dumps(value))
        argv += ["--" + name, str(path)]
    assert main(argv) == 2
    saved = json.loads((tmp_path / "failure.json").read_text())
    assert saved["execution_model"]["status"] == "not_measured" and saved["execution_model"]["reason"]
    assert neuron_tables.validate(saved["report"]) == report


@pytest.mark.parametrize("declared", [0, 99, True])
def test_explicit_execution_identity_must_exist_and_contain_the_instruction(declared):
    data = tables()
    data["ExecutionInfo"][1]["execution_start_ts"] = 15
    data["Instruction"][1]["execution_index"] = declared
    if declared == 0 and type(declared) is int:
        report = neuron_tables.analyze(data)
        assert report["joins"][1]["execution_index"] == 0 and report["unassigned_instruction_count"] == 0
    else:
        with pytest.raises(ValueError):
            neuron_tables.analyze(data)
    data["Instruction"][1]["execution_index"] = 1
    data["Instruction"][1]["start_ts"] = 0
    data["Instruction"][1]["duration_ns"] = data["Instruction"][1]["end_ts"]
    with pytest.raises(ValueError, match="timestamps contradict"):
        neuron_tables.analyze(data)


def test_model_import_stays_sdk_and_arrow_light():
    code = 'import sys; import kb_workflows.neuron_work; assert not {"torch","torch_neuronx","pyarrow","transformers","triton"} & sys.modules.keys()'
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert neuron_work.evaluator_digest() == evaluator_digest()
