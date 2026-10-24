"""Software protocol tests only: no device calibration or research experiments."""

import copy
import math
from contextlib import nullcontext

import pytest

from kb_workflows import calibration, headroom, performance
from kb_workflows.common import fingerprint
from kb_workflows.final_verification import PROTOCOL as FINAL_PROTOCOL


def test_headroom_config_paths_and_explicit_missing_evidence(tmp_path):
    from kb_workflows.config import parse_config

    (tmp_path / "calibration.json").write_text("{}")
    (tmp_path / "specs.json").write_text("{}")
    path = tmp_path / "run.toml"
    path.write_text('[run]\nscore_mode = "headroom"\nproblem_specs = "specs.json"\ndevice_calibration = "calibration.json"\n')
    config = parse_config(["--config", str(path), "--stage", "analyze"])
    assert config.device_calibration == str(tmp_path / "calibration.json")
    assert config.problem_specs == str(tmp_path / "specs.json")
    with pytest.raises(ValueError, match="Headroom scoring requires"):
        parse_config(["--score-mode", "headroom"])


def test_harbor_exports_models_only_to_verifier(tmp_path):
    from kb_workflows.config import RunConfig
    from kb_workflows.harbor import export_tasks

    specs, measured = tmp_path / "specs.json", tmp_path / "calibration.json"
    specs.write_text('{"level1": {"1": {"layer": "I"}, "2": {"layer": "D"}}}')
    measured.write_text('{"fixture": "not a real calibration"}')
    config = RunConfig(run_name="harbor-headroom", runs_dir=str(tmp_path), start=1, end=1, hardware="test", gpu_arch="Ampere",
                       stage="export-harbor", score_mode="headroom", problem_specs=str(specs), device_calibration=str(measured)).validate()
    target = export_tasks(config) / "l1-p1-s0"
    from kb_workflows.common import read_json

    child = read_json(target / "tests/config.json")["config"]
    assert child["problem_specs"] == "/tests/problem_specs.json"
    assert child["device_calibration"] == "/tests/device_calibration.json"
    assert read_json(target / "tests/problem_specs.json") == {"level1": {"1": {"layer": "I"}}}
    assert not (target / "environment/device_calibration.json").exists()


def test_device_driver_is_matched_by_physical_uuid_without_running_a_probe(monkeypatch):
    from types import SimpleNamespace

    import torch

    from kb_workflows import measurement

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name="test", total_memory=10, multi_processor_count=1, L2_cache_size=1, uuid="GPU-right"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(measurement.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="GPU-wrong, 999\nGPU-right, 580.1\n"))
    assert measurement.device_fingerprint("cuda:0")["driver_version"] == "580.1"
    monkeypatch.setattr(measurement.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    assert measurement.device_fingerprint("cuda:0")["driver_version"] is None


@pytest.fixture
def model():
    device = {"available": True, "uuid": "physical-test-device", "name": "test", "capability": [9, 0],
              "total_memory": 10**10, "multiprocessors": 10, "l2_bytes": 8, "torch_version": "test",
              "cuda_version": "test", "hip_version": None, "driver_version": "test"}
    probes = {"memory_bytes": calibration.probe_record(1000, [1] * 5, unit="bytes/s", parameters={}),
              "fp32_flops": calibration.probe_record(2000, [1] * 5, unit="operations/s", parameters={})}
    measured = calibration.seal(device, probes, [0.01] * 5, settings={"fixture": True}, digest="current")
    metadata = {"device_fingerprint": device, "precision": "fp32", "input_signature": [{"shape": [4]}]}
    spec = {"protocol": headroom.PROTOCOL, "layer": "I", "reference_sha256": "a" * 64,
            "precision": "fp32", "input_signature": metadata["input_signature"],
            "review": {"reviewer": "test-only", "rationale": "Protocol fixture, not a measured task certificate"},
            "calibration_sha256": measured["receipt_sha256"], "reference_work": {"memory_bytes": 1000, "fp32_flops": 2000},
            "algorithm_work": {"memory_bytes": 500, "fp32_flops": 1000}}
    return measured, metadata, spec


@pytest.mark.parametrize("damage", ["uuid", "software", "raw", "rate", "count", "unit", "launch", "unrun", "stale"])
def test_calibration_rejects_missing_or_inconsistent_measurements(model, damage):
    record, metadata, _ = copy.deepcopy(model)
    if damage == "uuid":
        metadata["device_fingerprint"]["uuid"] = "another-device"
    elif damage == "software":
        metadata["device_fingerprint"]["torch_version"] = "changed"
    elif damage == "raw":
        record["probes"]["memory_bytes"]["samples_ms"] = [2] * 5
    elif damage == "rate":
        record["probes"]["memory_bytes"]["rate"] *= 2
    elif damage == "count":
        record["probes"]["memory_bytes"]["samples_ms"] = [1]
    elif damage == "unit":
        record["probes"]["memory_bytes"]["unit"] = "GB/s"
    elif damage == "launch":
        record["launch_median_ms"] = 100
    elif damage == "unrun":
        record["experiments_run"] = False
    elif damage == "stale":
        record["evaluator_sha256"] = "old"
    # Recomputing a content hash alone cannot excuse inconsistent raw evidence.
    record["receipt_sha256"] = fingerprint({k: v for k, v in record.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError):
        calibration.validate(record, metadata["device_fingerprint"], digest="current")


def test_calibration_refuses_missing_physical_identity(model):
    record, metadata, _ = model
    assert calibration.validate(record, metadata["device_fingerprint"], digest="current") == record
    del metadata["device_fingerprint"]["uuid"]
    with pytest.raises(ValueError, match="physical device UUID"):
        calibration.identity(metadata["device_fingerprint"])


def test_calibration_runner_with_cpu_allocations_and_fake_device_timer(model, monkeypatch):
    import torch

    from src.validation import preserve_backend_settings

    adapter = calibration.CUDADevice.__new__(calibration.CUDADevice)
    adapter.device = torch.device("cpu")
    monkeypatch.setattr(adapter, "caps", lambda: model[1]["device_fingerprint"])
    calls = []
    def fake_timer(function, **kwargs):
        calls.append(function().shape)
        return [1] * kwargs["trials"]
    monkeypatch.setattr(adapter, "time", fake_timer)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10**9, 10**9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    from kb_workflows import measurement
    ticks = iter(range(100))
    monkeypatch.setattr(measurement, "CLOCK", lambda: next(ticks) / 1000)
    monkeypatch.setattr(calibration, "evaluator_digest", lambda: "current")
    with preserve_backend_settings():
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        measured = adapter.calibrate(matrix_sizes=[128], memory_bytes=64, trials=5)
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert len(calls) == 5
    assert set(measured["probes"]) == {"memory_bytes", "fp32_flops", "fp16_flops", "fp64_flops", "bf16_flops"}
    assert measured["probes"]["memory_bytes"]["work_per_call"] == 128
    assert measured["interpretation"].startswith("attained_microbenchmark_rates")


def test_cuda_event_adapter_sequence_with_fake_events(monkeypatch):
    import torch

    calls = []
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
        def record(self):
            calls.append("event")
        def synchronize(self):
            calls.append("end_sync")
        def elapsed_time(self, other):
            return 0.125
    adapter = calibration.CUDADevice.__new__(calibration.CUDADevice)
    adapter.device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("device_sync"))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    samples = adapter.time(lambda: calls.append("kernel"), trials=5, warmup=2)
    assert samples == [0.125] * 5
    assert calls == ["device_sync", "event", "kernel", "event", "end_sync"] * 7


@pytest.mark.parametrize("damage", ["source", "shape", "precision", "calibration", "review", "unknown_resource", "algorithm"])
def test_certificates_bind_context_and_review(model, damage):
    measured, metadata, spec = copy.deepcopy(model)
    if damage == "source":
        spec["reference_sha256"] = "b" * 64
    elif damage == "shape":
        spec["input_signature"] = [{"shape": [8]}]
    elif damage == "precision":
        spec["precision"] = "fp16"
    elif damage == "calibration":
        spec["calibration_sha256"] = "different"
    elif damage == "review":
        spec["review"] = {}
    elif damage == "unknown_resource":
        spec["reference_work"]["unmeasured_int8_ops"] = 1000
    else:
        spec["algorithm_work"]["memory_bytes"] = 10**9
    with pytest.raises(ValueError):
        headroom.certificate(spec, measured, metadata, source_sha256="a" * 64, digest="current")


def test_resource_maximum_headroom_formula_and_below_model_policy(model):
    measured, metadata, spec = model
    result = headroom.evaluate(spec, measured, metadata, source_sha256="a" * 64,
                               reference_ms=100, candidate_ms=10, digest="current")
    assert result["sol_reference_ms"] == 1  # Concurrent resources: max, not sum.
    assert result["score"] == pytest.approx(0.5)
    assert result["physical_lower_bound_proven"] is False
    assert headroom.score(100, 200, 1) == 0
    assert headroom.score(100, 0.1, 1) == 1
    result = headroom.evaluate(spec, measured, metadata, source_sha256="a" * 64,
                               reference_ms=100, candidate_ms=0.1, digest="current")
    assert result["below_model_estimate"] is True
    assert result["below_model_policy"].startswith("investigate_model")


def design_witness(measured, metadata):
    groups = []
    for repeat in range(2):
        group = {}
        for phase in ("standard", "hidden"):
            group[phase] = {"compiled": True, "correctness": True, "runtime": 1,
                            "runtime_stats": {"samples_ms": [1, 1]},
                            "metadata": {**copy.deepcopy(metadata), "kernel_sha256": "d" * 64,
                                         "reference_sha256": "e" * 64,
                                         "evaluation_config": {"evaluator_sha256": "current", "final_protocol": FINAL_PROTOCOL,
                                                               "final_nonce": f"nonce-{repeat}", "use_hidden_tests": phase == "hidden"}}}
        groups.append(group)
    return {"protocol": "independent_reachability_v1", "correctness": True, "hidden_correctness": True,
            "device_identity_sha256": calibration.identity(measured["device"]),
            "calibration_sha256": measured["receipt_sha256"], "reference_sha256": "a" * 64,
            "source_sha256": "c" * 64, "source_fingerprint": "d" * 64,
            "reference_fingerprint": "e" * 64, "input_signature": metadata["input_signature"],
            "precision": "fp32", "runtime_ms": 1, "verification": groups,
            "verification_receipt_sha256": fingerprint({"groups": groups})}


@pytest.mark.parametrize("damage", ["none", "missing", "false_correctness", "duplicated", "wrong_source", "slow", "summary"])
def test_design_requires_independent_reachability(model, damage):
    measured, metadata, spec = copy.deepcopy(model)
    spec["layer"] = "D"
    witness = design_witness(measured, metadata)
    spec["design"] = {"algorithm": "test algorithm", "comparison": "reviewed comparison",
                      "numerical_contract": "test fp32", "maximum_witness_gap": 3, "reachability": witness}
    if damage == "missing":
        witness["verification"] = []
    elif damage == "false_correctness":
        witness["verification"][1]["hidden"]["correctness"] = False
    elif damage == "duplicated":
        witness["verification"][1] = copy.deepcopy(witness["verification"][0])
    elif damage == "wrong_source":
        witness["verification"][1]["standard"]["metadata"]["kernel_sha256"] = "another-source"
    elif damage == "slow":
        for group in witness["verification"]:
            group["standard"].update(runtime=20, runtime_stats={"samples_ms": [20, 20]})
        witness["runtime_ms"] = 20
    elif damage == "summary":
        witness["runtime_ms"] = 0.7
    witness["verification_receipt_sha256"] = fingerprint({"groups": witness["verification"]})
    if damage == "none":
        result = headroom.certificate(spec, measured, metadata, source_sha256="a" * 64, digest="current")
        assert result["sol_target_ms"] == 0.5 and result["target_device_witnessed"]
    else:
        with pytest.raises(ValueError):
            headroom.certificate(spec, measured, metadata, source_sha256="a" * 64, digest="current")


def test_overlapping_timelines_and_conservative_origin_shares():
    trace = {"deviceProperties": [{"id": 0}], "traceEvents": [
        {"ph": "X", "name": "kernelbench_profile", "ts": 0, "dur": 100},
        {"ph": "X", "name": "custom", "cat": "kernel", "ts": 10, "dur": 50, "args": {"device": 0, "correlation": 1}},
        {"ph": "X", "name": "library", "cat": "kernel", "ts": 40, "dur": 40, "args": {"device": 0, "correlation": 2}},
        {"ph": "X", "name": "cudaLaunchKernel", "cat": "cuda_runtime", "ts": 0, "dur": 100, "args": {"device": 0}},
        {"ph": "X", "name": "other", "cat": "kernel", "ts": 0, "dur": 100, "args": {"device": 1}}]}
    result = performance.timeline(trace, device=0, origins={"1": "custom", "2": "library"})
    assert result["device_busy_ms"] == pytest.approx(0.07)
    assert result["kernel_count"] == 2
    assert result["custom_exclusive_fraction"] == pytest.approx(30 / 70)
    assert performance.timeline(trace, device=0)["unknown_origin_count"] == 2


@pytest.mark.parametrize("interval", [(math.nan, 2), (1, math.inf), (-1, 2), (2, 1), (True, 2)])
def test_invalid_intervals_fail(interval):
    with pytest.raises(ValueError):
        performance.interval_union([interval])


def test_unmeasured_work_is_not_fabricated_and_launch_only_is_explicit():
    result = performance.decomposition(10, 8, 2)
    assert (result["A"], result["E"], result["AE"], result["O"]) == (None, None, 4, 1.25)
    with pytest.raises(ValueError, match="source-bound work counters"):
        performance.decomposition(10, 8, 2, sol_execution_ms=4)
    with pytest.raises(ValueError, match="source-bound work counters"):
        performance.decomposition(10, 8, 2, sol_execution_ms=4,
            counters={"status": "measured", "source_sha256": "source"})
    ref = {"wall_ms": 10, "device_busy_ms": 8, "kernel_count": 10}
    cand = {"wall_ms": 9, "device_busy_ms": 8, "kernel_count": 2}
    assert performance.launch_only(ref, cand, launch_ms=0.2, speedup_ci95=[1.1, 1.2], noise_ci95=[0.99, 1.01])["launch_only"]
    assert not performance.launch_only(ref, cand, launch_ms=0.01, speedup_ci95=[1.1, 1.2], noise_ci95=[0.99, 1.01])["launch_only"]
