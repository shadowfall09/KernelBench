"""Reference-only calibration on CPU; physical execution remains unrun."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kb_workflows import engine, input_calibration, input_calibration_task, input_profiles, measurement, numerical_audit
from kb_workflows.calibration import IDENTITY_FIELDS
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.numerical import signal_audit
from kb_workflows.operator_specs import spec_for
from scripts import calibrate_inputs
from src import eval as native
from src.validation import compare_outputs


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


CAPS = {**dict.fromkeys(IDENTITY_FIELDS), "available": True, "uuid": "SIMULATED", "name": "SIMULATED",
        "driver_version": "fixture", "torch_version": torch.__version__}


class CPUAdapter:
    device = torch.device("cpu")
    simulated = True

    def caps(self):
        return copy.deepcopy(CAPS)


def calibration(tmp_path, pid=90, precision="fp32"):
    config = RunConfig(generation="local", hardware="RTX_A6000", start=pid, end=pid, precision=precision,
                       oracle_memory_mb=8, runs_dir=str(tmp_path)).validate()
    payload = engine.task_payload(config, pid, task="calibrate_inputs")

    def loader(source, context):
        Model, init, _ = native.load_original_model_and_inputs(source, context)
        # Never allocate original 32768 x 32768 arrays in software regression.
        return Model, init, lambda: [torch.randn(2, 256) * 0.01]

    result = input_calibration_task.calibrate(payload, adapter=CPUAdapter(), loader=loader)
    return result["record"], payload


def synthetic_format_fixture(record):
    """Valid serialization fixture, never evidence of a real measured cohort."""
    value = copy.deepcopy(record)
    value["simulated_execution"] = False
    value["receipt_sha256"] = fingerprint({key: val for key, val in value.items() if key != "receipt_sha256"})
    return value


@pytest.mark.parametrize("pid", range(1, 101))
def test_reviewed_distribution_adapters_preserve_all_l1_roles_shapes_and_reference_legality(pid):
    from test_operator_oracles import small_case

    _, model, args = small_case(pid)
    before = measurement.input_signature(args)
    for recipe in input_profiles.recipes(pid):
        transformed = input_profiles.transform(args, pid, recipe)
        assert measurement.input_signature(transformed) == before
        for first, actual in zip(args, transformed):
            if isinstance(first, torch.Tensor) and not first.is_floating_point():
                assert torch.equal(first, actual)
        output = model(*transformed)
        assert all(torch.isfinite(value).all() for value in (output if isinstance(output, tuple) else [output]))
        if pid == 13:
            assert all(torch.equal(value, value.t()) for value in transformed)
        if pid == 14:
            assert all(torch.equal(value, value.triu()) for value in transformed)
        if pid == 15:
            assert all(torch.equal(value, value.tril()) for value in transformed)
        if pid == 98:
            assert all((value > 0).all() and torch.allclose(value.sum(-1), torch.ones(value.shape[:-1]), atol=1e-6) for value in transformed)
        if pid == 100:
            assert torch.equal(args[1], transformed[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_product_profile_keeps_every_long_prefix_informative_and_varied(dtype):
    seen = []
    for seed in (42, 123, 456):
        torch.manual_seed(seed)
        source = torch.empty(2, 8192, dtype=dtype)
        values = input_profiles.transform([source], 90, "bounded_product_walk")[0]
        prefix = values.cumprod(1)
        assert prefix.dtype == dtype and torch.isfinite(prefix).all()
        assert prefix.abs().min() >= 0.125 and prefix.abs().max() <= 8
        assert signal_audit(prefix, absolute_tolerance=0.01)["informative"]
        seen.append(input_calibration_task.probe(values))
    assert len({fingerprint(value) for value in seen}) == 3


def test_reference_only_selection_preserves_failed_attempts_and_rejects_simulated_qualification(tmp_path):
    record, payload = calibration(tmp_path)
    assert [row["recipe"] for row in record["attempts"]] == ["identity", "bounded_product_walk"]
    assert record["qualified"] and record["selected_recipe"] == "bounded_product_walk"
    assert record["candidate_executed"] is False and record["timings_collected"] is False
    assert all(row["signal"]["leaves"][0]["fraction_above_atol"] < 0.9 for row in record["attempts"][0]["cases"])
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True)
    with pytest.raises(ValueError, match="Simulated"):
        input_calibration.validate_record(record, payload["ref_arch_src"])
    assert record["precision_contract"]["norm_atol"] == 0
    assert record["precision_contract"]["element_atol"] == 0


@pytest.mark.parametrize("damage", ["seed", "summary", "order", "wider_budget", "norm_limit", "device", "source",
                                   "input_signature", "diversity", "memory", "environment", "missing_leaf", "nonfinite_error", "scalar_metrics"])
def test_resealed_calibration_cannot_hide_changed_or_missing_reference_evidence(tmp_path, damage):
    record, payload = calibration(tmp_path)
    row = record["attempts"][-1]["cases"][0]
    if damage == "seed":
        row["seed"] = 1
    elif damage == "summary":
        record["selected_recipe"] = "identity"
    elif damage == "order":
        record["attempts"].reverse()
    elif damage == "wider_budget":
        record["precision_contract"]["norm_rtol"] *= 10
    elif damage == "norm_limit":
        row["reference_gate"]["leaves"][0]["norm_limit"] *= 10
    elif damage == "device":
        record["device"]["uuid"] = "changed"
    elif damage == "source":
        record["source_sha256"] = "0" * 64
    elif damage == "input_signature":
        row["raw_input_signature_sha256"] = "0" * 64
    elif damage == "diversity":
        for item in record["attempts"][-1]["cases"]:
            item["input_probe_sha256"] = "0" * 64
    elif damage == "nonfinite_error":
        row["reference_gate"]["leaves"][0]["reference"]["max_abs_error"] = float("nan")
    elif damage == "scalar_metrics":
        row["reference_gate"]["leaves"][0]["reference"] = 1
    elif damage == "missing_leaf":
        row["reference_gate"]["leaves"].clear()
    elif damage == "memory":
        row["estimated_host_bytes"] = 10**12
    else:
        record["environment_changes"] = ["default_dtype"]
    record["receipt_sha256"] = fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True)


def test_calibration_never_loads_a_candidate(tmp_path):
    _, payload = calibration(tmp_path)
    payload["kernel_src"] = "raise AssertionError('must not execute')"
    with pytest.raises(ValueError, match="no candidate"):
        input_calibration_task.calibrate(payload, adapter=CPUAdapter())


def test_runtime_uses_actual_loader_hook_strong_comparison_and_original_signature(tmp_path, monkeypatch):
    record, base = calibration(tmp_path)
    record = synthetic_format_fixture(record)
    payload = copy.deepcopy(base)
    payload["task"] = "evaluate"
    payload["eval_config"].update(input_profile=record, numerical_validation=True, measure_performance=False)
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: copy.deepcopy(CAPS))

    @input_calibration.task_policy
    def worker(request):
        namespace = {}
        Model, init, factory = native.load_original_model_and_inputs(request["ref_arch_src"], namespace)
        namespace.update(batch_size=2, input_shape=(256,))
        torch.manual_seed(71)
        args = factory()
        expected = Model(*init())(*args)
        assert expected.abs().min() >= 0.125
        assert compare_outputs(expected, torch.zeros_like(expected), 1.0) is not None
        return {"compiled": True, "correctness": True, "runtime": 7, "metadata": {"device_fingerprint": CAPS}}

    result = worker(payload)
    assert result["runtime"] == 7 and input_calibration.issue(result, payload) is None
    assert result["metadata"]["input_calibration"]["factory_calls"] == 1
    assert input_profiles.ACTIVE.get() is None
    result["metadata"]["input_calibration"]["factory_calls"] = 0
    assert input_calibration.issue(result, payload)
    _, policy = numerical_audit._spec({**payload, "eval_config": {**payload["eval_config"],
        "oracle_spec_sha256": fingerprint(spec_for(1, 90, payload['ref_arch_src']))}})
    assert policy["norm_atol"] == 0


def test_profile_artifact_changes_payload_baselines_frozen_and_research_contracts(tmp_path):
    from kb_workflows import experiment, final_verification, generation, harbor, study_data

    record, payload = calibration(tmp_path)
    fixture = synthetic_format_fixture(record)
    bundle = {"protocol": input_calibration.BUNDLE, "records": {"1/90/fp32": fixture}}
    path = tmp_path / "inputs.json"
    atomic_json(path, bundle)
    config = RunConfig(generation="local", hardware="RTX_A6000", input_calibration=str(path),
                       start=90, end=90, runs_dir=str(tmp_path), run_name="calibrated").validate()
    assert config.numerical_validation
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 90)).write_text("candidate")
    baseline = engine.task_payload(config, 90, task="baseline")
    assert baseline["eval_config"]["input_profile"]["receipt_sha256"] == fixture["receipt_sha256"]
    plain = copy.deepcopy(config)
    plain.input_calibration = ""
    assert fingerprint(baseline) != fingerprint(engine.task_payload(plain, 90, task="baseline"))
    assert experiment.protocol(config) != experiment.protocol(plain)
    assert final_verification._options(config) != final_verification._options(plain)
    assert study_data.evaluation_contract(config) != study_data.evaluation_contract(plain)
    job = generation.prepare_job(config, 90, 0)
    assert read_json(Path(read_json(job / "job_config.json")["input_calibration"])) == bundle
    config.generation = "harbor"
    exported = harbor.write_task(tmp_path / "harbor", config, 90)
    assert (exported / "tests/input_calibration.json").is_file()
    assert not (exported / "environment/input_calibration.json").exists()


def test_config_and_calibration_preview_are_sdk_light_and_do_not_allocate_inputs(tmp_path):
    config = tmp_path / "calibrate.toml"
    config.write_text('[run]\nlevel=1\nstart=90\nend=90\n[ gpu ]\nhardware="RTX_A6000"\n')
    output = tmp_path / "not-created.json"
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from scripts.calibrate_inputs import main
assert main(["--config",{str(config)!r},"--output",{str(output)!r}]) == 0
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["candidate_executed"] is False
    assert not output.exists()
    cfg = parse_config(["--config", str(config)])
    tasks = calibrate_inputs.prepare(cfg)
    assert "kernel_src" not in tasks[0][1]
    assert tasks[0][1]["request_id"] == "kb-" + fingerprint(tasks[0][1])


@pytest.mark.parametrize("phase", ["evaluate", "baseline", "paired_timing"])
def test_every_timed_output_uses_independent_truth_and_receipt_on_execution_device(tmp_path, monkeypatch, phase):
    from kb_workflows import paired_task

    record, payload = calibration(tmp_path)
    payload["task"] = phase
    payload["eval_config"].update(input_profile=synthetic_format_fixture(record), numerical_validation=True,
                                   measure_performance=True, num_perf_trials=2, paired_trials=10)
    observed = {**CAPS, "uuid": "SIMULATED-SECOND-LANE"}
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: observed)

    @input_calibration.task_policy
    def worker(request):
        namespace = {}
        Model, init, factory = native.load_original_model_and_inputs(request["ref_arch_src"], namespace)
        namespace.update(batch_size=2, input_shape=(256,))
        model = Model(*init())
        args = factory()

        def verify(values, output):
            expected = model(*values)
            input_calibration.verify_output(model, values, expected, output)

        if phase == "paired_timing":
            paired_task.collect_pairs(model, model, model, lambda _: factory(), model, device="cpu", trials=10,
                                      seed=42, verifier=verify)
        else:
            if phase == "baseline":
                verify(args, model(*args))
            measurement.synchronized_times(model, args, "cpu", num_trials=2,
                                           input_factory=lambda _: factory(), verifier=verify)
        return {"correctness": True, "baseline_correctness": phase == "baseline",
                "metadata": {"device_fingerprint": observed}}

    result = worker(payload)
    receipt = result["metadata"]["input_calibration"]
    assert receipt["execution_device_identity_sha256"] != receipt["calibration_device_identity_sha256"]
    assert receipt["independent_outputs_verified"] == {"evaluate": 5, "baseline": 6, "paired_timing": 39}[phase]
    assert input_calibration.issue(result, payload) is None
    receipt["independent_outputs_verified"] -= 1
    assert input_calibration.issue(result, payload)


def test_independent_timing_gate_rejects_zero_and_checks_memory_before_oracle(tmp_path, monkeypatch):
    from kb_workflows import operator_oracles

    record, payload = calibration(tmp_path)
    payload["eval_config"].update(input_profile=synthetic_format_fixture(record), numerical_validation=True)
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: CAPS)

    @input_calibration.task_policy
    def worker(request):
        namespace = {}
        Model, init, factory = native.load_original_model_and_inputs(request["ref_arch_src"], namespace)
        namespace.update(batch_size=2, input_shape=(256,))
        values = factory()
        model = Model(*init())
        expected = model(*values)
        with pytest.raises(ValueError, match="FP64 budget"):
            input_calibration.verify_output(model, values, expected, torch.zeros_like(expected))
        monkeypatch.setattr(operator_oracles, "estimate_bytes", lambda *args: 10**12)
        monkeypatch.setattr(operator_oracles, "execute", lambda *args: pytest.fail("oracle allocation after admission rejection"))
        with pytest.raises(ValueError, match="no resizing"):
            input_calibration.verify_output(model, values, expected, expected)
        return {"correctness": False}

    worker(payload)


def test_calibration_admission_does_not_transform_resize_or_compute_full_truth(tmp_path, monkeypatch):
    from kb_workflows import operator_oracles

    _, payload = calibration(tmp_path)
    Model, init, _ = native.load_original_model_and_inputs(payload["ref_arch_src"], {})
    monkeypatch.setattr(operator_oracles, "estimate_bytes", lambda *args: 10**12)
    monkeypatch.setattr(operator_oracles, "execute", lambda *args: pytest.fail("oracle exceeded host envelope"))
    monkeypatch.setattr(input_calibration_task, "transform", lambda *args: pytest.fail("transform exceeded host envelope"))
    result = input_calibration_task.calibrate(payload, adapter=CPUAdapter(),
        loader=lambda *args: (Model, init, lambda: [torch.randn(2, 256)]))
    record = result["record"]
    assert record["qualified"] is False and record["selected_recipe"] is None
    assert all(row["status"] == "not_measured" for attempt in record["attempts"] for row in attempt["cases"])
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True, require_qualified=False)


def test_baseline_without_complete_timed_fp64_evidence_is_not_resumable(tmp_path):
    record, _ = calibration(tmp_path)
    path = tmp_path / "inputs.json"
    atomic_json(path, {"protocol": input_calibration.BUNDLE, "records": {"1/90/fp32": synthetic_format_fixture(record)}})
    config = RunConfig(generation="local", hardware="RTX_A6000", input_calibration=str(path), start=90, end=90).validate()
    assert input_calibration.baseline_issue(config, {"baseline_correctness": True}, 90)
    config.baseline_suite = True
    baseline = {"baseline_correctness": True, "variants": {"eager": {"baseline_correctness": True}}}
    assert input_calibration.baseline_issue(config, baseline, 90)


def test_actual_baseline_entrypoint_checks_first_warmup_and_measured_outputs(tmp_path, monkeypatch):
    from kb_workflows import evaluate_task

    record, payload = calibration(tmp_path)
    payload["task"] = "baseline"
    payload["eval_config"].update(input_profile=synthetic_format_fixture(record), numerical_validation=True,
                                   num_perf_trials=2, baseline_suite=False)
    original_loader, timer = native.load_original_model_and_inputs, measurement.synchronized_times

    def small_loader(source, namespace):
        Model, init, factory = original_loader(source, namespace)
        namespace.update(batch_size=2, input_shape=(256,))
        return Model, init, factory

    flags = []
    monkeypatch.setattr(native, "load_original_model_and_inputs", small_loader)
    monkeypatch.setattr(native, "_process_input_tensor", lambda value, device, dtype: value.to(dtype=dtype) if isinstance(value, torch.Tensor) else value)
    monkeypatch.setattr(native, "move_model", lambda model, device, dtype: model.to(dtype=dtype))
    monkeypatch.setattr(native, "get_timing_stats", lambda samples, **kwargs: measurement.timing_stats(samples))
    monkeypatch.setattr(native, "get_memory_stats", lambda *args, **kwargs: {"peak_bytes": None, "simulated_execution": True})
    monkeypatch.setattr(measurement, "synchronize", lambda _: None)
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: CAPS)
    monkeypatch.setattr(measurement, "synchronized_times", lambda model, args, device, **kwargs: timer(model, args, "cpu", **kwargs))
    monkeypatch.setattr(evaluate_task, "configure_tf32", lambda module, enabled: flags.append(enabled))
    result = input_calibration.task_policy(evaluate_task.baseline)(payload)
    assert result["baseline_correctness"] is True and result["tf32"] is False
    assert result["num_trials"] == 2 and flags == [False, False]
    assert result["metadata"]["input_calibration"]["independent_outputs_verified"] == 6
    assert input_calibration.issue(result, payload) is None


def test_queue_accepts_reference_calibration_but_rejects_candidate_payloads(tmp_path):
    from kb_workflows.queue_worker import validate_payload

    _, payload = calibration(tmp_path)
    validate_payload(payload)
    payload["kernel_src"] = "raise AssertionError('never execute')"
    with pytest.raises(ValueError, match="no candidate"):
        validate_payload(payload)
