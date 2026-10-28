"""Small CPU controls and unscored serialization fixtures, never device fits."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kb_workflows import engine
from kb_workflows.agents import generation
from kb_workflows.benchmarks import reference_population
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation import measurement, timed_numerics, tool_checks
from kb_workflows.numerics import input_calibration
from kb_workflows.numerics import precision_calibration as pc
from kb_workflows.numerics import precision_calibration_task as task
from kb_workflows.numerics.numerical import PrecisionContract, precision_gate, reference_utilization
from src import eval as native
from src.input_groups import copy_inputs
from tests.numerics.test_input_calibration import CAPS, CPUAdapter, synthetic_format_fixture
from tests.numerics.test_input_calibration import calibration as input_record


def dyadic_inputs():
    return [torch.randint(-4, 5, (3, 7)).float() * .25, torch.randint(-4, 5, (7, 5)).float() * .25]


def calibrate(tmp_path, *, level=1, pid=2, precision="fp32", drift=False, bad=False, profile=None, geometry=False):
    config = RunConfig(level=level, start=pid, end=pid, generation="local", precision=precision,
                       runs_dir=str(tmp_path), run_name="precision-controls", num_perf_trials=2, paired_trials=10, oracle_memory_mb=64).validate()
    request = engine.task_payload(config, pid, task="calibrate_precision")
    if profile is not None:
        request["eval_config"].update(input_profile=profile, numerical_validation=True)
    calls = []

    def loader(source, namespace):
        Model, init, _ = native.load_original_model_and_inputs(source, namespace)
        if bad:
            Original = Model

            class BadFixture(Original):
                def forward(self, *values):
                    return super().forward(*values) + 1

            Model = BadFixture

        def inputs():
            calls.append(1)
            if level == 2:
                return [torch.rand(2, 64, 7, 9)]
            if level == 3:
                return [torch.rand(5, 2, 1024), torch.rand(2, 256)]
            if pid == 90:
                return [torch.randn(2, 256) * .01]
            if pid == 1:
                return [torch.randint(-4, 5, (3, 3)).float() * .25, torch.randint(-4, 5, (3, 3)).float() * .25]
            if drift and len(calls) > len(pc.FIT_SEEDS):
                return [torch.randn(3, 7), torch.randn(7, 5)]
            if geometry and len(calls) > len(pc.FIT_SEEDS):
                return [torch.randint(-4, 5, (4, 7)).float() * .25, torch.randint(-4, 5, (7, 5)).float() * .25]
            return dyadic_inputs()
        return Model, init, inputs

    record = task.calibrate(request, adapter=CPUAdapter(), loader=loader)["record"]
    return config, request, record


@pytest.fixture
def fitted(tmp_path):
    config, request, record = calibrate(tmp_path)
    assert record["qualified"] and record["factor"] == pc.FIT["minimum_factor"], record
    pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True)
    # Format fixture only: never written to the repository or used for measurements.
    fixture = copy.deepcopy(record)
    fixture["simulated_execution"] = False
    reseal(fixture)
    path = tmp_path / "unscored-precision-format-fixture.json"
    atomic_json(path, {"protocol": pc.BUNDLE, "records": {"1/2/fp32": fixture}})
    config.precision_calibration = str(path)
    config.validate()
    return config, request, fixture


def reseal(record):
    record["receipt_sha256"] = fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})


def test_reference_only_fit_tightens_all_terms_and_keeps_disjoint_check_seeds(tmp_path):
    _, request, record = calibrate(tmp_path)
    assert [row["seed"] for row in record["fit"]] == list(pc.FIT_SEEDS)
    assert [row["seed"] for row in record["check"]] == list(pc.CHECK_SEEDS)
    assert set(pc.FIT_SEEDS).isdisjoint(pc.CHECK_SEEDS)
    assert record["candidate_executed"] is False and record["timings_collected"] is False
    assert record["qualified"] and all(record["policy"][key] <= value for key, value in record["base_policy"].items())
    assert record["simulated_execution"] is True
    with pytest.raises(ValueError, match="Simulated"):
        pc.validate_record(record, request["ref_arch_src"], "fp32")


def test_heldout_error_never_refits_or_widens_a_frozen_training_policy(tmp_path):
    _, request, record = calibrate(tmp_path, drift=True)
    assert record["factor"] == pc.FIT["minimum_factor"]
    assert record["qualified"] is False
    assert any(row["status"] == "failed" for row in record["check"])
    assert record["policy"] == pc.fitted_policy(record["base_policy"], record["factor"])
    pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True, require_qualified=False)
    with pytest.raises(ValueError, match="no widening"):
        pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True)


def test_reference_outside_the_initial_cap_cannot_fit_a_larger_budget(tmp_path):
    _, request, record = calibrate(tmp_path, bad=True)
    assert record["qualified"] is False and record["factor"] is None and record["policy"] is None
    assert len(record["fit"]) == 3 and len(record["check"]) == 3
    assert all(row["status"] == "failed" for row in record["fit"])
    pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True, require_qualified=False)


@pytest.mark.parametrize("level,pid", [(2, 1), (3, 34)])
def test_original_composite_and_recurrent_constructors_produce_replayable_cpu_fit_records(tmp_path, level, pid):
    _, request, record = calibrate(tmp_path, level=level, pid=pid)
    pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True, require_qualified=False)
    assert all(row["oracle_receipt"] is not None for row in record["fit"] + record["check"])
    assert record["factor"] is None or record["factor"] <= 1


@pytest.mark.parametrize("dtype,precision", [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")])
def test_typed_stored_values_are_compared_to_unrounded_fp64_truth(tmp_path, dtype, precision):
    _, request, record = calibrate(tmp_path, precision=precision)
    pc.validate_record(record, request["ref_arch_src"], precision, allow_simulated=True)
    assert record["qualified"] and record["precision"] == precision
    assert all(row["utilization"]["floating_elements"] == 15 for row in record["fit"] + record["check"])


@pytest.mark.parametrize("damage", ["factor", "policy", "cap", "margin", "fit_seed", "check_seed", "repeat_fit", "overlap",
                                   "signature", "raw_signature", "device", "source", "oracle", "evaluator", "dtype", "profile",
                                   "fit_ratio", "check_ratio", "norm_limit", "counter", "extra_leaf", "fitted_gate", "qualified", "memory"])
def test_resealed_precision_records_cannot_change_caps_fit_data_holdout_or_bindings(fitted, damage):
    _, request, record = fitted
    row = record["fit"][0]
    if damage == "factor":
        record["factor"] *= 2
    elif damage == "policy":
        record["policy"]["norm_rtol"] *= 2
    elif damage == "cap":
        record["base_policy"]["norm_rtol"] *= 2
    elif damage == "margin":
        record["fit_method"]["margin"] = 100
    elif damage == "fit_seed":
        row["seed"] = 42
    elif damage == "check_seed":
        record["check"][0]["seed"] = pc.FIT_SEEDS[0]
    elif damage == "repeat_fit":
        row["input_probe_sha256"] = record["fit"][1]["input_probe_sha256"]
    elif damage == "overlap":
        record["check"][0]["input_probe_sha256"] = row["input_probe_sha256"]
    elif damage in {"signature", "raw_signature"}:
        row["input_signature_sha256" if damage == "signature" else "raw_input_signature_sha256"] = "0" * 64
    elif damage == "device":
        record["device"]["uuid"] = "another-device"
    elif damage == "source":
        record["source_sha256"] = "0" * 64
    elif damage == "oracle":
        record["oracle_spec_sha256"] = "0" * 64
    elif damage == "evaluator":
        record["evaluator_sha256"] = "0" * 64
    elif damage == "dtype":
        record["precision"] = "bf16"
    elif damage == "profile":
        record["input_profile_sha256"] = "0" * 64
    elif damage == "fit_ratio":
        row["utilization"]["norm_max"] = .5
    elif damage == "check_ratio":
        record["check"][0]["utilization"]["element_max"] = .5
    elif damage == "norm_limit":
        row["utilization"]["leaves"][0]["norm_limit"] *= 2
    elif damage == "counter":
        row["reference_gate"]["leaves"][0]["reference"]["elements_failed"] = 1
    elif damage == "extra_leaf":
        row["reference_gate"]["leaves"].append(copy.deepcopy(row["reference_gate"]["leaves"][0]))
    elif damage == "fitted_gate":
        record["check"][0]["fitted_gate"]["leaves"][0]["reference"]["elements_failed"] = 1
    elif damage == "qualified":
        record["qualified"] = False
    else:
        row["estimated_host_bytes"] = 10**12
    reseal(record)
    with pytest.raises(ValueError):
        pc.validate_record(record, request["ref_arch_src"], "fp32")


def test_profile_recipe_is_applied_once_and_the_fit_cannot_switch_profiles(tmp_path, monkeypatch):
    record, _ = input_record(tmp_path, pid=90)
    profile = synthetic_format_fixture(record)
    from kb_workflows.numerics import input_profiles

    original = input_profiles.transform
    seen = []

    def observe(*args, **kwargs):
        seen.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(input_profiles, "transform", observe)
    _, request, fitted = calibrate(tmp_path, pid=90, profile=profile)
    assert seen == [profile["selected_recipe"]] * 6
    pc.validate_record(fitted, request["ref_arch_src"], "fp32", profile, allow_simulated=True)
    with pytest.raises(ValueError, match="another task/policy"):
        pc.validate_record(fitted, request["ref_arch_src"], "fp32", None, allow_simulated=True)


@pytest.mark.parametrize("execution_uuid", [CAPS["uuid"], "another-instance-of-the-same-class"])
def test_runtime_uses_fitted_policy_and_requires_the_actual_device_receipt(fitted, monkeypatch, execution_uuid):
    config, _, record = fitted
    request = engine.task_payload(config, 2, task="baseline")
    assert request["eval_config"]["precision_profile"] == record
    caps = {**CAPS, "uuid": execution_uuid}
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: copy.deepcopy(caps))
    Model, init, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    reference, candidate = Model(*init()), Model(*init())
    args = dyadic_inputs()

    @timed_numerics.task_policy
    @pc.task_policy
    def worker(request):
        ticket = input_calibration.prepare_output(reference, args, scope="first")
        expected = reference(*copy_inputs(args))
        input_calibration.verify_output(reference, args, expected, candidate(*copy_inputs(args)), prepared=ticket)

        def verify(values, output):
            ticket = input_calibration.prepare_output(reference, values)
            expected = reference(*copy_inputs(values))
            input_calibration.verify_output(reference, values, expected, output, prepared=ticket)

        measurement.synchronized_times(candidate, args, "cpu", num_trials=2, input_factory=lambda _: dyadic_inputs(), verifier=verify)
        return {"baseline_correctness": True, "device_fingerprint": copy.deepcopy(caps), "metadata": {"simulated_execution": True}}

    result = worker(request)
    assert result["baseline_correctness"] and pc.issue(result, request) is None
    assert result["metadata"]["timed_numerical_audit"]["audit"]["policy"] == record["policy"]
    result["metadata"].pop("precision_calibration")
    assert pc.issue(result, request) and timed_numerics.issue(result, request)
    result["metadata"]["workflow_signature"] = fingerprint(request)
    assert not tool_checks.reusable(result, request)


def test_device_mismatch_fails_before_the_worker_or_candidate_runs(fitted, monkeypatch):
    config, _, _ = fitted
    request = engine.task_payload(config, 2, task="baseline")
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: {**CAPS, "uuid": "another-physical-device", "driver_version": "another-stack"})

    @pc.task_policy
    def forbidden(request):
        pytest.fail("Wrong-device precision fit must not invoke a candidate")

    with pytest.raises(ValueError, match="cannot transfer"):
        forbidden(request)


def test_fit_policy_freezes_before_candidates_and_child_jobs_get_only_their_row(fitted):
    config, _, record = fitted
    reference_population.freeze_run(config)
    path = config.run_dir / "precision_calibration_manifest.json"
    assert read_json(path)["bundle_sha256"] == fingerprint(read_json(Path(config.precision_calibration)))
    directory = generation.prepare_job(config, 2, 0)
    job = RunConfig(**read_json(directory / "job_config.json")).validate()
    assert pc.for_config(job, 2, engine.task_payload(job, 2, task="baseline")["ref_arch_src"]) == record
    assert "Effective device-fitted" in generation.build_prompt(config, 2)
    data = read_json(Path(config.precision_calibration))
    data["changed_after_candidate"] = True
    atomic_json(Path(config.precision_calibration), data)
    with pytest.raises(ValueError, match="Frozen precision"):
        reference_population.freeze_run(config)


def test_fit_cannot_be_adopted_after_candidate_generation(fitted):
    config, _, _ = fitted
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 2)).write_text("unscored fixture candidate")
    with pytest.raises(ValueError, match="must freeze before candidates"):
        reference_population.freeze_run(config)


def test_precision_file_paths_are_relative_and_same_class_multi_gpu_configs_remain_available(fitted, tmp_path):
    config, _, _ = fitted
    document = tmp_path / "device.toml"
    document.write_text('[run]\nlevel=1\nstart=2\nend=2\ngeneration="local"\nprecision_calibration=' + json.dumps(Path(config.precision_calibration).name) + '\n')
    parsed = parse_config(["--config", str(document)])
    assert parsed.precision_calibration == config.precision_calibration and parsed.numerical_validation
    config.gpus = "0,1"
    config.validate()
    reference_population.freeze_run(config)
    assert read_json(config.run_dir / "precision_calibration_manifest.json")["contract"]["gpus"] == "0,1"


@pytest.mark.parametrize("level,count", [(1, 100), (2, 100), (3, 50)])
def test_preview_is_sdk_light_and_never_executes_calibration(tmp_path, level, count):
    config = tmp_path / "preview.toml"
    config.write_text(f'[run]\nlevel={level}\nstart=1\nend={count}\ngeneration="local"\n')
    output = tmp_path / "absent" / "fit.json"
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from scripts.calibrate import main
assert main(["precision","--config",{str(config)!r},"--output",{str(output)!r}]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], text=True, capture_output=True, check=True)
    preview = json.loads(result.stdout)
    assert len(preview["problems"]) == count and preview["experiments_started"] is False
    assert not output.parent.exists()


def test_changed_nominal_geometry_does_not_produce_a_qualified_fit(tmp_path):
    _, request, record = calibrate(tmp_path, geometry=True)
    assert record["qualified"] is False
    assert len({row["input_signature_sha256"] for row in record["fit"] + record["check"]}) == 2
    with pytest.raises(ValueError, match="changed native dimensions"):
        pc.validate_record(record, request["ref_arch_src"], "fp32", allow_simulated=True)


def test_a_run_cannot_pool_fits_from_different_hardware_or_software_classes(fitted, tmp_path):
    from kb_workflows.analysis.calibration import identity

    config, _, two = fitted
    _, _, one = calibrate(tmp_path, pid=1)
    one["simulated_execution"] = False
    one["device"]["driver_version"] = "different-software-stack"
    one["device_identity_sha256"] = identity(one["device"])
    reseal(one)
    atomic_json(Path(config.precision_calibration), {"protocol": pc.BUNDLE, "records": {"1/1/fp32": one, "1/2/fp32": two}})
    config.start = 1
    with pytest.raises(ValueError, match="one hardware/software class"):
        reference_population.freeze_run(config)
    assert not config.run_dir.exists()


def test_harbor_exports_private_precision_data_and_only_the_effective_policy(fitted, tmp_path):
    from harbor.models.task.task import Task

    from kb_workflows.runtime import harbor

    config, _, record = fitted
    config.hardware = "RTX_A6000"
    directory = harbor.write_task(tmp_path / "unscored-harbor-task", config, 2)
    Task(directory)
    private = read_json(directory / "tests/precision_calibration.json")
    assert list(private["records"]) == ["1/2/fp32"] and private["records"]["1/2/fp32"] == record
    child = read_json(directory / "tests/config.json")["config"]
    assert child["precision_calibration"] == "/tests/precision_calibration.json" and child["numerical_validation"]
    assert (directory / "tests/repo/kb_workflows/numerics/precision_calibration_task.py").read_bytes() == (ROOT / "kb_workflows/numerics/precision_calibration_task.py").read_bytes()
    instruction = (directory / "instruction.md").read_text()
    assert "device-fitted" in instruction and "SIMULATED" not in instruction


def test_regrade_archives_and_rebinds_the_precision_fit_without_running_workers(fitted, tmp_path, monkeypatch):
    from kb_workflows.runtime import regrade

    monkeypatch.setattr(regrade, "ROOT", ROOT)
    config, _, _ = fitted
    config.hardware, config.gpu, config.gpu_arch = "RTX_A6000", "A6000", "Ampere"
    reference_population.freeze_run(config)
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    atomic_json(config.run_dir / "paper_analysis.json", {"problem_ids": [2], "declared_samples": 1})
    output = tmp_path / "unscored-regrade"
    plan = regrade.prepare(config.run_dir, copy.deepcopy(config), output=output)
    assert "precision_calibration_manifest.json" in plan["history_files"]
    assert "precision_calibration" in plan["verifier_inputs"]
    assert (output / "precision_calibration_manifest.json").is_file()
    assert plan["experiments_started"] is False
    regrade.validate(plan)


def test_study_and_final_contracts_bind_precision_file_contents(fitted, tmp_path):
    from kb_workflows.analysis import experiment, study_plan
    from kb_workflows.evaluation import final_verification

    config, _, _ = fitted
    config.hardware, config.gpu, config.gpu_arch, config.final_repeats = "RTX_A6000", "A6000", "Ampere", 2
    options = config.to_dict()
    plan = study_plan.make_plan({"base_config": options, "models": ["unrun"], "methods": ["one-shot"],
                                "sessions": 1, "vary": "wall", "budgets": [100]}, directory=tmp_path / "study")
    path = study_plan.write_plan(plan, tmp_path / "study")
    assert study_plan.execute_plan(path)["experiments_started"] is False
    assert final_verification._options(config)["precision_calibration_sha256"] == experiment.protocol(config)["precision_calibration_sha256"]
    document = read_json(Path(config.precision_calibration))
    document["changed_after_plan"] = True
    atomic_json(Path(config.precision_calibration), document)
    with pytest.raises(ValueError, match="precision calibration changed"):
        study_plan.execute_plan(path)


def test_fitted_cli_preview_validates_the_policy_without_importing_torch(fitted, tmp_path):
    config, _, record = fitted
    args = ["--generation", "local", "--start", "2", "--end", "2", "--dry-run",
            "--precision-calibration", config.precision_calibration, "--runs-dir", str(tmp_path / "absent")]
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main({args!r}) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["calibrated_precision"]["2"] == record["policy"]
    assert not (tmp_path / "absent").exists()


def test_utilization_handles_exact_discrete_and_complex_outputs_without_rounding_truth():
    contract = PrecisionContract(1e-3, 1e-6, 1e-3, 1e-6, 1e-5)
    value = {"float": torch.tensor([1 + 2j], dtype=torch.complex64), "index": torch.tensor([1])}
    truth = {"float": torch.tensor([1.00001 + 2.00001j], dtype=torch.complex128), "index": torch.tensor([1])}
    summary = reference_utilization(value, truth, contract)
    assert summary["floating_elements"] == 2 and 0 < summary["norm_max"] < 1
    assert precision_gate(value, value, truth, contract)["passed"]
