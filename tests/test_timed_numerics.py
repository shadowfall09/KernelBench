"""Unscored CPU timing/control tests; no original-size or device experiment."""

import copy
import subprocess
import sys

import pytest
import torch

from kb_workflows import engine, input_calibration, measurement, timed_numerics, tool_checks
from kb_workflows.common import ROOT, fingerprint
from kb_workflows.config import RunConfig
from kb_workflows.paired_task import collect_pairs
from src import eval as native
from src.input_groups import copy_inputs
from src.validation import compare_outputs


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def payload(tmp_path, *, level=1, pid=90, phase="evaluate"):
    config = RunConfig(level=level, start=pid, end=pid, generation="local", numerical_validation=True,
                       runs_dir=str(tmp_path), num_perf_trials=2, paired_trials=10, oracle_memory_mb=64).validate()
    request = engine.task_payload(config, pid, task="baseline")
    request["task"] = phase
    request["eval_config"].update(measure_performance=True)
    return request


def native_worker(request, *, wrong=False):
    Model, init, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    reference = Model(*init())
    candidate = copy.deepcopy(reference)
    args = [torch.rand(2, 16) * .1 + .9]

    @timed_numerics.task_policy
    def run(request):
        def verify(values, output):
            ticket = input_calibration.prepare_output(reference, values)
            expected = reference(*copy_inputs(values))
            input_calibration.verify_output(reference, values, expected, output, prepared=ticket)

        if request["task"] in {"baseline", "evaluate"}:
            ticket = input_calibration.prepare_output(reference, args, scope="first")
            expected = reference(*copy_inputs(args))
            input_calibration.verify_output(reference, args, expected, candidate(*copy_inputs(args)), prepared=ticket)

        def function(*values):
            output = candidate(*values)
            return output + .0001 if wrong else output

        times = measurement.synchronized_times(function, args, "cpu", num_trials=2,
            input_factory=lambda _: [torch.rand(2, 16) * .1 + .9], verifier=verify)
        return {"compiled": True, "baseline_correctness" if request["task"] == "baseline" else "correctness": True,
                "runtime": sum(times)/len(times), "metadata": {"simulated_execution": True}}

    return run(request)


@pytest.mark.parametrize("phase", ["baseline", "evaluate"])
def test_unprofiled_native_outputs_are_checked_on_first_warm_and_measured_calls(tmp_path, phase):
    request = payload(tmp_path, phase=phase)
    assert "input_profile" not in request["eval_config"]
    result = native_worker(request)
    assert result.get("correctness", result.get("baseline_correctness")) is True
    assert timed_numerics.issue(result, request) is None
    rows = result["metadata"]["timed_numerical_audit"]["audit"]["cases"]
    assert sum(row["scope"] == "timing" for row in rows) == 5
    assert sum(row["scope"] == "first" for row in rows) == 1
    assert all(row["precision_gate"]["reference_qualified"] and row["pre_reference_forward"] for row in rows)
    assert timed_numerics.ACTIVE.get() is None


def test_unprofiled_timing_rejects_error_accepted_by_legacy_comparison(tmp_path):
    values = torch.ones(2, 16)
    assert compare_outputs(values, values + .0001, 1e-3) is None
    with pytest.raises(ValueError, match="independent FP64"):
        native_worker(payload(tmp_path), wrong=True)
    assert timed_numerics.ACTIVE.get() is None


@pytest.mark.parametrize("damage", ["missing", "scope", "count", "first", "policy", "reference", "rms", "error_counter",
                                   "label", "seed", "memory", "prestate", "oracle", "source", "profile"])
def test_resealed_timed_receipts_cannot_skip_outputs_or_change_the_frozen_gate(tmp_path, damage):
    request = payload(tmp_path, phase="baseline")
    result = native_worker(request)
    document = result["metadata"]["timed_numerical_audit"]
    row = document["audit"]["cases"][0]
    if damage == "missing":
        result["metadata"].pop("timed_numerical_audit")
    elif damage == "scope":
        row["scope"] = "control"
    elif damage == "count":
        document["audit"]["cases"].pop()
    elif damage == "first":
        document["audit"]["cases"].pop(0)
        for i, item in enumerate(document["audit"]["cases"]):
            item["label"] = f"output/{i}"
    elif damage == "policy":
        document["audit"]["policy"]["norm_rtol"] *= 2
    elif damage == "reference":
        row["precision_gate"]["reference_qualified"] = False
    elif damage == "rms":
        row["precision_gate"]["leaves"][0]["candidate"]["rms_error"] = 1
    elif damage == "error_counter":
        row["precision_gate"]["leaves"][0]["candidate"]["elements_failed"] = 1
    elif damage == "label":
        row["label"] = "extra"
    elif damage == "seed":
        row["seed"] = 42
    elif damage == "memory":
        row["estimated_host_bytes"] = 10**12
    elif damage == "prestate":
        row["pre_reference_forward"] = False
    elif damage == "oracle":
        document["audit"]["oracle_spec_sha256"] = "0" * 64
    elif damage == "source":
        document["reference_sha256"] = "0" * 64
    else:
        document["input_profile_sha256"] = "0" * 64
    document["receipt_sha256"] = fingerprint({key: value for key, value in document.items() if key != "receipt_sha256"})
    assert timed_numerics.issue(result, request) is not None
    result["metadata"]["workflow_signature"] = fingerprint(request)
    assert not tool_checks.reusable(result, request)


@pytest.mark.parametrize("level,pid", [(1, 2), (2, 1), (3, 34)])
def test_actual_operator_composite_and_recurrent_oracles_accept_small_cpu_controls_without_profiles(tmp_path, level, pid):
    request = payload(tmp_path, level=level, pid=pid)
    Model, init, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    reference = Model(*init()).float()
    args = ([torch.randn(3, 7), torch.randn(7, 5)] if level == 1 else [torch.rand(2, 64, 7, 9)] if level == 2
            else [torch.rand(5, 2, 1024), torch.rand(2, 256)])
    request["eval_config"]["measure_performance"] = False

    @timed_numerics.task_policy
    def run(request):
        ticket = input_calibration.prepare_output(reference, args)
        expected = reference(*copy_inputs(args))
        input_calibration.verify_output(reference, args, expected, expected.clone() if isinstance(expected, torch.Tensor) else copy.deepcopy(expected), prepared=ticket)
        return {"correctness": True, "metadata": {"simulated_execution": True}}

    result = run(request)
    assert result["correctness"], result
    assert timed_numerics.issue(result, request) is None
    row = result["metadata"]["timed_numerical_audit"]["audit"]["cases"][0]
    if level > 1:
        assert row["composite_receipt" if level == 2 else "model_receipt"] is not None


def test_paired_preparer_uses_one_pre_forward_state_and_rng_for_all_three_outputs(tmp_path, monkeypatch):
    from test_model_input_profiles import StatefulRandom, counter_oracle

    counter_oracle(monkeypatch)
    request = payload(tmp_path, pid=90, phase="paired_timing")
    models = [StatefulRandom() for _ in range(4)]
    reference, candidate, control, oracle = models
    states = []

    def prepare(values):
        states.append(oracle.counter.item())
        return input_calibration.prepare_output(oracle, values)

    def verify(values, output, *, expected, prepared):
        input_calibration.verify_output(oracle, values, expected, output, prepared=prepared)

    @timed_numerics.task_policy
    def run(request):
        report = collect_pairs(reference, candidate, control, lambda _: [torch.ones(2, 4)], oracle,
            device="cpu", trials=10, seed=42, set_seed=torch.manual_seed, verifier=verify, prepare_verifier=prepare)
        return {"compiled": True, "correctness": True, "metadata": {"paired_timing": report, "simulated_execution": True}}

    result = run(request)
    assert result["correctness"] and timed_numerics.issue(result, request) is None
    assert states == list(range(13)) and all(model.counter.item() == 13 for model in models)
    assert len(result["metadata"]["timed_numerical_audit"]["audit"]["cases"]) == 39


def test_rng_restores_candidate_advancement_even_when_the_verifier_rejects():
    args = [torch.ones(2, 4)]
    torch.manual_seed(771)
    torch.rand_like(args[0])
    expected = torch.get_rng_state().clone()
    torch.manual_seed(771)

    def verify(values, output):
        torch.rand(19)
        raise ValueError("intentional bad output")

    with pytest.raises(ValueError, match="intentional"):
        measurement.synchronized_times(lambda x: x + torch.rand_like(x), args, "cpu", num_trials=1, verifier=verify)
    assert torch.equal(torch.get_rng_state(), expected)


def test_missing_verifier_cannot_return_a_positive_numerical_timing_result(tmp_path):
    request = payload(tmp_path)

    @timed_numerics.task_policy
    def unverified(request):
        return {"correctness": True, "metadata": {}}

    result = unverified(request)
    assert result["correctness"] is False
    assert result["metadata"]["timed_numerical_error"] == "timed_numerical_output_coverage_missing"


def test_baseline_suite_receipts_bind_each_variant_and_the_selected_winner(tmp_path):
    config = RunConfig(start=90, end=90, generation="local", numerical_validation=True,
                       runs_dir=str(tmp_path), num_perf_trials=2, paired_trials=10, oracle_memory_mb=64).validate()
    request = engine.task_payload(config, 90, task="baseline")
    request["eval_config"]["baseline_variant"] = "eager"
    result = native_worker(request)
    suite = {**copy.deepcopy(result), "winning_baseline": "eager", "variants": {"eager": copy.deepcopy(result)}}
    assert timed_numerics.baseline_issue(config, suite, 90) is None
    suite["winning_baseline"] = "compile"
    assert timed_numerics.baseline_issue(config, suite, 90) == "timed_numerical_receipt_missing_or_changed"


def test_actual_native_correctness_and_cold_calls_use_independent_tickets(tmp_path):
    request = payload(tmp_path)
    # Exercise the actual native fixed-input wrapper and paired-call body on CPU.
    request["eval_config"]["measure_performance"] = False
    Model, init, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    reference = Model(*init())
    args = [torch.rand(2, 16) * .1 + .9]

    @timed_numerics.task_policy
    def run(request):
        passed = native.run_correctness_on_fixed_inputs(reference, copy.deepcopy(reference), args, {},
            device="cpu", config_label="unscored_native", seed=99)
        return {"compiled": True, "correctness": passed, "metadata": {"simulated_execution": True}}

    result = run(request)
    assert result["correctness"]
    assert len(result["metadata"]["timed_numerical_audit"]["audit"]["cases"]) == 4
    assert timed_numerics.issue(result, request) is None


def test_first_native_candidate_error_cannot_hide_behind_later_correct_outputs(tmp_path):
    from src.reliability import check_paired_calls

    request = payload(tmp_path)
    request["eval_config"]["measure_performance"] = False
    Model, init, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    reference = Model(*init())
    args = [torch.rand(2, 16) * .1 + .9]

    class FirstWrong(Model):
        def __init__(self):
            super().__init__(*init())
            self.calls = 0

        def forward(self, *args):
            self.calls += 1
            value = super().forward(*args)
            return value + .0001 if self.calls == 1 else value

    @timed_numerics.task_policy
    def run(request):
        return check_paired_calls(reference, FirstWrong(), args, seed=99, tolerance=1e-3, device="cpu",
            set_seed=torch.manual_seed, metadata={}, label="unscored_cold", record_first_call=True, clock=measurement.CLOCK,
            **timed_numerics.pair_hooks(reference, args, seed=99, device="cpu", set_seed=torch.manual_seed, first=True))

    with pytest.raises(ValueError, match="independent FP64"):
        run(request)


@pytest.mark.parametrize("phase", ["evaluate", "baseline", "paired_timing"])
def test_neuron_simulator_checks_first_warm_and_timed_outputs_without_an_input_profile(tmp_path, monkeypatch, phase):
    from test_device_adapters import SimulatedNeuron

    from kb_workflows import neuron_worker

    request = payload(tmp_path, level=3, pid=4, phase=phase)
    if phase != "baseline":
        request["kernel_src"] = request["ref_arch_src"] + "\nModelNew = Model\n"
    request["eval_config"].update(device_backend="neuron", num_correct_trials=1, paired_baseline_variant="neuron_eager")
    device = SimulatedNeuron()
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: device.caps())
    load = neuron_worker.load_source

    def small_inputs(source, name, **kwargs):
        namespace = load(source, name, **kwargs)
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: [torch.rand(2, 1, 32, 32)]
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_inputs)

    @timed_numerics.task_policy
    def run(request):
        return neuron_worker.evaluate(request, adapter=device)

    result = run(request)
    assert result.get("correctness", result.get("baseline_correctness")) is True, result
    assert timed_numerics.issue(result, request) is None
    assert result["metadata"]["simulated_execution"] is True
    rows = result["metadata"]["timed_numerical_audit"]["audit"]["cases"]
    assert sum(row["scope"] == "timing" for row in rows) == (39 if phase == "paired_timing" else 5)


def test_import_and_preview_policy_do_not_import_device_sdks(tmp_path):
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows import timed_numerics
assert timed_numerics.ACTIVE.get() is None
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True)
