"""Model value portfolios on meta and explicit small CPU calibration controls."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from kb_workflows import engine
from kb_workflows.common import ROOT, fingerprint, problem_paths
from kb_workflows.config import RunConfig
from kb_workflows.numerics import input_calibration, input_calibration_task, input_profiles
from src import eval as native
from src.input_groups import copy_inputs, layout
from tests.numerics.test_input_calibration import CPUAdapter, synthetic_format_fixture


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("recipe", ["identity", "normal_small", "normal_unit", "symmetric_gain4"])
def test_recurrent_value_generation_preserves_shared_backing_offsets_objects_and_precision(dtype, recipe):
    backing = torch.arange(80, dtype=dtype)
    x, h = backing[7:31].reshape(2, 3, 4), backing[10:18].reshape(2, 4)
    args = [x, h]
    before = layout(args)
    torch.manual_seed(991)
    result = input_profiles.transform(args, 34, recipe, level=3)
    assert layout(result) == before and torch.equal(backing, torch.arange(80, dtype=dtype))
    assert result[0].untyped_storage()._cdata == result[1].untyped_storage()._cdata
    assert result[0].untyped_storage()._cdata != backing.untyped_storage()._cdata
    assert all(value.dtype == dtype and torch.isfinite(value).all() for value in result)
    assert torch.equal(result[0].flatten()[3:11], result[1].flatten())


def calibrate_model(tmp_path, level, pid, *, constant=False):
    config = RunConfig(level=level, generation="local", start=pid, end=pid, runs_dir=str(tmp_path), oracle_memory_mb=64).validate()
    payload = engine.task_payload(config, pid, task="calibrate_inputs")

    def loader(source, namespace):
        Model, init, _ = native.load_original_model_and_inputs(source, namespace)
        if constant:
            Original = Model

            class Constant(Original):
                def __init__(self, *args):
                    super().__init__(*args)
                    with torch.no_grad():
                        self.conv.weight.zero_()
                        self.conv.bias.zero_()
                        self.bias.fill_(1)

            Model = Constant
        def factory():
            if level == 2:
                shape = (2, 3, 5, 7, 7) if pid == 83 else (2, 64, 7, 9)
                return [torch.rand(shape)]
            if pid == 4:
                return [torch.rand(2, 1, 32, 32)]
            return [torch.rand(5, 2, 1024), torch.rand(2, 256)]
        return Model, init, factory

    record = input_calibration_task.calibrate(payload, adapter=CPUAdapter(), loader=loader)["record"]
    return payload, record


@pytest.mark.parametrize("level,pid", [(2, 1), (3, 4), (3, 34)])
def test_reference_only_calibration_uses_source_frozen_composite_and_model_truth(tmp_path, level, pid):
    payload, record = calibrate_model(tmp_path, level, pid)
    assert record["qualified"], record
    assert record["candidate_executed"] is False and record["timings_collected"] is False
    assert record["selection_policy"] == input_profiles.selection_policy(level, pid)
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True)
    rows = record["attempts"][-1]["cases"]
    assert {row["forward_seed"] for row in rows} == {42}
    assert len({row["output_probe_sha256"] for row in rows}) >= 2
    assert all(row["oracle_receipt"]["reference_numerical_forward_called"] is False for row in rows)
    for damage in ("forward_seed", "program", "no_outputs", "policy"):
        changed = copy.deepcopy(record)
        case = changed["attempts"][-1]["cases"][0]
        if damage == "forward_seed":
            case["forward_seed"] = 999
        elif damage == "program":
            case["oracle_receipt"]["program_sha256"] = "0" * 64
        elif damage == "no_outputs":
            for row in changed["attempts"][-1]["cases"]:
                row["output_probe_sha256"] = "0" * 64
        else:
            changed["selection_policy"]["minimum_fraction"] = 0
        changed["receipt_sha256"] = fingerprint({key: value for key, value in changed.items() if key != "receipt_sha256"})
        with pytest.raises(ValueError):
            input_calibration.validate_record(changed, payload["ref_arch_src"], allow_simulated=True)


def test_zero_formula_keeps_all_failed_profiles_and_fixed_forward_rng_masks(tmp_path):
    payload, record = calibrate_model(tmp_path, 2, 83)
    assert record["qualified"] is False and record["selected_recipe"] is None
    assert [row["recipe"] for row in record["attempts"]] == list(input_profiles.recipes(83, 2))
    assert all(row["signal"]["informative"] is False for attempt in record["attempts"] for row in attempt["cases"])
    for attempt in record["attempts"]:
        masks = [row["oracle_receipt"]["masks"] for row in attempt["cases"]]
        assert masks[0] and masks[0] == masks[1] == masks[2]
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True, require_qualified=False)


def test_constant_nonzero_reference_cannot_qualify_from_input_diversity_alone(tmp_path):
    payload, record = calibrate_model(tmp_path, 2, 1, constant=True)
    assert record["qualified"] is False and record["selected_recipe"] is None
    for attempt in record["attempts"]:
        assert all(row["reference_gate"]["passed"] and row["signal"]["informative"] for row in attempt["cases"])
        assert len({row["input_probe_sha256"] for row in attempt["cases"]}) == 3
        assert len({row["output_probe_sha256"] for row in attempt["cases"]}) == 1
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True, require_qualified=False)


def test_normal_extreme_profile_repairs_quantized_uniform_max_without_changing_the_task(tmp_path):
    config = RunConfig(start=49, end=49, precision="fp16", generation="local", runs_dir=str(tmp_path), oracle_memory_mb=8).validate()
    payload = engine.task_payload(config, 49, task="calibrate_inputs")

    def loader(source, namespace):
        Model, init, _ = native.load_original_model_and_inputs(source, namespace)
        return Model, init, lambda: [torch.rand(1, 32768)]

    record = input_calibration_task.calibrate(payload, adapter=CPUAdapter(), loader=loader)["record"]
    assert record["qualified"] is True and record["selected_recipe"] == "normal", record
    assert [row["recipe"] for row in record["attempts"]] == ["identity", "signed", "positive", "normal"]
    assert all(len({row["output_probe_sha256"] for row in attempt["cases"]}) == 1 for attempt in record["attempts"][:-1])
    input_calibration.validate_record(record, payload["ref_arch_src"], allow_simulated=True)


@pytest.mark.parametrize("level", [2, 3])
def test_model_calibration_preview_is_sdk_light_and_never_starts_experiments(tmp_path, level):
    config = tmp_path / "preview.toml"
    config.write_text(f'[run]\nlevel={level}\nstart=1\nend={100 if level == 2 else 50}\n')
    output = tmp_path / "unrun.json"
    script = f'''import sys,json
sys.path.insert(0,{str(ROOT)!r})
from scripts.calibrate import main
assert main(["inputs","--config",{str(config)!r},"--output",{str(output)!r}]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)["problems"]) == (100 if level == 2 else 50)
    assert not output.exists()


@pytest.mark.parametrize("level,pid", [(2, 1), (3, 4)])
def test_model_frozen_profiles_wrap_actual_nominal_factories_but_not_other_sources(tmp_path, monkeypatch, level, pid):
    from kb_workflows.evaluation import measurement

    payload, record = calibrate_model(tmp_path, level, pid)
    payload["task"] = "evaluate"
    payload["eval_config"].update(input_profile=synthetic_format_fixture(record), numerical_validation=True, measure_performance=False)
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: record["device"])

    @input_calibration.task_policy
    def worker(request):
        namespace = {}
        _, _, factory = native.load_original_model_and_inputs(request["ref_arch_src"], namespace)
        namespace.update(batch_size=2, height=7, width=9)
        values = factory()
        assert len(values) == 1
        with pytest.raises(ValueError, match="another reference"):
            input_profiles.wrap_factory(request["ref_arch_src"] + "\n# another", factory)
        return {"correctness": True, "metadata": {"device_fingerprint": record["device"]}}

    # LeNet's factory has no height/width globals and already declares 32x32;
    # only its batch is changed in this explicit software fixture.
    result = worker(payload)
    assert input_calibration.issue(result, payload) is None
    assert result["metadata"]["input_calibration"]["factory_calls"] == 1


def test_model_timed_ticket_is_required_and_uses_pre_reference_state_and_rng(tmp_path):
    payload, record = calibrate_model(tmp_path, 3, 34)
    payload["eval_config"].update(input_profile=synthetic_format_fixture(record), numerical_validation=True)
    source = problem_paths(3)[34].read_text()
    Model, init, _ = native.load_original_model_and_inputs(source, {})
    model = Model(*init())
    args = [torch.rand(5, 2, 1024), torch.rand(2, 256)]
    with input_profiles.context(source, 34, record["selected_recipe"], level=3) as state, torch.no_grad():
        state.update(payload=payload, independent_outputs_verified=0)
        rng = torch.get_rng_state().clone()
        prepared = input_calibration.prepare_output(model, args)
        assert torch.equal(rng, torch.get_rng_state())
        expected = model(*copy_inputs(args))
        input_calibration.verify_output(model, args, expected, expected.clone(), prepared=prepared)
        assert state["independent_outputs_verified"] == 1
        with pytest.raises(ValueError, match="pre-forward"):
            input_calibration.verify_output(model, args, expected, expected)
        changed = copy.deepcopy(prepared)
        changed["reference_id"] = 0
        with pytest.raises(ValueError, match="another reference"):
            input_calibration.verify_output(model, args, expected, expected, prepared=changed)


class StatefulRandom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.tensor(0.))

    def forward(self, x):
        self.counter.add_(1)
        return x + self.counter + torch.rand_like(x)


def counter_oracle(monkeypatch):
    """An unscored CPU oracle fixture for exercising orchestration, not L2 math."""
    from kb_workflows.numerics import specs as reference_oracles

    def prepare(model, args, **kwargs):
        class Oracle:
            estimated_host_bytes = 1

            def execute(self):
                return args[0].double() + model.counter.double() + 1 + torch.rand_like(args[0]).double()

        return Oracle()

    monkeypatch.setattr(reference_oracles, "prepare_oracle", prepare)


def test_timed_reference_ticket_tracks_continuous_state_and_restores_forward_rng(tmp_path, monkeypatch):
    from kb_workflows.evaluation.measurement import synchronized_times

    payload, record = calibrate_model(tmp_path, 2, 1)
    payload["eval_config"]["input_profile"] = synthetic_format_fixture(record)
    counter_oracle(monkeypatch)
    reference, candidate = StatefulRandom(), StatefulRandom()
    args = [torch.ones(2, 4)]
    with input_profiles.context(payload["ref_arch_src"], 1, record["selected_recipe"], level=2) as state:
        state.update(payload=payload, independent_outputs_verified=0)

        def verify(values, output):
            ticket = input_calibration.prepare_output(reference, values)
            expected = reference(*copy_inputs(values))
            input_calibration.verify_output(reference, values, expected, output, prepared=ticket)

        before = torch.get_rng_state().clone()
        times = synchronized_times(candidate, args, "cpu", num_trials=2, verifier=verify)
        after = torch.get_rng_state().clone()
        assert len(times) == 2 and state["independent_outputs_verified"] == 5
        assert reference.counter.item() == candidate.counter.item() == 5
        torch.set_rng_state(before)
        for _ in range(5):
            torch.rand_like(args[0])
        assert torch.equal(torch.get_rng_state(), after)


def test_neuron_style_pair_preparation_observes_each_current_reference_state(tmp_path, monkeypatch):
    from kb_workflows.evaluation.determinism import preserve_rng
    from src.reliability import check_paired_calls

    payload, record = calibrate_model(tmp_path, 2, 1)
    payload["eval_config"]["input_profile"] = synthetic_format_fixture(record)
    counter_oracle(monkeypatch)
    reference, candidate = StatefulRandom(), StatefulRandom()
    args, states, ticket = [torch.ones(2, 4)], [], None
    with input_profiles.context(payload["ref_arch_src"], 1, record["selected_recipe"], level=2) as state:
        state.update(payload=payload, independent_outputs_verified=0)

        def prepare_pair():
            nonlocal ticket
            states.append(reference.counter.item())
            with preserve_rng("cpu"):
                torch.manual_seed(991)
                ticket = input_calibration.prepare_output(reference, args)

        def verify_pair(values, outputs):
            for output in outputs.values():
                input_calibration.verify_output(reference, values, outputs["reference"], output, prepared=ticket)

        assert check_paired_calls(reference, candidate, args, seed=991, tolerance=1e-6, device="cpu", set_seed=torch.manual_seed,
            metadata={}, label="stateful-rng", prepare_pair=prepare_pair, verify_pair=verify_pair) is None
        assert states == [0, 1] and state["independent_outputs_verified"] == 4


def test_timing_input_snapshots_preserve_real_shared_views():
    from kb_workflows.evaluation.measurement import synchronized_times
    from src.validation import compare_outputs

    class Alias(torch.nn.Module):
        def forward(self, a, b):
            assert a.storage_offset() == 2 and b.storage_offset() == 5
            assert a.untyped_storage()._cdata == b.untyped_storage()._cdata
            a.add_(1)
            return b.clone()

    base = torch.arange(16.)
    args = [base[2:8], base[5:11]]

    def verify(values, output):
        assert compare_outputs(Alias()(*copy_inputs(values)), output, 0) is None

    times = synchronized_times(Alias(), args, "cpu", num_trials=2, verifier=verify)
    assert len(times) == 2 and torch.equal(base, torch.arange(16.))


@pytest.mark.parametrize("phase", ["evaluate", "baseline", "paired_timing"])
def test_model_profile_reaches_neuron_timing_and_both_native_pair_orders_on_simulator(tmp_path, monkeypatch, phase):
    from kb_workflows.common import kernel_name
    from kb_workflows.evaluation import measurement
    from kb_workflows.hardware import neuron_worker
    from tests.hardware.test_device_adapters import SimulatedNeuron

    original, record = calibrate_model(tmp_path, 3, 4)
    source = original["ref_arch_src"]
    config = RunConfig(level=3, start=4, end=4, generation="local", device_backend="neuron", hardware="Trainium2",
        enable_ncu=False, runs_dir=str(tmp_path / "runtime"), numerical_validation=True,
        num_correct_trials=1, num_perf_trials=2, paired_trials=10, oracle_memory_mb=64).validate()
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(3, 4)).write_text(source + "\nModelNew = Model\n")
    payload = engine.task_payload(config, 4, task=phase)
    payload["eval_config"]["input_profile"] = synthetic_format_fixture(record)
    if phase == "paired_timing":
        payload["eval_config"]["paired_baseline_variant"] = "neuron_eager"

    adapter = SimulatedNeuron()
    monkeypatch.setattr(measurement, "device_fingerprint", lambda _: adapter.caps())
    load = neuron_worker.load_source

    def small_inputs(text, name, **kwargs):
        namespace = load(text, name, **kwargs)
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: [torch.rand(2, 1, 32, 32)]
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_inputs)

    @input_calibration.task_policy
    def worker(request):
        return neuron_worker.evaluate(request, adapter=adapter)

    result = worker(payload)
    assert result.get("correctness", result.get("baseline_correctness")) is True, result
    assert result["metadata"]["simulated_execution"] is True
    assert input_calibration.issue(result, payload) is None
    assert result["metadata"]["input_calibration"]["independent_outputs_verified"] >= (39 if phase == "paired_timing" else 5)
