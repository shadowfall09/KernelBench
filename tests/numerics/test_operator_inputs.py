"""Actual frozen L1 classes, CPU toy values, and original-init meta-only checks."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from kb_workflows import engine
from kb_workflows.common import ROOT, fingerprint, kernel_name, problem_paths
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation import tool_checks
from kb_workflows.numerics import numerical_audit, operator_cases, operator_input_policy, operator_oracles
from kb_workflows.numerics.numerical import PrecisionContract, precision_gate, signal_audit
from src import eval as native
from src.validation import clone_tensors


@pytest.mark.parametrize("pid", range(1, 101))
def test_all_l1_reviewed_shapes_and_values_match_independent_formulas(pid):
    from tests.numerics.test_operator_oracles import small_case

    _, model, _ = small_case(pid, torch.float64)
    state = clone_tensors(model.state_dict())
    seen = []
    with torch.no_grad():
        for row in operator_cases.cases(pid, model, seed=42):
            assert row["reference_parameters_replaced"] is False and row["nominal_timing_changed"] is False
            assert 0 < row["estimated_input_bytes"] <= 256 * 1024**2
            args = [native._process_input_tensor(value, "cpu", torch.float64) for value in row["inputs"]]
            ref = copy.deepcopy(model)
            truth = operator_oracles.execute(pid, ref, args)
            expected = ref(*clone_tensors(args))
            gate = precision_gate(expected, clone_tensors(expected), truth, PrecisionContract(2e-5, 1e-7, 1e-4, 1e-7, 2e-6))
            assert gate["passed"] and gate["reference_qualified"], (pid, row["label"], gate)
            if row["label"] == "strided":
                assert all(not value.is_contiguous() and value.stride()[-1] == 2 for value in args if isinstance(value, torch.Tensor))
            if pid == 13:
                assert all(torch.equal(value, value.t()) for value in args)
            if pid in {14, 15}:
                assert all(torch.equal(value, value.triu() if pid == 14 else value.tril()) for value in args)
            if pid == 93:
                assert args[1].dtype == torch.bool and args[0].shape == args[1].shape
            if pid == 95:
                assert args[1].dtype == torch.int64 and ((args[1] == -100) | ((args[1] >= 0) & (args[1] < args[0].shape[1]))).all()
            if pid == 98:
                assert (args[0] > 0).all() and (args[1] >= 0).all()
                assert all(torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-6) for value in args)
            if pid == 100:
                assert args[1].ndim == 1 and set(args[1].tolist()) <= {-1, 1}
                assert args[0].shape[0] == args[0].shape[1] == args[1].shape[0]
            seen.append(row["label"])
    assert seen == list(operator_cases.labels(pid))
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("pid", [19, 20, 21, 22, 28])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_declared_nonfinite_activations_preserve_masks_signs_and_null_error_measurements(pid, dtype):
    from tests.numerics.test_operator_oracles import small_case

    _, model, _ = small_case(pid, dtype)
    rows = {row["label"]: row for row in operator_cases.cases(pid, model, seed=42)}
    for label in ("nonfinite_mixed", "nonfinite_only"):
        args = [value.to(dtype=dtype) for value in rows[label]["inputs"]]
        actual = model(*args)
        truth = operator_oracles.execute(pid, model, args)
        gate = precision_gate(actual, actual.clone(), truth, PrecisionContract(1e-2, 1e-7, 1e-2, 1e-7))
        assert gate["passed"] and gate["reference_qualified"]
        assert not precision_gate(actual, torch.nan_to_num(actual), truth, PrecisionContract(1e-2, 1e-7, 1e-2, 1e-7))["passed"]
        if label == "nonfinite_only":
            leaf = gate["leaves"][0]
            assert leaf["kind"] == "nonfinite_only" and leaf["oracle_rms"] is None and leaf["norm_limit"] is None
            assert leaf["reference"]["rms_error"] is None
            assert signal_audit(actual, absolute_tolerance=0)["informative"] is False


def payload(tmp_path, pid=2, **overrides):
    config = RunConfig(generation="local", runs_dir=str(tmp_path), start=pid, end=pid, **overrides).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / kernel_name(1, pid)).write_text("unused candidate")
    return engine.task_payload(config, pid, hidden=True), config


def execute_hidden_cpu(tmp_path, monkeypatch, *, candidate=None, numerical=False):
    from tests.numerics.test_operator_oracles import small_case

    request, config = payload(tmp_path, numerical_validation=numerical)
    _, model, inputs = small_case(2)
    monkeypatch.setattr(native, "load_hidden_inputs", lambda _: [clone_tensors(inputs)])

    @operator_input_policy.task_policy
    def worker(value):
        metadata = {}
        context = numerical_audit.audit(value) if numerical else None
        if context:
            collector = context.__enter__()
        try:
            passed = native.run_hidden_correctness_check(model, candidate or copy.deepcopy(model),
                str(ROOT / "hidden_tests/level1/2_hidden.py"), metadata, device="cpu", seeds=(42, 123, 456))
        finally:
            if context:
                context.__exit__(None, None, None)
        if numerical:
            metadata["numerical_audit"] = collector.document()
        return {"compiled": True, "correctness": passed, "metadata": metadata}

    return request, worker(request), config


def test_real_hidden_entrypoint_keeps_legacy_cases_and_requires_every_new_case_and_seed(tmp_path, monkeypatch):
    request, result, _ = execute_hidden_cpu(tmp_path, monkeypatch)
    assert result["correctness"] is True
    assert result["metadata"]["hidden_completed_checks"] == 3 * (1 + 3 + len(operator_cases.labels(2)))  # Legacy + prior stress + reviewed families.
    records = result["metadata"]["operator_inputs"]["cases"]
    assert len(records) == 3 * len(operator_cases.labels(2)) and operator_input_policy.issue(result, request) is None
    assert [row["family_label"] for row in records[:len(operator_cases.labels(2))]] == list(operator_cases.labels(2))
    strided = next(row for row in records if row["family_label"] == "strided")
    assert all(row["stride"][-1] == 2 for row in strided["input_signature"])
    for damage in ("case", "label", "seed", "memory", "passed", "native_call", "plan", "signature", "empty_stride", "non_integer_seed"):
        changed = copy.deepcopy(result)
        doc = changed["metadata"]["operator_inputs"]
        if damage == "signature":
            doc["cases"][0]["input_signature"][0]["shape"][0] += 1
        elif damage == "empty_stride":
            doc["cases"][0]["input_signature"][0]["stride"] = []
        elif damage == "non_integer_seed":
            doc["cases"][0]["seed"] = float(doc["cases"][0]["seed"])
        elif damage == "case":
            doc["cases"].pop()
        elif damage in {"label", "seed", "passed"}:
            doc["cases"][0][damage] = {"label": "wrong", "seed": 0, "passed": False}[damage]
        elif damage == "memory":
            doc["cases"][0]["estimated_input_bytes"] = 10**12
        elif damage == "plan":
            doc["plan_sha256"] = "0" * 64
        else:
            changed["metadata"]["reliability_checks"] = []
        assert operator_input_policy.issue(changed, request)
        assert tool_checks.issue(changed, request)


def test_shape_specialized_candidate_cannot_pass_original_shape_and_skip_odd_hidden_shapes(tmp_path, monkeypatch):
    class Specializes(torch.nn.Module):
        def forward(self, a, b):
            return a @ b if a.shape == (5, 7) else torch.zeros((*a.shape[:-1], b.shape[-1]))

    request, result, _ = execute_hidden_cpu(tmp_path, monkeypatch, candidate=Specializes())
    assert result["correctness"] is False
    assert any(label.startswith("operator/") for label in result["metadata"]["hidden_failed_configs"])
    assert operator_input_policy.issue(result, request) is None  # A current negative result may resume.


def test_new_native_cases_enter_the_existing_independent_numerical_control(tmp_path, monkeypatch):
    request, result, _ = execute_hidden_cpu(tmp_path, monkeypatch, numerical=True)
    cases = result["metadata"]["reliability_checks"]
    verdicts = numerical_audit.verdicts(result["metadata"]["numerical_audit"], cases, request)
    assert all(row["status"] == "passed" for row in verdicts.values())
    assert len(result["metadata"]["numerical_audit"]["cases"]) == len(cases)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_conversion_keeps_positive_strides_for_actual_candidate_inputs(dtype):
    value = torch.randn(3, 34)[:, ::2]
    converted = native._process_input_tensor(value, "cpu", dtype)
    assert converted.dtype == dtype and converted.stride() == value.stride() and not converted.is_contiguous()
    assert torch.equal(converted, value.to(dtype))
    assert clone_tensors(converted).stride() == value.stride()
    mask = value > 0
    assert native._process_input_tensor(mask, "cpu", dtype).dtype == torch.bool


def test_input_plans_and_budget_are_sdk_light_source_bound_and_frozen(tmp_path):
    from kb_workflows.analysis import experiment, study_data
    from kb_workflows.evaluation import final_verification
    from kb_workflows.runtime.queue_worker import validate_payload

    request, config = payload(tmp_path)
    validate_payload(request)
    changed = copy.deepcopy(request)
    changed["eval_config"]["operator_inputs"]["labels"].pop()
    with pytest.raises(ValueError, match="input-family plan"):
        validate_payload(changed)
    previous = final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config)
    config.input_family_memory_mb += 1
    assert previous != (final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config))
    assert fingerprint(engine.task_payload(config, 2, hidden=True)) != fingerprint(request)
    file = tmp_path / "preview.toml"
    file.write_text('[run]\nlevel=1\nstart=1\nend=100\ninput_family_memory_mb=256\n')
    script = f'''import sys,json
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--config",{str(file)!r},"--dry-run"]) == 0
assert not {{"torch","nki","triton","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)["problems"]) == 100
    assert parse_config(["--config", str(file)]).input_family_memory_mb == 256


@pytest.mark.parametrize("pid", [33, 34])
def test_near_constant_fp32_normalization_retains_reference_self_qualification_failure(pid):
    from kb_workflows.numerics.specs import spec_for
    from tests.numerics.test_operator_oracles import small_case

    source, model, _ = small_case(pid)
    row = next(row for row in operator_cases.cases(pid, model, seed=42) if row["label"] == "near_constant")
    with torch.no_grad():
        truth = operator_oracles.execute(pid, model, row["inputs"])
        expected = model(*row["inputs"])
    policy = PrecisionContract(**spec_for(1, pid, source)["precision"]["float32"])
    gate = precision_gate(expected, expected.clone(), truth, policy)
    assert not gate["passed"] and not gate["reference_qualified"] and gate["reference_issues"]


def test_original_layernorm_dimensions_are_not_shrunk_when_input_admission_fails(monkeypatch):
    source = problem_paths(1)[40].read_text()
    with torch.device("meta"):
        Model, init, _ = native.load_original_model_and_inputs(source, {})
        model = Model(*init())
    shape = tuple(model.ln.normalized_shape)
    monkeypatch.setattr(torch, "randn", lambda *a, **k: pytest.fail("Allocated over-budget reviewed inputs"))
    with pytest.raises(ValueError, match="no resizing"):
        next(operator_cases.cases(40, model, seed=42, memory_mb=1))
    assert tuple(model.ln.normalized_shape) == shape


@pytest.mark.parametrize("pid", [2, 13, 90, 93, 95, 98, 99, 100])
def test_private_inputs_replay_their_seed_without_consuming_global_random_state(pid):
    from src.validation import compare_outputs
    from tests.numerics.test_operator_oracles import small_case

    _, model, _ = small_case(pid)
    before = torch.get_rng_state().clone()
    first = list(operator_cases.cases(pid, model, seed=42))
    assert torch.equal(before, torch.get_rng_state())
    second = list(operator_cases.cases(pid, model, seed=42))
    changed = list(operator_cases.cases(pid, model, seed=123))
    assert all(compare_outputs(a["inputs"], b["inputs"], 0) is None for a, b in zip(first, second))
    assert compare_outputs(first[0]["inputs"], changed[0]["inputs"], 0) is not None


def test_all_nan_checks_qualify_propagation_without_inventing_finite_precision_or_signal(tmp_path):
    from tests.numerics.test_operator_oracles import small_case

    request, _ = payload(tmp_path, pid=19, numerical_validation=True)
    _, model, _ = small_case(19)
    collector = numerical_audit.Collector(request)
    collector.observe(model, copy.deepcopy(model), [torch.full((3, 17), float("nan"))], seed=42,
        device="cpu", set_seed=torch.manual_seed, tolerance=1e-2, label="nonfinite")
    cases = [{"label": "nonfinite", "seed": 42}]
    verdict = numerical_audit.verdicts(collector.document(), cases, request)
    assert verdict["precision_contract"]["status"] == "passed" and verdict["reference_signal"]["status"] == "failed"
    collector.observe(model, copy.deepcopy(model), [torch.randn(3, 17)], seed=42,
        device="cpu", set_seed=torch.manual_seed, tolerance=1e-2, label="finite")
    cases.append({"label": "finite", "seed": 42})
    assert all(row["status"] == "passed" for row in numerical_audit.verdicts(collector.document(), cases, request).values())
    damaged = collector.document()
    damaged["cases"][0]["precision_gate"]["leaves"][0]["reference"]["rms_error"] = 0
    assert numerical_audit.verdicts(damaged, cases, request)["precision_contract"]["status"] == "not_measured"


def test_shared_operator_policy_reaches_neuron_hidden_calls_on_explicit_simulator(tmp_path, monkeypatch):
    from kb_workflows.hardware import neuron_worker
    from tests.hardware.test_device_adapters import SimulatedNeuron

    config = RunConfig(device_backend="neuron", hardware="Trainium2", generation="local", enable_ncu=False,
        level=1, start=19, end=19, runs_dir=str(tmp_path), run_name="software-neuron").validate()
    config.run_dir.mkdir()
    source = problem_paths(1)[19].read_text()
    (config.run_dir / kernel_name(1, 19)).write_text(source + "\nModelNew = Model\n")
    request = engine.task_payload(config, 19, hidden=True)
    request["eval_config"]["hidden_seeds"] = [991, 992]
    load = neuron_worker.load_source

    def small_source(text, name, **kwargs):
        if name == "kernelbench_neuron_hidden":
            return {"get_hidden_inputs": lambda: [[torch.randn(3, 17)]]}
        namespace = load(text, name, **kwargs)
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: [torch.randn(3, 17)]
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_source)

    @operator_input_policy.task_policy
    def worker(value):
        return neuron_worker.evaluate(value, adapter=SimulatedNeuron())

    result = worker(request)
    assert result["correctness"] and result["metadata"]["simulated_execution"]
    assert len(result["metadata"]["operator_inputs"]["cases"]) == 2 * len(operator_cases.labels(19))
    assert operator_input_policy.issue(result, request) is None
