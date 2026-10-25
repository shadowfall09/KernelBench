"""L3 source algorithms on explicit CPU toy configurations and meta only."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from kb_workflows.evaluation.state_contract import _plain_tensors
from kb_workflows.numerics.model_oracles import Prepared
from kb_workflows.numerics.numerical import PrecisionContract, precision_gate
from src.input_groups import copy_inputs
from src.validation import clone_tensors
from tests.support._model_cases import SMALL_TASKS, small_model_case

POLICY = PrecisionContract(1e-8, 1e-9, 1e-7, 1e-9, 1e-9)


@pytest.mark.parametrize("pid", SMALL_TASKS)
@pytest.mark.parametrize("training", [True, False])
def test_source_model_algorithms_match_explicit_fp64_math_without_changing_state(pid, training):
    _, model, args = small_model_case(pid)
    model.train(training)
    saved = clone_tensors({**model.state_dict(), **_plain_tensors(model)})
    prepared = Prepared(model, args, problem_id=pid)
    with torch.no_grad():
        torch.manual_seed(42)
        truth = prepared.execute()
        torch.manual_seed(42)
        expected = copy.deepcopy(model)(*copy_inputs(args))
    gate = precision_gate(expected, clone_tensors(expected), truth, POLICY)
    assert gate["passed"] and gate["reference_qualified"], (pid, gate)
    assert all(torch.equal(value, saved[name]) for name, value in {**model.state_dict(), **_plain_tensors(model)}.items())
    assert prepared.receipt()["reference_numerical_forward_called"] is False


@pytest.mark.parametrize("pid", [28, 29, 30, 31, 32, 33, 34, *range(35, 43), 43, 44, 46, 47, 48, 49, 50])
def test_truth_does_not_call_source_or_fused_numerical_forward(pid, monkeypatch):
    import torch.nn.functional as F

    _, model, args = small_model_case(pid)
    prepared = Prepared(model, args, problem_id=pid)
    with torch.no_grad():
        expected = copy.deepcopy(model)(*copy_inputs(args))

    def forbidden(*args, **kwargs):
        raise AssertionError("Source/fused numerical forward entered the independent model oracle")

    for child in model.modules():
        monkeypatch.setattr(child, "forward", forbidden)
    if prepared.inner is not None:
        monkeypatch.setattr(prepared.inner.graph, "forward", forbidden)
    for name in ("linear", "layer_norm", "batch_norm", "softmax", "gelu", "normalize", "scaled_dot_product_attention",
                 "multi_head_attention_forward", "conv1d", "conv2d", "conv3d", "max_pool2d", "adaptive_avg_pool1d"):
        monkeypatch.setattr(F, name, forbidden)
    with torch.no_grad():
        truth = prepared.execute()
    assert precision_gate(expected, clone_tensors(expected), truth, POLICY)["passed"]


@pytest.mark.parametrize("pid", range(1, 51))
def test_frozen_model_arithmetic_accepts_reviewed_hidden_shapes_and_alternate_storage_precision(pid):
    from kb_workflows.numerics import model_input_cases
    from kb_workflows.numerics.operator_specs import spec_for
    from tests.support._model_cases import original_meta

    source, model, _ = original_meta(3, pid)
    spec = spec_for(3, pid, source)
    family = model_input_cases.reviewed_spec(3, pid, source)
    dimensions = next(iter(family["shape_variants"].values()))
    model.to(dtype=torch.bfloat16)
    args = [torch.empty(shape, device="meta", dtype=torch.bfloat16) for shape in dimensions]
    prepared = Prepared(model, args, problem_id=pid, expected_program=spec["program_sha256"])
    assert prepared.program_sha256 == spec["program_sha256"]


@pytest.mark.parametrize("pid", [14, 29, 30, 43, 44])
def test_conditional_native_random_masks_match_without_using_native_numeric_layers(pid):
    _, model, args = small_model_case(pid, torch.float32)
    for layer in model.modules():
        if type(layer) is torch.nn.Dropout:
            layer.p = .25
    prepared = Prepared(model, args, problem_id=pid)
    with torch.no_grad():
        torch.manual_seed(991)
        truth = prepared.execute()
        torch.manual_seed(991)
        expected = copy.deepcopy(model)(*copy_inputs(args))
    gate = precision_gate(expected, clone_tensors(expected), truth, PrecisionContract(1e-4, 1e-6, 1e-3, 1e-6, 1e-4))
    assert gate["passed"], gate
    assert len(prepared.masks) == len(prepared.events) and prepared.masks
    assert all(row["dtype"] == "torch.float32" for row in prepared.masks)


@pytest.mark.parametrize("pid", [48, 49])
def test_optional_initial_ssd_states_preserve_the_declared_chunk_recurrence(pid):
    _, model, args = small_model_case(pid)
    args.append(torch.randn(2, 1, 2, 3, 4, dtype=torch.float64))
    prepared = Prepared(model, args, problem_id=pid)
    with torch.no_grad():
        truth, expected = prepared.execute(), model(*copy_inputs(args))
    assert precision_gate(expected, clone_tensors(expected), truth, POLICY)["passed"]


def test_omitted_lstm_states_keep_the_source_default_fp32_failure_for_other_storage_types():
    _, model, args = small_model_case(35)
    with torch.no_grad(), pytest.raises(RuntimeError, match="same dtype"):
        model(args[0])
    # With FP32 storage the source's zero-state path is legal and independently checked.
    model.float()
    values = [args[0].float()]
    prepared = Prepared(model, values, problem_id=35)
    with torch.no_grad():
        truth, expected = prepared.execute(), model(*values)
    assert precision_gate(expected, expected.clone(), truth, PrecisionContract(1e-4, 1e-6, 1e-3, 1e-6, 1e-4))["passed"]


@pytest.mark.parametrize("damage", [None, "program", "source_forward", "mask", "dtype", "stride", "kept", "events", "mode", "boolean_probability"])
@pytest.mark.parametrize("pid", [14, 44])
def test_model_stochastic_receipts_are_replayed_against_frozen_toy_constructor_plans(tmp_path, monkeypatch, pid, damage):
    from kb_workflows.numerics import model_specs

    _, model, args = small_model_case(pid, torch.float32)
    for layer in model.modules():
        if type(layer) is torch.nn.Dropout:
            layer.p = .25
    prepared = Prepared(model, args, problem_id=pid)
    with torch.no_grad():
        prepared.execute()
    spec = {"program_sha256": prepared.program_sha256, "model_protocol": model_specs.PROTOCOL,
            "problem_id": pid, "program_modules": len(prepared.description["architecture"]["modules"])}
    path = tmp_path / "unscored-toy-model-programs.json"
    path.write_text(json.dumps({"protocol": "kernelbench_pro_model_programs_v1", "programs": {prepared.program_sha256: prepared.description}}))
    monkeypatch.setattr(model_specs, "PROGRAMS", path)
    receipt = copy.deepcopy(prepared.receipt())
    if damage == "program":
        receipt["program_sha256"] = "0" * 64
    elif damage == "source_forward":
        receipt["reference_numerical_forward_called"] = True
    elif damage == "mask":
        receipt["masks"].pop()
    elif damage in {"dtype", "stride", "kept"}:
        field, value = {"dtype": ("dtype", "torch.float16"), "stride": ("input_stride", []), "kept": ("kept", -1)}[damage]
        receipt["masks"][0][field] = value
    elif damage == "events":
        receipt["meta_dropout_calls"] = None
    elif damage == "mode":
        receipt["dropout_states"][0]["training"] = False
    elif damage == "boolean_probability":
        receipt["dropout_states"][0]["probability"] = True
    assert model_specs.receipt_valid(receipt, spec, precision="fp32", source_device="cpu") is (damage is None)


def original_request(tmp_path, memory=256):
    from kb_workflows import engine
    from kb_workflows.config import RunConfig
    from src.eval import load_original_model_and_inputs

    config = RunConfig(level=3, start=4, end=4, generation="local", runs_dir=str(tmp_path),
                       numerical_validation=True, oracle_memory_mb=memory).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    from kb_workflows.common import kernel_name

    (config.run_dir / kernel_name(3, 4)).write_text("CPU unit-test handoff; not an experiment")
    payload = engine.task_payload(config, 4)
    Model, init, _ = load_original_model_and_inputs(payload["ref_arch_src"], {})
    return payload, Model(*init())


@pytest.mark.parametrize("damage", [None, "model_receipt", "program", "inner", "meta_forward"])
def test_actual_l3_collector_requires_source_bound_model_evidence(tmp_path, damage):
    from kb_workflows.numerics import numerical_audit

    payload, model = original_request(tmp_path)
    collector = numerical_audit.Collector(payload)
    args = [torch.randn(2, 1, 32, 32)]
    before = torch.get_rng_state().clone()
    collector.observe(model, copy.deepcopy(model), args, seed=42, device="cpu", set_seed=torch.manual_seed,
                      tolerance=1e-2, label="actual-l3-input")
    assert torch.equal(before, torch.get_rng_state())
    document = collector.document()
    assert document["cases"][0]["status"] == "passed", document
    row = document["cases"][0]
    if damage == "model_receipt":
        del row["model_receipt"]
    elif damage == "program":
        row["model_receipt"]["program_sha256"] = "0" * 64
    elif damage == "inner":
        row["model_receipt"]["inner_receipt"] = None
    elif damage == "meta_forward":
        row["model_receipt"]["reference_numerical_forward_called"] = True
    result = numerical_audit.verdicts(document, [{"label": "actual-l3-input", "seed": 42}], payload)
    assert result["precision_contract"]["status"] == ("passed" if damage is None else "not_measured")


def test_l3_admission_precedes_actual_fp64_arithmetic(tmp_path, monkeypatch):
    from kb_workflows.numerics import numerical_audit

    payload, model = original_request(tmp_path, memory=1)
    monkeypatch.setattr(Prepared, "execute", lambda self: pytest.fail("Executed an over-budget model oracle"))
    collector = numerical_audit.Collector(payload)
    collector.observe(model, copy.deepcopy(model), [torch.randn(2, 1, 32, 32)], seed=42, device="cpu",
                      set_seed=torch.manual_seed, tolerance=1e-2, label="budget")
    row = collector.document()["cases"][0]
    assert row["status"] == "not_measured" and row["estimated_host_bytes"] > 1024**2


def test_l3_hidden_native_entrypoint_audits_every_legacy_and_additional_actual_case(tmp_path, monkeypatch):
    from kb_workflows.common import ROOT
    from kb_workflows.numerics import numerical_audit, operator_input_policy
    from src import eval as native

    payload, model = original_request(tmp_path)
    payload["eval_config"].update(use_hidden_tests=True, hidden_seeds=[991])
    monkeypatch.setattr(native, "load_hidden_inputs", lambda _: [[torch.randn(2, 1, 32, 32)]])

    @operator_input_policy.task_policy
    def worker(request):
        metadata = {}
        with numerical_audit.audit(request) as collector:
            passed = native.run_hidden_correctness_check(model, copy.deepcopy(model),
                str(ROOT / "hidden_tests/level3/4_hidden.py"), metadata, device="cpu", seeds=[991])
        metadata["numerical_audit"] = collector.document()
        return {"compiled": True, "correctness": passed, "metadata": metadata}

    result = worker(payload)
    assert result["correctness"] and operator_input_policy.issue(result, payload) is None
    metadata = result["metadata"]
    verdict = numerical_audit.verdicts(metadata["numerical_audit"], metadata["reliability_checks"], payload)
    assert all(row["status"] == "passed" for row in verdict.values()), verdict
    assert len(metadata["numerical_audit"]["cases"]) == 1 + len(payload["eval_config"]["operator_inputs"]["labels"])


def test_l3_preview_reads_all_50_plans_without_importing_device_sdks_or_allocating_inputs(tmp_path):
    from kb_workflows.common import ROOT

    script = f'''import sys,json
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--level","3","--numerical-validation","--runs-dir",{str(tmp_path / "absent")!r},"--dry-run"]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)["numerical_coverage"]["supported"]) == 50
    assert not (tmp_path / "absent").exists()
