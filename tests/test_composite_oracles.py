"""Independent L2 arithmetic: CPU toy execution and default-init meta checks."""

import copy
import json
import subprocess
import sys

import pytest
import torch
from _composite_cases import small_composite_case

from kb_workflows import composite_specs, engine, numerical_audit
from kb_workflows.common import ROOT, kernel_name, problem_paths
from kb_workflows.composite_oracles import Prepared
from kb_workflows.config import RunConfig
from kb_workflows.numerical import PrecisionContract, precision_gate
from kb_workflows.operator_specs import spec_for
from src.eval import load_original_model_and_inputs
from src.validation import clone_tensors

POLICY = PrecisionContract(1e-9, 1e-10, 1e-8, 1e-10, 1e-10)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("pid", range(1, 101))
def test_all_l2_graphs_match_actual_source_algorithms_in_explicit_small_configurations(pid):
    source, model, args = small_composite_case(pid)
    prepared = Prepared(model, args)
    before = clone_tensors(model.state_dict())
    with torch.no_grad():
        torch.manual_seed(42)
        truth = prepared.execute()
        torch.manual_seed(42)
        expected = copy.deepcopy(model)(*clone_tensors(args))
    gate = precision_gate(expected, clone_tensors(expected), truth, POLICY)
    assert gate["passed"] and gate["reference_qualified"], (pid, gate)
    assert prepared.receipt()["reference_numerical_forward_called"] is False
    assert prepared.estimated_host_bytes > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert spec_for(2, pid, source)["family"] == "composite"


@pytest.mark.parametrize("pid", range(1, 101))
def test_every_original_l2_initialization_and_nominal_shape_matches_its_frozen_meta_plan(pid):
    source = problem_paths(2)[pid].read_text()
    spec = spec_for(2, pid, source)
    with torch.device("meta"):
        Model, init, inputs = load_original_model_and_inputs(source, {})
        model, args = Model(*init()), inputs()
    prepared = Prepared(model, args, expected_program=spec["program_sha256"])
    assert prepared.description == composite_specs.program_for(spec)
    assert len(prepared.description["nodes"]) == spec["program_nodes"]
    assert prepared.estimated_host_bytes > 0
    assert all(value.device.type == "meta" for value in model.parameters())


@pytest.mark.parametrize("pid", [1, 3, 4, 6, 11, 17, 21, 23, 28, 30, 33, 34, 38, 52, 66, 72, 79, 83, 85, 94])
def test_numeric_execution_does_not_call_reference_graph_or_native_fused_layer_forward(pid, monkeypatch):
    import torch.nn.functional as F

    _, model, args = small_composite_case(pid)
    prepared = Prepared(model, args)
    with torch.no_grad():
        torch.manual_seed(42)
        expected = copy.deepcopy(model)(*clone_tensors(args))

    def forbidden(*args, **kwargs):
        raise AssertionError("An independent composite oracle called reference/fused numerical forward")

    for child in model.modules():
        monkeypatch.setattr(child, "forward", forbidden)
    monkeypatch.setattr(prepared.graph, "forward", forbidden)
    for name in ("conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                 "batch_norm", "instance_norm", "group_norm", "layer_norm", "gelu", "mish", "hardswish", "hardtanh",
                 "leaky_relu", "softplus", "softmax", "max_pool1d", "max_pool2d", "max_pool3d",
                 "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"):
        monkeypatch.setattr(F, name, forbidden)
    with torch.no_grad():
        torch.manual_seed(42)
        truth = prepared.execute()
    assert precision_gate(expected, clone_tensors(expected), truth, POLICY)["passed"]


@pytest.mark.parametrize("pid", [1, 12, 28, 30, 33, 41, 66, 83, 99])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_low_precision_storage_and_native_dropout_masks_are_retained(pid, dtype):
    source, model, args = small_composite_case(pid, dtype)
    prepared = Prepared(model, args)
    with torch.no_grad():
        torch.manual_seed(42)
        truth = prepared.execute()
        torch.manual_seed(42)
        expected = copy.deepcopy(model)(*clone_tensors(args))
    policy = PrecisionContract(**spec_for(2, pid, source)["precision"][str(dtype).removeprefix("torch.")])
    gate = precision_gate(expected, clone_tensors(expected), truth, policy)
    assert gate["passed"] and gate["reference_qualified"], (pid, dtype, gate)
    assert all(row["dtype"] == str(dtype) for row in prepared.masks)


def request(tmp_path, pid=1, memory=64):
    config = RunConfig(generation="local", level=2, start=pid, end=pid, numerical_validation=True,
                       oracle_memory_mb=memory, runs_dir=str(tmp_path), run_name="composite-software").validate()
    config.run_dir.mkdir(exist_ok=True)
    (config.run_dir / kernel_name(2, pid)).write_text("candidate bytes are not executed by these collector unit tests")
    payload = engine.task_payload(config, pid)
    source = payload["ref_arch_src"]
    Model, init, _ = load_original_model_and_inputs(source, {})
    model = Model(*init())
    return payload, model


def test_actual_l2_collector_uses_frozen_program_precision_signal_and_rng_restoration(tmp_path):
    payload, model = request(tmp_path)
    collector = numerical_audit.Collector(payload)
    args = [torch.randn(2, 64, 7, 9)]
    before = torch.get_rng_state().clone()
    collector.observe(model, copy.deepcopy(model), args, seed=42, device="cpu", set_seed=torch.manual_seed, tolerance=1e-2, label="actual_case")
    assert torch.equal(before, torch.get_rng_state())
    document = collector.document()
    assert document["cases"][0]["status"] == "passed", document
    verdict = numerical_audit.verdicts(document, [{"label": "actual_case", "seed": 42}], payload)
    assert all(row["status"] == "passed" for row in verdict.values())
    for damage in ("missing", "program", "reference_forward", "nodes"):
        changed = copy.deepcopy(document)
        receipt = changed["cases"][0]["composite_receipt"]
        if damage == "missing":
            del changed["cases"][0]["composite_receipt"]
        elif damage == "program":
            receipt["program_sha256"] = "0" * 64
        elif damage == "reference_forward":
            receipt["reference_numerical_forward_called"] = True
        else:
            receipt["node_count"] += 1
        assert numerical_audit.verdicts(changed, [{"label": "actual_case", "seed": 42}], payload)["precision_contract"]["status"] == "not_measured"


def test_reference_zero_composite_is_not_qualified_by_equal_candidate_outputs(tmp_path):
    payload, model = request(tmp_path, pid=83)
    collector = numerical_audit.Collector(payload)
    collector.observe(model, copy.deepcopy(model), [torch.randn(2, 3, 5, 7, 7)], seed=42, device="cpu",
        set_seed=torch.manual_seed, tolerance=1e-2, label="constant_zero")
    document = collector.document()
    assert document["cases"][0]["status"] == "passed", document
    assert document["cases"][0]["composite_receipt"]["masks"]
    verdict = numerical_audit.verdicts(document, [{"label": "constant_zero", "seed": 42}], payload)
    assert verdict["precision_contract"]["status"] == "passed"
    assert verdict["reference_signal"]["status"] == "failed"


def test_composite_admission_occurs_before_fp64_state_copy_or_native_mask_sampling(tmp_path, monkeypatch):
    payload, model = request(tmp_path, memory=1)
    collector = numerical_audit.Collector(payload)
    monkeypatch.setattr(Prepared, "execute", lambda *a: pytest.fail("An over-budget full-input oracle was executed"))
    collector.observe(model, copy.deepcopy(model), [torch.randn(2, 64, 7, 9)], seed=42, device="cpu",
        set_seed=torch.manual_seed, tolerance=1e-2, label="over_budget")
    row = collector.document()["cases"][0]
    assert row["status"] == "not_measured" and row["estimated_host_bytes"] > 1024**2
    verdict = numerical_audit.verdicts(collector.document(), [{"label": "over_budget", "seed": 42}], payload)
    assert all(value["status"] == "not_measured" for value in verdict.values())


def test_changed_initialization_and_unreviewed_operation_never_fall_back_to_reference_forward():
    source, model, args = small_composite_case(1)
    with pytest.raises(ValueError, match="reviewed source plan"):
        Prepared(model, args, expected_program=spec_for(2, 1, source)["program_sha256"])

    class Unsupported(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)

    with pytest.raises(ValueError, match="No reviewed function arithmetic"):
        Prepared(Unsupported(), [torch.ones(2)])
    with pytest.raises(ValueError, match="No independent"):
        spec_for(3, 1, "not supported")


def test_l2_preview_is_sdk_light_and_does_not_create_an_oracle_or_allocate_inputs(tmp_path):
    script = f'''import sys,json
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--level","2","--start","1","--end","100","--numerical-validation","--runs-dir",{str(tmp_path / 'absent')!r},"--dry-run"]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)["numerical_coverage"]["supported"]) == 100
    assert not (tmp_path / "absent").exists()
    with pytest.raises(ValueError, match="level 1 or 2"):
        RunConfig(level=3, numerical_validation=True).validate()


@pytest.mark.parametrize("damage", [None, "mask_missing", "shape", "dtype", "probability", "kept", "hash", "mode", "identity", "meta_forward"])
def test_native_dropout_receipt_schema_is_replayed_on_an_explicit_toy_plan(tmp_path, monkeypatch, damage):
    _, model, args = small_composite_case(66, torch.float32)
    prepared = Prepared(model, args)
    with torch.no_grad():
        torch.manual_seed(42)
        prepared.execute()
    descriptor = prepared.description
    spec = {"program_sha256": prepared.program_sha256, "composite_protocol": composite_specs.PROTOCOL,
            "program_nodes": len(descriptor["nodes"]),
            "stochastic_nodes": [row["name"] for row in descriptor["nodes"] if row["op"] == "call_module" and row["target"].startswith("Dropout:")]}
    file = tmp_path / "toy-program-schema.json"
    file.write_text(json.dumps({"protocol": "kernelbench_pro_composite_programs_v1", "programs": {prepared.program_sha256: descriptor}}))
    monkeypatch.setattr(composite_specs, "PROGRAMS", file)
    receipt = copy.deepcopy(prepared.receipt())
    mask = receipt["masks"][0]
    if damage == "mask_missing":
        receipt["masks"] = []
    elif damage == "shape":
        mask["shape"][0] += 1
    elif damage == "dtype":
        mask["dtype"] = "torch.float64"
    elif damage == "probability":
        mask["probability"] = 0.8
    elif damage == "kept":
        mask["kept"] = mask["elements"] + 1
    elif damage == "hash":
        mask["mask_sha256"] = "not-a-mask-digest"
    elif damage == "mode":
        receipt["dropout_states"][0]["training"] = False
    elif damage == "identity":
        mask["node"] = "another_dropout"
    elif damage == "meta_forward":
        receipt["shape_propagation_device"] = "cpu"
    assert composite_specs.receipt_valid(receipt, spec, precision="fp32", source_device="cpu") is (damage is None)


def test_tensor_literals_preserve_native_storage_rounding_before_fp64_arithmetic():
    class TensorLiteral(torch.nn.Module):
        def forward(self, x):
            return torch.min(x, torch.tensor(1.00000003, device=x.device))

    model = TensorLiteral()
    args = [torch.full((2, 3), 2.0, dtype=torch.float64)]
    with torch.no_grad():
        expected = model(*args)
        actual = Prepared(model, args).execute()
    assert torch.equal(expected, actual) and torch.equal(actual, torch.ones_like(actual))


def test_inplace_activation_aliases_and_explicit_gelu_approximation_are_not_erased():
    class Alias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = torch.nn.ReLU(inplace=True)
            self.gelu = torch.nn.GELU(approximate="tanh")

        def forward(self, x):
            view = x.view(x.shape)
            return self.gelu(self.activation(x) + view)

    model = Alias()
    args = [torch.tensor([[-2.0, 2.699, 0.5]], dtype=torch.float64)]
    with torch.no_grad():
        expected = copy.deepcopy(model)(*clone_tensors(args))
        actual = Prepared(model, args).execute()
    assert precision_gate(expected, expected.clone(), actual, POLICY)["passed"]
