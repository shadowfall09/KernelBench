"""L2/L3 geometry on meta and small CPU checks; no model/device experiments."""

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from kb_workflows import engine, model_input_cases, numerical_audit, operator_input_policy, reference_revisions
from kb_workflows.common import ROOT, fingerprint, kernel_name, problem_paths
from kb_workflows.config import RunConfig
from kb_workflows.measurement import input_signature
from src import eval as native
from src.validation import clone_tensors

TASKS = [(2, pid) for pid in range(1, 101)] + [(3, pid) for pid in range(1, 51)]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def original_meta(level, pid):
    original = torch.linspace

    def constructor_constants(*args, **kwargs):
        # Swin stores at most 24 scalar stochastic-depth probabilities via
        # .item(). Only this bounded constant vector is materialized on CPU;
        # all model parameters, buffers, inputs and outputs remain on meta.
        assert len(args) >= 3 and args[2] <= 4096
        return original(*args, **{**kwargs, "device": "cpu"})

    source = problem_paths(level)[pid].read_text()
    with torch.device("meta"), patch.object(torch, "linspace", constructor_constants):
        Model, init, inputs = native.load_original_model_and_inputs(source, {})
        return source, Model(*init()), inputs()


@pytest.mark.parametrize("level,pid", TASKS)
def test_all_original_l2_l3_factories_and_initializations_accept_frozen_input_families_on_meta(level, pid):
    with torch.no_grad():
        source, model, nominal = original_meta(level, pid)
        spec = model_input_cases.reviewed_spec(level, pid, source)
        assert [list(value.shape) for value in nominal] == spec["nominal_shapes"]
        before = {name: (value.shape, value.dtype) for name, value in model.state_dict().items()}
        rows = list(model_input_cases.cases(spec, seed=123, memory_mb=4096, allocation_device="meta"))
        assert [row["label"] for row in rows] == list(model_input_cases.labels(spec))
        for row in rows:
            args = row["inputs"]
            assert all(value.device.type == "meta" for value in args)
            assert [list(value.shape) for value in args] == model_input_cases.shapes(spec, row["label"])
            assert row["estimated_input_bytes"] == 16 * sum(value.numel() for value in args)
            if row["label"] == "strided":
                assert all(value.stride(-1) == 2 and not value.is_contiguous() for value in args)
            output = copy.deepcopy(model)(*args)
            outputs = output if isinstance(output, (tuple, list)) else [output]
            assert all(value.device.type == "meta" for value in outputs)
        assert before == {name: (value.shape, value.dtype) for name, value in model.state_dict().items()}


@pytest.mark.parametrize("level,pid", TASKS)
def test_all_model_input_contracts_bind_the_source_and_complete_shape_family(level, pid):
    source = problem_paths(level)[pid].read_text()
    spec = model_input_cases.reviewed_spec(level, pid, source)
    plan = operator_input_policy.plan(level, pid, source, 256)
    assert plan["reference_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert plan["model_family_sha256"] == fingerprint(spec)
    assert plan["labels"] == list(model_input_cases.labels(spec))
    assert bool(plan["fixed_shape_reason"]) == ((level, pid) in {(3, 33), (3, 48), (3, 49)})
    with pytest.raises(ValueError, match="reference changed"):
        operator_input_policy.plan(level, pid, source + "\n# change", 256)


def test_sequence_state_residual_and_constructor_bound_dimensions_are_coupled():
    for pid in range(34, 43):
        spec = model_input_cases.reviewed_spec(3, pid, problem_paths(3)[pid].read_text())
        for label in ("odd", "alternative_shape"):
            shapes = model_input_cases.shapes(spec, label)
            if pid == 34:
                assert shapes[0][1] == shapes[1][0] and shapes[0][0] % 2
            elif pid <= 38:
                assert shapes[0][1] % 2
                assert all(shapes[0][0] == shape[1] for shape in shapes[1:])
            else:
                assert shapes[0][0] % 2 and shapes[0][1] == shapes[1][1]
            assert all(shape[-1] == nominal[-1] for shape, nominal in zip(shapes, spec["nominal_shapes"]))
    for pid in (3, 34):
        spec = model_input_cases.reviewed_spec(2, pid, problem_paths(2)[pid].read_text())
        assert all(shapes[0][-1] == spec["nominal_shapes"][0][-1] for shapes in spec["shape_variants"].values())
    spec = model_input_cases.reviewed_spec(2, 26, problem_paths(2)[26].read_text())
    assert all(second[2:] == [n * 2 for n in first[2:]] and first[0] == second[0]
               for first, second in spec["shape_variants"].values())


@pytest.mark.parametrize("pid", [33, 48, 49])
def test_fixed_constructor_shapes_are_explicit_and_never_shrunk_for_admission(pid, monkeypatch):
    spec = model_input_cases.reviewed_spec(3, pid, problem_paths(3)[pid].read_text())
    assert spec["fixed_shape_reason"] and spec["shape_variants"]["fixed_shape"] == spec["nominal_shapes"]
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: pytest.fail("must admit before input allocation"))
    with pytest.raises(ValueError, match="memory envelope; no resizing"):
        next(model_input_cases.cases(spec, seed=42, memory_mb=1))


@pytest.mark.parametrize("level,pid,numerical", [(2, 1, False), (2, 1, True), (3, 34, False)])
def test_cpu_hidden_entrypoint_runs_each_family_and_replays_actual_execution(tmp_path, monkeypatch, level, pid, numerical):
    source = problem_paths(level)[pid].read_text()
    Model, init, _ = native.load_original_model_and_inputs(source, {})
    model = Model(*init())
    spec = model_input_cases.reviewed_spec(level, pid, source)
    tiny = next(model_input_cases.cases(spec, seed=42))["inputs"]
    monkeypatch.setattr(native, "load_hidden_inputs", lambda _: [clone_tensors(tiny)])
    config = RunConfig(level=level, generation="local", runs_dir=str(tmp_path), start=pid, end=pid,
                       numerical_validation=numerical).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / kernel_name(level, pid)).write_text("CPU unit-test candidate")
    request = engine.task_payload(config, pid, hidden=True)

    @operator_input_policy.task_policy
    def worker(payload):
        metadata = {}
        with numerical_audit.audit(payload) if numerical else nullcontext() as collector:
            passed = native.run_hidden_correctness_check(model, copy.deepcopy(model),
                str(ROOT / f"hidden_tests/level{level}/{pid}_hidden.py"), metadata, device="cpu", seeds=(42, 123, 456))
        if numerical:
            metadata["numerical_audit"] = collector.document()
        return {"compiled": True, "correctness": passed, "metadata": metadata}

    result = worker(request)
    assert result["correctness"] and operator_input_policy.issue(result, request) is None
    assert result["metadata"]["hidden_completed_checks"] == 3 * (1 + len(model_input_cases.labels(spec)))
    if numerical:
        verdicts = numerical_audit.verdicts(result["metadata"]["numerical_audit"],
                                           result["metadata"]["reliability_checks"], request)
        assert all(row["status"] == "passed" for row in verdicts.values())
    for damage in ("missing_case", "native_call", "shape", "dtype"):
        bad = copy.deepcopy(result)
        records = bad["metadata"]["operator_inputs"]["cases"]
        if damage == "missing_case":
            records.pop()
        elif damage == "native_call":
            bad["metadata"]["reliability_checks"].pop()
        else:
            row = records[0]
            actual = next(item for item in bad["metadata"]["reliability_checks"] if item["label"] == row["label"])
            if damage == "shape":
                row["input_signature"][0]["shape"][0] += 1
                row["estimated_input_bytes"] = 16 * sum(math.prod(item["shape"]) for item in row["input_signature"])
            else:
                row["input_signature"][0]["dtype"] = "torch.bfloat16"
            actual["input_signature"] = copy.deepcopy(row["input_signature"])
        assert operator_input_policy.issue(bad, request)


def test_cpu_value_families_and_seed_replay_are_independent_of_model_outputs():
    spec = model_input_cases.reviewed_spec(3, 34, problem_paths(3)[34].read_text())
    first = list(model_input_cases.cases(spec, seed=42))
    same = list(model_input_cases.cases(spec, seed=42))
    changed = list(model_input_cases.cases(spec, seed=123))
    assert all(torch.equal(a, b) for x, y in zip(first, same) for a, b in zip(x["inputs"], y["inputs"]))
    assert not torch.equal(first[0]["inputs"][0], changed[0]["inputs"][0])
    for row in first:
        signature = input_signature(row["inputs"])
        assert model_input_cases.signature_matches(spec, row["label"], signature, "fp32")
        for value in row["inputs"]:
            assert torch.isfinite(value).all()
            if row["label"] == "near_constant":
                assert set(value.unique().tolist()) == {1, 1 + 1 / 128}
            if row["label"] == "boundary":
                assert torch.count_nonzero(value) == 1 and value.reshape(-1)[-1] == 8


def hidden_module(pid):
    path = ROOT / f"hidden_tests/level1/{pid}_hidden.py"
    spec = importlib.util.spec_from_file_location(f"hidden_role_{pid}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("pid", [98, 100])
def test_legacy_float_probability_and_label_domains_keep_all_variant_slots(monkeypatch, generated, pid):
    from scripts import generate_hidden_inputs as generator

    source = problem_paths(1)[pid]
    Model, _, _ = native.load_original_model_and_inputs(source.read_text(), {})
    tiny = ([torch.tensor([[.1, .2, .7], [.2, .3, .5]]), torch.tensor([[.2, .3, .5], [.4, 0., .6]])] if pid == 98 else
            [torch.tensor([[.25, .5, 1.], [.75, 1.25, 2.], [-1., 1., .5]]), torch.tensor([-1., 1., -1.])])
    if generated:
        monkeypatch.setattr(generator, "execute_problem_file", lambda _: {
            "Model": Model, "get_init_inputs": lambda: [], "get_inputs": lambda: clone_tensors(tiny)})
        monkeypatch.setattr(generator, "check_config", lambda inputs, model: (bool(torch.isfinite(model(*inputs)).all()), ""))
        text, _ = generator.generate_hidden_file_content(str(source), pid, 1)
        namespace = {"__file__": str(ROOT / f"hidden_tests/level1/{pid}_hidden.py")}
        exec(compile(text, "generated-role-preserving-hidden", "exec"), namespace)
        namespace["_get_inputs"] = lambda: clone_tensors(tiny)
        rows = namespace["get_hidden_inputs"]()
    else:
        module = hidden_module(pid)
        monkeypatch.setattr(module, "_get_inputs", lambda: clone_tensors(tiny))
        rows = module.get_hidden_inputs()
    assert len(rows) == (3 if pid == 98 else 5)
    if pid == 98:
        for predictions, targets in rows:
            assert (predictions > 0).all() and (targets >= 0).all()
            assert all(torch.allclose(value.sum(-1), torch.ones(value.shape[0])) for value in (predictions, targets))
            assert torch.isfinite(Model()(predictions, targets))
        assert not torch.allclose(rows[1][0], tiny[0]) and not torch.allclose(rows[2][0], tiny[0])
    else:
        for index, factor in enumerate((1, 3, .01, -1)):
            assert torch.equal(rows[index][0], tiny[0] * factor) and torch.equal(rows[index][1], tiny[1])
        assert set(rows[-1][1].tolist()) == {-1, 1}
        assert not torch.equal(rows[3][0] * rows[3][1], tiny[0] * tiny[1])
    binding = reference_revisions.binding(1, pid, source.read_text())
    assert binding["revision"] == 2


@pytest.mark.parametrize("level", [2, 3])
def test_cli_plans_are_sdk_light_and_cover_every_model_task(level):
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--level",{str(level)!r},"--dry-run"]) == 0
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    document = json.loads(result.stdout)
    assert document["operator_input_coverage"]["task_count"] == (100 if level == 2 else 50)


def test_l3_shared_policy_reaches_neuron_on_explicit_cpu_simulator(tmp_path, monkeypatch):
    from test_device_adapters import SimulatedNeuron

    from kb_workflows import neuron_worker

    source = problem_paths(3)[34].read_text()
    spec = model_input_cases.reviewed_spec(3, 34, source)
    tiny = next(model_input_cases.cases(spec, seed=42))["inputs"]
    config = RunConfig(level=3, start=34, end=34, generation="local", runs_dir=str(tmp_path),
                       device_backend="neuron", hardware="Trainium2", enable_ncu=False).validate()
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(3, 34)).write_text(source + "\nModelNew = Model\n")
    payload = engine.task_payload(config, 34, hidden=True)
    payload["eval_config"]["hidden_seeds"] = [991]
    load = neuron_worker.load_source

    def small_source(text, name, **kwargs):
        if name == "kernelbench_neuron_hidden":
            return {"get_hidden_inputs": lambda: [clone_tensors(tiny)]}
        namespace = load(text, name, **kwargs)
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: clone_tensors(tiny)
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_source)

    @operator_input_policy.task_policy
    def worker(request):
        return neuron_worker.evaluate(request, adapter=SimulatedNeuron())

    result = worker(payload)
    assert result["correctness"] and result["metadata"]["simulated_execution"]
    assert len(result["metadata"]["operator_inputs"]["cases"]) == 6
    assert operator_input_policy.issue(result, payload) is None


@pytest.mark.parametrize("damage", ["rank", "dimension", "fixed_reason", "variants", "constraints"])
def test_malformed_model_family_metadata_is_rejected_before_device_import(tmp_path, monkeypatch, damage):
    document = json.loads(model_input_cases.MANIFEST.read_text())
    row = document["tasks"]["3/34"]
    if damage == "rank":
        row["shape_variants"]["odd"][0].pop()
    elif damage == "dimension":
        row["shape_variants"]["odd"][0][0] = True
    elif damage == "fixed_reason":
        row["fixed_shape_reason"] = True
    elif damage == "variants":
        row["shape_variants"]["alternative_shape"] = copy.deepcopy(row["shape_variants"]["odd"])
    else:
        row["shape_constraints"] = ""
    path = tmp_path / "families.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(model_input_cases, "MANIFEST", path)
    with pytest.raises(ValueError, match="(Invalid|distinct)"):
        operator_input_policy.plan(3, 34, problem_paths(3)[34].read_text(), 256)
