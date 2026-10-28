"""Reviewed scan shapes, masks and product signal on CPU; no original allocation."""

import copy
import importlib.util

import pytest
import torch

from kb_workflows.common import ROOT, problem_paths
from kb_workflows.numerics.numerical import PrecisionContract, precision_gate, signal_audit
from kb_workflows.numerics.operator_oracles import execute
from kb_workflows.scan_cases import TASKS, cases
from src import eval as native
from src.eval import load_original_model_and_inputs


@pytest.mark.parametrize("pid", TASKS)
@pytest.mark.parametrize("dtype,seed", [(torch.float32, 42), (torch.float16, 123), (torch.bfloat16, 456)])
def test_scan_cases_preserve_legal_shapes_roles_and_match_independent_fp64(pid, dtype, seed):
    source = problem_paths(1)[pid].read_text()
    Model, _, _ = load_original_model_and_inputs(source, {})
    model = Model(1)
    first = cases(pid, seed=seed)
    repeated = cases(pid, seed=seed)
    assert [row["label"] for row in first] == ["odd", "strided", "rank3", "singleton", "empty_axis"]
    for row, again in zip(first, repeated):
        assert all(torch.equal(a, b) for a, b in zip(row["inputs"], again["inputs"]))
        args = [value.to(dtype=dtype) if value.is_floating_point() else value for value in row["inputs"]]
        output = model(*args)
        truth = execute(pid, model, args)
        if dtype == torch.float32:
            policy = PrecisionContract(5e-6, 1e-7, 5e-5, 5e-7, 1e-6)
        else:
            from kb_workflows.numerics.specs import spec_for
            policy = PrecisionContract(**spec_for(1, pid, source)["precision"][str(dtype).removeprefix("torch.")])
        assert precision_gate(output, output.clone(), truth, policy)["passed"]
        assert output.shape == args[0].shape
        if pid == 93:
            assert args[1].dtype == torch.bool and args[1].shape == args[0].shape
            assert not args[1][0].any() and args[1][-1].all()
        if pid == 90 and output.numel():
            assert signal_audit(output, absolute_tolerance=1e-3)["informative"]
        if row["label"] == "strided":
            assert row["inputs"][0].stride(-1) == 2
            if pid == 93:
                assert row["inputs"][1].stride(-1) == 2


@pytest.mark.parametrize("pid", TASKS)
def test_actual_hidden_file_keeps_legacy_distributions_and_appends_family_cases(pid, monkeypatch):
    path = ROOT / "hidden_tests/level1" / f"{pid}_hidden.py"
    spec = importlib.util.spec_from_file_location(f"scan_hidden_{pid}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tiny = [torch.arange(12).reshape(3, 4).float()]
    if pid == 93:
        tiny.append(torch.ones(3, 4, dtype=torch.bool))
    monkeypatch.setattr(module, "_get_inputs", lambda: copy.deepcopy(tiny))
    configs = module.get_hidden_inputs()
    legacy_count = 3 if pid == 90 else 4
    assert len(configs) == legacy_count + 5
    factors = [1, 0.01, -1] if pid == 90 else [1, 3, 0.01, -1]
    assert all(torch.equal(values[0], tiny[0] * scale) for values, scale in zip(configs[:legacy_count], factors))
    if pid == 93:
        assert all(values[1].dtype == torch.bool for values in configs)
    Model, _, _ = load_original_model_and_inputs(problem_paths(1)[pid].read_text(), {})
    monkeypatch.setattr(native, "load_hidden_inputs", lambda _: module.get_hidden_inputs())
    metadata = {}
    assert native.run_hidden_correctness_check(Model(1), Model(1), str(path), metadata, device="cpu")
    assert metadata["hidden_completed_checks"] == 3 * len(configs)


def test_hidden_generator_retains_scan_family_cases_without_any_original_input_allocation(monkeypatch):
    from scripts import generate_hidden_inputs as generator

    source = problem_paths(1)[92]
    Model, _, _ = load_original_model_and_inputs(source.read_text(), {})
    monkeypatch.setattr(generator, "execute_problem_file", lambda _: {
        "Model": Model, "get_init_inputs": lambda: [1], "get_inputs": lambda: [torch.ones(3, 7)]})
    monkeypatch.setattr(generator, "check_config", lambda inputs, model: (bool(torch.isfinite(model(*inputs)).all()), ""))
    text, filtered = generator.generate_hidden_file_content(str(source), 92, 1)
    assert 'cases(92, seed=torch.initial_seed())' in text and filtered == []
    namespace = {"__file__": str(ROOT / "hidden_tests/level1/92_hidden.py")}
    exec(compile(text, "generated-hidden-cpu-fixture", "exec"), namespace)
    namespace["_get_inputs"] = lambda: [torch.ones(3, 7)]
    assert len(namespace["get_hidden_inputs"]()) == 9
