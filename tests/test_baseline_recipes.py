"""Source-bound library math and backend simulations; no device experiments."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from kb_workflows import baseline_policies, baseline_rewrites, baseline_variants, engine
from kb_workflows.common import ROOT, fingerprint, problem_paths, read_json
from kb_workflows.config import RunConfig
from src.validation import compare_outputs


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


RECIPES = [(1, 12), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (5, 1), (5, 3), (5, 27), (5, 30)]


@pytest.mark.parametrize("level,pid", RECIPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("values", ["signed", "zero", "near_constant"])
def test_reviewed_math_preserves_actual_reference_formula_roles_and_precision(level, pid, dtype, values):
    from test_operator_oracles import small_case

    from kb_workflows import production
    from src.eval import load_original_model_and_inputs

    if level == 1:
        source, model, args = small_case(pid, dtype=dtype)
    else:
        source = problem_paths(5)[pid].read_text()
        Model, _, _ = load_original_model_and_inputs(source, {})
        model = Model()
        precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
        args = production.make_inputs(pid, precision=precision, profile="small", seed=42)
    if values != "signed":
        # Change only the declared free primary activation, leaving masks,
        # scale/eps and normalization affine roles intact.
        args[0].fill_(0 if values == "zero" else 16)
        if values == "near_constant":
            args[0][..., ::2] += .25
    with torch.no_grad():
        expected = model(*copy.deepcopy(args))
        item = baseline_rewrites.entry_for(level, pid, source)
        for recipe in item["rewrites"]:
            forward = baseline_rewrites.framework_rewrite(copy.deepcopy(model), recipe)
            actual = forward(*copy.deepcopy(args))
            assert compare_outputs(expected, actual, 1e-3 if dtype == torch.float32 else 1e-2) is None
            assert actual.dtype == expected.dtype


@pytest.mark.parametrize("level,pid", [(level, pid) for level, count in ((1, 100), (2, 100), (3, 50), (5, 41)) for pid in range(1, count + 1)])
def test_every_task_has_frozen_source_and_each_declared_recipe_has_a_graph_counterpart(level, pid):
    source = problem_paths(level)[pid].read_text()
    options = baseline_rewrites.extension_options(level, pid, source)
    assert baseline_rewrites.entry_for(level, pid, source)
    assert len(options) == len(set(options)) and len(options) % 2 == 0
    for name, plan in options.items():
        assert plan == baseline_rewrites.plan_for(level, pid, source, name)
        if not name.endswith("_cuda_graph"):
            counterpart = options[name + "_cuda_graph"]
            assert counterpart["rewrite"] == plan["rewrite"] and counterpart["backend_policy"] == plan["backend_policy"]
            assert counterpart["outer_graph"] and not plan["outer_graph"]
    with pytest.raises(ValueError, match="reference changed"):
        baseline_rewrites.extension_options(level, pid, source + "\n# changed\n")


@pytest.mark.parametrize("level,pid,variant", [(1, 54, "channels_last"), (1, 55, "channels_last"),
    (2, 3, "channels_last"), (2, 10, "channels_last_cudnn_benchmark"), (2, 33, "cublaslt")])
def test_layout_and_backend_wrappers_preserve_reference_outputs_on_small_cpu_inputs(level, pid, variant, monkeypatch):
    from _composite_cases import small_composite_case
    from test_operator_oracles import small_case

    source, model, args = small_case(pid) if level == 1 else small_composite_case(pid, dtype=torch.float32)
    old = {name: value.clone() for name, value in model.state_dict().items()}
    state, observed = {"blas": "default"}, []

    def blas(value=None):
        if value is not None:
            state["blas"] = value
        return state["blas"]

    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", blas)
    plan = baseline_rewrites.plan_for(level, pid, source, variant)
    forward = baseline_variants.prepare_baseline(copy.deepcopy(model), args, "cpu", variant, plan=plan)[0]
    before = torch.backends.cudnn.benchmark
    with torch.no_grad():
        torch.manual_seed(42)
        expected = model(*copy.deepcopy(args))
        torch.manual_seed(42)
        actual = forward(*copy.deepcopy(args))
        observed.append((torch.backends.cudnn.benchmark, state["blas"]))
    assert compare_outputs(expected, actual, 1e-4) is None
    assert observed == [(before, "default")]
    assert all(torch.equal(value, old[name]) for name, value in model.named_parameters())


@pytest.mark.parametrize("policy", [{"cudnn_benchmark": True}, {"blas_preference": "cublaslt"},
    {"memory_format": "channels_last", "cudnn_benchmark": True}])
@pytest.mark.parametrize("error", [False, True])
def test_backend_preferences_are_observed_inside_call_and_restored_even_on_error(policy, error, monkeypatch):
    original = torch.backends.cudnn.benchmark
    state, records = {"blas": "default"}, []

    def blas(value=None):
        if value is not None:
            state["blas"] = value
        return state["blas"]

    class Model(torch.nn.Module):
        def forward(self, x):
            records.append((torch.backends.cudnn.benchmark, state["blas"], x.is_contiguous(memory_format=torch.channels_last)))
            if error:
                raise RuntimeError("fixture error")
            return x + 1

    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", blas)
    forward = baseline_policies.apply(Model(), policy)
    if error:
        with pytest.raises(RuntimeError, match="fixture error"):
            forward(torch.ones(2, 3, 4, 5))
    else:
        forward(torch.ones(2, 3, 4, 5))
    assert records == [(True if policy.get("cudnn_benchmark") else original,
                       policy.get("blas_preference", "default"), "memory_format" in policy)]
    assert torch.backends.cudnn.benchmark == original and state["blas"] == "default"


def test_layout_copy_is_inside_timed_call_and_weight_conversion_is_setup(monkeypatch):
    from kb_workflows import measurement

    clock = [0.]
    original = torch.Tensor.contiguous

    def copy_with_cost(value, *args, **kwargs):
        clock[0] += .003
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", copy_with_cost)
    monkeypatch.setattr(measurement, "CLOCK", lambda: clock[0])
    forward = baseline_policies.apply(torch.nn.Identity(), {"memory_format": "channels_last"})
    samples = measurement.synchronized_times(forward, [torch.randn(2, 3, 4, 5)], "cpu", num_warmup=0, num_trials=2)
    assert samples == pytest.approx([3., 3.])


def test_recipe_receipt_is_required_for_selection_resume_and_pairing(tmp_path, monkeypatch):
    config = RunConfig(generation="local", hardware="test", gpu_arch="Ampere", runs_dir=str(tmp_path),
                       start=12, end=12, baseline_portfolio="extended", resume=True).validate()
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    seen = []

    def evaluate(config, tasks):
        for key, request in tasks:
            variant = request["eval_config"]["baseline_variant"]
            seen.append(variant)
            # The source-bound row-scaling receipt is intentionally omitted.
            yield key, {"mean": .001 if variant == "library_rewrite" else 2., "cold_start_ms": 3.,
                        "baseline_correctness": True,
                        "reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                        "baseline_plan": None if variant == "library_rewrite" else baseline_rewrites.plan_for(
                            1, 12, request["ref_arch_src"], variant)}

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    path = engine.record_baseline(config)
    entry = next(iter(read_json(path)["level1"].values()))
    rejected = entry["variants"]["library_rewrite"]
    assert rejected["mean"] == .001 and rejected["baseline_correctness"] is False
    assert "recipe" in rejected["metadata"]["error"] and entry["winning_baseline"] != "library_rewrite"
    assert entry["suite_complete"] and len(seen) == len(baseline_variants.variants_for(config, 12))
    assert engine.paired_variant(config, 12) != "library_rewrite"
    variant = "cublaslt"
    request = engine.task_payload(config, 12, task="baseline")
    request["eval_config"]["baseline_variant"] = variant
    positive = {"baseline_correctness": True, "reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                "baseline_plan": baseline_rewrites.plan_for(1, 12, request["ref_arch_src"], variant)}
    assert baseline_rewrites.issue(positive, request) is None
    positive["baseline_plan"]["backend_policy"]["blas_preference"] = "default"
    assert baseline_rewrites.issue(positive, request)


def test_baseline_catalog_and_cli_preview_do_not_import_sdks(tmp_path):
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.config import RunConfig
from kb_workflows.baseline_variants import variants_for
from kb_workflows.cli import main
config=RunConfig(baseline_portfolio='extended').validate()
assert 'cublaslt' in variants_for(config,12)
assert main(['--baseline-portfolio','extended','--dry-run','--start','1','--end','100']) == 0
assert not {{'torch','triton','harbor','minisweagent','nki','torch_neuronx'}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["config"]["baseline_portfolio"] == "extended"
    assert report["baseline_variants"]["declared_variants"] > 600


@pytest.mark.parametrize("variant,pid", [("cublaslt", 12), ("channels_last", 54), ("library_rewrite", 40)])
def test_actual_native_entrypoint_rejects_wrong_policy_outputs_before_any_timing(tmp_path, monkeypatch, variant, pid):
    from kb_workflows import evaluate_task, measurement
    from src import eval as native

    class Reference(torch.nn.Module):
        def to(self, **kwargs):
            return self

        def forward(self):
            return torch.ones(2)

    config = RunConfig(runs_dir=str(tmp_path), start=pid, end=pid, baseline_portfolio="extended").validate()
    request = engine.task_payload(config, pid, task="baseline")
    request["eval_config"]["baseline_variant"] = variant
    monkeypatch.setattr(native, "load_original_model_and_inputs", lambda *a: (Reference, lambda: [], lambda: []))
    monkeypatch.setattr(baseline_policies, "apply", lambda *a: lambda: torch.zeros(2))
    monkeypatch.setattr(baseline_rewrites, "framework_rewrite", lambda *a: lambda: torch.zeros(2))
    monkeypatch.setattr(measurement, "synchronize", lambda device: None)
    monkeypatch.setattr(evaluate_task, "configure_tf32", lambda *a: None)
    monkeypatch.setattr(measurement, "synchronized_times", lambda *a, **kw: pytest.fail("Wrong baseline entered timing"))
    result = evaluate_task.baseline(request)
    assert result["correctness"] is False and "Baseline correctness failed" in result["metadata"]["error"]


@pytest.mark.parametrize("tamper", ["missing", "source", "recipe", "task", "graph", "variant"])
def test_saved_recipe_artifacts_reject_missing_changed_and_relabelled_evidence(tamper):
    source = problem_paths(5)[3].read_text()
    variant = "library_rewrite_production_var_mean_cuda_graph"
    plan = baseline_rewrites.plan_for(5, 3, source, variant)
    result = {"baseline_correctness": True, "baseline_variant": variant,
              "reference_sha256": fingerprint({"source": source}), "baseline_plan": plan}
    assert baseline_rewrites.artifact_issue(result) is None
    if tamper == "missing":
        result.pop("baseline_plan")
    elif tamper == "source":
        result["reference_sha256"] = "missing"
    elif tamper == "recipe":
        plan["rewrite"] = "production_layer_norm"
    elif tamper == "task":
        plan["task"] = [5, 27]
    elif tamper == "graph":
        plan["outer_graph"] = False
    else:
        result["baseline_variant"] = "library_rewrite"
    assert baseline_rewrites.artifact_issue(result)


@pytest.mark.parametrize("variant", ["neuron_eager", "neuron_compile", "neuron_library"])
def test_neuron_artifacts_keep_their_separate_source_validated_portfolio(variant):
    assert baseline_rewrites.artifact_issue({"baseline_correctness": True, "baseline_variant": variant}) is None
    with pytest.raises(ValueError):
        baseline_rewrites.plan_for(5, 3, problem_paths(5)[3].read_text(), variant)
