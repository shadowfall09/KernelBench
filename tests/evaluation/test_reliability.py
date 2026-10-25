"""Adversarial regressions for the research-informed correctness protocol."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from kb_workflows import engine
from kb_workflows.common import ROOT, atomic_json, evaluator_digest
from kb_workflows.config import RunConfig
from src import eval as native
from src.reliability import HIDDEN_SEEDS, PROTOCOL, STRESS_TASKS, stress_cases, stress_enabled


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *inputs):
        return self.function(*inputs)


def fixed(reference, candidate, inputs, metadata=None):
    return native.run_correctness_on_fixed_inputs(
        reference, candidate, inputs, {} if metadata is None else metadata, device="cpu")


def test_candidate_runs_before_any_reference_result_exists():
    cached = {}

    def reference(x):
        cached["answer"] = x.square()
        return cached["answer"]

    candidate = Function(lambda x: cached.get("answer", torch.zeros_like(x)))
    metadata = {}
    assert not fixed(Function(reference), candidate, [torch.ones(4)], metadata)
    assert metadata["reliability_checks"][0]["orders"] == ["candidate_first"]


def test_reference_cannot_overwrite_candidate_output_before_comparison():
    buffer = torch.zeros(4)
    reference = Function(lambda x: buffer.copy_(x))
    candidate = Function(lambda x: buffer.copy_(x + 1))
    assert not fixed(reference, candidate, [torch.ones(4)])


def test_second_candidate_call_is_also_verified():
    calls = [0]

    def candidate(x):
        calls[0] += 1
        return x if calls[0] == 1 else x + 1

    metadata = {}
    assert not fixed(nn.Identity(), Function(candidate), [torch.ones(4)], metadata)
    assert metadata["reliability_checks"][0]["orders"] == ["candidate_first", "reference_first"]


def test_stateful_models_advance_equally_and_rng_is_replayed():
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.tensor(0))

        def forward(self, x):
            self.count += 1
            return x + self.count + torch.rand_like(x)

    reference, candidate = Stateful(), Stateful()
    assert fixed(reference, candidate, [torch.ones(8)])
    assert reference.count.item() == candidate.count.item() == 2


def test_candidate_backend_change_is_restored_before_reference():
    initial = torch.backends.cudnn.benchmark

    def candidate(x):
        torch.backends.cudnn.benchmark = not initial
        return x

    def reference(x):
        assert torch.backends.cudnn.benchmark is initial
        return x

    assert fixed(Function(reference), Function(candidate), [torch.ones(3)])
    assert torch.backends.cudnn.benchmark is initial


def test_reference_exception_invalidates_validation():
    def broken(x):
        raise ValueError("invalid trusted case")

    metadata = {}
    assert not fixed(Function(broken), nn.Identity(), [torch.ones(3)], metadata)
    assert metadata["reliability_checks"][0]["failure_kind"] == "reference_error"


@pytest.mark.parametrize("configs", [None, [], {}, [torch.ones(2)]])
def test_missing_empty_or_malformed_hidden_suite_never_passes(monkeypatch, configs):
    monkeypatch.setattr(native, "load_hidden_inputs", lambda path: configs)
    metadata = {}
    with pytest.raises(RuntimeError):
        native.run_hidden_correctness_check(nn.Identity(), nn.Identity(), "untrusted.py", metadata, device="cpu")
    assert "hidden_validation_error" in metadata


def test_hidden_inputs_are_regenerated_under_every_seed(monkeypatch):
    generated = []

    def load(path):
        inputs = [torch.rand(5)]
        generated.append(inputs[0].clone())
        return [inputs]

    monkeypatch.setattr(native, "load_hidden_inputs", load)
    metadata = {}
    assert native.run_hidden_correctness_check(nn.Identity(), nn.Identity(), "untrusted.py", metadata, device="cpu")
    assert metadata["hidden_validation_seeds"] == list(HIDDEN_SEEDS)
    assert metadata["hidden_completed_checks"] == 3
    assert not torch.equal(generated[0], generated[1])
    assert all(r["passed"] and len(r["orders"]) == 2 for r in metadata["reliability_checks"])


@pytest.mark.parametrize("name", ["zero", "scaled_32", "sparse_outliers"])
def test_each_stress_distribution_can_reject_a_nominal_only_shortcut(monkeypatch, name):
    inputs = [torch.ones(2000)]
    # These shortcuts all pass the nominal input but violate the selected case.
    def candidate(x):
        if name == "zero":
            return torch.ones_like(x) if not x.any() else x
        if name == "scaled_32":
            return x.clamp(-10, 10)
        return torch.where(x.abs() > 40, torch.zeros_like(x), x)

    monkeypatch.setattr(native, "load_hidden_inputs", lambda path: [inputs])
    metadata = {}
    assert fixed(nn.Identity(), Function(candidate), inputs)
    path = ROOT / "hidden_tests/level1/19_hidden.py"
    assert not native.run_hidden_correctness_check(nn.Identity(), Function(candidate), str(path), metadata, device="cpu")
    assert any(name in label for label in metadata["hidden_failed_configs"])
    assert metadata["hidden_completed_checks"] == 12
    assert torch.equal(inputs[0], torch.ones(2000))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_outliers_are_reproducible_and_preserve_layout_and_other_arguments(dtype):
    x = torch.ones(16, 30, dtype=dtype)[:, ::2]
    mask = torch.tensor([True, False])
    inputs = [x, mask, 2]
    cases = dict(stress_cases(inputs, 42))
    again = dict(stress_cases(inputs, 42))
    assert torch.equal(cases["sparse_outliers"][0], again["sparse_outliers"][0])
    assert (cases["sparse_outliers"][0].abs() > 1).any()
    for changed in cases.values():
        assert changed[0].shape == x.shape and changed[0].stride() == x.stride()
        assert changed[0].dtype == x.dtype and changed[1] is mask and changed[2] == 2
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize("pid", [13, 14, 15, 90, 95, 98])
def test_restricted_domains_are_not_automatically_perturbed(pid):
    assert not stress_enabled(ROOT / f"hidden_tests/level1/{pid}_hidden.py")
    assert not stress_enabled("/tmp/level1/19_hidden.py")


def test_missing_hidden_path_has_explicit_validation_error():
    with pytest.raises(RuntimeError, match="hidden suite"):
        native.run_hidden_correctness_check(nn.Identity(), nn.Identity(), None, {}, device="cpu")


@pytest.mark.parametrize("pid", sorted(STRESS_TASKS))
def test_reviewed_reference_accepts_stress_cases_at_small_shapes(pid):
    path = next((ROOT / "KernelBench/level1").glob(f"{pid}_*.py"))
    context = {}
    exec(path.read_text(), context)
    for key in ("M", "N", "K", "L", "m", "n", "k", "b", "i", "j", "l", "batch_size", "dim1", "dim2"):
        if key in context:
            context[key] = 4
    # These tasks use `dim` as a length, while reduction tasks use it as an axis.
    if 19 <= pid <= 27:
        context["dim"] = 8
    model = context["Model"](*context["get_init_inputs"]())
    inputs = context["get_inputs"]()
    for _, changed in stress_cases(inputs, 42):
        output = model(*changed)
        assert torch.isfinite(output).all()
        assert fixed(model, model, changed)


def test_zero_standard_trials_are_rejected():
    result = native.run_and_check_correctness(nn.Identity(), nn.Identity(), lambda: [torch.ones(3)], {}, 0, device="cpu")
    assert not result.correctness


def test_candidate_module_cannot_replace_reference_globals(monkeypatch):
    source = '''import torch
from torch import nn
FACTOR = 2
class Model(nn.Module):
    def forward(self, x): return x * FACTOR
def get_init_inputs(): return []
def get_inputs(): return [torch.ones(4)]
'''
    candidate = '''import torch
from torch import nn
FACTOR = 3
class ModelNew(nn.Module):
    def forward(self, x): return x * FACTOR
'''
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_device"), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.get_device_name", return_value="simulated"), \
         patch.object(native, "graceful_eval_cleanup"):
        result = native.eval_kernel_against_ref(source, candidate, device=torch.device("cpu"))
    assert not result.correctness
    assert result.metadata["reliability_protocol"] == PROTOCOL


def test_stage_summary_keeps_failures_and_missing_samples_in_denominator(tmp_path):
    config = RunConfig(run_name="stages", runs_dir=str(tmp_path), start=1, end=1, num_samples=5).validate()
    atomic_json(config.run_dir / "eval_results.json", {
        "1_0": {"compiled": False},
        "1_1": {"compiled": True, "correctness": False},
        "1_2": {"compiled": True, "correctness": True, "runtime": 2},
        "1_3": {"compiled": True, "correctness": True, "runtime": 1},
    })
    atomic_json(config.run_dir / "eval_results_hidden.json", {
        "1_2": {"compiled": True, "correctness": False},
        "1_3": {"compiled": True, "correctness": True},
    })
    engine.write_pass_at_k(config)
    report = engine.write_validation_summary(config)
    assert report["counts"] == dict(declared_samples=5, compiled_samples=3, standard_correct_samples=2,
                                    hidden_correct_samples=1, correct_samples=1, timed_correct_samples=1)
    assert len(report["samples"]) == 5
    assert not report["score_eligibility_checked"] and not report["functional_validity_certified"]


def test_stage_summary_does_not_claim_hidden_checks_when_disabled(tmp_path):
    config = RunConfig(run_name="no-hidden", runs_dir=str(tmp_path), start=1, end=1, hidden_tests=False).validate()
    report = engine.write_validation_summary(config, {"1_0": {"compiled": True, "correctness": True}})
    assert report["counts"]["correct_samples"] == 1
    assert report["counts"]["hidden_correct_samples"] == 0
    assert report["samples"][0]["hidden_correct"] is None


def test_reliability_module_participates_in_resume_digest(monkeypatch):
    from pathlib import Path

    before = evaluator_digest()
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"different protocol" if path.name == "reliability.py" else original(path))
    assert evaluator_digest() != before
