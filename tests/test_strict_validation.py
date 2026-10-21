"""Known-good references and counterexamples for optional strict evidence gates."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from kb_workflows import evaluate_task, evidence, measurement, strict_validation
from kb_workflows.aggregation import sensitivity
from kb_workflows.common import ROOT, problem_paths
from kb_workflows.config import parse_config
from kb_workflows.numerical import PrecisionContract, precision_gate, signal_audit, to_oracle
from kb_workflows.strict_validation import contract_for, environment_guard, run_checks, state_checks
from src.eval import load_original_model_and_inputs, set_seed
from src.validation import compare_outputs

STRICT = PrecisionContract(norm_rtol=5e-6, norm_atol=1e-7, element_rtol=5e-5, element_atol=5e-7,
                           element_scale_rtol=1e-6)
REVIEWED = (1, 2, 3, *range(19, 28), 47, 48, 49, 51, 52)


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference(pid):
    source = problem_paths(1)[pid].read_text()
    Model, init, _ = load_original_model_and_inputs(source, {})
    return Model(*init()), contract_for(1, pid, source)


@pytest.mark.parametrize("pid", REVIEWED)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_reviewed_references_pass_actual_cpu_fp64_and_hidden_domain_checks(pid, dtype):
    model, contract = reference(pid)
    model = model.to(dtype=dtype)
    result = run_checks(model, copy.deepcopy(model), contract, hidden=True, seed=42,
                        device=torch.device("cpu"), dtype=dtype, set_seed=set_seed)
    assert all(item["status"] == "passed" for item in result.values()), result
    assert any("noncontiguous" in label for label in result["input_families"]["values"]["completed"])


def test_bfloat16_compute_cannot_claim_fp32_precision():
    ref, contract = reference(2)

    class Reduced(torch.nn.Module):
        def forward(self, a, b):
            return a.bfloat16().float() @ b.bfloat16().float()

    result = run_checks(ref, Reduced(), contract, hidden=False, seed=42, device=torch.device("cpu"),
                        dtype=torch.float32, set_seed=set_seed)
    assert result["precision_contract"]["status"] == "failed"


def test_gelu_approximation_passes_legacy_but_fails_explicit_oracle_budget():
    x = torch.tensor([-2.699, 2.699])
    exact = torch.nn.functional.gelu(x)
    approx = torch.nn.functional.gelu(x, approximate="tanh")
    assert compare_outputs(exact, approx, 1e-3) is None
    report = precision_gate(exact, approx, torch.nn.functional.gelu(x.double()), STRICT)
    assert report["passed"] is False
    assert report["leaves"][0]["reference"]["passed"]


def test_every_element_matters_and_failed_reference_is_not_a_qualified_oracle():
    expected = torch.ones(1000)
    wrong = expected.clone()
    wrong[-1] += 0.1
    report = precision_gate(expected, wrong, expected.double(), STRICT)
    assert not report["passed"] and report["leaves"][0]["candidate"]["elements_failed"] == 1
    bad_reference = precision_gate(wrong, expected, expected.double(), STRICT)
    assert not bad_reference["reference_qualified"]


def test_extreme_finite_values_cannot_overflow_rms_into_an_infinite_error_budget():
    truth = torch.full((5,), 1e308, dtype=torch.float64)
    correct = precision_gate(truth, truth.clone(), truth, STRICT)
    assert correct["passed"]
    assert correct["leaves"][0]["oracle_rms"] == pytest.approx(1e308)
    assert not precision_gate(truth, torch.zeros_like(truth), truth, STRICT)["passed"]


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "inf_sign", "integer", "structure", "oracle_dtype"])
def test_numerical_contract_preserves_structural_and_nonfinite_requirements(kind):
    ref = torch.tensor([1.0, 2.0])
    cand, truth = ref.clone(), ref.double()
    if kind == "shape":
        cand = cand.reshape(1, 2)
    elif kind == "dtype":
        cand = cand.double()
    elif kind == "nan":
        cand[0] = float("nan")
    elif kind == "inf_sign":
        ref[0], cand[0], truth[0] = float("inf"), float("-inf"), float("inf")
    elif kind == "integer":
        ref, cand, truth = ref.long(), torch.tensor([1, 3]), ref.long()
    elif kind == "structure":
        ref, cand, truth = [ref], (cand,), [truth]
    else:
        truth = truth.float()
    assert not precision_gate(ref, cand, truth, STRICT)["passed"]


def test_oracle_conversion_keeps_integer_bool_roles_and_never_mutates_input():
    data = {"x": torch.tensor([1.2]), "labels": torch.tensor([3]), "mask": torch.tensor([True])}
    converted = to_oracle(data)
    assert converted["x"].dtype == torch.float64
    assert converted["labels"].dtype == torch.int64 and converted["mask"].dtype == torch.bool
    converted["x"].zero_()
    assert data["x"].item() != 0


def test_signal_audit_distinguishes_weak_absolute_check_from_constant_semantics():
    weak = signal_audit(torch.full((10,), 1e-8), absolute_tolerance=1e-3)
    constant = signal_audit(torch.ones(10), absolute_tolerance=1e-3)
    assert not weak["informative"]
    assert constant["informative"] and constant["leaves"][0]["constant"]
    assert signal_audit(torch.tensor([2]), absolute_tolerance=1e-3)["informative"]


def test_identity_cache_detected_after_mutating_same_input_objects():
    ref, contract = reference(19)

    class Cached(torch.nn.Module):
        def forward(self, x):
            if not hasattr(self, "answer"):
                self.answer = torch.relu(x)
            return self.answer

    result = run_checks(ref, Cached(), contract, hidden=False, seed=42, device=torch.device("cpu"),
                        dtype=torch.float32, set_seed=set_seed)
    assert result["input_sensitivity"]["status"] == "failed"


def test_parameter_caches_and_missing_training_state_fail_the_declared_contract():
    class CachedLinear(torch.nn.Linear):
        def forward(self, x):
            if not hasattr(self, "saved"):
                self.saved = self.weight.detach().clone()
            return torch.nn.functional.linear(x, self.saved, self.bias)

    ref, cand = torch.nn.Linear(3, 2), CachedLinear(3, 2)
    cand.load_state_dict(ref.state_dict())
    result = state_checks(ref, cand, [torch.randn(5, 3)], seed=42, set_seed=set_seed,
                          tolerance=1e-5, device=torch.device("cpu"))
    assert result["status"] == "failed"
    assert any("parameter update" in issue for issue in result["values"]["issues"])

    class WrongState(torch.nn.BatchNorm1d):
        def forward(self, x):
            output = super().forward(x)
            if self.training:
                self.num_batches_tracked.zero_()
            return output

    ref, cand = torch.nn.BatchNorm1d(3), WrongState(3)
    cand.load_state_dict(ref.state_dict())
    result = state_checks(ref, cand, [torch.randn(5, 3)], seed=42, set_seed=set_seed,
                          tolerance=1e-5, device=torch.device("cpu"))
    assert result["status"] == "failed"
    assert any("num_batches_tracked" in issue for issue in result["values"]["issues"])


def test_environment_audit_restores_flags_dtype_and_env_without_serializing_values(monkeypatch):
    monkeypatch.setenv("KB_AUDIT_PRIVATE", "value-must-never-be-serialized")
    before = torch.get_default_dtype()
    benchmark = torch.backends.cudnn.benchmark
    changes = []
    with environment_guard(changes):
        os.environ["KB_AUDIT_PRIVATE"] = "changed-secret"
        os.environ["KB_AUDIT_ADDED"] = "another-secret"
        torch.set_default_dtype(torch.float64)
        torch.backends.cudnn.benchmark = not benchmark
    assert torch.get_default_dtype() == before and torch.backends.cudnn.benchmark == benchmark
    assert os.environ["KB_AUDIT_PRIVATE"] == "value-must-never-be-serialized"
    assert "KB_AUDIT_ADDED" not in os.environ
    assert "changed-secret" not in json.dumps(changes)
    assert {"default_dtype", "env.KB_AUDIT_PRIVATE", "env.KB_AUDIT_ADDED", "backend.benchmark"}.issubset(changes)


def test_missing_contract_and_reference_drift_are_explicit():
    assert contract_for(2, 99, "source") is None
    with pytest.raises(ValueError, match="snapshot"):
        contract_for(1, 2, "modified reference")


def test_evidence_levels_never_turn_missing_optional_tools_into_success():
    passed = evidence.check("passed", reason="fixture provides actual check outcome")
    checks = {name: copy.deepcopy(passed) for name in evidence.LEVELS["current"] + evidence.LEVELS["L0"]}
    receipt = evidence.receipt(checks, requested="L1")
    assert receipt["achieved_level"] == "L0" and not receipt["requested_level_met"]
    assert receipt["checks"]["kernel_origin"]["status"] == "not_measured"
    assert receipt["checks"]["kernel_origin"]["values"] is None
    settings = {"evidence_level": "L1", "strict_validation": True}
    result = {"metadata": {"evidence_receipt": receipt}}
    assert evidence.score_issue(result, settings, "standard")
    receipt["requested_level_met"] = True  # A forged summary cannot override its missing checks.
    assert evidence.score_issue(result, settings, "standard")


def test_evidence_scope_mismatch_and_strict_missing_checks_are_rejected():
    checks = {"correctness": evidence.check("passed", reason="Native correctness")}
    result = {"metadata": {"evidence_receipt": evidence.receipt(checks)}}
    assert evidence.score_issue(result, {"strict_validation": True}, "standard")
    assert evidence.score_issue(result, {"evidence_level": "L0"}, "hidden")
    assert evidence.score_issue(result, {}, "hidden") is None  # Existing policy is preserved.


@pytest.mark.parametrize("native_tampering", [False, True])
def test_actual_evaluate_entrypoint_attaches_strict_receipts_and_catches_first_import_tampering(tmp_path, monkeypatch, native_tampering):
    import src.eval as native

    before = torch.get_default_dtype()

    def native_run(**kwargs):
        if native_tampering:
            torch.set_default_dtype(torch.float64)
            os.environ["KB_FIRST_IMPORT_CHANGED"] = "secret-not-for-receipt"
        return SimpleNamespace(compiled=True, correctness=True, runtime=1.0, runtime_stats={},
                               peak_memory=0, memory_stats={}, metadata={})

    monkeypatch.setattr(native, "eval_kernel_against_ref", native_run)
    monkeypatch.setattr(evaluate_task, "validate_sources", lambda payload: "trusted reference")
    monkeypatch.setattr(measurement, "device_fingerprint", lambda device: {"available": False})
    monkeypatch.setattr(strict_validation, "evaluate_contract", lambda *args: {
        name: evidence.check("passed", reason="Stub at GPU contract boundary")
        for name in evidence.LEVELS["L0"] if name != "correctness"})
    payload = {"level": 1, "problem_id": 2, "kernel_src": "candidate", "build_root": str(tmp_path),
               "eval_config": {"precision": "fp32", "fp32_tolerance": 1e-3, "num_correct_trials": 1,
                               "num_perf_trials": 2, "strict_validation": True, "evidence_level": "L0"}}
    result = evaluate_task.evaluate(payload)
    assert result["correctness"] is (not native_tampering)
    assert result["metadata"]["evidence_receipt"]["requested_level_met"] is (not native_tampering)
    assert torch.get_default_dtype() == before and "KB_FIRST_IMPORT_CHANGED" not in os.environ
    assert "secret-not-for-receipt" not in json.dumps(result)


def test_unreviewed_task_cannot_get_a_strict_receipt_by_falling_back(monkeypatch):
    checks = strict_validation.evaluate_contract({"level": 3, "problem_id": 1}, "reference", "/unused")
    assert all(item["status"] == "not_measured" for item in checks.values())
    record = evidence.from_result({"correctness": True}, {"evidence_level": "L0"}, additional=checks)
    assert record["achieved_level"] == "current" and not record["requested_level_met"]


def test_cli_and_toml_require_explicit_strictness_and_keep_lighter_default(tmp_path):
    cfg = parse_config([])
    assert not cfg.strict_validation and cfg.evidence_level == "current"
    path = tmp_path / "strict.toml"
    path.write_text('[run]\nstrict_validation = true\nevidence_level = "L0"\n')
    cfg = parse_config(["--config", str(path)])
    assert cfg.strict_validation and cfg.evidence_level == "L0"
    assert parse_config(["--evidence-level", "L2"]).strict_validation


def test_contract_preflight_is_lightweight_and_rejects_unsupported_before_creating_run(tmp_path):
    code = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--evidence-level","L0","--start","4","--end","4","--runs-dir",{str(tmp_path)!r}]) == 2
assert "torch" not in sys.modules
assert main(["--evidence-level","L0","--start","1","--end","3","--dry-run"]) == 0
assert "torch" not in sys.modules
'''
    process = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert "Strict contracts unavailable" in process.stderr
    assert not list(tmp_path.iterdir())
    assert json.loads(process.stdout)["strict_coverage"]["supported"] == [1, 2, 3]


def test_aggregation_ablation_uses_identical_declared_slots_and_keeps_failure_costs():
    report = {"problem_ids": [1, 2], "declared_samples": 3,
              "generation_budget": {"totals": {"estimated_cost_usd": None}},
              "details": [{"problem_id": 1, "sample_id": 0, "score": 1, "status": "verified", "layer": "I"},
                          {"problem_id": 2, "sample_id": 0, "score": 1, "status": "failed", "layer": "I"}]}
    result = sensitivity(report)
    assert result["declared_slots"] == 6 and result["missing_slots"] == 4
    assert result["policies"]["mean"]["macro"] == pytest.approx(1 / 6)
    assert result["policies"]["median"]["macro"] == 0
    assert result["policies"]["best_of_n"]["macro"] == 0.5
    assert result["generation_budget"] == report["generation_budget"]
    assert result["requires_additional_model_calls"] is False


def test_aggregation_refuses_duplicate_slots_and_out_of_range_scores():
    row = {"problem_id": 1, "sample_id": 0, "score": 0.5, "status": "verified"}
    report = {"problem_ids": [1], "declared_samples": 1, "details": [row, row]}
    with pytest.raises(ValueError, match="Duplicate"):
        sensitivity(report)
    report["details"] = [{**row, "score": float("nan")}]
    with pytest.raises(ValueError, match="finite"):
        sensitivity(report)
