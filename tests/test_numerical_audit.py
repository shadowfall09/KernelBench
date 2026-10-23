"""Full-input oracle hooks on small CPU inputs; transport tests are simulated."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kb_workflows import evaluate_task, numerical_audit, operator_oracles, tool_checks, tool_runner
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, problem_paths, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.engine import task_payload
from kb_workflows.operator_specs import spec_for
from src.eval import load_original_model_and_inputs
from src.reliability import check_paired_calls


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def payload(pid=2):
    source = problem_paths(1)[pid].read_text()
    return {"task": "evaluate", "level": 1, "problem_id": pid, "ref_arch_src": source,
        "kernel_src": "frozen simulated candidate", "eval_config": {"precision": "fp32", "fp32_tolerance": 1e-3,
        "numerical_validation": True, "oracle_memory_mb": 16, "sanitizer_tools": "none", "sanitizer_timeout": 30,
        "timeout": 60, "oracle_spec_sha256": fingerprint(spec_for(1, pid, source))}}


def collect(request=None, candidate=None):
    request = request or payload()
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model()
    candidate = candidate if candidate is not None else copy.deepcopy(model)
    args = [torch.randn(3, 7), torch.randn(7, 5)]
    cases = []
    with numerical_audit.audit(request) as collector:
        for index, label in enumerate(("standard/0", "hidden/43/0")):
            collector.observe(model, candidate, args, seed=42 + index, device="cpu", tolerance=1e-3,
                              set_seed=torch.manual_seed, label=label)
            cases.append({"label": label, "seed": 42 + index})
    return collector.document(), cases


def test_actual_cpu_oracle_hook_preserves_inputs_rng_and_original_models():
    request = payload()
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model()
    args = [torch.randn(3, 7), torch.randn(7, 5)]
    saved = copy.deepcopy(args)
    metadata = {}
    with numerical_audit.audit(request) as collector:
        assert check_paired_calls(model, copy.deepcopy(model), args, seed=42, device="cpu", tolerance=1e-3,
            set_seed=torch.manual_seed, metadata=metadata, label="native/0") is None
    document = collector.document()
    assert all(torch.equal(a, b) for a, b in zip(saved, args))
    assert numerical_audit.ACTIVE.get() is None
    assert all(row["status"] == "passed" for row in numerical_audit.verdicts(document, metadata["reliability_checks"], request).values())
    assert document["dimensions_changed"] is False and document["timing_samples_collected"] is False
    json.dumps(document, allow_nan=False)
    before = torch.get_rng_state().clone()
    with numerical_audit.audit(request) as collector:
        collector.observe(model, copy.deepcopy(model), args, seed=43, device="cpu", tolerance=1e-3,
                          set_seed=torch.manual_seed, label="rng")
    assert torch.equal(before, torch.get_rng_state())


def test_wrong_arithmetic_and_bfloat16_compute_fail_full_input_fp32_contract():
    class Reduced(torch.nn.Module):
        def forward(self, a, b):
            return a.bfloat16().float() @ b.bfloat16().float()

    document, cases = collect(candidate=Reduced())
    checks = numerical_audit.verdicts(document, cases, payload())
    assert checks["precision_contract"]["status"] == "failed"
    assert document["cases"][0]["precision_gate"]["reference_qualified"]


def test_repaired_exclusive_scan_passes_reference_qualification():
    request = payload(92)
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model(1)
    with numerical_audit.audit(request) as collector:
        collector.observe(model, copy.deepcopy(model), [torch.randn(3, 7)], seed=42, device="cpu", tolerance=1e-3,
                          set_seed=torch.manual_seed, label="exclusive")
    document = collector.document()
    assert document["known_reference_issue"] is None
    assert document["cases"][0]["precision_gate"]["reference_qualified"] is True
    assert numerical_audit.verdicts(document, [{"label": "exclusive", "seed": 42}], request)["precision_contract"]["status"] == "passed"


def test_full_size_memory_admission_never_executes_or_silently_resizes(monkeypatch):
    request = payload()
    request["eval_config"]["oracle_memory_mb"] = 1
    seen = []
    original = operator_oracles.estimate_bytes

    def estimate(pid, model, inputs):
        seen.append([list(value.shape) for value in inputs])
        return original(pid, model, inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Over-budget oracle must not allocate or execute")

    monkeypatch.setattr(operator_oracles, "estimate_bytes", estimate)
    monkeypatch.setattr(operator_oracles, "execute", forbidden)
    model = torch.nn.Identity()
    inputs = [torch.ones(100, 100), torch.ones(100, 100)]
    with numerical_audit.audit(request) as collector:
        collector.observe(model, model, inputs, seed=42, device="cpu", tolerance=1e-3,
                          set_seed=torch.manual_seed, label="full")
    document = collector.document()
    assert seen == [[[100, 100], [100, 100]]]
    assert "no resizing" in document["cases"][0]["errors"][0]
    assert document["cases"][0]["status"] == "not_measured"
    checks = numerical_audit.verdicts(document, [{"label": "full", "seed": 42}], request)
    assert checks["precision_contract"]["status"] == "not_measured"


@pytest.mark.parametrize("damage", ["case", "seed", "policy", "budget", "source", "counter", "limit", "rms",
                                   "reference", "issues", "summary", "memory", "steps", "signal", "blank_leaves"])
def test_replaying_raw_numerical_evidence_rejects_missing_or_mutated_checks(damage):
    request = payload()
    document, cases = collect(request)
    row = document["cases"][0]
    gate = row["precision_gate"]
    if damage == "case":
        document["cases"].pop()
    elif damage == "seed":
        row["seed"] += 1
    elif damage == "policy":
        document["policy"] = {**document["policy"], "norm_rtol": 1}
    elif damage == "budget":
        document["oracle_memory_mb"] += 1
    elif damage == "source":
        document["oracle_spec_sha256"] = "0" * 64
    elif damage == "counter":
        gate["leaves"][0]["candidate"]["elements_failed"] = 1
    elif damage == "limit":
        gate["leaves"][0]["norm_limit"] *= 10
    elif damage == "rms":
        gate["leaves"][0]["candidate"]["rms_error"] = 1e5
    elif damage == "reference":
        gate["reference_qualified"] = False
    elif damage == "issues":
        gate["issues"].append("real failure")
    elif damage == "summary":
        row["status"] = "incomplete"
    elif damage == "memory":
        row["estimated_host_bytes"] = 10**12
    elif damage == "signal":
        row["reference_signal"]["informative"] = False
    elif damage == "blank_leaves":
        gate["leaves"] = []
    else:
        gate.pop("all_elements_required")
    checks = numerical_audit.verdicts(document, cases, request)
    assert checks["precision_contract"]["status"] == "not_measured"


def test_weak_all_zero_reference_signal_cannot_qualify_as_an_informative_test():
    request = payload()
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model()
    with numerical_audit.audit(request) as collector:
        collector.observe(model, copy.deepcopy(model), [torch.zeros(3, 7), torch.zeros(7, 5)], seed=42, device="cpu",
                          tolerance=1e-3, set_seed=torch.manual_seed, label="zero")
    checks = numerical_audit.verdicts(collector.document(), [{"label": "zero", "seed": 42}], request)
    assert checks["precision_contract"]["status"] == "passed"
    assert checks["reference_signal"]["status"] == "failed"


def test_configuration_payload_and_sdk_light_previews(tmp_path):
    cfg = parse_config(["--numerical-validation", "--oracle-memory-mb", "32", "--generation", "local"])
    assert cfg.numerical_validation and cfg.oracle_memory_mb == 32
    cfg.runs_dir, cfg.run_name = str(tmp_path), "oracle"
    cfg.run_dir.mkdir()
    (cfg.run_dir / kernel_name(1, 2)).write_text("candidate")
    request = task_payload(cfg, 2)
    assert request["eval_config"]["oracle_spec_sha256"] == fingerprint(spec_for(1, 2, request["ref_arch_src"]))
    assert request["eval_config"]["oracle_memory_mb"] == 32
    assert tool_checks.enabled(request)
    assert not parse_config(["--numerical-validation", "--no-numerical-validation"]).numerical_validation
    for invalid in (0, True, 1048577):
        with pytest.raises(ValueError, match="memory envelope"):
            RunConfig(oracle_memory_mb=invalid).validate()
    with pytest.raises(ValueError, match="level 1"):
        RunConfig(level=2, numerical_validation=True).validate()
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--numerical-validation","--level","1","--start","1","--end","100","--dry-run"]) == 0
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["config"]["numerical_validation"] is True


def test_cpu_supervisor_preserves_native_times_and_rechecks_numerical_receipts(monkeypatch):
    request = payload()
    request["eval_config"]["state_validation"] = True
    request["eval_config"]["determinism_checks"] = True
    request["eval_config"]["determinism_repeats"] = 3
    import hashlib
    request["eval_config"]["determinism_policy"] = {"mode": "bitwise", "atomics_allowed": False,
        "reference_sha256": hashlib.sha256(request["ref_arch_src"].encode()).hexdigest()}
    device = {"available": True, "uuid": "simulated", "driver_version": "fixture"}
    calls = []

    def dispatch(value):
        Model, _, _ = load_original_model_and_inputs(value["ref_arch_src"], {})
        model = Model()
        metadata = {"device_fingerprint": device, "reference_sha256": fingerprint({"source": value["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": value["kernel_src"]})}
        assert check_paired_calls(model, copy.deepcopy(model), [torch.randn(3, 7), torch.randn(7, 5)], seed=42,
            device="cpu", tolerance=1e-3, set_seed=torch.manual_seed, metadata=metadata, label="native/0") is None
        return {"compiled": True, "correctness": True, "runtime": 999 if "diagnostic_stage" in value["eval_config"] else 1,
                "runtime_stats": {"samples_ms": [1]}, "metadata": metadata}

    def run(command, timeout, log, **kwargs):
        index = command.index("-m")
        value = read_json(Path(command[index + 2]))
        calls.append(value["eval_config"].get("diagnostic_stage", "native"))
        result = evaluate_task.diagnostic_control(value) if calls[-1] == "control" else dispatch(value)
        atomic_json(Path(command[index + 3]), result)
        return {"exit_code": 0, "timed_out": False}

    monkeypatch.setattr(evaluate_task, "dispatch", dispatch)
    result = tool_runner.evaluate(request, run=run)
    assert calls == ["native", "control"]
    assert result["runtime"] == 1 and result["correctness"] is True
    checks = tool_checks.validate(result, request, allow_simulated=True)
    assert all(row["status"] == "passed" for row in checks.values())
    with pytest.raises(ValueError, match="Simulated"):
        tool_checks.validate(result, request)
    document = copy.deepcopy(result["metadata"]["tool_checks"])
    document["control"]["result"]["metadata"]["numerical_audit"]["cases"].pop()
    changed = tool_checks.finish(document, request)
    assert changed["correctness"] is False and changed["runtime"] == 1


def test_harbor_exports_the_oracle_manifest_and_current_policy(tmp_path):
    from kb_workflows import harbor

    cfg = RunConfig(generation="harbor", hardware="RTX_A6000", start=2, end=2,
                    numerical_validation=True, oracle_memory_mb=32).validate()
    directory = harbor.write_task(tmp_path / "harbor", cfg, 2)
    exported = read_json(directory / "tests/config.json")["config"]
    assert exported["numerical_validation"] is True and exported["oracle_memory_mb"] == 32
    assert (directory / "tests/repo/kb_workflows/operator_oracles.json").is_file()


def test_empty_tensor_errors_remain_null_and_do_not_replace_reference_signal():
    request = payload()
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model()
    with numerical_audit.audit(request) as collector:
        for label, args in (("empty", [torch.empty(3, 0), torch.empty(0, 5)]),
                            ("informative", [torch.ones(3, 7), torch.ones(7, 5)])):
            collector.observe(model, copy.deepcopy(model), args, seed=42, device="cpu", tolerance=1e-3,
                              set_seed=torch.manual_seed, label=label)
    document = collector.document()
    leaf = document["cases"][0]["precision_gate"]["leaves"][0]
    # This matmul has a nonempty all-zero result, unlike a zero output dimension.
    assert leaf["elements"] == 15
    request = payload(92)
    Model, _, _ = load_original_model_and_inputs(request["ref_arch_src"], {})
    model = Model(1)
    with numerical_audit.audit(request) as collector:
        for label, args in (("empty", [torch.empty(3, 0)]), ("informative", [torch.ones(3, 7)])):
            collector.observe(model, copy.deepcopy(model), args, seed=42, device="cpu", tolerance=1e-3,
                              set_seed=torch.manual_seed, label=label)
    document = collector.document()
    cases = [{"label": "empty", "seed": 42}, {"label": "informative", "seed": 42}]
    leaf = document["cases"][0]["precision_gate"]["leaves"][0]
    assert leaf["kind"] == "empty_tensor" and leaf["elements"] == 0
    assert leaf["oracle_rms"] is None and leaf["reference"]["rms_error"] is None
    assert all(row["status"] == "passed" for row in numerical_audit.verdicts(document, cases, request).values())
    assert document["cases"][0]["reference_signal"]["informative"] is False
    changed = copy.deepcopy(document)
    changed["cases"][0]["precision_gate"]["leaves"][0]["reference"]["rms_error"] = 0
    assert numerical_audit.verdicts(changed, cases, request)["precision_contract"]["status"] == "not_measured"
