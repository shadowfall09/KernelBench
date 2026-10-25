"""Native state hooks and source-bound orchestration, CPU/simulated only."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kb_workflows.common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.engine import task_payload
from kb_workflows.evaluation import evaluate_task, state_audit, tool_checks, tool_runner
from src.reliability import check_paired_calls


def collect(reference=None, candidate=None, *, labels=("standard/0", "hidden/42/0")):
    reference = reference if reference is not None else torch.nn.Linear(3, 2)
    candidate = candidate if candidate is not None else copy.deepcopy(reference)
    metadata = {}
    with state_audit.audit() as collector:
        for index, label in enumerate(labels):
            issue = check_paired_calls(reference, candidate, [torch.randn(5, 3)], seed=42 + index,
                tolerance=1e-5, device="cpu", set_seed=torch.manual_seed, metadata=metadata, label=label)
            assert issue is None
    return collector.document(), metadata["reliability_checks"]


def test_actual_native_hook_covers_every_standard_hidden_case_and_resets_context():
    document, cases = collect()
    assert [row["label"] for row in document["cases"]] == [row["label"] for row in cases]
    assert state_audit.verdict(document, cases)["status"] == "passed"
    assert state_audit.ACTIVE.get() is None
    assert document["timing_samples_collected"] is False
    assert all(len(row["check"]["values"]["events"]) == 11 for row in document["cases"])
    json.dumps(document, allow_nan=False)


def test_actual_hook_preserves_rng_and_original_module_state():
    reference = torch.nn.BatchNorm1d(3)
    candidate = copy.deepcopy(reference)
    snapshot = copy.deepcopy(reference.state_dict())
    inputs = [torch.randn(5, 3)]
    torch.manual_seed(700)
    before = torch.get_rng_state().clone()
    with state_audit.audit() as collector:
        state_audit.observe(reference, candidate, inputs, seed=42, tolerance=1e-5, device="cpu",
                            set_seed=torch.manual_seed, label="bn")
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(value, snapshot[name]) for name, value in reference.state_dict().items())
    assert collector.document()["check"]["status"] == "passed"


@pytest.mark.parametrize("damage", ["case", "seed", "omitted_event", "parameter_count", "mutated_binding",
                                   "omitted_binding", "probe_count", "event_seed", "summary", "tolerance"])
def test_receipt_replay_rejects_missing_or_modified_raw_state_evidence(damage):
    document, cases = collect()
    row = document["cases"][0]
    values = row["check"]["values"]
    if damage == "case":
        document["cases"].pop()
    elif damage == "seed":
        row["seed"] += 1
    elif damage == "omitted_event":
        values["events"].pop()
    elif damage == "parameter_count":
        values["parameter_count"] = 0
    elif damage == "mutated_binding":
        values["bindings"][0]["states"][0]["target"] = "different.parameter"
    elif damage == "omitted_binding":
        values["bindings"].pop()
    elif damage == "probe_count":
        values["events"][-1]["parameters_updated"] = 0
    elif damage == "event_seed":
        values["events"][-1]["seed"] += 1
    elif damage == "summary":
        values["issues"].append("actual failure hidden behind a passed summary")
    else:
        document["tolerance"] = 1.0
    verdict = state_audit.verdict(document, cases, expected_tolerance=1e-5)
    assert verdict["status"] == "not_measured"


def test_state_failure_is_retained_even_when_native_outputs_pass():
    reference = torch.nn.Linear(3, 2)

    class Cached(torch.nn.Linear):
        def forward(self, x):
            if not hasattr(self, "cache"):
                self.cache = self.weight.detach().clone()
            return torch.nn.functional.linear(x, self.cache, self.bias)

    candidate = Cached(3, 2)
    candidate.load_state_dict(reference.state_dict())
    document, cases = collect(reference, candidate)
    assert all(row["passed"] for row in cases)
    assert state_audit.verdict(document, cases)["status"] == "failed"


def test_state_hook_restores_process_environment_and_marks_changes_failed():
    class Changes(torch.nn.Identity):
        def forward(self, x):
            os.environ["KB_STATE_PRIVATE"] = "do-not-record-this-value"
            return x

    document, cases = collect(torch.nn.Identity(), Changes(), labels=("one",))
    # The native call itself is outside this diagnostic's restoration scope.
    os.environ.pop("KB_STATE_PRIVATE", None)
    assert state_audit.verdict(document, cases)["status"] == "failed"
    assert "do-not-record-this-value" not in json.dumps(document)
    assert "env.KB_STATE_PRIVATE" in json.dumps(document)


def test_state_config_previews_without_loading_torch_and_propagates_payloads(tmp_path):
    config = RunConfig(generation="local", evaluation="local", hardware="RTX_A6000", state_validation=True,
                       runs_dir=str(tmp_path), run_name="state", start=2, end=2).validate()
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 2)).write_text("candidate")
    for task in ("evaluate", "paired_timing", "qualify_small", "profile"):
        assert task_payload(config, 2, task=task)["eval_config"]["state_validation"] is True
    assert tool_checks.enabled(task_payload(config, 2))
    assert not tool_checks.enabled(task_payload(config, 2, task="baseline"))
    path = tmp_path / "state.toml"
    path.write_text('[run]\nstate_validation = true\n')
    assert parse_config(["--config", str(path)]).state_validation
    assert not parse_config(["--state-validation", "--no-state-validation"]).state_validation
    with pytest.raises(ValueError, match="explicit boolean"):
        RunConfig(state_validation=1).validate()
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--state-validation","--level","3","--start","1","--end","50","--dry-run"]) == 0
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["config"]["state_validation"] is True


def test_shared_control_can_collect_state_and_exact_repeats_together(monkeypatch):
    source = "frozen test reference"
    payload = {"ref_arch_src": source, "eval_config": {"state_validation": True, "determinism_checks": True,
        "determinism_repeats": 3, "determinism_policy": {"mode": "bitwise", "atomics_allowed": False,
            "reference_sha256": hashlib.sha256(source.encode()).hexdigest()}}}

    def dispatch(request):
        metadata = {}
        reference = torch.nn.Linear(3, 2)
        assert check_paired_calls(reference, copy.deepcopy(reference), [torch.randn(5, 3)], seed=42, tolerance=1e-5,
            device="cpu", set_seed=torch.manual_seed, metadata=metadata, label="standard/0") is None
        return {"compiled": True, "correctness": True, "metadata": metadata}

    monkeypatch.setattr(evaluate_task, "dispatch", dispatch)
    result = evaluate_task.diagnostic_control(payload)
    metadata = result["metadata"]
    assert state_audit.verdict(metadata["state_audit"], metadata["reliability_checks"])["status"] == "passed"
    assert tool_checks.repeat_verdict(metadata["repeat_audit"], payload["eval_config"]["determinism_policy"],
                                      3, metadata["reliability_checks"])["status"] == "passed"


def test_cpu_simulated_supervisor_retains_original_time_and_gates_state_receipts(tmp_path, monkeypatch):
    payload = {"task": "evaluate", "ref_arch_src": "reference", "kernel_src": "candidate", "eval_config": {
        "precision": "fp32", "fp32_tolerance": 1e-5, "state_validation": True, "sanitizer_tools": "none",
        "sanitizer_timeout": 30, "timeout": 60, "measure_performance": True}}
    device = {"available": True, "uuid": "simulated", "driver_version": "fixture"}
    calls = []

    def dispatch(request):
        metadata = {"reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": request["kernel_src"]}), "device_fingerprint": device}
        ref = torch.nn.Linear(3, 2)
        assert check_paired_calls(ref, copy.deepcopy(ref), [torch.randn(5, 3)], seed=42, tolerance=1e-5,
            device="cpu", set_seed=torch.manual_seed, metadata=metadata, label="standard/0") is None
        return {"compiled": True, "correctness": True, "runtime": 999 if "diagnostic_stage" in request["eval_config"] else 1,
                "runtime_stats": {"samples_ms": [1]}, "metadata": metadata}

    def run(command, timeout, log, **kwargs):
        index = command.index("-m")
        request = read_json(Path(command[index + 2]))
        calls.append(request["eval_config"].get("diagnostic_stage", "native"))
        result = evaluate_task.diagnostic_control(request) if calls[-1] == "control" else dispatch(request)
        atomic_json(Path(command[index + 3]), result)
        return {"exit_code": 0, "timed_out": False}

    monkeypatch.setattr(evaluate_task, "dispatch", dispatch)
    result = tool_runner.evaluate(payload, run=run)
    assert calls == ["native", "control"] and result["correctness"] is True and result["runtime"] == 1
    assert tool_checks.validate(result, payload, allow_simulated=True)["mode_state"]["status"] == "passed"
    with pytest.raises(ValueError, match="Simulated"):
        tool_checks.validate(result, payload)
    document = copy.deepcopy(result["metadata"]["tool_checks"])
    document["control"]["result"]["metadata"]["state_audit"]["cases"] = []
    changed = tool_checks.finish(document, payload)
    assert changed["correctness"] is False
    assert tool_checks.validate(changed, payload, allow_simulated=True)["mode_state"]["status"] == "not_measured"


def test_harbor_export_keeps_state_policy_in_native_verifier_config(tmp_path):
    from kb_workflows.runtime import harbor

    config = RunConfig(generation="harbor", hardware="RTX_A6000", state_validation=True, start=2, end=2).validate()
    task = harbor.write_task(tmp_path / "task", config, 2)
    assert read_json(task / "tests/config.json")["config"]["state_validation"] is True
    assert (task / "tests/repo/kb_workflows/evaluation/state_audit.py").is_file()
    assert (task / "tests/repo/kb_workflows/evaluation/state_contract.py").is_file()


def test_state_implementation_is_bound_into_evaluator_digest(monkeypatch):
    original = Path.read_bytes
    before = evaluator_digest()

    def changed(path):
        content = original(path)
        return content + b"\n# state implementation drift" if path.name == "state_contract.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert evaluator_digest() != before
