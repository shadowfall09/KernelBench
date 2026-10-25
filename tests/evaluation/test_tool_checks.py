"""CPU consistency and simulated sanitizer workflows; no GPU/tool experiments."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from kb_workflows import engine
from kb_workflows.common import atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation import determinism, tool_checks, tool_runner
from kb_workflows.evaluation.evidence import accepted, from_result


def test_tool_selection_is_explicit_and_ordered():
    assert tool_checks.select_tools("all") == tool_checks.TOOLS
    assert tool_checks.select_tools("initcheck,memcheck") == ("memcheck", "initcheck")
    assert tool_checks.select_tools("none") == ()
    for value in ("", "none,memcheck", "all,racecheck", "memcheck,memcheck", "shell-command", None):
        with pytest.raises(ValueError):
            tool_checks.select_tools(value)


@pytest.mark.parametrize("tool", tool_checks.TOOLS)
def test_vendor_report_parser_distinguishes_completed_clean_errors_and_missing_data(tool):
    if tool == "racecheck":
        clean = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        bad = "========= RACECHECK SUMMARY: 8 hazards displayed (0 errors, 1 warning)"
    else:
        clean, bad = "========= ERROR SUMMARY: 0 errors", "========= ERROR SUMMARY: 2 errors"
    assert tool_checks.parse_log(tool, "========= COMPUTE-SANITIZER\n" + clean)["status"] == "passed"
    assert tool_checks.parse_log(tool, "========= COMPUTE-SANITIZER\n" + bad)["status"] == "failed"
    assert tool_checks.parse_log(tool, clean)["status"] == "not_measured"
    assert tool_checks.parse_log(tool, "COMPUTE-SANITIZER\ninterrupted without a summary")["status"] == "not_measured"


def test_reviewed_atomic_policy_is_source_bound_and_has_no_implicit_tolerance_exemption(tmp_path):
    source = "reference source"
    row = {"mode": "bounded_atomics", "atomics_allowed": True, "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "atol": 1e-5, "rtol": 1e-5, "review": {"reviewer": "regression fixture", "rationale": "Test a declared reduction variation"}}
    path = tmp_path / "policies.json"
    document = {"protocol": determinism.POLICY_PROTOCOL, "tasks": {"1/2": row}}
    atomic_json(path, document)
    assert determinism.policy_for(path, 1, 2, source) == row
    assert determinism.policy_for(path, 1, 3, source)["mode"] == "bitwise"
    with pytest.raises(ValueError, match="reference changed"):
        determinism.policy_for(path, 1, 2, source + "modified")
    for field, bad in (("atomics_allowed", False), ("atol", float("nan")), ("review", {}), ("mode", "accept_all")):
        changed = copy.deepcopy(document)
        changed["tasks"]["1/2"][field] = bad
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            determinism.read_policies(path)


def test_exact_repeats_include_signed_zero_discrete_bits_dtype_structure_and_nonfinite_masks():
    policy = {"mode": "bitwise"}
    first = {"x": [torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0])], "i": torch.tensor([1, 2])}
    assert determinism.repeat_issue(first, copy.deepcopy(first), policy) is None
    changed = copy.deepcopy(first)
    changed["x"][0][-1] = -0.0
    assert "bits differ" in determinism.repeat_issue(first, changed, policy)
    changed = copy.deepcopy(first)
    changed["i"][0] = 2
    assert determinism.repeat_issue(first, changed, {"mode": "bounded_atomics", "atol": 100, "rtol": 100})
    assert determinism.repeat_issue(torch.ones(2), torch.ones(2, dtype=torch.float64), policy)
    assert determinism.repeat_issue([torch.ones(2)], (torch.ones(2),), policy)
    assert determinism.signature(torch.ones(2, dtype=torch.bfloat16))["bytes_sha256"]


def test_atomic_repeat_bounds_cannot_change_nonfinite_signs_or_hide_outliers():
    policy = {"mode": "bounded_atomics", "atol": 1e-5, "rtol": 1e-5}
    first = torch.tensor([0.2, 10, float("inf")])
    assert determinism.repeat_issue(first, first + torch.tensor([1e-6, 1e-6, 0]), policy) is None
    changed = first.clone()
    changed[1] += 1
    assert "variation bound exceeded" in determinism.repeat_issue(first, changed, policy)
    changed = first.clone()
    changed[-1] = -float("inf")
    assert "non-finite" in determinism.repeat_issue(first, changed, policy)


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return x + self.calls


def test_repeat_auditor_snapshots_state_and_restores_rng_without_advancing_original_models():
    from src.reliability import check_paired_calls

    reference, candidate = Counter(), Counter()
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    policy = {"mode": "bitwise"}
    with determinism.audit(policy, 5) as collector:
        issue = check_paired_calls(reference, candidate, [torch.arange(4.0)], device="cpu", seed=7,
            tolerance=1e-5, set_seed=torch.manual_seed, metadata={}, label="stateful")
    assert issue is None and reference.calls == candidate.calls == 2
    row = collector.document()["cases"][0]
    assert row["candidate_calls"] == row["reference_calls"] == 5
    assert accepted(collector.document()["check"]) and len(set(row["output_sha256"])) == 1
    assert torch.equal(before, torch.get_rng_state())


def test_repeat_audit_rejects_noise_and_environment_changes_even_when_native_tolerance_is_loose():
    changes = [0]
    class Reference(torch.nn.Module):
        def forward(self, x):
            return x + 1
    class Noisy(torch.nn.Module):
        def forward(self, x):
            changes[0] += 1
            return x + 1 + changes[0] * 1e-5
    with determinism.audit({"mode": "bitwise"}, 5) as collector:
        collector.observe(Reference(), Noisy(), [torch.ones(2)], device="cpu", seed=9, set_seed=torch.manual_seed,
                          tolerance=1e-3, label="varied")
    assert collector.document()["check"]["status"] == "failed"
    class Env(Reference):
        def forward(self, x):
            os.environ["KB_REPEAT_TEST"] = "mutated"
            return super().forward(x)
    with determinism.audit({"mode": "bitwise"}, 3) as collector:
        collector.observe(Reference(), Env(), [torch.ones(2)], device="cpu", seed=9, set_seed=torch.manual_seed,
                          tolerance=1e-3, label="environment")
    assert collector.document()["check"]["status"] == "failed" and "KB_REPEAT_TEST" not in os.environ


@pytest.fixture
def payload(tmp_path):
    config = RunConfig(hardware="RTX_A6000", generation="local", run_name="tool-fixture", runs_dir=str(tmp_path),
                       start=2, end=2, sanitizer_tools="all", determinism_checks=True, determinism_repeats=3).validate()
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 2)).write_text("import torch\nclass ModelNew(torch.nn.Module):\n def forward(self,a,b):return a@b\n")
    return engine.task_payload(config, 2)


DEVICE = {"available": True, "name": "SIMULATED", "uuid": "simulated-gpu", "driver_version": "fixture",
          "capability": [9, 0], "total_memory": 10**10, "multiprocessors": 1, "l2_bytes": 1000,
          "torch_version": "fixture", "cuda_version": "fixture", "hip_version": None}


def native_result(payload):
    metadata = {"reference_sha256": fingerprint({"source": payload["ref_arch_src"]}), "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                "device_fingerprint": DEVICE, "reliability_checks": [{"label": "standard/0", "passed": True,
                "seed": 42, "orders": ["candidate_first", "reference_first"]}], "timing_method": "synchronized_wall"}
    result = {"compiled": True, "correctness": True, "runtime": 1, "runtime_stats": {"samples_ms": [1], "num_trials": 1}, "metadata": metadata}
    metadata["evidence_receipt"] = from_result(result, payload["eval_config"])
    if payload["task"] == "paired_timing":
        metadata.pop("reliability_checks")
        metadata["paired_timing"] = {"protocol": "balanced_paired_wall_v1", "num_pairs": 10,
            "pairs": [{"seed": index, "order": "RCB", "reference_ms": 2, "candidate_ms": 1, "control_ms": 2} for index in range(10)],
            "all_outputs_verified": True, "varied_inputs": True, "output_consumed": True}
    return result


def fake_binary():
    version = "SIMULATED Compute Sanitizer 13.0"
    return {"status": "measured", "path": "/fixture/compute-sanitizer", "sha256": "a" * 64,
            "version": version, "version_sha256": hashlib.sha256(version.encode()).hexdigest()}


def fake_executor(*, bad_tool=None, omit_log=None, wrong_device=None, timeout=None):
    calls = []
    def run(command, limit, log, **kwargs):
        index = command.index("-m")
        source, target = Path(command[index + 2]), Path(command[index + 3])
        payload = read_json(source)
        stage = payload["eval_config"].get("diagnostic_stage", "native")
        calls.append((stage, payload, kwargs))
        result = native_result(payload)
        if stage != "native":
            result["runtime"] = 999999  # Diagnostic timing must never replace the original sample.
        if stage == wrong_device:
            result["metadata"]["device_fingerprint"] = {**DEVICE, "uuid": "wrong-physical-device"}
        if stage in tool_checks.TOOLS and stage != omit_log:
            if stage == "racecheck":
                summary = "RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)" if stage == bad_tool else "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            else:
                summary = "ERROR SUMMARY: 1 error" if stage == bad_tool else "ERROR SUMMARY: 0 errors"
            pattern = command[command.index("--log-file") + 1]
            Path(pattern.replace("%p", "123")).write_text("========= COMPUTE-SANITIZER\n========= " + summary + "\n")
        if stage == "control" and payload["eval_config"].get("determinism_checks"):
            policy, repeats = payload["eval_config"]["determinism_policy"], payload["eval_config"]["determinism_repeats"]
            result["metadata"]["repeat_audit"] = {"protocol": determinism.PROTOCOL, "policy": policy, "repeats": repeats,
                "cases": [{"status": "passed", "label": "standard/0", "seed": 42, "candidate_calls": repeats,
                           "reference_calls": repeats, "output_sha256": ["b" * 64] * repeats, "issues": []}]}
        atomic_json(target, result)
        Path(log).write_text("simulated worker")
        return {"exit_code": 124 if stage == timeout else 0, "timed_out": stage == timeout, "request_wall_seconds": 0.01}
    return run, calls


def test_complete_sanitizer_receipt_keeps_original_times_and_cannot_claim_real_execution(payload):
    run, calls = fake_executor()
    result = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    assert result["correctness"] and result["runtime"] == 1 and result["runtime_stats"]["samples_ms"] == [1]
    assert [row[0] for row in calls] == ["native", *tool_checks.TOOLS, "control"]
    assert all(row[1]["eval_config"]["measure_performance"] is False for row in calls[1:])
    checks = tool_checks.validate(result, payload, allow_simulated=True)
    assert all(accepted(item) for item in checks.values())
    assert checks["memory_safety"]["values"]["race_freedom_proven"] is False
    assert tool_checks.issue(result, payload) == "tool_checks_missing_stale_or_invalid"
    assert all(row[2]["environment"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1" for row in calls if row[0] in tool_checks.TOOLS)


@pytest.mark.parametrize("failure", ["error", "warning", "missing_log", "timeout", "device"])
def test_tool_failures_missing_logs_timeouts_and_device_changes_do_not_qualify(payload, failure):
    options = {"bad_tool": "memcheck"} if failure == "error" else {"bad_tool": "racecheck"} if failure == "warning" else {"omit_log": "initcheck"} if failure == "missing_log" else {"timeout": "synccheck"} if failure == "timeout" else {"wrong_device": "memcheck"}
    run, _ = fake_executor(**options)
    value = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    assert not value["correctness"]
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    expected = "failed" if failure in {"error", "warning"} else "not_measured"
    assert checks["memory_safety"]["status"] == expected


def test_mutated_tool_log_or_relabelled_source_invalidates_the_receipt(payload):
    run, _ = fake_executor()
    value = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    changed = copy.deepcopy(value)
    changed["metadata"]["tool_checks"]["tools"]["memcheck"]["logs"]["memcheck.123.log"]["text"] += "changed"
    with pytest.raises(ValueError, match="receipt"):
        tool_checks.validate(changed, payload, allow_simulated=True)
    payload["kernel_src"] += "\n# changed candidate\n"
    with pytest.raises(ValueError):
        tool_checks.validate(value, payload, allow_simulated=True)


def test_repeat_configuration_has_explicit_limits_and_neuron_does_not_accept_cuda_tools(tmp_path):
    config = parse_config(["--sanitizer-tools", "racecheck,memcheck", "--determinism-checks", "--determinism-repeats", "7"])
    assert config.sanitizer_tools == "memcheck,racecheck" and config.determinism_repeats == 7
    for value in (2, 101, True):
        with pytest.raises(ValueError):
            RunConfig(determinism_repeats=value).validate()
    with pytest.raises(ValueError, match="requires CUDA"):
        RunConfig(device_backend="neuron", hardware="Trainium2", generation="local", enable_ncu=False, sanitizer_tools="all").validate()


def test_sanitizer_command_tracks_children_without_serializing_launches_or_suppressing_errors(tmp_path):
    value = tool_runner.command("memcheck", "/vendor/tool", tmp_path / "memcheck.%p.log", [sys.executable, "-m", "worker"])
    assert value[value.index("--target-processes") + 1] == "all"
    assert value[value.index("--error-exitcode") + 1] == str(tool_checks.ERROR_EXIT)
    assert "--force-blocking-launches" not in value and "--suppressions" not in value


def test_parent_process_runner_records_a_real_cpu_child_without_gpu_or_vendor_tools(tmp_path):
    log = tmp_path / "worker.log"
    result = tool_runner.process([sys.executable, "-c", "print('CPU process regression')"], 5, log)
    assert result["exit_code"] == 0 and result["timed_out"] is False
    assert log.read_text().strip() == "CPU process regression"


def test_control_environment_removes_external_instrumentation_overrides(monkeypatch):
    monkeypatch.setenv("NV_COMPUTE_SANITIZER_BINARY_PATCHING", "0")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/untrusted/injection")
    clean = tool_runner.environment(sanitized=True)
    assert "NV_COMPUTE_SANITIZER_BINARY_PATCHING" not in clean and "CUDA_INJECTION64_PATH" not in clean
    assert clean["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"


@pytest.mark.parametrize("damage", ["truncated_log", "uncollected_log", "wrong_payload", "wrong_source", "nonzero_exit",
                                 "unbound_control", "missing_repeat_case", "extra_native_case", "different_repeat_seed", "repeat_bits", "different_tool_seed"])
def test_resealed_raw_evidence_still_cannot_qualify_incomplete_or_mismatched_checks(payload, damage):
    run, _ = fake_executor()
    document = copy.deepcopy(tool_runner.evaluate(payload, run=run, binary=fake_binary())["metadata"]["tool_checks"])
    mem = document["tools"]["memcheck"]
    if damage == "truncated_log":
        log = mem["logs"]["memcheck.123.log"]
        log["text"] = "COMPUTE-SANITIZER\nworker interrupted"
        log["sha256"] = hashlib.sha256(log["text"].encode()).hexdigest()
    elif damage == "uncollected_log":
        mem["error"] = "Another worker log exceeded the retention limit"
    elif damage == "wrong_payload":
        mem["payload_sha256"] = "f" * 64
    elif damage == "wrong_source":
        mem["result"]["metadata"]["kernel_sha256"] = "f" * 64
    elif damage == "nonzero_exit":
        mem["exit_code"] = 1
    elif damage == "unbound_control":
        document["control"]["payload_sha256"] = "f" * 64
    elif damage == "different_tool_seed":
        mem["result"]["metadata"]["reliability_checks"][0]["seed"] += 1
    else:
        control = document["control"]["result"]["metadata"]
        if damage == "missing_repeat_case":
            control["repeat_audit"]["cases"] = []
        elif damage == "extra_native_case":
            control["reliability_checks"].append({**control["reliability_checks"][0], "label": "standard/1"})
        elif damage == "different_repeat_seed":
            control["repeat_audit"]["cases"][0]["seed"] += 1
        else:
            control["repeat_audit"]["cases"][0]["output_sha256"][-1] = "c" * 64
    value = tool_checks.finish(document, payload)
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    assert not value["correctness"] and any(row["status"] != "passed" for row in checks.values())


def test_missing_executable_is_a_recorded_unknown_with_no_false_pass(payload):
    run, calls = fake_executor()
    value = tool_runner.evaluate(payload, run=run, binary={"status": "not_measured", "reason": "Tool absent in CPU regression"})
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    assert not value["correctness"] and checks["memory_safety"]["status"] == "not_measured"
    assert [stage for stage, _, _ in calls] == ["native", "control"]


def test_partial_tool_selection_does_not_establish_full_memory_safety(payload):
    payload["eval_config"]["sanitizer_tools"] = "memcheck"
    run, calls = fake_executor()
    value = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    assert value["correctness"] and accepted(checks["sanitizer_memcheck"])
    assert checks["memory_safety"]["status"] == "not_measured"
    assert [stage for stage, _, _ in calls] == ["native", "memcheck", "control"]


def test_repeat_only_needs_no_cuda_sanitizer_and_batch_index_does_not_modify_evidence(payload):
    payload["eval_config"]["sanitizer_tools"] = "none"
    run, calls = fake_executor()
    value = tool_runner.evaluate(payload, run=run)
    value["sample_id"] = 0
    value["metadata"].update(workflow_signature=fingerprint(payload), evaluation_config=payload["eval_config"])
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    assert value["correctness"] and accepted(checks["repeat_consistency"])
    assert "memory_safety" not in checks and [stage for stage, _, _ in calls] == ["native", "control"]


def test_matching_negative_results_resume_but_changed_raw_evidence_does_not(payload, monkeypatch):
    run, _ = fake_executor(bad_tool="memcheck")
    value = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    value["metadata"].update(workflow_signature=fingerprint(payload), evaluation_config=payload["eval_config"])
    validate = tool_checks.validate
    monkeypatch.setattr(tool_checks, "validate", lambda result, request: validate(result, request, allow_simulated=True))
    assert tool_checks.reusable(value, payload)
    value["metadata"]["tool_checks"]["tools"]["memcheck"]["logs"].clear()
    assert not tool_checks.reusable(value, payload)


def test_bounded_atomics_retains_native_accuracy_and_rejects_unreviewed_worker_policy():
    class Reference(torch.nn.Module):
        def forward(self, x):
            return x + 1
    class Wrong(torch.nn.Module):
        def forward(self, x):
            return x + 2
    policy = {"mode": "bounded_atomics", "atomics_allowed": True, "atol": 100, "rtol": 100}
    with determinism.audit(policy, 3) as collector:
        collector.observe(Reference(), Wrong(), [torch.ones(2)], device="cpu", seed=9,
                          set_seed=torch.manual_seed, tolerance=1e-5, label="accuracy")
    assert collector.document()["check"]["status"] == "failed"
    with pytest.raises(ValueError, match="reviewed"):
        determinism.validate_inline({**policy, "reference_sha256": hashlib.sha256(b"reference").hexdigest()}, "reference")


def test_declared_bounded_atomic_variation_can_pass_without_bitwise_identity():
    calls = [0]
    class Reference(torch.nn.Module):
        def forward(self, x):
            return x + 1
    class AtomicVariation(torch.nn.Module):
        def forward(self, x):
            calls[0] += 1
            return x + 1 + calls[0] * 1e-6
    with determinism.audit({"mode": "bounded_atomics", "atol": 1e-5, "rtol": 1e-5}, 5) as collector:
        collector.observe(Reference(), AtomicVariation(), [torch.ones(2)], device="cpu", seed=9,
                          set_seed=torch.manual_seed, tolerance=1e-4, label="bounded")
    assert accepted(collector.document()["check"])
    assert len(set(collector.document()["cases"][0]["output_sha256"])) > 1


def test_unavailable_model_snapshot_is_unknown_and_does_not_call_the_candidate():
    class Unsnapshotable(torch.nn.Module):
        def __deepcopy__(self, memo):
            raise RuntimeError("No snapshot support")
        def forward(self, x):
            raise AssertionError("An unknown snapshot must not execute candidate repeats")
    with determinism.audit({"mode": "bitwise"}, 3) as collector:
        collector.observe(Unsnapshotable(), Unsnapshotable(), [torch.ones(2)], device="cpu", seed=9,
                          set_seed=torch.manual_seed, tolerance=1e-5, label="snapshot")
    assert collector.document()["check"]["status"] == "not_measured"
    assert collector.document()["cases"][0]["candidate_calls"] == 0


def test_model_deepcopy_cannot_silently_reuse_original_state():
    class Shared(Counter):
        def __deepcopy__(self, memo):
            return self
    model = Shared()
    with determinism.audit({"mode": "bitwise"}, 3) as collector:
        collector.observe(Counter(), model, [torch.ones(2)], device="cpu", seed=9,
                          set_seed=torch.manual_seed, tolerance=1e-5, label="shared")
    assert collector.document()["check"]["status"] == "not_measured"
    assert model.calls == 0


def test_policy_file_contents_bind_generation_frozen_verification_studies_and_rl_inputs(tmp_path):
    from kb_workflows.agents import generation, rl_plan
    from kb_workflows.analysis import study_data
    from kb_workflows.analysis.experiment import protocol
    from kb_workflows.common import select_problems
    from kb_workflows.evaluation import final_verification

    reference = select_problems(1, 2, 2)[2].read_text()
    path = tmp_path / "policies.json"
    document = {"protocol": determinism.POLICY_PROTOCOL, "tasks": {"1/2": {
        "mode": "bounded_atomics", "atomics_allowed": True, "atol": 1e-6, "rtol": 1e-6,
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "review": {"reviewer": "fixture", "rationale": "Regression variation bound"}}}}
    atomic_json(path, document)
    config = RunConfig(generation="local", hardware="RTX_A6000", start=2, end=2, runs_dir=str(tmp_path),
                       determinism_checks=True, determinism_policy_file=str(path)).validate()
    readers = [lambda: protocol(config), lambda: generation.generation_signature(config, 2, 0),
               lambda: final_verification._options(config), lambda: study_data.evaluation_contract(config),
               lambda: rl_plan.verifier_inputs(config.to_dict())]
    before = [read() for read in readers]
    document["tasks"]["1/2"]["atol"] = 2e-6
    atomic_json(path, document)
    assert all(old != read() for old, read in zip(before, readers))
    job = generation.prepare_job(config, 2, 0)
    saved = read_json(job / "job_config.json")
    assert saved["determinism_policy_file"] == str(job / "determinism_policy.json")
    assert read_json(Path(saved["determinism_policy_file"])) == document


def test_harbor_exports_selected_policy_only_into_the_trusted_verifier(tmp_path):
    from kb_workflows.common import select_problems
    from kb_workflows.runtime import harbor

    source = select_problems(1, 2, 2)[2].read_text()
    path = tmp_path / "policies.json"
    row = {"mode": "bitwise", "reference_sha256": hashlib.sha256(source.encode()).hexdigest()}
    atomic_json(path, {"protocol": determinism.POLICY_PROTOCOL, "tasks": {"1/2": row, "1/3": row}})
    config = RunConfig(generation="harbor", hardware="RTX_A6000", start=2, end=2,
                       determinism_checks=True, determinism_policy_file=str(path)).validate()
    task = harbor.write_task(tmp_path / "harbor-task", config, 2)
    saved = read_json(task / "tests/config.json")["config"]
    assert saved["determinism_policy_file"] == "/tests/determinism_policies.json"
    assert read_json(task / "tests/determinism_policies.json")["tasks"] == {"1/2": row}
    assert not (task / "environment/determinism_policies.json").exists()


def test_real_cpu_timeout_is_explicit_and_does_not_create_vendor_measurements(tmp_path):
    value = tool_runner.process([sys.executable, "-c", "import time; time.sleep(30)"], 0.1, tmp_path / "timeout.log")
    assert value["timed_out"] is True and value["exit_code"] == 124


def test_sanitizer_discovery_cannot_spend_beyond_the_remaining_request(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tool_runner.shutil, "which", lambda value: sys.executable)
    def run(command, timeout, log, **kwargs):
        calls.append(timeout)
        Path(log).write_text("SIMULATED version\n")
        return {"exit_code": 0, "timed_out": False}
    assert tool_runner.binary_identity("configured", directory=tmp_path, run=run, timeout=0.25)["status"] == "measured"
    assert calls == [0.25]
    assert tool_runner.binary_identity("configured", directory=tmp_path, run=run, timeout=0)["status"] == "not_measured"
    assert calls == [0.25]


def test_paired_measurements_keep_their_original_samples_and_run_separate_correctness_checks(payload):
    payload["task"] = "paired_timing"
    run, calls = fake_executor()
    value = tool_runner.evaluate(payload, run=run, binary=fake_binary())
    checks = tool_checks.validate(value, payload, allow_simulated=True)
    assert value["correctness"] and all(accepted(row) for row in checks.values())
    assert len(value["metadata"]["paired_timing"]["pairs"]) == 10
    assert value["runtime"] == 1
    assert all(child["task"] == "evaluate" for _, child, _ in calls[1:])


@pytest.mark.parametrize("value", [None, [], {"protocol": tool_checks.PROTOCOL, "tools": [None]}])
def test_malformed_receipt_fails_closed_without_crashing_offline_analysis(payload, value):
    result = native_result(payload)
    result["metadata"]["tool_checks"] = value
    assert tool_checks.issue(result, payload) == "tool_checks_missing_stale_or_invalid"


def test_exact_scalar_repeat_signs_are_not_lost_in_json_equality():
    assert determinism.repeat_issue(0.0, -0.0, {"mode": "bitwise"})
    assert determinism.repeat_issue(complex(0.0, 1), complex(-0.0, 1), {"mode": "bitwise"})


@pytest.mark.parametrize("private_worker", [False, True])
def test_native_cli_dispatches_to_the_supervisor_and_cannot_bypass_required_receipts(payload, tmp_path, monkeypatch, private_worker):
    from kb_workflows.evaluation import evaluate_task

    source, target = tmp_path / "cli-request.json", tmp_path / "cli-result.json"
    atomic_json(source, payload)
    seen = []
    def dispatch(value):
        seen.append("native")
        return native_result(value)
    def supervise(value):
        seen.append("supervisor")
        return native_result(value)
    monkeypatch.setattr(evaluate_task, "dispatch", dispatch)
    monkeypatch.setattr(tool_runner, "evaluate", supervise)
    monkeypatch.setattr(sys, "argv", ["evaluate_task", str(source), str(target), *(["--native-worker"] if private_worker else [])])
    evaluate_task.main()
    result = read_json(target)
    assert seen == ["native" if private_worker else "supervisor"]
    assert tool_checks.issue(result, payload) == "tool_checks_missing_stale_or_invalid"


def test_native_control_collects_fresh_repeats_with_the_frozen_task_policy(payload, tmp_path, monkeypatch):
    from kb_workflows.evaluation import evaluate_task
    from src.reliability import check_paired_calls

    control = tool_checks.child_payload(payload, "f" * 32, "control")
    source, target = tmp_path / "control-request.json", tmp_path / "control-result.json"
    atomic_json(source, control)
    def dispatch(value):
        result = native_result(value)
        result["metadata"]["reliability_checks"] = []
        ref, cand = torch.nn.Identity(), torch.nn.Identity()
        assert check_paired_calls(ref, cand, [torch.ones(2)], device="cpu", seed=42, tolerance=1e-5,
                                 set_seed=torch.manual_seed, metadata=result["metadata"], label="standard/0") is None
        return result
    monkeypatch.setattr(evaluate_task, "dispatch", dispatch)
    monkeypatch.setattr(sys, "argv", ["evaluate_task", str(source), str(target), "--native-worker"])
    evaluate_task.main()
    metadata = read_json(target)["metadata"]
    verdict = tool_checks.repeat_verdict(metadata["repeat_audit"], control["eval_config"]["determinism_policy"], 3, metadata["reliability_checks"])
    assert accepted(verdict)
