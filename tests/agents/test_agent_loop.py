"""Exercise the real mini-swe-agent loop with simulated model/GPU responses."""

import json
import os
import subprocess
import sys

import pytest

from kb_workflows import engine
from kb_workflows.agents import agent_loop, agent_models, agent_task, generation
from kb_workflows.analysis.experiment import compare_runs, existing_budget, protocol, write_budget_report
from kb_workflows.common import atomic_json, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return RunConfig(run_name="loop", runs_dir=str(tmp_path), generation="local", start=90, end=90,
                     hardware="RTX_A6000", optimization_rounds=3, agent_profile=False, agent_timeout=30).validate()


def responses(calls, costs=None):
    def respond(messages, config, directory, timeout, remaining_cost):
        calls.append(messages.copy())
        return {"text": f"```python\nclass ModelNew:\n    version = {len(calls)}\n```", "cost": 0.1 if costs is None else costs[len(calls)-1],
                "input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    return respond


def evaluated(runtime, correct=True, *, origin=None):
    metadata = {"evidence_receipt": {"checks": {"kernel_origin": {"status": origin, "reason": "CPU feedback fixture",
        "values": {"track": "custom", "minimum_custom_fraction": 0.95}}}}} if origin else {}
    return {"compiled": True, "correctness": correct, "runtime": runtime, "metadata": metadata}


def test_real_mini_loop_repairs_and_retains_best_instead_of_last(config):
    directory = generation.prepare_job(config, 90, 0)
    calls, validations = [], []

    def evaluate(child, pid, index, remaining, task="evaluate"):
        validations.append(task)
        return [evaluated(8, False), evaluated(2), evaluated(1, False)][index-1]

    result = agent_loop.run_loop(directory, RunConfig(**read_json(directory / "job_config.json")).validate(), 90,
                                 response_fn=responses(calls), evaluator=evaluate)
    assert result["status"] == "success" and result["scaffold_version"] == "2.4.6"
    assert result["selected"]["round"] == 2
    assert "version = 2" in (directory / "runs/agent" / kernel_name(1, 90)).read_text()
    assert len(calls) == 3 and validations == ["evaluate"]*3
    assert json.loads(calls[1][-1]["content"])["feedback"]["correctness"] is False
    assert result["budget"]["model_calls"] == result["budget"]["candidate_submissions"] == 3
    assert result["budget"]["estimated_cost_usd"] == pytest.approx(0.3)
    assert result["budget"]["input_tokens"] == 30 and result["budget"]["output_tokens"] == 60
    trajectory = read_json(directory / "trajectory.json")
    assert trajectory["info"]["config"]["agent_type"] == "minisweagent.agents.default.DefaultAgent"
    assert not result["final_hidden_tests_run"]

    config.start = config.end = 1
    config.evidence_level = "L1"
    directory = generation.prepare_job(config, 1, 0)
    source_calls = []
    def source_evaluate(child, pid, index, remaining, task="evaluate"):
        return evaluated(4 / index, origin=["passed", "failed", "not_measured"][index - 1])
    result = agent_loop.run_loop(directory, config, 1, response_fn=responses(source_calls), evaluator=source_evaluate)
    assert result["selected"]["round"] == 1
    assert [row["status"] for row in result["rounds"]] == ["development_pass", "development_unqualified", "development_unqualified"]
    assert all(row["correctness"] for row in result["rounds"])
    assert not result["rounds"][1]["performance_eligible"]
    assert json.loads(source_calls[2][-1]["content"])["feedback"]["kernel_origin"]["status"] == "failed"


def test_one_shot_has_one_call_and_zero_development_or_profile_calls(config):
    config.generation_mode, config.optimization_rounds, config.agent_profile = "one-shot", 9, True
    directory = generation.prepare_job(config, 90, 0)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("One-shot must not get execution feedback")

    result = agent_loop.run_loop(directory, config, 90, response_fn=responses(calls), evaluator=forbidden)
    assert len(calls) == 1 and result["status"] == "success"
    assert result["budget"]["development_evaluations"] == result["budget"]["profile_requests"] == 0
    assert result["selected"]["development_verified"] is False
    assert "def get_inputs" not in calls[0][-1]["content"]


def test_profile_feedback_is_passed_to_next_round_and_cannot_win_by_profiler_time(config):
    config.agent_profile, config.optimization_rounds = True, 2
    directory = generation.prepare_job(config, 90, 0)
    calls, tasks = [], []

    def evaluate(child, pid, index, remaining, task="evaluate"):
        tasks.append(task)
        assert f"version = {index}" in (child.run_dir / kernel_name(1, 90)).read_text()
        return {"purpose": "diagnostics_only", "events": [{"name": "copy", "device_self_us": 1}]} if task == "profile" else evaluated(index)

    result = agent_loop.run_loop(directory, config, 90, response_fn=responses(calls), evaluator=evaluate)
    assert tasks == ["evaluate", "profile", "evaluate", "profile"]
    assert "diagnostics_only" in calls[1][-1]["content"]
    assert result["selected"]["round"] == 1 and result["budget"]["profile_requests"] == 2


def test_unavailable_profiler_does_not_invalidate_correct_candidate(config):
    config.agent_profile, config.optimization_rounds = True, 1
    directory = generation.prepare_job(config, 90, 0)

    def evaluate(child, pid, index, remaining, task="evaluate"):
        if task == "profile":
            raise RuntimeError("CUPTI unavailable")
        return evaluated(1)

    result = agent_loop.run_loop(directory, config, 90, response_fn=responses([]), evaluator=evaluate)
    assert result["status"] == "success" and result["rounds"][0]["profile"]["status"] == "unavailable"


def test_cost_cap_stops_before_another_call_and_reports_single_call_overshoot(config):
    config.agent_cost_limit = 0.05
    directory = generation.prepare_job(config, 90, 0)
    calls = []
    result = agent_loop.run_loop(directory, config, 90, response_fn=responses(calls), evaluator=lambda *a, **k: evaluated(1))
    assert len(calls) == 1 and result["exit_status"] == "LimitsExceeded"
    assert result["budget"]["estimated_cost_usd"] == 0.1


def test_unknown_cost_is_reported_unknown_and_stops_when_cost_cap_is_requested(config):
    config.agent_cost_limit = 1
    for provider_error in (False, True):
        directory = generation.prepare_job(config, 90, 0)
        calls = []
        respond = responses(calls, [None])

        def response(*args):
            return {**respond(*args), "provider_error": provider_error}

        result = agent_loop.run_loop(directory, config, 90, response_fn=response)
        assert len(calls) == 1 and result["status"] == "failed" and result["exit_status"] == "CostEvidenceMissing"
        assert result["budget"]["estimated_cost_usd"] is None
        assert read_json(directory / "trajectory.json")["info"]["model_stats"]["instance_cost"] is None


def test_model_error_keeps_best_and_records_incomplete_usage(config):
    directory = generation.prepare_job(config, 90, 0)
    calls = []
    respond = responses(calls)

    def fail_after_first(*args):
        if calls:
            raise TimeoutError("provider deadline")
        return respond(*args)

    result = agent_loop.run_loop(directory, config, 90, response_fn=fail_after_first, evaluator=lambda *a, **k: evaluated(1))
    assert result["status"] == "success" and result["selected"]["round"] == 1
    assert result["budget"]["model_calls"] == 2 and result["budget"]["estimated_cost_usd"] is None
    assert "TimeoutError" in result["error"]


def test_litellm_adapter_uses_sdk_response_and_never_sends_controller_extras(config, monkeypatch):
    import litellm

    from kb_workflows.agents.agent_models import litellm_response

    config.model = "test-provider/test-model"
    calls = []
    response = litellm.ModelResponse(model="observed-model", choices=[{"message": {"role": "assistant", "content": "class ModelNew: pass"}}],
                                   usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12})
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: calls.append(kwargs) or response)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.02)
    result = litellm_response([{"role": "user", "content": "task", "extra": {"private_controller_field": "ignored"}}], config, None, 12)
    assert calls[0]["messages"] == [{"role": "user", "content": "task"}]
    assert calls[0]["num_retries"] == 0 and calls[0]["timeout"] == 12
    assert result["models"] == ["observed-model"] and result["input_tokens"] == 9 and result["cost"] == 0.02


def test_development_payload_never_requests_hidden_inputs(config, monkeypatch):
    directory = generation.prepare_job(config, 90, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    output = child.run_dir / kernel_name(1, 90)
    output.parent.mkdir(parents=True)
    output.write_text("class ModelNew: pass")
    payloads = []

    def evaluate(cfg, tasks):
        payloads.extend(payload for _, payload in tasks)
        yield tasks[0][0], evaluated(1)

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    for task in ("evaluate", "profile"):
        agent_loop.development_evaluate(child, 90, 1, lambda: 12, task)
    assert all(p["eval_config"]["use_hidden_tests"] is False for p in payloads)
    assert all("hidden_source_sha256" not in p["eval_config"] and "kernel_src" in p for p in payloads)
    assert child.hidden_tests is True


@pytest.mark.parametrize("arguments", [["--agent-cost-limit", "nan"], ["--agent-cost-limit", "-1"],
    ["--agent-backend", "bad"], ["--generation-mode", "bad"], ["--generation", "harbor", "--generation-mode", "one-shot"]])
def test_invalid_protocol_settings_rejected(arguments):
    with pytest.raises((ValueError, SystemExit)):
        parse_config(arguments)


def test_protocol_and_budget_changes_invalidate_generation_resume(config):
    original = generation.generation_signature(config, 90, 0)
    config.generation_mode = "one-shot"
    assert generation.generation_signature(config, 90, 0) != original
    config.generation_mode = "feedback"
    config.agent_timeout += 1
    assert generation.generation_signature(config, 90, 0) != original


def test_missing_slots_are_not_reported_as_free_generation(config):
    report = write_budget_report(config, {})
    assert report["totals"]["estimated_cost_usd"] is None and not report["complete_resource_receipts"]
    config.generation = "harbor"
    assert protocol(config)["mode"] == "external" and protocol(config)["candidate_limit_per_sample"] is None


def test_source_changes_invalidate_generation_protocol_attribution(config):
    import hashlib

    config.run_dir.mkdir(parents=True)
    path = config.run_dir / kernel_name(1, 90)
    path.write_text("first source")
    write_budget_report(config, {"90_0": {"kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
    assert existing_budget(config)["provenance_valid"]
    path.write_text("modified by a later session")
    report = existing_budget(config)
    assert not report["provenance_valid"] and report["protocol"]["mode"] == "unattributed"


def test_comparison_separates_modes_and_budgets_but_allows_different_models(config):
    paths = []
    for name, mode, rounds, model in (("a", "one-shot", 5, "model-a"), ("b", "one-shot", 5, "model-b"),
                                      ("c", "feedback", 5, "model-a"), ("d", "feedback", 3, "model-a")):
        config.run_name, config.generation_mode, config.optimization_rounds, config.model = name, mode, rounds, model
        config.run_dir.mkdir(parents=True)
        atomic_json(config.run_dir / "run_config.json", config.to_dict())
        write_budget_report(config, {})
        paths.append(config.run_dir)
    groups = compare_runs(paths)["groups"]
    assert len(groups) == 3 and len(groups[0]["runs"]) == 2
    assert {row["declared_model"] for row in groups[0]["runs"]} == {"model-a", "model-b"}


def test_interactive_prepare_needs_no_model_or_gpu_execution(config, capsys):
    assert agent_task.main(["prepare", "--generation", "local", "--evaluation", "local",
                            "--gpu-profile", "RTX_A6000", "--run-name", config.run_name, "--runs-dir", config.runs_dir,
                            "--start", "90", "--end", "90", "--num-samples", "1"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert "p90_s0" in manifest["directory"]
    assert "model.py" in manifest["reference"] and not (config.run_dir / "eval_results_hidden.json").exists()


def test_interactive_bridge_standard_feedback_freeze_and_no_nested_agent(config, monkeypatch):
    directory = generation.prepare_job(config, 90, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    output = child.run_dir / kernel_name(1, 90)
    output.parent.mkdir(parents=True)
    output.write_text("class ModelNew: pass")
    destination = config.run_dir / kernel_name(1, 90)
    atomic_json(directory / "interactive_state.json", {"rounds": [], "destination": str(destination)})
    assert agent_task.verify_directory(directory, evaluator=lambda *a, **k: evaluated(1)) == 0
    assert agent_task.main(["finalize", "--directory", str(directory)]) == 0
    assert destination.read_text() == "class ModelNew: pass"
    assert agent_task.verify_directory(directory) == 2
    budget = read_json(config.run_dir / "generation_budget.json")
    assert budget["protocol"]["mode"] == "interactive" and budget["model_calls"] is None

    config.start = config.end = 1
    config.evidence_level = "L1"
    directory = generation.prepare_job(config, 1, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    output = child.run_dir / kernel_name(1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("qualified source")
    destination = config.run_dir / kernel_name(1, 1)
    atomic_json(directory / "interactive_state.json", {"rounds": [], "destination": str(destination)})
    assert agent_task.verify_directory(directory, evaluator=lambda *a, **k: evaluated(2, origin="passed")) == 0
    output.write_text("fast library source")
    assert agent_task.verify_directory(directory, evaluator=lambda *a, **k: evaluated(1, origin="failed")) == 1
    state = read_json(directory / "interactive_state.json")
    assert state["rounds"][-1]["correctness"] and not state["rounds"][-1]["performance_eligible"]
    assert state["best"]["source"] == "qualified source"
    assert agent_task.main(["finalize", "--directory", str(directory)]) == 0
    assert destination.read_text() == "qualified source"


def fake_claude(tmp_path, monkeypatch, sleepy=False):
    executable = tmp_path / "claude"
    content = '''import json, sys
from pathlib import Path
args=sys.argv[1:]
assert args[args.index('--tools')+1] == ''
assert '--strict-mcp-config' in args and '--safe-mode' in args
messages=json.loads(sys.stdin.read())
assert 'def get_inputs' not in messages[-1]['content']
'''
    if sleepy:
        content += '''import subprocess, time
child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
Path('nested.pid').write_text(str(child.pid))
time.sleep(60)
'''
    else:
        content += '''print(json.dumps({'result':'```python\\nclass ModelNew: pass\\n```','total_cost_usd':0.01,
    'usage':{'input_tokens':11,'output_tokens':7,'cache_read_input_tokens':0,'cache_creation_input_tokens':0},'num_turns':1}))
'''
    executable.write_text(f"#!{sys.executable}\n" + content)
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_real_worker_and_fake_cli_one_shot_multiple_samples_and_resume(config, tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    config.generation_mode, config.num_samples = "one-shot", 2
    assert generation.generate_run(config)
    report = read_json(config.run_dir / "generation_budget.json")
    assert report["totals"]["model_calls"] == 2 and report["totals"]["development_evaluations"] == 0
    assert report["totals"]["input_tokens"] == 22 and report["totals"]["estimated_cost_usd"] == 0.02
    assert report["protocol"]["mode"] == "one-shot"
    config.resume = True
    monkeypatch.setattr(generation, "run_agent", lambda *a, **k: pytest.fail("Matching resume should reuse generation"))
    assert generation.generate_run(config)


def test_explicit_repeat_preserves_prior_cost_and_is_not_labeled_one_shot(config, tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    config.generation_mode = "one-shot"
    assert generation.generate_run(config)
    assert generation.generate_run(config)
    report = read_json(config.run_dir / "generation_budget.json")
    assert report["totals"]["model_calls"] == 2
    assert report["totals"]["estimated_cost_usd"] == 0.02
    assert report["protocol"]["mode"] == "repeated-attempts" and report["protocol"]["base_mode"] == "one-shot"


def test_resume_keeps_a_completed_failure_instead_of_spending_another_budget(config, monkeypatch):
    calls = []
    monkeypatch.setattr(generation, "run_agent", lambda *a, **k: calls.append(1) or {"status": "failed"})
    assert not generation.generate_run(config)
    config.resume = True
    assert not generation.generate_run(config)
    assert len(calls) == 1


def test_wall_timeout_stops_nested_model_children(config, tmp_path, monkeypatch):
    import psutil

    fake_claude(tmp_path, monkeypatch, sleepy=True)
    config.generation_mode, config.agent_timeout = "one-shot", 2
    directory = generation.prepare_job(config, 90, 0)
    result = generation.run_agent(directory, config, 90, "0")
    assert result["exit_status"] == "WallTimeExceeded" and result["budget"] is None
    assert not (directory / "runs/agent" / kernel_name(1, 90)).exists()
    nested = directory / "nested.pid"
    assert nested.exists(), (directory / "agent.log").read_text()
    try:
        child = psutil.Process(int(nested.read_text()))
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_model_call_timeout_stops_nested_children_before_controller_timeout(config, tmp_path, monkeypatch):
    import psutil

    fake_claude(tmp_path, monkeypatch, sleepy=True)
    with pytest.raises(subprocess.TimeoutExpired):
        agent_models.claude_response([{"role": "user", "content": "public reference"}], config, tmp_path, timeout=2)
    pid = int((tmp_path / "nested.pid").read_text())
    try:
        child = psutil.Process(pid)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
