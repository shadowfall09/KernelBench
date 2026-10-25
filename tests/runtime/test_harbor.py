"""Harbor boundary, reward and unified-generation regression tests."""

import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb_workflows import cli, engine
from kb_workflows.agents import generation
from kb_workflows.analysis import paper_analysis
from kb_workflows.common import atomic_json, evaluator_digest, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.runtime import harbor, harbor_verifier


@pytest.fixture
def config(tmp_path):
    return RunConfig(run_name="harbor-test", runs_dir=str(tmp_path), generation="harbor",
                     stage="generate", start=90, end=90, hardware="RTX_A6000",
                     gpu="RTX_A6000", gpu_arch="Ampere", num_correct_trials=2, num_perf_trials=3).validate()


def test_export_keeps_trusted_sources_out_of_agent_image(config):
    task = harbor.write_task(config.run_dir / "task", config, 90)
    model = (task / "environment/model.py").read_text()
    assert "class Model" in model and "def get_inputs" not in model
    assert not (task / "environment/repo").exists()
    assert not (task / "environment/solution.py").exists()
    assert set(p.name for p in (task / "environment").iterdir()) <= {"model.py", "Dockerfile", "requirements.txt", "LICENSE", "NOTICE.md"}
    assert (task / "tests/repo/hidden_tests/level1/90_hidden.py").exists()
    assert (task / "tests/repo/src/eval.py").exists()
    assert (task / "tests/repo/kb_workflows/runtime/harbor_verifier.py").exists()
    assert "get_inputs" in (task / "solution/reference.py").read_text()
    for name in ("environment", "tests"):
        dockerfile = (task / name / "Dockerfile").read_text()
        assert "python3.12" in dockerfile and "ubuntu24.04" in dockerfile
    parsed = tomllib.loads((task / "task.toml").read_text())
    assert parsed["schema_version"] == "1.4"
    assert parsed["artifacts"] == ["/app/solution.py"]
    assert parsed["verifier"]["environment_mode"] == "separate"
    assert parsed["verifier"]["environment"]["gpus"] == 1
    assert parsed["environment"]["gpus"] == 1
    assert parsed["verifier"]["timeout_sec"] > config.timeout * 6
    for script in (task / "tests/test.sh", task / "solution/solve.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_export_validates_with_actual_harbor_task_model(config):
    Task = pytest.importorskip("harbor.models.task.task").Task
    config.harbor_memory_mb = 81920
    task = harbor.write_task(config.run_dir / "task", config, 90, device="GPU-device")
    loaded = Task(task)
    assert loaded.config.task.name == "kernelbench-pro/l1-p90-s0"
    assert loaded.config.verifier.environment_mode.value == "separate"
    assert loaded.config.environment.memory_mb == loaded.config.verifier.environment.memory_mb == 81920
    assert Task.is_valid_dir(task)
    assert 'device_ids: ["GPU-device"]' in (task / "tests/docker-compose.yaml").read_text()


def test_export_uses_native_samples_and_does_not_overwrite(config):
    config.num_samples = 2
    output = harbor.export_tasks(config)
    assert sorted(p.name for p in output.iterdir()) == ["l1-p90-s0", "l1-p90-s1"]
    for sid in range(2):
        spec = read_json(output / f"l1-p90-s{sid}/tests/config.json")
        assert spec["sample_id"] == sid
        assert spec["config"]["num_samples"] == 1
        assert spec["config"]["evaluation"] == "local"
        assert spec["config"]["gpus"] == "0"
        assert spec["evaluator_sha256"] == evaluator_digest()
    with pytest.raises(FileExistsError):
        harbor.export_tasks(config)


def test_export_timeout_accounts_for_expanded_baselines_pairing_and_all_final_repeats(config):
    config.baseline_portfolio, config.final_repeats, config.paired_timing = "extended", 3, True
    task = harbor.write_task(config.run_dir / "expanded-task", config, 90)
    data = tomllib.loads((task / "task.toml").read_text())
    assert data["verifier"]["timeout_sec"] == (6 + 2 + 6 + 3) * config.timeout + 180


def test_export_does_not_embed_host_authentication_configuration(config):
    config.profile = "private-profile"
    config.modal_secret = "private-secret-name"
    config.queue_url = "https://example-user:example-password@queue.invalid"
    config.harbor_command = "/private/harbor"
    task = harbor.write_task(config.run_dir / "task", config, 90)
    text = (task / "tests/config.json").read_text()
    assert "private-profile" not in text and "private-secret-name" not in text
    assert "example-password" not in text and "/private/harbor" not in text


def test_export_is_atomic_if_one_task_fails(config, monkeypatch):
    config.num_samples = 2
    real = harbor.write_task

    def write(path, cfg, pid, sid=0):
        if sid == 1:
            raise ValueError("bad second task")
        return real(path, cfg, pid, sid)

    monkeypatch.setattr(harbor, "write_task", write)
    with pytest.raises(ValueError):
        harbor.export_tasks(config)
    assert not (config.run_dir / "harbor_tasks").exists()


@pytest.mark.parametrize("arguments", [
    ["--generation", "harbor", "--no-hidden-tests"],
    ["--stage", "export-harbor", "--no-hidden-tests"],
    ["--harbor-environment", "unknown"],
    ["--harbor-agent", ""],
])
def test_invalid_harbor_configuration_is_rejected(arguments):
    with pytest.raises((ValueError, SystemExit)):
        parse_config(arguments)


def test_export_dry_run_has_no_files(config):
    assert cli.main(["--stage", "export-harbor", "--runs-dir", config.runs_dir,
                     "--run-name", config.run_name, "--start", "90", "--end", "90", "--dry-run"]) == 0
    assert not config.run_dir.exists()


@pytest.mark.parametrize("device", ["0,1", "x\ny", "$(id)", "../0"])
def test_device_selection_cannot_escape_compose_string(device):
    with pytest.raises(ValueError):
        harbor._compose(device)


def test_harbor_generation_imports_submission_and_resumes(config, monkeypatch):
    calls = []
    names = []
    config.num_samples = 2

    def popen(command, **kwargs):
        calls.append(command)
        jobs = Path(command[command.index("--jobs-dir") + 1])
        name = command[command.index("--job-name") + 1]
        target = jobs / name / "attempt/artifacts/app/solution.py"
        target.parent.mkdir(parents=True)
        target.write_text("class ModelNew: pass\n")
        task = Path(command[command.index("-p") + 1])
        names.append(tomllib.loads((task / "task.toml").read_text())["task"]["name"])
        assert 'device_ids: ["GPU-allocated"]' in (task / "environment/docker-compose.yaml").read_text()
        assert command[command.index("--n-attempts") + 1] == "1"
        return SimpleNamespace(wait=lambda **kw: 0, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(harbor.subprocess, "Popen", popen)
    monkeypatch.setattr(harbor, "terminate", lambda process: None)  # No OS process in this CLI double.
    monkeypatch.setattr(generation, "resolve_devices", lambda value, **kwargs: ["GPU-allocated"])
    assert generation.generate_run(config)
    assert (config.run_dir / kernel_name(1, 90)).read_text() == "class ModelNew: pass\n"
    config.resume = True
    assert generation.generate_run(config) and len(calls) == 2
    assert names == ["kernelbench-pro/l1-p90-s0", "kernelbench-pro/l1-p90-s1"]
    config.harbor_agent = "other-agent"
    assert generation.generate_run(config) and len(calls) == 4


@pytest.mark.parametrize("count", [0, 2])
def test_harbor_backend_rejects_missing_or_ambiguous_artifacts(config, monkeypatch, count):
    def popen(command, **kwargs):
        jobs = Path(command[command.index("--jobs-dir") + 1])
        name = command[command.index("--job-name") + 1]
        for index in range(count):
            output = jobs / name / f"attempt{index}/artifacts/app/solution.py"
            output.parent.mkdir(parents=True)
            output.write_text("candidate")
        return SimpleNamespace(wait=lambda **kw: 0, poll=lambda: 0)

    monkeypatch.setattr(harbor.subprocess, "Popen", popen)
    monkeypatch.setattr(harbor, "terminate", lambda process: None)
    result = harbor.run_harbor_agent(config.run_dir, config, 90, "0")
    assert result["status"] == "failed" and "submission" in result["error"]


def test_missing_harbor_cli_fails_before_baseline_or_generation(config, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda command: None)
    monkeypatch.setattr(engine, "record_baseline", lambda cfg: pytest.fail("Baseline started without Harbor"))
    assert cli.main(["--generation", "harbor", "--stage", "all", "--hardware", "RTX_A6000"]) == 2


@pytest.fixture
def verifier(config, tmp_path, monkeypatch):
    config.generation = "local"
    config.stage = "evaluate"
    config.evaluation = "local"
    config.resume = True
    spec = {"protocol": "kernelbench-pro-harbor-v1", "evaluator_sha256": evaluator_digest(),
            "config": config.to_dict(), "reward": "correctness"}
    manifest = tmp_path / "manifest.json"
    candidate = tmp_path / "solution.py"
    candidate.write_text("candidate source")
    state = dict(correct=True, gain=0.6, calls=[])

    def baseline(cfg):
        state["calls"].append("baseline")
        assert cfg.resume is False and cfg.evaluation == "local"

    def evaluate(cfg):
        state["calls"].append("evaluate")
        assert (cfg.run_dir / kernel_name(1, 90)).read_text() == "candidate source"
        result = {"90_0": {"compiled": True, "correctness": state["correct"]}}
        for name in ("eval_results.json", "eval_results_hidden.json"):
            atomic_json(cfg.run_dir / name, result)

    def analyze(cfg):
        state["calls"].append("analyze")
        return {"bounded_gain_macro": state["gain"], "details": [{"status": "verified"}]}

    monkeypatch.setattr(engine, "record_baseline", baseline)
    monkeypatch.setattr(engine, "evaluate_run", evaluate)
    monkeypatch.setattr(paper_analysis, "analyze_paper", analyze)
    atomic_json(manifest, spec)
    return spec, manifest, candidate, tmp_path / "logs", state


@pytest.mark.parametrize("mode,correct,gain,expected", [
    ("correctness", True, 0.6, 1.0), ("correctness", False, 0.0, 0.0),
    ("bounded_gain", True, 0.6, 0.6), ("bounded_gain", True, 0.0, 0.0),
])
def test_harbor_reward_reuses_native_verification(verifier, mode, correct, gain, expected):
    spec, manifest, candidate, logs, state = verifier
    spec["reward"] = mode
    state.update(correct=correct, gain=gain)
    atomic_json(manifest, spec)
    report = harbor_verifier.verify(manifest, candidate, logs)
    assert report["reward"] == expected
    assert float((logs / "reward.txt").read_text()) == expected
    assert state["calls"] == ["baseline", "evaluate", "analyze"]


@pytest.mark.parametrize("option", ["final_repeats", "paired_timing"])
def test_harbor_correctness_cannot_bypass_required_fresh_evidence(verifier, option):
    spec, manifest, candidate, logs, _ = verifier
    spec["config"][option] = 1 if option == "final_repeats" else True
    atomic_json(manifest, spec)
    report = harbor_verifier.verify(manifest, candidate, logs)
    assert report["correctness"] is False and report["reward"] == 0


@pytest.mark.parametrize("kind", ["version", "empty", "multiple", "hidden", "mode", "exception", "nonfinite"])
def test_invalid_verification_writes_zero_instead_of_stale_reward(verifier, kind, monkeypatch):
    spec, manifest, candidate, logs, state = verifier
    logs.mkdir()
    (logs / "reward.txt").write_text("1\n")
    (logs / "reward.json").write_text('{"reward": 1}')
    if kind == "version":
        spec["evaluator_sha256"] = "old"
    elif kind == "empty":
        candidate.write_text("")
    elif kind == "multiple":
        spec["config"]["num_samples"] = 2
    elif kind == "hidden":
        spec["config"]["hidden_tests"] = False
    elif kind == "mode":
        spec["reward"] = "bad"
    elif kind == "exception":
        monkeypatch.setattr(engine, "record_baseline", lambda cfg: (_ for _ in ()).throw(RuntimeError("GPU failed")))
    elif kind == "nonfinite":
        state["gain"] = float("nan")
    atomic_json(manifest, spec)
    report = harbor_verifier.verify(manifest, candidate, logs)
    assert report["reward"] == 0 and "error" in report
    assert (logs / "reward.txt").read_text().strip() == "0.0"
    assert not (logs / "reward.json").exists()
    assert json.loads((logs / "kernelbench_pro_report.json").read_text())["reward"] == 0


def test_native_cli_dispatches_export_without_model_or_gpu(config, monkeypatch):
    called = []
    monkeypatch.setattr(harbor, "export_tasks", lambda cfg: called.append(cfg) or Path("tasks"))
    assert cli.main(["--stage", "export-harbor", "--start", "90", "--end", "90",
                     "--gpu-profile", "RTX_A6000"]) == 0
    assert called[0].start == called[0].end == 90


def test_aws_profile_is_resolved_without_writing_credentials(config, monkeypatch):
    import sys

    config.profile = "test-profile"
    session = SimpleNamespace(region_name="us-east-1", get_credentials=lambda: SimpleNamespace(
        get_frozen_credentials=lambda: SimpleNamespace(access_key="example-access", secret_key="example-secret", token="example-session")))
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=lambda **kwargs: session))
    environment = harbor.runtime_environment(config)
    assert environment["AWS_ACCESS_KEY_ID"] == "example-access"
    assert environment["AWS_REGION"] == "us-east-1"
    assert "AWS_PROFILE" not in environment
    assert not config.run_dir.exists()


def test_bounded_reward_never_counts_an_incorrect_submission(verifier):
    spec, manifest, candidate, logs, state = verifier
    spec["reward"] = "bounded_gain"
    state.update(correct=False, gain=0.6)
    atomic_json(manifest, spec)
    assert harbor_verifier.verify(manifest, candidate, logs)["reward"] == 0
