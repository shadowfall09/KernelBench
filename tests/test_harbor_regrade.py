"""Pinned Harbor task/CLI contracts with recorded-artifact process simulation."""

import sys
from pathlib import Path

import pytest

from kb_workflows import harbor_regrade
from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.config import RunConfig

pytest.importorskip("harbor")


@pytest.fixture
def source_job(tmp_path):
    source = tmp_path / "old-job"
    source.mkdir()
    rows = []
    for index in range(2):
        name = f"original_{index}"
        row = {"trial_name": name, "task_name": "kernelbench-pro/l1-p2-s0"}
        directory = source / name
        directory.mkdir()
        atomic_json(directory / "config.json", {"SIMULATED": True})
        atomic_json(directory / "result.json", row)
        atomic_json(directory / "artifacts/manifest.json", [{"source": "/app/solution.py", "destination": "artifacts/app/solution.py",
            "type": "file", "status": "ok", "service": None}, {"source": "/logs/artifacts", "destination": "artifacts/logs/artifacts",
            "type": "directory", "status": "empty", "exclude": [], "service": None}])
        (directory / "artifacts/app").mkdir()
        (directory / "artifacts/app/solution.py").write_text(f"# SIMULATED original candidate {index}\n")
        rows.append(row)
    atomic_json(source / "result.json", {"trial_results": rows})
    atomic_json(source / "config.json", {"SIMULATED": True})
    return source


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(harbor_regrade, "resolve_devices", lambda value: ["GPU-simulated"])
    return RunConfig(hardware="RTX_A6000", generation="local", evaluation="docker",
                     harbor_command=str(Path(sys.executable).with_name("harbor"))).validate()


def test_harbor_plan_preserves_all_attempts_and_exports_current_verifiers_without_execution(source_job, config, tmp_path):
    from types import SimpleNamespace

    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.trial.regrade import RegradeTrial, check_task_regradable, read_artifact_manifest

    before = harbor_regrade.snapshot(source_job)
    plan = harbor_regrade.prepare(source_job, config, output=tmp_path / "new-job")
    assert len(plan["trials"]) == 2 and len(list((tmp_path / "new-job/tasks").iterdir())) == 1
    task = next((tmp_path / "new-job/tasks").iterdir())
    assert Task.is_valid_dir(task) and check_task_regradable(task) is None
    assert read_artifact_manifest(source_job / "original_0")[0].status == "ok"
    assert RegradeTrial._artifact_coverage_problems(SimpleNamespace(agent_env_paths=EnvironmentPaths()),
        source_record_dir=source_job / "original_0", source_artifacts_dir=source_job / "original_0/artifacts",
        entries=read_artifact_manifest(source_job / "original_0"), declared_artifacts=Task(task).config.artifacts) == []
    preview = harbor_regrade.execute(plan)
    assert preview["experiments_started"] is False and preview["declared_trials"] == 2
    assert preview["command"][1:3] == ["job", "regrade"]
    assert harbor_regrade.snapshot(source_job) == before
    assert not (tmp_path / "new-job/harbor_regrade_execution.json").exists()


@pytest.mark.parametrize("damage", ["missing_trial", "missing_artifact", "bad_manifest", "symlink", "non_kernelbench"])
def test_missing_or_untrusted_source_records_cannot_shrink_the_harbor_population(source_job, config, tmp_path, damage):
    if damage == "missing_trial":
        (source_job / "original_0/result.json").unlink()
    elif damage == "missing_artifact":
        (source_job / "original_0/artifacts/app/solution.py").unlink()
    elif damage == "bad_manifest":
        atomic_json(source_job / "original_0/artifacts/manifest.json", [])
    elif damage == "symlink":
        (source_job / "linked").symlink_to(tmp_path, target_is_directory=True)
    else:
        data = read_json(source_job / "result.json")
        data["trial_results"][0]["task_name"] = "another/task"
        atomic_json(source_job / "result.json", data)
    with pytest.raises(ValueError):
        harbor_regrade.prepare(source_job, config, output=tmp_path / "new-job")


@pytest.mark.parametrize("damage", ["source", "current_verifier", "sdk", "generation_command"])
def test_stale_current_verifiers_or_changed_regrade_inputs_reject_execution(source_job, config, tmp_path, monkeypatch, damage):
    plan = harbor_regrade.prepare(source_job, config, output=tmp_path / "new-job")
    if damage == "source":
        (source_job / "original_0/artifacts/app/solution.py").write_text("# changed")
    elif damage == "current_verifier":
        path = next((tmp_path / "new-job/tasks").iterdir()) / "tests/test.sh"
        path.write_text("exit 0")
    elif damage == "sdk":
        monkeypatch.setattr(harbor_regrade, "sdk_stamp", lambda: {"version": "changed"})
    else:
        plan["command"][2] = "start"
        plan["plan_sha256"] = fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
        atomic_json(tmp_path / "new-job/harbor_regrade_plan.json", plan)
    with pytest.raises(ValueError):
        harbor_regrade.execute(plan)
    assert not (tmp_path / "new-job/harbor_regrade_execution.json").exists()


def fake_launch(plan, *, damage=None):
    seen = []
    def launch(command, **options):
        seen.append((command, options))
        root = Path(plan["output_directory"]) / "jobs/regrade"
        rows = []
        for index, old in enumerate(plan["trials"]):
            path = root / f"new_{index}" / "artifacts/app/solution.py"
            path.parent.mkdir(parents=True)
            raw = (Path(plan["source_job"]) / old["trial_name"] / "artifacts/app/solution.py").read_bytes()
            path.write_bytes(raw if damage != "candidate" else b"# replaced candidate")
            row = {"trial_name": f"new_{index}", "task_name": old["task_name"], "config": {"source_trial": {
                "action": "regrade", "type": "local", "path": str(Path(plan["source_job"]) / old["trial_name"])}}}
            if damage == "generation":
                row["config"]["source_trial"] = {}
            rows.append(row)
        atomic_json(root / "result.json", {"trial_results": rows[:1] if damage == "missing" else rows})
        class Process:
            def wait(self, **kwargs):
                return 0
            def poll(self):
                return 0
        return Process()
    return launch, seen


def test_harbor_simulation_reuses_exact_source_artifacts_and_records_a_new_attempt(source_job, config, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-enter-regrade")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-enter-regrade")
    plan = harbor_regrade.prepare(source_job, config, output=tmp_path / "new-job")
    launch, seen = fake_launch(plan)
    report = harbor_regrade.execute(plan, execute=True, launch=launch)
    assert report["status"] == "completed" and report["new_model_calls"] == 0 and report["simulated_execution"] is True
    assert len(seen) == 1 and "ANTHROPIC_API_KEY" not in seen[0][1]["env"] and "AWS_SECRET_ACCESS_KEY" not in seen[0][1]["env"]
    with pytest.raises(ValueError, match="new recorded attempt"):
        harbor_regrade.execute(plan, execute=True, launch=launch)


@pytest.mark.parametrize("damage", ["missing", "candidate", "generation"])
def test_harbor_regrade_cannot_relabel_fewer_or_different_or_newly_generated_candidates(source_job, config, tmp_path, damage):
    plan = harbor_regrade.prepare(source_job, config, output=tmp_path / "new-job")
    launch, _ = fake_launch(plan, damage=damage)
    with pytest.raises(ValueError):
        harbor_regrade.execute(plan, execute=True, launch=launch)
    record = read_json(tmp_path / "new-job/harbor_regrade_execution.json")
    assert record["status"] == "failed" and record["new_model_calls"] == 0


def test_pinned_harbor_cli_accepts_the_current_regrade_flags_without_starting_a_job():
    from harbor.cli.main import app
    from typer.main import get_command

    command = get_command(app).commands["job"].commands["regrade"]
    flags = {flag for parameter in command.params for flag in parameter.opts}
    assert {"--task-path", "--env", "--jobs-dir", "--job-name", "--n-concurrent"} <= flags
