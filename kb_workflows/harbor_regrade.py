"""Prepare current Harbor verifiers and regrade recorded artifacts only."""

from __future__ import annotations

import importlib.metadata
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import replace
from pathlib import Path, PurePosixPath

from .common import atomic_json, evaluator_digest, fingerprint, read_json, safe_name
from .config import RunConfig
from .harbor import task_name, write_task
from .regrade import MAX_SOURCE_BYTES, digest
from .transports import resolve_devices, terminate

PROTOCOL = "kernelbench_pro_harbor_regrade_v1"
HARBOR_VERSION = "0.24.0"


def snapshot(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Harbor recorded jobs/tasks must not contain symlinks")
        if path.is_file():
            result[str(path.relative_to(directory))] = digest(path)
    return result


def cohort(source):
    source = Path(source)
    if not (source / "config.json").is_file():
        raise ValueError("Harbor source job config.json is missing")
    result = read_json(source / "result.json", {})
    rows = result.get("trial_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Harbor source lacks its completed declared trial population")
    trials, names = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Harbor source trial record is malformed")
        name, task = safe_name(row["trial_name"]), row.get("task_name", "")
        match = re.fullmatch(r"kernelbench-pro/l([1235])-p([1-9][0-9]*)-s([0-9]+)", task)
        if name in names or not match:
            raise ValueError("Harbor regrade requires unique recorded KernelBench Pro trials")
        names.add(name)
        level, pid, sid = map(int, match.groups())
        directory = source / name
        saved = read_json(directory / "result.json", {})
        if saved.get("task_name") != task or saved.get("trial_name") != name or not (directory / "config.json").is_file():
            raise ValueError("Harbor trial and aggregate records disagree")
        if saved.get("step_results"):
            raise ValueError("KernelBench Pro regrades require its single-artifact, single-step task format")
        manifest = read_json(directory / "artifacts/manifest.json")
        if not isinstance(manifest, list) or any(not isinstance(item, dict) or not isinstance(item.get("source"), str)
                or not isinstance(item.get("destination"), str) or not PurePosixPath(item["source"]).is_absolute()
                or PurePosixPath(item["destination"]).is_absolute() or ".." in PurePosixPath(item["source"]).parts
                or ".." in PurePosixPath(item["destination"]).parts for item in manifest):
            raise ValueError("Harbor artifact manifest has invalid or escaping paths")
        declarations = {item["source"].rstrip("/"): item for item in manifest if item.get("service") in (None, "main")}
        solution, convention = declarations.get("/app/solution.py", {}), declarations.get("/logs/artifacts", {})
        if (solution.get("destination") != "artifacts/app/solution.py" or solution.get("status") != "ok" or solution.get("type") != "file"
                or convention.get("destination") != "artifacts/logs/artifacts" or convention.get("status") not in {"ok", "empty"}
                or convention.get("type") != "directory" or convention.get("exclude") not in (None, [])):
            raise ValueError("Harbor trial lacks successful artifact-collection evidence; do not shrink the cohort")
        if convention["status"] == "ok" and not (directory / "artifacts/logs/artifacts").is_dir():
            raise ValueError("Harbor convention artifact bytes are missing")
        artifact = directory / "artifacts/app/solution.py"
        if artifact.is_symlink() or not artifact.is_file() or not 0 < artifact.stat().st_size <= MAX_SOURCE_BYTES:
            raise ValueError("Harbor recorded solution is absent, linked or oversized")
        artifact.read_text()
        trials.append({"trial_name": name, "task_name": task, "level": level, "problem_id": pid, "sample_id": sid,
                       "artifact_sha256": digest(artifact)})
    actual = {path.name for path in source.iterdir() if path.is_dir() and (path / "result.json").is_file()}
    if actual != names:
        raise ValueError("Harbor job aggregate omitted or added recorded trials")
    return trials


def sdk_stamp():
    try:
        package = importlib.metadata.distribution("harbor")
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "source_files": {}}
    return {"version": package.version, "source_files": {name: digest(package.locate_file(name)) for name in (
        "harbor/cli/main.py", "harbor/cli/jobs.py", "harbor/job.py", "harbor/trial/trial.py", "harbor/trial/regrade.py",
        "harbor/agents/nop.py", "harbor/models/task/verifier_mode.py")}}


def invocation(executable, source, output):
    return [executable, "job", "regrade", str(source), "--task-path", str(output / "tasks"), "--env", "docker",
            "--jobs-dir", str(output / "jobs"), "--job-name", "regrade", "--n-concurrent", "1"]


def time_limit(tasks, trials):
    total = 60.0
    for row in trials:
        path = tasks / task_name(row["level"], row["problem_id"], row["sample_id"]) / "task.toml"
        value = tomllib.loads(path.read_text())
        if value["task"]["name"] != row["task_name"]:
            raise ValueError("Regrade verifier task name changed")
        total += value["verifier"]["timeout_sec"] + value["verifier"]["environment"]["build_timeout_sec"]
    return total


def prepare(source, verifier, *, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Harbor regrading requires an existing source and a new disjoint output directory")
    trials, source_files = cohort(source), snapshot(source)
    config = replace(verifier, generation="local", stage="export-harbor", model="", small_model="", profile="", bedrock=False,
                     regrade_manifest="").validate()
    config.require_gpu_target()
    if config.harbor_environment != "docker":
        raise ValueError("This local recorded-job regrader requires the Harbor Docker provider")
    output.mkdir(parents=True, mode=0o700)
    tasks = output / "tasks"
    tasks.mkdir()
    selected = {}
    devices = resolve_devices(config.gpus)
    for trial in trials:
        selected.setdefault(trial["task_name"], trial)
    for index, trial in enumerate(selected.values()):
        child = replace(config, level=trial["level"], start=trial["problem_id"], end=trial["problem_id"], num_samples=1,
                        suite="production41" if trial["level"] == 5 else "full", suite_role="scored",
                        task_track=config.task_track if trial["level"] == 5 else "all").validate()
        write_task(tasks / task_name(trial["level"], trial["problem_id"], trial["sample_id"]), child,
                   trial["problem_id"], trial["sample_id"], device=devices[index % len(devices)])
    executable = shutil.which(config.harbor_command)
    command = invocation(str(Path(executable).resolve()) if executable else config.harbor_command, source, output)
    plan = {"protocol": PROTOCOL, "source_job": str(source), "output_directory": str(output), "trials": trials,
        "source_files": source_files, "task_files": snapshot(tasks), "evaluator_sha256": evaluator_digest(),
        "sdk": sdk_stamp(), "command": command, "executable_sha256": digest(executable) if executable else None,
        "config": config.to_dict(), "experiments_started": False, "new_model_calls": 0,
        "timeout_seconds": time_limit(tasks, trials),
        "scope": "upstream_Harbor_regrade_restores_artifacts_and_uses_NopAgent; no_agent_generation_or_historical_cost_recertification"}
    plan["plan_sha256"] = fingerprint(plan)
    atomic_json(output / "harbor_regrade_plan.json", plan)
    return plan


def validate(plan):
    if (not isinstance(plan, dict) or plan.get("protocol") != PROTOCOL or plan.get("evaluator_sha256") != evaluator_digest()
            or plan.get("plan_sha256") != fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise ValueError("Harbor regrade plan is changed or stale")
    if RunConfig(**plan["config"]).validate().to_dict() != plan["config"]:
        raise ValueError("Harbor regrade configuration is not the frozen resolved protocol")
    root, source = Path(plan["output_directory"]), Path(plan["source_job"])
    if plan["command"] != invocation(plan["command"][0], source, root):
        raise ValueError("Harbor regrading cannot replace artifact verification with agent execution")
    if plan.get("timeout_seconds") != time_limit(root / "tasks", plan["trials"]):
        raise ValueError("Harbor regrade request budget differs from its frozen verifier tasks")
    if (read_json(root / "harbor_regrade_plan.json") != plan or snapshot(source) != plan["source_files"]
            or cohort(source) != plan["trials"] or snapshot(root / "tasks") != plan["task_files"] or sdk_stamp() != plan["sdk"]):
        raise ValueError("Recorded job, current tasks or Harbor SDK changed after the regrade plan was frozen")
    return root


def environment():
    import os

    keys = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
            "XDG_RUNTIME_DIR", "TMPDIR", "CUDA_VISIBLE_DEVICES")
    return {key: os.environ[key] for key in keys if key in os.environ}


def execute(plan, *, execute=False, launch=None):
    root = validate(plan)
    if not execute:
        return {"experiments_started": False, "declared_trials": len(plan["trials"]), "command": plan["command"],
                "timeout_seconds": plan["timeout_seconds"]}
    if plan["sdk"]["version"] != HARBOR_VERSION:
        raise ValueError("Install the pinned Harbor 0.24.0 SDK before executing a regrade")
    executable = Path(plan["command"][0])
    if not executable.is_file() or digest(executable) != plan["executable_sha256"]:
        raise ValueError("Harbor executable is unavailable or changed")
    journal = root / "harbor_regrade_execution.json"
    if journal.exists() or (root / "jobs/regrade").exists():
        raise ValueError("Harbor regrading needs a new recorded attempt; completed/interrupted attempts are not silently retried")
    timeout = plan["timeout_seconds"]
    report = {"protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"], "new_model_calls": 0, "status": "running",
              "experiments_started": True, "simulated_execution": launch is not None, "timeout_seconds": timeout}
    atomic_json(journal, report)
    started = time.monotonic()
    try:
        with (root / "harbor_regrade.log").open("w") as log:
            process = (launch or subprocess.Popen)(plan["command"], env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            finally:
                terminate(process)
        validate(plan)
        if code != 0:
            raise RuntimeError("Harbor regrade returned a nonzero exit status")
        output = read_json(root / "jobs/regrade/result.json", {})
        rows = output.get("trial_results", [])
        if sorted(row["task_name"] for row in rows) != sorted(row["task_name"] for row in plan["trials"]):
            raise ValueError("Harbor regrade results lost or added source trials")
        expected, observed = {str((Path(plan["source_job"]) / row["trial_name"]).resolve()): row for row in plan["trials"]}, set()
        for row in rows:
            origin = row.get("config", {}).get("source_trial", {})
            path = str(Path(origin.get("path", "")).resolve())
            if (origin.get("action") != "regrade" or origin.get("type") != "local" or path not in expected
                    or path in observed or row["task_name"] != expected[path]["task_name"]):
                raise ValueError("Harbor output does not preserve the complete source-trial derivation")
            observed.add(path)
            artifact = root / "jobs/regrade" / safe_name(row["trial_name"]) / "artifacts/app/solution.py"
            if artifact.is_symlink() or not artifact.is_file() or digest(artifact) != expected[path]["artifact_sha256"]:
                raise ValueError("Harbor regrade changed the frozen candidate artifact")
        report.update(status="completed", exit_code=code, result_files=snapshot(root / "jobs/regrade"))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["request_wall_seconds"] = time.monotonic() - started
        report["receipt_sha256"] = fingerprint({key: value for key, value in report.items() if key != "receipt_sha256"})
        atomic_json(journal, report)
    return report
