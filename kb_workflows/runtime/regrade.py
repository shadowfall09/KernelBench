"""Frozen-artifact regrading; no agent/model runs and no execution by default."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from ..benchmarks.reference_population import select_run_problems
from ..common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name, read_json, safe_name
from ..config import RunConfig

PROTOCOL = "kernelbench_pro_artifact_regrade_v1"
EXTERNAL = ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file")
HISTORY = ("run_config.json", "reference_population.json", "precision_calibration_manifest.json", "paper_analysis.json", "generation_budget.json", "generation_attempts.json", "generation_summary.json",
           "eval_results.json", "eval_results_hidden.json", "final_manifest.json", "regrade_manifest.json", "baseline_snapshot.json")
MAX_SOURCE_BYTES = 8 * 1024**2


def history_name(name):
    return name in HISTORY or bool(re.fullmatch(r"work/p[0-9]+_s[0-9]+/(trajectory|agent_result|development_history|interactive_state)\.json", name))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def population(source, saved):
    samples = saved.get("num_samples", 1)
    if type(samples) is not int or samples < 1:
        raise ValueError("Source run has no positive declared sample count")
    analysis = read_json(source / "paper_analysis.json", {})
    if isinstance(analysis.get("problem_ids"), list) and analysis["problem_ids"]:
        ids = analysis["problem_ids"]
        if analysis.get("declared_samples") != samples:
            raise ValueError("Source population and saved sample count disagree")
    else:
        budget = read_json(source / "generation_budget.json", {})
        slots = budget.get("slots") or read_json(source / "eval_results.json", {})
        if not isinstance(slots, dict) or not slots:
            raise ValueError("Source run lacks a recorded complete population; source files alone cannot define its denominator")
        pairs = []
        for key in slots:
            match = re.fullmatch(r"([1-9][0-9]*)_([0-9]+)", key)
            if not match:
                raise ValueError("Invalid recorded source slot")
            pairs.append(tuple(map(int, match.groups())))
        ids = sorted({pid for pid, _ in pairs})
        if set(pairs) != {(pid, sid) for pid in ids for sid in range(samples)}:
            raise ValueError("Source run lost declared sample slots")
    if (any(type(pid) is not int or pid < 1 for pid in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Source task population is invalid or duplicated")
    return sorted(ids), samples


def prepare(source, verifier, *, output):
    verifier = replace(verifier).validate()
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Regrading requires an existing source run and a new, disjoint output directory")
    saved = read_json(source / "run_config.json")
    if not isinstance(saved, dict) or type(saved.get("level")) is not int or saved["level"] not in {1, 2, 3, 5}:
        raise ValueError("Source run configuration is missing")
    history_paths = {name: source / name for name in HISTORY if (source / name).is_file()}
    for filename in ("trajectory.json", "agent_result.json", "development_history.json", "interactive_state.json"):
        for path in source.glob("work/p*_s*/" + filename):
            name = str(path.relative_to(source))
            if history_name(name):
                history_paths[name] = path
    for name, path in history_paths.items():
        if path.is_symlink() or any((source / Path(*Path(name).parts[:index])).is_symlink() for index in range(1, len(Path(name).parts))):
            raise ValueError("Source provenance records cannot contain symlinks")
    history_hashes = {name: digest(path) for name, path in history_paths.items()}
    ids, samples = population(source, saved)
    workload = {name: saved[name] for name in ("level", "suite", "suite_role", "task_track") if name in saved}
    config = replace(verifier, **workload, start=saved.get("start", min(ids)) if verifier.reference_population else min(ids),
        end=saved.get("end", max(ids)) if verifier.reference_population else max(ids), num_samples=samples,
        runs_dir=str(output.parent), run_name=output.name, stage="evaluate", generation="local", resume=False,
        model="", small_model="", profile="", bedrock=False, agent_token_limit=0, agent_output_token_limit=0,
        agent_device_seconds=0, baseline=verifier.baseline + "-regrade-" + fingerprint({"output": str(output)})[:12],
        regrade_manifest=str(output / "regrade_manifest.json")).validate()
    config.require_gpu_target()
    selected = select_run_problems(config)
    if list(selected) != ids:
        raise ValueError("Current selection differs from the recorded source population; no silent task omission or addition")
    candidates = {}
    for pid in ids:
        for sid in range(samples):
            path = source / kernel_name(config.level, pid, sid)
            if path.is_symlink():
                raise ValueError("A source candidate is a symlink")
            status = "present" if path.is_file() and path.stat().st_size else "missing"
            if status == "present" and path.stat().st_size > MAX_SOURCE_BYTES:
                status = "oversized"
            if status == "present":
                try:
                    path.read_text()
                except UnicodeError:
                    status = "invalid_utf8"
            candidates[f"{pid}_{sid}"] = {"problem_id": pid, "sample_id": sid, "filename": path.name,
                "status": status, "source_sha256": digest(path) if path.is_file() else None}
    output.mkdir(parents=True, mode=0o700)
    history = output / "source_history"
    history.mkdir()
    files = {}
    for name, path in history_paths.items():
        if path.is_symlink():
            raise ValueError("Source provenance records cannot be symlinks")
        if path.is_file():
            target = history / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            target.chmod(0o444)
            files[name] = digest(target)
            if files[name] != history_hashes[name] or digest(path) != history_hashes[name]:
                raise ValueError("Historical evidence changed while being frozen")
    if saved.get("hardware") and saved.get("baseline"):
        original_baseline = ROOT / "results/timing" / safe_name(saved["hardware"]) / (safe_name(saved["baseline"]) + ".json")
        if original_baseline.is_file():
            atomic_json(history / "baseline_snapshot.json", read_json(original_baseline))
            (history / "baseline_snapshot.json").chmod(0o444)
            files["baseline_snapshot.json"] = digest(history / "baseline_snapshot.json")
    external = {}
    for name in EXTERNAL:
        filename = getattr(config, name)
        if filename:
            target = output / "verifier_inputs" / f"{name}.json"
            data = read_json(Path(filename))
            atomic_json(target, data)
            target.chmod(0o444)
            setattr(config, name, str(target))
            external[name] = digest(target)
    from ..benchmarks.reference_population import freeze_run

    freeze_run(config)
    artifacts = output / "frozen_candidates"
    artifacts.mkdir()
    for row in candidates.values():
        if row["status"] != "present":
            continue
        raw = (source / row["filename"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            raise ValueError("Source candidate changed while being frozen")
        frozen = artifacts / (row["source_sha256"] + ".py")
        if not frozen.exists():
            frozen.write_bytes(raw)
            frozen.chmod(0o444)
        path = output / row["filename"]
        path.write_bytes(raw)
        path.chmod(0o444)
    plan = {"protocol": PROTOCOL, "source_run": str(source), "output_directory": str(output), "config": config.to_dict(),
        "source_population": ids, "declared_samples": samples, "slots": candidates, "history_files": files,
        "verifier_inputs": external, "evaluator_sha256": evaluator_digest(),
        "references": {str(pid): {"filename": path.name, "sha256": digest(path)} for pid, path in selected.items()},
        "hidden_sources": {str(pid): digest(ROOT / "hidden_tests" / f"level{config.level}" / f"{pid}_hidden.py")
                           for pid in ids} if config.hidden_tests else {},
        "source_generation_protocol": read_json(history / "generation_budget.json", {}).get("protocol"),
        "baseline_template": verifier.baseline,
        "new_model_calls": 0, "historical_generation_cost_recertified": False,
        "score_change_scope": "updated_verifier_protocol; no_causal_performance_or_same_cost_claim", "experiments_started": False}
    plan["plan_sha256"] = fingerprint(plan)
    atomic_json(output / "regrade_manifest.json", plan)
    atomic_json(output / "run_config.json", config.to_dict())
    return plan


def validate(plan):
    if (not isinstance(plan, dict) or plan.get("protocol") != PROTOCOL or plan.get("evaluator_sha256") != evaluator_digest()
            or plan.get("plan_sha256") != fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise ValueError("Artifact regrade plan is missing, changed or stale")
    config = RunConfig(**plan["config"]).validate()
    root = config.run_dir
    if (str(root) != plan["output_directory"] or str(root / "regrade_manifest.json") != config.regrade_manifest
            or read_json(root / "run_config.json") != config.to_dict()
            or read_json(root / "regrade_manifest.json") != plan):
        raise ValueError("Regrade plan/configuration moved or changed")
    selected = select_run_problems(config)
    if list(selected) != plan["source_population"] or config.num_samples != plan["declared_samples"]:
        raise ValueError("Regrade denominator changed")
    if set(plan["slots"]) != {f"{pid}_{sid}" for pid in selected for sid in range(config.num_samples)}:
        raise ValueError("Regrade lost declared slots")
    for pid, path in selected.items():
        if plan["references"].get(str(pid)) != {"filename": path.name, "sha256": digest(path)}:
            raise ValueError("Regrade reference changed")
        if config.hidden_tests and plan["hidden_sources"].get(str(pid)) != digest(ROOT / "hidden_tests" / f"level{config.level}" / f"{pid}_hidden.py"):
            raise ValueError("Regrade hidden source changed")
    for name, checksum in plan["history_files"].items():
        path = root / "source_history" / name
        if not history_name(name) or path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError("Frozen historical provenance changed")
    for name, checksum in plan["verifier_inputs"].items():
        path = Path(getattr(config, name))
        if name not in EXTERNAL or path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError("Frozen verifier input changed")
    for key, row in plan["slots"].items():
        pid, sid = map(int, key.split("_"))
        if row["filename"] != kernel_name(config.level, pid, sid) or row["problem_id"] != pid or row["sample_id"] != sid:
            raise ValueError("Regrade source identity changed")
        path = root / row["filename"]
        if row["status"] == "present":
            if not isinstance(row.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
                raise ValueError("Invalid frozen regrade source checksum")
            frozen = root / "frozen_candidates" / (row["source_sha256"] + ".py")
            if any(p.is_symlink() or not p.is_file() or digest(p) != row["source_sha256"] for p in (path, frozen)):
                raise ValueError("Frozen regrade candidate changed")
        elif row["status"] not in {"missing", "oversized", "invalid_utf8"} or path.exists() or path.is_symlink():
            raise ValueError("A missing/rejected source cannot be silently added to the frozen run")
    return config


def protocol(config):
    plan = read_json(Path(config.regrade_manifest))
    validate(plan)
    matches_config(config, plan)
    return {"version": PROTOCOL, "mode": "artifact-regrade", "scaffold": "current_native_verifier",
        "source_plan_sha256": plan["plan_sha256"], "source_generation_protocol": plan["source_generation_protocol"],
        "new_model_calls": 0, "selection": "all_recorded_slots_including_missing", "historical_cost_recertified": False}


def matches_config(config, plan):
    supplied = config.to_dict()
    if supplied["stage"] not in {"baseline", "evaluate", "analyze"}:
        raise ValueError("Regrading cannot start agent generation")
    supplied["stage"] = plan["config"]["stage"]
    if supplied != plan["config"]:
        raise ValueError("Current configuration differs from the frozen regrade protocol")


def measurement_files(config):
    return ["eval_results.json"] + (["eval_results_hidden.json"] if config.hidden_tests else []) + (
        ["final_manifest.json", "final_results.json"] if config.final_repeats else []) + (["paired_results.json"] if config.paired_timing else [])


def baseline_entries(config):
    data = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {}).get(f"level{config.level}", {})
    return {key: fingerprint(data.get(row["filename"], {})) for key, row in read_json(Path(config.regrade_manifest))["references"].items()}


def issue(config):
    if not config.regrade_manifest:
        return None
    try:
        plan = read_json(Path(config.regrade_manifest))
        validate(plan)
        matches_config(config, plan)
        value = read_json(config.run_dir / "regrade_measurements.json", {})
        if (value.get("protocol") != PROTOCOL or value.get("plan_sha256") != plan["plan_sha256"]
                or value.get("simulated_execution") is not False
                or value.get("receipt_sha256") != fingerprint({key: item for key, item in value.items() if key != "receipt_sha256"})
                or value.get("files") != {name: digest(config.run_dir / name) for name in measurement_files(config)}
                or value.get("baseline_entries") != baseline_entries(config)):
            return "regrade_measurement_proof_missing_changed_or_simulated"
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return "regrade_plan_or_artifacts_missing_changed_or_stale"
    return None


def execute(plan, *, execute=False, dispatch=None):
    config = validate(plan)
    if not execute:
        return {"experiments_started": False, "declared_slots": len(plan["slots"]), "plan_sha256": plan["plan_sha256"]}
    journal = config.run_dir / "regrade_execution.json"
    if journal.exists():
        raise ValueError("A regrade execution already exists; make a new plan rather than spend an unrecorded retry")
    injected = dispatch is not None
    report = {"protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"], "experiments_started": True,
        "simulated_execution": dispatch is not None, "new_model_calls": 0, "stages": [], "status": "running"}
    atomic_json(journal, report)
    all_started = time.monotonic()
    if dispatch is None:
        from ..agents.generation import build_docker_image
        from ..engine import analyze_run, evaluate_run, record_baseline

        functions = {"image_build": build_docker_image, "baseline": record_baseline, "evaluate": evaluate_run, "analyze": analyze_run}
        def dispatch(phase, options):
            return functions[phase](options)
    context = nullcontext()
    if not injected and config.evaluation == "modal":
        from .modal_backend import modal_session

        context = modal_session(config)
    try:
        with context:
            phases = (["image_build"] if config.evaluation == "docker" and config.build_image else []) + ["baseline", "evaluate"]
            for name in phases:
                validate(plan)
                started = time.monotonic()
                completed = False
                try:
                    dispatch(name, config)
                    completed = True
                finally:
                    report["stages"].append({"name": name, "request_wall_seconds": time.monotonic() - started,
                                             "status": "completed" if completed else "failed"})
                    atomic_json(journal, report)
            validate(plan)
            measured = {"protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"], "simulated_execution": injected,
                "files": {name: digest(config.run_dir / name) for name in measurement_files(config)},
                "baseline_entries": baseline_entries(config), "new_model_calls": 0}
            measured["receipt_sha256"] = fingerprint(measured)
            atomic_json(config.run_dir / "regrade_measurements.json", measured)
            started = time.monotonic()
            completed = False
            try:
                dispatch("analyze", config)
                completed = True
            finally:
                report["stages"].append({"name": "analyze", "request_wall_seconds": time.monotonic() - started,
                                         "status": "completed" if completed else "failed"})
        validate(plan)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["total_request_wall_seconds"] = time.monotonic() - all_started
        report["result_files"] = {name: digest(config.run_dir / name) for name in (
            "eval_results.json", "eval_results_hidden.json", "final_manifest.json", "final_results.json", "paired_results.json",
            "paper_analysis.json", "eval_results_scored.json") if (config.run_dir / name).is_file()}
        report["receipt_sha256"] = fingerprint({key: value for key, value in report.items() if key != "receipt_sha256"})
        atomic_json(journal, report)
    return report
