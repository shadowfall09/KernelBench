"""Reviewable multi-model/multi-session plans; execution is always explicit."""

from __future__ import annotations

from pathlib import Path

from .budget_curves import VARY
from .common import atomic_json, evaluator_digest, fingerprint, read_json, safe_name, select_problems
from .config import RunConfig, file_options, parser
from .study_data import study_digest

PROTOCOL = "kernelbench_pro_study_plan_v1"
METHODS = {"one-shot", "feedback", "harbor"}


def make_plan(specification, *, directory, base_directory=None):
    directory = Path(directory).resolve()
    base_directory = Path(base_directory or Path.cwd()).resolve()
    document = dict(specification)
    if set(document) - {"name", "base_config", "models", "methods", "sessions", "vary", "budgets", "levels", "harbor_agent"}:
        raise ValueError("Unknown study specification fields")
    name = safe_name(document.get("name", "study"))
    options = document.get("base_config", {})
    if isinstance(options, str):
        path = Path(options)
        options = file_options(path if path.is_absolute() else base_directory / path, parser())
    if not isinstance(options, dict):
        raise ValueError("base_config must be a RunConfig object or TOML path")
    options = dict(options)
    for field in ("problem_specs", "device_calibration"):
        if options.get(field) and not Path(options[field]).is_absolute():
            options[field] = str((base_directory / options[field]).resolve())
    base = RunConfig(**options)
    models, methods = document.get("models"), document.get("methods", ["one-shot", "feedback", "harbor"])
    budgets, sessions, vary = document.get("budgets"), document.get("sessions", 3), document.get("vary", "wall")
    levels = document.get("levels", [base.level])
    if (not isinstance(models, list) or not models or any(not isinstance(model, str) or not model for model in models)
            or len(set(models)) != len(models) or not isinstance(methods, list) or not methods
            or set(methods) - METHODS or len(set(methods)) != len(methods)
            or type(sessions) is not int or sessions < 1 or vary not in VARY
            or not isinstance(budgets, list) or not budgets or len(set(budgets)) != len(budgets)
            or any((type(value) is not int if vary != "device_lease_seconds" else type(value) not in (int, float))
                   or not 0 < value < float("inf") for value in budgets)
            or not isinstance(levels, list) or not levels or len(set(levels)) != len(levels)):
        raise ValueError("Invalid models, methods, levels, sessions, or positive integer budget grid")
    if vary != "wall" and "harbor" in methods:
        raise ValueError("Harbor does not certify native candidate/token/device reservations; use a wall-time grid")
    jobs, skipped = [], []
    for mi, model in enumerate(models):
        for method in methods:
            for bi, budget in enumerate(budgets):
                if vary == "candidates" and method == "one-shot" and budget != 1:
                    skipped.append({"model": model, "method": method, "budget": budget, "reason": "one-shot has exactly one candidate"})
                    continue
                for si in range(sessions):
                    for level in levels:
                        config = RunConfig(**base.to_dict())
                        config.level, config.model = level, model
                        config.run_name = f"{name}_m{mi}_{method}_b{bi}_s{si}_l{level}"
                        config.runs_dir = str(directory / "runs")
                        config.stage, config.dry_run, config.resume = "all", False, True
                        field = {"wall": "agent_timeout", "candidates": "optimization_rounds", "total_tokens": "agent_token_limit",
                                 "output_tokens": "agent_output_token_limit", "device_lease_seconds": "agent_device_seconds"}[vary]
                        setattr(config, field, budget)
                        if method == "harbor":
                            config.generation, config.generation_mode = "harbor", "external"
                            config.harbor_agent = document.get("harbor_agent", base.harbor_agent)
                        else:
                            config.generation_mode = method
                            if config.generation == "harbor":
                                config.generation = "local"
                            if method == "one-shot":
                                config.optimization_rounds, config.agent_profile = 1, False
                        config.validate()
                        config.require_gpu_target()
                        selected = select_problems(level, config.start, config.end, suite=config.suite, suite_role=config.suite_role,
                                                   track=config.task_track)
                        jobs.append({"config": config.to_dict(), "session": f"session_{si:03}", "method": method,
                                     "problem_ids": list(selected), "declared_budget": budget,
                                     "path": str(config.run_dir), "config_sha256": fingerprint(config.to_dict())})
    if not jobs:
        raise ValueError("The study grid has no executable protocol points")
    plan = {"protocol": PROTOCOL, "name": name, "vary": vary, "jobs": jobs, "skipped_points": skipped,
            "plan_implementation_sha256": study_digest(),
            "evaluator_sha256": evaluator_digest(), "experiments_started": False,
            "session_independence": "separate_declared_attempts; provider_randomness_not_controlled",
            "gpu_time_or_token_hard_limits_certified": False}
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def write_plan(plan, directory):
    directory = Path(directory)
    atomic_json(directory / "plan.json", plan)
    atomic_json(directory / "study.json", {"runs": [{"path": job["path"], "session": job["session"]} for job in plan["jobs"]]})
    return directory / "plan.json"


def execute_plan(path, *, execute=False, runner=None, keep_going=False):
    path = Path(path)
    plan = read_json(path)
    if (not isinstance(plan, dict) or plan.get("protocol") != PROTOCOL or plan.get("plan_sha256") !=
            fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
            or plan.get("evaluator_sha256") != evaluator_digest() or plan.get("plan_implementation_sha256") != study_digest()):
        raise ValueError("Plan is malformed, changed, or stale; regenerate and review it")
    if not execute:
        return {"experiments_started": False, "job_count": len(plan["jobs"]), "plan_sha256": plan["plan_sha256"]}
    if runner is None:
        from .cli import main as runner
    results = []
    for index, job in enumerate(plan["jobs"]):
        if fingerprint(job["config"]) != job["config_sha256"]:
            raise ValueError("Study job config changed")
        config_file = path.parent / "configs" / f"{index:05}.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        # The native CLI already understands exact RunConfig fields via TOML.
        import json

        gpu_fields = {"hardware", "gpu", "gpu_arch", "gpus"}
        lines = ["[run]"]
        for key, value in job["config"].items():
            if key in gpu_fields or value is None:
                continue
            lines.append(f"{key} = {json.dumps(value)}")
        config = RunConfig(**job["config"])
        # TOML's explicit GPU mapping uses the maintained external schema.
        lines += ["", "[gpu]", f"hardware = {json.dumps(config.hardware)}", f"architecture = {json.dumps(config.gpu_arch)}",
                  f"devices = {json.dumps([int(device) for device in config.gpus.split(',')])}"]
        if config.gpu:
            lines.append(f"modal_type = {json.dumps(config.gpu)}")
        config_file.write_text("\n".join(lines) + "\n")
        code = runner(["--config", str(config_file)])
        results.append({"path": job["path"], "session": job["session"], "exit_code": code, "config_sha256": job["config_sha256"]})
        atomic_json(path.parent / "execution.json", {"plan_sha256": plan["plan_sha256"], "experiments_started": True, "jobs": results})
        if code and not keep_going:
            break
    return {"plan_sha256": plan["plan_sha256"], "experiments_started": True, "jobs": results}
