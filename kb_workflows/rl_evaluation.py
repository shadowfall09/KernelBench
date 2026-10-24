"""Explicit regrading plans and held-out native workflow jobs for trained checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .common import atomic_json, evaluator_digest, fingerprint, read_json, safe_name
from .local_model import checkpoint_digest
from .rl_plan import dataset_rows, task_config, validate_plan
from .rl_rewards import evaluate_group, validate_reward

PROTOCOL = "kernelbench_pro_rl_regrade_plan_v1"


def make_regrade_plan(training_plan, specification, *, directory, base_directory=None):
    validate_plan(training_plan)
    if set(specification) - {"cases", "repeats", "split"}:
        raise ValueError("Unknown reward regrade specification fields")
    split, repeats, cases = specification.get("split", "heldout"), specification.get("repeats", 30), specification.get("cases")
    if split not in {"train", "heldout"} or type(repeats) is not int or repeats < 1 or not isinstance(cases, list) or not cases:
        raise ValueError("Explicit cases, positive repetitions and a named split are required")
    root, rows, names = Path(base_directory or Path.cwd()).resolve(), [], set()
    for case in cases:
        if set(case) - {"case_id", "task_key", "source", "expected_invalid", "expected_block_checks", "rationale",
                        "independent_runtime_ms", "anchor_review"}:
            raise ValueError("Unknown reward case fields")
        name = safe_name(case["case_id"])
        if name in names or case["task_key"] not in training_plan["splits"][split]:
            raise ValueError("Duplicate case or source task outside the selected split")
        names.add(name)
        source = Path(case["source"]).expanduser()
        source = (source if source.is_absolute() else root / source).resolve()
        if not source.is_file():
            raise ValueError("A regrade candidate source is missing")
        rows.append({**case, "case_id": name, "source": str(source), "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    result = {"protocol": PROTOCOL, "training_plan_sha256": training_plan["plan_sha256"], "cases": rows,
              "split": split, "repeats": repeats, "output_directory": str(Path(directory).resolve()),
              "evaluator_sha256": evaluator_digest(), "experiments_started": False}
    result["plan_sha256"] = fingerprint(result)
    return result


def execute_regrade(training_plan, plan, *, execute=False, resume=False, group_evaluator=None):
    validate_plan(training_plan)
    if (plan.get("protocol") != PROTOCOL or plan.get("plan_sha256") != fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
            or plan.get("training_plan_sha256") != training_plan["plan_sha256"] or plan.get("evaluator_sha256") != evaluator_digest()):
        raise ValueError("Reward regrade plan is stale or changed")
    for case in plan["cases"]:
        if hashlib.sha256(Path(case["source"]).read_bytes()).hexdigest() != case["source_file_sha256"]:
            raise ValueError("Predeclared regrade source changed")
    if not execute:
        return {"experiments_started": False, "declared_cases": len(plan["cases"]), "repeats": plan["repeats"]}
    root = Path(plan["output_directory"])
    grouped = {}
    for case in plan["cases"]:
        grouped.setdefault(case["task_key"], []).append(case)
    result = {"plan_sha256": plan["plan_sha256"], "experiments_started": True, "groups": [], "cases": {case["case_id"]: [] for case in plan["cases"]}}
    for repeat in range(plan["repeats"]):
        for key, cases in grouped.items():
            directory = root / f"repeat_{repeat:03}" / key.replace("/", "_")
            target = directory / "reward.json"
            source_record = {"regrade_plan_sha256": plan["plan_sha256"], "cases": [
                {"case_id": case["case_id"], "sha256": case["source_file_sha256"]} for case in cases]}
            if resume and target.is_file():
                receipt = validate_reward(target)
                if (receipt["plan_sha256"] != training_plan["plan_sha256"] or receipt["task_key"] != key
                        or receipt["split"] != plan["split"] or receipt.get("purpose") != "audit"):
                    raise ValueError("Regrade receipt belongs to another task/plan")
                # Raw source-file digests also distinguish malformed sources whose parsed checksum is null.
                declared = read_json(directory / "regrade_sources.json", {})
                if declared != source_record:
                    raise ValueError("Regrade sources differ from the completed evidence")
            else:
                receipt = (group_evaluator or evaluate_group)(training_plan, key, [Path(case["source"]).read_text() for case in cases],
                    directory=directory, split=plan["split"], purpose="audit")
                atomic_json(directory / "regrade_sources.json", source_record)
            result["groups"].append({"task_key": key, "repeat": repeat, "receipt": str(target), "sha256": receipt["receipt_sha256"]})
            for sid, case in enumerate(cases):
                result["cases"][case["case_id"]].append({"receipt": str(target), "sample_id": sid})
            atomic_json(root / "regrade_results.json", result)
    audit = {"attacks": [], "reliability": {"kernels": []}}
    for case in plan["cases"]:
        samples = result["cases"][case["case_id"]]
        if case.get("expected_invalid") is True:
            audit["attacks"].append({**samples[0], "attack": case["case_id"], "expected_invalid": True,
                "expected_block_checks": case.get("expected_block_checks"), "rationale": case.get("rationale")})
        else:
            audit["reliability"]["kernels"].append({"kernel": case["case_id"], "repetitions": samples,
                "independent_runtime_ms": case.get("independent_runtime_ms"), "anchor_review": case.get("anchor_review")})
    if not audit["attacks"]:
        audit.pop("attacks")
    if not audit["reliability"]["kernels"]:
        audit.pop("reliability")
    atomic_json(root / "audit_input.json", audit)
    return result


def heldout_plan(training_plan, checkpoints, *, directory, sessions=3, samples=1):
    """Use ordinary one-shot/feedback final evaluation, without optimizer access to held-out tasks."""
    from .reference_population import receipt
    from .study_data import study_digest
    from .study_plan import PROTOCOL as STUDY_PROTOCOL

    validate_plan(training_plan)
    if (not isinstance(checkpoints, dict) or not checkpoints or type(sessions) is not int or sessions < 1
            or type(samples) is not int or samples < 1):
        raise ValueError("Held-out evaluation requires named local checkpoints and positive sessions/samples")
    root, jobs, identities = Path(directory).resolve(), [], {}
    for label, checkpoint in checkpoints.items():
        safe_name(label)
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Held-out checkpoint directory is missing")
        identities[label] = {"path": str(path), "sha256": checkpoint_digest(path)}
        for session in range(sessions):
            for task in dataset_rows(training_plan, "heldout"):
                key = task["task_key"]
                config = task_config(training_plan, key, directory=root / "runs" / f"{label}_s{session}_{key.replace('/', '_')}", samples=samples)
                config.stage, config.generation, config.agent_backend = "all", "local", "local-transformers"
                config.model, config.agent_temperature, config.resume = str(path), training_plan["training"]["temperature"], True
                config.validate()
                jobs.append({"config": config.to_dict(), "session": f"session_{session:03}", "method": config.generation_mode,
                             "problem_ids": [config.start], "declared_budget": config.agent_timeout,
                             "reference_population": receipt(config),
                             "path": str(config.run_dir), "config_sha256": fingerprint(config.to_dict())})
    plan = {"protocol": STUDY_PROTOCOL, "name": "rl_heldout", "vary": "wall", "jobs": jobs, "skipped_points": [],
            "plan_implementation_sha256": study_digest(), "evaluator_sha256": evaluator_digest(), "experiments_started": False,
            "training_plan_sha256": training_plan["plan_sha256"], "checkpoints": identities,
            "session_independence": "separate_declared_attempts; checkpoint_selection_must_be_declared_before_heldout_execution",
            "gpu_time_or_token_hard_limits_certified": False}
    plan["plan_sha256"] = fingerprint(plan)
    return plan
