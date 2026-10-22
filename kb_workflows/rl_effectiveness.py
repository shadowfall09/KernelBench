"""Preregistered reward comparisons with policy lineage and paired training-seed inference."""

from __future__ import annotations

import copy
import itertools
import random
from pathlib import Path
from statistics import mean

from .common import atomic_json, evaluator_digest, fingerprint, read_json
from .rl_lineage import comparison_contract
from .rl_lineage import validate as validate_lineage
from .rl_plan import implementation_digest, validate_plan, write_plan
from .rl_plan import make_plan as training_plan
from .study_statistics import aggregate, finite, interval

PROTOCOL = "kernelbench_pro_reward_experiment_v1"
ARMS = ("qualified", "raw_speedup")


def sign_flip(differences, *, seed=42, draws=20000):
    """Two-sided paired sign randomization; requires exchangeability/sign symmetry under H0."""
    if not differences or any(not finite(value) for value in differences):
        raise ValueError("Paired randomization needs finite training-seed differences")
    observed = abs(mean(differences))
    exact = len(differences) <= 16
    if exact:
        signs = itertools.product((-1, 1), repeat=len(differences))
        values = [abs(mean(value * sign for value, sign in zip(differences, row))) for row in signs]
        p = sum(value >= observed - 1e-12 for value in values) / len(values)
    else:
        rng = random.Random(seed)
        values = [abs(mean(value * rng.choice((-1, 1)) for value in differences)) for _ in range(draws)]
        p = (1 + sum(value >= observed - 1e-12 for value in values)) / (1 + len(values))
    return {"p_two_sided": p, "exact": exact, "draws": len(values), "paired_training_seeds": len(differences),
            "assumption": "paired_training_seed_label_exchangeability_or_sign_symmetry_under_null; not_guaranteed_by_seed_labels"}


def paired_effect(values, layers, *, trials=2000, seed=42):
    """Pair training seeds/tasks; resample evaluation sessions and candidate slots independently per arm."""
    if type(trials) is not int or trials < 200 or set(values) != set(ARMS) or not layers:
        raise ValueError("Effect analysis needs two arms, tasks and >=200 draws")
    seeds = sorted(values[ARMS[0]])
    if not seeds or seeds != sorted(values[ARMS[1]]):
        raise ValueError("All preregistered training seeds must be retained in both arms")
    population = set(layers)
    for arm in ARMS:
        for key in seeds:
            sessions = values[arm][key]
            if not sessions:
                raise ValueError("Each policy needs declared evaluation sessions")
            reference_sessions = values[ARMS[0]][seeds[0]]
            if set(sessions) != set(reference_sessions):
                raise ValueError("All policies need the same declared evaluation sessions")
            for tasks in sessions.values():
                if set(tasks) != population or any(not slots or any(not finite(value) or not 0 <= value <= 1 for value in slots) for slots in tasks.values()):
                    raise ValueError("Every held-out session needs every task and all declared bounded slots")
                reference_tasks = next(iter(reference_sessions.values()))
                if any(len(slots) != len(reference_tasks[task]) for task, slots in tasks.items()):
                    raise ValueError("All policies need the same declared candidate slots per task")
    means = {arm: {key: {task: mean(mean(session[task]) for session in values[arm][key].values())
                         for task in population} for key in seeds} for arm in ARMS}
    seed_scores = {arm: {key: aggregate(list(means[arm][key].items()), layers, "layer_macro") for key in seeds} for arm in ARMS}
    differences = [seed_scores["qualified"][key] - seed_scores["raw_speedup"][key] for key in seeds]
    strata = {}
    for task in sorted(population):
        strata.setdefault(layers[task], []).append(task)
    rng, draws = random.Random(seed), []
    for _ in range(trials):
        selected = [rng.choice(seeds) for _ in seeds]
        tasks = [rng.choice(group) for group in strata.values() for _ in group]
        arm_scores = {}
        for arm in ARMS:
            scores = []
            for key in selected:
                sessions = list(values[arm][key].values())
                sessions = [rng.choice(sessions) for _ in sessions]
                rows = [(task, mean(mean(rng.choice(session[task]) for _ in session[task]) for session in sessions)) for task in tasks]
                scores.append(aggregate(rows, layers, "layer_macro"))
            arm_scores[arm] = mean(scores)
        draws.append(arm_scores["qualified"] - arm_scores["raw_speedup"])
    ci, test = interval(draws), sign_flip(differences, seed=seed)
    return {"score_means": {arm: mean(seed_scores[arm].values()) for arm in ARMS}, "paired_seed_scores": seed_scores,
            "qualified_minus_raw": mean(differences), "ci95": ci, "randomization": test,
            "positive_effect_supported": len(seeds) >= 2 and ci[0] > 0 and test["p_two_sided"] < 0.05,
            "bootstrap": "paired_training_seed_and_layer_stratified_tasks; independent_evaluation_sessions_and_slots_per_arm",
            "training_seed_count": len(seeds), "task_count": len(population), "trials": trials}


def make_experiment(specification, *, directory, base_directory=None):
    if not isinstance(specification, dict) or set(specification) - {"base", "seeds", "evaluation_sessions", "evaluation_samples"}:
        raise ValueError("Unknown reward experiment fields")
    seeds = specification.get("seeds")
    sessions, samples = specification.get("evaluation_sessions", 3), specification.get("evaluation_samples", 1)
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or type(sessions) is not int or sessions < 1 or type(samples) is not int or samples < 1):
        raise ValueError("Declare unique nonnegative training seeds and positive held-out sessions/samples")
    root = Path(directory).resolve()
    jobs, contract = [], None
    for arm in ARMS:
        for seed in seeds:
            spec = copy.deepcopy(specification["base"])
            spec.setdefault("training", {}).update(reward_mode=arm, seed=seed)
            if arm == "raw_speedup":
                spec["training"].update(certificate_bonus=0.0, scaling_contracts="")
            plan = training_plan(spec, directory=root / "training" / arm / f"seed_{seed}", base_directory=base_directory)
            current = comparison_contract(plan)
            if contract is None:
                contract = current
            elif current != contract:
                raise ValueError("Training arms differ outside the declared reward treatment and random seed")
            jobs.append({"arm": arm, "seed": seed, "plan": plan, "checkpoint": str(Path(plan["output_directory"]) / "final_model")})
    experiment = {"protocol": PROTOCOL, "directory": str(root), "seeds": seeds, "jobs": jobs, "comparison_contract": contract,
                  "evaluation_sessions": sessions, "evaluation_samples": samples, "evaluator_sha256": evaluator_digest(),
                  "rl_implementation_sha256": implementation_digest(), "experiments_started": False,
                  "declared_training_wall_upper_seconds": sum(job["plan"]["training"]["wall_seconds"] for job in jobs),
                  "declared_generated_output_upper_bound": sum(job["plan"]["declared_generated_output_upper_bound"] for job in jobs)}
    experiment["experiment_sha256"] = fingerprint(experiment)
    return experiment


def validate_experiment(experiment):
    if (experiment.get("protocol") != PROTOCOL or experiment.get("experiment_sha256") !=
            fingerprint({key: value for key, value in experiment.items() if key != "experiment_sha256"})
            or experiment.get("evaluator_sha256") != evaluator_digest() or experiment.get("rl_implementation_sha256") != implementation_digest()):
        raise ValueError("Reward experiment is changed or stale")
    expected = {(arm, seed) for arm in ARMS for seed in experiment["seeds"]}
    if {(job["arm"], job["seed"]) for job in experiment["jobs"]} != expected or len(experiment["jobs"]) != len(expected):
        raise ValueError("Preregistered reward experiment lost or duplicated a trial")
    for job in experiment["jobs"]:
        validate_plan(job["plan"])
        if (job["plan"]["training"]["reward_mode"] != job["arm"] or job["plan"]["training"]["seed"] != job["seed"]
                or comparison_contract(job["plan"]) != experiment["comparison_contract"]):
            raise ValueError("A reward trial differs from the preregistered comparison")


def write_experiment(experiment, path):
    validate_experiment(experiment)
    path = Path(path)
    for job in experiment["jobs"]:
        write_plan(job["plan"], path.parent / "training_plans" / f"{job['arm']}_{job['seed']}.json")
    atomic_json(path, experiment)
    return path


def execute_training(experiment, *, execute=False, executor=None, keep_going=False):
    from .rl_training import execute_plan

    validate_experiment(experiment)
    if not execute:
        return {"experiments_started": False, "trial_count": len(experiment["jobs"]),
                "declared_training_wall_upper_seconds": experiment["declared_training_wall_upper_seconds"]}
    root, rows = Path(experiment["directory"]), []
    for job in experiment["jobs"]:
        path = root / "training_plans" / f"{job['arm']}_{job['seed']}.json"
        write_plan(job["plan"], path)
        try:
            validated = validate_lineage(job["checkpoint"])
        except (ValueError, OSError, KeyError):
            result = (executor or execute_plan)(path, execute=True)
        else:
            result = {"exit_code": 0, "completion_verified": True, "reused_completed_policy": True,
                      "lineage_sha256": validated["receipt"]["receipt_sha256"]}
        rows.append({"arm": job["arm"], "seed": job["seed"], "plan_sha256": job["plan"]["plan_sha256"], "result": result})
        atomic_json(root / "training_execution.json", {"experiment_sha256": experiment["experiment_sha256"], "trials": rows})
        if result.get("exit_code") and not keep_going:
            break
    return {"experiments_started": True, "trials": rows}


def evaluation_plan(experiment, *, directory, audit_only=False):
    from .rl_evaluation import heldout_plan

    validate_experiment(experiment)
    missing, checkpoints, lineages, software = [], {}, {}, None
    for job in experiment["jobs"]:
        label = f"{job['arm']}_{job['seed']}"
        try:
            verified = validate_lineage(job["checkpoint"], allow_injected=audit_only, require_supervisor=not audit_only)
        except (ValueError, OSError, KeyError) as exc:
            missing.append({"arm": job["arm"], "seed": job["seed"], "reason": str(exc)})
            continue
        if verified["receipt"]["plan_sha256"] != job["plan"]["plan_sha256"]:
            raise ValueError("Held-out policy is not the predeclared final model of its training trial")
        if software is None:
            software = verified["receipt"]["software_versions"]
        elif software != verified["receipt"]["software_versions"]:
            raise ValueError("Training software versions differ across reward arms/seeds")
        checkpoints[label] = job["checkpoint"]
        lineages[label] = {"checkpoint": job["checkpoint"], "sha256": verified["receipt"]["receipt_sha256"],
                           "formal_comparison_eligible": verified["formal_comparison_eligible"], "arm": job["arm"], "training_seed": job["seed"]}
    if missing:
        return {"status": "not_ready", "missing_trials": missing, "experiments_started": False}
    result = heldout_plan(experiment["jobs"][0]["plan"], checkpoints, directory=directory,
                          sessions=experiment["evaluation_sessions"], samples=experiment["evaluation_samples"])
    result.update(reward_experiment=experiment, training_lineages=lineages, audit_only=audit_only)
    result["plan_sha256"] = fingerprint({key: value for key, value in result.items() if key != "plan_sha256"})
    return result


def analyze(evaluation, *, trials=2000, seed=42):
    """Recompute held-out evidence only; incomplete preregistered trials never become a smaller study."""
    from .rl_rewards import validate_reward
    from .study_data import collect_run
    from .study_plan import execute_plan

    experiment = evaluation["reward_experiment"]
    validate_experiment(experiment)
    root = Path(evaluation["jobs"][0]["path"]).parent.parent
    # The maintained study runner verifies all current frozen checkpoint/lineage identities.
    plan_path = root / "plan.json"
    if read_json(plan_path) != evaluation:
        raise ValueError("Effect analysis requires the exact saved held-out plan")
    execute_plan(plan_path, execute=False)
    execution = read_json(root / "execution.json", {})
    completed = {row["path"]: row for row in execution.get("jobs", [])}
    if execution.get("plan_sha256") != evaluation["plan_sha256"] or any(job["path"] not in completed for job in evaluation["jobs"]):
        return {"status": "not_measured", "reason": "All preregistered held-out jobs need execution receipts", "experiments_started": False}
    values = {arm: {} for arm in ARMS}
    layers, observed, source_receipts, apparatus = {}, set(), [], []
    by_checkpoint = {row["checkpoint"]: row for row in evaluation["training_lineages"].values()}
    for job in evaluation["jobs"]:
        if completed[job["path"]].get("config_sha256") != job["config_sha256"]:
            raise ValueError("Held-out execution config differs from its preregistration")
        saved = read_json(Path(job["path"]) / "run_config.json")
        if saved != job["config"]:
            raise ValueError("Held-out run configuration is missing or changed")
        result = collect_run(job["path"], session=job["session"])
        identity = by_checkpoint[job["config"]["model"]]
        arm, training_seed = identity["arm"], str(identity["training_seed"])
        session = values[arm].setdefault(training_seed, {}).setdefault(job["session"], {})
        if set(session) & set(result["tasks"]):
            raise ValueError("Duplicate held-out tasks cannot become best-of-N")
        session.update(result["tasks"])
        for task, layer in result["layers"].items():
            if task in layers and layers[task] != layer:
                raise ValueError("Held-out score strata changed between arms")
            layers[task] = layer
        observed.add((arm, training_seed, job["session"]))
        source_receipts.append(result["receipt_sha256"])
        apparatus.append(result["formal_comparison_eligible"])
        budget = read_json(Path(job["path"]) / "generation_budget.json", {})
        if budget.get("protocol", {}).get("mode") != job["config"]["generation_mode"]:
            raise ValueError("Held-out inference changed its declared generation method")
        expected_checkpoint = next(item["sha256"] for item in evaluation["checkpoints"].values()
                                   if item["path"] == job["config"]["model"])
        for receipt in budget.get("slots", {}).values():
            if receipt and receipt.get("candidate_submissions", 0) and receipt.get("local_checkpoint_sha256") != [expected_checkpoint]:
                raise ValueError("Held-out generation used different local model weights")
    expected_tasks = set(experiment["jobs"][0]["plan"]["splits"]["heldout"])
    expected_trials = {(arm, str(key), f"session_{i:03}") for arm in ARMS for key in experiment["seeds"] for i in range(experiment["evaluation_sessions"])}
    if observed != expected_trials or set(layers) != expected_tasks:
        raise ValueError("The held-out study omitted a predeclared seed/session/task")
    effect = paired_effect(values, layers, trials=trials, seed=seed)
    blocked, resources = {arm: {} for arm in ARMS}, {arm: [] for arm in ARMS}
    for job in experiment["jobs"]:
        lineage = validate_lineage(job["checkpoint"], allow_injected=evaluation.get("audit_only", False),
                                   require_supervisor=not evaluation.get("audit_only", False))
        total = len(lineage["journal"]["groups"]) * job["plan"]["training"]["group_size"]
        known = failures = 0
        for group in lineage["journal"]["groups"]:
            if group.get("reward_path"):
                reward = validate_reward(group["reward_path"])
                known += reward["candidate_count"]
                failures += sum(bool(row.get("failed_checks")) for row in reward["components"])
        blocked[job["arm"]][str(job["seed"])] = {"declared_attempted_slots": total, "measured_slots": known,
            "failed_check_slots": failures, "unknown_slots": total - known,
            "rate_bounds": [failures / total, (failures + total - known) / total] if total else None}
        resources[job["arm"]].append({"training_seed": job["seed"], "consumed_groups": lineage["receipt"]["consumed_groups"],
                                      "active_optimizer_steps": lineage["receipt"]["global_step"], "supervised": lineage["supervised"]})
    complete_blocks = all(row["declared_attempted_slots"] and row["unknown_slots"] == 0 for arm in blocked.values() for row in arm.values())
    block_diffs = [blocked["raw_speedup"][str(key)]["rate_bounds"][0] - blocked["qualified"][str(key)]["rate_bounds"][0]
                   for key in experiment["seeds"]] if complete_blocks else []
    block_test = sign_flip(block_diffs, seed=seed) if block_diffs else None
    formal = not evaluation.get("audit_only", False) and all(apparatus) and all(row["formal_comparison_eligible"] for row in evaluation["training_lineages"].values())
    result = {"status": "analyzed", "effect": effect, "training_blocked_checks": blocked, "blocked_check_randomization": block_test,
              "blocked_check_reduction_supported": bool(formal and block_test and mean(block_diffs) > 0 and block_test["p_two_sided"] < 0.05),
              "heldout_improvement_supported": formal and effect["positive_effect_supported"],
              "formal_comparison_eligible": formal, "resources": resources,
              "budget_policy": "same_preregistered_optimizer_data_and_declared_limits; actual_costs_retained; equal_GPU_busy_time_not_certified",
              "heldout_source_receipts": source_receipts, "experiment_sha256": experiment["experiment_sha256"], "experiments_started": False}
    result["receipt_sha256"] = fingerprint(result)
    return result
