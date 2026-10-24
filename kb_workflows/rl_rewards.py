"""Staged native rewards and interleaved frozen remeasurement of one rollout group."""

from __future__ import annotations

import ast
import hashlib
import random
import secrets
import time
from pathlib import Path

from .agent_models import source_from_response
from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, kernel_name, read_json
from .evidence import score_issue
from .rl_plan import task_config, validate_plan
from .study_statistics import finite

PROTOCOL = "kernelbench_pro_rl_reward_v1"


def document_digest(value):
    return fingerprint({"document": value})


def parse_source(text):
    try:
        source = source_from_response(text)
        tree = ast.parse(source)
        if not any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body):
            raise ValueError("Response must define a top-level ModelNew class")
        return source, None
    except (SyntaxError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _default_dispatch(config, payload):
    from .engine import evaluate_tasks

    return next(iter(evaluate_tasks(config, [("rl-" + fingerprint(payload), payload)])))[1]


def components(row, standard, baseline, *, raw_clip):
    """An explicit weak public-speedup ablation stays distinct from qualified rewards."""
    eligible = row.get("status") == "verified"
    value = row.get("score", 0) if eligible else 0
    if not finite(value) or not 0 <= value <= 1:
        raise ValueError("Qualified native scores must be finite and bounded")
    speedup = (baseline.get("mean") / standard["runtime"] if standard.get("compiled") is True
               and standard.get("correctness") is True and finite(standard.get("runtime")) and standard["runtime"] > 0
               and finite(baseline.get("mean")) and baseline["mean"] > 0 else None)
    return {"qualified": value, "raw_speedup": min(speedup, raw_clip) / raw_clip if speedup is not None else 0.0,
            "public_speedup": speedup, "qualified_status": row.get("status", "failed_or_missing"),
            "qualification_accepted": eligible, "certificate_bonus": 0.0}


def evaluate_group(plan, task_key, completions, *, directory, split="train", purpose=None, dispatch=None, analyzer=None,
                   certificate_collector=None):
    """Execution entrypoint; callers must explicitly choose to run a training/evaluation job."""
    from .engine import paired_payload, task_payload
    from .final_verification import freeze, payload_for
    from .paper_analysis import analyze_paper

    training = validate_plan(plan)
    purpose = purpose or ("training" if split == "train" else "heldout")
    if purpose not in {"training", "heldout", "audit", "certificate"} or purpose == "training" and split != "train":
        raise ValueError("Invalid reward evaluation purpose")
    if task_key not in plan["splits"].get(split, []) or not isinstance(completions, list) or not completions:
        raise ValueError("Reward task must belong to its explicit split")
    if purpose == "training" and len(completions) != training.group_size:
        raise ValueError("Training requires the complete declared rollout group")
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("A reward group cannot overwrite or silently reuse earlier evaluation attempts")
    directory.mkdir(parents=True, exist_ok=True)
    config = task_config(plan, task_key, directory=directory, samples=len(completions))
    from .reference_population import freeze_run

    frozen_population = freeze_run(config)
    atomic_json(directory / "run_config.json", config.to_dict())
    external_inputs = {}
    for name in ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file"):
        filename = getattr(config, name)
        data = read_json(Path(filename)) if filename else {}
        atomic_json(directory / f"input_{name}.json", data)
        if filename:
            external_inputs[name] = {"path": filename, "sha256": document_digest(data)}
    baseline_path = ROOT / "results/timing" / config.hardware / f"{config.baseline}.json"
    reference = select_reference(config)
    baseline_entry = read_json(baseline_path, {}).get(f"level{config.level}", {}).get(reference.name, {})
    frozen_baseline = fingerprint(baseline_entry)
    atomic_json(directory / "reward_baseline.json", baseline_entry)
    dispatch = dispatch or _default_dispatch
    pid = config.start
    sources, errors = zip(*(parse_source(text) for text in completions))
    checksums = [hashlib.sha256(source.encode()).hexdigest() if source else None for source in sources]
    request = {"protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"], "task_key": task_key, "split": split, "purpose": purpose,
               "evaluator_sha256": evaluator_digest(), "reference_sha256": plan["tasks"][task_key]["reference_sha256"],
               "hidden_source_sha256": plan["tasks"][task_key]["hidden_source_sha256"],
               "external_inputs": external_inputs,
               "source_sha256": checksums, "declared_samples": len(sources), "static_errors": list(errors)}
    atomic_json(directory / "reward_request.json", request)
    for sid, source in enumerate(sources):
        if source:
            (directory / kernel_name(config.level, pid, sid)).write_text(source)
    records = {name: {} for name in ("small", "standard", "hidden", "final", "paired")}
    filenames = {"small": "small_results.json", "standard": "eval_results.json", "hidden": "eval_results_hidden.json",
                 "final": "final_results.json", "paired": "paired_results.json"}
    schedule = []

    def run(phase, key, payload):
        started = time.monotonic()
        try:
            result = dispatch(config, payload)
            if not isinstance(result, dict) or not isinstance(result.get("metadata", {}), dict):
                raise ValueError("Malformed verifier result")
        except Exception as exc:
            result = failure(str(exc), error_name=type(exc).__name__)
        result.setdefault("metadata", {}).update(workflow_signature=fingerprint(payload), evaluation_config=payload["eval_config"])
        records[phase][key] = result
        atomic_json(directory / filenames[phase], records[phase])
        device = result["metadata"].get("device_fingerprint", {})
        schedule.append({"phase": phase, "slot": key, "request_sha256": fingerprint(payload),
                         "request_wall_seconds": time.monotonic() - started, "device_uuid": device.get("uuid"),
                         "timed_correct_result": result.get("compiled") is True and result.get("correctness") is True
                             and finite(result.get("runtime")) and result["runtime"] > 0})
        atomic_json(directory / "reward_schedule.json", schedule)
        return result

    active = []
    for sid, source in enumerate(sources):
        key = f"{pid}_{sid}"
        if source is None:
            records["standard"][key] = failure(errors[sid])
            continue
        good = True
        if config.strict_validation:
            small = task_payload(config, pid, sid, task="qualify_small")
            result = run("small", key, small)
            good = (result.get("compiled") is True and result.get("correctness") is True
                    and score_issue(result, small["eval_config"], "standard") is None)
        if not good and training.reward_mode == "qualified":
            records["standard"][key] = failure("Small-instance qualification failed; expensive stages skipped")
            continue
        result = run("standard", key, task_payload(config, pid, sid))
        if result.get("compiled") is True and result.get("correctness") is True:
            secret = run("hidden", key, task_payload(config, pid, sid, hidden=True))
            if secret.get("compiled") is True and secret.get("correctness") is True:
                active.append(sid)
    for phase in ("small", "standard", "hidden"):
        atomic_json(directory / filenames[phase], records[phase])

    manifest = freeze(config)
    # Equal workload seeds across a prompt group, fresh per repeat and unavailable to development tools.
    if active:
        common = manifest["slots"][f"{pid}_{active[0]}"]["repeats"]
        for entry in manifest["slots"].values():
            for repeat, shared in zip(entry["repeats"], common):
                repeat.update(seed=shared["seed"], hidden_seeds=list(shared["hidden_seeds"]))
        atomic_json(directory / "final_manifest.json", manifest)
    order_seed = secrets.randbits(128)
    rng = random.Random(order_seed)
    orders = []
    for repeat in range(config.final_repeats):
        order = active.copy()
        rng.shuffle(order)
        orders.append(order)
        for hidden in (False, True):
            for sid in order if not hidden else reversed(order):
                key = f"{pid}_{sid}:{repeat}:{'hidden' if hidden else 'standard'}"
                run("final", key, payload_for(config, pid, sid, repeat, hidden=hidden))
        for sid in order:
            try:
                payload = paired_payload(config, pid, sid, repeat=repeat)
            except (ValueError, OSError, KeyError) as exc:
                records["paired"][f"{pid}_{sid}:{repeat}"] = failure(str(exc))
                continue
            run("paired", f"{pid}_{sid}:{repeat}", payload)
    for phase in ("final", "paired"):
        atomic_json(directory / filenames[phase], records[phase])
    report = (analyzer or analyze_paper)(config)
    details = {row["sample_id"]: row for row in report["details"]}
    reference = select_reference(config)
    baseline = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {}).get(f"level{config.level}", {}).get(reference.name, {})
    uuids = [row["device_uuid"] for row in schedule if row["phase"] in {"standard", "final", "paired"} and row["timed_correct_result"]]
    baseline_uuid = baseline.get("device_fingerprint", {}).get("uuid")
    same_device = bool(uuids) and isinstance(baseline_uuid, str) and bool(baseline_uuid) and all(value == baseline_uuid for value in uuids)
    values = [components(details.get(sid, {}), records["standard"].get(f"{pid}_{sid}", {}), baseline,
                         raw_clip=training.raw_speedup_clip) for sid in range(len(sources))]
    for value in values:
        value["reward"] = value[training.reward_mode] if same_device else 0.0
        if not same_device:
            value["device_group_status"] = "missing_or_different_physical_device_receipt"
    certificates, certificate_seconds = {}, 0.0
    if training.certificate_bonus:
        from .scaling_certificate import collect, read_contracts

        contract = read_contracts(training.scaling_contracts)["tasks"].get(task_key)
        for sid, value in enumerate(values):
            if not contract or not same_device or not value["qualification_accepted"] or value["qualified"] < training.certificate_min_score:
                continue
            started = time.monotonic()
            try:
                receipt = (certificate_collector or collect)(plan, task_key, sources[sid], contract, directory=directory / "certificates" / str(sid),
                                  anchor_metadata=records["standard"][f"{pid}_{sid}"]["metadata"], dispatch=dispatch)
                certificates[str(sid)] = {"path": f"certificates/{sid}/certificate.json", "receipt_sha256": receipt["receipt_sha256"],
                                          "passed": receipt["fit"]["passed"]}
                if receipt["fit"]["passed"]:
                    value["certificate_bonus"] = min(training.certificate_bonus, 1 - value["qualified"])
                    value["reward"] = value["qualified"] + value["certificate_bonus"]
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                certificates[str(sid)] = {"passed": False, "error": str(exc)}
            finally:
                certificate_seconds += time.monotonic() - started
    for sid, value in enumerate(values):
        failed = {"static_source"} if errors[sid] else set()
        for phase in ("small", "standard", "hidden"):
            evidence = records[phase].get(f"{pid}_{sid}", {}).get("metadata", {}).get("evidence_receipt", {}).get("checks", {})
            failed.update(name for name, item in evidence.items() if item.get("status") == "failed")
        value["failed_checks"] = sorted(failed)
    result = {**request, "reward_mode": training.reward_mode, "components": values, "rewards": [value["reward"] for value in values],
              "interleaving": {"scope": "fresh_processes_interleaved_by_repeat; not_simultaneous_resident_models",
                               "orders": orders, "order_seed": order_seed, "common_seeds": True,
                               "same_physical_device_verified": same_device, "drift_eliminated": False},
              "scaling_certificates": certificates,
              "source_files": {name: document_digest(read_json(directory / name, {})) for name in
                    ("run_config.json", "small_results.json", "eval_results.json", "eval_results_hidden.json",
                     "final_manifest.json", "final_results.json", "paired_results.json", "paper_analysis.json", "reward_schedule.json", "reward_baseline.json",
                     "input_problem_specs.json", "input_device_calibration.json", "input_input_calibration.json", "input_precision_calibration.json", "input_reference_population.json",
                     "input_task_dimensions.json", "input_determinism_policy_file.json") + (("reference_population.json",) if frozen_population else ())
                     + (("precision_calibration_manifest.json",) if config.precision_calibration else ())},
              "request_wall_seconds": sum(row["request_wall_seconds"] for row in schedule) + certificate_seconds,
              "certificate_request_wall_seconds": certificate_seconds, "gpu_execution_seconds": None,
              "candidate_count": len(sources), "development_feedback_contains_hidden_results": False}
    validate_plan(plan)
    current_baseline = read_json(baseline_path, {}).get(f"level{config.level}", {}).get(reference.name, {})
    if fingerprint(current_baseline) != frozen_baseline or fingerprint(read_json(directory / "reward_baseline.json", {})) != frozen_baseline:
        raise ValueError("Baseline evidence changed while evaluating the rollout group")
    for sid, checksum in enumerate(checksums):
        if checksum and hashlib.sha256((directory / kernel_name(config.level, pid, sid)).read_bytes()).hexdigest() != checksum:
            raise ValueError("Candidate source changed while evaluating the rollout group")
    result["baseline_file"] = {"path": str(baseline_path.relative_to(ROOT)), "level": config.level,
                               "reference": reference.name, "sha256": frozen_baseline}
    result["receipt_sha256"] = fingerprint(result)
    atomic_json(directory / "reward.json", result)
    return result


def select_reference(config):
    from .common import select_problems

    return select_problems(config.level, problem_ids=[config.start])[config.start]


def baseline_coverage(plan, split="train"):
    """Read current baseline evidence before loading policy weights or starting generation."""
    from .baseline_variants import variants_for
    from .engine import task_payload
    from .paper_analysis import measurement_evidence

    validate_plan(plan)
    if split not in plan["splits"]:
        raise ValueError("Unknown baseline preparation split")
    rows = []
    for key in plan["splits"][split]:
        config = task_config(plan, key, directory=Path(plan["output_directory"]) / "baseline_preview")
        reference = select_reference(config)
        baseline = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {}).get(f"level{config.level}", {}).get(reference.name, {})
        expected = task_payload(config, config.start, task="baseline")
        variants = baseline.get("variants", {})
        issue = None
        if (not baseline.get("baseline_suite") or not baseline.get("suite_complete")
                or baseline.get("metadata", {}).get("workflow_signature") != fingerprint(expected)
                or not isinstance(baseline.get("device_fingerprint", {}).get("uuid"), str)
                or not baseline["device_fingerprint"]["uuid"]):
            issue = "missing_stale_or_device_unbound_baseline"
        elif set(variants) != set(variants_for(config, config.start)):
            issue = "baseline_portfolio_incomplete"
        else:
            for name, value in variants.items():
                payload = task_payload(config, config.start, task="baseline")
                payload["eval_config"]["baseline_variant"] = name
                if (value.get("baseline_correctness") is not True or measurement_evidence(value) is not None
                        or value.get("metadata", {}).get("workflow_signature") != fingerprint(payload)):
                    issue = "baseline_variant_missing_or_incompatible_evidence"
                    break
        rows.append({"task_key": key, "ready": issue is None, "reason": issue})
    return rows


def prepare_baselines(plan, *, split="train", execute=False):
    rows = baseline_coverage(plan, split)
    if not execute:
        return {"experiments_started": False, "baselines": rows}
    from .engine import record_baseline

    for row in rows:
        if not row["ready"]:
            config = task_config(plan, row["task_key"], directory=Path(plan["output_directory"]) / "baselines" / row["task_key"].replace("/", "_"))
            config.stage = "baseline"
            record_baseline(config)
    return {"experiments_started": True, "baselines": baseline_coverage(plan, split)}


def validate_reward(path):
    path = Path(path)
    receipt = read_json(path)
    if (not isinstance(receipt, dict) or receipt.get("protocol") != PROTOCOL or receipt.get("receipt_sha256") !=
            fingerprint({key: value for key, value in receipt.items() if key != "receipt_sha256"})
            or receipt.get("evaluator_sha256") != evaluator_digest()):
        raise ValueError("Reward receipt is missing or changed")
    if any(document_digest(read_json(path.parent / name, {})) != digest for name, digest in receipt["source_files"].items()):
        raise ValueError("Underlying native reward evidence changed")
    for item in receipt.get("external_inputs", {}).values():
        if document_digest(read_json(Path(item["path"]))) != item["sha256"]:
            raise ValueError("Reward verifier input files changed")
    for sid, item in receipt.get("scaling_certificates", {}).items():
        if item.get("path"):
            from .scaling_certificate import validate as validate_scaling

            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Scaling receipt path escapes the reward group")
            certificate = validate_scaling(path.parent / relative)
            if (certificate["receipt_sha256"] != item["receipt_sha256"] or certificate["fit"]["passed"] != item["passed"]
                    or certificate["candidate_sha256"] != receipt["source_sha256"][int(sid)]
                    or certificate["parent_plan_sha256"] != receipt["plan_sha256"]):
                raise ValueError("Scaling bonus has different candidate/plan evidence")
    for sid, checksum in enumerate(receipt["source_sha256"]):
        level, pid = map(int, receipt["task_key"].split("/"))
        source = path.parent / kernel_name(level, pid, sid)
        if checksum and (not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum):
            raise ValueError("Reward candidate source changed")
    from .common import select_problems

    reference = select_problems(level, problem_ids=[pid])[pid]
    hidden = ROOT / "hidden_tests" / f"level{level}" / f"{pid}_hidden.py"
    baseline = receipt["baseline_file"]
    relative = Path(baseline["path"])
    current_baseline = read_json(ROOT / relative, {}).get(f"level{level}", {}).get(reference.name, {}) if not relative.is_absolute() and ".." not in relative.parts else {}
    if (relative.is_absolute() or ".." in relative.parts or baseline.get("level") != level or baseline.get("reference") != reference.name
            or fingerprint(current_baseline) != baseline["sha256"]
            or hashlib.sha256(reference.read_bytes()).hexdigest() != receipt["reference_sha256"]
            or not hidden.is_file() or hashlib.sha256(hidden.read_bytes()).hexdigest() != receipt["hidden_source_sha256"]):
        raise ValueError("Reward task or baseline provenance changed")
    return receipt
