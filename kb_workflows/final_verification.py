"""Frozen-source independent remeasurement with replayable fresh-seed manifests."""

from __future__ import annotations

import hashlib
import random
import re
import secrets
import uuid
from pathlib import Path
from statistics import mean

from .common import atomic_json, evaluator_digest, fingerprint, kernel_name, read_json, select_problems
from .tool_checks import reusable

PROTOCOL = "frozen_final_remeasurement_v1"


def _options(config):
    from .isolation import options as isolation_options

    options = {name: getattr(config, name) for name in (
        "level", "suite", "suite_role", "task_profile", "task_track", "task_memory_mb", "start", "end", "num_samples", "precision", "fp32_tolerance",
        "num_correct_trials", "num_perf_trials", "hidden_tests", "strict_validation", "evidence_level",
        "baseline", "baseline_suite", "baseline_portfolio", "cold_calls", "gpu_arch", "hardware", "device_backend", "device_python",
        "paired_timing", "paired_trials", "final_repeats", "evaluation", "timeout",
        "sanitizer_tools", "sanitizer_command", "sanitizer_timeout", "determinism_checks", "determinism_repeats")}
    options["isolation"] = isolation_options(config)
    if config.task_dimensions:
        options["task_dimensions_sha256"] = fingerprint({"dimensions": read_json(Path(config.task_dimensions))})
    if config.determinism_policy_file:
        options["determinism_policy_sha256"] = fingerprint(read_json(Path(config.determinism_policy_file)))
    return options


def _slots(config):
    selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track)
    slots = {}
    for pid, reference in selected.items():
        for sid in range(config.num_samples):
            source = config.run_dir / kernel_name(config.level, pid, sid)
            slots[f"{pid}_{sid}"] = {"problem_id": pid, "sample_id": sid,
                                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() and source.stat().st_size else None,
                                    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    return slots


def freeze(config):
    if config.final_repeats < 1:
        raise ValueError("Final remeasurement is not enabled")
    expected = _slots(config)
    # Resume is a transport policy, not a change to the experiment.
    options = _options(config)
    signature = fingerprint({"config": options, "evaluator": evaluator_digest(), "slots": expected})
    target = config.run_dir / "final_manifest.json"
    prior = read_json(target, {})
    if config.resume and prior.get("signature") == signature:
        for entry in prior.get("slots", {}).values():
            if entry["source_sha256"]:
                if not re.fullmatch("[0-9a-f]{64}", entry["source_sha256"]):
                    raise ValueError("Invalid frozen source checksum")
                path = config.run_dir / "final_artifacts" / (entry["source_sha256"] + ".py")
                if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
                    raise ValueError("Frozen final source is missing or was modified")
        return prior
    artifact_dir = config.run_dir / "final_artifacts"
    if artifact_dir.is_symlink():
        raise ValueError("Frozen artifact directory must not be a symlink")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compatible_prior = (config.resume and prior.get("protocol") == PROTOCOL
                        and prior.get("config_sha256") == fingerprint(options)
                        and prior.get("evaluator_sha256") == evaluator_digest())
    for key, entry in expected.items():
        checksum = entry["source_sha256"]
        old = prior.get("slots", {}).get(key, {})
        reuse = compatible_prior and all(old.get(name) == value for name, value in entry.items())
        if checksum:
            original = config.run_dir / kernel_name(config.level, entry["problem_id"], entry["sample_id"])
            frozen = artifact_dir / (checksum + ".py")
            if frozen.is_symlink():
                raise ValueError("Frozen source must not be a symlink")
            if reuse and (not frozen.is_file() or hashlib.sha256(frozen.read_bytes()).hexdigest() != checksum):
                raise ValueError("Frozen source was changed; cannot reuse final evidence")
            if not reuse:
                frozen.write_bytes(original.read_bytes())
        entry["repeats"] = old["repeats"] if reuse else [{"nonce": uuid.uuid4().hex, "seed": secrets.randbits(31),
                             "hidden_seeds": list(secrets.SystemRandom().sample(range(1 << 31), 3))}
                            for _ in range(config.final_repeats)]
    manifest = {"protocol": PROTOCOL, "signature": signature, "slots": expected,
                "config_sha256": fingerprint(options),
                "evaluator_sha256": evaluator_digest(), "repeats": config.final_repeats,
                "selection": "all_declared_samples; no_fastest_repeat_selection"}
    atomic_json(target, manifest)
    return manifest


def payload_for(config, pid, sid, repeat, *, hidden=False, task="evaluate", variant=None):
    from .engine import task_payload

    manifest = read_json(config.run_dir / "final_manifest.json", {})
    options = _options(config)
    if (manifest.get("protocol") != PROTOCOL or manifest.get("evaluator_sha256") != evaluator_digest()
            or manifest.get("config_sha256") != fingerprint(options)):
        raise ValueError("Missing or stale final manifest")
    reference = select_problems(config.level, problem_ids=[pid])[pid]
    source = config.run_dir / kernel_name(config.level, pid, sid)
    current = {"problem_id": pid, "sample_id": sid,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
               "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    entry = manifest["slots"][f"{pid}_{sid}"]
    if any(entry.get(key) != current[key] for key in current) or not entry["source_sha256"]:
        raise ValueError("Final submission source or reference changed after freezing")
    frozen = config.run_dir / "final_artifacts" / (entry["source_sha256"] + ".py")
    if frozen.is_symlink() or hashlib.sha256(frozen.read_bytes()).hexdigest() != entry["source_sha256"]:
        raise ValueError("Frozen submission was modified")
    payload = task_payload(config, pid, sid, hidden=hidden, task=task)
    payload["kernel_src"] = frozen.read_text()
    plan = entry["repeats"][repeat]
    payload["eval_config"].update(final_protocol=PROTOCOL, final_nonce=plan["nonce"], final_repeat=repeat,
                                  evaluation_seed=plan["seed"], hidden_seeds=plan["hidden_seeds"])
    if variant is not None:
        payload["eval_config"]["paired_baseline_variant"] = variant
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload


def run(config):
    from .engine import evaluate_tasks

    manifest = freeze(config)
    path = config.run_dir / "final_results.json"
    results = read_json(path, {}) if config.resume else {}
    tasks = []
    for key, entry in manifest["slots"].items():
        if not entry["source_sha256"]:
            continue
        for repeat in range(config.final_repeats):
            for hidden in (False, True) if config.hidden_tests else (False,):
                label = f"{key}:{repeat}:{'hidden' if hidden else 'standard'}"
                payload = payload_for(config, entry["problem_id"], entry["sample_id"], repeat, hidden=hidden)
                if config.resume and reusable(results.get(label, {}), payload):
                    continue
                tasks.append(("final-" + label, payload))
    for label, result in evaluate_tasks(config, tasks):
        results[label.removeprefix("final-")] = result
        atomic_json(path, results)
    atomic_json(path, results)
    return results


def validate(config, pid, sid, baseline):
    from .evidence import score_issue
    from .isolation import runtime_issue
    from .paper_analysis import compatible, current_result, measurement_evidence

    if not config.final_repeats:
        return None, []
    results = read_json(config.run_dir / "final_results.json", {})
    standard = []
    for repeat in range(config.final_repeats):
        for hidden in (False, True) if config.hidden_tests else (False,):
            label = f"{pid}_{sid}:{repeat}:{'hidden' if hidden else 'standard'}"
            result = results.get(label, {})
            if result.get("compiled") is not True or result.get("correctness") is not True:
                return "final_remeasurement_failed_or_missing", []
            try:
                payload = payload_for(config, pid, sid, repeat, hidden=hidden)
            except (KeyError, IndexError, ValueError, OSError):
                return "final_manifest_changed_or_missing", []
            if current_result(result, payload) or score_issue(result, payload["eval_config"], "hidden" if hidden else "standard"):
                return "final_remeasurement_evidence_mismatch", []
            if runtime_issue(baseline, result, payload["eval_config"]):
                return "final_remeasurement_runtime_mismatch", []
            if not hidden:
                if compatible(baseline, result, evaluator_digest()) or measurement_evidence(result, candidate=True):
                    return "final_remeasurement_measurement_mismatch", []
                standard.append(result)
    return None, standard


def correctness_passed(config, pid, sid):
    from .paper_analysis import current_result

    if not config.final_repeats:
        return True
    results = read_json(config.run_dir / "final_results.json", {})
    for repeat in range(config.final_repeats):
        for hidden in (False, True) if config.hidden_tests else (False,):
            result = results.get(f"{pid}_{sid}:{repeat}:{'hidden' if hidden else 'standard'}", {})
            if result.get("compiled") is not True or result.get("correctness") is not True:
                return False
            try:
                payload = payload_for(config, pid, sid, repeat, hidden=hidden)
            except (ValueError, KeyError, IndexError, OSError):
                return False
            if current_result(result, payload):
                return False
    return True


def timing_summary(results, bootstrap_trials=2000):
    """Resample independent processes before timing samples, retaining every repeat."""
    groups = [record["runtime_stats"]["samples_ms"] for record in results]
    if not groups or any(not values for values in groups):
        raise ValueError("Missing independent timing samples")
    rng = random.Random(42)
    draws = []
    for _ in range(bootstrap_trials):
        means = []
        for _ in groups:
            chosen = rng.choice(groups)
            means.append(mean(rng.choice(chosen) for _ in chosen))
        draws.append(mean(means))
    draws.sort()
    return {"mean": mean(mean(values) for values in groups), "num_trials": sum(map(len, groups)),
            "independent_repeats": len(groups), "mean_ci95_ms": [draws[int(0.025 * (bootstrap_trials - 1))],
                                                                    draws[int(0.975 * (bootstrap_trials - 1))]],
            "ci_method": "independent_repeat_then_sample_percentile_bootstrap"}
