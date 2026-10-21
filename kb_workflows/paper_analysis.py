"""Failure-inclusive measurement scoring alongside the original Verified metrics."""

from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import mean

from .common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name, read_json, select_problems
from .measurement import amortized

DEVICE_FIELDS = ("name", "capability", "total_memory", "multiprocessors", "l2_bytes", "torch_version", "cuda_version", "hip_version")


def same_device_class(first, second):
    return bool(first.get("available") and second.get("available")) and all(
        name in first and name in second and first[name] == second[name] for name in DEVICE_FIELDS
    )


def valid_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def compatible(baseline, candidate, current_digest):
    metadata = candidate.get("metadata", {})
    if baseline.get("baseline_correctness") is not True or (baseline.get("baseline_suite") and not baseline.get("suite_complete")):
        return "baseline_not_validated_or_incomplete"
    if not same_device_class(baseline.get("device_fingerprint", {}), metadata.get("device_fingerprint", {})):
        return "device_or_software_mismatch"
    if baseline.get("timing_method") != "synchronized_wall" or metadata.get("timing_method") != "synchronized_wall":
        return "timing_protocol_mismatch"
    if baseline.get("precision") != metadata.get("precision"):
        return "precision_mismatch"
    if (baseline.get("metadata", {}).get("evaluation_config", {}).get("fp32_tolerance")
            != metadata.get("evaluation_config", {}).get("fp32_tolerance")):
        return "precision_contract_mismatch"
    if baseline.get("reference_sha256") != metadata.get("reference_sha256") or not metadata.get("reference_sha256"):
        return "reference_mismatch"
    if baseline.get("input_signature") != metadata.get("input_signature") or "input_signature" not in metadata:
        return "input_signature_mismatch"
    for result in (baseline, candidate):
        if result.get("metadata", {}).get("evaluation_config", {}).get("evaluator_sha256") != current_digest:
            return "stale_or_unknown_evaluator"
    return None


def bounded_gain(reference, candidate, target=None):
    denominator = max(math.log(reference / target), math.log(1.25)) if target else math.log(1.25)
    return max(0, min(1, math.log(reference / candidate) / denominator))


def bootstrap_ci(problem_scores, trials=1000):
    """Resample problems then declared sample slots, keeping missing slots as zeros."""
    if not problem_scores:
        return None
    rng = random.Random(42)
    draws = []
    for _ in range(trials):
        problems = [rng.choice(problem_scores) for _ in problem_scores]
        draws.append(mean(mean(rng.choice(slots) for _ in slots) for slots in problems))
    draws.sort()
    return [draws[int(0.025 * (trials - 1))], draws[int(0.975 * (trials - 1))]]


def measurement_evidence(result, candidate=False):
    metadata = result.get("metadata", {}) if candidate else result
    stats = result.get("runtime_stats", {}) if candidate else result
    receipt = metadata.get("timing_receipt", {})
    if not all(receipt.get(key) is True for key in
               ("output_consumed", "fresh_inputs", "varied_input_contents", "timed_outputs_verified")):
        return "timing_evidence_missing"
    samples = stats.get("samples_ms")
    if (not isinstance(samples, list) or not samples or len(samples) != stats.get("num_trials")
            or not all(map(valid_time, samples))):
        return "timing_samples_missing_or_invalid"
    observed = result.get("runtime") if candidate else result.get("mean")
    if not valid_time(observed) or not math.isclose(mean(samples), observed, rel_tol=1e-9, abs_tol=1e-12):
        return "timing_summary_does_not_match_samples"
    if len(receipt.get("input_seeds", [])) < len(samples):
        return "timing_input_seeds_missing"
    if not valid_time(metadata.get("cold_start_ms")):
        return "cold_cost_missing"
    return None


def current_result(result, payload):
    metadata = result.get("metadata", {})
    if metadata.get("workflow_signature") != fingerprint(payload):
        return "stale_or_incompatible_workflow"
    if metadata.get("evaluation_config") != payload["eval_config"]:
        return "evaluation_config_mismatch"
    if metadata.get("reference_sha256") != fingerprint({"source": payload["ref_arch_src"]}):
        return "reference_mismatch"
    if metadata.get("kernel_sha256") != fingerprint({"source": payload["kernel_src"]}):
        return "kernel_source_changed_or_missing"
    return None


def analyze_paper(config):
    from .engine import task_payload

    selected = select_problems(config.level, config.start, config.end)
    standard = read_json(config.run_dir / "eval_results.json", {})
    hidden = read_json(config.run_dir / "eval_results_hidden.json", {}) if config.hidden_tests else standard
    baselines = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {})
    baselines = baselines.get(f"level{config.level}", {})
    specs = read_json(Path(config.problem_specs), {}) if config.problem_specs else {}
    specs = specs.get(f"level{config.level}", {})
    digest = evaluator_digest()
    groups, rows, scored = {}, [], {}
    for pid, path in selected.items():
        layer = specs.get(str(pid), {}).get("layer", "unclassified")
        if layer not in {"I", "F", "D", "unclassified"}:
            raise ValueError(f"Unknown layer for {pid}: {layer}")
        baseline = baselines.get(path.name, {})
        slots = []
        for sid in range(config.num_samples):
            key = f"{pid}_{sid}"
            raw, secret = standard.get(key, {}), hidden.get(key, {})
            row = {"problem_id": pid, "sample_id": sid, "layer": layer, "score": 0}
            rows.append(row)
            slots.append(0)
            status = "verified"
            if not (raw.get("compiled") and raw.get("correctness") and secret.get("compiled") and secret.get("correctness")):
                status = "failed_or_missing"
            kernel = config.run_dir / kernel_name(config.level, pid, sid)
            if status == "verified" and not kernel.is_file():
                status = "kernel_source_changed_or_missing"
            if status == "verified":
                payload = task_payload(config, pid, sid)
                status = current_result(raw, payload) or status
            if status == "verified" and config.hidden_tests:
                try:
                    issue = current_result(secret, task_payload(config, pid, sid, hidden=True))
                except FileNotFoundError:
                    issue = "hidden_source_missing"
                if issue:
                    status = "stale_or_unknown_hidden_check"
            if status == "verified":
                status = compatible(baseline, raw, digest) or status
            if status == "verified":
                expected_baseline = fingerprint(task_payload(config, pid, task="baseline"))
                if baseline.get("metadata", {}).get("workflow_signature") != expected_baseline:
                    status = "stale_or_incompatible_baseline_workflow"
            if status == "verified":
                status = measurement_evidence(raw, candidate=True) or measurement_evidence(baseline) or status
            # A future audit-enabled result cannot claim its unfinished gates here.
            if status == "verified" and raw.get("metadata", {}).get("evaluation_config", {}).get("anti_cheat"):
                status = "anti_cheat_evidence_not_implemented_in_this_version"
            if status == "verified":
                options = baseline.get("variants", {"single": baseline})
                def current_variant(name, value):
                    expected = task_payload(config, pid, task="baseline")
                    if baseline.get("baseline_suite"):
                        expected["eval_config"]["baseline_variant"] = name
                    return value.get("metadata", {}).get("workflow_signature") == fingerprint(expected)

                cold_options = {
                    name: amortized(value["mean"], value["cold_start_ms"], config.cold_calls)
                    for name, value in options.items()
                    if compatible(value, raw, digest) is None and measurement_evidence(value) is None
                    and current_variant(name, value)
                }
                if not cold_options:
                    status = "validated_amortized_baseline_missing"
                else:
                    winner = min(cold_options, key=cold_options.get)
                    chosen = options[winner]
                    reference, runtime = cold_options[winner], amortized(
                        raw["runtime"], raw["metadata"]["cold_start_ms"], config.cold_calls)
                    b_ci = chosen.get("mean_ci95_ms")
                    c_ci = raw.get("runtime_stats", {}).get("mean_ci95_ms")
                    separated = (isinstance(b_ci, list) and isinstance(c_ci, list) and len(b_ci) == len(c_ci) == 2
                                 and chosen.get("num_trials", 0) >= 2 and raw.get("runtime_stats", {}).get("num_trials", 0) >= 2
                                 and all(map(valid_time, b_ci + c_ci)) and c_ci[1] < b_ci[0])
                    row.update(speedup=baseline["mean"] / raw["runtime"], cold_calls=config.cold_calls,
                               cold_winning_baseline=winner, amortized_speedup=reference / runtime,
                               significantly_faster=bool(separated), score_cost_policy="declared_K_amortized; cold_cost_point_estimate")
                    row["score"] = bounded_gain(reference, runtime) if separated else 0
                    slots[-1] = row["score"]
            row["status"] = status
            entry = {**raw, "metadata": dict(raw.get("metadata", {})), "correctness": status == "verified"}
            entry["metadata"].update(
                raw_correctness=raw.get("correctness", False), score_gate_status=status,
                score_gate_evaluator_sha256=digest, score_kernel_name=kernel.name,
                score_standard_sha256=fingerprint(raw), score_hidden_sha256=fingerprint(secret),
                score_baseline_sha256=fingerprint(baseline), score_hardware=config.hardware,
                score_baseline_name=config.baseline, score_reference_name=path.name, score_level=config.level,
                score_hidden_required=config.hidden_tests, score_policy="measurement_and_compatibility_v1",
                score_problem_id=pid, score_run_config_sha256=fingerprint(read_json(config.run_dir / "run_config.json", {})),
            )
            scored[key] = entry
        groups.setdefault(layer, []).append(slots)
    by_layer = {layer: mean(mean(slots) for slots in values) for layer, values in groups.items()}
    report = {
        "kind": "failure_inclusive_measurement_scoring_v1", "problem_ids": list(selected),
        "declared_samples": config.num_samples, "cold_calls": config.cold_calls, "hidden_gated": config.hidden_tests,
        "aggregation": "problem_mean_then_layer_macro; failures_and_missing_slots_zero; no_best_of_n",
        "bounded_gain_by_layer": by_layer, "bounded_gain_macro": mean(by_layer.values()),
        "layer_ci95": {layer: bootstrap_ci(values) for layer, values in groups.items()},
        "cross_device_comparison_certified": False, "anti_cheat_verified": False,
        "gate_scope": "current_sources_settings_measurement_evidence_device_and_software; not_H1_H14",
        "details": rows,
    }
    atomic_json(config.run_dir / "paper_analysis.json", report)
    atomic_json(config.run_dir / "eval_results_scored.json", scored)
    print(f"[Scoring] bounded gain macro={report['bounded_gain_macro']:.4f}; report: {config.run_dir / 'paper_analysis.json'}")
    return report
