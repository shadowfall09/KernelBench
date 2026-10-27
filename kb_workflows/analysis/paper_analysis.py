"""Failure-inclusive measurement scoring alongside the original Verified metrics."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from statistics import mean

from ..benchmarks.reference_population import select_run_problems
from ..benchmarks.suites import selection_receipt, suite_rows
from ..common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name, read_json
from ..evaluation.evidence import score_issue
from ..evaluation.measurement import amortized

DEVICE_FIELDS = ("name", "capability", "total_memory", "multiprocessors", "l2_bytes", "torch_version", "cuda_version", "hip_version")


def same_device_class(first, second):
    return bool(first.get("available") and second.get("available")) and all(
        name in first and name in second and first[name] == second[name] for name in DEVICE_FIELDS
    ) and first.get("driver_version") == second.get("driver_version")


def valid_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def compatible(baseline, candidate, current_digest):
    from ..evaluation.baseline_rewrites import artifact_issue
    from ..runtime.isolation import runtime_issue

    metadata = candidate.get("metadata", {})
    if reason := runtime_issue(baseline, candidate, metadata.get("evaluation_config", {})):
        return reason
    if baseline.get("baseline_correctness") is not True or (baseline.get("baseline_suite") and not baseline.get("suite_complete")):
        return "baseline_not_validated_or_incomplete"
    if reason := artifact_issue(baseline):
        return reason
    if not same_device_class(baseline.get("device_fingerprint", {}), metadata.get("device_fingerprint", {})):
        return "device_or_software_mismatch"
    if baseline.get("timing_method") != "synchronized_wall" or metadata.get("timing_method") != "synchronized_wall":
        return "timing_protocol_mismatch"
    if (metadata.get("evaluation_config", {}).get("evidence_level") in {"L1", "L2"}
            or metadata.get("origin_capture") or baseline.get("metadata", {}).get("origin_capture")):
        from ..provenance.origin_task import timing_observer

        try:
            if timing_observer(baseline, baseline=True) != timing_observer(candidate):
                return "origin_timing_mode_mismatch"
        except (OSError, ValueError):
            return "origin_timing_observer_missing_or_invalid"
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
    options = result.get("metadata", {}).get("evaluation_config", {})
    if options.get("numerical_validation"):
        # Candidate/current-result and baseline gates additionally validate the
        # complete source-bound payload; this keeps legacy metric imports gated.
        if not isinstance(result.get("metadata", {}).get("timed_numerical_audit"), dict):
            return "timed_numerical_evidence_missing"
    metadata = result.get("metadata", {}) if candidate else result
    if metadata.get("simulated_execution") is True or result.get("metadata", {}).get("simulated_execution") is True:
        return "simulated_device_execution_cannot_qualify"
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
    from ..evaluation.tool_checks import issue
    from ..runtime.isolation import issue as isolation_issue

    metadata = result.get("metadata", {})
    if metadata.get("workflow_signature") != fingerprint(payload):
        return "stale_or_incompatible_workflow"
    if metadata.get("evaluation_config") != payload["eval_config"]:
        return "evaluation_config_mismatch"
    if metadata.get("reference_sha256") != fingerprint({"source": payload["ref_arch_src"]}):
        return "reference_mismatch"
    if metadata.get("kernel_sha256") != fingerprint({"source": payload["kernel_src"]}):
        return "kernel_source_changed_or_missing"
    if reason := isolation_issue(result, payload):
        return reason
    if reason := issue(result, payload):
        return reason
    return None


def analyze_paper(config):
    from ..benchmarks.reference_population import freeze_run

    freeze_run(config)
    from ..agents.resource_budgets import active
    from ..agents.resource_budgets import score_issue as resource_score_issue
    from ..engine import task_payload
    from ..evaluation.aggregation import sensitivity
    from .experiment import existing_budget, protocol

    selected = select_run_problems(config)
    standard = read_json(config.run_dir / "eval_results.json", {})
    hidden = read_json(config.run_dir / "eval_results_hidden.json", {}) if config.hidden_tests else standard
    baselines = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {})
    baselines = baselines.get(f"level{config.level}", {})
    specs = read_json(Path(config.problem_specs), {}) if config.problem_specs else {}
    specs = specs.get(f"level{config.level}", {})
    calibration = read_json(Path(config.device_calibration), {}) if config.device_calibration else {}
    research_rows = suite_rows(config.level, config.suite_role, suite=config.suite) if config.suite != "full" else {}
    digest = evaluator_digest()
    groups, bounded_groups, track_groups, rows, scored, certificate_coverage = {}, {}, {}, [], {}, {}
    external_files = {option: fingerprint(read_json(Path(getattr(config, option))))
                      for option in ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file", "regrade_manifest") if getattr(config, option)}
    auxiliary_files = {}
    for name in ((["final_manifest.json", "final_results.json"] if config.final_repeats else [])
                 + (["paired_results.json"] if config.paired_timing else [])
                 + (["generation_budget.json", "generation_attempts.json", "generation_summary.json"] if active(config) else [])):
        auxiliary_files[name] = fingerprint(read_json(config.run_dir / name, {}))
    for pid, path in selected.items():
        layer = specs.get(str(pid), {}).get("layer", research_rows.get(pid, {}).get("layer", "unclassified"))
        bound_record = specs.get(str(pid), {}).get("headroom_certificate", {})
        layer_source = ("problem_specs" if "layer" in specs.get(str(pid), {}) else
                        ("provisional_suite_taxonomy" if pid in research_rows else "unclassified"))
        if layer not in {"I", "F", "D", "unclassified"}:
            raise ValueError(f"Unknown layer for {pid}: {layer}")
        baseline = baselines.get(path.name, {})
        task_track = research_rows.get(pid, {}).get("track", "forward")
        bound_issue = None
        if config.score_mode == "headroom":
            from .headroom import certificate

            try:
                info = certificate(bound_record, calibration, baseline,
                                   source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), digest=digest)
                layer, layer_source = info["layer"], "reviewed_headroom_certificate"
                certificate_coverage[str(pid)] = {"valid": True, "certificate_sha256": info["certificate_sha256"]}
            except (KeyError, TypeError, ValueError) as exc:
                bound_issue = str(exc)
                layer, layer_source = "unclassified", "invalid_headroom_certificate"
                certificate_coverage[str(pid)] = {"valid": False, "reason": bound_issue}
        slots, bounded_slots = [], []
        for sid in range(config.num_samples):
            key = f"{pid}_{sid}"
            raw, secret = standard.get(key, {}), hidden.get(key, {})
            row = {"problem_id": pid, "sample_id": sid, "layer": layer, "track": task_track, "layer_source": layer_source, "score": 0}
            rows.append(row)
            slots.append(0)
            bounded_slots.append(0)
            status = "verified"
            if not (raw.get("compiled") and raw.get("correctness") and secret.get("compiled") and secret.get("correctness")):
                status = "failed_or_missing"
            kernel = config.run_dir / kernel_name(config.level, pid, sid)
            if status == "verified" and not kernel.is_file():
                status = "kernel_source_changed_or_missing"
            if status == "verified":
                status = resource_score_issue(config, pid, sid) or status
            if status == "verified" and config.regrade_manifest:
                from ..runtime.regrade import issue as regrade_issue

                status = regrade_issue(config) or status
            if status == "verified":
                if bound_issue is not None:
                    status = "headroom_certificate_missing_or_invalid"
                    row["headroom_error"] = bound_issue
            if status == "verified":
                try:
                    payload = task_payload(config, pid, sid)
                    status = current_result(raw, payload) or status
                except (ValueError, KeyError, OSError):
                    status = "reference_or_input_family_contract_changed"
            if status == "verified" and config.hidden_tests:
                try:
                    issue = current_result(secret, task_payload(config, pid, sid, hidden=True))
                    if issue is None:
                        from ..runtime.isolation import runtime_issue

                        issue = runtime_issue(raw, secret, payload["eval_config"])
                except FileNotFoundError:
                    issue = "hidden_source_missing"
                if issue:
                    status = "stale_or_unknown_hidden_check"
            if status == "verified" and config.throughput_validation:
                contrast = raw.get("metadata", {}).get("tool_checks", {}).get("checks", {}).get("throughput_contrast", {})
                row["throughput_contrast"] = contrast.get("values")
                if contrast.get("status") != "passed":
                    status = "throughput_diagnostics_missing_or_invalid"
                elif contrast.get("values", {}).get("review_required") is True:
                    status = "throughput_discrepancy_requires_review"
            if status == "verified":
                status = compatible(baseline, raw, digest) or status
            if status == "verified":
                from ..evaluation.baseline_rewrites import baseline_issue as recipe_baseline_issue
                from ..evaluation.timed_numerics import baseline_issue as timed_baseline_issue
                from ..numerics.input_calibration import baseline_issue as calibrated_baseline_issue
                from ..runtime.isolation import baseline_issue

                status = (baseline_issue(config, baseline, pid) or calibrated_baseline_issue(config, baseline, pid)
                          or recipe_baseline_issue(config, baseline, pid) or timed_baseline_issue(config, baseline, pid) or status)
            if status == "verified":
                expected_baseline = fingerprint(task_payload(config, pid, task="baseline"))
                if baseline.get("metadata", {}).get("workflow_signature") != expected_baseline:
                    status = "stale_or_incompatible_baseline_workflow"
            if status == "verified":
                status = measurement_evidence(raw, candidate=True) or measurement_evidence(baseline) or status
            final_results = []
            measurement = {}
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
                    winner = (min(cold_options, key=lambda name: options[name]["mean"])
                              if config.score_mode == "headroom" else min(cold_options, key=cold_options.get))
                    chosen = options[winner]
                    paired_variant = winner if baseline.get("baseline_suite") else chosen.get("baseline_variant", "eager")
                    paired = None
                    if config.paired_timing:
                        from ..evaluation.paired_task import validate_for_score

                        issue, paired = validate_for_score(config, pid, sid, paired_variant, chosen)
                        if issue:
                            status = issue
                    if status == "verified":
                        from ..evaluation.evidence import measurement_checks, receipt

                        measurement = measurement_checks(config, pid, sid, chosen, variant=paired_variant, paired=paired)
                        attribution = measurement.get("launch_attribution", {})
                        if attribution.get("values") is not None:
                            row["launch_attribution"] = attribution["values"]
                            if attribution["values"]["launch_only"]:
                                status = "launch_only"
                            elif attribution["values"]["mixed_repeat_classification"]:
                                status = "launch_attribution_inconsistent"
                        options = task_payload(config, pid, sid)["eval_config"]
                        evidence_issue = score_issue(raw, options, "standard", additional=measurement)
                        if config.hidden_tests:
                            evidence_issue = score_issue(secret, options, "hidden", additional=measurement) or evidence_issue
                        if status == "verified":
                            status = evidence_issue or status
                        if measurement:
                            row["measurement_checks"] = measurement
                            row["assessed_evidence"] = receipt({**raw.get("metadata", {}).get("evidence_receipt", {}).get("checks", {}),
                                                               **measurement}, requested=config.evidence_level)
                    if status == "verified" and config.final_repeats:
                        from ..evaluation.final_verification import validate

                        issue, final_results = validate(config, pid, sid, chosen, measurement=measurement)
                        status = issue or status
                    # Final repeats use every declared process, never their fastest member.
                    final_stats = None
                    if final_results:
                        from ..evaluation.final_verification import timing_summary

                        final_stats = timing_summary(final_results)
                    candidate_steady = final_stats["mean"] if final_stats else raw["runtime"]
                    baseline_steady = chosen["mean"]
                    if paired:
                        candidate_steady, baseline_steady = paired["candidate_mean_ms"], paired["reference_mean_ms"]
                    reference, runtime = cold_options[winner], amortized(
                        candidate_steady, raw["metadata"]["cold_start_ms"], config.cold_calls)
                    if paired:
                        reference = amortized(baseline_steady, chosen["cold_start_ms"], config.cold_calls)
                    b_ci = chosen.get("mean_ci95_ms")
                    c_stats = final_stats or raw.get("runtime_stats", {})
                    c_ci = c_stats.get("mean_ci95_ms")
                    separated = (isinstance(b_ci, list) and isinstance(c_ci, list) and len(b_ci) == len(c_ci) == 2
                                 and chosen.get("num_trials", 0) >= 2 and c_stats.get("num_trials", 0) >= 2
                                 and all(map(valid_time, b_ci + c_ci)) and c_ci[1] < b_ci[0])
                    if paired:
                        separated = paired["significantly_faster"]
                        row.update(paired_speedup_ci95=paired["speedup_ci95"], noise_ci95=paired["reference_control_ci95"],
                                   paired_trial_count=paired["num_pairs"], paired_repeat_groups=paired["independent_repeat_groups"])
                    if final_results:
                        row.update(final_remeasurement_count=len(final_results), final_runtime_mean_ms=candidate_steady)
                    row.update(speedup=baseline["mean"] / raw["runtime"], cold_calls=config.cold_calls,
                               cold_winning_baseline=min(cold_options, key=cold_options.get), selected_baseline=winner,
                               baseline_selection="minimum_steady_mean" if config.score_mode == "headroom" else "minimum_K_amortized",
                               amortized_speedup=reference / runtime,
                               significantly_faster=bool(separated), score_cost_policy="declared_K_amortized; cold_cost_point_estimate")
                    row["bounded_gain"] = bounded_gain(reference, runtime) if separated and status == "verified" else 0
                    bounded_slots[-1] = row["bounded_gain"]
                    row["score"] = row["bounded_gain"]
                    if config.score_mode == "headroom" and status == "verified":
                        from .headroom import evaluate

                        try:
                            bound = evaluate(bound_record, calibration, raw["metadata"],
                                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                reference_ms=baseline_steady, candidate_ms=candidate_steady, digest=digest)
                            row.update(headroom=bound, headroom_score=bound["score"] if separated else 0,
                                       layer_source="reviewed_headroom_certificate",
                                       score_cost_policy="headroom_steady_state; amortized_bounded_gain_reported_separately")
                            row["score"] = row["headroom_score"]
                        except (KeyError, TypeError, ValueError) as exc:
                            status = "headroom_certificate_missing_or_invalid"
                            row.update(headroom_error=str(exc), score=0, headroom_score=0)
                    slots[-1] = row["score"]
            gpu_files = {}
            if config.generation == "harbor" and config.agent_output_token_limit:
                from ..runtime.harbor_budget import quota_artifact_hashes

                try:
                    summary = read_json(config.run_dir / "generation_summary.json")[key]
                    gpu_files = quota_artifact_hashes(config.run_dir, pid, sid, summary, level=config.level)
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    if status == "verified":
                        status = "resource_budget_harbor_output_evidence_missing_or_invalid"
                        row["score"], slots[-1], bounded_slots[-1] = 0, 0, 0
            if config.agent_gpu_busy_seconds:
                from ..agents.resource_budgets import gpu_artifact_hashes

                try:
                    snapshot = read_json(config.run_dir / "generation_budget.json")["slots"][key]["resource_budgets"]["gpu_busy_quota"]
                    gpu_files = gpu_artifact_hashes(config.run_dir, pid, sid, snapshot)
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    if status == "verified":
                        status = "resource_budget_gpu_evidence_changed_or_missing"
                        row["score"], slots[-1], bounded_slots[-1] = 0, 0, 0
            row["status"] = status
            entry = {**raw, "metadata": dict(raw.get("metadata", {})), "correctness": status == "verified"}
            if "assessed_evidence" in row:
                entry["metadata"].update(evidence_receipt=row["assessed_evidence"],
                    worker_evidence_receipt_sha256=fingerprint(raw.get("metadata", {}).get("evidence_receipt", {})),
                    score_measurement_checks=measurement)
            entry["metadata"].update(
                raw_correctness=raw.get("correctness", False), score_gate_status=status,
                score_gate_evaluator_sha256=digest, score_kernel_name=kernel.name,
                score_standard_sha256=fingerprint(raw), score_hidden_sha256=fingerprint(secret),
                score_baseline_sha256=fingerprint(baseline), score_hardware=config.hardware,
                score_baseline_name=config.baseline, score_reference_name=path.name, score_level=config.level,
                score_hidden_required=config.hidden_tests, score_policy="measurement_and_compatibility_v1",
                score_problem_id=pid, score_run_config_sha256=fingerprint(read_json(config.run_dir / "run_config.json", {})),
                score_auxiliary_files={**auxiliary_files, **gpu_files},
                score_external_files=external_files,
                score_regrade_measurements_sha256=fingerprint(read_json(config.run_dir / "regrade_measurements.json", {})) if config.regrade_manifest else None,
            )
            scored[key] = entry
        groups.setdefault(layer, []).append(slots)
        track_groups.setdefault(task_track, {}).setdefault(layer, []).append(slots)
        bounded_groups.setdefault(layer, []).append(bounded_slots)
    by_layer = {layer: mean(mean(slots) for slots in values) for layer, values in groups.items()}
    label_source = ("problem_specs_with_suite_fallback" if config.problem_specs and research_rows else
                    ("problem_specs" if config.problem_specs else ("provisional_suite_taxonomy" if research_rows else "unclassified")))
    if config.score_mode == "headroom":
        label_source = "reviewed_headroom_certificates; invalid_models_unclassified"
    report = {
        "kind": "failure_inclusive_measurement_scoring_v1", "problem_ids": list(selected),
        "suite": selection_receipt(config), "leaderboard_eligible": config.suite_role == "scored" and
            (config.suite != "production41" or config.task_track != "all") and
            (config.score_mode != "headroom" or all(item["valid"] for item in certificate_coverage.values())),
        "track_comparison_policy": "separate_tracks; all_track_collection_macro_is_diagnostic",
        "score_by_track": {track: {"by_layer": {layer: mean(mean(slots) for slots in values) for layer, values in layers.items()},
                                  "macro": mean(mean(mean(slots) for slots in values) for values in layers.values())}
                           for track, layers in track_groups.items()},
        "headroom_certificate_coverage": certificate_coverage,
        "layer_source": label_source,
        "headroom_score_implemented": True, "score_mode": config.score_mode,
        "bounded_gain_baseline_policy": "same_selected_variant; K_optimized_in_bounded_gain_mode; steady_optimized_in_headroom_mode",
        "strict_validation": config.strict_validation, "requested_evidence_level": config.evidence_level,
        "kernel_track": config.kernel_track, "origin_minimum_custom_fraction": config.origin_minimum_custom_fraction,
        "experiment": existing_budget(config).get("protocol", protocol(config)), "generation_budget": existing_budget(config),
        "generation_receipts_available": bool(existing_budget(config)),
        "declared_samples": config.num_samples, "cold_calls": config.cold_calls, "hidden_gated": config.hidden_tests,
        "aggregation": "problem_mean_then_layer_macro; failures_and_missing_slots_zero; no_best_of_n",
        "score_by_layer": by_layer, "score_macro": mean(by_layer.values()),
        "bounded_gain_by_layer": {layer: mean(mean(slots) for slots in values) for layer, values in bounded_groups.items()},
        "bounded_gain_macro": mean(mean(mean(slots) for slots in values) for values in bounded_groups.values()),
        "headroom_by_layer": by_layer if config.score_mode == "headroom" else None,
        "headroom_macro": mean(by_layer.values()) if config.score_mode == "headroom" else None,
        "layer_ci95": {layer: bootstrap_ci(values) for layer, values in groups.items()},
        "cross_device_comparison_certified": False, "anti_cheat_verified": False,
        "gate_scope": "current_sources_settings_measurement_evidence_device_and_software; not_H1_H14",
        "details": rows,
    }
    report["aggregation_sensitivity"] = sensitivity(report)
    atomic_json(config.run_dir / "paper_analysis.json", report)
    atomic_json(config.run_dir / "eval_results_scored.json", scored)
    print(f"[Scoring] {config.score_mode} macro={report['score_macro']:.4f}; report: {config.run_dir / 'paper_analysis.json'}")
    return report
