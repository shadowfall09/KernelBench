"""Experiment labels and resource receipts, independent of agent SDK imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import atomic_json, kernel_name, read_json, select_problems
from .suites import selection_receipt


def protocol(config):
    external = config.generation == "harbor"
    return {
        "version": "kernelbench-pro-agent-v1",
        "mode": "external" if external else config.generation_mode,
        "scaffold": f"harbor/{config.harbor_agent}" if external else "mini-swe-agent==2.4.6",
        "model_backend": "harbor" if external else config.agent_backend,
        "model": config.model,
        "samples_per_problem": config.num_samples,
        "candidate_limit_per_sample": None if external else (1 if config.generation_mode == "one-shot" else config.optimization_rounds),
        "wall_limit_seconds_per_sample": config.agent_timeout,
        "estimated_cost_limit_usd_per_sample": config.agent_cost_limit if not external else None,
        "development_feedback": "none" if config.generation_mode == "one-shot" else ("external" if external else "standard_only"),
        "profiling": config.agent_profile if not external and config.generation_mode == "feedback" else False,
        "held_out_feedback": False if not external else None,
        "selection": "single_candidate" if config.generation_mode == "one-shot" else ("external" if external else "best_correct_development_mean"),
    }


def write_budget_report(config, summary):
    ids = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role)
    slots = [summary.get(f"{pid}_{sid}", {}) for pid in ids for sid in range(config.num_samples)]
    history = read_json(config.run_dir / "generation_attempts.json", {})
    attempts = history.get("slots", {})
    receipts = [attempt.get("budget") for values in attempts.values() for attempt in values] if attempts else [slot.get("budget") for slot in slots]
    active_protocol = protocol(config)
    if any(len(values) > 1 for values in attempts.values()):
        active_protocol = {**active_protocol, "base_mode": active_protocol["mode"], "mode": "repeated-attempts",
                           "attempt_counts": {key: len(values) for key, values in attempts.items()}}
    fields = ("model_calls", "candidate_submissions", "development_evaluations", "profile_requests", "input_tokens", "output_tokens", "estimated_cost_usd")
    totals = {}
    for field in fields:
        values = [item.get(field) if isinstance(item, dict) else None for item in receipts]
        totals[field] = sum(values) if values and all(isinstance(v, (int, float)) for v in values) else None
    if history.get("historical_attempts_unknown"):
        totals = {key: None for key in totals}
    report = {"protocol": active_protocol, "suite": selection_receipt(config), "declared_slots": len(slots), "all_slots_have_receipts": all(isinstance(v, dict) for v in receipts),
              "complete_resource_receipts": all(value is not None for value in totals.values()),
              "unknown_fields": [key for key, value in totals.items() if value is None],
              "attempt_history": history,
              "totals": totals, "slots": {f"{pid}_{sid}": summary.get(f"{pid}_{sid}", {}).get("budget")
                                            for pid in ids
                                            for sid in range(config.num_samples)},
              "kernel_sha256": {f"{pid}_{sid}": summary.get(f"{pid}_{sid}", {}).get("kernel_sha256")
                                for pid in ids for sid in range(config.num_samples)},
              "observed_models": sorted({model for item in receipts if isinstance(item, dict) for model in item.get("models") or []}) or None,
              "gpu_execution_seconds": None, "gpu_time_note": "Request wall time includes transport/queue/compile; GPU-only billing is not measured",
              "final_evaluation_budget_included": False}
    atomic_json(config.run_dir / "generation_budget.json", report)
    return report


def compare_runs(directories):
    """Group reports only when the declared workload, hardware and resource limits match."""
    from .config import RunConfig

    groups = {}
    for directory in map(Path, directories):
        settings = read_json(directory / "run_config.json")
        config = RunConfig(**settings)
        config.runs_dir, config.run_name = str(directory.resolve().parent), directory.name
        budget = existing_budget(config)
        scoring = read_json(directory / "paper_analysis.json", {})
        declared = dict(budget.get("protocol", {"mode": "unattributed"}))
        declared.pop("model", None)
        contract = {"protocol": declared,
                    "suite": selection_receipt(config),
                    "layer_assignments": {str(row["problem_id"]): row.get("layer") for row in scoring.get("details", [])},
                    "hardware": config.hardware, "precision": config.precision, "baseline": config.baseline,
                    "level": config.level, "problems": list(select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role)),
                    "hidden_tests": config.hidden_tests, "num_correct_trials": config.num_correct_trials,
                    "num_perf_trials": config.num_perf_trials, "cold_calls": config.cold_calls,
                    "fp32_tolerance": config.fp32_tolerance}
        key = json.dumps(contract, sort_keys=True)
        entry = groups.setdefault(key, {"contract": contract, "runs": []})
        entry["runs"].append({"run": directory.name, "declared_model": config.model, "observed_models": budget.get("observed_models"),
                              "bounded_gain_macro": scoring.get("bounded_gain_macro"),
                              "generation_totals": budget.get("totals"), "resource_receipts_complete": budget.get("complete_resource_receipts", False),
                              "provenance_valid": budget.get("provenance_valid", False)})
    return {"comparison_policy": "separate_by_protocol_budget_model_workload_precision_and_hardware; no_cross_group_ranking",
            "groups": list(groups.values())}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare native run reports without pooling different experiment settings")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.runs), indent=2, ensure_ascii=False))


def existing_budget(config):
    report = read_json(config.run_dir / "generation_budget.json", {})
    if not report:
        return {}
    current = {}
    for pid in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role):
        for sid in range(config.num_samples):
            path = config.run_dir / kernel_name(config.level, pid, sid)
            current[f"{pid}_{sid}"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    expected = report.get("kernel_sha256", {})
    report["provenance_valid"] = all(key in expected and expected[key] == value for key, value in current.items())
    if not report["provenance_valid"]:
        report["declared_protocol"] = report.get("protocol")
        report["protocol"] = {"mode": "unattributed", "reason": "Generation receipts do not match current candidate sources"}
    return report
