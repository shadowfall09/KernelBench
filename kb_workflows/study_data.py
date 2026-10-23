"""Export fresh, evidence-gated run scores for offline research analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .common import ROOT, evaluator_digest, fingerprint, read_json, select_problems
from .config import RunConfig
from .experiment import existing_budget, protocol
from .study_statistics import finite

PROTOCOL = "kernelbench_pro_study_snapshot_v1"
METRICS = {"score", "bounded_gain", "headroom_score", "qualified_correctness", "qualified_fast_p"}
RESOURCE_FIELDS = ("model_calls", "candidate_submissions", "input_tokens", "output_tokens", "total_tokens",
                   "estimated_cost_usd", "generation_wall_seconds", "development_request_wall_seconds")


def study_digest():
    directory = Path(__file__).parent
    return fingerprint({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                        for name in ("study_data.py", "study_statistics.py", "study_plan.py", "budget_curves.py")})


def evaluation_contract(config):
    from .isolation import options
    from .tool_checks import diagnostic_options

    fields = ("hardware", "gpu_arch", "device_backend", "precision", "fp32_tolerance", "baseline", "baseline_suite", "baseline_portfolio",
              "hidden_tests", "num_correct_trials", "num_perf_trials", "strict_validation", "evidence_level",
              "cold_calls", "paired_timing", "paired_trials", "final_repeats", "score_mode", "suite", "suite_role",
              "task_track", "task_profile", "task_memory_mb", "num_samples", "evaluation", "timeout",
              "sanitizer_tools", "sanitizer_command", "sanitizer_timeout", "determinism_checks", "determinism_repeats")
    result = {key: getattr(config, key) for key in fields}
    result.update(diagnostic_options(config))
    result["verifier_isolation"] = options(config)
    result["evaluator_sha256"] = evaluator_digest()
    path = ROOT / "results/timing" / config.hardware / f"{config.baseline}.json"
    result["baseline_dataset_sha256"] = fingerprint(read_json(path, {}))
    result["external_files"] = {key: fingerprint(read_json(Path(getattr(config, key)), {}))
                                for key in ("problem_specs", "device_calibration", "input_calibration", "task_dimensions", "determinism_policy_file", "regrade_manifest") if getattr(config, key)}
    return result


def row_value(row, metric, speedup_threshold):
    if row.get("status") != "verified":
        return 0.0
    if metric == "qualified_correctness":
        return 1.0
    if metric == "qualified_fast_p":
        return float(row.get("significantly_faster") is True and finite(row.get("amortized_speedup"))
                     and row["amortized_speedup"] > speedup_threshold)
    value = row.get(metric)
    if not finite(value) or not 0 <= value <= 1:
        raise ValueError(f"A verified slot lacks the requested bounded metric: {metric}")
    return value


def collect_run(directory, *, session, label=None, metric="score", speedup_threshold=1.0):
    """Recompute analysis only. This never calls generation or evaluation runners."""
    from .paper_analysis import analyze_paper

    if (metric not in METRICS or not finite(speedup_threshold) or speedup_threshold < 1
            or not isinstance(session, str) or not session):
        raise ValueError("Unknown metric, threshold, or session label")
    directory = Path(directory).resolve()
    saved = read_json(directory / "run_config.json")
    if not isinstance(saved, dict):
        raise ValueError(f"Missing run configuration: {directory}")
    config = RunConfig(**saved)
    config.runs_dir, config.run_name = str(directory.parent), directory.name
    config.stage = "analyze"
    config.validate()
    report = analyze_paper(config)
    selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role,
                               track=config.task_track)
    if report.get("problem_ids") != list(selected) or report.get("declared_samples") != config.num_samples:
        raise ValueError("Analysis population does not match the declared run")
    budget = existing_budget(config)
    observed = budget.get("observed_models") or []
    model = observed[0] if len(observed) == 1 else config.model
    declared_protocol = dict(budget.get("protocol") or protocol(config))
    declared_protocol.pop("model", None)
    # Retry multiplicity remains explicit evidence and actual cost, not a per-task method name.
    attempts = declared_protocol.pop("attempt_counts", None)
    tasks, seen, layers, identities = {}, set(), {}, {}
    for pid, path in selected.items():
        key = f"{config.level}/{pid}"
        tasks[key] = [0.0] * config.num_samples
        identities[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in report.get("details", []):
        pid, sid = row.get("problem_id"), row.get("sample_id")
        key = f"{config.level}/{pid}"
        if key not in tasks or type(sid) is not int or not 0 <= sid < config.num_samples or (pid, sid) in seen:
            raise ValueError("Duplicate or out-of-population study score slot")
        seen.add((pid, sid))
        layer = row.get("layer", "unclassified")
        if key in layers and layers[key] != layer:
            raise ValueError("One task cannot change layers across samples")
        layers[key] = layer
        tasks[key][sid] = row_value(row, metric, speedup_threshold)
    for key in tasks:
        layers.setdefault(key, "unclassified")
    eligible = (bool(model) and len(observed) <= 1 and budget.get("provenance_valid") is True
                and budget.get("all_slots_have_receipts") is True and report.get("leaderboard_eligible") is True
                and declared_protocol.get("mode") not in {None, "unattributed", "interactive"}
                and (declared_protocol.get("mode") != "external" or budget.get("external_receipts_valid") is True))
    record = {"protocol": PROTOCOL, "directory": str(directory), "label": label or model or directory.name,
              "analysis_implementation_sha256": study_digest(),
              "model_identity": model or f"unattributed:{directory.name}", "observed_models": observed,
              "model_identity_source": "provider_receipt" if len(observed) == 1 else "declared_configuration",
              "session": session, "metric": metric, "speedup_threshold": speedup_threshold,
              "evaluation_contract": evaluation_contract(config), "generation_protocol": declared_protocol,
              "retry_attempt_counts": attempts, "tasks": tasks, "layers": layers, "reference_sha256": identities,
              "resource_totals": {key: budget.get("totals", {}).get(key) for key in RESOURCE_FIELDS},
              "declared_slots": len(tasks) * config.num_samples, "formal_comparison_eligible": eligible,
              "qualification": "current_sources_settings_baseline_and_evidence_recomputed_without_workload_execution",
              "source_receipts": {name: fingerprint(read_json(directory / name, {})) for name in
                  ("run_config.json", "paper_analysis.json", "eval_results_scored.json", "generation_budget.json")},
              "budget_unknown_fields": budget.get("unknown_fields"), "experiments_started_by_export": False}
    record["receipt_sha256"] = fingerprint(record)
    return record


def validate_record(record):
    if (record.get("protocol") != PROTOCOL or record.get("receipt_sha256") !=
            fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})):
        raise ValueError("Study record is malformed or changed after export")
    if not record.get("tasks") or set(record["tasks"]) != set(record.get("reference_sha256", {})):
        raise ValueError("Study record is missing declared tasks or source identities")
    if set(record["tasks"]) != set(record.get("layers", {})) or not isinstance(record.get("session"), str) or not record["session"]:
        raise ValueError("Study record is missing strata or session identity")
    for slots in record["tasks"].values():
        if not isinstance(slots, list) or not slots or any(not finite(value) or not 0 <= value <= 1 for value in slots):
            raise ValueError("Invalid declared score slots")
    if record.get("declared_slots") != sum(len(slots) for slots in record["tasks"].values()):
        raise ValueError("Study slot denominator changed")


def assert_session_identity(record, seen):
    receipt = record.get("source_receipts", {}).get("generation_budget.json")
    if receipt:
        identity = (record["model_identity"], receipt, tuple(sorted(record["tasks"])))
        if identity in seen and seen[identity] != record["session"]:
            raise ValueError("The same generation evidence cannot establish independent sessions")
        seen[identity] = record["session"]


def discrimination_groups(records, *, audit_only=False):
    """Combine disjoint levels/ranges into sessions; never take a best run."""
    groups, evidence_sessions = {}, {}
    for record in records:
        validate_record(record)
        if not audit_only and not record.get("formal_comparison_eligible"):
            raise ValueError("Unattributed or incomplete generation evidence; use audit-only for exploratory diagnostics")
        contract = {"evaluation": record["evaluation_contract"], "generation": record["generation_protocol"],
                    "metric": record["metric"], "speedup_threshold": record["speedup_threshold"]}
        assert_session_identity(record, evidence_sessions)
        group = groups.setdefault(fingerprint(contract), {"contract": contract, "subjects": {}, "layers": {},
                                                         "reference_sha256": {}, "source_receipts": [],
                                                         "formal_comparison_eligible": True})
        group["formal_comparison_eligible"] &= record["formal_comparison_eligible"]
        subject = group["subjects"].setdefault(record["model_identity"], {"sessions": {}, "label": record["label"]})
        session = subject["sessions"].setdefault(record["session"], {})
        if set(session) & set(record["tasks"]):
            raise ValueError("Duplicate task within a subject/session; repeated runs cannot become best-of-N")
        session.update(record["tasks"])
        for key, value in record["reference_sha256"].items():
            if key in group["reference_sha256"] and group["reference_sha256"][key] != value:
                raise ValueError("Subjects reference different source snapshots")
            group["reference_sha256"][key] = value
            if key in group["layers"] and group["layers"][key] != record["layers"][key]:
                raise ValueError("Subjects use incompatible task strata")
            group["layers"][key] = record["layers"][key]
        group["source_receipts"].append(record["receipt_sha256"])
    return list(groups.values())


def collect_manifest(path, *, metric="score", speedup_threshold=1.0):
    path = Path(path).resolve()
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list) or not document["runs"]:
        raise ValueError("Study manifest needs a nonempty runs array")
    records, seen = [], set()
    for entry in document["runs"]:
        directory = Path(entry["path"])
        if not directory.is_absolute():
            directory = path.parent / directory
        directory = directory.resolve()
        if directory in seen:
            raise ValueError("A study cannot reuse one run directory as multiple independent sessions")
        seen.add(directory)
        records.append(collect_run(directory, session=entry["session"], label=entry.get("label"),
                                   metric=metric, speedup_threshold=speedup_threshold))
    return records
