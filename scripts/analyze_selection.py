"""Analyze current saved scores; never regenerate kernels or run a workload."""

import argparse
from pathlib import Path

from kb_workflows.analysis.study_data import METRICS, collect_manifest, discrimination_groups, study_digest
from kb_workflows.analysis.study_statistics import paired_binary_design, selection_analysis
from kb_workflows.benchmarks.suites import load_manifest
from kb_workflows.common import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--selection-suite", choices=("kernelbench105", "production41"))
    selection.add_argument("--selection", type=Path, help="JSON array of level/problem_id objects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=sorted(METRICS), default="score")
    parser.add_argument("--speedup-threshold", type=float, default=1.0)
    parser.add_argument("--aggregation", choices=("task_mean", "layer_macro"), default="task_mean")
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-partial-selection", action="store_true")
    parser.add_argument("--assumed-discordance", type=float, help="Optional paired-binary normal design assumption, not a measured power result")
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.assumed_discordance is not None and args.metric not in {"qualified_correctness", "qualified_fast_p"}:
        raise ValueError("Paired binary power assumptions require a binary metric")
    records = collect_manifest(args.manifest, metric=args.metric, speedup_threshold=args.speedup_threshold)
    rows = ([row for row in load_manifest(args.selection_suite)["problems"] if row["role"] == "scored"]
            if args.selection_suite else read_json(args.selection))
    selected = [f"{row['level']}/{row['problem_id']}" for row in rows]
    if len(set(selected)) != len(selected):
        raise ValueError("Selection contains duplicate task IDs")
    output = []
    for group in discrimination_groups(records, audit_only=args.audit_only):
        missing = sorted(set(selected) - set(group["layers"]))
        if missing and not args.allow_partial_selection:
            raise ValueError(f"Study does not contain the full requested selection: {missing}")
        present = [key for key in selected if key in group["layers"]]
        if len(group["subjects"]) < 2:
            analysis = {"status": "not_computed", "reason": "At least two compatible subjects are needed"}
        else:
            analysis = selection_analysis(group["subjects"], group["layers"], present, aggregation=args.aggregation,
                                          trials=args.bootstrap_trials, seed=args.seed)
        if args.assumed_discordance is not None:
            analysis["power_design_assumption"] = {
                name: paired_binary_design(size, args.assumed_discordance, alpha=args.alpha, power=args.target_power)
                if size >= 2 else {"reason": "At least two tasks required"}
                for name, size in (("full", len(group["layers"])), ("selected", len(present)))}
        output.append({**group, "analysis": analysis, "missing_selection_tasks": missing,
                       "selection_is_partial": bool(missing)})
    atomic_json(args.output, {"groups": output, "records": records, "experiments_started": False,
                              "analysis_implementation_sha256": study_digest(),
                              "historical_seven_model_ranking_reproduced": False})
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
