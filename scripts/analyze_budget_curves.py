"""Compute score/budget curves from saved independent protocol points."""

import argparse
from pathlib import Path

from kb_workflows.analysis.budget_curves import AXES, VARY, budget_curves, plot_curves
from kb_workflows.analysis.study_data import METRICS, collect_manifest
from kb_workflows.common import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--axis", choices=sorted(AXES), default="declared_wall_seconds")
    parser.add_argument("--vary", choices=sorted(VARY), default="wall")
    parser.add_argument("--metric", choices=sorted(METRICS), default="score")
    parser.add_argument("--speedup-threshold", type=float, default=1.0)
    parser.add_argument("--aggregation", choices=("task_mean", "layer_macro"), default="task_mean")
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    records = collect_manifest(args.manifest, metric=args.metric, speedup_threshold=args.speedup_threshold)
    report = budget_curves(records, axis=args.axis, vary=args.vary, aggregation=args.aggregation,
                           trials=args.bootstrap_trials, seed=args.seed, audit_only=args.audit_only)
    report["source_records"] = records
    atomic_json(args.output, report)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_curves(report, args.plot)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
