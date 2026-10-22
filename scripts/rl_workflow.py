"""Plan GRPO/regrading/held-out jobs and analyze receipts without starting experiments by default."""

import argparse
import json
from pathlib import Path

from kb_workflows.common import atomic_json, read_json
from kb_workflows.rl_audit import analyze
from kb_workflows.rl_evaluation import execute_regrade, heldout_plan, make_regrade_plan
from kb_workflows.rl_plan import make_plan, read_plan, write_plan
from kb_workflows.rl_rewards import prepare_baselines
from kb_workflows.rl_training import execute_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    planning = actions.add_parser("plan")
    planning.add_argument("--spec", type=Path, required=True)
    planning.add_argument("--output-dir", type=Path, required=True)
    train = actions.add_parser("train")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--execute", action="store_true")
    train.add_argument("--resume-checkpoint", type=Path)
    baselines = actions.add_parser("baselines")
    baselines.add_argument("--training-plan", type=Path, required=True)
    baselines.add_argument("--split", choices=("train", "heldout"), default="train")
    baselines.add_argument("--execute", action="store_true")
    regrade_plan = actions.add_parser("regrade-plan")
    regrade_plan.add_argument("--training-plan", type=Path, required=True)
    regrade_plan.add_argument("--spec", type=Path, required=True)
    regrade_plan.add_argument("--output-dir", type=Path, required=True)
    regrade = actions.add_parser("regrade")
    regrade.add_argument("--training-plan", type=Path, required=True)
    regrade.add_argument("--plan", type=Path, required=True)
    regrade.add_argument("--execute", action="store_true")
    regrade.add_argument("--resume", action="store_true")
    heldout = actions.add_parser("heldout-plan")
    heldout.add_argument("--training-plan", type=Path, required=True)
    heldout.add_argument("--checkpoints", type=Path, required=True)
    heldout.add_argument("--output-dir", type=Path, required=True)
    heldout.add_argument("--sessions", type=int, default=3)
    heldout.add_argument("--samples", type=int, default=1)
    audit = actions.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    experiment = actions.add_parser("experiment-plan")
    experiment.add_argument("--spec", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, required=True)
    train_experiment = actions.add_parser("experiment-train")
    train_experiment.add_argument("--experiment", type=Path, required=True)
    train_experiment.add_argument("--execute", action="store_true")
    train_experiment.add_argument("--keep-going", action="store_true")
    evaluate_experiment = actions.add_parser("experiment-evaluation-plan")
    evaluate_experiment.add_argument("--experiment", type=Path, required=True)
    evaluate_experiment.add_argument("--output-dir", type=Path, required=True)
    evaluate_experiment.add_argument("--audit-only", action="store_true")
    analyze_experiment = actions.add_parser("experiment-analyze")
    analyze_experiment.add_argument("--plan", type=Path, required=True)
    analyze_experiment.add_argument("--output", type=Path, required=True)
    analyze_experiment.add_argument("--bootstrap-trials", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan = make_plan(read_json(args.spec), directory=args.output_dir / "training", base_directory=args.spec.resolve().parent)
        write_plan(plan, args.output_dir / "plan.json")
        result = {"plan": str(args.output_dir / "plan.json"), **execute_plan(args.output_dir / "plan.json")}
    elif args.action == "train":
        result = execute_plan(args.plan, execute=args.execute, resume=args.resume_checkpoint)
    elif args.action == "baselines":
        result = prepare_baselines(read_plan(args.training_plan), split=args.split, execute=args.execute)
    elif args.action == "regrade-plan":
        plan = make_regrade_plan(read_plan(args.training_plan), read_json(args.spec), directory=args.output_dir / "regrades",
                                 base_directory=args.spec.resolve().parent)
        atomic_json(args.output_dir / "plan.json", plan)
        result = {"plan": str(args.output_dir / "plan.json"), "experiments_started": False}
    elif args.action == "regrade":
        result = execute_regrade(read_plan(args.training_plan), read_json(args.plan), execute=args.execute, resume=args.resume)
    elif args.action == "heldout-plan":
        from kb_workflows.study_plan import write_plan as write_study

        plan = heldout_plan(read_plan(args.training_plan), read_json(args.checkpoints), directory=args.output_dir,
                            sessions=args.sessions, samples=args.samples)
        write_study(plan, args.output_dir)
        result = {"plan": str(args.output_dir / "plan.json"), "job_count": len(plan["jobs"]), "experiments_started": False}
    elif args.action.startswith("experiment-"):
        from kb_workflows import rl_effectiveness

        if args.action == "experiment-plan":
            result = rl_effectiveness.make_experiment(read_json(args.spec), directory=args.output_dir,
                                                     base_directory=args.spec.resolve().parent)
            rl_effectiveness.write_experiment(result, args.output_dir / "experiment.json")
            result = {"experiment": str(args.output_dir / "experiment.json"),
                      **rl_effectiveness.execute_training(result, execute=False)}
        elif args.action == "experiment-train":
            result = rl_effectiveness.execute_training(read_json(args.experiment), execute=args.execute, keep_going=args.keep_going)
        elif args.action == "experiment-evaluation-plan":
            result = rl_effectiveness.evaluation_plan(read_json(args.experiment), directory=args.output_dir, audit_only=args.audit_only)
            if result.get("status") != "not_ready":
                from kb_workflows.study_plan import write_plan as write_study

                write_study(result, args.output_dir)
                result = {"plan": str(args.output_dir / "plan.json"), "experiments_started": False}
        else:
            result = rl_effectiveness.analyze(read_json(args.plan), trials=args.bootstrap_trials)
            atomic_json(args.output, result)
    else:
        result = analyze(read_json(args.manifest))
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
