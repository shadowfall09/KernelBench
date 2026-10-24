"""Preview or explicitly collect vendor counters for one saved public task."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.reference_population import select_run_problems
from kb_workflows.work_counters import MODEL, execution_model, recipe


def prepare(config, pid, sid, policy, *, target="candidate", seed=42):
    metrics = recipe(policy)
    if (config.device_backend != "cuda" or config.regrade_manifest or target not in {"candidate", "reference"}
            or pid not in select_run_problems(config) or not 0 <= sid < config.num_samples
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Counter diagnostic needs one selected public CUDA task/sample and uint32 seed")
    payload = task_payload(config, pid, sid, task="counter_audit")
    payload["eval_config"].update(counter_recipe=policy, counter_target=target, counter_seed=seed, measure_performance=False)
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload, sorted(metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=int)
    parser.add_argument("--sample", default=0, type=int)
    parser.add_argument("--target", choices=("candidate", "reference"), default="candidate")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = parse_config(["--config", str(args.config), "--stage", "evaluate", "--generation", "local"])
        payload, metrics = prepare(config, args.problem, args.sample, read_json(args.recipe), target=args.target, seed=args.seed)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Counter output must be a new path")
        preview = {"task": "counter_audit", "metrics": metrics, "target": args.target, "seed": args.seed,
                   "request_sha256": fingerprint(payload), "backend": config.evaluation,
                   "calibration_sha256": fingerprint(read_json(args.calibration)) if args.calibration else None,
                   "output": str(args.output.resolve()), "experiments_started": False, "scored_wall_times_replaced": False}
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        runtime = copy.deepcopy(config)
        runtime.runs_dir, runtime.run_name = str(args.output.resolve().parent), args.output.stem + "-counter-worker"
        runtime.run_dir.mkdir(parents=True, exist_ok=False)
        _, result = next(evaluate_tasks(runtime, [("counters", payload)]))
        if args.calibration and result.get("status") == "ok":
            try:
                result["execution_model"] = execution_model(result["counters"], read_json(args.calibration))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                result["execution_model"] = {"status": "not_measured", "reason": str(exc)}
        atomic_json(args.output, result)
        modeled = result.get("execution_model", {}).get("protocol") == MODEL
        model_status = "not_requested" if not args.calibration else ("measured" if modeled else "not_measured")
        print(json.dumps({**preview, "experiments_started": True, "result_status": result.get("status", "error"),
                          "execution_model_status": model_status}, indent=2))
        return 0 if result.get("status") == "ok" and (not args.calibration or modeled) else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
