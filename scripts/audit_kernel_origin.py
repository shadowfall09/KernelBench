"""Preview or explicitly collect untimed CUDA origin diagnostics for a saved candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.reference_population import select_run_problems


def prepare(config, problem, sample, *, seeds, minimum_custom_fraction):
    if config.device_backend != "cuda":
        raise ValueError("CUDA origin diagnostics cannot substitute for Neuron binary attribution")
    if config.regrade_manifest:
        raise ValueError("Origin diagnostics are separate from frozen artifact regrading")
    selected = select_run_problems(config)
    if problem not in selected or not 0 <= sample < config.num_samples:
        raise ValueError("Requested candidate is outside the configured task/sample population")
    if (not seeds or len(seeds) > 100 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Seeds must be 1..100 distinct uint32 values")
    import math

    if (type(minimum_custom_fraction) not in (int, float) or not math.isfinite(minimum_custom_fraction)
            or not 0 < minimum_custom_fraction <= 1):
        raise ValueError("Minimum custom fraction must be in (0, 1]")
    payload = task_payload(config, problem, sample, task="origin_audit")
    payload["eval_config"].update(measure_performance=False, origin_seeds=seeds,
                                  origin_minimum_custom_fraction=minimum_custom_fraction)
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=int)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--minimum-custom-fraction", type=float, default=0.95)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = parse_config(["--config", str(args.config), "--stage", "evaluate", "--generation", "local"])
        payload = prepare(config, args.problem, args.sample, seeds=[int(part) for part in args.seeds.split(",")],
                          minimum_custom_fraction=args.minimum_custom_fraction)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Origin output must be a new path; saved evidence is not overwritten")
        preview = {"task": "origin_audit", "request_sha256": fingerprint(payload), "backend": config.evaluation,
                   "output": str(args.output.resolve()), "input_seeds": payload["eval_config"]["origin_seeds"],
                   "experiments_started": False, "scored_wall_times_replaced": False,
                   "formal_qualification_implemented": False}
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        # Do not write logs/config/scores into the source experiment directory.
        diagnostic_config = copy.deepcopy(config)
        diagnostic_config.runs_dir = str(args.output.resolve().parent)
        diagnostic_config.run_name = args.output.stem + "-origin-worker"
        logs = diagnostic_config.run_dir
        logs.mkdir(parents=True, exist_ok=False)
        _, result = next(evaluate_tasks(diagnostic_config, [("origin", payload)]))
        atomic_json(args.output, result)
        print(json.dumps({**preview, "experiments_started": True, "result_status": result.get("status", "error")}, indent=2))
        return 0 if result.get("status") == "ok" else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
