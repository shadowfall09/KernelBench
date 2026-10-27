"""Preview or explicitly collect untimed CUDA origin diagnostics for a saved candidate."""

from __future__ import annotations

import argparse
import copy
import json
import secrets
from pathlib import Path

from kb_workflows.benchmarks.reference_population import select_run_problems
from kb_workflows.common import atomic_json, fingerprint
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.provenance.origin_task import read_scored


def prepare(config, problem, sample, *, minimum_custom_fraction=None, task="evaluate", hidden=False, variant=None, role="candidate",
            scored_result=None):
    if config.device_backend != "cuda":
        raise ValueError("CUDA origin diagnostics cannot substitute for Neuron binary attribution")
    if config.regrade_manifest:
        raise ValueError("Origin diagnostics are separate from frozen artifact regrading")
    selected = select_run_problems(config)
    if problem not in selected or not 0 <= sample < config.num_samples:
        raise ValueError("Requested candidate is outside the configured task/sample population")
    if task not in {"evaluate", "paired_timing"} or hidden and task != "evaluate":
        raise ValueError("Origin capture requires evaluation or public paired timing")
    if hidden and not config.hidden_tests:
        raise ValueError("Hidden origin capture requires configured final hidden verification")
    if task == "paired_timing" and variant is None:
        raise ValueError("Paired origin capture requires an explicit baseline variant")
    if role not in {"candidate", "reference"} or role == "reference" and task != "paired_timing":
        raise ValueError("Reference capture requires the explicit paired baseline")
    import math

    if minimum_custom_fraction is None:
        minimum_custom_fraction = config.origin_minimum_custom_fraction
    if (type(minimum_custom_fraction) not in (int, float) or not math.isfinite(minimum_custom_fraction)
            or not 0 < minimum_custom_fraction <= 1):
        raise ValueError("Minimum custom fraction must be in (0, 1]")
    payload = task_payload(config, problem, sample, task=task, hidden=hidden)
    if scored_result is not None:
        frozen = scored_result.get("metadata", {}).get("native_call_receipt", {}).get("native_request", {}).get("eval_config", {})
        if "final_protocol" in frozen:
            from kb_workflows.evaluation.final_verification import PROTOCOL, payload_for

            repeat = frozen.get("final_repeat")
            if frozen["final_protocol"] != PROTOCOL or type(repeat) is not int or not 0 <= repeat < config.final_repeats:
                raise ValueError("Scored final result has an invalid frozen repeat")
            payload = payload_for(config, problem, sample, repeat, hidden=hidden, task=task, variant=variant)
    if task == "paired_timing":
        from kb_workflows.evaluation.baseline_rewrites import plan_for

        plan_for(config.level, problem, payload["ref_arch_src"], variant)
        payload["eval_config"]["paired_baseline_variant"] = variant
    count = config.num_perf_trials + 3 if task == "evaluate" and not hidden else 0
    seeds = [secrets.randbits(63) for _ in range(count)] if scored_result is None else []
    payload.update(task="origin_audit", _origin_target=task, _origin_timing_seeds=seeds)
    if role == "reference":
        payload["_origin_role"] = role
    payload["eval_config"]["origin_minimum_custom_fraction"] = minimum_custom_fraction
    if scored_result is not None:
        from kb_workflows.provenance.origin_task import validate_scored

        frozen, _ = validate_scored(scored_result, payload)
        payload.update(_origin_timing_seeds=frozen["_origin_timing_seeds"], _scored_result=scored_result)
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=int)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--task", choices=("evaluate", "paired_timing"), default="evaluate")
    parser.add_argument("--hidden", action="store_true", help="Trusted final hidden task, never development feedback")
    parser.add_argument("--baseline-variant", help="Required for paired_timing; use the selected source-bound baseline variant")
    parser.add_argument("--role", choices=("candidate", "reference"), default="candidate")
    parser.add_argument("--scored-result", type=Path, help="Existing L1/L2 native result or result map; reuse its actual calls/seeds")
    parser.add_argument("--minimum-custom-fraction", type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = parse_config(["--config", str(args.config), "--stage", "evaluate", "--generation", "local"])
        scored = read_scored(args.scored_result, args.problem, args.sample, args.task)
        payload = prepare(config, args.problem, args.sample, minimum_custom_fraction=args.minimum_custom_fraction,
                          task=args.task, hidden=args.hidden, variant=args.baseline_variant, role=args.role, scored_result=scored)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Origin output must be a new path; saved evidence is not overwritten")
        preview = {"task": "origin_audit", "request_sha256": fingerprint(payload), "backend": config.evaluation,
                   "output": str(args.output.resolve()), "native_task": payload["_origin_target"],
                   "hidden": payload["eval_config"]["use_hidden_tests"], "timing_seed_count": len(payload["_origin_timing_seeds"]),
                   "experiments_started": False, "scored_wall_times_replaced": False,
                   "formal_qualification_implemented": False}
        preview["actual_scored_calls_requested"] = scored is not None
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
