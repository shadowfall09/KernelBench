"""Preview or explicitly collect vendor counters for one saved public task."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.engine import evaluate_tasks
from kb_workflows.provenance.counter_task import MODEL, artifact_path, execution_models, native_contract
from kb_workflows.provenance.origin_task import read_scored
from kb_workflows.provenance.work_counters import recipe
from scripts.audit_kernel_origin import configuration
from scripts.audit_kernel_origin import prepare as prepare_origin


def prepare(config, pid, sid, policy, *, target="candidate", task="evaluate", hidden=False, variant=None, scored_result=None):
    metrics = recipe(policy)
    payload = prepare_origin(config, pid, sid, task=task, hidden=hidden, variant=variant, role=target,
                             scored_result=scored_result)
    payload.update(task="counter_audit", _native_counter={"recipe": policy})
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload, sorted(metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--regrade-manifest", type=Path, help="Use the validated frozen verifier configuration")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=int)
    parser.add_argument("--sample", default=0, type=int)
    parser.add_argument("--target", choices=("candidate", "reference"), default="candidate")
    parser.add_argument("--task", choices=("evaluate", "paired_timing"), default="evaluate")
    parser.add_argument("--hidden", action="store_true", help="Trusted final hidden phase; never development feedback")
    parser.add_argument("--baseline-variant", help="Required for paired timing; the actual source-bound baseline")
    parser.add_argument("--scored-result", type=Path, help="Existing L1/L2 native result/map; bind counters to its actual call plan")
    parser.add_argument("--result-key", help="Exact slot/repeat/phase key in the scored result map")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to the scored receipt's canonical archive when --scored-result is supplied")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = configuration(args.config, args.regrade_manifest)
        scored = read_scored(args.scored_result, args.problem, args.sample, args.task, result_key=args.result_key)
        payload, metrics = prepare(config, args.problem, args.sample, read_json(args.recipe), target=args.target,
                                   task=args.task, hidden=args.hidden, variant=args.baseline_variant, scored_result=scored)
        canonical = artifact_path(config.run_dir, scored) if scored is not None else None
        if args.output is None:
            if canonical is None:
                raise ValueError("--output is required without an actual --scored-result")
            args.output = canonical
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Counter output must be a new path")
        preview = {"task": "counter_audit", "metrics": metrics, "target": args.target, "native_task": args.task,
                   "native_contract_sha256": native_contract(payload),
                   "request_sha256": fingerprint(payload), "backend": config.evaluation,
                   "calibration_sha256": fingerprint(read_json(args.calibration)) if args.calibration else None,
                   "output": str(args.output.resolve()), "experiments_started": False, "scored_wall_times_replaced": False}
        preview.update(hidden=args.hidden, formal_archive_path=str(canonical.resolve()) if canonical else None)
        preview["actual_scored_calls_requested"] = args.scored_result is not None
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        runtime = copy.deepcopy(config)
        runtime.runs_dir, runtime.run_name = str(args.output.resolve().parent), args.output.stem + "-counter-worker"
        runtime.run_dir.mkdir(parents=True, exist_ok=False)
        _, result = next(evaluate_tasks(runtime, [("counters", payload)]))
        if args.calibration and result.get("status") == "ok":
            try:
                result["execution_models"] = execution_models(result, read_json(args.calibration))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                result["execution_models"] = {"status": "not_measured", "reason": str(exc)}
        atomic_json(args.output, result)
        modeled = result.get("execution_models", {}).get("protocol") == MODEL
        model_status = "not_requested" if not args.calibration else ("measured" if modeled else "not_measured")
        print(json.dumps({**preview, "experiments_started": True, "result_status": result.get("status", "error"),
                          "execution_model_status": model_status}, indent=2))
        return 0 if result.get("status") == "ok" and (not args.calibration or modeled) else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
