"""Preview or explicitly fit per-task/device/dtype reference precision budgets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.precision_calibration import BUNDLE, CHECK_SEEDS, FIT, FIT_SEEDS
from kb_workflows.reference_population import select_run_problems


def prepare(config):
    if config.level not in {1, 2, 3} or config.precision_calibration:
        raise ValueError("Fit original L1/L2/L3 references without an existing precision calibration")
    tasks = []
    for pid in select_run_problems(config):
        payload = task_payload(config, pid, task="calibrate_precision")
        payload["eval_config"].update(measure_performance=False)
        payload["request_id"] = "kb-" + fingerprint(payload)
        tasks.append((f"precision-{pid}", payload))
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = parse_config(["--config", str(args.config), "--stage", "baseline", "--generation", "local"])
        tasks = prepare(config)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Precision calibration output must be a new path")
        preview = {"task": "calibrate_precision", "problems": [value["problem_id"] for _, value in tasks],
            "fit_seeds": FIT_SEEDS, "check_seeds": CHECK_SEEDS, "fit_method": FIT,
            "precision": config.precision, "oracle_memory_mb": config.oracle_memory_mb, "output": str(args.output.resolve()),
            "candidate_executed": False, "timings_collected": False, "experiments_started": False}
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        runtime = copy.deepcopy(config)
        runtime.runs_dir, runtime.run_name = str(args.output.resolve().parent), args.output.stem + "-precision-calibration"
        runtime.run_dir.mkdir(parents=True, exist_ok=False)
        document = {"protocol": BUNDLE, "records": {}, "errors": {}}
        for key, result in evaluate_tasks(runtime, tasks):
            if result.get("status") != "ok" or not isinstance(result.get("record"), dict):
                document["errors"][key] = result
                continue
            record = result["record"]
            document["records"][f"{record['level']}/{record['problem_id']}/{record['precision']}"] = record
        atomic_json(args.output, document)
        print(json.dumps({**preview, "experiments_started": True,
                          "qualified": sum(row.get("qualified") is True for row in document["records"].values()),
                          "failed_workers": len(document["errors"])}, indent=2))
        return 0 if not document["errors"] and all(row["qualified"] for row in document["records"].values()) else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
