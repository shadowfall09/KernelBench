"""Preview or explicitly calibrate reference inputs; never execute a candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint, select_problems
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.input_calibration import BUNDLE, SEEDS
from kb_workflows.input_profiles import recipes


def prepare(config):
    if config.level not in {1, 2, 3} or config.input_calibration:
        raise ValueError("Calibrate original L1/L2/L3 inputs without an existing input calibration file")
    selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role)
    tasks = []
    for pid in selected:
        payload = task_payload(config, pid, task="calibrate_inputs")
        payload["eval_config"].update(measure_performance=False)
        payload["request_id"] = "kb-" + fingerprint(payload)
        tasks.append((f"inputs-{pid}", payload))
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
            raise ValueError("Input calibration output must be a new path")
        preview = {"task": "calibrate_inputs", "problems": [payload["problem_id"] for _, payload in tasks],
            "recipes": {payload["problem_id"]: recipes(payload["problem_id"], payload["level"]) for _, payload in tasks},
            "seeds": SEEDS, "oracle_memory_mb": config.oracle_memory_mb, "output": str(args.output.resolve()),
            "candidate_executed": False, "timings_collected": False, "experiments_started": False}
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        runtime = copy.deepcopy(config)
        runtime.runs_dir, runtime.run_name = str(args.output.resolve().parent), args.output.stem + "-input-calibration"
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
