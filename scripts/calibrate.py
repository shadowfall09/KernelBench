"""Device, input and precision calibration through one explicit entrypoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from kb_workflows.benchmarks.reference_population import select_run_problems
from kb_workflows.common import atomic_json, fingerprint, select_problems
from kb_workflows.config import parse_config
from kb_workflows.engine import evaluate_tasks, task_payload
from kb_workflows.numerics import input_calibration, precision_calibration
from kb_workflows.numerics.input_profiles import recipes


def prepare(config, kind):
    if kind not in {"inputs", "precision"} or config.level not in {1, 2, 3} or config.precision_calibration or kind == "inputs" and config.input_calibration:
        raise ValueError("Calibrate original L1/L2/L3 references without an existing conflicting calibration")
    selected = (select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role)
                if kind == "inputs" else select_run_problems(config))
    tasks = []
    for pid in selected:
        payload = task_payload(config, pid, task="calibrate_" + kind)
        payload["eval_config"]["measure_performance"] = False
        payload["request_id"] = "kb-" + fingerprint(payload)
        tasks.append((f"{kind}-{pid}", payload))
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="kind", required=True)
    for kind in ("device", "inputs", "precision"):
        sub = modes.add_parser(kind)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--execute", action="store_true")
        if kind == "device":
            sub.add_argument("--backend", choices=("cuda", "neuron"), default="cuda")
            sub.add_argument("--device", help="Defaults to the selected backend's device 0")
            sub.add_argument("--matrix-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
            sub.add_argument("--memory-mib", type=int, default=256)
            sub.add_argument("--trials", type=int, default=15)
        else:
            sub.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.kind == "device":
            args.device = args.device or args.backend + ":0"
            prefix, _, ordinal = args.device.partition(":")
            if prefix != args.backend:
                raise ValueError("Device name and backend must agree")
            if not args.execute:
                print(json.dumps({**vars(args), "output": str(args.output), "experiments_started": False}, indent=2))
                return 0
            from kb_workflows.hardware.devices import adapter

            result = adapter(args.backend, ordinal or "0").calibrate(matrix_sizes=args.matrix_sizes,
                memory_bytes=args.memory_mib * 1024**2, trials=args.trials)
            atomic_json(args.output, result)
            print(args.output)
            return 0
        config = parse_config(["--config", str(args.config), "--stage", "baseline", "--generation", "local"])
        tasks = prepare(config, args.kind)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Calibration output must be a new path")
        preview = {"task": "calibrate_" + args.kind, "problems": [p["problem_id"] for _, p in tasks],
                   "oracle_memory_mb": config.oracle_memory_mb, "output": str(args.output.resolve()),
                   "candidate_executed": False, "timings_collected": False, "experiments_started": False}
        module = input_calibration if args.kind == "inputs" else precision_calibration
        if args.kind == "inputs":
            preview.update(recipes={p["problem_id"]: recipes(p["problem_id"], p["level"]) for _, p in tasks}, seeds=module.SEEDS)
        else:
            preview.update(fit_seeds=module.FIT_SEEDS, check_seeds=module.CHECK_SEEDS, fit_method=module.FIT, precision=config.precision)
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        runtime = copy.deepcopy(config)
        runtime.runs_dir, runtime.run_name = str(args.output.resolve().parent), args.output.stem + "-" + args.kind + "-calibration"
        runtime.run_dir.mkdir(parents=True, exist_ok=False)
        document = {"protocol": module.BUNDLE, "records": {}, "errors": {}}
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
