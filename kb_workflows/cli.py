from __future__ import annotations

import json
import sys
from contextlib import nullcontext

from .common import ROOT, atomic_json, read_json, select_problems
from .config import parse_config


def main(argv=None):
    try:
        config = parse_config(argv)
        selected = select_problems(config.level, config.start, config.end)
        if config.hidden_tests and config.stage in {"all", "evaluate"}:
            for pid in selected:
                path = ROOT / "hidden_tests" / f"level{config.level}" / f"{pid}_hidden.py"
                if not path.is_file():
                    raise ValueError(f"Hidden tests missing: {path}")
        if config.dry_run:
            print(
                json.dumps(
                    {
                        "config": config.to_dict(),
                        "problems": list(selected),
                        "evaluation_phases": ["standard", "hidden"] if config.hidden_tests else ["standard"],
                        "run_dir": str(config.run_dir),
                    },
                    indent=2,
                )
            )
            return 0
        if config.stage in {"all", "analyze", "baseline"} and not config.hardware:
            raise ValueError("Specify --hardware for the actual verifier GPU to select/record its baseline")
        from .engine import analyze_run, evaluate_run, record_baseline
        from .generation import build_docker_image, generate_run

        needs_image = (config.generation == "docker" and config.stage in {"all", "generate"}) or (
            config.evaluation == "docker" and config.stage in {"all", "evaluate", "baseline"}
        )
        if needs_image and config.build_image:
            build_docker_image(config)
        cloud = (config.evaluation == "modal" and config.stage in {"all", "evaluate", "baseline"}) or (
            config.generation == "modal" and config.stage in {"all", "generate"}
        )
        if cloud:
            from .modal_backend import modal_session
        context = modal_session(config) if cloud and config.stage != "analyze" else nullcontext()
        config.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config.run_dir / "run_config.json", config.to_dict())
        generation_ok = True
        with context:
            baseline_path = ROOT / "results" / "timing" / config.hardware / f"{config.baseline}.json"
            baseline_data = read_json(baseline_path, {}).get(f"level{config.level}", {})
            missing_baseline = any(
                not isinstance(baseline_data.get(path.name), dict)
                or not (baseline_data[path.name].get("mean") or 0) > 0
                for path in selected.values()
            )
            if config.stage == "baseline" or config.record_baseline or (config.stage == "all" and missing_baseline):
                print("[baseline] Recording the requested problems on the verifier hardware", flush=True)
                record_baseline(config)
            if config.stage in {"all", "generate"}:
                generation_ok = generate_run(config)
            if config.stage in {"all", "evaluate"}:
                evaluate_run(config)
            if config.stage in {"all", "analyze"}:
                analyze_run(config)
        print(f"Results: {config.run_dir}", flush=True)
        return 0 if generation_ok else 1
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
