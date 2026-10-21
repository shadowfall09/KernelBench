from __future__ import annotations

import json
import shutil
import sys
import time
from contextlib import nullcontext

from .common import ROOT, atomic_json, evaluator_digest, read_json, select_problems
from .config import parse_config
from .suites import selection_receipt


def main(argv=None):
    try:
        config = parse_config(argv)
        selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role)
        strict_coverage = None
        if config.strict_validation:
            from .contract_specs import coverage

            strict_coverage = coverage(config.level, selected)
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
                        "suite": selection_receipt(config),
                        "strict_coverage": strict_coverage,
                        "evaluation_phases": ["standard", "hidden"] if config.hidden_tests else ["standard"],
                        "run_dir": str(config.run_dir),
                    },
                    indent=2,
                )
            )
            return 0
        if (strict_coverage and strict_coverage["unsupported"]
                and config.stage in {"all", "generate", "evaluate", "export-harbor"}):
            raise ValueError(f"Strict contracts unavailable for level {config.level} IDs {strict_coverage['unsupported']}; choose reviewed tasks")
        if config.stage != "analyze":
            config.require_gpu_target()
        if config.stage == "export-harbor":
            from .harbor import export_tasks

            print(f"Harbor tasks: {export_tasks(config)}", flush=True)
            return 0
        if config.generation == "harbor" and config.stage in {"all", "generate"} and not shutil.which(config.harbor_command):
            raise ValueError("Harbor executable not found; install the harbor extra or set --harbor-command")
        if config.generation == "local" and config.stage in {"all", "generate"}:
            import importlib.util

            if importlib.util.find_spec("minisweagent") is None:
                raise ValueError("mini-swe-agent is missing; install the project requirements")
            if config.agent_backend == "claude-code" and not shutil.which("claude"):
                raise ValueError("Claude Code CLI is missing; install it or select --agent-backend litellm")
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
        phases = {}

        def phase(name, function):
            started = time.monotonic()
            try:
                return function(config)
            finally:
                phases[name] = time.monotonic() - started
                atomic_json(config.run_dir / "phase_budget.json", {"phase_wall_seconds": phases,
                            "includes": "phase request wait, compilation and execution; excludes image build and environment startup",
                            "gpu_execution_seconds": None, "cloud_billing_certified": False})
        generation_ok = True
        with context:
            baseline_path = ROOT / "results" / "timing" / config.hardware / f"{config.baseline}.json"
            baseline_data = read_json(baseline_path, {}).get(f"level{config.level}", {})
            missing_baseline = any(
                not isinstance(baseline_data.get(path.name), dict)
                or not (baseline_data[path.name].get("mean") or 0) > 0
                or baseline_data[path.name].get("metadata", {}).get("evaluation_config", {}).get("evaluator_sha256") != evaluator_digest()
                or (config.baseline_suite and not baseline_data[path.name].get("suite_complete"))
                for path in selected.values()
            )
            if config.stage == "baseline" or config.record_baseline or (config.stage == "all" and missing_baseline):
                print("[baseline] Recording the requested problems on the verifier hardware", flush=True)
                phase("baseline", record_baseline)
            if config.stage in {"all", "generate"}:
                generation_ok = phase("generation", generate_run)
            if config.stage in {"all", "evaluate"}:
                phase("final_evaluation", evaluate_run)
            if config.stage in {"all", "analyze"}:
                phase("analysis", analyze_run)
        print(f"Results: {config.run_dir}", flush=True)
        return 0 if generation_ok else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
