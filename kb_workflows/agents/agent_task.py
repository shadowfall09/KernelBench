"""Interactive coding-agent bridge. Development feedback and final submission are separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

from ..analysis.experiment import protocol
from ..benchmarks.reference_population import select_run_problems
from ..common import atomic_json, kernel_name, read_json
from ..config import RunConfig, parse_config
from ..evaluation.evidence import origin_feedback
from .resource_budgets import active


def require_interactive_budget(config):
    if active(config):
        raise ValueError("Interactive sessions have unobserved calls; resource reservations require the native controller")


def verify_directory(directory, *, evaluator=None):
    """Used by a current Claude Code/Codex session, without recursively launching another agent."""
    from .agent_loop import development_evaluate

    directory = Path(directory).resolve()
    config = RunConfig(**read_json(directory / "job_config.json")).validate()
    require_interactive_budget(config)
    if config.generation_mode == "one-shot":
        print("One-shot mode does not allow development feedback")
        return 2
    state = read_json(directory / "interactive_state.json", {})
    now = time.time()
    state.setdefault("started_at", now)
    state.setdefault("rounds", [])
    if state.get("frozen") or len(state["rounds"]) >= config.optimization_rounds or now - state["started_at"] >= config.agent_timeout:
        print("Development budget exhausted or submission frozen; finalize the retained candidate")
        return 2
    output = config.run_dir / kernel_name(config.level, config.start)
    if not output.is_file() or output.is_symlink() or not output.stat().st_size:
        print(f"Write a nonempty ModelNew implementation to {output}")
        return 2
    index = len(state["rounds"]) + 1
    round_dir = directory / "rounds" / f"{index:03}"
    round_dir.mkdir(parents=True, exist_ok=True)
    source = output.read_text()
    (round_dir / "candidate.py").write_text(source)
    state["rounds"].append({"round": index, "status": "started"})
    atomic_json(directory / "interactive_state.json", state)

    def remaining():
        limit = config.agent_timeout - (time.time() - state["started_at"])
        if limit <= 0:
            raise TimeoutError("Development wall budget exhausted")
        return limit

    run = evaluator or development_evaluate
    if config.evaluation == "modal":
        from ..runtime.modal_backend import modal_session

        context = modal_session(config)
    else:
        context = nullcontext()
    with context:
        result = run(config, config.start, index, remaining)
    atomic_json(round_dir / "standard.json", result)
    runtime = result.get("runtime", -1)
    good = (result.get("compiled") is True and result.get("correctness") is True
            and type(runtime) in (int, float) and math.isfinite(runtime) and runtime > 0)
    row = {"round": index, "correctness": result.get("compiled") is True and result.get("correctness") is True,
           "runtime_ms": runtime if good else None,
           "error": str(result.get("metadata", {}).get("error", ""))[:2000], "feedback_scope": "standard_only"}
    origin = origin_feedback(result, config.to_dict())
    if origin is not None:
        row["kernel_origin"] = origin
        good = good and origin["status"] == "passed"
    row["performance_eligible"] = good
    if good and (not state.get("best") or runtime < state["best"]["runtime_ms"]):
        state["best"] = {"round": index, "runtime_ms": runtime, "source": source}
    if good and config.agent_profile:
        try:
            if config.evaluation == "modal":
                from ..runtime.modal_backend import modal_session

                with modal_session(config):
                    row["profile"] = run(config, config.start, index, remaining, task="profile")
            else:
                row["profile"] = run(config, config.start, index, remaining, task="profile")
        except Exception as exc:
            row["profile"] = {"status": "unavailable", "error": type(exc).__name__}
    state["rounds"][-1] = row
    atomic_json(directory / "interactive_state.json", state)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0 if good else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare/finalize one task for the coding agent in the current session")
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--directory", type=Path)
    args, options = parser.parse_known_args(argv)
    if args.action == "prepare":
        from .generation import prepare_job

        config = parse_config(options)
        require_interactive_budget(config)
        config.require_gpu_target()
        selected = select_run_problems(config)
        if config.start != config.end or config.num_samples != 1 or list(selected) != [config.start]:
            raise ValueError("Interactive preparation requires --start N --end N --num-samples 1")
        if config.generation != "local" or config.evaluation not in {"local", "docker", "queue", "modal"}:
            raise ValueError("Interactive preparation requires --generation local")
        directory = prepare_job(config, config.start, 0)
        atomic_json(directory / "interactive_state.json", {"started_at": time.time(), "rounds": [], "frozen": False,
                                                           "destination": str(config.run_dir / kernel_name(config.level, config.start))})
        print(json.dumps({"directory": str(directory), "reference": str(directory / "model.py"),
                          "candidate": str(directory / "runs/agent" / kernel_name(config.level, config.start)),
                          "feedback": f"cd {directory} && python verify.py"}, indent=2))
        return 0
    if options or args.directory is None:
        parser.error("finalize requires --directory and no run options")
    directory = args.directory.resolve()
    state = read_json(directory / "interactive_state.json")
    config = RunConfig(**read_json(directory / "job_config.json")).validate()
    require_interactive_budget(config)
    if state.get("frozen"):
        raise ValueError("Submission is already frozen")
    source = (config.run_dir / kernel_name(config.level, config.start)).read_text() if config.generation_mode == "one-shot" else state.get("best", {}).get("source")
    if not source:
        raise ValueError("No development-correct candidate to finalize")
    destination = Path(state["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    state["frozen"] = True
    atomic_json(directory / "interactive_state.json", state)
    # This session has unobserved model/tool calls. Never label it as measured one-shot or mini-swe-agent.
    interactive_protocol = {**protocol(config), "mode": "interactive", "scaffold": "current_coding_session",
                            "model_backend": "current_session", "model": "", "development_mode": config.generation_mode,
                            "estimated_cost_limit_usd_per_sample": None}
    atomic_json(destination.parent / "generation_budget.json", {"protocol": interactive_protocol,
                "model_calls": None, "estimated_cost_usd": None, "development_evaluations": len(state["rounds"]),
                "kernel_sha256": {f"{config.start}_0": hashlib.sha256(source.encode()).hexdigest()},
                "generation_wall_seconds": time.time() - state["started_at"], "final_hidden_tests_run": False})
    print(f"Frozen candidate: {destination}; run the native evaluate/analyze stages for final results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
