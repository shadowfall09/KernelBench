"""Bounded mini-swe-agent loop with controller-owned development feedback."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from contextlib import nullcontext
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from ..analysis.experiment import protocol
from ..common import atomic_json, fingerprint, kernel_name, read_json
from ..config import RunConfig
from .agent_models import KernelModel
from .resource_budgets import BudgetExhausted, Ledger, device_lease, settle_device_lease


def development_evaluate(config, pid, index, remaining, task="evaluate", *, lease=None):
    """All feedback tasks are standard-only; final hidden checks have no route here."""
    from ..engine import evaluate_tasks, task_payload

    limit = min(config.timeout, remaining(), lease["execution_timeout_seconds"]) if lease else max(1, min(config.timeout, int(remaining())))
    if limit <= 0:
        raise BudgetExhausted("Development execution window expired before dispatch")
    child = replace(config, hidden_tests=False, timeout=limit,
                    queue_wait_timeout=limit, queue_http_timeout=min(config.queue_http_timeout, limit))
    payload = task_payload(child, pid, task=task)
    if lease:
        payload["eval_config"]["device_lease"] = lease
    return next(iter(evaluate_tasks(child, [(f"development-{index}-{task}", payload)])))[1]


class KernelEnvironment:
    def __init__(self, directory, config, pid, remaining, evaluator=development_evaluate):
        self.directory, self.config, self.pid = Path(directory), config, pid
        self.remaining, self.evaluator = remaining, evaluator
        self.output = config.run_dir / kernel_name(config.level, pid)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.unlink(missing_ok=True)
        self.rounds = []
        self.best = None
        self.limit = 1 if config.generation_mode == "one-shot" else config.optimization_rounds
        self.development_seconds = 0.0
        self.device_budget = Ledger(config.agent_device_seconds, "device_lease_seconds") if config.agent_device_seconds else None
        self.requests = {"evaluate": 0, "profile": 0}

    def evaluate(self, index, task="evaluate"):
        lease = device_lease(self.device_budget, self.config.timeout, self.remaining()) if self.device_budget else None
        self.requests[task] += 1
        started = time.monotonic()
        try:
            result = self.evaluator(self.config, self.pid, index, self.remaining, task=task,
                                    **({"lease": lease} if lease else {}))
        except Exception:
            if lease:
                self.device_budget.settle(lease["nonce"])
            raise
        finally:
            self.development_seconds += time.monotonic() - started
        if lease and not settle_device_lease(self.device_budget, lease, result):
            atomic_json(self.directory / "rounds" / f"{index:03}" / f"{task}_resource_failure.json", result)
            raise BudgetExhausted("Missing or exceeded parent-owned device lease receipt")
        return result

    def execute(self, action):
        from minisweagent.exceptions import Submitted

        self.remaining()
        if len(self.rounds) >= self.limit:
            raise RuntimeError("Candidate budget exceeded")
        index = len(self.rounds) + 1
        row = {"round": index, "feedback_scope": "none" if self.config.generation_mode == "one-shot" else "standard_only"}
        self.rounds.append(row)
        round_dir = self.directory / "rounds" / f"{index:03}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if action.get("error"):
            row.update(status="response_error", error=action["error"])
        else:
            source = action["source"]
            (round_dir / "candidate.py").write_text(source)
            row["kernel_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            self.output.write_text(source)
            if self.config.generation_mode == "one-shot":
                self.best = {"round": index, "source": source, "runtime_ms": None, "development_verified": False}
                row["status"] = "generated_without_feedback"
            else:
                row["evaluation_requested"] = True
                result = self.evaluate(index)
                atomic_json(round_dir / "standard.json", result)
                self.remaining()
                runtime = result.get("runtime")
                good = (result.get("compiled") is True and result.get("correctness") is True
                        and type(runtime) in (int, float) and math.isfinite(runtime) and runtime > 0)
                row.update(compiled=result.get("compiled", False), correctness=result.get("correctness", False),
                           runtime_ms=runtime if good else None, status="development_pass" if good else "development_failed",
                           error=str(result.get("metadata", {}).get("error", ""))[:2000])
                if good and (self.best is None or runtime < self.best["runtime_ms"]):
                    self.best = {"round": index, "source": source, "runtime_ms": runtime, "development_verified": True}
                if self.best:
                    atomic_json(self.directory / "best_candidate.json", self.best)
                if good and self.config.agent_profile:
                    row["profile_requested"] = True
                    try:
                        row["profile"] = self.evaluate(index, task="profile")
                    except BudgetExhausted:
                        raise
                    except Exception as exc:
                        row["profile"] = {"status": "unavailable", "error": type(exc).__name__}
                        if self.device_budget:
                            raise BudgetExhausted("Device profile request failed without complete usage evidence") from exc
        atomic_json(round_dir / "feedback.json", row)
        # Save the best even if a later call times out or produces broken code.
        self.restore_best()
        atomic_json(self.directory / "development_history.json", self.rounds)
        if len(self.rounds) >= self.limit or action.get("stop"):
            raise Submitted({"role": "exit", "content": "Candidate budget complete",
                             "extra": {"exit_status": "CandidateLimit" if not action.get("stop") else action.get("exit_status", "CostEvidenceMissing"),
                                       "submission": str(self.output) if self.best else ""}})
        return {"feedback": row, "best_development": {k: v for k, v in (self.best or {}).items() if k != "source"},
                "remaining_candidates": self.limit - len(self.rounds), "remaining_seconds": self.remaining(),
                "instruction": "Return a complete improved ModelNew. Hidden tests are reserved for final evaluation."}

    def restore_best(self):
        if self.best:
            self.output.write_text(self.best["source"])
            atomic_json(self.directory / "best_candidate.json", self.best)
        else:
            self.output.unlink(missing_ok=True)

    @staticmethod
    def get_template_vars(**kwargs):
        return kwargs

    def serialize(self):
        return {"info": {"environment": "kernelbench-pro-standard-feedback", "hidden_feedback": False}}


def run_loop(directory, config, pid, *, response_fn=None, evaluator=development_evaluate):
    from minisweagent.agents.default import DefaultAgent

    from .generation import build_prompt

    directory = Path(directory)
    started = time.monotonic()

    def remaining():
        seconds = config.agent_timeout - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("Generation wall-clock budget exhausted")
        return seconds

    model = KernelModel(config, directory, remaining, response_fn=response_fn)
    environment = KernelEnvironment(directory, config, pid, remaining, evaluator=evaluator)
    agent = DefaultAgent(model, environment, system_template=(
        "You write GPU kernels. Return one complete Python code block defining torch.nn.Module ModelNew. "
        "Preserve the reference semantics, shapes, dtypes and parameters. "
        "You have no shell, file or evaluation tools. The controller compiles/tests each submitted source when feedback is enabled. "
        "Use development feedback to diagnose errors and bottlenecks; never infer hidden-test success."),
        instance_template="{{task}}", step_limit=environment.limit, cost_limit=config.agent_cost_limit,
        wall_time_limit_seconds=config.agent_timeout, output_path=directory / "trajectory.json")
    outcome, error = {}, None
    try:
        task = build_prompt(config, pid) + "\nPublic reference:\n```python\n" + (directory / "model.py").read_text() + "\n```"
        if config.evaluation == "modal":
            from ..runtime.modal_backend import modal_session

            context = modal_session(config)
        else:
            context = nullcontext()
        with context:
            outcome = agent.run(task)
    except BudgetExhausted as exc:
        error = str(exc)
        outcome = {"exit_status": "ResourceBudgetLimit"}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        environment.restore_best()
        atomic_json(directory / "development_history.json", environment.rounds)
    usage = model.usage()
    if environment.device_budget:
        usage["resource_budgets"]["device_lease"] = environment.device_budget.snapshot()
    budget = {**usage, "candidate_submissions": sum("kernel_sha256" in row for row in environment.rounds),
              "development_evaluations": environment.requests["evaluate"],
              "profile_requests": environment.requests["profile"],
              "generation_wall_seconds": time.monotonic() - started, "development_request_wall_seconds": environment.development_seconds,
              "gpu_execution_seconds": None, "billing_cost_certified": False,
              "cost_limit_policy": "check_between_model_calls; one_call_may_exceed_estimated_cap",
              "protocol": protocol(config)}
    result = {"status": "success" if environment.best else "failed", "budget": budget,
              "exit_status": outcome.get("exit_status", "Error"), "error": error,
              "scaffold_version": version("mini-swe-agent"), "rounds": environment.rounds,
              "selected": {k: v for k, v in (environment.best or {}).items() if k != "source"},
              "final_hidden_tests_run": False, "protocol_sha256": fingerprint(protocol(config))}
    atomic_json(directory / "agent_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    config = RunConfig(**read_json(args.directory / "job_config.json")).validate()
    result = run_loop(args.directory, config, config.start)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
