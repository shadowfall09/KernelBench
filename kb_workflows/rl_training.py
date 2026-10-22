"""Explicit local TRL GRPO entrypoint; planning/importing this module never trains."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .common import ROOT, atomic_json, fingerprint, read_json
from .local_model import checkpoint_digest
from .resource_budgets import BudgetExhausted, Ledger
from .rl_plan import TRL_VERSION, dataset_rows, read_plan, task_config, validate_plan
from .rl_rewards import baseline_coverage, evaluate_group, parse_source, validate_reward

JOURNAL_PROTOCOL = "kernelbench_pro_rl_journal_v1"
CHECKPOINT_FILE = "kernelbench_training.json"


@contextmanager
def offline_environment():
    values = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
              "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Journal:
    """Reserve each group before generation; interrupted groups remain charged."""

    def __init__(self, plan):
        self.plan, self.training = plan, validate_plan(plan)
        self.root = Path(plan["output_directory"])
        self.path = self.root / "training_journal.json"
        if not self.path.exists() and any((self.root / name).exists() for name in ("training_result.json", "trainer", "episodes", "rewards")):
            raise ValueError("Existing training artifacts lack their reservation journal; use a new output directory")
        self.document = read_json(self.path, {"protocol": JOURNAL_PROTOCOL, "plan_sha256": plan["plan_sha256"], "groups": []})
        if self.document.get("protocol") != JOURNAL_PROTOCOL or self.document.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Training journal belongs to another plan")
        self.pending = None
        applied = [row for row in self.document["groups"] if type(row.get("optimizer_step_after")) is int]
        self.active_group_ids = ([row["id"] for row in applied] if
                                 [row["optimizer_step_after"] for row in applied] == list(range(1, len(applied) + 1)) else [])
        if not self.path.exists():
            atomic_json(self.path, self.document)

    def reserve(self, step):
        if self.pending is not None:
            raise ValueError("Prior rollout group did not finish")
        if len(self.document["groups"]) >= self.training.max_steps:
            raise BudgetExhausted("Declared rollout group budget exhausted, including interrupted groups")
        self.pending = {"id": uuid.uuid4().hex, "optimizer_step_before": step, "status": "generation_started",
                        "started_at": time.time(), "group_size": self.training.group_size, "usage_complete": False}
        self.pending["started_monotonic"] = time.monotonic()
        self.document["groups"].append(self.pending)
        atomic_json(self.path, self.document)
        return self.pending

    def settle(self, receipt_path, *, completion_tokens, episode_directories):
        if self.pending is None:
            raise ValueError("Reward callback has no pre-generation reservation")
        receipt = validate_reward(receipt_path)
        self.pending.update(status="rewarded", reward_path=str(Path(receipt_path).resolve()), reward_sha256=receipt["receipt_sha256"],
                            completion_sequence_tokens=list(completion_tokens), episode_directories=episode_directories,
                            usage_scope="completion_sequence_includes_tool_observations; not_model_output_or_billing",
                            usage_complete=False, updated_at=time.time())
        atomic_json(self.path, self.document)
        self.pending = None

    def optimizer_applied(self, step):
        if (not self.document["groups"] or self.pending is not None or type(step) is not int
                or step != len(self.active_group_ids) + 1 or self.document["groups"][-1].get("status") != "rewarded"
                or self.document["groups"][-1].get("optimizer_step_before") != step - 1):
            raise ValueError("Optimizer update lacks its complete rewarded predecessor")
        row = self.document["groups"][-1]
        row["optimizer_step_after"] = step
        self.active_group_ids.append(row["id"])
        atomic_json(self.path, self.document)


class DevelopmentEnvironment:
    """Only submit_kernel is exposed as a tool by TRL's environment introspection."""

    def __init__(self, plan, *, evaluator=None):
        self._plan, self._evaluator = plan, evaluator
        self._controller, self._directory, self._task_key = None, None, None

    def reset(self, task_key=None, **kwargs):
        from .agent_loop import KernelEnvironment, development_evaluate

        if task_key not in self._plan["splits"]["train"]:
            raise ValueError("Development reset cannot enter held-out tasks")
        self._task_key = task_key
        self._directory = Path(self._plan["output_directory"]) / "episodes" / uuid.uuid4().hex
        config = task_config(self._plan, task_key, directory=self._directory / "candidate")
        self._directory.mkdir(parents=True)
        self._started = time.monotonic()

        def remaining():
            seconds = config.agent_timeout - (time.monotonic() - self._started)
            if seconds <= 0:
                raise BudgetExhausted("Episode wall budget exhausted")
            return seconds
        self._controller = KernelEnvironment(self._directory, config, config.start, remaining,
                                             evaluator=self._evaluator or development_evaluate)
        self._closed = False
        return None

    def submit_kernel(self, source: str) -> str:
        """Submit a complete ModelNew source for standard-only development feedback.

        Args:
            source: Complete Python source defining ModelNew.

        Returns:
            Standard correctness, timing and remaining development budget, without hidden results.
        """
        from minisweagent.exceptions import Submitted

        if self._controller is None or self._closed:
            return json.dumps({"status": "closed", "hidden_feedback": False})
        code, error = parse_source(source)
        try:
            result = self._controller.execute({"source": code} if code else {"error": error})
        except Submitted:
            self._closed = True
            result = {"status": "candidate_limit", "selected": {key: value for key, value in (self._controller.best or {}).items() if key != "source"}}
        except Exception as exc:
            self._closed = True
            self._controller.restore_best()
            result = {"status": "stopped", "error": type(exc).__name__}
        atomic_json(self._directory / "episode.json", {"task_key": self._task_key, "rounds": self._controller.rounds,
                    "closed": self._closed, "hidden_feedback": False})
        return json.dumps(result, ensure_ascii=False)

    def _source(self, terminal_text):
        self._closed = True
        if self._controller and self._controller.best:
            source, selected = self._controller.best["source"], "development_best"
        elif self._controller and self._controller.rounds:
            source, selected = "", "no_correct_candidate"
        else:
            source, selected = terminal_text, "terminal_without_development"
            try:
                self._controller.remaining()
            except BudgetExhausted:
                source, selected = "", "terminal_after_episode_deadline"
        code, _ = parse_source(source)
        atomic_json(self._directory / "episode.json", {"task_key": self._task_key, "rounds": self._controller.rounds,
                    "closed": True, "hidden_feedback": False, "selected": selected,
                    "selected_source_sha256": fingerprint({"source": code}) if code else None,
                    "candidate_submissions": sum("kernel_sha256" in row for row in self._controller.rounds)
                        + int(selected == "terminal_without_development" and code is not None)})
        return source


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(row.get("content") or "" for row in completion
                         if isinstance(row, dict) and row.get("role") == "assistant" and isinstance(row.get("content"), str))
    return ""


class RewardFunction:
    __name__ = "kernelbench_terminal_reward"

    def __init__(self, plan, journal, *, group_evaluator=None):
        self.plan, self.journal = plan, journal
        self.evaluate = group_evaluator or evaluate_group
        self.expected = {row["task_key"]: row for row in dataset_rows(plan)}

    def __call__(self, *, prompts, completions, completion_ids, task_key, prompt_sha256, environments=None, **kwargs):
        training = validate_plan(self.plan)
        if (len(completions) != training.group_size or len(task_key) != len(completions) or len(set(task_key)) != 1
                or task_key[0] not in self.expected or len(prompts) != len(completions)
                or len(completion_ids) != len(completions) or len(prompt_sha256) != len(completions)):
            raise ValueError("Reward callback requires a complete same-task training group")
        expected = self.expected[task_key[0]]
        if any(value != expected["prompt_sha256"] for value in prompt_sha256):
            raise ValueError("Reward task/prompt provenance changed")
        if any(fingerprint({"messages": prompt}) != expected["prompt_sha256"] for prompt in prompts):
            raise ValueError("Actual reward prompts differ from the frozen training dataset")
        if any(not isinstance(ids, list) or len(ids) > training.max_completion_tokens
               or any(type(token) is not int or token < 0 for token in ids) for ids in completion_ids):
            raise ValueError("Completion token sequences are malformed or exceed the declared limit")
        if self.journal.pending is None:
            raise ValueError("Generation was not reserved before the reward callback")
        texts = [completion_text(item) for item in completions]
        if training.mode == "one-shot" and time.monotonic() - self.journal.pending["started_monotonic"] > self.plan["verifier_config"]["agent_timeout"]:
            texts = [""] * len(texts)
        directories = []
        if training.mode == "feedback":
            if not isinstance(environments, list) or len(environments) != len(texts):
                raise ValueError("Feedback training needs the corresponding isolated episode environments")
            if any(type(env) is not DevelopmentEnvironment or env._task_key != task_key[0] for env in environments):
                raise ValueError("Episode environment belongs to a different task")
            texts = [env._source(text) for env, text in zip(environments, texts)]
            directories = [str(env._directory) for env in environments]
        directory = Path(self.plan["output_directory"]) / "rewards" / self.journal.pending["id"]
        result = self.evaluate(self.plan, task_key[0], texts, directory=directory)
        self.journal.settle(directory / "reward.json", completion_tokens=[len(ids) for ids in completion_ids],
                            episode_directories=directories)
        return result["rewards"]


def checkpoint_record(plan, journal, directory, step):
    directory = Path(directory)
    record = {"plan_sha256": plan["plan_sha256"], "step": step, "journal_prefix": fingerprint(journal.document),
              "consumed_groups": len(journal.document["groups"]),
              "active_group_ids": list(journal.active_group_ids),
              "artifact_sha256": checkpoint_digest(directory, exclude=(CHECKPOINT_FILE,))}
    record["receipt_sha256"] = fingerprint(record)
    atomic_json(directory / CHECKPOINT_FILE, record)
    return record


def validate_checkpoint(plan, journal, directory):
    directory = Path(directory).resolve()
    root = Path(plan["output_directory"]).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("Resume checkpoint must be an artifact of this training run")
    record = read_json(directory / CHECKPOINT_FILE, {})
    if (record.get("plan_sha256") != plan["plan_sha256"] or record.get("receipt_sha256") !=
            fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})
            or record.get("artifact_sha256") != checkpoint_digest(directory, exclude=(CHECKPOINT_FILE,))):
        raise ValueError("Training checkpoint provenance is missing or stale")
    prefix = {**journal.document, "groups": journal.document["groups"][:record["consumed_groups"]]}
    if fingerprint(prefix) != record["journal_prefix"]:
        raise ValueError("Reward history before the checkpoint changed")
    indexed = {row["id"]: row for row in prefix["groups"]}
    active = record.get("active_group_ids", [])
    if (len(active) != record["step"] or len(set(active)) != len(active)
            or any(key not in indexed or indexed[key].get("optimizer_step_after") != i + 1 for i, key in enumerate(active))):
        raise ValueError("Checkpoint lacks a consistent active optimizer path")
    for row in journal.document["groups"]:
        if row.get("reward_path") and validate_reward(row["reward_path"])["receipt_sha256"] != row.get("reward_sha256"):
            raise ValueError("Training reward history changed")
    state = read_json(directory / "trainer_state.json", {})
    if state.get("global_step") != record["step"]:
        raise ValueError("Trainer step and checkpoint receipt disagree")
    return directory


def native_backend(plan, journal, reward):
    """Load actual local model/trainer only after explicit execution."""
    from packaging.version import Version

    if version("trl") != TRL_VERSION or not Version("5.2") <= Version(version("transformers")) < Version("6"):
        raise ValueError(f"RL requires trl=={TRL_VERSION} and transformers>=5.2,<6")
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    training = validate_plan(plan)
    coverage = baseline_coverage(plan)
    if any(not row["ready"] for row in coverage):
        raise ValueError("Training baselines are missing or stale; review/prepare them with rl_workflow baselines before loading the policy")
    tokenizer = AutoTokenizer.from_pretrained(training.checkpoint, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = dataset_rows(plan)
    tools = [DevelopmentEnvironment(plan).submit_kernel] if training.mode == "feedback" else None
    for row in rows:
        tokens = tokenizer.apply_chat_template(row["prompt"], tools=tools, tokenize=True, add_generation_prompt=True)
        if len(tokens) > training.max_prompt_tokens:
            raise ValueError("Public reference exceeds max_prompt_tokens; refusing silent truncation")
    model = AutoModelForCausalLM.from_pretrained(training.checkpoint, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype="float32" if training.use_cpu else "auto")

    class Callbacks(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            journal.reserve(state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            journal.optimizer_applied(state.global_step)

        def on_save(self, args, state, control, **kwargs):
            checkpoint_record(plan, journal, Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step)

    options = GRPOConfig(output_dir=str(Path(plan["output_directory"]) / "trainer"), max_steps=training.max_steps,
                         per_device_train_batch_size=training.group_size, gradient_accumulation_steps=1,
                         generation_batch_size=training.group_size, num_generations=training.group_size,
                         max_completion_length=training.max_completion_tokens, learning_rate=training.learning_rate,
                         temperature=training.temperature, beta=training.beta, num_iterations=1,
                         loss_type="dapo", scale_rewards="group", seed=training.seed, data_seed=training.seed,
                         save_steps=training.save_steps, save_strategy="steps", logging_steps=1,
                         report_to="none", push_to_hub=False, disable_dropout=True, use_cpu=training.use_cpu,
                         bf16=False, fp16=False, dataloader_num_workers=0,
                         max_tool_calling_iterations=plan["verifier_config"]["optimization_rounds"] if tools else None,
                         generation_kwargs={"num_beams": 1, "max_time": plan["verifier_config"]["agent_timeout"]},
                         **({"save_safetensors": True} if "save_safetensors" in GRPOConfig.__dataclass_fields__ else {}))
    return GRPOTrainer(model=model, processing_class=tokenizer, args=options, train_dataset=Dataset.from_list(rows),
                       reward_funcs=reward, callbacks=[Callbacks()],
                       **({"environment_factory": lambda: DevelopmentEnvironment(plan)} if tools else {}))


def run_worker(plan, *, resume=None, backend_factory=None, group_evaluator=None, supervisor_nonce=None):
    training = validate_plan(plan)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed rollout groups cannot certify this serial device schedule; launch one training process")
    root = Path(plan["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "training.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another worker owns this training run") from exc
        journal = Journal(plan)
        if journal.document["groups"] and resume is None:
            raise ValueError("Existing training attempts require an explicit receipt-bound checkpoint")
        if resume:
            resume = str(validate_checkpoint(plan, journal, resume))
            journal.active_group_ids = list(read_json(Path(resume) / CHECKPOINT_FILE)["active_group_ids"])
        if len(journal.document["groups"]) >= training.max_steps:
            raise BudgetExhausted("All rollout groups were already consumed, including interrupted attempts")
        reward = RewardFunction(plan, journal, group_evaluator=group_evaluator)
        started = time.monotonic()
        result = {"plan_sha256": plan["plan_sha256"], "training_started": True, "status": "error",
                  "attempt_id": uuid.uuid4().hex, "supervisor_nonce": supervisor_nonce,
                  "execution_kind": "native_trl" if backend_factory is None and group_evaluator is None else "injected_backend_or_verifier"}
        result["software_versions"] = {}
        for name in ("torch", "transformers", "trl", "datasets", "accelerate"):
            try:
                result["software_versions"][name] = version(name)
            except PackageNotFoundError:
                result["software_versions"][name] = None
        atomic_json(root / "training_result.json", result)
        with offline_environment():
            backend = (backend_factory or native_backend)(plan, journal, reward)
            try:
                outcome = backend.train(resume_from_checkpoint=resume)
                result.update(status="completed", global_step=outcome.global_step)
            except BudgetExhausted as exc:
                result.update(status="rollout_budget_exhausted", error=str(exc), global_step=len(journal.active_group_ids))
            finally:
                result.update(request_wall_seconds=time.monotonic() - started,
                              consumed_groups=len(journal.document["groups"]), max_groups=training.max_steps,
                              generation_usage_complete=False, gpu_time_or_billing_certified=False)
                atomic_json(root / "training_result.json", result)
            final = root / "final_model"
            backend.save_model(str(final))
            if getattr(backend, "processing_class", None):
                backend.processing_class.save_pretrained(str(final))
            result.update(final_model=str(final), final_model_sha256=checkpoint_digest(final))
            from .rl_lineage import emit

            lineage_path, lineage = emit(plan, journal, result, attempt_id=result["attempt_id"],
                                         execution_kind=result["execution_kind"], supervisor_nonce=supervisor_nonce)
            result.update(lineage_path=str(lineage_path), lineage_sha256=lineage["receipt_sha256"])
            atomic_json(root / "training_result.json", result)
        return result


def _execute_plan(path, *, execute=False, resume=None, worker_runner=None):
    plan = read_plan(path)
    if not execute:
        return {"training_started": False, "plan_sha256": plan["plan_sha256"], "tasks": plan["splits"],
                "trainer": plan["trainer"], "declared_generated_output_upper_bound": plan["declared_generated_output_upper_bound"]}
    root = Path(plan["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    wall_path = root / "training_wall_budget.json"
    if not wall_path.exists() and any((root / name).exists() for name in ("training_journal.json", "training_result.json", "trainer")):
        raise ValueError("Existing training attempts lack their wall-time receipt; do not reset the budget")
    previous = read_json(wall_path, {})
    if previous and previous.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Training wall budget belongs to another plan")
    ledger = Ledger(plan["training"]["wall_seconds"], "training_process_wall_seconds")
    snapshot = previous.get("ledger", {})
    ledger.entries = list(snapshot.get("entries", []))
    # A lost supervisor cannot prove unused time; charge its complete pending reservation.
    for nonce, reserved in snapshot.get("pending", {}).items():
        ledger.pending[nonce] = reserved
        ledger.settle(nonce)
    if ledger.remaining <= 0:
        raise BudgetExhausted("Training wall budget is exhausted or prior usage is unknown")
    if ledger.remaining <= 5:
        raise BudgetExhausted("Insufficient training wall budget including process cleanup")
    nonce = ledger.reserve(ledger.remaining)
    atomic_json(wall_path, {"plan_sha256": plan["plan_sha256"], "ledger": ledger.snapshot()})
    started = time.monotonic()
    try:
        if worker_runner:
            code = worker_runner(plan, resume)
        else:
            from .generation import terminate_agent_tree

            command = [sys.executable, "-m", "kb_workflows.rl_training", "--plan", str(Path(path).resolve()),
                       "--worker", "--supervisor-nonce", nonce]
            if resume:
                command += ["--resume-checkpoint", str(Path(resume).resolve())]
            with (root / "training.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=ledger.pending[nonce] - 5)
                except subprocess.TimeoutExpired:
                    code = 124
                finally:
                    terminate_agent_tree(process)
    finally:
        ledger.settle(nonce, time.monotonic() - started)
        atomic_json(wall_path, {"plan_sha256": plan["plan_sha256"], "ledger": ledger.snapshot()})
    from .rl_lineage import supervisor
    from .rl_lineage import validate as validate_lineage

    worker = read_json(root / "training_result.json", {})
    matched = worker.get("supervisor_nonce") == nonce and worker.get("plan_sha256") == plan["plan_sha256"]
    if not matched:
        worker = {}
    supervised = supervisor(plan, nonce, ledger.snapshot(), code, worker)
    completion_verified, error = False, None
    if worker.get("lineage_path") and worker.get("final_model"):
        try:
            validated = validate_lineage(worker["final_model"], receipt_path=worker["lineage_path"], allow_injected=True)
            completion_verified = validated["receipt"]["receipt_sha256"] == worker.get("lineage_sha256")
        except (ValueError, OSError, KeyError) as exc:
            error = str(exc)
    elif code == 0:
        error = "Worker exited without a current policy-lineage receipt"
    return {"training_started": True, "exit_code": code if code else (0 if completion_verified else 1),
            "worker_exit_code": code, "result": worker, "completion_verified": completion_verified, "error": error,
            "supervisor_nonce": nonce, "supervisor_sha256": supervised["receipt_sha256"],
            "wall_budget": ledger.snapshot(), "physical_preemption_certified": False}


def execute_plan(path, *, execute=False, resume=None, worker_runner=None):
    if not execute:
        return _execute_plan(path, execute=False, resume=resume, worker_runner=worker_runner)
    plan = read_plan(path)
    root = Path(plan["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another supervisor owns this training budget") from exc
        return _execute_plan(path, execute=True, resume=resume, worker_runner=worker_runner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-nonce", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        result = run_worker(read_plan(args.plan), resume=args.resume_checkpoint, supervisor_nonce=args.supervisor_nonce)
        code = 0 if result["status"] == "completed" else 1
    else:
        result = execute_plan(args.plan, execute=args.execute, resume=args.resume_checkpoint)
        code = result.get("exit_code", 0)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
