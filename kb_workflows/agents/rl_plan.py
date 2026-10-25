"""Frozen, nonexecuting plans for local GRPO and separate held-out evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..benchmarks.suites import suite_rows
from ..common import ROOT, atomic_json, evaluator_digest, fingerprint, read_json, select_problems
from ..config import RunConfig, file_options, parser
from ..numerics.contract_specs import contract_for
from .generation import build_prompt, prompt_source
from .local_model import checkpoint_digest

PROTOCOL = "kernelbench_pro_grpo_plan_v1"
TRL_VERSION = "1.8.0"


@dataclass
class TrainingConfig:
    checkpoint: str
    mode: str = "feedback"
    reward_mode: str = "qualified"
    group_size: int = 4
    max_steps: int = 100
    max_prompt_tokens: int = 8192
    max_completion_tokens: int = 4096
    learning_rate: float = 1e-6
    temperature: float = 0.8
    beta: float = 0.0
    seed: int = 42
    save_steps: int = 10
    wall_seconds: float = 14400.0
    use_cpu: bool = True
    family_disjoint: bool = True
    raw_speedup_clip: float = 4.0
    certificate_bonus: float = 0.0
    certificate_min_score: float = 0.5
    scaling_contracts: str = ""

    def validate(self):
        for name in ("group_size", "max_steps", "max_prompt_tokens", "max_completion_tokens", "save_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (2 if name == "group_size" else 1):
                raise ValueError(f"Invalid positive training option: {name}")
        for name in ("learning_rate", "temperature", "wall_seconds", "raw_speedup_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid finite positive training option: {name}")
        if type(self.beta) not in (int, float) or not math.isfinite(self.beta) or self.beta < 0 or type(self.seed) is not int:
            raise ValueError("Invalid GRPO beta or seed")
        if any(type(getattr(self, name)) is not bool for name in ("use_cpu", "family_disjoint")):
            raise ValueError("Training booleans must be explicit booleans")
        if self.mode not in {"one-shot", "feedback"} or self.reward_mode not in {"qualified", "raw_speedup"}:
            raise ValueError("Unknown episode or reward mode")
        if any(type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name))
               or not 0 <= getattr(self, name) <= 1 for name in ("certificate_bonus", "certificate_min_score")):
            raise ValueError("Certificate bonus and score threshold must lie in [0,1]")
        if self.certificate_bonus and (self.reward_mode != "qualified" or not self.scaling_contracts):
            raise ValueError("Certificate bonuses require qualified rewards and reviewed scaling contracts")
        if self.scaling_contracts:
            from ..analysis.scaling_certificate import read_contracts

            self.scaling_contracts = str(Path(self.scaling_contracts).expanduser().resolve())
            read_contracts(self.scaling_contracts)
        self.checkpoint = str(Path(self.checkpoint).expanduser().resolve())
        if not Path(self.checkpoint).is_dir():
            raise ValueError("Training requires an existing offline checkpoint directory")
        return self


def implementation_digest():
    directory = Path(__file__).parent.parent
    return fingerprint({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                        for name in ("agents/rl_plan.py", "agents/rl_rewards.py", "agents/rl_training.py", "agents/rl_audit.py", "agents/rl_evaluation.py",
                                     "agents/rl_lineage.py", "agents/rl_effectiveness.py", "analysis/scaling_certificate.py")})


def task_config(plan, key, *, directory, samples=1):
    row = plan["tasks"][key]
    options = dict(plan["verifier_config"])
    population_contract = {name: options[name] for name in ("suite", "suite_role", "task_track")} if options.get("reference_population") else {}
    options.update(level=row["level"], suite="production41" if row["level"] == 5 else "full", suite_role="scored",
                   task_track=row.get("track", "all"), start=row["problem_id"], end=row["problem_id"], num_samples=samples,
                   run_name=Path(directory).name, runs_dir=str(Path(directory).resolve().parent), stage="evaluate", resume=False,
                   generation_mode=plan["training"]["mode"], optimization_rounds=options["optimization_rounds"])
    if population_contract:
        options.update(**population_contract, reference_population_scope="task")
    return RunConfig(**options).validate()


def verifier_inputs(options):
    return {name: fingerprint({"document": read_json(Path(options[name]))}) for name in
            ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file", "regrade_manifest") if options.get(name)}


def _task(row, base):
    if not isinstance(row, dict) or set(row) - {"level", "problem_id", "family"}:
        raise ValueError("Each task requires explicit level/problem_id and optional reviewed family")
    level, pid = row.get("level"), row.get("problem_id")
    if type(level) is not int or type(pid) is not int or level not in {1, 2, 3, 5}:
        raise ValueError("Invalid training task identity")
    path = select_problems(level, problem_ids=[pid])[pid]
    hidden = ROOT / "hidden_tests" / f"level{level}" / f"{pid}_hidden.py"
    if not hidden.is_file():
        raise ValueError("All RL tasks require held-out correctness inputs")
    source = path.read_text()
    contract = contract_for(level, pid, source)
    production = suite_rows(5, "scored", suite="production41").get(pid, {}) if level == 5 else {}
    if (base.strict_validation or base.evidence_level != "current") and contract is None:
        raise ValueError(f"Reviewed L0 auxiliary contract unavailable for {level}/{pid}")
    family = production.get("family") or (contract or {}).get("family")
    if family in {"square_matmul", "batched_matmul"}:
        family = "matmul"
    declared = row.get("family")
    if declared is not None and (not isinstance(declared, str) or not declared.strip()):
        raise ValueError("Declared family must be a nonempty reviewed label")
    if declared and family and declared != family:
        raise ValueError("Declared family contradicts the repository task contract")
    family = family or declared
    if not family:
        raise ValueError("Unclassified tasks require a reviewed family label for split auditing")
    return {"level": level, "problem_id": pid, "family": family,
            "family_source": "repository_contract" if contract or production else "user_declared_review",
            "track": production.get("track", "all"), "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "hidden_source_sha256": hashlib.sha256(hidden.read_bytes()).hexdigest()}


def make_plan(specification, *, directory, base_directory=None):
    if not isinstance(specification, dict) or set(specification) - {"training", "verifier_config", "train", "heldout"}:
        raise ValueError("Unknown RL specification fields")
    root = Path(base_directory or Path.cwd()).resolve()
    settings = dict(specification.get("training", {}))
    if not settings.get("checkpoint"):
        raise ValueError("An explicit local checkpoint is required")
    checkpoint = Path(settings["checkpoint"]).expanduser()
    settings["checkpoint"] = str(checkpoint if checkpoint.is_absolute() else root / checkpoint)
    if settings.get("scaling_contracts"):
        source = Path(settings["scaling_contracts"]).expanduser()
        settings["scaling_contracts"] = str(source if source.is_absolute() else root / source)
    training = TrainingConfig(**settings).validate()
    options = specification.get("verifier_config", {})
    if isinstance(options, str):
        path = Path(options)
        options = file_options(path if path.is_absolute() else root / path, parser())
    else:
        options = dict(options)
        for name in ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file", "regrade_manifest"):
            if options.get(name) and not Path(options[name]).is_absolute():
                options[name] = str((root / options[name]).resolve())
    defaults = dict(generation="local", evaluation="local", strict_validation=True, evidence_level="L0",
                    paired_timing=True, final_repeats=3, parallel=1)
    base = RunConfig(**{**defaults, **options}).validate()
    if base.regrade_manifest:
        raise ValueError("RL verification cannot reuse an artifact-regrade population as new generated candidates")
    base.require_gpu_target()
    if training.certificate_bonus and base.score_mode != "headroom":
        raise ValueError("A design certificate bonus requires reviewed Headroom D/SOL bounds")
    if (base.evaluation not in {"local", "docker"} or len(base.gpus.split(",")) != 1 or base.parallel != 1
            or not base.hidden_tests or not base.paired_timing or base.final_repeats < 2 or not base.score_gate_required):
        raise ValueError("RL rewards require serial one-device local/Docker verification, hidden checks, paired timing and final repeats >=2")
    if any((base.agent_token_limit, base.agent_output_token_limit, base.agent_device_seconds)):
        raise ValueError("Native per-call reservations cannot certify TRL generation; RL uses its explicit step/token bounds and receipts")
    directory = Path(directory).resolve()
    if directory.is_relative_to(Path(training.checkpoint)) or Path(training.checkpoint).is_relative_to(directory):
        raise ValueError("Training output and initial checkpoint directories must be separate")
    tasks, splits = {}, {}
    for split in ("train", "heldout"):
        declared = specification.get(split)
        if not isinstance(declared, list) or not declared:
            raise ValueError("Nonempty explicit train and heldout task lists are required")
        splits[split] = []
        for item in declared:
            row = _task(item, base)
            key = f"{row['level']}/{row['problem_id']}"
            if key in tasks or any(row["reference_sha256"] == prior["reference_sha256"] for prior in tasks.values()):
                raise ValueError("Task identity/reference overlaps across or within splits")
            tasks[key] = row
            splits[split].append(key)
    if training.family_disjoint and ({tasks[key]["family"] for key in splits["train"]}
                                      & {tasks[key]["family"] for key in splits["heldout"]}):
        raise ValueError("Declared task families overlap train and heldout; review the split")
    plan = {"protocol": PROTOCOL, "training": asdict(training), "verifier_config": base.to_dict(), "tasks": tasks, "splits": splits,
            "output_directory": str(directory), "initial_checkpoint_sha256": checkpoint_digest(training.checkpoint),
            "evaluator_sha256": evaluator_digest(), "rl_implementation_sha256": implementation_digest(),
            "trainer": f"trl=={TRL_VERSION}", "experiments_started": False,
            "verifier_inputs": verifier_inputs(base.to_dict()),
            "scaling_contracts_sha256": fingerprint({"document": read_json(Path(training.scaling_contracts))}) if training.scaling_contracts else None,
            "split_scope": "explicit_task_and_reference_identity; family_labels_are_reviewed_not_semantic_proof",
            "declared_generated_output_upper_bound": training.max_steps * training.group_size * training.max_completion_tokens
                * (1 + base.optimization_rounds if training.mode == "feedback" else 1),
            "gpu_time_or_billing_certified": False}
    # Validate every row against the same native configuration used by rewards.
    for key in tasks:
        task_config(plan, key, directory=directory / "preview", samples=training.group_size)
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get("protocol") != PROTOCOL or plan.get("plan_sha256") !=
            fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
            or plan.get("evaluator_sha256") != evaluator_digest() or plan.get("rl_implementation_sha256") != implementation_digest()):
        raise ValueError("RL plan changed or implementation/evaluator is stale")
    training = TrainingConfig(**plan["training"]).validate()
    if verifier_inputs(plan["verifier_config"]) != plan.get("verifier_inputs"):
        raise ValueError("Frozen verifier input files changed")
    scaling_sha = fingerprint({"document": read_json(Path(training.scaling_contracts))}) if training.scaling_contracts else None
    if scaling_sha != plan.get("scaling_contracts_sha256"):
        raise ValueError("Reviewed scaling contracts changed")
    if checkpoint_digest(training.checkpoint) != plan["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint assets changed")
    for key, row in plan["tasks"].items():
        path = select_problems(row["level"], problem_ids=[row["problem_id"]])[row["problem_id"]]
        hidden = ROOT / "hidden_tests" / f"level{row['level']}" / f"{row['problem_id']}_hidden.py"
        if (hashlib.sha256(path.read_bytes()).hexdigest() != row["reference_sha256"]
                or not hidden.is_file() or hashlib.sha256(hidden.read_bytes()).hexdigest() != row["hidden_source_sha256"]):
            raise ValueError(f"RL task source changed: {key}")
    return training


def dataset_rows(plan, split="train"):
    if split not in {"train", "heldout"}:
        raise ValueError("Unknown RL split")
    rows = []
    for key in plan["splits"][split]:
        task = plan["tasks"][key]
        config = task_config(plan, key, directory=Path(plan["output_directory"]) / "preview")
        path = select_problems(task["level"], problem_ids=[task["problem_id"]])[task["problem_id"]]
        text = build_prompt(config, task["problem_id"])
        text += "\nUse submit_kernel for standard-only feedback when tools are provided; final hidden outcomes are terminal rewards only."
        text += "\nPublic reference:\n```python\n" + prompt_source(path.read_text(), task["level"], task["problem_id"]) + "\n```"
        prompt = [{"role": "user", "content": text}]
        rows.append({"task_key": key, "prompt": prompt, "prompt_sha256": fingerprint({"messages": prompt})})
    return rows


def write_plan(plan, path):
    validate_plan(plan)
    atomic_json(Path(path), plan)
    return Path(path)


def read_plan(path):
    plan = read_json(Path(path))
    validate_plan(plan)
    return plan
