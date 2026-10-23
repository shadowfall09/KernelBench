"""RL planning, real native scoring and trainer wiring with simulated actors/workers only."""

import copy
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _operator_evidence import attach_operator_evidence

from kb_workflows import engine, paper_analysis, rl_audit, rl_evaluation, rl_plan, rl_rewards, rl_training
from kb_workflows.baseline_variants import VARIANTS, select_baseline
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig
from kb_workflows.evidence import LEVELS, check, from_result
from kb_workflows.measurement import timing_stats
from kb_workflows.paired_statistics import paired_summary
from kb_workflows.resource_budgets import BudgetExhausted

SOURCE = "import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self, a, b): return a @ b\n"


@pytest.fixture
def specification(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    return {"training": {"checkpoint": str(checkpoint), "mode": "one-shot", "group_size": 2, "max_steps": 4,
                         "max_completion_tokens": 64, "use_cpu": True},
            "verifier_config": {"hardware": "RTX_A6000", "final_repeats": 2, "paired_trials": 10,
                                "num_perf_trials": 3, "agent_timeout": 30, "optimization_rounds": 2, "agent_profile": False},
            "train": [{"level": 1, "problem_id": 2}], "heldout": [{"level": 1, "problem_id": 26}]}


@pytest.fixture
def plan(specification, tmp_path):
    return rl_plan.make_plan(specification, directory=tmp_path / "training")


@pytest.fixture
def native_verifier(plan, tmp_path, monkeypatch):
    root = tmp_path / "verifier"
    for pid in (2, 26):
        target = root / "hidden_tests/level1" / f"{pid}_hidden.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "hidden_tests/level1" / f"{pid}_hidden.py", target)
    for module in (engine, paper_analysis, rl_rewards):
        monkeypatch.setattr(module, "ROOT", root)
    device = {"available": True, "name": "SIMULATED_GPU", "capability": [8, 6], "total_memory": 10**10,
              "multiprocessors": 84, "l2_bytes": 10**6, "torch_version": "unit-test", "cuda_version": "unit-test",
              "hip_version": None, "uuid": "simulated-device-uuid", "driver_version": "unit-test"}
    timing_receipt = {"output_consumed": True, "fresh_inputs": True, "varied_input_contents": True,
                      "timed_outputs_verified": True, "input_seeds": [101, 102, 103]}
    signature = [{"shape": [2, 2], "dtype": "torch.float32"}]
    for key in plan["tasks"]:
        config = rl_plan.task_config(plan, key, directory=tmp_path / "preview", samples=2)
        variants = {}
        for variant in VARIANTS:
            payload = engine.task_payload(config, config.start, task="baseline")
            payload["eval_config"]["baseline_variant"] = variant
            variants[variant] = {**timing_stats([10, 10, 10]), "baseline_correctness": True, "precision": "fp32", "tf32": False,
                "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}), "input_signature": signature,
                "device_fingerprint": device, "cold_start_ms": 10, "timing_method": "synchronized_wall",
                "timing_receipt": timing_receipt, "metadata": {"evaluation_config": payload["eval_config"], "workflow_signature": fingerprint(payload)}}
        baseline = select_baseline(variants)
        baseline["suite_complete"] = True
        baseline["metadata"]["workflow_signature"] = fingerprint(engine.task_payload(config, config.start, task="baseline"))
        reference = select_problems(1, problem_ids=[config.start])[config.start]
        target = root / "results/timing" / config.hardware / f"{config.baseline}.json"
        data = read_json(target, {"level1": {}})
        data["level1"][reference.name] = baseline
        atomic_json(target, data)
    seen = []

    def dispatch(config, payload):
        seen.append(copy.deepcopy(payload))
        sid = payload["sample_id"]
        source = payload["kernel_src"]
        bad = "WRONG" in source
        if payload["task"] == "qualify_small":
            runtime = -1
        else:
            runtime = 5 + sid / 4
        result = {"compiled": True, "correctness": not bad, "runtime": runtime,
                  "runtime_stats": timing_stats([runtime] * 3) if runtime > 0 else {},
                  "metadata": {"reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
                      "kernel_sha256": fingerprint({"source": source}), "precision": "fp32", "input_signature": signature,
                      "device_fingerprint": copy.deepcopy(device), "timing_method": "synchronized_wall", "cold_start_ms": 5,
                      "timing_receipt": copy.deepcopy(timing_receipt)}}
        passed = {name: check("passed", reason="Simulated verifier evidence") for name in LEVELS["L0"] if name != "correctness"}
        if bad:
            passed["precision_contract"] = check("failed", reason="Simulated wrong candidate")
        result["metadata"]["evidence_receipt"] = from_result(result, payload["eval_config"],
            scope="hidden" if payload["eval_config"].get("use_hidden_tests") else "standard", additional=passed)
        if payload["task"] == "paired_timing":
            pairs = [{"reference_ms": 10, "candidate_ms": runtime, "control_ms": 10, "seed": i, "order": "RCB"} for i in range(10)]
            result["metadata"]["paired_timing"] = {**paired_summary(pairs), "baseline_variant": payload["eval_config"]["paired_baseline_variant"],
                "output_consumed": True, "varied_inputs": True, "all_outputs_verified": True}
        return attach_operator_evidence(payload, result)
    return dispatch, seen


def evaluated(plan, native_verifier, directory, sources=None, **kwargs):
    dispatch, _ = native_verifier
    return rl_rewards.evaluate_group(plan, "1/2", sources or [SOURCE, SOURCE + "\n# second\n"],
                                     directory=directory, dispatch=dispatch, **kwargs)


def test_plan_binds_assets_split_and_real_native_configuration(plan):
    assert not plan["experiments_started"] and plan["trainer"] == "trl==1.8.0"
    assert plan["splits"] == {"train": ["1/2"], "heldout": ["1/26"]}
    assert plan["declared_generated_output_upper_bound"] == 4 * 2 * 64
    rows = rl_plan.dataset_rows(plan)
    assert len(rows) == 1 and rows[0]["task_key"] == "1/2"
    assert set(rows[0]) == {"task_key", "prompt", "prompt_sha256"}
    assert "SOL_alg" not in rows[0]["prompt"][0]["content"]


@pytest.mark.parametrize("change", ["overlap", "family", "missing_checkpoint", "remote", "multi_device", "no_hidden", "no_paired", "native_tokens"])
def test_invalid_or_unenforceable_training_plan_rejected(specification, tmp_path, change):
    spec = copy.deepcopy(specification)
    if change == "overlap":
        spec["heldout"] = spec["train"]
    elif change == "family":
        spec["heldout"] = [{"level": 1, "problem_id": 1}]
    elif change == "missing_checkpoint":
        spec["training"]["checkpoint"] = str(tmp_path / "absent")
    elif change == "remote":
        spec["verifier_config"]["evaluation"] = "queue"
    elif change == "multi_device":
        spec["verifier_config"]["gpus"] = "0,1"
    elif change == "no_hidden":
        spec["verifier_config"]["hidden_tests"] = False
    elif change == "no_paired":
        spec["verifier_config"]["paired_timing"] = False
    else:
        spec["verifier_config"]["agent_token_limit"] = 100
    with pytest.raises(ValueError):
        rl_plan.make_plan(spec, directory=tmp_path / "training")


def test_nonexecuting_cli_is_lightweight_and_does_not_load_any_model(specification, tmp_path):
    spec = tmp_path / "spec.json"
    atomic_json(spec, specification)
    code = f'''import sys
sys.path.insert(0, {str(ROOT)!r})
from scripts.rl_workflow import main
assert main(['plan','--spec',{str(spec)!r},'--output-dir',{str(tmp_path / 'planned')!r}]) == 0
assert not any(name in sys.modules for name in ('torch','trl','transformers','datasets'))
'''
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["training_started"] is False
    assert not (tmp_path / "planned/training").exists()


def test_checkpoint_change_and_plan_change_invalidate_execution(plan, tmp_path):
    path = tmp_path / "plan.json"
    rl_plan.write_plan(plan, path)
    assert not rl_training.execute_plan(path)["training_started"]
    changed = copy.deepcopy(plan)
    changed["training"]["max_steps"] += 1
    with pytest.raises(ValueError, match="stale"):
        rl_plan.validate_plan(changed)
    Path(plan["training"]["checkpoint"], "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="assets changed"):
        rl_training.execute_plan(path, execute=True, worker_runner=lambda *args: pytest.fail("Must not start a worker"))


def test_staged_group_drives_real_native_scoring_and_reuses_common_hidden_seeds(plan, native_verifier, tmp_path):
    result = evaluated(plan, native_verifier, tmp_path / "group")
    assert result["rewards"] == [1, 1]
    assert result["interleaving"]["same_physical_device_verified"]
    _, seen = native_verifier
    assert [row["task"] for row in seen[:3]] == ["qualify_small", "evaluate", "evaluate"]
    assert all(not row["eval_config"]["use_hidden_tests"] for row in seen if row["task"] == "qualify_small")
    final = [row for row in seen if "final_repeat" in row["eval_config"]]
    for repeat in (0, 1):
        seeds = {row["eval_config"]["evaluation_seed"] for row in final if row["eval_config"]["final_repeat"] == repeat}
        assert len(seeds) == 1
    assert rl_rewards.validate_reward(tmp_path / "group/reward.json")["rewards"] == [1, 1]


def test_wrong_rollout_stays_zero_without_poisoning_valid_sibling(plan, native_verifier, tmp_path):
    result = evaluated(plan, native_verifier, tmp_path / "group", [SOURCE, SOURCE + "\n# WRONG\n"])
    assert result["rewards"] == [1, 0]
    assert "precision_contract" in result["components"][1]["failed_checks"]
    _, seen = native_verifier
    assert [row["task"] for row in seen if row["sample_id"] == 1] == ["qualify_small"]
    report = read_json(tmp_path / "group/paper_analysis.json")
    assert report["bounded_gain_macro"] == 0.5 and report["declared_samples"] == 2


def test_static_errors_are_zero_without_executing_that_source(plan, native_verifier, tmp_path):
    result = evaluated(plan, native_verifier, tmp_path / "group", [SOURCE, "this is invalid Python!!!"])
    assert result["rewards"] == [1, 0] and result["components"][1]["failed_checks"] == ["static_source"]
    assert all(row["sample_id"] == 0 for row in native_verifier[1])


@pytest.mark.parametrize("damage", ["candidate", "baseline", "receipt", "hidden", "native_results"])
def test_reward_receipt_cannot_survive_changes_to_its_evidence(plan, native_verifier, tmp_path, damage):
    directory = tmp_path / "group"
    evaluated(plan, native_verifier, directory)
    if damage == "candidate":
        (directory / kernel_name(1, 2)).write_text(SOURCE + "\n# changed\n")
    elif damage == "baseline":
        path = next((rl_rewards.ROOT / "results/timing/RTX_A6000").glob("*.json"))
        path.write_text("{}")
    elif damage == "receipt":
        record = read_json(directory / "reward.json")
        record["rewards"] = [0, 0]
        atomic_json(directory / "reward.json", record)
    elif damage == "hidden":
        (rl_rewards.ROOT / "hidden_tests/level1/2_hidden.py").write_text("changed")
    else:
        atomic_json(directory / "eval_results.json", {})
    with pytest.raises(ValueError):
        rl_rewards.validate_reward(directory / "reward.json")


def test_unknown_or_changed_physical_device_stops_group_reward(plan, native_verifier, tmp_path):
    dispatch, _ = native_verifier
    def changed(config, payload):
        result = dispatch(config, payload)
        if payload["sample_id"] == 1:
            result["metadata"]["device_fingerprint"]["uuid"] = "other-device"
        return result
    result = evaluated(plan, (changed, []), tmp_path / "group")
    assert result["rewards"] == [0, 0] and not result["interleaving"]["same_physical_device_verified"]


def test_raw_speedup_ablation_is_explicitly_distinct_from_the_qualified_score():
    values = rl_rewards.components({"status": "hidden_failed", "score": 0},
        {"compiled": True, "correctness": True, "runtime": 2}, {"mean": 10}, raw_clip=4)
    assert values["qualified"] == 0 and values["raw_speedup"] == 1 and not values["qualification_accepted"]


def test_feedback_tool_exposes_only_standard_feedback_and_respects_candidate_budget(specification, native_verifier, tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    specification["training"]["mode"] = "feedback"
    plan = rl_plan.make_plan(specification, directory=tmp_path / "feedback")
    calls = []
    def development(config, pid, index, remaining, task="evaluate"):
        calls.append(index)
        return {"compiled": True, "correctness": index == 1, "runtime": 1 if index == 1 else -1, "metadata": {}}
    env = rl_training.DevelopmentEnvironment(plan, evaluator=development)
    env.reset(task_key="1/2")
    assert [name for name, member in inspect.getmembers(env, predicate=inspect.ismethod) if not name.startswith("_")] == ["reset", "submit_kernel"]
    first = json.loads(env.submit_kernel(SOURCE))
    assert first["feedback"]["feedback_scope"] == "standard_only"
    env.submit_kernel(SOURCE + "\n# wrong later candidate\n")
    assert json.loads(env.submit_kernel(SOURCE))["status"] == "closed" and calls == [1, 2]
    assert env._source("unbudgeted terminal override") == SOURCE
    with pytest.raises(ValueError, match="held-out"):
        env.reset(task_key="1/26")


def test_journal_charges_interrupted_groups_before_generation(plan):
    journal = rl_training.Journal(plan)
    for step in range(4):
        fresh = rl_training.Journal(plan)
        fresh.reserve(step)
    with pytest.raises(BudgetExhausted):
        rl_training.Journal(plan).reserve(4)
    saved = read_json(journal.path)
    assert len(saved["groups"]) == 4 and all(row["status"] == "generation_started" for row in saved["groups"])


def test_real_trl_configuration_accepts_injected_offline_loaders_without_training(plan, native_verifier, monkeypatch):
    import transformers
    import trl

    calls = []
    tokenizer = SimpleNamespace(pad_token_id=0, apply_chat_template=lambda *a, **k: [1, 2], padding_side="left")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: calls.append(kwargs) or tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: calls.append(kwargs) or object())
    captured = {}
    monkeypatch.setattr(trl, "GRPOTrainer", lambda **kwargs: captured.update(kwargs) or SimpleNamespace())
    journal = rl_training.Journal(plan)
    reward = rl_training.RewardFunction(plan, journal)
    rl_training.native_backend(plan, journal, reward)
    assert all(row["local_files_only"] and row["trust_remote_code"] is False for row in calls)
    assert captured["args"].num_generations == 2 and captured["args"].loss_type == "dapo"
    assert captured["args"].report_to == [] and captured["args"].push_to_hub is False
    assert set(captured["train_dataset"]["task_key"]) == {"1/2"}


def test_fake_trainer_exercises_reward_callback_saving_and_provenance(plan, native_verifier, tmp_path):
    def group(*args, **kwargs):
        return rl_rewards.evaluate_group(*args, **kwargs, dispatch=native_verifier[0])
    def backend_factory(plan, journal, reward):
        class Backend:
            def train(self, **kwargs):
                assert rl_training.os.environ["HF_HUB_OFFLINE"] == "1"
                journal.reserve(0)
                row = rl_plan.dataset_rows(plan)[0]
                rewards = reward(prompts=[row["prompt"]] * 2, completions=[SOURCE, SOURCE + "\n# alternate\n"],
                                 completion_ids=[[1, 2], [1, 2, 3]], task_key=[row["task_key"]] * 2,
                                 prompt_sha256=[row["prompt_sha256"]] * 2)
                assert rewards == [1, 1]
                journal.optimizer_applied(1)
                return SimpleNamespace(global_step=1)
            def save_model(self, directory):
                Path(directory).mkdir(parents=True)
                (Path(directory) / "model.safetensors").write_bytes(b"SIMULATED_CHECKPOINT_NOT_A_MODEL")
        return Backend()
    result = rl_training.run_worker(plan, backend_factory=backend_factory, group_evaluator=group)
    assert result["status"] == "completed" and result["consumed_groups"] == 1
    assert result["generation_usage_complete"] is False
    journal = rl_training.Journal(plan)
    checkpoint = Path(plan["output_directory"]) / "trainer/checkpoint-1"
    checkpoint.mkdir(parents=True)
    atomic_json(checkpoint / "trainer_state.json", {"global_step": 1})
    (checkpoint / "model.safetensors").write_bytes(b"FAKE")
    rl_training.checkpoint_record(plan, journal, checkpoint, 1)
    assert rl_training.validate_checkpoint(plan, journal, checkpoint) == checkpoint
    journal.reserve(1)  # Interrupted work after a valid checkpoint stays in the budget.
    assert rl_training.validate_checkpoint(plan, journal, checkpoint) == checkpoint
    (checkpoint / "model.safetensors").write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="provenance"):
        rl_training.validate_checkpoint(plan, journal, checkpoint)


def test_reward_callback_rejects_heldout_task_without_dispatch(plan):
    journal = rl_training.Journal(plan)
    journal.reserve(0)
    reward = rl_training.RewardFunction(plan, journal, group_evaluator=lambda *a, **k: pytest.fail("Held-out task reached training reward"))
    with pytest.raises(ValueError, match="training group"):
        reward(prompts=[[], []], completions=[SOURCE, SOURCE], completion_ids=[[1], [1]], task_key=["1/26"] * 2,
               prompt_sha256=["fake"] * 2)


def test_callback_rejects_changed_prompts_before_evaluation(plan):
    journal = rl_training.Journal(plan)
    journal.reserve(0)
    row = rl_plan.dataset_rows(plan)[0]
    reward = rl_training.RewardFunction(plan, journal, group_evaluator=lambda *a, **k: pytest.fail("Changed prompts reached verifier"))
    with pytest.raises(ValueError, match="Actual reward prompts"):
        reward(prompts=[[{"role": "user", "content": "different task"}]] * 2, completions=[SOURCE] * 2,
               completion_ids=[[1], [1]], task_key=["1/2"] * 2, prompt_sha256=[row["prompt_sha256"]] * 2)


def test_training_refuses_distributed_groups_before_loading_a_model(plan, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="Distributed"):
        rl_training.run_worker(plan, backend_factory=lambda *a: pytest.fail("Model was loaded"))


def test_prompt_length_guard_runs_before_model_loading(plan, native_verifier, monkeypatch):
    import transformers

    tokenizer = SimpleNamespace(pad_token_id=0, apply_chat_template=lambda *a, **k: list(range(9000)), padding_side="left")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("Overlong prompt loaded model weights"))
    journal = rl_training.Journal(plan)
    with pytest.raises(ValueError, match="silent truncation"):
        rl_training.native_backend(plan, journal, rl_training.RewardFunction(plan, journal))


def test_training_wall_supervisor_does_not_start_by_default_and_retains_attempts(plan, tmp_path):
    path = tmp_path / "plan.json"
    rl_plan.write_plan(plan, path)
    calls = []
    assert not rl_training.execute_plan(path, worker_runner=lambda *args: calls.append(args))["training_started"]
    assert not calls
    result = rl_training.execute_plan(path, execute=True, worker_runner=lambda *args: calls.append(args) or 0)
    assert result["exit_code"] == 1 and result["worker_exit_code"] == 0 and len(calls) == 1
    assert not result["completion_verified"] and "current policy-lineage" in result["error"]
    assert len(result["wall_budget"]["entries"]) == 1 and result["wall_budget"]["compliant"]


def test_lost_supervisor_burns_unobserved_training_wall_reservation(plan, tmp_path):
    from kb_workflows.resource_budgets import Ledger

    path = tmp_path / "plan.json"
    rl_plan.write_plan(plan, path)
    ledger = Ledger(plan["training"]["wall_seconds"], "training_process_wall_seconds")
    ledger.reserve(ledger.remaining)
    atomic_json(Path(plan["output_directory"]) / "training_wall_budget.json", {"plan_sha256": plan["plan_sha256"], "ledger": ledger.snapshot()})
    with pytest.raises(BudgetExhausted, match="unknown"):
        rl_training.execute_plan(path, execute=True, worker_runner=lambda *a: pytest.fail("Restart reset an unknown budget"))


def test_icc_and_attack_intervals_do_not_claim_unknown_or_single_kernel_reliability():
    assert rl_audit.intraclass_correlation([[0, 0, 0], [1, 1, 1]]) == 1
    assert rl_audit.intraclass_correlation([[1, 1], [1, 1]]) is None
    assert rl_audit.intraclass_correlation([[0, 1], [1, 0], [0.5, 0.5]]) < 0
    with pytest.raises(ValueError):
        rl_audit.intraclass_correlation([[1, 2]])
    assert rl_audit.wilson(0, 1)[1] > 0.7
    assert rl_audit.wilson(0, 200)[1] < 0.02
    assert rl_audit.analyze({})["reliability"]["status"] == "not_measured"


def test_attack_audit_requires_the_expected_failed_check(plan, native_verifier, tmp_path):
    directory = tmp_path / "group"
    evaluated(plan, native_verifier, directory, [SOURCE, SOURCE + "\n# WRONG\n"])
    attack = {"receipt": str(directory / "reward.json"), "sample_id": 1, "attack": "zero-output", "expected_invalid": True,
              "expected_block_checks": ["precision_contract"], "rationale": "Simulated candidate violates the numerical contract"}
    result = rl_audit.attacks([attack])
    assert result["qualification_success_rate"] == 0 and not result["interval_target_met"]
    attack["expected_block_checks"] = ["not_measured_memory_safety"]
    with pytest.raises(ValueError, match="infrastructure"):
        rl_audit.attacks([attack])


def test_regrade_plans_preserve_declared_candidates_and_resume_only_current_evidence(plan, native_verifier, tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text(SOURCE)
    regrade = rl_evaluation.make_regrade_plan(plan, {"split": "train", "repeats": 2,
        "cases": [{"case_id": "candidate", "task_key": "1/2", "source": str(source)}]}, directory=tmp_path / "regrades")
    assert not rl_evaluation.execute_regrade(plan, regrade)["experiments_started"]
    calls = []
    def group(*args, **kwargs):
        calls.append(kwargs)
        return rl_rewards.evaluate_group(*args, **kwargs, dispatch=native_verifier[0])
    result = rl_evaluation.execute_regrade(plan, regrade, execute=True, group_evaluator=group)
    assert len(result["cases"]["candidate"]) == len(calls) == 2
    audit = read_json(tmp_path / "regrades/audit_input.json")
    assert len(audit["reliability"]["kernels"][0]["repetitions"]) == 2
    reliability = rl_audit.reliability(audit["reliability"])
    assert reliability["icc_absolute_single"] is None and not reliability["minimum_30_repetitions_met"]
    rl_evaluation.execute_regrade(plan, regrade, execute=True, resume=True, group_evaluator=lambda *a, **k: pytest.fail("Unexpected repeated work"))
    source.write_text(SOURCE + "\n# modified\n")
    with pytest.raises(ValueError, match="source changed"):
        rl_evaluation.execute_regrade(plan, regrade, execute=True, resume=True)


def test_heldout_plan_contains_only_heldout_tasks_and_never_launches_a_job(plan, tmp_path):
    result = rl_evaluation.heldout_plan(plan, {"initial": plan["training"]["checkpoint"]}, directory=tmp_path / "heldout", sessions=2)
    assert len(result["jobs"]) == 2 and not result["experiments_started"]
    assert all(row["config"]["start"] == row["config"]["end"] == 26 for row in result["jobs"])
    assert all(row["config"]["agent_backend"] == "local-transformers" for row in result["jobs"])
    assert result["checkpoints"]["initial"]["sha256"] == plan["initial_checkpoint_sha256"]


def test_heldout_execution_rejects_checkpoint_changes_before_any_job(plan, tmp_path):
    from kb_workflows import study_plan

    evaluation = rl_evaluation.heldout_plan(plan, {"initial": plan["training"]["checkpoint"]}, directory=tmp_path / "heldout", sessions=1)
    path = study_plan.write_plan(evaluation, tmp_path / "heldout")
    assert not study_plan.execute_plan(path)["experiments_started"]
    Path(plan["training"]["checkpoint"], "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checkpoint changed"):
        study_plan.execute_plan(path, execute=True, runner=lambda *a: pytest.fail("Changed checkpoint reached held-out execution"))


def test_small_worker_uses_only_auxiliary_contract_and_current_evidence(plan, tmp_path, monkeypatch):
    from kb_workflows import evaluate_task, strict_validation

    directory = tmp_path / "small"
    config = rl_plan.task_config(plan, "1/2", directory=directory)
    directory.mkdir()
    (directory / kernel_name(1, 2)).write_text(SOURCE)
    payload = engine.task_payload(config, 2, task="qualify_small")
    monkeypatch.setattr(evaluate_task, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_task, "configure_tf32", lambda *args: None)
    import kb_workflows.measurement as measurement
    monkeypatch.setattr(measurement, "device_fingerprint", lambda *args: {"uuid": "simulated"})
    monkeypatch.setattr(strict_validation, "evaluate_contract", lambda *args: {
        name: check("passed", reason="Simulated reviewed auxiliary check") for name in LEVELS["L0"] if name != "correctness"})
    result = evaluate_task.qualify_small(payload)
    assert result["correctness"] and result["runtime"] == -1
    assert result["metadata"]["evidence_receipt"]["requested_level_met"]
    assert result["metadata"]["qualification_scope"].startswith("reviewed_auxiliary_cases")


def test_missing_baselines_stop_before_any_checkpoint_load(plan, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("Missing baselines loaded a checkpoint"))
    journal = rl_training.Journal(plan)
    with pytest.raises(ValueError, match="baselines are missing"):
        rl_training.native_backend(plan, journal, rl_training.RewardFunction(plan, journal))
    assert not rl_rewards.prepare_baselines(plan)["experiments_started"]


def test_missing_journal_cannot_reset_existing_training_attempts(plan):
    journal = rl_training.Journal(plan)
    journal.reserve(0)
    atomic_json(journal.root / "training_result.json", {"status": "interrupted"})
    journal.path.unlink()
    with pytest.raises(ValueError, match="lack their reservation journal"):
        rl_training.Journal(plan)


def test_baseline_change_during_candidate_execution_cannot_produce_a_reward(plan, native_verifier, tmp_path):
    dispatch, _ = native_verifier
    changed = False
    def tamper(config, payload):
        nonlocal changed
        result = dispatch(config, payload)
        if not changed:
            changed = True
            path = next((rl_rewards.ROOT / "results/timing/RTX_A6000").glob("*.json"))
            atomic_json(path, {})
        return result
    with pytest.raises(ValueError, match="Baseline evidence changed"):
        evaluated(plan, (tamper, []), tmp_path / "group")


def test_another_task_baseline_update_does_not_invalidate_this_training_history(plan, native_verifier, tmp_path):
    directory = tmp_path / "group"
    evaluated(plan, native_verifier, directory)
    path = next((rl_rewards.ROOT / "results/timing/RTX_A6000").glob("*.json"))
    data = read_json(path)
    other = select_problems(1, problem_ids=[26])[26].name
    data["level1"][other]["mean"] = 20
    atomic_json(path, data)
    assert rl_rewards.validate_reward(directory / "reward.json")["rewards"] == [1, 1]


def fake_supervised_training(plan, native_verifier, *, plan_path=None, resume=None):
    """Exercise the supervisor/worker protocol without loading or updating a model."""
    path = plan_path or Path(plan["output_directory"]).parent / "training-plan.json"
    rl_plan.write_plan(plan, path)
    def runner(current, checkpoint):
        wall = read_json(Path(current["output_directory"]) / "training_wall_budget.json")
        nonce = next(iter(wall["ledger"]["pending"]))
        def group(*args, **kwargs):
            return rl_rewards.evaluate_group(*args, **kwargs, dispatch=native_verifier[0])
        def factory(current, journal, reward):
            class FakeBackend:
                def train(self, **kwargs):
                    step = len(journal.active_group_ids)
                    journal.reserve(step)
                    row = rl_plan.dataset_rows(current)[0]
                    size = current["training"]["group_size"]
                    reward(prompts=[row["prompt"]] * size, completions=[SOURCE + f"\n# seed {current['training']['seed']}\n"] * size,
                           completion_ids=[[1, 2]] * size, task_key=[row["task_key"]] * size, prompt_sha256=[row["prompt_sha256"]] * size)
                    journal.optimizer_applied(step + 1)
                    return SimpleNamespace(global_step=step + 1)
                def save_model(self, directory):
                    directory = Path(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "model.safetensors").write_text(f"SIMULATED {current['training']['reward_mode']} {current['training']['seed']} {len(journal.active_group_ids)}")
            return FakeBackend()
        result = rl_training.run_worker(current, resume=checkpoint, backend_factory=factory, group_evaluator=group, supervisor_nonce=nonce)
        return 0 if result["status"] == "completed" else 1
    return rl_training.execute_plan(path, execute=True, resume=resume, worker_runner=runner)


def test_policy_lineage_links_native_reward_artifacts_and_supervisor_without_claiming_native_training(plan, native_verifier):
    from kb_workflows import rl_lineage

    result = fake_supervised_training(plan, native_verifier)
    assert result["exit_code"] == 0 and result["completion_verified"]
    checkpoint = Path(result["result"]["final_model"])
    with pytest.raises(ValueError, match="Injected"):
        rl_lineage.validate(checkpoint)
    validated = rl_lineage.validate(checkpoint, allow_injected=True)
    assert validated["supervised"] and not validated["formal_comparison_eligible"]
    assert validated["receipt"]["global_step"] == len(validated["receipt"]["active_group_ids"]) == 1
    assert validated["receipt"]["consumed_groups"] == 1
    group = validated["journal"]["groups"][0]
    Path(group["reward_path"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        rl_lineage.validate(checkpoint, allow_injected=True)


def test_old_training_result_does_not_satisfy_a_new_supervisor_attempt(plan, native_verifier, tmp_path):
    first = fake_supervised_training(plan, native_verifier)
    path = tmp_path / "plan.json"
    rl_plan.write_plan(plan, path)
    second = rl_training.execute_plan(path, execute=True, worker_runner=lambda *args: 0)
    assert first["completion_verified"] and not second["completion_verified"]
    assert second["exit_code"] == 1 and second["result"] == {}
    assert second["supervisor_nonce"] != first["supervisor_nonce"]


def test_resume_keeps_discarded_optimizer_costs_but_restores_the_checkpoint_policy_path(plan, native_verifier):
    from kb_workflows import rl_lineage

    first = fake_supervised_training(plan, native_verifier)
    journal = rl_training.Journal(plan)
    checkpoint = Path(plan["output_directory"]) / "trainer/checkpoint-1"
    checkpoint.mkdir(parents=True)
    atomic_json(checkpoint / "trainer_state.json", {"global_step": 1})
    (checkpoint / "model.safetensors").write_text("SIMULATED")
    rl_training.checkpoint_record(plan, journal, checkpoint, 1)
    discarded = journal.reserve(1)
    directory = journal.root / "rewards" / discarded["id"]
    evaluated(plan, native_verifier, directory)
    journal.settle(directory / "reward.json", completion_tokens=[2, 2], episode_directories=[])
    journal.optimizer_applied(2)
    result = fake_supervised_training(plan, native_verifier, resume=checkpoint)
    lineage = rl_lineage.validate(result["result"]["final_model"], allow_injected=True)
    assert lineage["receipt"]["global_step"] == 2 and lineage["receipt"]["consumed_groups"] == 3
    assert discarded["id"] not in lineage["receipt"]["active_group_ids"]
    assert first["result"]["lineage_sha256"] != result["result"]["lineage_sha256"]


def test_training_effect_uses_training_seeds_and_does_not_promote_a_tiny_positive_ci():
    from kb_workflows.rl_effectiveness import paired_effect, sign_flip

    assert sign_flip([1] * 2)["p_two_sided"] == 0.5
    assert sign_flip([1] * 6)["p_two_sided"] == pytest.approx(1 / 32)
    def data(count):
        return {arm: {str(seed): {"s0": {"1/26": [value, value]}} for seed in range(count)}
                for arm, value in (("qualified", 0.75), ("raw_speedup", 0.25))}
    small = paired_effect(data(2), {"1/26": "I"}, trials=200)
    assert small["ci95"][0] > 0 and not small["positive_effect_supported"]
    assert paired_effect(data(6), {"1/26": "I"}, trials=200)["positive_effect_supported"]
    missing = data(2)
    del missing["raw_speedup"]["1"]
    with pytest.raises(ValueError, match="retained"):
        paired_effect(missing, {"1/26": "I"}, trials=200)
    missing = data(2)
    missing["raw_speedup"]["1"]["s1"] = {"1/26": [0.2, 0.2]}
    with pytest.raises(ValueError, match="same declared evaluation sessions"):
        paired_effect(missing, {"1/26": "I"}, trials=200)
    missing = data(2)
    missing["raw_speedup"]["1"]["s0"]["1/26"].pop()
    with pytest.raises(ValueError, match="same declared candidate slots"):
        paired_effect(missing, {"1/26": "I"}, trials=200)


def test_reward_experiment_is_preregistered_and_unrun_trials_are_not_zero_scores(specification, tmp_path):
    from kb_workflows import rl_effectiveness

    experiment = rl_effectiveness.make_experiment({"base": specification, "seeds": [11, 22], "evaluation_sessions": 1}, directory=tmp_path / "experiment")
    assert len(experiment["jobs"]) == 4 and not experiment["experiments_started"]
    assert not rl_effectiveness.execute_training(experiment)["experiments_started"]
    plan = rl_effectiveness.evaluation_plan(experiment, directory=tmp_path / "evaluation")
    assert plan["status"] == "not_ready" and len(plan["missing_trials"]) == 4
    changed = copy.deepcopy(experiment)
    changed["jobs"].pop()
    changed["experiment_sha256"] = fingerprint({key: value for key, value in changed.items() if key != "experiment_sha256"})
    with pytest.raises(ValueError, match="lost or duplicated"):
        rl_effectiveness.validate_experiment(changed)


def test_lineage_bound_heldout_effect_report_runs_only_analysis_on_simulated_evidence(specification, native_verifier, tmp_path, monkeypatch):
    import hashlib

    from kb_workflows import rl_effectiveness, study_plan
    from kb_workflows.experiment import protocol, write_budget_report
    from kb_workflows.local_model import checkpoint_digest

    experiment = rl_effectiveness.make_experiment({"base": specification, "seeds": [11, 22], "evaluation_sessions": 1}, directory=tmp_path / "experiment")
    rl_effectiveness.write_experiment(experiment, tmp_path / "experiment/experiment.json")
    def executor(path, **kwargs):
        return fake_supervised_training(rl_plan.read_plan(path), native_verifier, plan_path=path)
    executed = rl_effectiveness.execute_training(experiment, execute=True, executor=executor)
    assert len(executed["trials"]) == 4 and all(row["result"]["completion_verified"] for row in executed["trials"])
    assert rl_effectiveness.evaluation_plan(experiment, directory=tmp_path / "evaluation")["status"] == "not_ready"
    evaluation = rl_effectiveness.evaluation_plan(experiment, directory=tmp_path / "evaluation", audit_only=True)
    study_plan.write_plan(evaluation, tmp_path / "evaluation")
    assert rl_effectiveness.analyze(evaluation, trials=200)["status"] == "not_measured"
    executions = []
    for job in evaluation["jobs"]:
        directory = Path(job["path"])
        config = RunConfig(**job["config"])
        arm = next(item["arm"] for item in evaluation["training_lineages"].values() if item["checkpoint"] == config.model)
        text = SOURCE + ("\n# WRONG\n" if arm == "raw_speedup" else "\n# valid\n")
        rl_rewards.evaluate_group(experiment["jobs"][0]["plan"], "1/26", [text], directory=directory, split="heldout", dispatch=native_verifier[0])
        atomic_json(directory / "run_config.json", config.to_dict())
        kernel = directory / kernel_name(1, 26)
        receipt = {"protocol": protocol(config), "model_calls": 1, "candidate_submissions": 1, "input_tokens": 3,
                   "output_tokens": 3, "total_tokens": 6, "generation_wall_seconds": 1, "development_request_wall_seconds": 0,
                   "estimated_cost_usd": None, "models": [config.model], "local_checkpoint_sha256": [checkpoint_digest(config.model)]}
        summary = {"26_0": {"status": "success", "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(), "budget": receipt}}
        atomic_json(directory / "generation_attempts.json", {"slots": {"26_0": [{"status": "success", "budget": receipt}]}})
        atomic_json(directory / "generation_summary.json", summary)
        write_budget_report(config, summary)
        executions.append({"path": str(directory), "session": job["session"], "config_sha256": job["config_sha256"], "exit_code": 0})
    atomic_json(tmp_path / "evaluation/execution.json", {"plan_sha256": evaluation["plan_sha256"], "jobs": executions})
    from kb_workflows import study_data
    monkeypatch.setattr(study_data, "ROOT", rl_rewards.ROOT)
    monkeypatch.setattr(engine, "evaluate_tasks", lambda *a, **k: pytest.fail("Analysis launched a workload"))
    report = rl_effectiveness.analyze(evaluation, trials=200)
    assert report["status"] == "analyzed" and report["effect"]["qualified_minus_raw"] == 1
    assert not report["formal_comparison_eligible"] and not report["heldout_improvement_supported"]
    assert len(report["training_blocked_checks"]["qualified"]) == 2
    budget_path = Path(evaluation["jobs"][0]["path"]) / "generation_budget.json"
    budget = read_json(budget_path)
    budget["slots"]["26_0"]["local_checkpoint_sha256"] = None
    atomic_json(budget_path, budget)
    with pytest.raises(ValueError, match="different local model weights"):
        rl_effectiveness.analyze(evaluation, trials=200)


def test_reordered_regrade_cases_cannot_reuse_old_sample_assignments(plan, native_verifier, tmp_path):
    cases = []
    for index in (0, 1):
        source = tmp_path / f"candidate{index}.py"
        source.write_text(SOURCE + f"\n# version {index}\n")
        cases.append({"case_id": f"candidate{index}", "task_key": "1/2", "source": str(source)})
    spec = {"split": "train", "repeats": 1, "cases": cases}
    directory = tmp_path / "regrade"
    initial = rl_evaluation.make_regrade_plan(plan, spec, directory=directory)
    rl_evaluation.execute_regrade(plan, initial, execute=True,
        group_evaluator=lambda *a, **k: rl_rewards.evaluate_group(*a, **k, dispatch=native_verifier[0]))
    spec["cases"] = list(reversed(cases))
    reordered = rl_evaluation.make_regrade_plan(plan, spec, directory=directory)
    with pytest.raises(ValueError, match="sources differ"):
        rl_evaluation.execute_regrade(plan, reordered, execute=True, resume=True)
