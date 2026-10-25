"""Artifact provenance and simulated regrading, without model/device experiments."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from kb_workflows.analysis.experiment import protocol
from kb_workflows.common import atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig
from kb_workflows.runtime import regrade


@pytest.fixture
def source(tmp_path):
    config = RunConfig(generation="local", hardware="RTX_A6000", run_name="original", runs_dir=str(tmp_path),
                       start=1, end=2, num_samples=2).validate()
    config.run_dir.mkdir()
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    atomic_json(config.run_dir / "paper_analysis.json", {"problem_ids": [1, 2], "declared_samples": 2, "score_macro": 0.8})
    atomic_json(config.run_dir / "generation_budget.json", {"protocol": {"mode": "feedback", "model": "SIMULATED historical model"},
                                                           "totals": {"model_calls": 8, "estimated_cost_usd": None}})
    atomic_json(config.run_dir / "work/p1_s0/trajectory.json", {"messages": ["SIMULATED historical trajectory"]})
    for pid, sid in ((1, 0), (1, 1), (2, 0)):
        (config.run_dir / kernel_name(1, pid, sid)).write_text("# simulated identical frozen candidate\n")
    return config


def target():
    return RunConfig(generation="local", hardware="RTX_A6000").validate()


def tree_hashes(path):
    return {str(p.relative_to(path)): regrade.digest(p) for p in path.rglob("*") if p.is_file()}


def test_regrade_freezes_all_declared_slots_preserves_missing_and_does_not_modify_source(source, tmp_path):
    before = tree_hashes(source.run_dir)
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    config = regrade.validate(plan)
    assert plan["source_population"] == [1, 2] and len(plan["slots"]) == 4
    assert plan["slots"]["2_1"]["status"] == "missing"
    assert not (config.run_dir / kernel_name(1, 2, 1)).exists()
    assert len(list((config.run_dir / "frozen_candidates").iterdir())) == 1
    assert regrade.execute(plan) == {"experiments_started": False, "declared_slots": 4, "plan_sha256": plan["plan_sha256"]}
    assert tree_hashes(source.run_dir) == before
    assert not (config.run_dir / "regrade_execution.json").exists()
    assert config.baseline != target().baseline
    assert config.model == "" and config.agent_token_limit == config.agent_output_token_limit == config.agent_device_seconds == 0
    assert protocol(config)["mode"] == "artifact-regrade" and protocol(config)["new_model_calls"] == 0
    assert not (config.run_dir / "generation_budget.json").exists()
    assert read_json(config.run_dir / "source_history/generation_budget.json")["totals"]["model_calls"] == 8
    assert read_json(config.run_dir / "source_history/work/p1_s0/trajectory.json")["messages"] == ["SIMULATED historical trajectory"]


@pytest.mark.parametrize("damage", ["candidate", "frozen_candidate", "history", "reference", "evaluator", "population", "add_missing"])
def test_stale_or_changed_frozen_regrade_inputs_cannot_start(source, tmp_path, damage, monkeypatch):
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    root = Path(plan["output_directory"])
    if damage == "candidate":
        path = root / plan["slots"]["1_0"]["filename"]
        path.chmod(0o644)
        path.write_text("# edited")
    elif damage == "frozen_candidate":
        path = root / "frozen_candidates" / (plan["slots"]["1_0"]["source_sha256"] + ".py")
        path.chmod(0o644)
        path.write_text("# edited")
    elif damage == "history":
        path = root / "source_history/generation_budget.json"
        path.chmod(0o644)
        path.write_text("{}")
    elif damage == "evaluator":
        monkeypatch.setattr(regrade, "evaluator_digest", lambda: "changed")
    elif damage == "add_missing":
        (root / kernel_name(1, 2, 1)).write_text("# later candidate must not fill a missing original slot")
    else:
        if damage == "reference":
            plan["references"]["1"]["sha256"] = "f" * 64
        else:
            plan["slots"].pop("2_1")
        plan["plan_sha256"] = fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
        atomic_json(root / "regrade_manifest.json", plan)
    with pytest.raises(ValueError):
        regrade.execute(plan)
    assert not (root / "regrade_execution.json").exists()


def test_output_cannot_overwrite_source_or_an_existing_plan(source, tmp_path):
    with pytest.raises(ValueError):
        regrade.prepare(source.run_dir, target(), output=source.run_dir / "nested")
    output = tmp_path / "regraded"
    regrade.prepare(source.run_dir, target(), output=output)
    with pytest.raises(ValueError):
        regrade.prepare(source.run_dir, target(), output=output)


def test_frozen_plan_can_regrade_after_original_run_was_archived(source, tmp_path):
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    shutil.rmtree(source.run_dir)
    assert len(regrade.validate(plan).to_dict()) > 0
    assert regrade.execute(plan)["experiments_started"] is False


def test_source_population_must_be_recorded_and_complete(source, tmp_path):
    (source.run_dir / "paper_analysis.json").unlink()
    with pytest.raises(ValueError, match="population"):
        regrade.prepare(source.run_dir, target(), output=tmp_path / "missing-population")
    atomic_json(source.run_dir / "eval_results.json", {"1_0": {}, "1_1": {}, "2_0": {}})
    with pytest.raises(ValueError, match="lost declared"):
        regrade.prepare(source.run_dir, target(), output=tmp_path / "partial-population")


def test_current_selection_cannot_silently_shrink_historical_population(source, tmp_path):
    value = read_json(source.run_dir / "paper_analysis.json")
    value["problem_ids"] = [1, 3]
    atomic_json(source.run_dir / "paper_analysis.json", value)
    with pytest.raises(ValueError, match="selection"):
        regrade.prepare(source.run_dir, target(), output=tmp_path / "selection")


def test_new_verifier_inputs_are_frozen_by_content_and_mutation_rejects_the_plan(source, tmp_path):
    path = tmp_path / "policies.json"
    atomic_json(path, {"protocol": "kernelbench_pro_determinism_policies_v1", "tasks": {}})
    verifier = replace(target(), determinism_policy_file=str(path)).validate()
    plan = regrade.prepare(source.run_dir, verifier, output=tmp_path / "regraded")
    config = regrade.validate(plan)
    path.write_text("{}")
    assert regrade.validate(plan)
    frozen = Path(config.determinism_policy_file)
    frozen.chmod(0o644)
    frozen.write_text("{}")
    with pytest.raises(ValueError):
        regrade.validate(plan)


def test_simulated_execution_runs_only_verification_phases_and_cannot_establish_formal_regrade_evidence(source, tmp_path):
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    seen = []
    def dispatch(phase, config):
        seen.append(phase)
        if phase == "evaluate":
            for name in regrade.measurement_files(config):
                atomic_json(config.run_dir / name, {"SIMULATED": True})
        if phase == "analyze":
            atomic_json(config.run_dir / "paper_analysis.json", {"SIMULATED": True})
    result = regrade.execute(plan, execute=True, dispatch=dispatch)
    assert seen == ["baseline", "evaluate", "analyze"] and result["status"] == "completed"
    assert result["new_model_calls"] == 0 and result["simulated_execution"]
    config = regrade.validate(plan)
    assert regrade.issue(config) == "regrade_measurement_proof_missing_changed_or_simulated"
    with pytest.raises(ValueError, match="already exists"):
        regrade.execute(plan, execute=True, dispatch=dispatch)


def test_regrade_failure_retains_execution_cost_journal_without_retrying_or_generating(source, tmp_path):
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    def dispatch(phase, config):
        if phase == "evaluate":
            raise RuntimeError("SIMULATED infrastructure failure")
    with pytest.raises(RuntimeError):
        regrade.execute(plan, execute=True, dispatch=dispatch)
    record = read_json(Path(plan["output_directory"]) / "regrade_execution.json")
    assert record["status"] == "failed" and record["stages"][0]["name"] == "baseline"
    assert not (Path(plan["output_directory"]) / "regrade_measurements.json").exists()


def test_current_config_cannot_relabel_a_frozen_regrade_or_reduce_its_denominator(source, tmp_path):
    plan = regrade.prepare(source.run_dir, target(), output=tmp_path / "regraded")
    config = regrade.validate(plan)
    assert protocol(replace(config, stage="analyze"))["mode"] == "artifact-regrade"
    with pytest.raises(ValueError, match="configuration differs"):
        protocol(replace(config, end=1))


def test_cli_native_planning_and_run_preview_never_launch_models_or_verifiers(source, tmp_path, monkeypatch, capsys):
    from kb_workflows import engine
    from kb_workflows.agents import generation
    from scripts.regrade_artifacts import main

    monkeypatch.setattr(engine, "record_baseline", lambda *a, **kw: pytest.fail("Preview must not start device verification"))
    monkeypatch.setattr(generation, "generate_run", lambda *a, **kw: pytest.fail("Regrading must not start a model"))
    config = tmp_path / "target.toml"
    config.write_text('[gpu]\nprofile="RTX_A6000"\n[run]\ngeneration="local"\n')
    output = tmp_path / "cli-regraded"
    assert main(["native-plan", "--source-run", str(source.run_dir), "--config", str(config), "--output", str(output)]) == 0
    assert main(["native-run", "--manifest", str(output / "regrade_manifest.json")]) == 0
    assert '"experiments_started": false' in capsys.readouterr().out
    assert not (output / "eval_results.json").exists()
    assert not (output / "regrade_execution.json").exists()
