"""CPU format/control fixtures only; never measured reference qualification."""

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_input_calibration import CPUAdapter, calibration, synthetic_format_fixture

from kb_workflows import engine, experiment, generation, input_calibration, input_calibration_task
from kb_workflows import reference_population as population
from kb_workflows.common import ROOT, atomic_json, fingerprint, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.suites import selection_receipt
from src import eval as native


def reseal(record):
    record["receipt_sha256"] = fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})


@pytest.fixture
def cohort(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    good, _ = calibration(tmp_path, 90)
    config = RunConfig(generation="local", hardware="RTX_A6000", start=89, end=90, precision="fp32",
                       oracle_memory_mb=8, runs_dir=str(tmp_path), run_name="population", hidden_tests=False).validate()
    request = engine.task_payload(config, 89, task="calibrate_inputs")

    def loader(source, context):
        Original, init, _ = native.load_original_model_and_inputs(source, context)

        class BadFixture(Original):
            def forward(self, value):
                return torch.zeros_like(value)

        return BadFixture, init, lambda: [torch.randn(2, 256)]

    bad = input_calibration_task.calibrate(request, adapter=CPUAdapter(), loader=loader)["record"]
    assert not bad["qualified"] and len(bad["attempts"]) == 3
    # Deliberately unscored serialization fixtures. These files never leave tmp_path.
    bundle = {"protocol": input_calibration.BUNDLE, "records": {
        "1/89/fp32": synthetic_format_fixture(bad), "1/90/fp32": synthetic_format_fixture(good)}, "errors": {}}
    calibration_path = tmp_path / "unscored-calibration-format-fixture.json"
    atomic_json(calibration_path, bundle)
    config.input_calibration = str(calibration_path)
    config.validate()
    manifest = population.build(config)
    manifest_path = tmp_path / "unscored-population-format-fixture.json"
    atomic_json(manifest_path, manifest)
    config.reference_population = str(manifest_path)
    config.validate()
    yield config, manifest, bundle
    torch.set_num_threads(previous)


def test_reference_eligibility_freezes_complete_ledger_and_first_recipe_without_candidates(cohort):
    config, manifest, _ = cohort
    assert manifest["qualified"] == [90] and manifest["excluded"] == [89]
    assert [row["problem_id"] for row in manifest["ledger"]] == [89, 90]
    failed = manifest["ledger"][0]
    assert failed["qualified"] is False and failed["selected_recipe"] is None
    assert all("reference_precision_failed" in row["reasons"] for row in failed["attempts"])
    assert all("output_samples_not_distinct" in row["reasons"] for row in failed["attempts"])
    assert manifest["candidate_executed"] is False and manifest["timings_collected"] is False
    assert not config.run_dir.exists()
    assert list(population.select_run_problems(config)) == [90]
    population.freeze_run(config)
    snapshot = read_json(config.run_dir / population.SNAPSHOT)
    assert snapshot["manifest"] == manifest and snapshot["active_problem_ids"] == [90]
    assert snapshot["num_samples"] == 1
    assert selection_receipt(config)["reference_population"]["excluded"] == [89]
    assert experiment.protocol(config)["reference_population"]["manifest_sha256"] == manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="outside"):
        engine.task_payload(config, 89, task="baseline")


@pytest.mark.parametrize("damage", ["missing", "prefix", "worker", "error", "unmeasured", "missing_probe", "memory", "simulated", "all_bad", "source", "device"])
def test_incomplete_or_rebound_calibration_is_not_an_exclusion(cohort, damage):
    config, _, bundle = cohort
    config.reference_population = ""
    bad, good = bundle["records"].values()
    if damage == "missing":
        del bundle["records"]["1/89/fp32"]
    elif damage == "prefix":
        bad["attempts"].pop()
    elif damage == "worker":
        bundle["errors"]["inputs-89"] = {"error": "worker unavailable"}
    elif damage == "error":
        bad["attempts"][0]["cases"][0]["errors"].append("CUDA OOM")
    elif damage == "unmeasured":
        bad["attempts"][0]["cases"][0]["status"] = "not_measured"
    elif damage == "missing_probe":
        bad["attempts"][0]["cases"][0].pop("output_probe_sha256")
    elif damage == "memory":
        bad["attempts"][0]["cases"][0]["estimated_host_bytes"] = 10**12
    elif damage == "simulated":
        bad["simulated_execution"] = True
    elif damage == "source":
        bad["source_sha256"] = "0" * 64
    elif damage == "device":
        bad["device"]["uuid"] = "changed"
    else:
        for attempt in good["attempts"]:
            for row in attempt["cases"]:
                row["output_probe_sha256"] = "0" * 64
        # A short failed prefix is also insufficient to freeze an empty cohort.
        good.update(qualified=False, selected_recipe=None)
    reseal(bad)
    reseal(good)
    atomic_json(Path(config.input_calibration), bundle)
    with pytest.raises(ValueError):
        population.build(config)
    assert not config.run_dir.exists()


@pytest.mark.parametrize("damage", ["ledger", "excluded", "qualified", "selection", "evaluator", "policy", "calibration"])
def test_resealed_manifest_cannot_redefine_reference_eligibility(cohort, damage):
    config, manifest, bundle = cohort
    if damage == "ledger":
        manifest["ledger"][0]["attempts"].clear()
    elif damage == "excluded":
        manifest["excluded"].clear()
    elif damage == "qualified":
        manifest["qualified"].append(89)
    elif damage == "selection":
        manifest["selection"]["references"].pop("89")
    elif damage == "evaluator":
        manifest["evaluator_sha256"] = "0" * 64
    elif damage == "policy":
        manifest["post_candidate_failures"] = "remove_failures"
    else:
        bundle["unrelated_metadata"] = "changed after preregistration"
        atomic_json(Path(config.input_calibration), bundle)
    manifest["manifest_sha256"] = fingerprint({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    atomic_json(Path(config.reference_population), manifest)
    with pytest.raises(ValueError):
        population.select_run_problems(config)


@pytest.mark.parametrize("artifact", ["level_1_problem_90_sample_0_kernel.py", "work", "generation_summary.json", "eval_results.json", "final_manifest.json", "paper_analysis.json"])
def test_population_cannot_be_adopted_after_candidates_or_results(cohort, artifact):
    config, _, _ = cohort
    config.run_dir.mkdir()
    (config.run_dir / artifact).write_text("{}")
    with pytest.raises(ValueError, match="after candidate"):
        population.freeze_run(config)
    assert not (config.run_dir / population.SNAPSHOT).exists()


@pytest.mark.parametrize("change", ["manifest", "samples", "scope", "disable"])
def test_frozen_run_keeps_denominator_even_when_candidate_fails(cohort, change):
    config, _, _ = cohort
    population.freeze_run(config)
    atomic_json(config.run_dir / "eval_results.json", {"90_0": {"compiled": False, "correctness": False}})
    if change == "samples":
        config.num_samples = 2
    elif change == "scope":
        config.start = config.end = 90
        config.reference_population_scope = "task"
    elif change == "disable":
        config.reference_population = ""
    else:
        snapshot = config.run_dir / population.SNAPSHOT
        data = read_json(snapshot)
        data["active_problem_ids"] = []
        atomic_json(snapshot, data)
    with pytest.raises(ValueError, match="Frozen run"):
        population.select_run_problems(config)


def test_missing_candidate_remains_a_zero_scored_slot_and_excluded_reference_is_visible(cohort, monkeypatch):
    from kb_workflows import paper_analysis

    config, manifest, _ = cohort
    population.freeze_run(config)
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    monkeypatch.setattr(paper_analysis, "ROOT", config.run_dir.parent / "no-baselines")
    report = paper_analysis.analyze_paper(config)
    assert report["problem_ids"] == [90] and report["declared_samples"] == 1
    assert len(report["details"]) == 1 and report["details"][0]["score"] == 0
    assert report["suite"]["reference_population"]["excluded"] == manifest["excluded"]


def test_native_missing_candidate_and_baseline_scheduler_use_frozen_population(cohort, monkeypatch):
    config, _, _ = cohort
    engine.evaluate_run(config)  # No candidate => no device worker is launched.
    results = read_json(config.run_dir / "eval_results.json")
    assert list(results) == ["90_0"] and results["90_0"]["correctness"] is False
    passes = read_json(config.run_dir / "pass_at_k_results.json")
    assert list(passes["problems"]) == ["90"]
    calls = []

    def no_execution(_config, tasks):
        calls.extend(tasks)
        return iter(())

    monkeypatch.setattr(engine, "evaluate_tasks", no_execution)
    monkeypatch.setattr(engine, "ROOT", config.run_dir.parent / "unscored-baseline-fixture")
    config.baseline_suite = False
    engine.record_baseline(config)
    assert calls and {payload["problem_id"] for _, payload in calls} == {90}
    assert all(payload["eval_config"]["reference_population"] for _, payload in calls)


def test_regrade_archives_parent_population_and_preserves_excluded_endpoint(cohort, tmp_path):
    from kb_workflows import regrade

    config, manifest, _ = cohort
    population.freeze_run(config)
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    atomic_json(config.run_dir / "paper_analysis.json", {"problem_ids": [90], "declared_samples": 1})
    verifier = replace(config, run_name="unscored-regrade", gpu="A6000", gpu_arch="Ampere")
    target = tmp_path / "regrade"
    plan = regrade.prepare(config.run_dir, verifier, output=target)
    assert plan["source_population"] == [90] and list(plan["slots"]) == ["90_0"]
    assert plan["config"]["start"] == 89 and plan["config"]["end"] == 90
    assert "reference_population.json" in plan["history_files"]
    assert "reference_population" in plan["verifier_inputs"]
    assert read_json(target / population.SNAPSHOT)["manifest"] == manifest
    regrade.validate(plan)


def test_parent_and_private_agent_subset_bind_the_same_population(cohort):
    config, manifest, _ = cohort
    parent = engine.task_payload(config, 90, task="baseline")
    directory = generation.prepare_job(config, 90, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    assert child.reference_population == ""
    child_payload = engine.task_payload(child, 90, task="baseline")
    assert child_payload["eval_config"]["reference_population"] == parent["eval_config"]["reference_population"]
    subset = read_json(directory / "input_calibration.json")
    assert list(subset["records"]) == ["1/90/fp32"]
    assert subset["population_manifest"] == manifest and subset["delegated_task"] == "1/90/fp32"
    assert population.validate_inline(child_payload, child_payload["eval_config"]["input_profile"]) == manifest["manifest_sha256"]
    child.start = 89
    with pytest.raises(ValueError, match="restricted"):
        child.validate()


def test_explicit_single_task_delegation_preserves_full_parent_ledger(cohort):
    config, manifest, _ = cohort
    child = replace(config, start=90, end=90, reference_population_scope="task", run_name="rl-group", num_samples=4).validate()
    assert list(population.select_run_problems(child)) == [90]
    assert population.manifest_for(child) == manifest
    population.freeze_run(child)
    assert read_json(child.run_dir / population.SNAPSHOT)["num_samples"] == 4
    child.start = child.end = 89
    with pytest.raises(ValueError, match="one qualified"):
        child.validate()


def test_harbor_private_export_keeps_parent_population_and_sdk_light_replay(cohort, tmp_path):
    from kb_workflows.harbor import export_tasks

    config, manifest, _ = cohort
    config.hidden_tests = True
    output = export_tasks(config)
    assert sorted(path.name for path in output.iterdir()) == ["l1-p90-s0"]
    private = output / "l1-p90-s0" / "tests"
    subset = read_json(private / "input_calibration.json")
    assert subset["population_manifest"] == manifest
    spec = read_json(private / "config.json")
    assert spec["config"]["reference_population"] == ""
    assert (private / "repo/kb_workflows/reference_population.py").read_bytes() == (ROOT / "kb_workflows/reference_population.py").read_bytes()
    assert (private / "repo/scripts/freeze_reference_population.py").is_file()
    assert not (output / "l1-p90-s0/environment/input_calibration.json").exists()
    child = RunConfig(**spec["config"])
    child.input_calibration = str(private / "input_calibration.json")
    child.runs_dir = str(tmp_path / "harbor-replay")
    child.validate()
    assert population.binding(child, 90) == population.binding(config, 90)


def test_rl_task_config_delegates_qualified_parent_and_rejects_excluded_task(cohort, tmp_path):
    from kb_workflows.rl_plan import task_config, verifier_inputs

    config, manifest, _ = cohort
    plan = {"verifier_config": config.to_dict(), "training": {"mode": "feedback"}, "tasks": {
        "1/90": {"level": 1, "problem_id": 90}, "1/89": {"level": 1, "problem_id": 89}}}
    child = task_config(plan, "1/90", directory=tmp_path / "rl-group", samples=4)
    assert child.reference_population_scope == "task"
    assert population.manifest_for(child) == manifest
    assert "reference_population" in verifier_inputs(config.to_dict())
    with pytest.raises(ValueError, match="one qualified"):
        task_config(plan, "1/89", directory=tmp_path / "bad-group")
    assert not (tmp_path / "rl-group").exists()


def test_final_manifest_and_budget_bind_population_without_shrinking_failed_slots(cohort):
    from kb_workflows import final_verification

    config, manifest, _ = cohort
    config.final_repeats = 2
    population.freeze_run(config)
    frozen = final_verification.freeze(config)
    assert list(frozen["slots"]) == ["90_0"]
    assert frozen["slots"]["90_0"]["source_sha256"] is None
    budget = experiment.write_budget_report(config, {})
    assert budget["declared_slots"] == 1 and list(budget["slots"]) == ["90_0"]
    assert budget["protocol"]["reference_population"]["manifest_sha256"] == manifest["manifest_sha256"]
    assert experiment.existing_budget(config)["provenance_valid"]


def test_study_plan_binds_reference_population_before_any_runner(cohort, tmp_path):
    from kb_workflows import study_plan

    config, manifest, _ = cohort
    options = config.to_dict()
    options.update(gpu="A6000", gpu_arch="Ampere")
    plan = study_plan.make_plan({"base_config": options, "models": ["unrun-model"], "methods": ["one-shot"],
                                  "sessions": 1, "vary": "wall", "budgets": [100]}, directory=tmp_path / "study")
    assert plan["jobs"][0]["problem_ids"] == [90]
    assert plan["jobs"][0]["reference_population"]["manifest_sha256"] == manifest["manifest_sha256"]
    path = study_plan.write_plan(plan, tmp_path / "study")
    assert study_plan.execute_plan(path)["experiments_started"] is False
    changed = read_json(path)
    changed["jobs"][0]["reference_population"]["manifest_sha256"] = "0" * 64
    changed["plan_sha256"] = fingerprint({key: value for key, value in changed.items() if key != "plan_sha256"})
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="population changed"):
        study_plan.execute_plan(path)


def test_runtime_population_receipt_is_required_and_task_bound(cohort, monkeypatch):
    config, manifest, _ = cohort
    payload = engine.task_payload(config, 90, task="baseline")
    payload["task"] = "evaluate"
    payload["eval_config"].update(measure_performance=False)
    monkeypatch.setattr("kb_workflows.measurement.device_fingerprint", lambda _: copy.deepcopy(payload["eval_config"]["input_profile"]["device"]))

    @input_calibration.task_policy
    def fake_worker(request):
        return {"correctness": True, "metadata": {}, "device_fingerprint": request["eval_config"]["input_profile"]["device"]}

    result = fake_worker(payload)
    # This software fixture does not call a nominal factory: test only population binding.
    receipt = result["metadata"]["input_calibration"]
    assert receipt["reference_population_sha256"] == manifest["manifest_sha256"]
    payload["task"] = "qualify_small"
    assert input_calibration.issue(result, payload) is None
    receipt.pop("reference_population_sha256")
    assert input_calibration.issue(result, payload) == "calibrated_input_execution_evidence_missing"
    payload["eval_config"]["reference_population"]["active_key"] = "1/89/fp32"
    with pytest.raises(ValueError, match="outside"):
        input_calibration.validate_inline(payload)


def test_population_freezer_and_preview_are_sdk_light_and_do_not_run_experiments(cohort, tmp_path):
    config, _, _ = cohort
    config.reference_population = ""
    path = tmp_path / "preview.toml"
    path.write_text('[run]\nlevel=1\nstart=89\nend=90\ngeneration="local"\nhidden_tests=false\ninput_calibration=' + json.dumps(config.input_calibration) + '\n')
    output = tmp_path / "new-population.json"
    script = f'''import json,sys
sys.path.insert(0,{str(ROOT)!r})
from scripts.freeze_reference_population import main
assert main(["--config",{str(path)!r},"--output",{str(output)!r}]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["experiments_started"] is False and not output.exists()
    script = script.replace(']) == 0', ',"--write"]) == 0')
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["written"] is True and output.exists()
    path.write_text(path.read_text() + 'reference_population="new-population.json"\n')
    parsed = parse_config(["--config", str(path), "--dry-run"])
    assert parsed.reference_population == str(output)
    assert list(population.select_run_problems(parsed)) == [90]
    assert not parsed.run_dir.exists()
