"""Scoring and recovery contracts using persisted results and real workflow IDs."""

import copy
import shutil

import pytest

from kb_workflows import common, engine, paper_analysis
from kb_workflows.common import atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.measurement import timing_stats


@pytest.fixture
def run(tmp_path, monkeypatch):
    reference = tmp_path / "KernelBench/level1/1_problem.py"
    reference.parent.mkdir(parents=True)
    reference.write_text("reference source")
    hidden = tmp_path / "hidden_tests/level1/1_hidden.py"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("hidden source")
    config = RunConfig(run_name="scoring", runs_dir=str(tmp_path / "runs"), hardware="test",
                       start=1, end=1, num_samples=3, baseline_suite=False).validate()
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(paper_analysis, "ROOT", tmp_path)
    # Default parameters on select_problems were bound to the original checkout.
    def select(*args, **kwargs):
        return {1: reference}
    monkeypatch.setattr(engine, "select_problems", select)
    monkeypatch.setattr(paper_analysis, "select_problems", select)
    monkeypatch.setattr(engine, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(common, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(paper_analysis, "evaluator_digest", lambda: "current")
    config.run_dir.mkdir(parents=True)
    for sid in range(config.num_samples):
        (config.run_dir / kernel_name(1, 1, sid)).write_text(f"candidate source {sid}")
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    device = {"available": True, "name": "GPU", "capability": [9, 0], "total_memory": 10**10,
              "multiprocessors": 100, "l2_bytes": 10**6, "torch_version": "test", "cuda_version": "test", "hip_version": None}
    receipt = {"output_consumed": True, "fresh_inputs": True, "varied_input_contents": True,
               "timed_outputs_verified": True, "input_seeds": [101, 102, 103]}
    payload = engine.task_payload(config, 1, task="baseline")
    baseline = {**timing_stats([10, 10, 10]), "baseline_correctness": True, "precision": "fp32", "tf32": True,
                "reference_sha256": fingerprint({"source": reference.read_text()}), "input_signature": [{"shape": [4]}],
                "device_fingerprint": device, "cold_start_ms": 10, "timing_receipt": copy.deepcopy(receipt),
                "metadata": {"evaluation_config": payload["eval_config"], "workflow_signature": fingerprint(payload)}}
    baseline_path = tmp_path / "results/timing/test" / f"{config.baseline}.json"
    atomic_json(baseline_path, {"level1": {reference.name: baseline}})
    standard, secret = {}, {}
    for sid in range(config.num_samples):
        for hidden_, results in [(False, standard), (True, secret)]:
            payload = engine.task_payload(config, 1, sid, hidden=hidden_)
            results[f"1_{sid}"] = {"compiled": True, "correctness": True, "runtime": 5,
                                   "runtime_stats": timing_stats([5, 5, 5]),
                                   "metadata": {**copy.deepcopy(baseline["metadata"]),
                                                "evaluation_config": payload["eval_config"], "workflow_signature": fingerprint(payload),
                                                "reference_sha256": baseline["reference_sha256"], "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                                                "input_signature": baseline["input_signature"], "device_fingerprint": copy.deepcopy(device),
                                                "timing_method": "synchronized_wall", "precision": "fp32", "cold_start_ms": 5,
                                                "timing_receipt": copy.deepcopy(receipt)}}
    def save():
        atomic_json(baseline_path, {"level1": {reference.name: baseline}})
        atomic_json(config.run_dir / "eval_results.json", standard)
        atomic_json(config.run_dir / "eval_results_hidden.json", secret)

    save()
    return config, baseline, standard, secret, save


def test_failures_and_missing_slots_stay_in_denominator(run):
    config, _, standard, _, save = run
    standard["1_1"]["correctness"] = False
    del standard["1_2"]
    save()
    report = paper_analysis.analyze_paper(config)
    assert report["bounded_gain_macro"] == pytest.approx(1 / 3)
    assert len(report["details"]) == 3
    assert [r["status"] for r in report["details"]] == ["verified", "failed_or_missing", "failed_or_missing"]
    assert report["anti_cheat_verified"] is False
    assert common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]


@pytest.mark.parametrize("kind", ["kernel", "reference", "hidden", "version", "settings", "timing_receipt", "samples", "baseline", "incomplete_suite", "precision", "device", "software", "cold"])
def test_missing_or_incompatible_evidence_blocks_scoring(run, kind):
    config, baseline, standard, secret, save = run
    raw = standard["1_0"]
    if kind == "kernel":
        (config.run_dir / kernel_name(1, 1)).write_text("changed kernel")
    elif kind == "reference":
        (common.ROOT / "KernelBench/level1/1_problem.py").write_text("changed reference")
    elif kind == "hidden":
        (common.ROOT / "hidden_tests/level1/1_hidden.py").write_text("changed hidden source")
    elif kind == "version":
        raw["metadata"]["evaluation_config"]["evaluator_sha256"] = "old"
    elif kind == "settings":
        raw["metadata"]["evaluation_config"]["fp32_tolerance"] = 0.5
    elif kind == "timing_receipt":
        raw["metadata"]["timing_receipt"].pop("timed_outputs_verified")
    elif kind == "samples":
        raw["runtime_stats"]["samples_ms"] = [50, 50, 50]
    elif kind == "baseline":
        baseline["metadata"]["workflow_signature"] = "old"
    elif kind == "incomplete_suite":
        baseline.update(baseline_suite=True, suite_complete=False)
    elif kind == "precision":
        baseline["precision"] = "bf16"
    elif kind == "device":
        raw["metadata"]["device_fingerprint"]["name"] = "other GPU"
    elif kind == "software":
        raw["metadata"]["device_fingerprint"]["torch_version"] = "other torch"
    else:
        raw["metadata"].pop("cold_start_ms")
    save()
    row = paper_analysis.analyze_paper(config)["details"][0]
    assert row["score"] == 0 and row["status"] != "verified"
    assert not common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]


@pytest.mark.parametrize("kind", ["kernel", "reference", "hidden", "raw_result", "baseline", "run_config", "gate_file"])
def test_legacy_gate_rejects_evidence_changed_after_analysis(run, kind):
    config, baseline, standard, _, save = run
    paper_analysis.analyze_paper(config)
    assert common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]
    if kind == "kernel":
        (config.run_dir / kernel_name(1, 1)).write_text("changed")
    elif kind == "reference":
        (common.ROOT / "KernelBench/level1/1_problem.py").write_text("changed")
    elif kind == "hidden":
        (common.ROOT / "hidden_tests/level1/1_hidden.py").write_text("changed")
    elif kind == "raw_result":
        standard["1_0"]["runtime"] = 0.01
        save()
    elif kind == "baseline":
        baseline["mean"] = 100
        save()
    elif kind == "run_config":
        atomic_json(config.run_dir / "run_config.json", {**config.to_dict(), "precision": "bf16"})
    else:
        (config.run_dir / "eval_results_scored.json").unlink()
    assert not common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]


def test_uncertainty_and_declared_cold_budget_affect_score(run):
    config, _, standard, _, save = run
    # Preserve actual steady samples while varying uncertainty and cold cost.
    standard["1_0"]["runtime_stats"] = timing_stats([0.1, 5, 9.9])
    standard["1_1"]["metadata"]["cold_start_ms"] = 100000
    save()
    report = paper_analysis.analyze_paper(config)
    assert report["details"][0]["score"] == 0
    assert report["details"][1]["score"] == 0
    assert report["details"][1]["speedup"] == 2
    assert report["details"][1]["amortized_speedup"] < 1
    assert report["details"][2]["score"] == 1


def test_hidden_failure_cannot_be_overwritten_by_legacy_merge(run):
    from scripts.benchmark_eval_analysis import merge_hidden_eval

    config, _, standard, secret, save = run
    standard["1_0"]["correctness"] = False
    save()
    paper_analysis.analyze_paper(config)
    merged = merge_hidden_eval(common.apply_score_gate(standard, config.run_dir), secret)
    assert not merged["1_0"]["correctness"]


def test_optional_layers_keep_missing_slots_and_bootstrap(run):
    config, _, _, _, _ = run
    specs = config.run_dir / "specs.json"
    atomic_json(specs, {"level1": {"1": {"layer": "F"}}})
    config.problem_specs = str(specs)
    report = paper_analysis.analyze_paper(config)
    assert report["bounded_gain_by_layer"] == {"F": 1}
    assert report["layer_ci95"]["F"] == [1, 1]
    assert not report["cross_device_comparison_certified"]


def test_batch_analysis_writes_gate_before_legacy_metrics(run, monkeypatch):
    from scripts import benchmark_eval_analysis as legacy

    config, _, _, _, _ = run
    monkeypatch.setattr(legacy, "__file__", str(common.ROOT / "scripts/benchmark_eval_analysis.py"))
    report = engine.analyze_run(config)
    assert report["bounded_gain_macro"] == 1
    assert read_json(config.run_dir / "eval_results_scored.json")["1_0"]["correctness"]
    assert (config.run_dir / "analysis_summary.json").is_file()


def test_worker_rejects_mismatched_evaluator_version(run, monkeypatch):
    from kb_workflows import evaluate_task

    config, _, _, _, _ = run
    monkeypatch.setattr(evaluate_task, "evaluator_digest", lambda: "current")
    payload = engine.task_payload(config, 1)
    payload["eval_config"]["evaluator_sha256"] = "old"
    with pytest.raises(ValueError, match="versions differ"):
        evaluate_task.validate_sources(payload)


def test_resume_reuses_current_results_and_reruns_changed_evaluator(run, monkeypatch):
    config, _, _, _, _ = run
    config.resume = True
    calls = []

    def evaluate(config, tasks):
        calls.extend(tasks)
        for key, payload in tasks:
            yield key, {"compiled": True, "correctness": True,
                        "metadata": {"workflow_signature": fingerprint(payload), "evaluation_config": payload["eval_config"]}}

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    engine.evaluate_run(config)
    assert not calls
    monkeypatch.setattr(engine, "evaluator_digest", lambda: "changed-evaluator")
    engine.evaluate_run(config)
    assert len(calls) == 6  # Three samples in both standard and hidden phases.
    calls.clear()
    engine.evaluate_run(config)
    assert not calls


def test_full_pipeline_rerecords_stale_existing_baseline(run, monkeypatch):
    from kb_workflows import cli, generation

    config, baseline, _, _, save = run
    config.stage, config.generation, config.evaluation = "all", "local", "queue"
    baseline["metadata"]["evaluation_config"]["evaluator_sha256"] = "old"
    save()
    calls = []
    monkeypatch.setattr(cli, "ROOT", common.ROOT)
    monkeypatch.setattr(cli, "parse_config", lambda argv: config)
    monkeypatch.setattr(cli, "select_problems", engine.select_problems)
    monkeypatch.setattr(cli, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(engine, "record_baseline", lambda config: calls.append("baseline"))
    monkeypatch.setattr(engine, "evaluate_run", lambda config: None)
    monkeypatch.setattr(engine, "analyze_run", lambda config: None)
    monkeypatch.setattr(generation, "generate_run", lambda config: True)
    assert cli.main([]) == 0
    assert calls == ["baseline"]


def test_child_verifier_accepts_serialized_options_and_omits_parent_layers(run):
    from kb_workflows import generation

    config, _, _, _, _ = run
    specs = config.run_dir / "layers.json"
    atomic_json(specs, {"level1": {"1": {"layer": "F"}}})
    config.problem_specs = str(specs)
    directory = generation.prepare_job(config, 1, 0)
    child = read_json(directory / "job_config.json")
    assert not child["problem_specs"]
    args = []
    for key, value in child.items():
        if value is not None:
            args += ["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)]
    assert parse_config(args).score_gate_required


def test_changed_measurement_code_invalidates_resume_digest(tmp_path, monkeypatch):
    source_root = common.ROOT
    for name in ("src", "kb_workflows"):
        shutil.copytree(source_root / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(source_root / "requirements.txt", tmp_path / "requirements.txt")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    before = common.evaluator_digest()
    path = tmp_path / "kb_workflows/measurement.py"
    path.write_text(path.read_text() + "\n# changed measurement protocol\n")
    assert common.evaluator_digest() != before
