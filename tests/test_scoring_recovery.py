"""Scoring and recovery contracts using persisted results and real workflow IDs."""

import copy
import shutil

import pytest
from _operator_evidence import attach_operator_evidence

from kb_workflows import cli, common, engine, evaluate_task, final_verification, generation, paper_analysis, study_data
from kb_workflows.common import atomic_json, fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.measurement import timing_stats


@pytest.fixture
def run(tmp_path, monkeypatch):
    reference = tmp_path / "KernelBench/level1/1_problem.py"
    reference.parent.mkdir(parents=True)
    reference.write_text(common.problem_paths(1)[1].read_text())
    hidden = tmp_path / "hidden_tests/level1/1_hidden.py"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("hidden source")
    config = RunConfig(run_name="scoring", runs_dir=str(tmp_path / "runs"), hardware="test", gpu_arch="Ampere",
                       start=1, end=1, num_samples=3, baseline_suite=False).validate()
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(paper_analysis, "ROOT", tmp_path)
    # Default parameters on select_problems were bound to the original checkout.
    def select(*args, **kwargs):
        return {1: reference}
    monkeypatch.setattr(engine, "select_problems", select)
    monkeypatch.setattr(engine, "select_run_problems", lambda config: select())
    monkeypatch.setattr(paper_analysis, "select_run_problems", lambda config: select())
    monkeypatch.setattr(engine, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(common, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(paper_analysis, "evaluator_digest", lambda: "current")
    config.run_dir.mkdir(parents=True)
    for sid in range(config.num_samples):
        (config.run_dir / kernel_name(1, 1, sid)).write_text(f"candidate source {sid}")
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    device = {"available": True, "name": "GPU", "capability": [9, 0], "total_memory": 10**10,
              "multiprocessors": 100, "l2_bytes": 10**6, "torch_version": "test", "cuda_version": "test", "hip_version": None,
              "uuid": "physical-test-device", "driver_version": "test"}
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
            attach_operator_evidence(payload, results[f"1_{sid}"])
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


@pytest.mark.parametrize("review", [False, True])
def test_throughput_review_excludes_performance_credit_and_preserves_saved_native_verdict(run, monkeypatch, review):
    """Analysis of already-qualified synthetic receipts; no device claims."""
    config, baseline, standard, secret, save = run
    config.throughput_validation = True
    expected = engine.task_payload(config, 1, task="baseline")
    baseline["metadata"].update(evaluation_config=expected["eval_config"], workflow_signature=fingerprint(expected))
    for results, hidden in ((standard, False), (secret, True)):
        for sid in range(config.num_samples):
            expected = engine.task_payload(config, 1, sid, hidden=hidden)
            results[f"1_{sid}"]["metadata"].update(evaluation_config=expected["eval_config"], workflow_signature=fingerprint(expected))
    for result in standard.values():
        result["metadata"]["tool_checks"] = {"checks": {"throughput_contrast": {
            "status": "passed", "values": {"review_required": review, "threshold": .1}}}}
    # Raw receipt validation is covered by the supervisor/contrast regressions;
    # this test isolates score policy after that validation has succeeded.
    monkeypatch.setattr(paper_analysis, "current_result", lambda *args: None)
    save()
    report = paper_analysis.analyze_paper(config)
    assert report["bounded_gain_macro"] == (0 if review else 1)
    assert {row["status"] for row in report["details"]} == ({"throughput_discrepancy_requires_review"} if review else {"verified"})
    assert all(row["correctness"] for row in read_json(config.run_dir / "eval_results.json").values())


def test_rl_public_hidden_gap_regrades_identical_saved_sources_without_execution(run, monkeypatch):
    from dataclasses import replace

    from kb_workflows import rl_audit, study_data

    config, _, _, hidden, save = run
    hidden["1_0"]["correctness"] = False
    save()
    monkeypatch.setattr(study_data, "ROOT", common.ROOT)
    monkeypatch.setattr(study_data, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(study_data, "select_run_problems", engine.select_run_problems)
    monkeypatch.setattr(engine, "evaluate_tasks", lambda *a, **k: pytest.fail("Audit launched an evaluator"))
    public = replace(config, hidden_tests=False, run_name="public")
    shutil.copytree(config.run_dir, public.run_dir)
    atomic_json(public.run_dir / "run_config.json", public.to_dict())
    result = rl_audit.public_hidden_gap(public.run_dir, config.run_dir, trials=100)
    assert result["mean_public_minus_final"] == pytest.approx(1 / 3)
    assert result["failures_retained"] and not result["experiments_started"]
    (public.run_dir / kernel_name(1, 1)).write_text("different candidate")
    with pytest.raises(ValueError, match="identical candidate"):
        rl_audit.public_hidden_gap(public.run_dir, config.run_dir, trials=100)


@pytest.mark.parametrize("damage", [None, "receipt", "history", "usage", "signature"])
def test_resource_budget_proof_is_required_and_invalidates_both_score_gates(run, monkeypatch, damage):
    import hashlib

    from kb_workflows.experiment import protocol, write_budget_report
    from kb_workflows.resource_budgets import Ledger

    config, _, standard, _, _ = run
    config.agent_backend, config.model, config.agent_output_token_limit = "litellm", "fake/provider", 10
    config.validate()
    monkeypatch.setattr(generation, "generation_signature", lambda *args: "cap-signature")
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    summary, history = {}, {"slots": {}}
    for sid in range(config.num_samples):
        ledger = Ledger(10, "tokens", integer=True)
        ledger.settle(ledger.reserve(10), 3)
        receipt = {"protocol": {**protocol(config), "samples_per_problem": 1}, "model_calls": 1, "output_tokens": 3,
                   "resource_budgets": {"output_tokens": ledger.snapshot()}}
        key = f"1_{sid}"
        summary[key] = {"status": "success", "signature": "cap-signature", "budget": receipt,
                        "kernel_sha256": hashlib.sha256((config.run_dir / kernel_name(1, 1, sid)).read_bytes()).hexdigest()}
        history["slots"][key] = [{"status": "success", "signature": "cap-signature", "budget": receipt}]
    atomic_json(config.run_dir / "generation_summary.json", summary)
    atomic_json(config.run_dir / "generation_attempts.json", history)
    write_budget_report(config, summary)
    assert paper_analysis.analyze_paper(config)["bounded_gain_macro"] == pytest.approx(1)
    assert common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]
    if damage is None:
        return
    if damage == "receipt":
        (config.run_dir / "generation_budget.json").unlink()
    elif damage == "history":
        history["slots"]["1_0"].append(history["slots"]["1_0"][0])
        atomic_json(config.run_dir / "generation_attempts.json", history)
    elif damage == "usage":
        summary["1_0"]["budget"]["output_tokens"] = 30
        atomic_json(config.run_dir / "generation_summary.json", summary)
    else:
        summary["1_0"]["signature"] = "different-config-or-evaluator"
        atomic_json(config.run_dir / "generation_summary.json", summary)
    assert not common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]
    report = paper_analysis.analyze_paper(config)
    assert report["details"][0]["score"] == 0 and report["details"][0]["status"].startswith("resource_budget")


def test_study_export_recomputes_current_qualification_without_running_workloads(run, monkeypatch):
    import hashlib

    from kb_workflows.experiment import write_budget_report

    config, _, _, _, _ = run
    config.model = "declared-test-model"
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    monkeypatch.setattr(study_data, "ROOT", common.ROOT)
    monkeypatch.setattr(study_data, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(study_data, "select_run_problems", engine.select_run_problems)
    monkeypatch.setattr(engine, "evaluate_tasks", lambda *a, **k: pytest.fail("Analysis started evaluation"))
    monkeypatch.setattr(generation, "generate_run", lambda *a, **k: pytest.fail("Analysis started generation"))
    receipts = {}
    for sid in range(config.num_samples):
        source = config.run_dir / kernel_name(1, 1, sid)
        receipts[f"1_{sid}"] = {"kernel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "budget": {"model_calls": 1, "candidate_submissions": 1,
                                           "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                                           "estimated_cost_usd": 0.01, "generation_wall_seconds": 3,
                                           "development_request_wall_seconds": 1, "models": ["actual-test-model"]}}
    write_budget_report(config, receipts)
    exported = study_data.collect_run(config.run_dir, session="replica-0")
    assert exported["tasks"] == {"1/1": [1, 1, 1]}
    assert exported["resource_totals"]["total_tokens"] == 90
    assert exported["formal_comparison_eligible"] and exported["model_identity"] == "actual-test-model"
    (config.run_dir / kernel_name(1, 1)).write_text("changed source after evaluation")
    updated = study_data.collect_run(config.run_dir, session="replica-0")
    assert updated["tasks"] == {"1/1": [0, 1, 1]} and not updated["formal_comparison_eligible"]


@pytest.mark.parametrize("damage", ["none", "missing", "scope", "summary"])
def test_persisted_strict_evidence_is_required_at_scoring_boundary(run, damage):
    from kb_workflows import evidence

    config, baseline, standard, secret, save = run
    config.strict_validation, config.evidence_level = True, "L0"
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    payload = engine.task_payload(config, 1, task="baseline")
    baseline["metadata"].update(evaluation_config=payload["eval_config"], workflow_signature=fingerprint(payload))
    for hidden, results in ((False, standard), (True, secret)):
        for sid in range(config.num_samples):
            payload = engine.task_payload(config, 1, sid, hidden=hidden)
            checks = {name: evidence.check("passed", reason="Persisted evidence fixture") for name in evidence.LEVELS["L0"]}
            results[f"1_{sid}"]["metadata"].update(
                evaluation_config=payload["eval_config"], workflow_signature=fingerprint(payload),
                evidence_receipt=evidence.receipt(checks, requested="L0", scope="hidden" if hidden else "standard"))
    record = standard["1_0"]["metadata"]["evidence_receipt"]
    if damage == "missing":
        del standard["1_0"]["metadata"]["evidence_receipt"]
    elif damage == "scope":
        record["scope"] = "hidden"
    elif damage == "summary":
        record["checks"]["precision_contract"] = evidence.check("not_measured", reason="Missing oracle proof")
        record["requested_level_met"] = True
    save()
    report = paper_analysis.analyze_paper(config)
    assert report["details"][0]["status"] == "verified" if damage == "none" else report["details"][0]["status"] != "verified"
    assert report["bounded_gain_macro"] == pytest.approx(1 if damage == "none" else 2 / 3)
    assert report["aggregation_sensitivity"]["declared_slots"] == config.num_samples


@pytest.mark.parametrize("damage", ["none", "final_missing", "final_hidden", "frozen", "paired_missing", "paired_summary"])
def test_final_and_paired_evidence_drive_scores_and_legacy_gate(run, monkeypatch, damage):
    from kb_workflows.paired_statistics import paired_summary

    config, baseline, standard, secret, save = run
    config.baseline_suite, config.paired_timing, config.paired_trials, config.final_repeats = True, True, 12, 2
    monkeypatch.setattr(final_verification, "select_problems", engine.select_problems)
    monkeypatch.setattr(final_verification, "evaluator_digest", lambda: "current")
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    generic = engine.task_payload(config, 1, task="baseline")
    baseline["metadata"].update(evaluation_config=generic["eval_config"], workflow_signature=fingerprint(generic))
    variant = copy.deepcopy(baseline)
    variant_payload = copy.deepcopy(generic)
    variant_payload["eval_config"]["baseline_variant"] = "eager"
    variant["metadata"].update(evaluation_config=variant_payload["eval_config"], workflow_signature=fingerprint(variant_payload))
    baseline.update(baseline_suite=True, suite_complete=True, variants={"eager": variant})
    for hidden, results in ((False, standard), (True, secret)):
        for sid in range(config.num_samples):
            payload = engine.task_payload(config, 1, sid, hidden=hidden)
            results[f"1_{sid}"]["metadata"].update(evaluation_config=payload["eval_config"], workflow_signature=fingerprint(payload))
    manifest = final_verification.freeze(config)
    final, paired = {}, {}
    for sid in range(config.num_samples):
        for repeat in range(config.final_repeats):
            for hidden, originals in ((False, standard), (True, secret)):
                payload = final_verification.payload_for(config, 1, sid, repeat, hidden=hidden)
                record = copy.deepcopy(originals[f"1_{sid}"])
                record["metadata"].update(evaluation_config=payload["eval_config"], workflow_signature=fingerprint(payload))
                attach_operator_evidence(payload, record)
                final[f"1_{sid}:{repeat}:{'hidden' if hidden else 'standard'}"] = record
            payload = engine.paired_payload(config, 1, sid, repeat=repeat, variant="eager")
            summary = paired_summary([{"reference_ms": 10, "candidate_ms": 5, "control_ms": 10} for _ in range(12)])
            summary.update(baseline_variant="eager", output_consumed=True, varied_inputs=True, all_outputs_verified=True)
            metadata = {**copy.deepcopy(standard[f"1_{sid}"]["metadata"]), "paired_timing": summary,
                        "evaluation_config": payload["eval_config"], "workflow_signature": fingerprint(payload)}
            paired[f"1_{sid}:{repeat}"] = {"compiled": True, "correctness": True, "runtime": 5, "metadata": metadata}
    if damage == "final_missing":
        del final["1_0:1:standard"]
    elif damage == "final_hidden":
        final["1_0:1:hidden"]["correctness"] = False
    elif damage == "frozen":
        path = config.run_dir / "final_artifacts" / (manifest["slots"]["1_0"]["source_sha256"] + ".py")
        path.write_text("changed frozen source")
    elif damage == "paired_missing":
        del paired["1_0:1"]
    elif damage == "paired_summary":
        paired["1_0:1"]["metadata"]["paired_timing"]["speedup_ci95"] = [100, 100]
    atomic_json(config.run_dir / "final_results.json", final)
    atomic_json(config.run_dir / "paired_results.json", paired)
    save()
    report = paper_analysis.analyze_paper(config)
    assert report["bounded_gain_macro"] == pytest.approx(1 if damage == "none" else 2 / 3)
    assert report["details"][1]["paired_repeat_groups"] == 2
    gated = common.apply_score_gate(standard, config.run_dir)
    assert gated["1_0"]["correctness"] is (damage == "none")
    assert gated["1_1"]["correctness"]
    if damage == "none":
        from kb_workflows import calibration, headroom

        monkeypatch.setattr(headroom, "ROOT", common.ROOT)
        monkeypatch.setattr(headroom, "select_problems", engine.select_problems)
        monkeypatch.setattr(headroom, "evaluator_digest", lambda: "current")
        measured = calibration.seal(baseline["device_fingerprint"],
            {"memory_bytes": calibration.probe_record(1000, [1] * 5, unit="bytes/s", parameters={})},
            [0.01] * 5, settings={"test_fixture": True}, digest="current")
        witness = headroom.witness_from_run(config, 1, 0, measured)
        assert witness["runtime_ms"] == 5 and len(witness["verification"]) == 2
        assert witness["source_sha256"] == manifest["slots"]["1_0"]["source_sha256"]
    final["1_1:0:standard"]["runtime"] = 999
    atomic_json(config.run_dir / "final_results.json", final)
    assert not common.apply_score_gate(standard, config.run_dir)["1_1"]["correctness"]


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


@pytest.mark.parametrize("damage", ["none", "missing_certificate", "different_device", "design_without_witness", "rate"])
def test_headroom_scoring_uses_reviewed_certificate_and_invalidates_external_inputs(run, damage):
    import hashlib
    import math

    from kb_workflows import calibration, headroom

    config, baseline, standard, secret, save = run
    baseline["device_fingerprint"]["uuid"] = "physical-test-device"
    baseline["device_fingerprint"]["driver_version"] = "test"
    for records in (standard, secret):
        for record in records.values():
            record["metadata"]["device_fingerprint"]["uuid"] = "physical-test-device"
            record["metadata"]["device_fingerprint"]["driver_version"] = "test"
    measured = calibration.seal(baseline["device_fingerprint"],
        {"memory_bytes": calibration.probe_record(1000, [1] * 5, unit="bytes/s", parameters={})},
        [0.01] * 5, settings={"test_fixture": True}, digest="current")
    spec = {"protocol": headroom.PROTOCOL, "layer": "F", "reference_sha256": hashlib.sha256(engine.select_problems(1)[1].read_bytes()).hexdigest(),
            "precision": "fp32", "input_signature": baseline["input_signature"],
            "review": {"reviewer": "unit-test", "rationale": "Synthetic protocol fixture"},
            "calibration_sha256": measured["receipt_sha256"],
            "reference_work": {"memory_bytes": 1000}, "algorithm_work": {"memory_bytes": 1000}}
    if damage == "missing_certificate":
        spec = {}
    elif damage == "different_device":
        measured["device"]["uuid"] = "other-device"
    elif damage == "design_without_witness":
        spec["layer"] = "D"
    elif damage == "rate":
        measured["probes"]["memory_bytes"]["rate"] = 1e20
    calibration_path, specs_path = config.run_dir / "calibration.json", config.run_dir / "specs.json"
    atomic_json(calibration_path, measured)
    atomic_json(specs_path, {"level1": {"1": {"layer": "F", "headroom_certificate": spec}}})
    config.score_mode, config.device_calibration, config.problem_specs = "headroom", str(calibration_path), str(specs_path)
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    save()
    report = paper_analysis.analyze_paper(config)
    if damage == "none":
        assert report["headroom_macro"] == pytest.approx(math.log(2) / math.log(10))
        assert report["leaderboard_eligible"]
        assert report["headroom_by_layer"] == {"F": pytest.approx(math.log(2) / math.log(10))}
        assert report["bounded_gain_macro"] == 1
        assert common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]
        measured["launch_median_ms"] = 500
        atomic_json(calibration_path, measured)
        assert not common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]
    else:
        assert report["headroom_macro"] == 0
        assert not report["leaderboard_eligible"]
        assert report["details"][0]["layer"] == "unclassified"
        assert report["details"][0]["status"] == "headroom_certificate_missing_or_invalid"
        assert not common.apply_score_gate(standard, config.run_dir)["1_0"]["correctness"]


def test_batch_analysis_writes_gate_before_legacy_metrics(run, monkeypatch):
    from scripts import benchmark_eval_analysis as legacy

    config, _, _, _, _ = run
    monkeypatch.setattr(legacy, "__file__", str(common.ROOT / "scripts/benchmark_eval_analysis.py"))
    report = engine.analyze_run(config)
    assert report["bounded_gain_macro"] == 1
    assert read_json(config.run_dir / "eval_results_scored.json")["1_0"]["correctness"]
    assert (config.run_dir / "analysis_summary.json").is_file()


def test_worker_rejects_mismatched_evaluator_version(run, monkeypatch):
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
            result = {"compiled": True, "correctness": True,
                        "metadata": {"workflow_signature": fingerprint(payload), "evaluation_config": payload["eval_config"]}}
            yield key, attach_operator_evidence(payload, result)

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

    config, baseline, _, _, save = run
    config.stage, config.generation, config.evaluation = "all", "local", "queue"
    baseline["metadata"]["evaluation_config"]["evaluator_sha256"] = "old"
    save()
    calls = []
    monkeypatch.setattr(cli, "ROOT", common.ROOT)
    monkeypatch.setattr(cli, "parse_config", lambda argv: config)
    monkeypatch.setattr(cli, "select_run_problems", engine.select_run_problems)
    monkeypatch.setattr(cli, "evaluator_digest", lambda: "current")
    monkeypatch.setattr(engine, "record_baseline", lambda config: calls.append("baseline"))
    monkeypatch.setattr(engine, "evaluate_run", lambda config: None)
    monkeypatch.setattr(engine, "analyze_run", lambda config: None)
    monkeypatch.setattr(generation, "generate_run", lambda config: True)
    assert cli.main([]) == 0
    assert calls == ["baseline"]


def test_child_verifier_accepts_serialized_options_and_omits_parent_layers(run):

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
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(source_root / "scripts/regrade_artifacts.py", tmp_path / "scripts/regrade_artifacts.py")
    shutil.copyfile(source_root / "scripts/audit_kernel_origin.py", tmp_path / "scripts/audit_kernel_origin.py")
    shutil.copyfile(source_root / "scripts/audit_work_counters.py", tmp_path / "scripts/audit_work_counters.py")
    shutil.copyfile(source_root / "scripts/calibrate_inputs.py", tmp_path / "scripts/calibrate_inputs.py")
    shutil.copyfile(source_root / "scripts/calibrate_precision.py", tmp_path / "scripts/calibrate_precision.py")
    shutil.copyfile(source_root / "scripts/freeze_reference_population.py", tmp_path / "scripts/freeze_reference_population.py")
    for name in ("requirements.txt", "requirements-neuron-worker.txt"):
        shutil.copyfile(source_root / name, tmp_path / name)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    before = common.evaluator_digest()
    path = tmp_path / "kb_workflows/measurement.py"
    path.write_text(path.read_text() + "\n# changed measurement protocol\n")
    assert common.evaluator_digest() != before
