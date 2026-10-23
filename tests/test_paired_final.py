"""Paired timing, independent frozen-source repeats and expanded reference baselines."""

import copy
from collections import Counter
from dataclasses import replace

import pytest
import torch
from _operator_evidence import attach_operator_evidence

from kb_workflows import baseline_variants, engine, final_verification, measurement
from kb_workflows.baseline_rewrites import framework_rewrite, rewrite_for
from kb_workflows.common import atomic_json, fingerprint, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.paired_statistics import paired_summary
from kb_workflows.paired_task import collect_pairs
from src.eval import load_original_model_and_inputs
from src.validation import compare_outputs


def test_paired_intervals_distinguish_gain_from_reference_noise():
    same = [{"reference_ms": 2, "candidate_ms": 2, "control_ms": 2} for _ in range(12)]
    record = paired_summary(same)
    assert record["speedup_ci95"] == [1, 1] and not record["significantly_faster"]
    faster = [{**row, "candidate_ms": 1} for row in same]
    assert paired_summary(faster)["significantly_faster"]
    drifting = [{**row, "control_ms": 0.5} for row in faster]
    assert not paired_summary(drifting)["significantly_faster"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_paired_stats_reject_invalid_samples(value):
    rows = [{"reference_ms": 1, "candidate_ms": value, "control_ms": 1} for _ in range(12)]
    with pytest.raises(ValueError):
        paired_summary(rows)


def test_bootstrap_keeps_independent_repeats_as_groups():
    rows = [{"reference_ms": 2, "candidate_ms": 1 if group else 4, "control_ms": 2, "repeat": group}
            for group in range(2) for _ in range(12)]
    report = paired_summary(rows)
    assert report["independent_repeat_groups"] == 2
    assert report["speedup_ci95"][0] <= 0.5 and report["speedup_ci95"][1] >= 2
    assert not report["significantly_faster"]


@pytest.mark.parametrize("warmup", [0, 3])
def test_balanced_real_cpu_calls_exclude_preparation_and_verify_outputs(monkeypatch, warmup):
    clock, seen = [0.0], []
    monkeypatch.setattr(measurement, "CLOCK", lambda: clock[0])

    def factory(index):
        clock[0] += 10
        return [torch.arange(4).float() + index]

    def oracle(x):
        clock[0] += 100
        return x.square()

    def timed(name, duration):
        def call(x):
            clock[0] += duration
            seen.append(name)
            return x.square()
        return call

    result = collect_pairs(timed("R", 0.004), timed("C", 0.002), timed("B", 0.004), factory, oracle,
                           device="cpu", trials=12, seed=42, warmup=warmup, clock=lambda: clock[0])
    assert len(seen) == 3 * (12 + warmup)
    assert Counter(row["order"] for row in result["pairs"]) == {key: 2 for key in ("RCB", "CRB", "BCR", "BRC", "CBR", "RBC")}
    assert result["reference_mean_ms"] == pytest.approx(4)
    assert result["candidate_mean_ms"] == pytest.approx(2)
    assert result["significantly_faster"]


def test_paired_wrong_outputs_are_not_timed_as_valid_speedups():
    with pytest.raises(ValueError, match="Paired trial"):
        collect_pairs(lambda x: x + 1, lambda x: torch.zeros_like(x), lambda x: x + 1,
                      lambda trial: [torch.ones(4)], lambda x: x + 1, device="cpu", trials=10, seed=42)


@pytest.mark.parametrize("pid", [36, 38, 39])
@pytest.mark.parametrize("kind", ["signed", "zero", "near_constant"])
def test_reviewed_framework_rewrites_preserve_actual_reference_semantics(pid, kind):
    source = select_problems(1, problem_ids=[pid])[pid].read_text()
    Model, init, _ = load_original_model_and_inputs(source, {})
    model = Model(*init())
    value = torch.randn(2, 64, 3, 5)
    if kind == "zero":
        value.zero_()
    elif kind == "near_constant":
        value.fill_(128)
        value[..., ::2] += 1
    rewritten = framework_rewrite(model, rewrite_for(1, pid, source))
    assert compare_outputs(model(value), rewritten(value), 1e-5) is None
    with pytest.raises(ValueError, match="reference changed"):
        rewrite_for(1, pid, source + "\n# changed")


def test_expanded_portfolio_requests_all_graph_combinations_and_reviewed_rewrites(tmp_path, monkeypatch):
    cfg = RunConfig(run_name="extended", runs_dir=str(tmp_path), hardware="RTX_A6000", start=36, end=36,
                    baseline_portfolio="extended").validate()
    assert "extended" in cfg.baseline
    seen = []
    monkeypatch.setattr(engine, "ROOT", tmp_path)

    def evaluate(config, tasks):
        for key, payload in tasks:
            seen.append(payload)
            from kb_workflows.baseline_rewrites import plan_for

            yield key, {"mean": 1, "cold_start_ms": 2, "baseline_correctness": True,
                        "baseline_plan": plan_for(payload["level"], payload["problem_id"], payload["ref_arch_src"], payload["eval_config"]["baseline_variant"]),
                        "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
                        "metadata": {"evaluation_config": payload["eval_config"]}}

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    target = engine.record_baseline(cfg)
    assert len(seen) == 8 and len({row["request_id"] for row in seen}) == 8
    entry = next(iter(read_json(target)["level1"].values()))
    assert entry["suite_complete"] and len(entry["requested_variants"]) == 8
    assert "compile_max_autotune_cuda_graph" in entry["variants"]


def test_outer_graph_owns_replay_instead_of_nesting_inductor_graphs(monkeypatch):
    options = []
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: options.append(kwargs) or model)
    monkeypatch.setattr(baseline_variants, "cuda_graph_model", lambda model, args, device: model)
    model = torch.nn.Identity()
    _, backend, mode = baseline_variants.prepare_baseline(model, [torch.ones(2)], "cpu", "compile_max_autotune_cuda_graph")
    assert backend == "inductor" and mode == "max-autotune-no-cudagraphs"
    assert options == [{"backend": "inductor", "mode": "max-autotune-no-cudagraphs"}]


def final_config(tmp_path, samples=2):
    cfg = RunConfig(run_name="frozen", runs_dir=str(tmp_path), hardware="RTX_A6000", start=19, end=19,
                    final_repeats=2, num_samples=samples).validate()
    cfg.run_dir.mkdir(parents=True)
    for sid in range(samples):
        (cfg.run_dir / kernel_name(1, 19, sid)).write_text(f"# candidate {sid}\n")
    return cfg


def test_freezing_uses_unique_fresh_seeds_and_survives_analysis_phase(tmp_path):
    cfg = final_config(tmp_path)
    manifest = final_verification.freeze(cfg)
    plans = [plan for entry in manifest["slots"].values() for plan in entry["repeats"]]
    assert len({plan["nonce"] for plan in plans}) == 4
    assert all(len(set(plan["hidden_seeds"])) == 3 for plan in plans)
    analysis = replace(cfg, stage="analyze", resume=True)
    payload = final_verification.payload_for(analysis, 19, 0, 1, hidden=True)
    assert payload["eval_config"]["evaluation_seed"] == plans[1]["seed"]
    assert payload["eval_config"]["hidden_seeds"] == plans[1]["hidden_seeds"]
    assert payload["kernel_src"] == "# candidate 0\n"


def test_frozen_source_edits_invalidate_payload_and_resume(tmp_path):
    cfg = final_config(tmp_path, 1)
    manifest = final_verification.freeze(cfg)
    frozen = cfg.run_dir / "final_artifacts" / (manifest["slots"]["19_0"]["source_sha256"] + ".py")
    frozen.write_text("# modified")
    with pytest.raises(ValueError, match="modified"):
        final_verification.payload_for(cfg, 19, 0, 0)
    cfg.resume = True
    with pytest.raises(ValueError, match="modified"):
        final_verification.freeze(cfg)


def test_final_worker_tasks_are_distinct_resume_and_only_changed_sample_retries(tmp_path, monkeypatch):
    cfg = final_config(tmp_path)
    seen = []

    def evaluate(config, tasks):
        seen.extend(tasks)
        for key, payload in tasks:
            result = {"compiled": True, "correctness": True, "runtime": 1,
                        "metadata": {"workflow_signature": fingerprint(payload), "evaluation_config": payload["eval_config"],
                                     "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
                                     "kernel_sha256": fingerprint({"source": payload["kernel_src"]})}}
            yield key, attach_operator_evidence(payload, result)

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    results = final_verification.run(cfg)
    assert len(seen) == len(results) == 8 and len({payload["request_id"] for _, payload in seen}) == 8
    assert final_verification.correctness_passed(cfg, 19, 0)
    cfg.resume = True
    seen.clear()
    final_verification.run(cfg)
    assert seen == []
    (cfg.run_dir / kernel_name(1, 19, 0)).write_text("# new candidate\n")
    seen.clear()
    final_verification.run(cfg)
    assert len(seen) == 4 and all(payload["sample_id"] == 0 for _, payload in seen)
    record = read_json(cfg.run_dir / "final_results.json")
    record["19_0:1:hidden"]["correctness"] = False
    atomic_json(cfg.run_dir / "final_results.json", record)
    assert not final_verification.correctness_passed(cfg, 19, 0)
    assert final_verification.correctness_passed(cfg, 19, 1)


def test_new_settings_parse_validate_and_have_distinct_protocols(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[run]\nbaseline_portfolio="extended"\npaired_timing=true\npaired_trials=12\nfinal_repeats=2\n')
    cfg = parse_config(["--config", str(path)])
    assert cfg.paired_timing and cfg.final_repeats == 2 and cfg.paired_trials == 12
    for options in ({"final_repeats": -1}, {"paired_trials": 1}, {"paired_timing": True, "baseline_suite": False}):
        with pytest.raises(ValueError):
            RunConfig(**options).validate()


def test_final_summary_keeps_slow_repeats_and_process_uncertainty():
    rows = [{"runtime_stats": {"samples_ms": [1, 1]}}, {"runtime_stats": {"samples_ms": [9, 9]}}]
    report = final_verification.timing_summary(rows)
    assert report["mean"] == 5 and report["num_trials"] == 4 and report["independent_repeats"] == 2
    assert report["mean_ci95_ms"] == [1, 9]


def test_fresh_hidden_seed_schedule_is_used_by_actual_correctness_entrypoint(monkeypatch):
    from src import eval as native

    monkeypatch.setattr(native, "load_hidden_inputs", lambda path: [[torch.randn(2, 3)]])
    metadata = {}
    model = torch.nn.Identity()
    assert native.run_hidden_correctness_check(model, copy.deepcopy(model), "/not-a-stress-contract.py", metadata,
                                               device=torch.device("cpu"), seeds=[718, 941, 2112])
    assert metadata["hidden_validation_seeds"] == [718, 941, 2112]
    assert metadata["hidden_completed_checks"] == 3
    with pytest.raises(ValueError):
        native.run_hidden_correctness_check(model, model, "/unused", {}, device=torch.device("cpu"), seeds=[42, 42])
