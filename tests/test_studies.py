"""Protocol, statistical and accounting regressions; no real experiments."""

import json
import math

import pytest

from kb_workflows import budget_curves, study_data, study_plan
from kb_workflows import study_statistics as stats
from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.config import parse_config
from kb_workflows.harbor_budget import harbor_budget


def record(model="a", session="s0", budget=60, values=None, level=1, *, eligible=True, resource=100):
    values = values or [0.0, 0.5, 1.0]
    tasks = {f"{level}/{i + 1}": [value, value] for i, value in enumerate(values)}
    result = {"protocol": study_data.PROTOCOL, "model_identity": model, "label": model, "session": session,
              "evaluation_contract": {"hardware": "test", "num_samples": 2},
              "generation_protocol": {"mode": "feedback", "scaffold": "mini-swe-agent==2.4.6", "model_backend": "test",
                                      "wall_limit_seconds_per_sample": budget, "candidate_limit_per_sample": 5},
              "metric": "score", "speedup_threshold": 1.0, "tasks": tasks,
              "layers": {key: "I" for key in tasks}, "reference_sha256": {key: "a" * 64 for key in tasks},
              "resource_totals": {"total_tokens": resource, "generation_wall_seconds": resource, "model_calls": 6},
              "declared_slots": len(tasks) * 2, "formal_comparison_eligible": eligible}
    return seal(result)


def seal(value):
    value["receipt_sha256"] = fingerprint({key: item for key, item in value.items() if key != "receipt_sha256"})
    return value


@pytest.mark.parametrize("vary,axis,field", [
    ("total_tokens", "declared_total_tokens", "total_token_limit_per_sample"),
    ("output_tokens", "declared_output_tokens", "output_token_limit_per_sample"),
    ("device_lease_seconds", "declared_device_lease_seconds", "development_device_lease_seconds_per_sample"),
])
def test_resource_limit_curves_group_the_varied_cap_into_one_series(vary, axis, field):
    records = [record(budget=60), record(budget=60)]
    for item, cap in zip(records, [10, 20]):
        item["generation_protocol"][field] = cap
        seal(item)
    output = budget_curves.budget_curves(records, vary=vary, axis=axis, trials=100)
    series = output["groups"][0]["series"]
    assert len(series) == 1 and [row["x"] for row in series[0]["points"]] == [10, 20]


def test_ties_and_constant_rankings_are_not_fake_perfect_agreement():
    assert stats.ranks([1, 1, 2]) == [1.5, 1.5, 3]
    assert stats.spearman([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    assert stats.kendall_b([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    assert stats.spearman([0, 0, 0], [1, 2, 3]) is None
    assert stats.kendall_b([0, 0, 0], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        stats.ranks([math.nan, 1])


@pytest.mark.parametrize("aggregation", ["task_mean", "layer_macro"])
def test_selection_reversal_is_detected_with_fixed_failures_and_intervals(aggregation):
    records = [record("a", values=[0.1, 1, 1]), record("b", values=[0.4, 0.5, 0.5]), record("c", values=[0.8, 0, 0])]
    group = study_data.discrimination_groups(records)[0]
    result = stats.selection_analysis(group["subjects"], group["layers"], ["1/1"], aggregation=aggregation, trials=100)
    assert result["agreement"]["spearman"] == pytest.approx(-1)
    assert result["agreement"]["kendall_tau_b"] == pytest.approx(-1)
    assert all(pair["observed_strict_order_reversed"] for pair in result["pairwise"])
    assert result["scores"]["c"]["full"] == pytest.approx(0.8 / 3)
    assert result["scores"]["c"]["selected_ci95"] == [0.8, 0.8]
    assert not result["historical_seven_model_ranking_reproduced"]
    identical = stats.selection_analysis(group["subjects"], group["layers"], list(group["layers"]), trials=100)
    assert identical["agreement"]["spearman"] == 1
    assert identical["agreement"]["spearman_ci95"] == [1, 1]


def test_all_failure_population_has_undefined_correlations_and_zero_scores():
    group = study_data.discrimination_groups([record("a", values=[0, 0]), record("b", values=[0, 0])])[0]
    result = stats.selection_analysis(group["subjects"], group["layers"], ["1/1"], trials=100)
    assert result["agreement"]["spearman"] is None and result["agreement"]["spearman_ci95"] is None
    assert result["agreement"]["spearman_defined_draws"] == 0
    assert result["discrimination"]["selected_subject_std"] == 0
    assert result["redundancy"]["pairs"] == []


def test_power_design_is_an_explicit_approximation_and_zero_discordance_is_not_zero_mde():
    smaller = stats.paired_binary_design(30, 0.3)
    larger = stats.paired_binary_design(146, 0.3)
    assert larger["minimum_gap_normal_approximation"] < smaller["minimum_gap_normal_approximation"]
    assert not larger["empirical_power_verified"] and not larger["operator_family_correlation_adjusted"]
    assert stats.paired_binary_design(146, 0)["minimum_gap_normal_approximation"] is None


@pytest.mark.parametrize("damage", ["missing", "unequal_samples", "nan", "duplicate_selection", "outside_selection"])
def test_invalid_populations_do_not_silently_shrink_the_denominator(damage):
    group = study_data.discrimination_groups([record("a"), record("b")])[0]
    selected = ["1/1"]
    if damage == "missing":
        del group["subjects"]["b"]["sessions"]["s0"]["1/2"]
    elif damage == "unequal_samples":
        group["subjects"]["b"]["sessions"]["s0"]["1/2"].append(0)
    elif damage == "nan":
        group["subjects"]["b"]["sessions"]["s0"]["1/2"][0] = math.nan
    elif damage == "duplicate_selection":
        selected *= 2
    else:
        selected = ["9/9"]
    with pytest.raises(ValueError):
        stats.selection_analysis(group["subjects"], group["layers"], selected, trials=100)


def test_sessions_and_levels_merge_without_selecting_the_best_attempt():
    records = [record(model, session, level=level, values=values)
               for model in ("a", "b") for session, values in (("s0", [0, 0]), ("s1", [1, 1])) for level in (1, 2)]
    group = study_data.discrimination_groups(records)[0]
    assert set(group["subjects"]["a"]["sessions"]["s0"]) == {"1/1", "1/2", "2/1", "2/2"}
    result = stats.selection_analysis(group["subjects"], group["layers"], ["1/1", "2/1"], trials=100)
    assert result["scores"]["a"]["full"] == 0.5 and result["scores"]["a"]["session_count"] == 2
    assert result["scores"]["a"]["full_ci95"][0] == 0 and result["scores"]["a"]["full_ci95"][1] == 1
    with pytest.raises(ValueError, match="Duplicate task"):
        study_data.discrimination_groups([*records, records[0]])


def test_copied_generation_receipts_do_not_create_independent_sessions(tmp_path, monkeypatch):
    first, second = record("a", "s0"), record("a", "s1")
    for value in (first, second):
        value["source_receipts"] = {"generation_budget.json": "same-actual-attempt"}
        seal(value)
    with pytest.raises(ValueError, match="independent sessions"):
        study_data.discrimination_groups([first, second])
    with pytest.raises(ValueError, match="independent sessions"):
        budget_curves.budget_curves([first, second], trials=100)
    manifest = tmp_path / "study.json"
    atomic_json(manifest, {"runs": [{"path": "one-run", "session": "s0"}, {"path": "./one-run", "session": "s1"}]})
    monkeypatch.setattr(study_data, "collect_run", lambda *a, **k: first)
    with pytest.raises(ValueError, match="one run directory"):
        study_data.collect_manifest(manifest)


def test_stale_or_unattributed_records_require_explicit_audit_mode():
    bad = record("a", eligible=False)
    with pytest.raises(ValueError, match="Unattributed"):
        study_data.discrimination_groups([bad])
    assert not study_data.discrimination_groups([bad], audit_only=True)[0]["formal_comparison_eligible"]
    bad["tasks"]["1/1"][0] = 0.9
    with pytest.raises(ValueError, match="changed"):
        study_data.validate_record(bad)


def test_incompatible_protocols_and_sources_are_not_combined():
    first, second = record("a"), record("b", budget=120)
    assert len(study_data.discrimination_groups([first, second])) == 2
    second = record("b")
    second["reference_sha256"]["1/1"] = "b" * 64
    seal(second)
    with pytest.raises(ValueError, match="different source snapshots"):
        study_data.discrimination_groups([first, second])


def test_curves_preserve_regression_and_only_integrate_shared_budget_domains():
    records = [record("a", budget=60, values=[0, 0]), record("a", budget=120, values=[1, 1]),
               record("a", budget=240, values=[0.5, 0.5]), record("b", budget=120, values=[0.2, 0.2]),
               record("b", budget=180, values=[0.4, 0.4])]
    report = budget_curves.budget_curves(records, trials=100)
    group = report["groups"][0]
    assert group["shared_auc_domain"] == (120, 180)
    a, b = group["series"]
    assert [point["score"] for point in a["points"]] == [0, 1, 0.5]
    assert a["auc_normalized"] == pytest.approx(0.875)
    assert b["auc_normalized"] == pytest.approx(0.3)
    assert not report["monotonic_envelope_or_best_of_history_used"]


@pytest.mark.parametrize("axis", ["observed_total_tokens", "observed_generation_wall_seconds"])
def test_unknown_resource_is_not_free_and_observed_axes_are_normalized_per_slot(axis):
    first, second = record("a", budget=60, values=[0, 1], resource=80), record("a", budget=120, values=[0, 1], resource=None)
    report = budget_curves.budget_curves([first, second], axis=axis, trials=100)
    series = report["groups"][0]["series"][0]
    assert [point["x"] for point in series["points"]] == [20, None]
    assert series["auc_normalized"] is None
    assert "unknown" in series["points"][1]["budget_status"]


def test_curve_denominator_and_no_extrapolation():
    first, second = record("a"), record("a", budget=120)
    del second["tasks"]["1/1"]
    del second["reference_sha256"]["1/1"]
    del second["layers"]["1/1"]
    second["declared_slots"] -= 2
    seal(second)
    with pytest.raises(ValueError, match="identical task population"):
        budget_curves.budget_curves([first, second], trials=100)
    with pytest.raises(ValueError, match="without extrapolation"):
        budget_curves.area([(1, 0), (2, 1)], 0, 2)
    with pytest.raises(ValueError, match="same observed"):
        budget_curves.area([(1, 0), (1, 1), (2, 1)], 1, 2)


def specification():
    return {"name": "regression", "base_config": {"hardware": "RTX_A6000", "generation": "local", "start": 90, "end": 90},
            "models": ["test/model-a", "test/model-b"], "methods": ["one-shot", "feedback", "harbor"],
            "sessions": 2, "budgets": [60, 120], "vary": "wall"}


def test_plan_and_default_cli_are_nonexecuting_and_real_cli_can_parse_every_job(tmp_path):
    from scripts.plan_study import main

    plan = study_plan.make_plan(specification(), directory=tmp_path)
    assert len(plan["jobs"]) == 24
    path = study_plan.write_plan(plan, tmp_path)
    assert study_plan.execute_plan(path)["experiments_started"] is False
    assert main(["--plan", str(path)]) == 0 and not (tmp_path / "execution.json").exists()
    parsed = []
    def simulated(args):
        config = parse_config(args)
        parsed.append(config)
        return 0
    result = study_plan.execute_plan(path, execute=True, runner=simulated)
    assert len(result["jobs"]) == len(parsed) == 24
    assert {config.agent_timeout for config in parsed} == {60, 120}
    assert {config.generation_mode for config in parsed} == {"one-shot", "feedback", "external"}
    assert all(config.optimization_rounds == 1 and not config.agent_profile for config in parsed if config.generation_mode == "one-shot")
    assert len(read_json(tmp_path / "study.json")["runs"]) == 24


def test_plan_cannot_mislabel_one_shot_or_external_candidate_limits(tmp_path):
    spec = specification()
    spec.update(vary="candidates", budgets=[1, 3, 5], methods=["one-shot", "feedback"], models=["test/model"], sessions=1)
    plan = study_plan.make_plan(spec, directory=tmp_path)
    assert len(plan["jobs"]) == 4 and len(plan["skipped_points"]) == 2
    spec["methods"].append("harbor")
    with pytest.raises(ValueError, match="does not certify"):
        study_plan.make_plan(spec, directory=tmp_path)


def test_modified_plan_is_rejected_and_failure_stops_following_jobs(tmp_path):
    path = study_plan.write_plan(study_plan.make_plan(specification(), directory=tmp_path), tmp_path)
    result = study_plan.execute_plan(path, execute=True, runner=lambda args: 2)
    assert len(result["jobs"]) == 1 and result["jobs"][0]["exit_code"] == 2
    plan = read_json(path)
    plan["jobs"][0]["config"]["agent_timeout"] = 999
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="changed"):
        study_plan.execute_plan(path)


@pytest.mark.parametrize("missing", [False, True])
def test_harbor_accounting_includes_cache_once_and_preserves_missing_step_tokens(tmp_path, missing):
    contexts = [{"n_input_tokens": 100, "n_cache_tokens": 80, "n_output_tokens": 10, "cost_usd": 0.1,
                 "model_usage": {"actual-model": {}}},
                {"n_input_tokens": None if missing else 200, "n_cache_tokens": 150, "n_output_tokens": 20, "cost_usd": 0.2}]
    raw = {"step_results": [{"agent_result": context} for context in contexts],
           "agent_execution": {"started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:03Z"}}
    atomic_json(tmp_path / "trial/result.json", raw)
    result = harbor_budget(tmp_path)
    assert result["generation_wall_seconds"] == 3 and result["models"] == ["actual-model"]
    assert result["total_tokens"] == (None if missing else 330)
    assert result["resource_receipt_verified"] is (not missing)
    assert result["model_calls"] is None


def test_missing_harbor_report_and_invalid_timing_do_not_create_zero_cost(tmp_path):
    assert harbor_budget(tmp_path)["estimated_cost_usd"] is None
    atomic_json(tmp_path / "trial/result.json", {"agent_result": {"n_input_tokens": 1, "n_output_tokens": 2},
                 "agent_execution": {"started_at": "2026-01-01T00:00:03Z", "finished_at": "2026-01-01T00:00:00Z"}})
    result = harbor_budget(tmp_path)
    assert result["generation_wall_seconds"] is None and not result["resource_receipt_verified"]


@pytest.mark.parametrize("complete", [False, True])
def test_claude_token_totals_include_cache_and_do_not_guess_omitted_usage(tmp_path, monkeypatch, complete):
    from types import SimpleNamespace

    from kb_workflows import agent_models
    from kb_workflows.config import RunConfig

    usage = {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 10}
    if complete:
        usage["cache_creation_input_tokens"] = 3
    result = json.dumps({"result": "class ModelNew: pass", "usage": usage, "num_turns": 1})
    process = SimpleNamespace(communicate=lambda *a, **k: (result, None), returncode=0, poll=lambda: 0)
    monkeypatch.setattr(agent_models.subprocess, "Popen", lambda *a, **k: process)
    response = agent_models.claude_response([{"role": "user", "content": "task"}], RunConfig(), tmp_path, 10)
    assert response["total_tokens"] == (20 if complete else None)


def test_plot_export_uses_saved_points_only(tmp_path):
    pytest.importorskip("matplotlib")
    report = budget_curves.budget_curves([record("a"), record("a", budget=120)], trials=100)
    path = tmp_path / "curve.svg"
    budget_curves.plot_curves(report, path)
    assert "<svg" in path.read_text()


def test_selection_cli_requires_complete_selection_or_marks_partial_scope(tmp_path, monkeypatch):
    from scripts import analyze_selection

    monkeypatch.setattr(analyze_selection, "collect_manifest", lambda *a, **k: [record("a"), record("b")])
    selected, output = tmp_path / "selected.json", tmp_path / "selection.json"
    atomic_json(selected, [{"level": 1, "problem_id": 1}, {"level": 1, "problem_id": 999}])
    args = ["--manifest", "unused.json", "--selection", str(selected), "--output", str(output), "--bootstrap-trials", "100"]
    with pytest.raises(ValueError, match="full requested selection"):
        analyze_selection.main(args)
    assert not output.exists()
    assert analyze_selection.main([*args, "--allow-partial-selection"]) == 0
    report = read_json(output)
    assert report["groups"][0]["missing_selection_tasks"] == ["1/999"]
    assert report["groups"][0]["selection_is_partial"] and not report["historical_seven_model_ranking_reproduced"]


def test_budget_cli_exports_report_and_static_figure_without_running_a_model(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from scripts import analyze_budget_curves

    monkeypatch.setattr(analyze_budget_curves, "collect_manifest", lambda *a, **k: [record("a"), record("a", budget=120)])
    output, plot = tmp_path / "curve.json", tmp_path / "curve.svg"
    assert analyze_budget_curves.main(["--manifest", "unused.json", "--output", str(output), "--plot", str(plot),
                                        "--bootstrap-trials", "100"]) == 0
    assert not read_json(output)["experiments_started_by_analysis"]
    assert "<svg" in plot.read_text()
