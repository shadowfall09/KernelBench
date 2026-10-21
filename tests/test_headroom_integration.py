"""Research selection must agree across task execution, scores, budgets and provenance."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter

import pytest

from kb_workflows import engine, generation, harbor, suites
from kb_workflows.common import ROOT, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.experiment import compare_runs, write_budget_report
from kb_workflows.paper_analysis import analyze_paper
from scripts.check_motivations import validate


def research_config(tmp_path, **options):
    return RunConfig(run_name="selection", runs_dir=str(tmp_path), hardware="RTX_A6000",
                     suite="kernelbench105", start=1, end=10, **options).validate()


@pytest.mark.parametrize("level,counts", [(1, (39, 2, 59)), (2, (40, 3, 57)), (3, (26, 0, 24))])
def test_partition_covers_current_references_without_overlap(level, counts):
    roles = {}
    for role, count in zip(suites.ROLES, counts):
        if count == 0:
            with pytest.raises(ValueError, match="No problems remain"):
                select_problems(level, suite="kernelbench105", suite_role=role)
            continue
        roles[role] = select_problems(level, suite="kernelbench105", suite_role=role)
        assert len(roles[role]) == count
    identities = [pid for values in roles.values() for pid in values]
    assert len(identities) == len(set(identities)) == len(select_problems(level))


def test_selection_taxonomy_is_preserved_without_certification():
    main = [row for row in suites.load_manifest()["problems"] if row["role"] == "scored"]
    assert Counter(row["layer"] for row in main) == {"I": 30, "F": 56, "D": 19}
    assert suites.load_manifest()["selection_seed"] == 20261002


def test_range_intersects_suite_and_rejects_empty_or_unknown_request(tmp_path):
    cfg = research_config(tmp_path)
    assert list(select_problems(cfg.level, cfg.start, cfg.end, suite=cfg.suite)) == [2, 3, 6, 8, 9, 10]
    for options in ({"start": 4, "end": 5, "suite": "kernelbench105"}, {"suite": "unknown"},
                    {"suite": "full", "suite_role": "probe"}):
        with pytest.raises(ValueError):
            select_problems(1, **options)


def test_reference_edits_cannot_silently_reuse_reviewed_selection(tmp_path):
    source = select_problems(1, 2, 2)[2]
    target = tmp_path / "KernelBench/level1" / source.name
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    assert list(select_problems(1, 2, 2, root=tmp_path, suite="kernelbench105")) == [2]
    target.write_text(target.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="reference snapshot mismatch"):
        select_problems(1, 2, 2, root=tmp_path, suite="kernelbench105")


def test_toml_and_cli_choose_role_without_changing_default_scope(tmp_path):
    path = tmp_path / "suite.toml"
    path.write_text('[run]\nsuite = "kernelbench105"\nsuite_role = "probe"\nlevel = 2\n')
    cfg = parse_config(["--config", str(path), "--suite-role", "extended"])
    assert (cfg.suite, cfg.suite_role, cfg.level) == ("kernelbench105", "extended", 2)
    assert RunConfig().validate().suite == "full"
    with pytest.raises(ValueError):
        RunConfig(suite_role="probe").validate()


def test_dry_run_is_lightweight_and_creates_no_output(tmp_path):
    code = f'''import sys
sys.path.insert(0, {str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--suite", "kernelbench105", "--suite-role", "probe", "--level", "2",
             "--dry-run", "--runs-dir", {str(tmp_path / "absent")!r}]) == 0
assert not any(name in sys.modules for name in ("torch", "harbor", "modal", "minisweagent"))
'''
    report = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(report.stdout)["problems"] == [23, 80, 83]
    assert not (tmp_path / "absent").exists()


def test_missing_candidates_keep_selection_denominator_in_evaluation_and_score(tmp_path):
    cfg = research_config(tmp_path, num_samples=2)
    engine.evaluate_run(cfg)
    validation = read_json(cfg.run_dir / "validation_summary.json")
    assert validation["counts"]["declared_samples"] == 12
    assert validation["counts"]["correct_samples"] == 0
    assert len(read_json(cfg.run_dir / "pass_at_k_results.json")["problems"]) == 6
    report = analyze_paper(cfg)
    assert report["problem_ids"] == [2, 3, 6, 8, 9, 10]
    assert len(report["details"]) == 12 and report["bounded_gain_macro"] == 0
    assert report["layer_source"] == "provisional_suite_taxonomy"
    assert report["headroom_score_implemented"] and report["score_mode"] == "bounded_gain"
    assert report["headroom_macro"] is None
    assert report["suite"]["layer_certificates"] is False


def test_probe_scores_and_comparisons_are_separate_from_main_selection(tmp_path):
    cfg = research_config(tmp_path, suite_role="probe")
    cfg.end = 100
    cfg.run_dir.mkdir()
    report = analyze_paper(cfg)
    assert report["problem_ids"] == [19, 21]
    assert report["leaderboard_eligible"] is False
    write_budget_report(cfg, {})
    (cfg.run_dir / "run_config.json").write_text(json.dumps(cfg.to_dict()))
    comparison = compare_runs([cfg.run_dir])
    assert comparison["groups"][0]["contract"]["suite"]["role"] == "probe"


def test_baseline_generation_budget_and_child_scope_agree(tmp_path, monkeypatch):
    cfg = research_config(tmp_path, baseline_suite=False)
    seen = []

    def evaluate(config, tasks):
        seen.extend(payload["problem_id"] for _, payload in tasks)
        return iter(())

    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    engine.record_baseline(cfg)
    assert seen == [2, 3, 6, 8, 9, 10]
    budget = write_budget_report(cfg, {})
    assert budget["declared_slots"] == 6
    assert set(budget["slots"]) == {f"{pid}_0" for pid in seen}
    cfg.generation = "local"
    directory = generation.prepare_job(cfg, 2, 0)
    job = read_json(directory / "job_config.json")
    assert (job["suite"], job["suite_role"], job["start"], job["end"]) == ("full", "scored", 2, 2)


def test_native_generation_runs_every_selected_sample_only(tmp_path, monkeypatch):
    cfg = research_config(tmp_path, generation="local", num_samples=2, parallel=1)
    seen = []

    def generate(directory, config, pid, device):
        seen.append(pid)
        path = directory / "runs/agent" / kernel_name(config.level, pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# simulated generated source")
        return {"status": "success"}

    monkeypatch.setattr(generation, "run_agent", generate)
    monkeypatch.setattr(generation, "resolve_devices", lambda devices: ["simulated-device"])
    assert generation.generate_run(cfg)
    assert Counter(seen) == {pid: 2 for pid in [2, 3, 6, 8, 9, 10]}
    assert len(read_json(cfg.run_dir / "generation_summary.json")) == 12
    budget = read_json(cfg.run_dir / "generation_budget.json")
    assert budget["declared_slots"] == 12 and budget["suite"]["name"] == "kernelbench105"


def test_real_harbor_export_keeps_only_selected_probes_and_portable_children(tmp_path):
    Task = pytest.importorskip("harbor.models.task.task").Task
    cfg = research_config(tmp_path, suite_role="probe")
    cfg.start, cfg.end = 19, 21
    output = harbor.export_tasks(cfg)
    assert sorted(path.name for path in output.iterdir()) == ["l1-p19-s0", "l1-p21-s0"]
    for task in output.iterdir():
        assert Task.is_valid_dir(task)
        child = read_json(task / "tests/config.json")["config"]
        assert (child["suite"], child["suite_role"]) == ("full", "scored")
        assert (task / "tests/repo/kb_workflows/suites/kernelbench105.json").is_file()
    assert read_json(cfg.run_dir / "harbor_export.json")["suite"]["role"] == "probe"


def test_partial_explicit_layers_do_not_claim_all_labels_were_certified(tmp_path):
    cfg = research_config(tmp_path)
    path = tmp_path / "layers.json"
    path.write_text(json.dumps({"level1": {"2": {"layer": "D"}}}))
    cfg.problem_specs = str(path)
    cfg.run_dir.mkdir()
    report = analyze_paper(cfg)
    by_pid = {row["problem_id"]: row for row in report["details"]}
    assert by_pid[2]["layer"] == "D" and by_pid[2]["layer_source"] == "problem_specs"
    assert by_pid[3]["layer_source"] == "provisional_suite_taxonomy"
    assert report["suite"]["layer_certificates"] is False


def test_audit_and_selection_artifacts_match_provenance_and_totals():
    directory = ROOT / "research/headroom/evidence"
    provenance = read_json(directory / "source_manifest.json")
    for name, checksum in provenance["outputs"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum
    with (directory / "audit_implementations.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    stats = read_json(directory / "audit_statistics.json")
    assert len(rows) == stats["implementations"] == 1574
    assert Counter(row["source_review_verdict"] for row in rows) == stats["by_severity"]
    assert all(row["raw_source_available"] == "False" for row in rows)
    assert sum("noise" in row["flags"].split(";") for row in rows) == 346
    assert stats["by_severity"]["噪声"] == 217
    assert provenance["candidate_gpu_runs"] is False


def test_every_change_has_a_current_motivation_and_real_local_evidence():
    report = validate()
    assert report["historical_changes"] == 67
    assert report["implemented_changes"] == 92
    assert report["planned_designs"] == 10
    assert report["in_progress_designs"] == 7
    assert report["completed_designs"] == 6


def test_motivation_validation_detects_inventory_drift(tmp_path):
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    path = tmp_path / "docs/FEATURE_INVENTORY.md"
    path.write_text(path.read_text().replace("严格比较输出结构、shape 和 dtype", "unmapped change", 1))
    # Coverage and titles are validated before evidence access.
    with pytest.raises(ValueError):
        validate(tmp_path)
