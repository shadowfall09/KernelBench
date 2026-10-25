"""Task repair semantics and source lineage, with CPU/simulated execution only."""

import copy
import hashlib
import importlib
import json
import subprocess
import sys

import pytest
import torch

from kb_workflows.benchmarks import reference_revisions, suites
from kb_workflows.common import ROOT, evaluator_digest, fingerprint, kernel_name, problem_paths
from kb_workflows.config import RunConfig
from kb_workflows.engine import task_payload
from kb_workflows.evaluation import evaluate_task, tool_checks
from kb_workflows.numerics.numerical import PrecisionContract, precision_gate
from kb_workflows.numerics.operator_oracles import execute
from kb_workflows.runtime.queue_worker import validate_payload
from src.eval import load_original_model_and_inputs


def model92(dim):
    Model, _, _ = load_original_model_and_inputs(problem_paths(1)[92].read_text(), {})
    return Model(dim)


def request(tmp_path, pid=92, *, hidden=False, task="evaluate"):
    config = RunConfig(generation="local", hardware="RTX_A6000", runs_dir=str(tmp_path), run_name=f"rev-{pid}",
                       start=pid, end=pid).validate()
    config.run_dir.mkdir(exist_ok=True)
    (config.run_dir / kernel_name(1, pid)).write_text("candidate")
    return task_payload(config, pid, task=task, hidden=hidden)


@pytest.mark.parametrize("shape,axis", [((3, 7), 0), ((3, 7), 1), ((3, 7), -1), ((3, 7), -2),
                                      ((2, 3, 5), 1), ((2, 3, 5), -1), ((4,), 0), ((1,), -1),
                                      ((0,), 0), ((3, 0), 1), ((0, 7), 0), ((3, 0, 5), 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int16, torch.bool])
def test_repaired_exclusive_scan_preserves_shape_axis_dtype_promotion_and_empty_scans(shape, axis, dtype):
    model = model92(axis)
    inputs = torch.arange(max(1, torch.tensor(shape).prod().item()))[:torch.tensor(shape).prod().item()].reshape(shape)
    inputs = ((inputs % 5) - 2).to(dtype)
    before = inputs.clone()
    actual = model(inputs)
    expected = torch.zeros_like(inputs, dtype=inputs.cumsum(axis).dtype)
    # A prefix independently excludes the current element at every position.
    for position in range(inputs.shape[axis]):
        expected.select(axis, position).copy_(inputs.narrow(axis, 0, position).sum(axis))
    assert actual.shape == inputs.shape and actual.dtype == expected.dtype
    assert torch.equal(actual, expected) and torch.equal(inputs, before)
    truth = execute(92, model, [inputs])
    assert truth.shape == actual.shape
    assert torch.equal(actual.to(truth.dtype), truth)


def test_repaired_scan_supports_noncontiguous_inputs_and_rejects_the_legacy_axis_shortcut():
    model = model92(1)
    x = torch.randn(3, 34)[:, ::2]
    truth = execute(92, model, [x])
    expected = model(x)
    assert precision_gate(expected, expected.clone(), truth, PrecisionContract(5e-6, 1e-7, 5e-5, 5e-7, 1e-6))["passed"]
    legacy = torch.cat((torch.zeros_like(x.select(1, 0).unsqueeze(1)), x), dim=1)[:-1].cumsum(1)
    assert legacy.shape != expected.shape
    assert not precision_gate(legacy, legacy.clone(), truth, PrecisionContract(5e-6, 1e-7, 5e-5, 5e-7, 1e-6))["reference_qualified"]


def test_current_revision_registry_matches_every_physical_source_and_hidden_file():
    data = reference_revisions.registry()
    assert {(row["level"], row["problem_id"]) for row in data["revisions"]} == {(1, pid) for pid in (*range(89, 94), 98, 100)}
    for row in data["revisions"]:
        source = problem_paths(row["level"])[row["problem_id"]]
        hidden = ROOT / "hidden_tests" / f"level{row['level']}" / f"{row['problem_id']}_hidden.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["reference_sha256"]
        assert hashlib.sha256(hidden.read_bytes()).hexdigest() == row["hidden_sha256"]
        assert reference_revisions.binding(row["level"], row["problem_id"], source.read_text())["id"] == row["id"]
    assert reference_revisions.binding(1, 2, "unrevised source") is None


@pytest.mark.parametrize("damage", ["id", "before_hash", "gap", "first_revision", "evidence"])
def test_malformed_or_discontinuous_revision_history_is_rejected(tmp_path, monkeypatch, damage):
    data = reference_revisions.registry()
    row = copy.deepcopy(data["revisions"][0])
    if damage == "id":
        data["revisions"].append(row)
    elif damage in {"before_hash", "gap"}:
        row["id"] += "-next"
        row["revision"] = 3 if damage == "before_hash" else 4
        row["reference_before_sha256"] = "0" * 64
        data["revisions"].append(row)
    elif damage == "first_revision":
        data["revisions"][0]["revision"] = 3
    else:
        data["revisions"][0]["evidence"] = []
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(reference_revisions, "REGISTRY", path)
    with pytest.raises(ValueError):
        reference_revisions.registry()


@pytest.mark.parametrize("pid", range(89, 94))
@pytest.mark.parametrize("hidden", [False, True])
def test_native_queue_payloads_carry_source_and_hidden_revision(tmp_path, pid, hidden):
    payload = request(tmp_path, pid, hidden=hidden)
    expected = reference_revisions.validate_payload(payload)
    assert expected["revision"] == 2 and expected["id"].startswith(f"l1-{pid}-")
    validate_payload(payload)
    assert evaluate_task.validate_sources(payload) == payload["ref_arch_src"]
    changed = copy.deepcopy(payload)
    changed["eval_config"].pop("reference_revision")
    with pytest.raises(ValueError, match="reference revision"):
        validate_payload(changed)
    if hidden:
        changed = copy.deepcopy(payload)
        changed["eval_config"]["hidden_source_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="Hidden tests"):
            reference_revisions.validate_payload(changed)


def test_retired_source_is_rejected_even_with_a_current_evaluator_digest(tmp_path):
    payload = request(tmp_path)
    payload["ref_arch_src"] = "old or changed task source"
    payload["eval_config"]["evaluator_sha256"] = evaluator_digest()
    with pytest.raises(ValueError, match="current declared revision"):
        reference_revisions.validate_payload(payload)


@pytest.mark.parametrize("task", ["evaluate", "baseline", "paired_timing", "profile", "qualify_small", "origin_audit"])
def test_shared_dispatch_records_revision_for_every_native_task_without_running_a_device(tmp_path, monkeypatch, task):
    payload = request(tmp_path, task=task)
    called = []

    def worker(value):
        called.append(value)
        return {"compiled": True, "correctness": True, "metadata": {}, "runtime": 1}

    modules = {"paired_timing": ("kb_workflows.evaluation.paired_task", "evaluate_paired"),
               "profile": ("kb_workflows.runtime.profile_task", "profile"),
               "origin_audit": ("kb_workflows.provenance.origin_task", "audit")}
    if task in modules:
        module, name = modules[task]
        monkeypatch.setattr(importlib.import_module(module), name, worker)
    else:
        monkeypatch.setattr(evaluate_task, "evaluate" if task == "evaluate" else task, worker)
    result = evaluate_task.dispatch(payload)
    assert called == [payload]
    assert result["metadata"]["reference_revision"] == payload["eval_config"]["reference_revision"]
    assert hasattr(evaluate_task.dispatch, "__wrapped__")


def test_positive_scoring_and_resume_require_the_current_revision_receipt(tmp_path):
    payload = request(tmp_path)
    result = {"compiled": True, "correctness": True, "metadata": {"workflow_signature": fingerprint(payload)}}
    assert tool_checks.issue(result, payload) == "task_reference_revision_evidence_missing_or_changed"
    assert not tool_checks.reusable(result, payload)
    result["metadata"]["reference_revision"] = payload["eval_config"]["reference_revision"]
    assert tool_checks.issue(result, payload) is None
    assert tool_checks.reusable(result, payload)
    previous = copy.deepcopy(result)
    previous["metadata"]["reference_revision"]["revision"] = 1
    assert not tool_checks.reusable(previous, payload)
    negative = {"correctness": False, "metadata": {"workflow_signature": fingerprint(payload)}}
    assert tool_checks.reusable(negative, payload)


def test_suite_revision_preserves_task_selection_and_old_research_evidence():
    manifest = suites.load_manifest()
    assert manifest["version"] == 2 and manifest["selection_seed"] == 20261002
    assert manifest["selection_reference_snapshot"] == "92c27f4c020a4860d5ca546b88a50b007f629e3f"
    assert {role: sum(row["role"] == role for row in manifest["problems"]) for role in suites.ROLES} == {
        "scored": 105, "probe": 5, "extended": 140}
    repaired = [row for row in manifest["problems"] if row.get("reference_revision")]
    assert {row["problem_id"] for row in repaired} == {*range(89, 94), 98, 100}
    receipt = suites.selection_receipt(RunConfig(suite="kernelbench105"))
    assert receipt["reference_revision_registry_sha256"] == hashlib.sha256(reference_revisions.REGISTRY.read_bytes()).hexdigest()
    archive = json.loads((ROOT / "research/headroom/evidence/source_manifest.json").read_text())
    assert hashlib.sha256((ROOT / "research/headroom/evidence/selection.csv").read_bytes()).hexdigest() == archive["outputs"]["selection.csv"]


def test_revision_previews_and_registry_load_without_torch(tmp_path):
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--suite","kernelbench105","--level","1","--start","89","--end","93","--dry-run"]) == 0
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["suite"]["version"] == 2


def test_harbor_exports_current_revision_and_both_hidden_and_source_files(tmp_path):
    from kb_workflows.runtime import harbor

    config = RunConfig(generation="harbor", hardware="RTX_A6000", start=92, end=92).validate()
    directory = harbor.write_task(tmp_path / "task", config, 92)
    assert (directory / "tests/repo/kb_workflows/benchmarks/reference_revisions.json").is_file()
    assert (directory / "tests/repo/hidden_tests/level1/92_hidden.py").read_bytes() == (ROOT / "hidden_tests/level1/92_hidden.py").read_bytes()
    assert (directory / "tests/repo/KernelBench/level1/92_cumsum_exclusive.py").read_bytes() == problem_paths(1)[92].read_bytes()


@pytest.mark.parametrize("option,value", [("state_validation", True), ("numerical_validation", True), ("oracle_memory_mb", 32)])
def test_diagnostic_policy_changes_invalidate_all_frozen_generation_study_and_rl_contracts(tmp_path, option, value):
    from kb_workflows.agents import generation
    from kb_workflows.analysis import experiment, study_data
    from kb_workflows.evaluation import final_verification

    config = RunConfig(generation="local", hardware="RTX_A6000", start=2, end=2, runs_dir=str(tmp_path)).validate()
    readers = [lambda: experiment.protocol(config), lambda: final_verification._options(config),
               lambda: study_data.evaluation_contract(config),
               lambda: generation.generation_signature(config, 2, 0)]
    before = [read() for read in readers]
    setattr(config, option, value)
    assert all(old != read() for old, read in zip(before, readers))


@pytest.mark.parametrize("option,value", [("state_validation", True), ("numerical_validation", True), ("oracle_memory_mb", 32)])
def test_actual_rl_plan_binds_diagnostic_settings_without_loading_or_training_a_model(tmp_path, option, value):
    from kb_workflows.agents import rl_plan

    checkpoint = tmp_path / "metadata-only-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    specification = {"training": {"checkpoint": str(checkpoint), "mode": "one-shot", "group_size": 2, "max_steps": 2},
        "verifier_config": {"hardware": "RTX_A6000", "final_repeats": 2, "paired_trials": 10},
        "train": [{"level": 1, "problem_id": 2}], "heldout": [{"level": 1, "problem_id": 26}]}
    directory = tmp_path / "plan-output"
    before = rl_plan.make_plan(specification, directory=directory)
    specification["verifier_config"][option] = value
    after = rl_plan.make_plan(specification, directory=directory)
    assert before["plan_sha256"] != after["plan_sha256"]
    assert after["verifier_config"][option] == value
    rl_plan.validate_plan(after)
    tampered = copy.deepcopy(before)
    tampered["verifier_config"][option] = value
    with pytest.raises(ValueError, match="plan changed"):
        rl_plan.validate_plan(tampered)


@pytest.mark.parametrize("option,value", [("state_validation", True), ("numerical_validation", True), ("oracle_memory_mb", 32)])
def test_run_comparison_cannot_pool_different_diagnostic_policies(tmp_path, monkeypatch, option, value):
    from kb_workflows.analysis import experiment
    from kb_workflows.common import atomic_json

    paths = []
    for index in range(2):
        config = RunConfig(generation="local", hardware="RTX_A6000", start=2, end=2,
                           runs_dir=str(tmp_path), run_name=f"run-{index}").validate()
        if index:
            setattr(config, option, value)
        config.run_dir.mkdir()
        atomic_json(config.run_dir / "run_config.json", config.to_dict())
        paths.append(config.run_dir)
    monkeypatch.setattr(experiment, "existing_budget", lambda _: {"protocol": {"mode": "one-shot"}})
    result = experiment.compare_runs(paths)
    assert len(result["groups"]) == 2


@pytest.mark.parametrize("key,value", [("level", "1"), ("level", True), ("problem_id", "92"),
                                      ("ref_arch_src", 92), ("eval_config", [])])
def test_malformed_request_types_cannot_bypass_repaired_task_identity(tmp_path, key, value):
    payload = request(tmp_path)
    payload[key] = value
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_changed_shared_hidden_generator_cannot_keep_the_old_task_revision(tmp_path, monkeypatch):
    from pathlib import Path

    payload = request(tmp_path)
    original = Path.read_bytes

    def changed(path):
        raw = original(path)
        return raw + b"\n# changed family generator" if path.name == "scan_cases.py" else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="Hidden input dependency"):
        reference_revisions.validate_payload(payload)
