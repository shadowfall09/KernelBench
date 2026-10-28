"""CPU dispatch controls and simulated process receipts; no CUDA experiments."""

import copy
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from kb_workflows import engine
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation import allocation_audit as audit
from kb_workflows.evaluation import evaluate_task, tool_checks, tool_runner
from src import eval as native
from src.reliability import check_paired_calls

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128,
          torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]
GOOD = '''import torch
CACHE = torch.empty(3, 17)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.workspace = torch.empty(3, 17)
    def forward(self, x):
        output = torch.empty_like(x)
        output.copy_(torch.relu(x))
        return output
'''


def payload(tmp_path, *, source=GOOD, **options):
    config = RunConfig(generation="local", hardware="RTX_A6000", runs_dir=str(tmp_path),
                       start=19, end=19, allocation_validation=True, **options).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / kernel_name(1, 19)).write_text(source)
    return engine.task_payload(config, 19), config


@contextmanager
def collector(tmp_path, pattern):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "a" * 32, audit.STAGES[pattern])
    with patch.dict(os.environ, tool_runner.environment(allocation=True), clear=True), audit.audit(child, pattern) as capture:
        yield capture, child


# Allocation frontends and dtype handling are independent dimensions. Cover
# every frontend/pattern for each dtype class, and every remaining dtype once.
@pytest.mark.parametrize("dtype,pattern,factory",
    [(dtype, pattern, factory) for dtype in (torch.float32, torch.complex64, torch.int64, torch.bool)
     for pattern in audit.PATTERNS for factory in ("empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided")]
    + [(dtype, pattern, "empty") for dtype in DTYPES
       if dtype not in (torch.float32, torch.complex64, torch.int64, torch.bool) for pattern in audit.PATTERNS])
def test_all_reviewed_dense_allocation_frontends_poison_native_storage_without_switching_algorithms(tmp_path, dtype, pattern, factory):
    base = torch.ones(3, 5, dtype=dtype)
    deterministic = torch.are_deterministic_algorithms_enabled()
    with collector(tmp_path, pattern) as (capture, _), audit.candidate_scope("constructor"):
        if factory == "empty":
            output = torch.empty((3, 5), dtype=dtype)
        elif factory == "empty_like":
            output = torch.empty_like(base)
        elif factory == "empty_strided":
            output = torch.empty_strided((3, 5), (10, 2), dtype=dtype)
        elif factory == "new_empty":
            output = base.new_empty((3, 5))
        else:
            output = base.new_empty_strided((3, 5), (10, 2))
    if dtype == torch.bool:
        assert (output == (pattern == "nan_max")).all()
    elif output.is_complex():
        assert (torch.isnan(output.real).all() and torch.isnan(output.imag).all()) if pattern == "nan_max" else (output == complex(-.5, .5)).all()
    elif output.is_floating_point():
        assert torch.isnan(output).all() if pattern == "nan_max" else (output == -.5).all()
    else:
        assert (output == (torch.iinfo(dtype).max if pattern == "nan_max" else torch.iinfo(dtype).min)).all()
    row = capture.document()["windows"][0]
    assert row["mode_present_after"] and row["environment_changes"] == []
    assert sum(group["allocations"] for group in row["allocations"]) == 1
    assert sum(group["elements"] for group in row["allocations"]) == 15
    assert torch.are_deterministic_algorithms_enabled() == deterministic


@pytest.mark.parametrize("pattern", audit.PATTERNS)
def test_resize_growth_poison_preserves_initialized_prefix_and_shrink_is_unchanged(tmp_path, pattern):
    original = torch.arange(4.)
    with collector(tmp_path, pattern) as (capture, _), audit.candidate_scope("constructor"):
        value = original.clone()
        value.resize_(9)
        assert torch.equal(value[:4], original)
        assert torch.isnan(value[4:]).all() if pattern == "nan_max" else (value[4:] == -.5).all()
        value.resize_(3)
        assert torch.equal(value, original[:3])
    groups = capture.document()["windows"][0]["allocations"]
    assert len(groups) == 1 and groups[0]["elements"] == 5 and groups[0]["status"] == "poisoned"


def cpu_worker(request):
    from tests.evaluation.test_tool_checks import DEVICE

    Model, _, _ = native.load_original_model_and_inputs(request["ref_arch_src"], {})
    ModelNew = native.load_custom_model(request["kernel_src"], {})
    with audit.candidate_scope("constructor"):
        model = ModelNew()
    metadata = {"reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                "kernel_sha256": fingerprint({"source": request["kernel_src"]}), "device_fingerprint": copy.deepcopy(DEVICE)}
    args = [torch.linspace(-2, 2, 51).reshape(3, 17)]
    issue = check_paired_calls(Model(), model, args, device="cpu", seed=42, set_seed=torch.manual_seed,
                               tolerance=1e-5, metadata=metadata, label="standard/0")
    return {"compiled": True, "correctness": issue is None, "metadata": metadata,
            "runtime": 7, "runtime_stats": {"samples_ms": [7], "num_trials": 1}}


@pytest.mark.parametrize("pattern", audit.PATTERNS)
def test_actual_cpu_native_import_constructor_and_calls_produce_replayable_full_coverage(tmp_path, monkeypatch, pattern):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "a" * 32, audit.STAGES[pattern])
    monkeypatch.setattr(evaluate_task, "dispatch", cpu_worker)
    with patch.dict(os.environ, tool_runner.environment(allocation=True), clear=True):
        result = evaluate_task.allocation_control(child)
    assert result["correctness"]
    record = result["metadata"]["allocation_audit"]
    windows = record["windows"]
    assert [row["phase"] for row in windows] == ["import", "constructor", "forward", "forward"]
    assert all(sum(group["allocations"] for group in row["allocations"]) == 1 for row in windows)
    check = audit.verdict(record, child, result["metadata"]["reliability_checks"], pattern=pattern)
    assert check["status"] == "passed" and check["values"]["poisoned_allocations"] == 4
    assert check["values"]["complete_device_memory_coverage"] is False


@pytest.mark.parametrize("kind", ["partial", "constructor_cache", "import_cache", "all_nan"])
def test_poisoned_controls_reject_partial_writes_and_persistent_unwritten_caches(tmp_path, monkeypatch, kind):
    if kind == "partial":
        source = GOOD.replace("output.copy_(torch.relu(x))", "output[:, :8].copy_(torch.relu(x[:, :8]))")
    elif kind == "constructor_cache":
        source = GOOD.replace("output = torch.empty_like(x)", "output = self.workspace").replace("output.copy_(torch.relu(x))", "pass")
    elif kind == "import_cache":
        source = GOOD.replace("output = torch.empty_like(x)", "output = CACHE").replace("output.copy_(torch.relu(x))", "pass")
    else:
        source = GOOD.replace("output.copy_(torch.relu(x))", "pass")
    request, _ = payload(tmp_path, source=source)
    monkeypatch.setattr(evaluate_task, "dispatch", cpu_worker)
    with patch.dict(os.environ, tool_runner.environment(allocation=True), clear=True):
        for pattern in audit.PATTERNS:
            child = tool_checks.child_payload(request, "a" * 32, audit.STAGES[pattern])
            result = evaluate_task.allocation_control(child)
            assert not result["correctness"]
            assert audit.verdict(result["metadata"]["allocation_audit"], child,
                result["metadata"]["reliability_checks"], pattern=pattern)["status"] == "failed"


def simulated_executor(monkeypatch, *, failure=None):
    from tests.evaluation.test_tool_checks import native_result

    calls = []
    monkeypatch.setattr(evaluate_task, "dispatch", cpu_worker)

    def execute(command, timeout, log, *, environment):
        index = command.index("-m")
        source, target = map(Path, command[index + 2:index + 4])
        request = json.loads(source.read_text())
        stage = request["eval_config"].get("diagnostic_stage", "native")
        calls.append((stage, copy.deepcopy(environment)))
        if stage in audit.STAGES.values():
            with patch.dict(os.environ, environment, clear=True):
                result = evaluate_task.allocation_control(request)
            if failure == "missing":
                result["metadata"].pop("allocation_audit")
            elif failure == "device":
                result["metadata"]["device_fingerprint"]["uuid"] = "OTHER"
            elif failure == "seed":
                result["metadata"]["allocation_audit"]["windows"][-1]["seed"] += 1
        else:
            result = copy.deepcopy(native_result(request))
        atomic_json(target, result)
        Path(log).write_text("CPU simulated diagnostic process")
        return {"exit_code": 124 if failure == "timeout" and stage in audit.STAGES.values() else 0,
                "timed_out": failure == "timeout" and stage in audit.STAGES.values(), "request_wall_seconds": .01}
    return execute, calls


def test_parent_owns_two_separate_patterns_environment_and_preserves_scored_times(tmp_path, monkeypatch):
    request, _ = payload(tmp_path)
    run, calls = simulated_executor(monkeypatch)
    result = tool_runner.evaluate(request, run=run)
    assert result["correctness"] and result["runtime"] == 1 and result["runtime_stats"]["samples_ms"] == [1]
    assert [stage for stage, _ in calls] == ["native", *audit.STAGES.values(), "control"]
    for stage, environment in calls:
        if stage in audit.STAGES.values():
            assert all(environment[name] == value for name, value in audit.ENVIRONMENT.items())
            assert "PYTORCH_CUDA_ALLOC_CONF" not in environment
    assert all(row["status"] == "passed" for row in tool_checks.validate(result, request, allow_simulated=True).values())
    assert tool_checks.issue(result, request) == "tool_checks_missing_stale_or_invalid"


@pytest.mark.parametrize("failure", ["missing", "device", "seed", "timeout"])
def test_missing_or_changed_simulated_control_evidence_cannot_qualify(tmp_path, monkeypatch, failure):
    request, _ = payload(tmp_path)
    run, _ = simulated_executor(monkeypatch, failure=failure)
    result = tool_runner.evaluate(request, run=run)
    assert not result["correctness"]
    checks = tool_checks.validate(result, request, allow_simulated=True)
    assert all(checks[name]["status"] == "not_measured" for name in audit.STAGES.values())


@pytest.mark.parametrize("damage", ["environment", "setup", "mode", "pattern", "device_init", "counters", "unsupported"])
def test_receipt_replay_rejects_removed_or_changed_poison_windows(tmp_path, monkeypatch, damage):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "a" * 32, audit.STAGES["nan_max"])
    monkeypatch.setattr(evaluate_task, "dispatch", cpu_worker)
    with patch.dict(os.environ, tool_runner.environment(allocation=True), clear=True):
        result = evaluate_task.allocation_control(child)
    document = result["metadata"]["allocation_audit"]
    if damage == "environment":
        document["environment"]["CUDA_LAUNCH_BLOCKING"] = False
    elif damage == "setup":
        document["windows"].pop(0)
    elif damage == "mode":
        document["windows"][-1]["mode_present_after"] = False
    elif damage == "pattern":
        document["pattern"] = "finite_min"
    elif damage == "device_init":
        document["cuda_initialized_before_audit"] = True
    else:
        group = document["windows"][-1]["allocations"][0]
        group["elements" if damage == "counters" else "status"] = -1 if damage == "counters" else "unsupported_fill"
    assert audit.verdict(document, child, result["metadata"]["reliability_checks"], pattern="nan_max")["status"] == "not_measured"


def test_configuration_freezing_and_preview_require_no_tensor_sdk(tmp_path):
    from kb_workflows.analysis import experiment, study_data
    from kb_workflows.evaluation import final_verification

    config = parse_config(["--allocation-validation"])
    assert config.allocation_validation and not parse_config(["--no-allocation-validation"]).allocation_validation
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="Allocation validation"):
            RunConfig(allocation_validation=value).validate()
    with pytest.raises(ValueError, match="CUDA"):
        RunConfig(generation="local", device_backend="neuron", hardware="Trainium2", enable_ncu=False,
                  allocation_validation=True).validate()
    config = RunConfig(generation="local")
    before = final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config)
    config.allocation_validation = True
    assert before != (final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config))
    file = tmp_path / "allocation.toml"
    file.write_text('[run]\nallocation_validation=true\n')
    assert parse_config(["--config", str(file)]).allocation_validation
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--allocation-validation","--dry-run","--start","19","--end","19"]) == 0
assert not {{"torch","triton","nki"}} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.int64, torch.bool])
def test_second_pattern_detects_unwritten_outputs_even_when_first_pattern_matches_the_reference(tmp_path, dtype):
    if dtype == torch.bool:
        fill = True
    elif dtype == torch.int64:
        fill = torch.iinfo(dtype).max
    elif dtype.is_complex:
        fill = complex(float("nan"), float("nan"))
    else:
        fill = float("nan")

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, fill)

    class Unwritten(torch.nn.Module):
        def forward(self, x):
            return torch.empty_like(x)

    results = []
    for pattern in audit.PATTERNS:
        with collector(tmp_path, pattern):
            metadata = {}
            issue = check_paired_calls(Reference(), Unwritten(), [torch.ones(3, 5, dtype=dtype)], seed=42,
                device="cpu", set_seed=torch.manual_seed, tolerance=0, metadata=metadata, label="first_pattern_collision")
            results.append(issue is None)
    assert results == [True, False]


def test_quantized_and_noncontiguous_resize_allocations_remain_explicitly_unsupported(tmp_path):
    value = torch.arange(6.).reshape(2, 3).t()
    with collector(tmp_path, "nan_max") as (capture, _), audit.candidate_scope("constructor"):
        torch._empty_affine_quantized([3, 5], scale=1, zero_point=0, dtype=torch.qint8)
        value.resize_(9)
    groups = capture.document()["windows"][0]["allocations"]
    assert {row["status"] for row in groups} == {"unsupported_tensor", "unsupported_resize_growth"}


def test_poison_scope_restores_environment_rng_and_deterministic_configuration(tmp_path):
    before = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with collector(tmp_path, "nan_max") as (capture, _), audit.candidate_scope("constructor"):
        torch.empty(3, 5)
        os.environ["KB_ALLOCATION_TEST"] = "fixture"
        torch.use_deterministic_algorithms(not deterministic, warn_only=True)
    assert "KB_ALLOCATION_TEST" not in os.environ
    assert torch.equal(before, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert set(capture.document()["windows"][0]["environment_changes"]) == {"env.KB_ALLOCATION_TEST", "deterministic_policy"}


def test_native_queue_preflight_and_private_harbor_bundle_share_the_policy(tmp_path):
    from kb_workflows.runtime import harbor
    from kb_workflows.runtime.queue_worker import validate_payload

    request, _ = payload(tmp_path)
    validate_payload(request)
    for value in (1, "yes"):
        changed = copy.deepcopy(request)
        changed["eval_config"]["allocation_validation"] = value
        with pytest.raises(ValueError, match="Allocation validation"):
            validate_payload(changed)
    changed = copy.deepcopy(request)
    changed["eval_config"]["device_backend"] = "neuron"
    with pytest.raises(ValueError, match="CUDA"):
        validate_payload(changed)
    config = RunConfig(generation="harbor", hardware="RTX_A6000", allocation_validation=True, start=19, end=19).validate()
    task = harbor.write_task(tmp_path / "harbor-task", config, 19)
    assert json.loads((task / "tests/config.json").read_text())["config"]["allocation_validation"] is True
    assert (task / "tests/repo/kb_workflows/evaluation/allocation_audit.py").read_bytes() == (ROOT / "kb_workflows/evaluation/allocation_audit.py").read_bytes()


def test_meta_allocations_record_unknown_storage_without_materializing_or_poisoning(tmp_path):
    with collector(tmp_path, "nan_max") as (capture, _), audit.candidate_scope("constructor"):
        value = torch.empty(3, 5, device="meta")
    assert value.device.type == "meta"
    group = capture.document()["windows"][0]["allocations"][0]
    assert group["status"] == "metadata_only" and group["elements"] is None


def test_poison_control_cannot_run_a_baseline_or_collect_performance(tmp_path):
    request, _ = payload(tmp_path)
    request["eval_config"]["diagnostic_stage"] = audit.STAGES["nan_max"]
    for task, measured in (("baseline", False), ("evaluate", True)):
        request["task"] = task
        request["eval_config"]["measure_performance"] = measured
        with pytest.raises(ValueError, match="untimed correctness"):
            audit.Collector(request, "nan_max")
