"""Small CPU functional tests and explicit device simulations; no experiments."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from kb_workflows import cache_audit as audit
from kb_workflows import cache_recipes, engine, tool_checks, tool_runner
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, problem_paths
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.input_reuse import MutableInputs, description, isolated
from src import eval as native


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def payload(tmp_path, **options):
    config = RunConfig(generation="local", runs_dir=str(tmp_path), start=19, end=19, cache_validation=True,
                       num_correct_trials=2, **options).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    source = problem_paths(1)[19].read_text()
    (config.run_dir / kernel_name(1, 19)).write_text(source + "\nModelNew=Model\n")
    return engine.task_payload(config, 19), config


def execute(request, candidate=None, *, trials=2):
    from test_tool_checks import DEVICE

    capture = audit.Collector(request, sync=lambda device: None) if request["eval_config"].get("diagnostic_stage") == audit.STAGE else None
    token = audit.ACTIVE.set(capture)
    try:
        metadata = {"device_fingerprint": copy.deepcopy(DEVICE),
                    "reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": request["kernel_src"]})}
        result = native.run_and_check_correctness(torch.nn.ReLU(), candidate or torch.nn.ReLU(),
            lambda: [torch.randn(3, 17)], metadata, num_correct_trials=trials, device="cpu", precision=torch.float32)
        if capture:
            metadata["input_cache_audit"] = capture.document()
        return {"compiled": result.compiled, "correctness": result.correctness, "metadata": metadata,
                "runtime": 1.0, "runtime_stats": {"samples_ms": [1.0]}}
    finally:
        audit.ACTIVE.reset(token)


def test_native_factory_same_objects_every_case_and_rng_restoration(tmp_path):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    result = execute(child)
    document = result["metadata"]["input_cache_audit"]
    cases = result["metadata"]["reliability_checks"]
    assert result["correctness"] and len(document["cases"]) == 2
    assert audit.verdict(document, child, cases, allow_simulated=True)["status"] == "passed"
    assert audit.verdict(document, child, cases)["status"] == "not_measured"
    assert all(len({probe["input_sha256"] for probe in row["probes"]}) == 3 for row in document["cases"])
    with patch.object(audit, "observe"):
        plain = execute(child)
    assert [r["seed"] for r in plain["metadata"]["reliability_checks"]] == [r["seed"] for r in cases]


@pytest.mark.parametrize("key", ["id", "data_ptr"])
def test_candidate_cached_by_object_or_pointer_fails_updated_contents(tmp_path, key):
    class Cached(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.saved, self.retain = {}, []

        def forward(self, x):
            self.retain.append(x)  # Prevent allocator reuse from confusing this fixture.
            index = id(x) if key == "id" else x.data_ptr()
            if index not in self.saved:
                self.saved[index] = torch.relu(x)
            return self.saved[index]

    request, _ = payload(tmp_path)
    assert execute(request, Cached())["correctness"]
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    result = execute(child, Cached())
    assert result["correctness"]  # Separate diagnostic fails; native verdict is retained.
    assert audit.verdict(result["metadata"]["input_cache_audit"], child,
        result["metadata"]["reliability_checks"], allow_simulated=True)["status"] == "failed"


def test_aliases_strides_offsets_and_repeated_objects_are_isolated_and_reused():
    base = torch.arange(48.).reshape(6, 8)
    first, second = base[1:5, ::2], base[2:6, 1::2]
    args = [first, second, first, base[0:1, :].expand(3, 8), 1.0]
    group = MutableInputs(args)
    assert description(group.values) == description(args)
    assert group.values[0] is group.values[2]
    assert group.values[0].untyped_storage()._cdata == group.values[1].untyped_storage()._cdata
    updated = isolated(args)
    updated[0].fill_(7)
    updated[1].fill_(9)
    flags = group.update(updated)
    assert flags == {"same_tensor_objects": True, "storage_preserved_during_update": True, "geometry_restorations": 0}
    assert torch.equal(group.values[0], updated[0]) and torch.equal(group.values[1], updated[1])
    assert not torch.equal(first, updated[0])
    group.values[0].transpose_(0, 1)
    assert group.update(updated)["geometry_restorations"] == 1
    assert group.values[0].stride() == first.stride()


def test_nonleaf_and_gradient_inputs_do_not_copy_graphs_or_share_grad_storage():
    leaf = torch.randn(3, requires_grad=True)
    leaf.grad = torch.ones_like(leaf)
    copied = isolated([leaf, leaf * 2])
    assert copied[0].grad is not leaf.grad and torch.equal(copied[0].grad, leaf.grad)
    assert all(value.is_leaf and value.requires_grad for value in copied)
    group = MutableInputs(copied)
    group.values[0].data = torch.zeros(3)
    assert group.update(copied)["geometry_restorations"] == 1


@pytest.mark.parametrize("level,pid", [(1, pid) for pid in range(1, 101)] + [(2, pid) for pid in range(1, 101)])
def test_reviewed_hidden_recipes_keep_native_functional_consistency(level, pid):
    from _composite_cases import small_composite_case
    from test_operator_oracles import small_case

    source, reference, args = small_case(pid) if level == 1 else small_composite_case(pid, dtype=torch.float32)
    policy = cache_recipes.plan(level, pid, source)
    candidate, r_group, c_group = copy.deepcopy(reference), MutableInputs(args), MutableInputs(args)
    for probe in range(3):
        values = args if probe == 0 else cache_recipes.change(isolated(args), policy, probe=probe, reference=reference)
        r_group.update(values)
        c_group.update(values)
        with torch.no_grad():
            torch.manual_seed(123)
            expected = reference(*r_group.values)
            torch.manual_seed(123)
            actual = candidate(*c_group.values)
        from src.validation import compare_outputs

        assert compare_outputs(expected, actual, 1e-3) is None
        if level == 1 and pid == 98:
            assert all(torch.all(value > 0) and torch.allclose(value.sum(-1), torch.ones(value.shape[:-1])) for value in values)
        if level == 1 and pid in {95, 100}:
            assert torch.equal(values[1], args[1])
        if level == 1 and pid in {13, 14, 15}:
            for value in values:
                assert torch.equal(value, value.t() if pid == 13 else value.triu() if pid == 14 else value.tril())


@pytest.mark.parametrize("tamper", ["missing", "seed", "objects", "order", "input_hash", "source", "memory", "schema", "native_signature"])
def test_receipt_replay_rejects_missing_changed_policy_and_coverage(tmp_path, tamper):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    result = execute(child)
    document, cases = result["metadata"]["input_cache_audit"], result["metadata"]["reliability_checks"]
    row = document["cases"][0]
    if tamper == "missing":
        document["cases"].pop()
    elif tamper == "seed":
        row["probes"][0]["seed"] += 1
    elif tamper == "objects":
        row["probes"][0]["updates"]["C"]["same_tensor_objects"] = False
    elif tamper == "order":
        row["probes"][0]["order"] = "RC"
    elif tamper == "input_hash":
        row["probes"][0]["input_sha256"] = "missing"
    elif tamper == "source":
        child["ref_arch_src"] += "\n"
    elif tamper == "memory":
        row["estimated_storage_bytes"] += 1
    elif tamper == "schema":
        row["output_schema"]["shape"] = [999]
    else:
        cases[0]["input_signature"][0]["shape"] = [999]
    assert audit.verdict(document, child, cases, allow_simulated=True)["status"] == "not_measured"


@pytest.mark.parametrize("failure", [None, "device", "missing", "timeout"])
def test_supervisor_separate_worker_preserves_times_and_requires_post_control(tmp_path, failure):
    request, _ = payload(tmp_path)
    stages = []

    def run(command, timeout, log, *, environment):
        index = command.index("-m")
        source, target = map(Path, command[index + 2:index + 4])
        child = json.loads(source.read_text())
        stage = child["eval_config"].get("diagnostic_stage", "native")
        stages.append(stage)
        result = execute(child)
        if stage == audit.STAGE:
            if failure == "device":
                result["metadata"]["device_fingerprint"]["uuid"] = "other-device"
            elif failure == "missing":
                result["metadata"].pop("input_cache_audit")
        atomic_json(target, result)
        Path(log).write_text("explicit CPU simulation")
        return {"exit_code": 124 if failure == "timeout" and stage == audit.STAGE else 0,
                "timed_out": failure == "timeout" and stage == audit.STAGE, "request_wall_seconds": .01}

    result = tool_runner.evaluate(request, run=run)
    assert stages == ["native", audit.STAGE, "control"]
    assert result["correctness"] is (failure is None)
    assert result["runtime"] == 1 and result["runtime_stats"]["samples_ms"] == [1.]
    assert tool_checks.validate(result, request, allow_simulated=True)["same_object_inputs"]["status"] == (
        "passed" if failure is None else "not_measured")
    assert tool_checks.issue(result, request) == "tool_checks_missing_stale_or_invalid"


def test_cli_source_bindings_memory_and_no_device_imports(tmp_path):
    assert parse_config(["--cache-validation", "--cache-memory-mb", "64"]).cache_memory_mb == 64
    for values in ({"cache_validation": 1}, {"cache_memory_mb": 0}, {"cache_memory_mb": True}):
        with pytest.raises(ValueError):
            RunConfig(**values).validate()
    request, _ = payload(tmp_path, cache_memory_mb=1)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    capture = audit.Collector(child, sync=lambda device: None)
    capture.mutable = lambda args: pytest.fail("Admission must precede isolated storage allocation")
    capture.factory = lambda: []
    x = torch.ones(1024, 1024)
    capture.observe(torch.nn.ReLU(), torch.nn.ReLU(), [x], reference_output=x,
                    seed=42, tolerance=1e-3, device="cpu", set_seed=torch.manual_seed, label="large")
    assert capture.rows[0]["status"] == "not_measured" and "envelope" in capture.rows[0]["reason"]


def test_neuron_worker_uses_same_object_hook_on_explicit_cpu_simulator(tmp_path, monkeypatch):
    from test_device_adapters import SimulatedNeuron

    from kb_workflows import evaluate_task, neuron_worker

    request, _ = payload(tmp_path, device_backend="neuron", hardware="Trainium2", enable_ncu=False)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    capture = audit.Collector(child, sync=lambda device: None)
    load = neuron_worker.load_source

    def small_source(text, name, **kwargs):
        namespace = load(text, name, **kwargs)
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: [torch.randn(3, 17)]
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_source)
    monkeypatch.setattr(audit, "Collector", lambda value: capture)
    monkeypatch.setattr(evaluate_task, "dispatch", lambda value: neuron_worker.evaluate(value, adapter=SimulatedNeuron()))
    result = evaluate_task.cache_control(child)
    assert result["correctness"] and result["metadata"]["simulated_execution"]
    assert all(row["status"] == "passed" for row in result["metadata"]["input_cache_audit"]["cases"])


@pytest.mark.parametrize("pid", range(1, 42))
def test_production_recipes_keep_discrete_scales_probabilities_and_backward_roles(pid):
    from kb_workflows import production
    from kb_workflows.production_specs import task

    source = problem_paths(5)[pid].read_text()
    policy, spec = cache_recipes.plan(5, pid, source), task(pid)
    args = production.make_inputs(pid, profile="small", seed=123)
    mutable = {index for index, _ in policy["production_mutable_roles"]}
    for probe in (1, 2):
        values = cache_recipes.change(isolated(args), policy, probe=probe, reference=None)
        assert description(values) == description(args)
        for index, (old, new) in enumerate(zip(args, values)):
            if index not in mutable and isinstance(old, torch.Tensor):
                assert torch.equal(old, new)
        output = production.execute(pid, *values)
        leaves = output if isinstance(output, tuple) else (output,)
        assert all(torch.isfinite(value).all() for value in leaves)
        if spec["gradient_indices"]:
            assert len(leaves) == len(spec["gradient_indices"])


def test_continuous_state_randomness_and_mutable_output_snapshots(tmp_path):
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.tensor(0))
            self.buffer = torch.empty(3, 17)

        def forward(self, x):
            self.count.add_(1)
            self.buffer.copy_(x + self.count + torch.rand_like(x))
            return self.buffer

    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    capture = audit.Collector(child, sync=lambda device: None)
    reference, candidate = Stateful(), Stateful()
    inputs = [torch.randn(3, 17)]
    capture.factory = lambda: [torch.randn(3, 17)]
    state = torch.get_rng_state().clone()
    capture.observe(reference, candidate, inputs, reference_output=inputs[0], seed=42,
                    tolerance=1e-3, device="cpu", set_seed=torch.manual_seed, label="state")
    assert capture.rows[0]["status"] == "passed"
    assert reference.count.item() == candidate.count.item() == 3
    assert torch.equal(state, torch.get_rng_state())
    assert len({probe["output_sha256"]["C"] for probe in capture.rows[0]["probes"]}) == 3


def test_candidate_import_and_forward_environment_changes_are_restored_and_rejected(tmp_path):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    with audit.audit(child) as capture:
        native.load_custom_model("import os\nos.environ['KB_CACHE_FIXTURE']='changed'\n", {})
    assert "KB_CACHE_FIXTURE" not in os.environ
    assert capture.document()["native_environment_changes"]


def test_freeze_queue_and_private_harbor_preserve_policy_without_sdk_imports(tmp_path):
    from kb_workflows import experiment, final_verification, harbor, study_data
    from kb_workflows.queue_worker import validate_payload

    request, config = payload(tmp_path)
    validate_payload(request)
    altered = copy.deepcopy(request)
    altered["eval_config"]["input_cache_plan"]["recipe"] = "anything"
    with pytest.raises(ValueError, match="cache plan"):
        validate_payload(altered)
    before = final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config)
    config.cache_memory_mb += 1
    assert before != (final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config))
    config.generation = "harbor"
    directory = harbor.write_task(tmp_path / "private", config, 19)
    assert json.loads((directory / "tests/config.json").read_text())["config"]["cache_validation"]
    assert (directory / "tests/repo/kb_workflows/cache_audit.py").is_file()
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--cache-validation","--dry-run","--start","19","--end","19"]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)


def test_broadcast_backing_storage_is_not_confused_with_logical_storage(tmp_path):
    from kb_workflows.measurement import input_signature

    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    capture = audit.Collector(child, sync=lambda device: None)
    capture.factory = lambda: [torch.randn(1, 8).expand(3, 8)]
    args = capture.factory()
    capture.observe(torch.nn.ReLU(), torch.nn.ReLU(), args, reference_output=torch.relu(args[0]),
                    seed=42, tolerance=1e-3, device="cpu", set_seed=torch.manual_seed, label="broadcast")
    cases = [{"label": "broadcast", "seed": 42, "passed": True, "input_signature": input_signature(args)}]
    row = capture.rows[0]
    assert row["input_storage_bytes"] == 32 and row["input_logical_bytes"] == 96
    assert audit.verdict(capture.document(), child, cases, allow_simulated=True)["status"] == "passed"


def test_combined_state_cache_and_throughput_receipts_remain_separate(tmp_path):
    from test_throughput_audit import _execute_parent_request

    from kb_workflows import state_audit, throughput_audit

    request, _ = payload(tmp_path, state_validation=True, throughput_validation=True)

    def run(command, timeout, log, *, environment):
        index = command.index("-m")
        source, target = map(Path, command[index + 2:index + 4])
        child = json.loads(source.read_text())
        stage = child["eval_config"].get("diagnostic_stage", "native")
        if stage in throughput_audit.STAGES.values():
            mode = next(mode for mode, value in throughput_audit.STAGES.items() if value == stage)
            result = _execute_parent_request(child, mode)
        elif stage == "control":
            with state_audit.audit(child) as collector:
                result = _execute_parent_request(child)
            result["metadata"]["state_audit"] = collector.document()
        elif stage == audit.STAGE:
            result = execute(child, trials=1)
        else:
            result = _execute_parent_request(child)
        atomic_json(target, result)
        Path(log).write_text("explicit combined CPU simulation")
        return {"exit_code": 0, "timed_out": False, "request_wall_seconds": .01}

    result = tool_runner.evaluate(request, run=run)
    checks = tool_checks.validate(result, request, allow_simulated=True)
    assert result["correctness"]
    assert all(checks[name]["status"] == "passed" for name in ("mode_state", "same_object_inputs", "throughput_contrast"))


@pytest.mark.parametrize("pid", [13, 90, 93, 95, 98, 100])
def test_native_hidden_hook_uses_source_bound_roles_without_a_public_factory(tmp_path, pid):
    from test_operator_oracles import small_case

    from src.reliability import check_paired_calls

    source, reference, args = small_case(pid)
    request, _ = payload(tmp_path)
    request.update(problem_id=pid, ref_arch_src=source)
    request["eval_config"].update(use_hidden_tests=True, input_cache_plan=cache_recipes.plan(1, pid, source))
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGE)
    capture = audit.Collector(child, sync=lambda device: None)
    token, metadata = audit.ACTIVE.set(capture), {}
    try:
        with torch.no_grad():
            issue = check_paired_calls(reference, copy.deepcopy(reference), args, seed=123, tolerance=1e-3,
                device="cpu", set_seed=torch.manual_seed, metadata=metadata, label="hidden/reviewed")
    finally:
        audit.ACTIVE.reset(token)
    assert issue is None and capture.factory is None
    assert capture.rows[0]["value_policy"] == "reviewed_value_only_recipe"
    assert audit.verdict(capture.document(), child, metadata["reliability_checks"], allow_simulated=True)["status"] == "passed"
