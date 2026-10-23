"""CPU functional and queued-device simulations; no hardware/model experiments."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from kb_workflows import engine, tool_checks, tool_runner
from kb_workflows import throughput_audit as audit
from kb_workflows.common import ROOT, atomic_json, fingerprint, kernel_name, problem_paths
from kb_workflows.config import RunConfig, parse_config
from src import eval as native

SOURCE = '''import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
'''


class Queue:
    def __init__(self, barrier_ms=0):
        self.now, self.pending, self.barrier, self.calls = 0., 0., barrier_ms / 1000, []

    def clock(self):
        return self.now

    def sync(self, device):
        self.now += self.pending + self.barrier
        self.pending = 0
        self.calls.append("sync")

    def launch(self):
        self.now += .0001
        self.pending += .002
        self.calls.append("launch")


def payload(tmp_path, **options):
    config = RunConfig(generation="local", hardware="RTX_A6000", runs_dir=str(tmp_path), start=19, end=19,
                       num_correct_trials=1, throughput_validation=True, **options).validate()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / kernel_name(1, 19)).write_text(SOURCE)
    return engine.task_payload(config, 19), config


def execute_mode(request, mode, *, barrier_ms=0, source=None, collector_factory=None):
    from test_tool_checks import DEVICE

    child = tool_checks.child_payload(request, "b" * 32, audit.STAGES[mode])
    queue = Queue(barrier_ms)
    capture = (collector_factory or audit.Collector)(child, mode, clock=queue.clock, sync=queue.sync)
    token = audit.ACTIVE.set(capture)
    try:
        Model, _, _ = native.load_original_model_and_inputs(child["ref_arch_src"], {})
        Candidate = native.load_custom_model(source or child["kernel_src"], {})

        class Scheduled(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner, self.queue = inner, queue

            def forward(self, *args):
                self.queue.launch()
                output = self.inner(*args)
                self.training = self.inner.training  # The timing wrapper delegates the candidate's mode.
                return output

        reference, candidate = Scheduled(Model()), Scheduled(Candidate())
        metadata = {"device_fingerprint": copy.deepcopy(DEVICE),
                    "reference_sha256": fingerprint({"source": child["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": child["kernel_src"]})}
        result = native.run_and_check_correctness(reference, candidate, lambda: [torch.randn(3, 17)], metadata,
            num_correct_trials=1, device="cpu", precision=torch.float32)
        metadata["throughput_audit"] = capture.document()
        output = {"compiled": result.compiled, "correctness": result.correctness, "metadata": metadata,
                  "runtime": 1., "runtime_stats": {"samples_ms": [1.], "num_trials": 1}}
        return output, child, queue
    finally:
        audit.ACTIVE.reset(token)


def contrast(tmp_path, *, barrier_ms=0, source=None):
    request, config = payload(tmp_path)
    documents, requests, cases, outputs = {}, {}, {}, {}
    for mode in audit.MODES:
        result, child, queue = execute_mode(request, mode, barrier_ms=barrier_ms, source=source)
        assert result["correctness"]
        outputs[mode] = result
        documents[mode] = result["metadata"]["throughput_audit"]
        requests[mode] = child
        cases[mode] = result["metadata"]["reliability_checks"]
    return audit.verdict(documents, requests, cases, allow_simulated=True), documents, requests, cases, request, config


def test_actual_native_cpu_observer_measures_queued_completion_and_every_output(tmp_path):
    verdict, docs, requests, cases, _, _ = contrast(tmp_path)
    assert verdict["status"] == "passed" and not verdict["values"]["review_required"]
    for name in ("R", "C"):
        assert verdict["values"]["models"][name]["latency_mean_ms"] == pytest.approx(2.1)
        assert verdict["values"]["models"][name]["burst_mean_ms"] == pytest.approx(2.1)
    for mode, document in docs.items():
        row = document["case"]
        assert row["status"] == "passed" and row["warmup_calls"] == 3
        assert len(row["rounds"]) == 4 and row["estimated_storage_bytes"] > 0
        assert [r["order"] for r in row["rounds"]].count("RC") == 2
        for record in row["rounds"]:
            assert record["outputs_verified"] == 8 and len(set(record["input_sha256"])) == 8
            assert record["output_sha256"]["R"] == record["output_sha256"]["C"]
            assert all(len(sample["call_samples_ms"]) == (8 if mode == "latency" else 0) for sample in record["samples"].values())
    assert audit.verdict(docs, requests, cases)["status"] == "not_measured"


def test_legitimate_sync_overhead_is_a_review_flag_without_a_wrong_answer_claim(tmp_path):
    verdict, _, _, _, _, _ = contrast(tmp_path, barrier_ms=1)
    assert verdict["status"] == "passed" and verdict["values"]["review_required"]
    for row in verdict["values"]["models"].values():
        assert row["latency_mean_ms"] == pytest.approx(3.1)
        assert row["burst_mean_ms"] == pytest.approx(2.225)
    assert verdict["values"]["complete_gpu_integrity_proven"] is False


def test_stable_snapshots_preserve_each_call_when_candidate_reuses_one_output_buffer(tmp_path):
    source = '''import torch
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.empty(3,17)
    def forward(self,x):
        self.output.copy_(torch.relu(x))
        return self.output
'''
    verdict, docs, _, _, _, _ = contrast(tmp_path, source=source)
    assert verdict["status"] == "passed"
    assert len(set(docs["burst"]["case"]["rounds"][0]["output_sha256"]["C"])) == 8


@pytest.mark.parametrize("kind", ["wrong_output", "dtype", "shape", "tuple", "environment"])
def test_wrong_timed_outputs_and_settings_are_retained_as_failed_evidence(tmp_path, kind):
    body = {"wrong_output": "return torch.zeros_like(x)", "dtype": "return torch.relu(x).double()",
            "shape": "return torch.relu(x[:, :1])", "tuple": "return (torch.relu(x),)",
            "environment": "import os; os.environ['KB_THROUGHPUT_TEST']='fixture'; return torch.relu(x)"}[kind]
    source = SOURCE.replace("return torch.relu(x)", body)
    request, _ = payload(tmp_path)
    result, _, _ = execute_mode(request, "burst", source=source)
    assert result["metadata"]["throughput_audit"]["case"]["status"] == "failed"
    remaining = os.environ.get("KB_THROUGHPUT_TEST")
    assert remaining is None


@pytest.mark.parametrize("damage", ["repeat", "seed", "input_hash", "output_hash", "sum", "threads", "grad", "training", "memory", "schema", "mode", "nonce", "case", "snapshot"])
def test_receipt_replay_rejects_missing_or_changed_modes_calls_inputs_storage_and_samples(tmp_path, damage):
    _, docs, requests, cases, _, _ = contrast(tmp_path)
    row = docs["burst"]["case"]
    if damage == "repeat":
        row["rounds"].pop()
    elif damage == "seed":
        row["rounds"][0]["input_seeds"][0] += 1
    elif damage == "input_hash":
        row["rounds"][0]["input_sha256"][0] = "c" * 64
    elif damage == "output_hash":
        row["rounds"][0]["output_sha256"]["C"].pop()
    elif damage == "sum":
        row["rounds"][0]["samples"]["C"]["total_ms"] *= 2
    elif damage == "threads":
        row["rounds"][0]["cpu_threads_after_calls"]["C"][0] += 1
    elif damage == "grad":
        row["rounds"][0]["no_grad_after_calls"]["C"][0] = False
    elif damage == "training":
        row["rounds"][0]["training_after_calls"]["C"][0] = not row["training_before"]["C"]
    elif damage == "memory":
        row["estimated_storage_bytes"] -= 1
    elif damage == "schema":
        row["output_schema"]["storage_bytes"] -= 1
    elif damage == "mode":
        docs["burst"]["mode"] = "latency"
    elif damage == "nonce":
        requests["burst"]["eval_config"]["diagnostic_nonce"] = "c" * 32
    elif damage == "case":
        row["seed"] += 1
    else:
        docs["burst"]["output_snapshot_policy"] = "only_the_last_output"
    assert audit.verdict(docs, requests, cases, allow_simulated=True)["status"] == "not_measured"


def test_memory_admission_precedes_any_numeric_warmup_or_batch_allocation(tmp_path):
    request, _ = payload(tmp_path, throughput_memory_mb=1)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGES["burst"])
    queue = Queue()
    capture = audit.Collector(child, "burst", clock=queue.clock, sync=queue.sync)
    capture.factory = lambda: pytest.fail("over-budget inputs were allocated")
    capture.observe(torch.nn.ReLU(), torch.nn.ReLU(), [torch.empty(3, 100000, device="meta")],
                    seed=42, device="cpu", set_seed=torch.manual_seed, tolerance=1e-3, label="large_original_case")
    row = capture.document()["case"]
    assert row["status"] == "not_measured" and row["estimated_storage_bytes"] > 1024**2 and row["warmup_calls"] == 0
    assert queue.calls == []


def test_no_hidden_factory_or_instrumented_launch_blocking_contrast_is_allowed(tmp_path, monkeypatch):
    request, config = payload(tmp_path)
    hidden = engine.task_payload(config, 19, hidden=True)
    assert not audit.validate_inline(hidden) and not tool_checks.enabled(hidden)
    child = tool_checks.child_payload(hidden, "b" * 32, audit.STAGES["burst"])
    with pytest.raises(ValueError, match="public correctness"):
        audit.Collector(child, "burst")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result, _, _ = execute_mode(request, "burst")
    assert result["metadata"]["throughput_audit"]["case"]["status"] == "not_measured"


@pytest.mark.parametrize("latency,burst,flag", [([1.], [1.1], False), ([1.], [.9], False), ([1.], [1.10001], True),
                                             ([1., 1.], [1., .89], True), ([1., 1.], [1.11, .89], True)])
def test_fixed_ten_percent_review_threshold_checks_each_repeat_and_float_boundary(latency, burst, flag):
    assert audit.summarize(latency, burst)["review_required"] is flag


def simulated_executor(*, barrier_ms=0, failure=None):
    calls = []

    def run(command, timeout, log, *, environment):
        index = command.index("-m")
        source, target = map(Path, command[index + 2:index + 4])
        request = json.loads(source.read_text())
        stage = request["eval_config"].get("diagnostic_stage", "native")
        calls.append(stage)
        mode = next((mode for mode, name in audit.STAGES.items() if name == stage), None)
        with patch.dict(os.environ, environment, clear=True):
            if mode:
                result = _execute_parent_request(request, mode, barrier_ms)
            else:
                result = _execute_parent_request(request)
            if failure == "device" and mode:
                result["metadata"]["device_fingerprint"]["uuid"] = "other-device"
            elif failure == "missing" and mode:
                result["metadata"].pop("throughput_audit")
        atomic_json(target, result)
        Path(log).write_text("simulated throughput worker")
        return {"exit_code": 124 if failure == "timeout" and mode else 0,
                "timed_out": failure == "timeout" and bool(mode), "request_wall_seconds": .01}
    return run, calls


def _execute_parent_request(request, mode=None, barrier_ms=0):
    # Execute CPU observation with the supervisor's actual stage/nonce.
    from test_tool_checks import DEVICE

    queue = Queue(barrier_ms)
    capture = audit.Collector(request, mode, clock=queue.clock, sync=queue.sync) if mode else None
    token = audit.ACTIVE.set(capture)
    try:
        class Scheduled(torch.nn.Module):
            def forward(self, x):
                queue.launch()
                return torch.relu(x)
        metadata = {"device_fingerprint": copy.deepcopy(DEVICE),
                    "reference_sha256": fingerprint({"source": request["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": request["kernel_src"]})}
        result = native.run_and_check_correctness(Scheduled(), Scheduled(), lambda: [torch.randn(3, 17)], metadata,
            num_correct_trials=1, device="cpu", precision=torch.float32)
        if capture is not None:
            metadata["throughput_audit"] = capture.document()
        return {"compiled": result.compiled, "correctness": result.correctness, "metadata": metadata,
                "runtime": 1., "runtime_stats": {"samples_ms": [1.], "num_trials": 1}}
    finally:
        audit.ACTIVE.reset(token)


@pytest.mark.parametrize("failure", [None, "device", "missing", "timeout"])
def test_supervisor_owns_two_fresh_modes_retains_native_times_and_replays_process_identity(tmp_path, monkeypatch, failure):
    request, _ = payload(tmp_path)
    run, calls = simulated_executor(failure=failure)
    result = tool_runner.evaluate(request, run=run)
    assert calls == ["native", *audit.STAGES.values(), "control"]
    assert result["runtime"] == 1 and result["runtime_stats"]["samples_ms"] == [1.]
    assert result["correctness"] is (failure is None)
    checks = tool_checks.validate(result, request, allow_simulated=True)
    assert checks["throughput_contrast"]["status"] == ("passed" if failure is None else "not_measured")
    assert tool_checks.issue(result, request) == "tool_checks_missing_stale_or_invalid"


def test_policies_freeze_cli_harbor_and_queue_without_loading_device_sdks(tmp_path):
    from kb_workflows import experiment, final_verification, harbor, study_data
    from kb_workflows.queue_worker import validate_payload

    config = parse_config(["--throughput-validation", "--throughput-calls", "4", "--throughput-repeats", "6"])
    assert config.throughput_validation and config.throughput_calls == 4 and config.throughput_repeats == 6
    for option, value in [("throughput_validation", 1), ("throughput_calls", True), ("throughput_calls", 1),
                          ("throughput_repeats", 3), ("throughput_repeats", 5), ("throughput_memory_mb", 0)]:
        with pytest.raises(ValueError, match="[Tt]hroughput"):
            RunConfig(**{option: value}).validate()
    before = final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config)
    config.throughput_calls += 1
    assert before != (final_verification._options(config), experiment.protocol(config), study_data.evaluation_contract(config))
    request, config = payload(tmp_path)
    validate_payload(request)
    incomplete = copy.deepcopy(request)
    incomplete["eval_config"].pop("throughput_calls")
    with pytest.raises(ValueError, match="resolved throughput"):
        validate_payload(incomplete)
    config.generation = "harbor"
    task = harbor.write_task(tmp_path / "private", config, 19)
    assert json.loads((task / "tests/config.json").read_text())["config"]["throughput_validation"]
    assert (task / "tests/repo/kb_workflows/throughput_audit.py").is_file()
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(["--throughput-validation","--dry-run","--start","19","--end","19"]) == 0
assert not {{"torch","triton","nki","torch_neuronx"}} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("behavior", ["stateful", "random", "structured"])
def test_continuous_state_rng_and_structured_snapshots_are_checked_without_resets(tmp_path, behavior):
    request, _ = payload(tmp_path)
    child = tool_checks.child_payload(request, "b" * 32, audit.STAGES["burst"])
    queue = Queue()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls, self.queue = 0, queue

        def forward(self, x):
            self.calls += 1
            self.queue.launch()
            if behavior == "stateful":
                return x + self.calls
            if behavior == "random":
                return x + torch.rand_like(x)
            return {"value": x.t(), "statistics": (x.sum(), x.long(), x > 0)}

    capture = audit.Collector(child, "burst", clock=queue.clock, sync=queue.sync)
    capture.factory = lambda: [torch.randn(3, 17)]
    reference, candidate = Model(), Model()
    before = torch.get_rng_state().clone()
    capture.observe(reference, candidate, [torch.ones(3, 17)], seed=42, device="cpu", set_seed=torch.manual_seed,
                    tolerance=1e-3, label="controlled_unit_case")
    row = capture.document()["case"]
    assert row["status"] == "passed"
    assert reference.calls == candidate.calls == 3 + 8 * 4
    assert torch.equal(before, torch.get_rng_state())
    assert all(record["output_sha256"]["R"] == record["output_sha256"]["C"] for record in row["rounds"])


@pytest.mark.parametrize("change", ["grad", "training", "late_wrong_output"])
def test_passing_warmup_does_not_exempt_later_calls_or_allow_mode_and_grad_changes(tmp_path, change):
    body = {"grad": "torch.set_grad_enabled(True); return torch.relu(x)",
            "training": "self.eval(); return torch.relu(x)", "late_wrong_output": "return torch.zeros_like(x)"}[change]
    source = '''import torch
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self,x):
        self.calls += 1
        if self.calls > 3:
            BODY
        return torch.relu(x)
'''.replace("BODY", body)
    request, _ = payload(tmp_path)
    before = torch.is_grad_enabled()
    result, _, _ = execute_mode(request, "burst", source=source)
    row = result["metadata"]["throughput_audit"]["case"]
    assert row["warmup_calls"] == 3 and row["status"] == "failed"
    assert torch.is_grad_enabled() == before


def test_neuron_entrypoint_uses_native_factory_and_backend_synchronization_on_explicit_simulator(tmp_path, monkeypatch):
    from test_device_adapters import SimulatedNeuron

    from kb_workflows import evaluate_task, neuron_worker

    config = RunConfig(generation="local", device_backend="neuron", hardware="Trainium2", enable_ncu=False,
        throughput_validation=True, num_correct_trials=1, start=19, end=19, runs_dir=str(tmp_path)).validate()
    config.run_dir.mkdir(parents=True)
    source = problem_paths(1)[19].read_text()
    (config.run_dir / kernel_name(1, 19)).write_text(source + "\nModelNew=Model\n")
    request = tool_checks.child_payload(engine.task_payload(config, 19), "b" * 32, audit.STAGES["burst"])
    queue = Queue()
    capture = audit.Collector(request, "burst", clock=queue.clock, sync=queue.sync)
    load = neuron_worker.load_source

    def small_source(text, name, **kwargs):
        namespace = load(text, name, **kwargs)
        field = "Model" if name == "kernelbench_neuron_reference" else "ModelNew"
        Original = namespace[field]

        forward = Original.forward

        def scheduled(self, *args):
            if not any(isinstance(value, torch.Tensor) and value.device.type == "meta" for value in args):
                queue.launch()
            return forward(self, *args)

        Original.forward = scheduled
        if name == "kernelbench_neuron_reference":
            namespace["get_inputs"] = lambda: [torch.randn(3, 17)]
        return namespace

    monkeypatch.setattr(neuron_worker, "load_source", small_source)
    monkeypatch.setattr(audit, "Collector", lambda *a, **kw: capture)
    monkeypatch.setattr(evaluate_task, "dispatch", lambda value: neuron_worker.evaluate(value, adapter=SimulatedNeuron()))
    result = evaluate_task.throughput_control(request)
    assert result["correctness"] and result["metadata"]["simulated_execution"]
    assert result["metadata"]["throughput_audit"]["case"]["status"] == "passed"
