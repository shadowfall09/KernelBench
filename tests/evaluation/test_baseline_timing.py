"""CPU and simulated-CUDA regressions for baseline and wall timing workflows."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from kb_workflows import engine
from kb_workflows.common import atomic_json, read_json
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation import measurement
from kb_workflows.evaluation.baseline_variants import VARIANTS, cuda_graph_model, select_baseline
from src.validation import compare_outputs


def test_wall_timer_waits_for_completion_and_refreshes_input(monkeypatch):
    clock = [0.0]
    pending = [0.0]
    seen = []

    def sync(device):
        clock[0] += pending[0]
        pending[0] = 0

    def forward(x):
        seen.append(x)  # Retain objects, so Python cannot reuse their IDs.
        clock[0] += 0.001
        pending[0] = 0.004  # Work on another stream completes at synchronization.
        return x + 1

    monkeypatch.setattr(measurement, "CLOCK", lambda: clock[0])
    monkeypatch.setattr(measurement, "_SYNC", sync)
    original = torch.ones(4)
    receipt = {}
    times = measurement.synchronized_times(forward, [original], "cuda:0", 1, 3, receipt)
    assert times == pytest.approx([5, 5, 5])
    assert len({id(x) for x in seen}) == 4
    assert all(x.data_ptr() != original.data_ptr() for x in seen)
    assert receipt["output_consumed"] and receipt["fresh_inputs"]


def test_output_consumption_preserves_noncontiguous_storage():
    x = torch.arange(12.0).reshape(3, 4).T
    assert not x.is_contiguous()
    assert measurement.consume_output(x).item() == x[0, 0].item()
    assert len(measurement.consume_output({"x": x, "y": [x]})) == 2


def test_lazy_tensor_subclass_is_rejected():
    class Lazy(torch.Tensor):
        pass

    ordinary = torch.ones(2)
    lazy = ordinary.as_subclass(Lazy)
    assert "materialized" in compare_outputs(ordinary, lazy, 1e-3)
    with pytest.raises(TypeError, match="materialized"):
        measurement.consume_output(lazy)


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_timing_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        measurement.timing_stats(samples)


def test_raw_samples_and_uncertainty_are_retained():
    result = measurement.timing_stats([1, 2, 3])
    assert result["mean"] == 2
    assert result["samples_ms"] == [1, 2, 3]
    assert result["mean_ci95_ms"][0] < 2 < result["mean_ci95_ms"][1]


def test_baseline_winner_excludes_incorrect_and_invalid_times():
    results = {
        "bad": {"mean": 0.01, "baseline_correctness": False},
        "nan": {"mean": float("nan"), "baseline_correctness": True},
        "eager": {"mean": 3, "baseline_correctness": True},
        "compile": {"mean": 1, "baseline_correctness": True, "metadata": {"receipt": 7}},
    }
    selected = select_baseline(results)
    assert selected["winning_baseline"] == "compile"
    selected["metadata"]["new"] = True
    assert "new" not in results["compile"]["metadata"]
    assert select_baseline({"bad": results["bad"]})["mean"] == -1


def test_incorrect_compiled_baseline_is_rejected_before_timing(tmp_path, monkeypatch):
    from kb_workflows.evaluation.evaluate_task import baseline
    from src import eval as native

    class Reference(torch.nn.Module):
        def to(self, **kwargs):
            return self

        def forward(self):
            return torch.ones(2)

    config = RunConfig(run_name="bad-baseline", runs_dir=str(tmp_path), start=1, end=1, baseline_suite=True).validate()
    payload = engine.task_payload(config, 1, task="baseline")
    payload["eval_config"]["baseline_variant"] = "compile_default"
    monkeypatch.setattr(native, "load_original_model_and_inputs", lambda *args: (Reference, lambda: [], lambda: []))
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: lambda: torch.zeros(2))
    monkeypatch.setattr(measurement, "synchronize", lambda device: None)
    monkeypatch.setattr("kb_workflows.evaluation.evaluate_task.configure_tf32", lambda *args: None)
    monkeypatch.setattr(measurement, "synchronized_times", lambda *args, **kwargs: pytest.fail("Incorrect variant was timed"))
    result = baseline(payload)
    assert result["correctness"] is False
    assert "Baseline correctness failed" in result["metadata"]["error"]


def test_graph_replay_copies_inputs_and_keeps_previous_output(monkeypatch):
    class Stream:
        def wait_stream(self, stream):
            pass

    captured = {}

    class Graph:
        def replay(self):
            captured["output"].copy_(captured["input"].square())

    def model(x):
        output = x.square()
        captured.update(input=x, output=output)
        return output

    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    replay = cuda_graph_model(model, [torch.ones(3)], "cpu")
    previous = replay(torch.full((3,), 2.0))
    current = replay(torch.full((3,), 3.0))
    assert torch.equal(previous, torch.full((3,), 4.0))
    assert torch.equal(current, torch.full((3,), 9.0))


def test_suite_uses_distinct_tasks_and_resumes_only_complete_receipts(tmp_path, monkeypatch):
    config = RunConfig(run_name="suite", runs_dir=str(tmp_path), hardware="test", gpu_arch="Ampere", start=1, end=1,
                       baseline_suite=True, resume=True).validate()
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    seen = []

    def evaluate(config, tasks):
        seen.extend(tasks)
        for key, payload in tasks:
            variant = payload["eval_config"]["baseline_variant"]
            yield key, {"mean": 1 if variant == "eager" else 2, "baseline_correctness": True,
                        "precision_gate": {"status": "passed"},
                        "metadata": {"evaluation_config": payload["eval_config"]}}

    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    path = engine.record_baseline(config)
    assert len(seen) == len(VARIANTS)
    assert len({payload["request_id"] for _, payload in seen}) == len(VARIANTS)
    results = read_json(path)
    receipt = next(iter(results["level1"].values()))
    assert receipt["suite_complete"] and receipt["winning_baseline"] == "eager"
    seen.clear()
    engine.record_baseline(config)
    assert not seen
    receipt["suite_complete"] = False
    atomic_json(path, results)
    engine.record_baseline(config)
    assert len(seen) == len(VARIANTS)


def test_backend_precision_is_restored_between_timing_calls():
    settings = []
    def forward(x):
        settings.append(torch.backends.cudnn.benchmark)
        torch.backends.cudnn.benchmark = not settings[-1]
        return x + 1

    before = torch.backends.cudnn.benchmark
    measurement.synchronized_times(forward, [torch.ones(2)], "cpu", 0, 3)
    assert settings == [before] * 3 and torch.backends.cudnn.benchmark == before


def test_varied_timing_closes_content_cache_loophole():
    class Cached:
        def __init__(self):
            self.computations = 0

        def __call__(self, x):
            if not hasattr(self, "input") or not torch.equal(x, self.input):
                self.input, self.output = x.clone(), x.square()
                self.computations += 1
            return self.output

    model = Cached()
    receipt = {}

    def verifier(args, output):
        assert compare_outputs(args[0].square(), output, 1e-5) is None

    measurement.synchronized_times(model, [], "cpu", 1, 4, receipt,
                                    input_factory=lambda trial: [torch.full((4,), float(trial + 1))], verifier=verifier)
    assert model.computations == 5 and receipt["varied_input_contents"] and receipt["timed_outputs_verified"]


def test_verification_and_input_preparation_are_outside_timing(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(measurement, "CLOCK", lambda: clock[0])

    def inputs(trial):
        clock[0] += 1
        return [torch.tensor([float(trial + 2)])]

    def forward(x):
        clock[0] += 0.002
        return x.square()

    def verifier(args, output):
        clock[0] += 3
        assert torch.equal(args[0].square(), output)

    times = measurement.synchronized_times(forward, [], "cpu", 0, 3,
                                           input_factory=inputs, verifier=verifier)
    assert times == pytest.approx([2, 2, 2])


@pytest.mark.parametrize("cached", [False, True])
def test_native_timing_validates_new_inputs_and_records_cold_cost(monkeypatch, cached):
    from src import eval as native

    source = '''import torch
from torch import nn
class Model(nn.Module):
    def forward(self, x): return x.square()
def get_init_inputs(): return []
def get_inputs(): return [torch.rand(4) + .2]
'''
    forward = "return x * x" if not cached else '''if not hasattr(self, "answer"): self.answer = x * x
        return self.answer'''
    candidate = f'''import torch
from torch import nn
class ModelNew(nn.Module):
    def forward(self, x):
        {forward}
'''
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_device"), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.get_device_name", return_value="simulated"), \
         patch.object(native, "graceful_eval_cleanup"), \
         patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=16):
        result = native.eval_kernel_against_ref(source, candidate, device=torch.device("cpu"),
                                                measure_performance=True, num_perf_trials=3)
    assert result.correctness is (not cached)
    if cached:
        assert result.runtime == -1
        assert "Timed output did not match varied input" in str(result.metadata["error_during_performance"])
    else:
        receipt = result.metadata["timing_receipt"]
        assert receipt["timed_outputs_verified"] and receipt["varied_input_contents"]
        assert len(receipt["input_seeds"]) == 6
        assert result.metadata["cold_start_ms"] >= result.metadata["candidate_setup_ms"]
        assert len(result.runtime_stats["samples_ms"]) == 3
        assert result.metadata["memory_receipt"]["output_verified"]
        assert result.peak_memory == result.memory_stats["raw_peak_bytes"] == 16


def test_declared_call_budget_changes_amortized_baseline():
    variants = {"eager": {"mean": 2, "cold_start_ms": 2, "baseline_correctness": True},
                "compile": {"mean": 1, "cold_start_ms": 1000, "baseline_correctness": True}}
    assert select_baseline(variants, 1)["cold_winning_baseline"] == "eager"
    assert select_baseline(variants, 10000)["cold_winning_baseline"] == "compile"
    assert measurement.amortized(1, 1000, 1000) == pytest.approx(1.999)


def test_memory_probe_isolates_inputs_verification_peak_and_post_call_rng(monkeypatch):
    import random

    import numpy as np

    from src import eval as native

    peak = [0]
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: peak.__setitem__(0, 0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peak[0])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "CPU memory-boundary fixture")
    for wrong, rejected in ((False, False), (True, False), (False, True)):
        torch.manual_seed(17)
        random.seed(57)
        np.random.seed(87)
        generator = torch.Generator().manual_seed(17)
        torch.rand(4, generator=generator)
        next_torch = torch.rand(4, generator=generator)
        python = random.Random(57)
        python.random()
        next_python = python.random()
        numpy = np.random.RandomState(87)
        numpy.random_sample()
        next_numpy = numpy.random_sample()
        args, receipt = torch.zeros(4), {}

        def candidate(x):
            assert not torch.is_grad_enabled()
            x.add_(1)
            random.random()
            np.random.random_sample()
            peak[0] = 123
            return 2 * x + torch.rand_like(x) + int(wrong)

        def verify(fresh, actual):
            assert not torch.is_grad_enabled() and torch.equal(fresh[0], torch.zeros(4))
            expected = 2 * (fresh[0] + 1) + torch.rand_like(fresh[0])
            random.random()
            np.random.random_sample()
            peak[0] = 999  # The trusted verifier allocates after the candidate peak was read.
            if rejected or not torch.equal(expected, actual):
                raise ValueError("memory output/oracle rejected")

        if wrong or rejected:
            with pytest.raises(RuntimeError, match="Memory probe output validation failed"):
                native.get_memory_stats(candidate, args, device="cpu", verifier=verify, receipt=receipt, input_seed=17)
            assert receipt.get("output_verified") is not True
        else:
            result = native.get_memory_stats(candidate, args, device="cpu", verifier=verify, receipt=receipt, input_seed=17)
            assert result["peak_bytes"] == receipt["peak_bytes"] == 123
            assert receipt["output_verified"] and receipt["verification_outside_peak"]
        assert torch.equal(args, torch.ones(4))
        assert torch.equal(torch.rand(4), next_torch)
        assert random.random() == next_python and np.random.random_sample() == next_numpy
    # A simulated CUDA allocator includes a live 32-byte verification snapshot.
    # Its pre-reset allocation still contributes to max_memory_allocated.
    allocated = iter((64, 96))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: next(allocated))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 160)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: torch.set_rng_state(state))
    monkeypatch.setattr(measurement, "_SYNC", lambda device: None)
    receipt = {}
    def verify(values, output):
        assert torch.equal(values[0] + 1, output)
    result = native.get_memory_stats(lambda x: x + 1, torch.zeros(4), device="cuda:0", verifier=verify, receipt=receipt)
    assert result["peak_bytes"] == 128 and result["verification_input_bytes"] == 32
    assert result["raw_peak_bytes"] == receipt["raw_peak_bytes"] == 160


def test_wall_baseline_cli_defaults_and_explicit_single_baseline():
    assert parse_config([]).baseline == "baseline_time_suite_fp32_wall"
    assert parse_config(["--no-baseline-suite"]).baseline == "baseline_time_torch_tf32_wall"
    assert parse_config(["--cold-calls", "1"]).cold_calls == 1
    for args in (["--cold-calls", "0"], ["--compile-backend", "inductor"]):
        with pytest.raises(ValueError):
            parse_config(args)
