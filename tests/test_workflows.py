import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import pytest

from kb_workflows import engine
from kb_workflows.common import ROOT, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.generation import build_prompt, generate_run, prepare_job, prompt_source
from kb_workflows.legacy import convert_arguments
from kb_workflows.queue_worker import JobStore, handler_for, worker_loop
from kb_workflows.transports import http_json, local_evaluate, resolve_devices


@pytest.fixture
def config(tmp_path):
    return RunConfig(
        run_name="test",
        runs_dir=str(tmp_path),
        start=1,
        end=2,
        generation="local",
        evaluation="queue",
        queue_poll_interval=0.01,
    ).validate()


def test_problem_range_and_cli_validation():
    assert list(select_problems(1, 1, 2)) == [1, 2]
    assert list(select_problems(1, 100, 100)) == [100]
    with pytest.raises(ValueError, match="missing IDs"):
        select_problems(1, 100, 101)
    for arguments in (["--parallel", "0"], ["--gpus", "0,0"], ["--run-name", "../other"], ["--level", "4"]):
        with pytest.raises(ValueError):
            parse_config(arguments)
    assert parse_config(["--no-hidden-tests"]).hidden_tests is False
    assert parse_config([]).baseline == "baseline_time_torch_tf32"
    assert parse_config(["--precision", "bf16"]).baseline == "baseline_time_torch_bf16"


def test_legacy_options_use_same_configuration():
    args = convert_arguments(
        [
            "level=2",
            "subset=(3,5)",
            "dataset_src=local",
            "num_parallel=2",
            "gpu_arch=['Ampere']",
            "use_hidden_tests=True",
            "queue_base_url=http://localhost:8000",
        ]
    )
    config = parse_config(args)
    assert (config.level, config.start, config.end, config.parallel, config.gpu_arch) == (2, 3, 5, 2, "Ampere")


def test_gpu_indices_respect_slurm_allocation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert resolve_devices("0,1") == ["3", "7"]
    with pytest.raises(ValueError, match="outside"):
        resolve_devices("2")


def test_input_blind_prompt_keeps_model_but_removes_test_config(config):
    source = select_problems(1, 90, 90)[90].read_text()
    stripped = prompt_source(source, 1, 90)
    assert "class Model" in stripped
    assert "def get_inputs" not in stripped
    assert "def get_init_inputs" not in stripped
    prompt = build_prompt(config, 90)
    assert "problem 90" in prompt and kernel_name(1, 90) in prompt
    assert "python verify.py" in prompt and "hidden input distributions" in prompt


def test_prepared_job_is_independent_of_parent_run(config):
    directory = prepare_job(config, 1, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    assert child.stage == "evaluate" and child.run_name == "agent" and child.start == child.end == 1
    assert not child.resume and not child.build_image
    assert child.run_dir == directory / "runs" / "agent"
    assert (directory / "verify.py").exists()


def test_queue_persistence_idempotency_and_restart(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    payload = {
        "request_id": "one",
        "ref_arch_src": "source",
        "kernel_src": "kernel",
        "eval_config": {"timeout": 10},
        "level": 1,
        "problem_id": 1,
    }
    store.submit(payload)
    store.submit(payload)
    assert store.counts() == {"queued": 1}
    assert store.claim()[0] == "one"
    restored = JobStore(store.path)
    assert restored.get("one")["status"] == "queued"
    with pytest.raises(ValueError, match="different content"):
        restored.submit({**payload, "kernel_src": "changed"})
    restored.complete("one", {"correctness": False})
    assert JobStore(store.path).get("one")["result"] == {"correctness": False}


def test_http_queue_full_pipeline_and_resume(config, tmp_path):
    store = JobStore(tmp_path / "queue.sqlite3")
    stop = threading.Event()
    calls = []

    def evaluator(payload, device, log):
        calls.append(payload)
        # This transport test simulates a standard pass followed by a hidden failure.
        hidden = payload["eval_config"]["use_hidden_tests"]
        return {
            "compiled": True,
            "correctness": not hidden,
            "runtime": -1 if hidden else 2.0,
            "runtime_stats": {"mean": 2.0},
            "peak_memory": -1 if hidden else 1024,
            "memory_stats": {"peak_bytes": 1024},
            "metadata": {},
        }

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    worker = threading.Thread(target=worker_loop, args=(store, "0", stop, evaluator))
    service = threading.Thread(target=server.serve_forever)
    worker.start()
    service.start()
    config.queue_url = f"http://127.0.0.1:{server.server_port}"
    config.run_dir.mkdir(parents=True)
    for pid in (1, 2):
        (config.run_dir / kernel_name(1, pid)).write_text("class ModelNew: pass")
    try:
        assert http_json("GET", config.queue_url + "/health")["status"] == "ok"
        engine.evaluate_run(config)
        assert len(calls) == 4
        results = read_json(config.run_dir / "eval_results.json")
        hidden = read_json(config.run_dir / "eval_results_hidden.json")
        assert set(results) == {"1_0", "2_0"}
        assert results["1_0"]["peak_memory"] == 1024
        assert hidden["1_0"]["correctness"] is False
        config.resume = True
        engine.evaluate_run(config)
        assert len(calls) == 4
        (config.run_dir / kernel_name(1, 1)).write_text("class ModelNew: changed = True")
        engine.evaluate_run(config)
        assert len(calls) == 6  # Only changed P1, in both phases.
        with pytest.raises(RuntimeError, match="HTTP 400"):
            http_json("POST", config.queue_url + "/v1/verify", {"invalid": True})
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        service.join(timeout=5)


def test_missing_kernel_is_recorded_as_failure(config):
    config.evaluation = "local"
    engine.evaluate_run(config)
    for filename in ("eval_results.json", "eval_results_hidden.json"):
        results = read_json(config.run_dir / filename)
        assert set(results) == {"1_0", "2_0"}
        assert all(not entry["correctness"] for entry in results.values())


def test_native_evaluator_receives_hidden_path_and_retains_memory(config, tmp_path):
    from kb_workflows.evaluate_task import evaluate
    from src import eval as native

    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(1, 1)).write_text("class ModelNew: pass")
    payload = engine.task_payload(config, 1, hidden=True)
    payload["build_root"] = str(tmp_path / "build")
    result = native.KernelExecResult(compiled=True, correctness=True, peak_memory=123, memory_stats={"peak_bytes": 123})
    with patch.object(native, "eval_kernel_against_ref", autospec=True, return_value=result) as function:
        actual = evaluate(payload)
    assert function.call_args.kwargs["hidden_test_path"] == str(ROOT / "hidden_tests/level1/1_hidden.py")
    assert function.call_args.kwargs["measure_performance"] is False
    assert actual["peak_memory"] == 123 and actual["memory_stats"]["peak_bytes"] == 123
    changed = {**payload, "ref_arch_src": "different reference"}
    with pytest.raises(ValueError, match="reference sources differ"):
        evaluate(changed)
    payload["eval_config"]["hidden_source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hidden test sources differ"):
        evaluate(payload)


def test_timeout_returns_a_failure_and_terminates_task():
    payload = {"eval_config": {"timeout": 0.01}}
    result = local_evaluate(payload)
    assert not result["correctness"] and result["metadata"]["error"] == "Evaluation timed out"


def test_generation_resume_and_failed_overwrite(config, monkeypatch):
    config.end = 1
    config.generation = "local"
    count = []

    def fake_agent(directory, config, pid, device=None):
        count.append(pid)
        output = directory / "runs/agent" / kernel_name(1, pid)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("class ModelNew: pass")
        return {"status": "success"}

    monkeypatch.setattr("kb_workflows.generation.run_agent", fake_agent)
    assert generate_run(config)
    config.resume = True
    assert generate_run(config)
    assert count == [1]
    config.model = "different-model"
    assert generate_run(config)
    assert count == [1, 1]


def test_hidden_gating_and_invalid_runtime_metrics():
    from scripts.benchmark_eval_analysis import build_baseline_lookup, compute_sample_metrics, merge_hidden_eval

    standard = {"1_0": {"compiled": True, "correctness": True, "runtime": 2, "peak_memory": 1024}}
    assert merge_hidden_eval(standard, {})["1_0"]["correctness"] is False
    merged = merge_hidden_eval(standard, {"1_0": {"correctness": True, "runtime": -1}})
    assert merged["1_0"]["runtime"] == 2 and merged["1_0"]["peak_memory"] == 1024
    assert build_baseline_lookup(
        {"level1": {"1_a.py": {"mean": 2}, "2_b.py": None, "3_c.py": "invalid", "4_d.py": {"mean": -1}}}, 1
    ) == {1: 2}
    entry = {1: {"compiled": True, "correctness": True, "runtime": -1}}
    metrics = compute_sample_metrics(entry, [1], {1: 1.0}, [1.0])
    assert metrics["correctness_rate"] == 1 and metrics["gmsr"] == 0 and metrics["fast_p_1.0"] == 0


def test_dry_run_does_not_create_files_or_load_cloud_sdk(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_batch.py"),
            "--generation",
            "modal",
            "--evaluation",
            "modal",
            "--level",
            "1",
            "--end",
            "1",
            "--runs-dir",
            str(tmp_path),
            "--run-name",
            "dry",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["problems"] == [1]
    assert not (tmp_path / "dry").exists()


def test_baseline_uses_tf32_and_records_compatible_memory_fields(config, monkeypatch):
    from contextlib import ExitStack
    from unittest.mock import MagicMock

    from kb_workflows.evaluate_task import baseline
    from src import eval as native

    model = MagicMock()
    model.to.return_value = model
    payload = engine.task_payload(config, 1, task="baseline")
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(native, "load_original_model_and_inputs", return_value=(lambda: model, lambda: [], lambda: []))
        )
        stack.enter_context(patch.object(native, "time_execution_with_cuda_event", return_value=[1.0, 2.0]))
        stack.enter_context(
            patch.object(native, "get_timing_stats", return_value={"mean": 1.5, "hardware": "test-gpu"})
        )
        stack.enter_context(patch.object(native, "get_memory_stats", return_value={"peak_bytes": 1024}))
        configure = stack.enter_context(patch("kb_workflows.evaluate_task.configure_tf32"))
        result = baseline(payload)
    assert configure.call_args.args[1] is True
    assert result["tf32"] is True and result["peak_memory"] == 1024
    assert result["memory_stats"]["peak_bytes"] == 1024 and result["time_unit"] == "ms"


def test_subset_analysis_hidden_gating_and_denominator(config, tmp_path, monkeypatch):
    import scripts.benchmark_eval_analysis as analysis
    from kb_workflows.common import atomic_json

    config.run_dir.mkdir(parents=True)
    monkeypatch.setattr(analysis, "__file__", str(tmp_path / "scripts" / "benchmark_eval_analysis.py"))
    atomic_json(
        tmp_path / "results/timing/test-gpu/baseline_time_torch_tf32.json",
        {"level1": {"1_a.py": {"mean": 4.0, "peak_memory": 2048}, "2_b.py": {"mean": 3.0}}},
    )
    atomic_json(
        config.run_dir / "eval_results.json",
        {
            "1_0": {"compiled": True, "correctness": True, "runtime": 2.0, "peak_memory": 1024},
            "2_0": {"compiled": True, "correctness": True, "runtime": 1.0},
        },
    )
    atomic_json(config.run_dir / "eval_results_hidden.json", {"1_0": {"correctness": False}})
    analysis.analyze_multi_sample_eval(
        config.run_name,
        "test-gpu",
        "baseline_time_torch_tf32",
        1,
        use_hidden_eval=True,
        problem_ids=[1],
        runs_dir=config.runs_dir,
    )
    report = read_json(config.run_dir / "analysis_summary.json")
    assert report["problem_ids"] == [1]
    assert report["per_sample"]["0"]["correctness_rate"] == 0
    assert report["details"][0][4] == 2.0  # Preserve standard runtime after hidden failure.
    assert report["details"][0][5] is None


def test_pass_at_k_includes_missing_samples_and_hidden_failures(config):
    from kb_workflows.common import atomic_json

    config.end = 1
    config.num_samples = 5
    atomic_json(
        config.run_dir / "eval_results.json", {f"1_{sid}": {"compiled": True, "correctness": True} for sid in range(4)}
    )
    atomic_json(config.run_dir / "eval_results_hidden.json", {f"1_{sid}": {"correctness": True} for sid in range(2)})
    engine.write_pass_at_k(config)
    result = read_json(config.run_dir / "pass_at_k_results.json")
    assert result["problems"]["1"]["correct_samples"] == 2
    assert result["averages"]["avg_pass@1"] == pytest.approx(0.4)
    assert result["averages"]["avg_pass@5"] == 1


def test_failed_overwrite_cannot_reuse_previous_kernel(config, monkeypatch):
    config.end = 1
    old = prepare_job(config, 1, 0) / "runs/agent" / kernel_name(1, 1)
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("stale kernel")
    destination = config.run_dir / kernel_name(1, 1)
    destination.write_text("old final kernel")
    monkeypatch.setattr("kb_workflows.generation.run_agent", lambda *args: {"status": "failed"})
    assert not generate_run(config)
    assert not destination.exists()
