import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import pytest

from kb_workflows import engine
from kb_workflows.agents.generation import build_prompt, generate_run, prepare_job, prompt_source
from kb_workflows.common import ROOT, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.runtime.queue_worker import JobStore, handler_for, worker_loop
from kb_workflows.runtime.transports import http_json, local_evaluate, resolve_devices


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
        baseline_suite=False,
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
    assert parse_config([]).baseline == "baseline_time_suite_fp32_wall"
    assert parse_config(["--precision", "bf16"]).baseline == "baseline_time_suite_bf16_wall"


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

    def evaluator(payload, device, log, *, cancelled=None):
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
        from kb_workflows.agents import resource_budgets as budgets
        from kb_workflows.common import fingerprint
        from kb_workflows.runtime.transports import queue_evaluate

        ledger = budgets.Ledger(6, "device_lease_seconds")
        lease = budgets.device_lease(ledger, 1, 6)
        payload = engine.task_payload(config, 1)
        payload["eval_config"].update(timeout=1, device_lease=lease)
        payload["request_id"] = "kb-" + fingerprint(payload)
        result = queue_evaluate(payload, config)
        assert result["correctness"] and budgets.settle_device_lease(ledger, lease, result)
        assert 0 < ledger.spent < 6
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
    from kb_workflows.evaluation.evaluate_task import evaluate
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

    monkeypatch.setattr("kb_workflows.agents.generation.run_agent", fake_agent)
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

    import torch

    from kb_workflows.evaluation import measurement
    from kb_workflows.evaluation.evaluate_task import baseline
    from src import eval as native

    class Reference(torch.nn.Module):
        def to(self, **kwargs):
            return self

        def forward(self):
            return torch.ones(2)

    model = Reference()
    payload = engine.task_payload(config, 1, task="baseline")
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(native, "load_original_model_and_inputs", return_value=(lambda: model, lambda: [], lambda: []))
        )
        stack.enter_context(patch.object(measurement, "synchronized_times", return_value=[1.0, 2.0]))
        stack.enter_context(
            patch.object(native, "get_timing_stats", return_value={"mean": 1.5, "hardware": "test-gpu"})
        )
        stack.enter_context(patch.object(native, "get_memory_stats", return_value={"peak_bytes": 1024}))
        configure = stack.enter_context(patch("kb_workflows.evaluation.evaluate_task.configure_tf32"))
        stack.enter_context(patch.object(measurement, "synchronize"))
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
    monkeypatch.setattr("kb_workflows.agents.generation.run_agent", lambda *args: {"status": "failed"})
    assert not generate_run(config)
    assert not destination.exists()


def test_queue_lease_state_machine_refuses_replay_and_unknown_restart_usage(tmp_path):
    from kb_workflows.agents import resource_budgets as budgets
    from kb_workflows.common import fingerprint

    store = JobStore(tmp_path / "lease.sqlite3")
    ledger = budgets.Ledger(20, "device_lease_seconds")
    lease = budgets.device_lease(ledger, 5, 20)
    payload = {"request_id": "leased", "ref_arch_src": "source", "kernel_src": "kernel",
               "level": 1, "problem_id": 1, "eval_config": {"timeout": 5, "device_lease": lease}}
    store.submit(payload)
    store.submit(payload)
    with pytest.raises(ValueError, match="already bound"):
        store.submit({**payload, "request_id": "another"})
    with pytest.raises(ValueError, match="does not match"):
        store.cancel("leased", "different-source")
    cancelled = store.cancel("leased", fingerprint(payload))
    assert cancelled["status"] == "cancelled" and store.claim() is None
    assert budgets.settle_device_lease(ledger, lease, cancelled["result"]) and ledger.spent == 0
    store.complete("leased", {"correctness": True})
    assert store.get("leased")["result"]["correctness"] is False

    lease = budgets.device_lease(ledger, 5, 20)
    payload["request_id"], payload["eval_config"]["device_lease"] = "expired", lease
    store.submit(payload)
    store.expiries["expired"] = 0
    assert store.claim() is None
    assert budgets.settle_device_lease(ledger, lease, store.get("expired")["result"])
    lease = budgets.device_lease(ledger, 5, 20)
    payload["request_id"], payload["eval_config"]["device_lease"] = "interrupted", lease
    store.submit(payload)
    assert store.claim()[0] == "interrupted"
    restored = JobStore(store.path)
    assert restored.claim() is None and restored.get("interrupted")["status"] == "failed"
    assert not budgets.settle_device_lease(ledger, lease, restored.get("interrupted")["result"])
    assert ledger.spent == 10
    assert restored.cancel("leased")["status"] == "cancelled"
    lease = budgets.device_lease(ledger, 5, 10)
    payload["request_id"], payload["eval_config"]["device_lease"] = "racing-cancel", lease
    restored.submit(payload)
    assert restored.claim()[0] == "racing-cancel"
    restored.cancel("racing-cancel")
    restored.complete("racing-cancel", budgets.stamp_device_lease({"correctness": True}, lease, 1))
    result = restored.get("racing-cancel")["result"]
    assert not result["correctness"] and budgets.settle_device_lease(ledger, lease, result)
    payload["request_id"] = "failed-cleanup"
    payload["eval_config"].update(timeout=3, device_lease=budgets.device_lease(ledger, 3, 9))
    restored.submit(payload)
    def broken(*args, cancelled):
        raise TimeoutError("Cannot confirm process cleanup")
    stopped = threading.Event()
    worker_loop(restored, "0", stopped, evaluator=broken)
    assert stopped.is_set() and restored.worker_error
    assert restored.get("failed-cleanup")["status"] == "failed"
    with pytest.raises(RuntimeError, match="uncertain"):
        restored.submit({**payload, "request_id": "cannot-start"})


def test_queue_cancellation_and_expiry_stop_cpu_worker_descendants(config, tmp_path, monkeypatch):
    import time
    from pathlib import Path

    from kb_workflows.agents import resource_budgets as budgets
    from kb_workflows.common import fingerprint
    from kb_workflows.runtime import transports

    real_popen = subprocess.Popen
    config.start = config.end = 1
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(1, 1)).write_text("class ModelNew: pass")
    for mode in ("cancel", "expire"):
        marker = tmp_path / f"{mode}.pid"
        child_code = ("import os,signal,time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)")
        parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(60)"
        processes = []

        def spawn(command, *args, **kwargs):
            assert command[1:3] == ["-m", "kb_workflows.evaluation.evaluate_task"]
            process = real_popen([sys.executable, "-c", parent_code], *args, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(transports.subprocess, "Popen", spawn)
        store = JobStore(tmp_path / f"{mode}.sqlite3")
        stop = threading.Event()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
        worker = threading.Thread(target=worker_loop, args=(store, "0", stop))
        service = threading.Thread(target=server.serve_forever)
        worker.start()
        service.start()
        base = f"http://127.0.0.1:{server.server_port}"
        ledger = budgets.Ledger(7, "device_lease_seconds")
        lease = budgets.device_lease(ledger, 2, 7)
        payload = engine.task_payload(config, 1)
        payload["eval_config"].update(timeout=2, device_lease=lease)
        payload["request_id"] = "kb-" + fingerprint(payload)
        route = base + "/v1/verify/" + payload["request_id"]
        try:
            http_json("POST", base + "/v1/verify", payload)
            deadline = time.monotonic() + 4
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "CPU worker did not start before lease expiry"
            if mode == "cancel":
                reply = http_json("POST", route + "/cancel", {"request_sha256": fingerprint(payload)})
                assert reply["status"] in {"cancelling", "cancelled"}
            deadline = time.monotonic() + 8
            while store.get(payload["request_id"])["status"] != "cancelled" and time.monotonic() < deadline:
                time.sleep(0.01)
            reply = http_json("GET", route)
            assert reply["status"] == "cancelled" and not reply["result"]["correctness"]
            assert budgets.settle_device_lease(ledger, lease, reply["result"]) and 0 < ledger.spent < 7
            pid = int(marker.read_text())
            deadline = time.monotonic() + 2
            while Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z" and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not Path(f"/proc/{pid}/stat").exists() or Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
            assert all(process.poll() is not None for process in processes)
        finally:
            stop.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
            service.join(timeout=5)
            for process in processes:
                transports.terminate(process)
            assert not worker.is_alive() and not service.is_alive()
