from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .common import ROOT, failure


def resolve_devices(gpus):
    requested = gpus.split(",")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return requested
    allocated = visible.split(",")
    if not visible or any(int(index) >= len(allocated) for index in requested):
        raise ValueError("Requested GPU index is outside CUDA_VISIBLE_DEVICES")
    return [allocated[int(index)] for index in requested]


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()


def local_evaluate(payload, device="0", log_path=None):
    """Isolate CUDA state and compilation per task; kill children on timeout."""
    timeout = payload["eval_config"]["timeout"]
    with tempfile.TemporaryDirectory(prefix="kb-task-") as temporary:
        directory = Path(temporary)
        source, target = directory / "request.json", directory / "result.json"
        source.write_text(json.dumps(payload))
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(device),
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        actual_log = Path(log_path) if log_path is not None else directory / "evaluator.log"
        with open(actual_log, "w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "kb_workflows.evaluate_task", str(source), str(target)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return failure("Evaluation timed out", timeout_s=timeout)
            finally:
                terminate(process)
        if target.exists():
            return json.loads(target.read_text())
        return failure(
            "Evaluator exited without a result",
            exit_code=process.returncode,
            evaluator_log_tail=actual_log.read_text(errors="replace")[-4000:],
        )


def http_json(method, url, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(0.25 * 2**attempt)


def queue_evaluate(payload, config):
    base = config.queue_url.rstrip("/")
    submitted = http_json("POST", base + config.queue_submit_path, payload, config.queue_http_timeout)
    request_id = submitted["request_id"]
    deadline = time.monotonic() + config.queue_wait_timeout
    path = config.queue_result_path.format(request_id=urllib.parse.quote(request_id, safe=""))
    while time.monotonic() < deadline:
        response = http_json("GET", base + path, timeout=config.queue_http_timeout)
        if response["status"] in {"completed", "failed"}:
            return response.get("result") or failure(response.get("error", "Queue task failed"))
        time.sleep(config.queue_poll_interval)
    return failure("Queue wait timed out", queue_request_id=request_id, timeout_s=config.queue_wait_timeout)


def docker_evaluate(payload, config, device="0", log_path=None):
    name = "kb-eval-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="kb-docker-eval-") as temporary:
        directory = Path(temporary)
        (directory / "request.json").write_text(json.dumps(payload))
        cache = ROOT / "cache" / "workflow"
        cache.mkdir(parents=True, exist_ok=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--gpus",
                f"device={device}",
                "-v",
                f"{directory}:/task",
                "-v",
                f"{cache}:/app/KernelBench/cache/workflow",
                config.docker_image,
                "python",
                "-m",
                "kb_workflows.evaluate_task",
                "/task/request.json",
                "/task/result.json",
            ]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=payload["eval_config"]["timeout"])
            except subprocess.TimeoutExpired:
                return failure("Docker evaluation timed out", timeout_s=payload["eval_config"]["timeout"])
            finally:
                terminate(process)
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        target = directory / "result.json"
        return (
            json.loads(target.read_text())
            if target.exists()
            else failure("Docker evaluator produced no result", exit_code=process.returncode)
        )
