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

from ..agents.resource_budgets import CLEANUP_SECONDS, record_device_lease
from ..common import ROOT, failure, fingerprint

QUEUE_PROTOCOL = "kernelbench_pro_queue_v2"


def resolve_devices(gpus, *, backend="cuda"):
    requested = gpus.split(",")
    if backend == "neuron":
        from ..hardware.neuron_device import core_ids

        visible = os.environ.get("NEURON_RT_VISIBLE_CORES")
        if visible is None:
            return requested
        allocated = core_ids(visible)
        if any(int(index) >= len(allocated) for index in requested):
            raise ValueError("Requested device is outside NEURON_RT_VISIBLE_CORES")
        return [str(allocated[int(index)]) for index in requested]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return requested
    allocated = visible.split(",")
    if not visible or any(int(index) >= len(allocated) for index in requested):
        raise ValueError("Requested GPU index is outside CUDA_VISIBLE_DEVICES")
    return [allocated[int(index)] for index in requested]


def terminate(process):
    # Descendants can outlive the leader, including after it handles SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=CLEANUP_SECONDS - 1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


@record_device_lease
def local_evaluate(payload, device="0", log_path=None, *, cancelled=None):
    """Isolate CUDA state and compilation per task; kill children on timeout."""
    timeout = payload["eval_config"]["timeout"]
    from .isolation import enabled

    if enabled(payload):
        raise ValueError("A local process cannot establish the requested restricted Docker policy")
    if cancelled and cancelled():
        return failure("Evaluation cancelled before execution", cancelled=True)
    with tempfile.TemporaryDirectory(prefix="kb-task-") as temporary:
        directory = Path(temporary)
        source, target = directory / "request.json", directory / "result.json"
        source.write_text(json.dumps(payload))
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(device),
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        if payload["eval_config"].get("device_backend", "cuda") == "neuron":
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment.update(NEURON_RT_VISIBLE_CORES=str(device), NEURON_LOGICAL_NC_CONFIG="1",
                               TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION="1")
        interpreter = payload.get("worker_python") or sys.executable
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        actual_log = Path(log_path) if log_path is not None else directory / "evaluator.log"
        with open(actual_log, "w") as log:
            if cancelled and cancelled():
                return failure("Evaluation cancelled before execution", cancelled=True)
            process = subprocess.Popen(
                [interpreter, "-m", "kb_workflows.evaluation.evaluate_task", str(source), str(target)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if cancelled and cancelled():
                        return failure("Evaluation cancelled", cancelled=True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return failure("Evaluation timed out", timeout_s=timeout)
                    try:
                        process.wait(timeout=min(0.1, remaining) if cancelled else remaining)
                        if cancelled and cancelled():
                            return failure("Evaluation cancelled", cancelled=True)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                terminate(process)
        if target.exists():
            return json.loads(target.read_text())
        return failure(
            "Evaluator exited without a result",
            exit_code=process.returncode,
            evaluator_log_tail=actual_log.read_text(errors="replace")[-4000:],
        )


def http_json(method, url, payload=None, timeout=30, *, deadline=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        remaining = deadline - time.monotonic() if deadline is not None else timeout
        if remaining <= 0:
            raise TimeoutError("Queue HTTP deadline expired")
        try:
            with urllib.request.urlopen(request, timeout=min(timeout, remaining)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        delay = 0.25 * 2**attempt
        time.sleep(min(delay, max(0, deadline - time.monotonic())) if deadline is not None else delay)


def queue_evaluate(payload, config):
    from ..agents.resource_budgets import validate_device_lease

    lease = validate_device_lease(payload)
    base = config.queue_url.rstrip("/")
    request_id = payload["request_id"]
    deadline = time.monotonic() + config.queue_wait_timeout
    path = config.queue_result_path.format(request_id=urllib.parse.quote(request_id, safe=""))
    signature = fingerprint(payload)
    submitted, terminal = False, False
    result = failure("Queue wait timed out", queue_request_id=request_id, timeout_s=config.queue_wait_timeout)

    def checked(response):
        if (not isinstance(response, dict) or response.get("request_id") != request_id
                or (lease and (response.get("protocol") != QUEUE_PROTOCOL or response.get("request_sha256") != signature))):
            raise ValueError("Queue response does not bind the requested task and lease protocol")
        return response

    try:
        if lease:
            health = http_json("GET", base + "/health", timeout=config.queue_http_timeout, deadline=deadline)
            if health.get("protocol") != QUEUE_PROTOCOL or health.get("status") != "ok":
                raise ValueError("Device budgets require a queue with cancellation and bounded-lease support")
        # Even a lost POST response may have admitted work. Cancel by our own ID.
        submitted = True
        response = http_json("POST", base + config.queue_submit_path, payload, config.queue_http_timeout, deadline=deadline)
        if not lease and isinstance(response, dict) and isinstance(response.get("request_id"), str) and response["request_id"]:
            # Legacy compatible services may assign their own job IDs.
            request_id = response["request_id"]
            path = config.queue_result_path.format(request_id=urllib.parse.quote(request_id, safe=""))
            result["metadata"]["queue_request_id"] = request_id
        response = checked(response)
        while True:
            if response["status"] in {"completed", "failed", "cancelled"}:
                terminal = True
                result = response.get("result") or failure(response.get("error", "Queue task failed"))
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(config.queue_poll_interval, remaining))
            response = checked(http_json("GET", base + path, timeout=config.queue_http_timeout, deadline=deadline))
    except Exception as exc:
        result = failure(str(exc), error_name=type(exc).__name__, queue_request_id=request_id)
    finally:
        if submitted and not terminal:
            cleanup_deadline = time.monotonic() + CLEANUP_SECONDS
            try:
                response = checked(http_json("POST", base + path + "/cancel", {"request_sha256": signature},
                                             config.queue_http_timeout, deadline=cleanup_deadline))
                while response["status"] in {"queued", "running", "cancelling"}:
                    time.sleep(min(0.1, max(0, cleanup_deadline - time.monotonic())))
                    response = checked(http_json("GET", base + path, timeout=config.queue_http_timeout, deadline=cleanup_deadline))
                confirmed = (response["status"] in {"completed", "failed", "cancelled"}
                             and not (response.get("result") or {}).get("metadata", {}).get("resource_usage_unknown"))
                metadata = result.setdefault("metadata", {})
                metadata["queue_cancellation_confirmed"] = confirmed
                if confirmed:
                    # Preserve server-parent usage, while a timed-out call stays failed.
                    receipt = (response.get("result") or {}).get("metadata", {}).get("device_lease_receipt")
                    if receipt is not None:
                        metadata["device_lease_receipt"] = receipt
            except Exception as exc:
                result.setdefault("metadata", {}).update(queue_cancellation_confirmed=False, queue_cancel_error=str(exc))
    return result


@record_device_lease
def docker_evaluate(payload, config, device="0", log_path=None):
    from .isolation import enabled

    if enabled(payload):
        from .restricted_transport import evaluate

        return evaluate(payload, config, device, log_path)
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
                "kb_workflows.evaluation.evaluate_task",
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
