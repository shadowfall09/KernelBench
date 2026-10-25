"""Persistent HTTP verification queue; one active task per configured device."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..agents.resource_budgets import record_device_lease, stamp_device_lease, validate_device_lease
from ..common import ROOT, failure, fingerprint
from .transports import QUEUE_PROTOCOL, device_environment, local_evaluate, resolve_devices


class JobStore:
    def __init__(self, path: Path, capacity=256, *, backend="cuda"):
        if backend not in {"cuda", "neuron"}:
            raise ValueError("Queue backend must be cuda or neuron")
        self.backend = backend
        self.path = path
        self.capacity = capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.expiries = {}
        self.worker_error = None
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, signature TEXT, payload TEXT, status TEXT, result TEXT, created REAL)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS leases (nonce TEXT PRIMARY KEY, job TEXT UNIQUE)")
            connection.execute("CREATE TABLE IF NOT EXISTS worker (id INTEGER PRIMARY KEY CHECK(id=1), backend TEXT NOT NULL)")
            saved = connection.execute("SELECT backend FROM worker WHERE id=1").fetchone()
            legacy = saved is None and connection.execute("SELECT 1 FROM jobs LIMIT 1").fetchone() is not None
            if saved is not None and saved["backend"] != backend or legacy and backend != "cuda":
                raise ValueError("Queue state belongs to another device backend; use a separate state path")
            connection.execute("INSERT OR IGNORE INTO worker VALUES (1,?)", (backend,))
            for row in connection.execute("SELECT * FROM jobs WHERE status IN ('queued','running','cancelling')").fetchall():
                leased = json.loads(row["payload"])["eval_config"].get("device_lease") is not None
                if leased or row["status"] == "cancelling":
                    # A lost parent's usage/cleanup cannot be reconstructed or replayed.
                    connection.execute("UPDATE jobs SET status='failed', result=? WHERE id=?", (
                        json.dumps(failure("Queue restarted with unresolved work; resource usage is unknown")), row["id"]))
                elif row["status"] == "running":
                    connection.execute("UPDATE jobs SET status='queued' WHERE id=?", (row["id"],))

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def submit(self, payload):
        if payload["eval_config"].get("device_backend", "cuda") != self.backend:
            raise ValueError("Queue task backend differs from the configured worker")
        if "worker_python" in payload:
            raise ValueError("Queue interpreter is owned by the worker configuration")
        lease = validate_device_lease(payload)
        signature = fingerprint(payload)
        request_id = payload["request_id"]
        with self.lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self.worker_error:
                raise RuntimeError(self.worker_error)
            existing = connection.execute("SELECT signature FROM jobs WHERE id=?", (request_id,)).fetchone()
            if existing:
                if existing["signature"] != signature:
                    raise ValueError("request_id already exists with different content")
            else:
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')"
                ).fetchone()[0]
                if active >= self.capacity:
                    raise OverflowError("Queue is full; retry later")
                if lease:
                    try:
                        connection.execute("INSERT INTO leases VALUES (?,?)", (lease["nonce"], request_id))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Device lease is already bound to another queue request") from exc
                    self.expiries[request_id] = time.monotonic() + lease["execution_timeout_seconds"]
                connection.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, 'queued', NULL, ?)",
                    (request_id, signature, json.dumps(payload), time.time()),
                )
        return self.get(request_id)

    def get(self, request_id):
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (request_id,)).fetchone()
        if row is None:
            return None
        return {
            "protocol": QUEUE_PROTOCOL,
            "device_backend": self.backend,
            "request_id": row["id"],
            "request_sha256": row["signature"],
            "status": row["status"],
            "created_at": row["created"],
            **({"result": json.loads(row["result"])} if row["result"] else {}),
        }

    def claim(self):
        with self.lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in connection.execute("SELECT id, payload FROM jobs WHERE status='queued' ORDER BY created").fetchall():
                payload = json.loads(row["payload"])
                if payload["eval_config"].get("device_backend", "cuda") != self.backend or "worker_python" in payload:
                    connection.execute("UPDATE jobs SET status='failed', result=? WHERE id=?", (
                        json.dumps(failure("Stored task violates the worker backend/interpreter contract")), row["id"]))
                    self.expiries.pop(row["id"], None)
                    continue
                if time.monotonic() >= self.expiries.get(row["id"], float("inf")):
                    result = self.not_started(payload, "Queue device lease expired before execution")
                    connection.execute("UPDATE jobs SET status='cancelled', result=? WHERE id=?", (json.dumps(result), row["id"]))
                    self.expiries.pop(row["id"], None)
                    continue
                connection.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                return row["id"], payload
        return None

    @staticmethod
    def not_started(payload, reason):
        result = failure(reason, cancelled=True, worker_started=False)
        lease = validate_device_lease(payload)
        return stamp_device_lease(result, lease, 0.0) if lease else result

    def cancel(self, request_id, signature=None):
        with self.lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (request_id,)).fetchone()
            if row is None:
                return None
            if signature is not None and signature != row["signature"]:
                raise ValueError("Cancellation request source does not match the queued task")
            if row["status"] == "queued":
                result = self.not_started(json.loads(row["payload"]), "Queue task cancelled before execution")
                connection.execute("UPDATE jobs SET status='cancelled', result=? WHERE id=?", (json.dumps(result), request_id))
            elif row["status"] == "running":
                connection.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (request_id,))
            self.expiries.pop(request_id, None)
        return self.get(request_id)

    def cancelled(self, request_id):
        if time.monotonic() >= self.expiries.get(request_id, float("inf")):
            self.cancel(request_id)
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (request_id,)).fetchone()
        return row is None or row["status"] in {"cancelling", "cancelled", "failed"}

    def complete(self, request_id, result):
        with self.lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (request_id,)).fetchone()
            if row is None or row["status"] not in {"running", "queued", "cancelling"}:
                return
            status = "failed" if result.get("metadata", {}).get("resource_usage_unknown") else "completed"
            if row["status"] == "cancelling" and not result.get("metadata", {}).get("resource_usage_unknown"):
                result = failure("Queue task cancelled", **{**result.get("metadata", {}), "cancelled": True})
                status = "cancelled"
            connection.execute(
                "UPDATE jobs SET status=?, result=? WHERE id=?",
                (status, json.dumps(result, allow_nan=False), request_id),
            )
            self.expiries.pop(request_id, None)

    def counts(self):
        with self.connect() as connection:
            return dict(connection.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(200, {"status": "unavailable" if store.worker_error else "ok",
                                     "protocol": QUEUE_PROTOCOL, "device_backend": store.backend, "jobs": store.counts()})
            elif path.startswith("/v1/verify/"):
                result = store.get(unquote(path[len("/v1/verify/") :]))
                self.send_json(200 if result else 404, result or {"error": "Unknown request_id"})
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            cancelling = path.startswith("/v1/verify/") and path.endswith("/cancel")
            if path != "/v1/verify" and not cancelling:
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Body must be between 1 byte and 16 MiB")
                payload = json.loads(self.rfile.read(length))
                if cancelling:
                    signature = payload.get("request_sha256") if isinstance(payload, dict) else None
                    if not isinstance(signature, str) or not signature:
                        raise ValueError("Cancellation requires the queued request fingerprint")
                    result = store.cancel(unquote(path[len("/v1/verify/"):-len("/cancel")]), signature)
                    self.send_json(200 if result else 404, result or {"error": "Unknown request_id"})
                    return
                validate_payload(payload)
                self.send_json(200, store.submit(payload))
            except OverflowError as exc:
                self.send_json(429, {"error": str(exc)})
            except RuntimeError as exc:
                self.send_json(503, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})

    return Handler


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    for key in ("request_id", "ref_arch_src", "eval_config", "level", "problem_id"):
        if key not in payload:
            raise ValueError(f"Missing {key}")
    if not isinstance(payload["eval_config"], dict) or not isinstance(payload["ref_arch_src"], str):
        raise ValueError("eval_config must be a dictionary and ref_arch_src a string")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    if payload.get("task", "evaluate") not in {"evaluate", "baseline", "profile", "paired_timing", "qualify_small", "origin_audit", "calibrate_inputs", "calibrate_precision", "counter_audit"}:
        raise ValueError("Unknown task")
    if payload.get("task", "evaluate") in {"evaluate", "profile", "paired_timing", "qualify_small", "origin_audit", "counter_audit"} and not isinstance(payload.get("kernel_src"), str):
        raise ValueError("kernel_src must be provided")
    if payload.get("task") in {"calibrate_inputs", "calibrate_precision"} and "kernel_src" in payload:
        raise ValueError("Reference calibration accepts no candidate")
    if payload["eval_config"].get("timeout", 0) <= 0:
        raise ValueError("Evaluation timeout must be positive")
    validate_device_lease(payload)
    from ..benchmarks.reference_revisions import validate_payload as validate_revision

    validate_revision(payload)
    from ..numerics.input_calibration import validate_inline

    validate_inline(payload)
    from ..numerics.precision_calibration import validate_inline as validate_precision

    validate_precision(payload)
    if payload.get("task") == "counter_audit":
        from ..provenance.work_counters import recipe

        recipe(payload.get("_native_counter", {}).get("recipe"))
    from ..numerics.operator_input_policy import validate_inline as validate_operator_inputs

    validate_operator_inputs(payload)
    from ..evaluation.allocation_audit import validate_inline as validate_allocations

    validate_allocations(payload)
    from ..evaluation.throughput_audit import validate_inline as validate_throughput

    validate_throughput(payload)
    from ..evaluation.cache_audit import validate_inline as validate_cache

    validate_cache(payload)
    from ..evaluation.state_policies import validate_inline as validate_state

    validate_state(payload)


def worker_loop(store, device, stop, evaluator=local_evaluate):
    evaluator = record_device_lease(evaluator)
    while not stop.is_set():
        job = store.claim()
        if not job:
            stop.wait(0.1)
            continue
        request_id, payload = job
        def cancelled():
            if stop.is_set():
                store.cancel(request_id)
            return store.cancelled(request_id)
        try:
            result = evaluator(payload, device, store.path.parent / "logs" / f"{fingerprint(payload)}.log", cancelled=cancelled)
        except Exception as exc:
            result = failure(str(exc), error_name=type(exc).__name__, resource_usage_unknown=True)
        if not isinstance(result, dict) or not isinstance(result.get("metadata", {}), dict):
            result = failure("Malformed evaluator response")
        cancelled()
        store.complete(request_id, result)
        if result.get("metadata", {}).get("resource_usage_unknown"):
            store.worker_error = "Queue worker stopped after uncertain process cleanup"
            stop.set()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, help="Shared RunConfig TOML; selects device backend, IDs and vendor Python")
    parser.add_argument("--device", "--gpu", dest="device", type=int, help="One device ordinal within the current allocation")
    parser.add_argument("--dry-run", action="store_true", help="Preview configuration without importing a device SDK")
    parser.add_argument("--state", type=Path, default=ROOT / "cache" / "verify_queue" / "jobs.sqlite3")
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args(argv)
    if args.capacity <= 0:
        parser.error("--capacity must be positive")
    try:
        from ..config import RunConfig, file_options
        from ..config import parser as run_parser

        options = file_options(args.config, run_parser()) if args.config else {}
        fields = {"device_backend", "device_python", "gpus", "gpu", "gpu_arch", "hardware"}
        # Reuse shared parsing/device validation without loading controller-only
        # calibration, model or study artifacts on the verification machine.
        config = RunConfig(**{key: value for key, value in options.items() if key in fields}, generation="local",
                           enable_ncu=False, stage="evaluate").validate() if args.config else None
        backend = config.device_backend if config else "cuda"
        requested = str(args.device) if args.device is not None else config.gpus if config else "0"
        if not requested.isdecimal():
            raise ValueError("A worker requires one nonnegative configured device or an explicit --device")
        device = resolve_devices(requested, backend=backend)[0]
        interpreter = config.device_python if config and config.device_python else sys.executable
        if args.dry_run:
            print(json.dumps({"protocol": QUEUE_PROTOCOL, "device_backend": backend, "device": device,
                              "python": interpreter, "state": str(args.state), "device_validated": False,
                              "experiments_started": False}))
            return
        # Re-exec before importing the SDK. Preserve the configured venv path,
        # not merely the binary it symlinks to, and retain repository imports.
        if os.path.abspath(interpreter) != os.path.abspath(sys.executable):
            os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
            os.execv(interpreter, [interpreter, "-m", "kb_workflows.runtime.queue_worker", *list(sys.argv[1:] if argv is None else argv)])
            return
        if sys.version_info[:2] != (3, 12):
            raise ValueError("Verification workers require Python 3.12")
        os.environ.update(device_environment(device, backend=backend))
        from ..hardware.devices import adapter

        caps = adapter(backend, "0").caps()
        if caps.get("available") is not True:
            raise ValueError("Configured verification device is unavailable")
        if backend == "neuron" and caps.get("name") != config.hardware:
            raise ValueError("Configured Neuron hardware differs from the observed device")
        store = JobStore(args.state, args.capacity, backend=backend)
    except (ValueError, OSError, ImportError, IndexError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    stop = threading.Event()
    worker = threading.Thread(target=worker_loop, args=(store, device, stop), daemon=True)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store))
    worker.start()
    print(f"KernelBench Pro {backend} worker ({device}): http://{args.host}:{args.port}; state: {args.state}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    main()
