"""Persistent HTTP verification queue; one active task per worker GPU."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import ROOT, fingerprint
from .transports import local_evaluate


class JobStore:
    def __init__(self, path: Path, capacity=256):
        self.path = path
        self.capacity = capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, signature TEXT, payload TEXT, status TEXT, result TEXT, created REAL)"
            )
            connection.execute("UPDATE jobs SET status='queued' WHERE status='running'")

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def submit(self, payload):
        signature = fingerprint(payload)
        request_id = payload["request_id"]
        with self.lock, self.connect() as connection:
            existing = connection.execute("SELECT signature FROM jobs WHERE id=?", (request_id,)).fetchone()
            if existing:
                if existing["signature"] != signature:
                    raise ValueError("request_id already exists with different content")
            else:
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                if active >= self.capacity:
                    raise OverflowError("Queue is full; retry later")
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
            "request_id": row["id"],
            "status": row["status"],
            "created_at": row["created"],
            **({"result": json.loads(row["result"])} if row["result"] else {}),
        }

    def claim(self):
        with self.lock, self.connect() as connection:
            row = connection.execute(
                "SELECT id, payload FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if row:
                connection.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                return row["id"], json.loads(row["payload"])
        return None

    def complete(self, request_id, result):
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='completed', result=? WHERE id=?",
                (json.dumps(result, allow_nan=False), request_id),
            )

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
                self.send_json(200, {"status": "ok", "jobs": store.counts()})
            elif path.startswith("/v1/verify/"):
                result = store.get(unquote(path[len("/v1/verify/") :]))
                self.send_json(200 if result else 404, result or {"error": "Unknown request_id"})
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/verify":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Body must be between 1 byte and 16 MiB")
                payload = json.loads(self.rfile.read(length))
                validate_payload(payload)
                self.send_json(200, store.submit(payload))
            except OverflowError as exc:
                self.send_json(429, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})

    return Handler


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    for key in ("request_id", "ref_arch_src", "eval_config", "level", "problem_id"):
        if key not in payload:
            raise ValueError(f"Missing {key}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    if payload.get("task", "evaluate") not in {"evaluate", "baseline"}:
        raise ValueError("Unknown task")
    if payload.get("task", "evaluate") == "evaluate" and not isinstance(payload.get("kernel_src"), str):
        raise ValueError("kernel_src must be provided")
    if payload["eval_config"].get("timeout", 0) <= 0:
        raise ValueError("Evaluation timeout must be positive")


def worker_loop(store, device, stop, evaluator=local_evaluate):
    from .common import failure

    while not stop.is_set():
        job = store.claim()
        if not job:
            stop.wait(0.1)
            continue
        request_id, payload = job
        try:
            result = evaluator(payload, device, store.path.parent / "logs" / f"{fingerprint(payload)}.log")
        except Exception as exc:
            result = failure(str(exc), error_name=type(exc).__name__)
        store.complete(request_id, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", default="0", help="CUDA device ID, relative to the worker's CUDA_VISIBLE_DEVICES")
    parser.add_argument("--state", type=Path, default=ROOT / "cache" / "verify_queue" / "jobs.sqlite3")
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args(argv)
    if args.capacity <= 0:
        parser.error("--capacity must be positive")
    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required; start this worker inside a GPU allocation")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if os.environ.get("CUDA_VISIBLE_DEVICES") else None
    device = devices[int(args.gpu)] if devices else args.gpu
    store = JobStore(args.state, args.capacity)
    stop = threading.Event()
    worker = threading.Thread(target=worker_loop, args=(store, device, stop), daemon=True)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store))
    worker.start()
    print(f"Verified worker: http://{args.host}:{args.port}; state: {args.state}", flush=True)
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
