"""Parent-owned diagnostic processes; instrumented times never enter scores."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, read_json
from .tool_checks import ERROR_EXIT, MAX_LOG_BYTES, PROTOCOL, child_payload, coverage, derive, finish, select_tools


def process(command, timeout, log, *, environment=None):
    """Keep children in the evaluator's process group and bound per-stage cleanup."""
    import psutil

    started = time.monotonic()
    with Path(log).open("w") as output:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out, code = True, 124
        finally:
            if child.poll() is None:
                try:
                    descendants = psutil.Process(child.pid).children(recursive=True)
                except psutil.Error:
                    descendants = []
                for target in reversed(descendants):
                    try:
                        target.terminate()
                    except psutil.Error:
                        pass
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                _, alive = psutil.wait_procs(descendants, timeout=1)
                for target in alive:
                    try:
                        target.kill()
                    except psutil.Error:
                        pass
    return {"exit_code": code, "timed_out": timed_out, "request_wall_seconds": time.monotonic() - started}


def binary_identity(command, *, run=process, directory, timeout=5):
    if timeout <= 0:
        return {"status": "not_measured", "reason": "No request budget remains for sanitizer version discovery"}
    path = shutil.which(command)
    if path is None:
        return {"status": "not_measured", "reason": "Configured compute-sanitizer executable is unavailable"}
    target = Path(path).resolve()
    if not target.is_file():
        return {"status": "not_measured", "reason": "Sanitizer executable is not a regular file"}
    log = Path(directory) / "sanitizer-version.txt"
    result = run([str(target), "--version"], min(5, timeout), log, environment=environment())
    version = log.read_text(errors="replace") if log.is_file() else ""
    if result.get("exit_code") != 0 or result.get("timed_out") or not version.strip():
        return {"status": "not_measured", "reason": "Sanitizer version query did not complete"}
    return {"status": "measured", "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "version": version, "version_sha256": hashlib.sha256(version.encode()).hexdigest()}


def environment(*, sanitized=False, allocation=False):
    values = {key: value for key, value in os.environ.items() if not key.startswith("NV_COMPUTE_SANITIZER_")}
    for key in ("CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH"):
        values.pop(key, None)
    if sanitized or allocation:
        values["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    if allocation:
        from .allocation_audit import ENVIRONMENT

        values.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        values.update(ENVIRONMENT)
    return values


def command(tool, binary, log_pattern, worker):
    options = [binary, "--tool", tool, "--target-processes", "all", "--require-cuda-init", "yes",
               "--error-exitcode", str(ERROR_EXIT), "--log-file", str(log_pattern)]
    if tool == "memcheck":
        options += ["--padding", "32"]
    if tool == "racecheck":
        options += ["--racecheck-detect-level", "warn", "--racecheck-report", "analysis"]
    return [*options, "--", *worker]


def evaluate(payload, *, run=None, binary=None):
    """Used by the native CLI supervisor on every supported execution transport."""
    from .allocation_audit import validate_inline

    validate_inline(payload)
    from .throughput_audit import validate_inline as throughput_enabled

    throughput = throughput_enabled(payload)
    from .cache_audit import validate_inline as cache_enabled

    cache = cache_enabled(payload)
    injected = run is not None or binary is not None
    run = run or process
    settings = payload["eval_config"]
    started = time.monotonic()
    # The outer transport owns its normal request timeout. Leave cleanup time
    # inside that same allocation rather than granting tools an extra budget.
    deadline = started + settings["timeout"] - 5

    def remaining():
        return max(0.0, deadline - time.monotonic())

    with tempfile.TemporaryDirectory(prefix="kb-diagnostics-") as temporary:
        root = Path(temporary)

        def worker(request, stage, prefix=None, *, instrumented=False):
            directory = root / stage
            directory.mkdir()
            source, target, log = directory / "request.json", directory / "result.json", directory / "worker.log"
            atomic_json(source, request)
            invocation = [sys.executable, "-m", "kb_workflows.evaluate_task", str(source), str(target), "--native-worker"]
            if prefix:
                invocation = command(prefix, resolved["path"], directory / f"{prefix}.%p.log", invocation)
            limit = min(remaining(), settings["sanitizer_timeout"]) if stage != "native" else remaining()
            if limit <= 0:
                response = {"exit_code": 124, "timed_out": True, "request_wall_seconds": 0}
            else:
                try:
                    response = run(invocation, limit, log, environment=environment(
                        sanitized=instrumented, allocation=stage.startswith("allocation_")))
                except (OSError, ValueError, RuntimeError) as exc:
                    response = {"exit_code": None, "timed_out": False, "request_wall_seconds": None, "error": str(exc)}
            result = read_json(target, {})
            response.update(payload_sha256=fingerprint(request), result=result,
                            coverage=coverage(result, paired=request.get("task") == "paired_timing"))
            if prefix:
                logs = {}
                for path in sorted(directory.glob(f"{prefix}.*.log")):
                    if path.is_symlink() or path.stat().st_size > MAX_LOG_BYTES:
                        response["error"] = "Sanitizer log is a symlink or exceeds the retained log limit"
                        continue
                    value = path.read_text(errors="replace")
                    logs[path.name] = {"text": value, "sha256": hashlib.sha256(value.encode()).hexdigest()}
                response["logs"] = logs
            return response

        native = worker(payload, "native")
        base = native["result"]
        if native.get("exit_code") != 0 or native.get("timed_out") or not isinstance(base, dict) or not base:
            return failure("Native worker failed before optional diagnostics", native_process=native)
        if base.get("compiled") is not True or base.get("correctness") is not True:
            return base
        nonce = uuid.uuid4().hex
        tools = select_tools(settings.get("sanitizer_tools", "none"))
        document = {"protocol": PROTOCOL, "evaluator_sha256": evaluator_digest(), "payload_sha256": fingerprint(payload),
            "nonce": nonce, "native_result": copy.deepcopy(base), "native_process": {key: value for key, value in native.items() if key != "result"},
            "simulated_execution": injected, "tools": {}, "control": {}, "allocation_controls": {}, "throughput_controls": {}, "input_cache_control": {},
            "timing_policy": "only_original_uninstrumented_native_samples_are_scored",
            "coverage_scope": "tested_worker_and_descendants; no_proof_of_race_freedom_or_kernel_authorship",
            "security_isolation_certified": False}
        resolved = binary if binary is not None else (binary_identity(settings["sanitizer_command"], run=run, directory=root, timeout=remaining()) if tools else {})
        document["binary"] = resolved
        for tool in tools:
            request = child_payload(payload, nonce, tool)
            if resolved.get("status") != "measured":
                record = {"exit_code": None, "timed_out": False, "payload_sha256": fingerprint(request),
                          "result": {}, "coverage": None, "logs": {}, "error": resolved.get("reason")}
            else:
                record = worker(request, tool, tool, instrumented=True)
            document["tools"][tool] = record
        if settings.get("allocation_validation"):
            from .allocation_audit import STAGES

            for pattern, stage in STAGES.items():
                document["allocation_controls"][pattern] = worker(child_payload(payload, nonce, stage), stage)
        if throughput:
            from .throughput_audit import STAGES as THROUGHPUT_STAGES

            for mode, stage in THROUGHPUT_STAGES.items():
                document["throughput_controls"][mode] = worker(child_payload(payload, nonce, stage), stage)
        if cache:
            document["input_cache_control"] = worker(child_payload(payload, nonce, "input_cache"), "input_cache")
        preliminary = derive(document, payload)
        if (settings.get("determinism_checks") or settings.get("state_validation") or settings.get("numerical_validation") or settings.get("allocation_validation") or throughput or cache
                or all(preliminary[f"sanitizer_{tool}"]["status"] == "passed" for tool in tools)):
            document["control"] = worker(child_payload(payload, nonce, "control"), "control")
        if tools and not injected and resolved.get("status") == "measured":
            target = Path(resolved["path"])
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != resolved["sha256"]:
                document["binary"] = {"status": "not_measured", "reason": "Sanitizer executable changed during diagnostics"}
        document["request_wall_seconds"] = time.monotonic() - started
        return finish(document, payload)
