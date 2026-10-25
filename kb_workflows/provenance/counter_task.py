"""Parent-owned Nsight Compute invocation; never used as scored latency."""

from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
import time
from pathlib import Path

from ..common import atomic_json, fingerprint, read_json
from .work_counters import MAX_BYTES, binding_valid, recipe, record

CONFIGURATION = "config_file_off; kernel_replay; profiler_API_single_call; base_units; all_children"


def command(binary, source, target, report, metrics):
    return [binary, "--config-file", "off", "--profile-from-start", "off", "--replay-mode", "kernel",
            "--target-processes", "all", "--csv", "--page", "raw", "--print-metric-name", "name",
            "--print-units", "base", "--print-metric-instances", "none", "--print-summary", "none",
            "--metrics", ",".join(sorted(metrics)), "--log-file", str(report),
            sys.executable, "-m", "kb_workflows.provenance.counter_worker", str(source), str(target)]


def validate_call(payload, child, device):
    from ..analysis.calibration import identity

    settings, binding = payload["eval_config"], binding_valid(child.get("binding"))
    source = payload["ref_arch_src"] if settings["counter_target"] == "reference" else payload["kernel_src"]
    policy = {"native_tolerance": settings["fp32_tolerance"] if settings["precision"] == "fp32" else 1e-2}
    if settings.get("numerical_validation"):
        from ..numerics.numerical_audit import _spec

        _, policy = _spec(payload)
    expected = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
                "evaluation_contract_sha256": fingerprint({"settings": settings}), "precision": settings["precision"],
                "target": settings["counter_target"], "seed": settings["counter_seed"],
                "policy_sha256": fingerprint(policy), "device_identity_sha256": identity(device)}
    if any(binding.get(k) != v for k, v in expected.items()) or not isinstance(child.get("input_signature"), list) or binding["input_signature_sha256"] != fingerprint({"signature": child["input_signature"]}):
        raise ValueError("Counter child changed source, inputs, policy, seed or execution device")


def _collect(payload, *, directory, run=None, binary=None, device_caps=None):
    from ..evaluation.tool_runner import binary_identity, environment, process

    settings = payload["eval_config"]
    if settings.get("use_hidden_tests") or settings.get("device_backend", "cuda") != "cuda":
        raise ValueError("Nsight counters require a separate public CUDA diagnostic")
    metrics = recipe(settings["counter_recipe"])
    simulated = run is not None or binary is not None or device_caps is not None
    run = run or process
    deadline = time.monotonic() + settings["timeout"]
    tool = binary_identity(binary or "ncu", run=run, directory=directory, timeout=max(0, deadline - time.monotonic()))
    if tool.get("status") != "measured":
        raise ValueError("Nsight Compute executable/version unavailable or discovery budget expired")
    if device_caps is None:
        from ..evaluation.measurement import device_fingerprint

        device_caps = device_fingerprint("cuda:0")
    source, target, report = directory / "request.json", directory / "worker.json", directory / "counters.csv"
    atomic_json(source, payload)
    argv = command(tool["path"], source, target, report, metrics)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Counter request has no remaining execution budget")
    result = run(argv, remaining, directory / "application.log", environment=environment())
    if (result.get("exit_code") != 0 or result.get("timed_out") is not False
            or not report.is_file() or report.is_symlink() or report.stat().st_size > MAX_BYTES
            or not target.is_file() or target.is_symlink() or target.stat().st_size > MAX_BYTES):
        raise ValueError("Nsight counter process timed out, failed or left missing/oversized reports")
    if hashlib.sha256(Path(tool["path"]).read_bytes()).hexdigest() != tool["sha256"]:
        raise ValueError("Nsight executable changed during capture")
    child = read_json(target)
    if child.get("status") != "ok":
        raise ValueError("Counter-call qualification failed: " + str(child.get("error")))
    validate_call(payload, child, device_caps)
    tool.update(name="ncu", configuration=CONFIGURATION, binary_unchanged=True, exit_code=0, timed_out=False)
    counters = record(report.read_bytes(), settings["counter_recipe"], child["binding"], tool,
                      simulated=simulated or child.get("simulated") is not False)
    # The command/report/call survive the temporary directory for review.
    return {"status": "ok", "purpose": "work_counters_diagnostics_only", "counters": counters,
            "call": child, "command": argv, "scored_wall_times_replaced": False,
            "origin_or_complete_native_coverage_proven": False}


def audit(payload, *, run=None, binary=None, device_caps=None):
    with tempfile.TemporaryDirectory(prefix="kb-counter-parent-") as temporary:
        directory = Path(temporary)
        try:
            return _collect(payload, directory=directory, run=run, binary=binary, device_caps=device_caps)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, UnicodeError) as exc:
            artifacts = {}
            for name in ("counters.csv", "application.log", "sanitizer-version.txt", "worker.json"):
                path = directory / name
                if not path.is_file() or path.is_symlink():
                    continue
                size = path.stat().st_size
                with path.open("rb") as stream:
                    raw = stream.read(min(size, 64 * 1024))
                artifacts[name] = {"bytes": size, "prefix_base64": base64.b64encode(raw).decode(),
                                   "prefix_sha256": hashlib.sha256(raw).hexdigest(), "complete": len(raw) == size}
            return {"status": "not_measured", "purpose": "work_counters_diagnostics_only", "reason": str(exc),
                    "failure_artifacts": artifacts, "scored_wall_times_replaced": False,
                    "origin_or_complete_native_coverage_proven": False}
