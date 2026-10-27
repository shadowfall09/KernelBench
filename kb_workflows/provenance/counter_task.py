"""Native-sequence Nsight collection and replay; diagnostic times never score."""

from __future__ import annotations

import base64
import copy
import hashlib
import os
import re
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ..common import atomic_json, fingerprint, read_json
from . import origin_task
from .work_counters import (
    LAUNCH_METRICS,
    MAX_BYTES,
    bind_report_binaries,
    execution_values,
    parse_csv,
    recipe,
    work_totals,
)

CONFIGURATION = "config_file_off; kernel_replay; profiler_API_native_calls; nvtx_rename; launch_metrics; base_units; all_children; cache_none; clocks_none; kernel_names_off"
PROTOCOL = "kernelbench_pro_native_work_counters_v2"
MODEL = "kernelbench_pro_native_execution_models_v1"


class Profiler:
    """Capture trusted functions before import; one profiler/NVTX window per call."""
    def __init__(self, payload, *, functions=None):
        self.nonce = payload["_native_counter"]["nonce"]
        if not isinstance(self.nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", self.nonce):
            raise ValueError("Native counters require a fresh parent-owned capture nonce")
        self.simulated, self.pid = functions is not None, os.getpid()
        if functions is None:
            import torch

            functions = (torch.cuda.profiler.start, torch.cuda.profiler.stop, torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop)
        self.start, self.stop, self.push, self.pop = functions
        self.windows = []

    @contextmanager
    def call(self, *, seed, signature, training):
        window = {"id": len(self.windows) + 1, "seed": seed, "input_signature": signature, "training": training,
                  "started": False, "stopped": False, "range_balanced": False}
        window["marker"] = f"kb_native_{self.nonce}_{window['id']}"
        self.windows.append(window)
        depth = self.push(window["marker"])
        if type(depth) is not int or depth < 0:
            raise ValueError("NVTX push did not create a valid range")
        try:
            self.start()
            window["started"] = True
            try:
                yield window
            finally:
                self.stop()
                window["stopped"] = True
        finally:
            window["range_balanced"] = self.pop() == depth


def command(binary, source, target, report, metrics):
    return [binary, "--config-file", "off", "--profile-from-start", "off", "--replay-mode", "kernel",
            "--cache-control", "none", "--clock-control", "none", "--rename-kernels", "off",
            "--target-processes", "all", "--nvtx", "--print-nvtx-rename", "kernel", "--print-kernel-base", "mangled",
            "--csv", "--page", "raw", "--print-metric-name", "name", "--print-units", "base",
            "--print-metric-instances", "none", "--print-summary", "none", "--metrics", ",".join(sorted(set(metrics) | set(LAUNCH_METRICS))),
            "--export", str(Path(report).with_suffix(".ncu-rep")),
            "--log-file", str(report), sys.executable, "-m", "kb_workflows.evaluation.evaluate_task",
            str(source), str(target), "--native-worker"]


def _receipt(value):
    return fingerprint({k: v for k, v in value.items() if k not in {"receipt_sha256", "metadata", "execution_models"}})


def native_contract(payload):
    """Review binds native settings/role/variant; actual inputs bind separately."""
    return fingerprint({k: v for k, v in origin_task.native_request(payload).items() if k != "_origin_timing_seeds"})


def _observations(value, *, allow_simulated=False):
    origin = value["origin"]
    summary = origin_task.replay(origin, allow_simulated=allow_simulated)
    payload, child = origin["native_request"], value["native_run"]
    rows = origin_task.validate_native(child, payload, stage="counter", simulated=value["simulated"])
    if rows != origin["native_runs"]["profile"]["calls"] or child["binding"] != origin["native_runs"]["profile"]["binding"]:
        raise ValueError("Counter process changed the complete native input/output/mode sequence or device")
    windows = child["counter_windows"]
    if (type(child.get("pid")) is not int or child["pid"] <= 0 or len(windows) != len(rows)
            or not isinstance(value.get("nonce"), str) or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])):
        raise ValueError("Missing counter process, nonce or call windows")
    markers = {}
    for window, row in zip(windows, rows, strict=True):
        if (window.get("id") != row["id"] or window.get("marker") != f"kb_native_{value['nonce']}_{row['id']}"
                or window.get("native_call_sha256") != fingerprint(row) or window.get("seed") != row["seed"]
                or fingerprint({"signature": window.get("input_signature")}) != row["input_signature_sha256"]
                or any(window.get(key) is not True for key in ("started", "stopped", "range_balanced", "output_verified", "no_grad"))
                or any(window.get(key) != row[key] for key in ("training", "training_after"))):
            raise ValueError("Native counter window is incomplete or rebound")
        markers[window["marker"]] = []
    parsed = parse_csv(value["raw_csv"].encode(), value["recipe"], pid=child["pid"], launch_identity=True)
    binaries, binary_issue = {}, "Missing or excessive native Nsight binary report"
    archive = value.get("raw_report")
    if archive is not None:
        if (not isinstance(archive, dict) or not isinstance(archive.get("base64"), str)
                or len(archive["base64"]) > 4 * ((MAX_BYTES + 2) // 3)):
            raise ValueError("Invalid or excessive native Nsight binary archive")
        raw = base64.b64decode(archive["base64"], validate=True)
        if (len(raw) > MAX_BYTES or type(archive.get("bytes")) is not int or len(raw) != archive["bytes"]
                or hashlib.sha256(raw).hexdigest() != archive.get("sha256")):
            raise ValueError("Native Nsight binary archive bytes changed")
        try:
            binaries = bind_report_binaries(raw, parsed["actions"], pid=child["pid"])
            binary_issue = None
        except ValueError as exc:
            binary_issue = str(exc)[:300]
    for action in parsed["actions"]:
        marker = action["identity"]["Kernel Name"]
        if marker not in markers:
            raise ValueError("Counter action has no unique trusted native NVTX call; nested/unscoped work is unsupported")
        markers[marker].append(action)
    observations = []
    for row, window, activity in zip(rows, windows, summary["calls"], strict=True):
        actions = markers[window["marker"]]
        if activity["id"] != row["id"] or not actions or len(actions) != activity["kernel_count"]:
            raise ValueError("Native counter launches omit or change source-capture call coverage")
        observations.append({"id": row["id"], "native_call_sha256": fingerprint(row),
                             "action_ids": [action["identity"]["ID"] for action in actions],
                             "launch_identities": [action["launch_identity"] for action in actions],
                             "kernel_identities": [binaries[action["identity"]["ID"]] for action in actions] if binaries else None,
                             "binary_identity_issue": binary_issue,
                             "profile_memory_activity": activity["memory_activity"],
                             "work": work_totals(actions, value["recipe"]),
                             "non_kernel_activity_count": activity["activity_count"] - activity["kernel_count"]})
    return observations


def validate(value, *, allow_simulated=False):
    try:
        if (value.get("protocol") != PROTOCOL or value.get("status") != "ok" or value.get("receipt_sha256") != _receipt(value)
                or type(value.get("simulated")) is not bool or value["simulated"] and not allow_simulated
                or value.get("scored_wall_times_replaced") is not False or value.get("formal_qualification_implemented") is not False
                or not value["simulated"] and (value["native_run"].get("counter_simulated") is not False
                                               or value["origin"]["native_coverage"].get("simulated") is not False)):
            raise ValueError("Native counter report is missing, changed or simulated")
        tool = value["tool"]
        if (tool.get("name") != "ncu" or not re.fullmatch(r"[0-9a-f]{64}", tool.get("sha256", ""))
                or not isinstance(tool.get("version"), str) or not tool["version"].strip()
                or tool.get("version_sha256") != hashlib.sha256(tool["version"].encode()).hexdigest()
                or tool.get("configuration") != CONFIGURATION or tool.get("binary_unchanged") is not True
                or tool.get("exit_code") != 0 or tool.get("timed_out") is not False):
            raise ValueError("Native counters lack a completed unchanged Nsight executable identity")
        if _observations(value, allow_simulated=allow_simulated) != value["calls"]:
            raise ValueError("Native counter totals differ from raw actions and actual calls")
        return value
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Incomplete or invalid native counter report") from exc


def execution_models(value, calibration, *, allow_simulated=False):
    validate(value, allow_simulated=allow_simulated)
    payload = value["origin"]["native_request"]
    if value["recipe"]["review"].get("native_contract_sha256") != native_contract(payload):
        raise ValueError("Complete native work review must bind the native configuration, role and baseline variant")
    settings = payload["eval_config"]
    policy = {"native_tolerance": settings["fp32_tolerance"] if settings["precision"] == "fp32" else 1e-2}
    if settings.get("numerical_validation"):
        from ..numerics.numerical_audit import _spec

        _, policy = _spec(payload)
    models = []
    source = payload["ref_arch_src"] if payload.get("_origin_role") == "reference" else payload["kernel_src"]
    for observed, row in zip(value["calls"], value["native_run"]["calls"], strict=True):
        if observed["non_kernel_activity_count"]:
            raise ValueError("Native memcpy/memset lacks counter-process memory identity and transfer calibration; logical payload is not DRAM traffic")
        binding = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
                   "device": value["native_run"]["binding"]["device_fingerprint"], "precision": settings["precision"],
                   "input_signature_sha256": row["input_signature_sha256"], "policy_sha256": fingerprint(policy)}
        latency, components = execution_values(binding, value["recipe"], observed["work"], calibration,
                                               digest=settings["evaluator_sha256"])
        models.append({"id": row["id"], "native_call_sha256": fingerprint(row), "sol_execution_ms": latency, "components_ms": components})
    result = {"protocol": MODEL, "counter_receipt_sha256": value["receipt_sha256"], "calibration": calibration, "calls": models,
              "simulated": value["simulated"], "hardware_counters_collected": not value["simulated"],
              "interpretation": "reviewed_per_native_call_demands_over_attained_rates; not_a_physical_lower_bound"}
    result["receipt_sha256"] = fingerprint(result)
    return result


def scored_models(value, result, calibration, *, allow_simulated=False):
    """Join counters to actual timers and source-profile workload observations.

    Original symbols/configurations and optional cubin bytes come from the same
    ncu collection. Missing binary evidence stays unknown; memory work is separate.
    """
    from .kernel_origin import workload_identity

    validate(value, allow_simulated=allow_simulated)
    payload = value["origin"]["native_request"]
    if payload.get("_origin_role", "candidate") != "candidate":
        raise ValueError("Scored counter models require the actual candidate role")
    request, observed = origin_task.validate_scored(result, payload, allow_simulated=allow_simulated)
    actual = origin_task.scored_origin(result, allow_simulated=allow_simulated)
    profiled = origin_task.replay(value["origin"], allow_simulated=allow_simulated)
    profile_calls = value["origin"]["native_runs"]["profile"]["calls"]
    if (observed["calls"] != profile_calls
            or observed["binding"]["device_fingerprint"] != value["native_run"]["binding"]["device_fingerprint"]):
        raise ValueError("Counter source profile differs from actual inputs, states, outputs, modes or physical device")
    models = execution_models(value, calibration, allow_simulated=allow_simulated)
    joined = []
    timings = {row["id"]: row for row in observed["native_timings"]}
    for call, measured, replayed, counted, model in zip(observed["calls"], actual["calls"], profiled["calls"], value["calls"], models["calls"], strict=True):
        identity = workload_identity(measured)
        if (identity != workload_identity(replayed) or measured["id"] != call["id"] or replayed["id"] != call["id"]
                or model["native_call_sha256"] != fingerprint(call)):
            raise ValueError("Source-profile kernel binary, entrypoint, launch configuration or multiplicity differs from scored work")
        expected = [{key: row["kernel_identity"][key] for key in ("symbol", "configuration")} for row in measured["activities"]]
        if sorted(map(fingerprint, expected)) != sorted(map(fingerprint, counted["launch_identities"])):
            raise ValueError("Counter-process original symbol, launch configuration or multiplicity differs from scored work")
        if counted["kernel_identities"] is not None:
            expected = [row["kernel_identity"] for row in measured["activities"]]
            counted_kernels = [{key: row[key] for key in ("binary_sha256", "symbol", "configuration")}
                               for row in counted["kernel_identities"]]
            if sorted(map(fingerprint, expected)) != sorted(map(fingerprint, counted_kernels)):
                raise ValueError("Counter-process cubin binary differs from actual scored work")
        busy = measured["device_busy_ns"] / 1e6
        wall = timings.get(call["id"], {}).get("wall_ms")
        if not busy or wall is not None and busy > wall * (1 + 1e-6):
            raise ValueError("Actual scored activity is missing or exceeds its own measured timer")
        joined.append({**model, "phase": call["phase"], "seed": call["seed"], "workload": identity,
                       "scored_device_busy_ms": busy, "scored_wall_ms": wall,
                       "O": wall / busy if wall is not None else None})
    simulated = bool(value["simulated"] or value["origin"]["native_coverage"]["simulated"]
                     or result["metadata"]["origin_capture"]["raw"].get("simulated") is not False)
    binary_observed = all(call["kernel_identities"] is not None for call in value["calls"])
    output = {"protocol": "kernelbench_pro_scored_execution_models_v1", "calls": joined,
              "status": "simulated" if simulated else "measured",
              "counter_receipt_sha256": value["receipt_sha256"],
              "actual_result_sha256": fingerprint(result), "actual_origin_sha256": actual["raw_sha256"],
              "profile_origin_sha256": profiled["raw_sha256"], "calibration_sha256": calibration["receipt_sha256"],
              "evaluator_sha256": request["eval_config"]["evaluator_sha256"],
              "simulated": simulated,
              "counter_process_launch_identity_observed": True,
              "counter_process_binary_identity_observed": binary_observed,
              "counter_process_workload_identity_observed": binary_observed, "formal_qualification_implemented": False,
              "scope": "actual_scored_calls_to_same_collection_cubin_symbol_configuration" if binary_observed else
                       "actual_scored_calls_to_profile_binary_and_counter_symbol_configuration; counter_process_binary_unproven",
              "scored_wall_times_replaced": False}
    output["receipt_sha256"] = fingerprint(output)
    return output


def audit(payload, *, run=None, binary=None):
    from ..evaluation.tool_runner import binary_identity, environment, process

    origin = None
    with tempfile.TemporaryDirectory(prefix="kb-native-counter-") as temporary:
        directory = Path(temporary)
        try:
            settings = payload["eval_config"]
            if settings.get("use_hidden_tests") or settings.get("device_backend", "cuda") != "cuda":
                raise ValueError("Native Nsight counters require public CUDA evaluation or paired timing")
            metrics = recipe(payload["_native_counter"]["recipe"])
            runner = run or process
            deadline = time.monotonic() + settings["timeout"]
            tool = binary_identity(binary or "ncu", run=runner, directory=directory, timeout=max(0, deadline - time.monotonic()))
            if tool.get("status") != "measured":
                raise ValueError("Nsight executable/version unavailable within the original budget")
            base = copy.deepcopy(payload)
            base.pop("_native_counter", None)
            base["task"] = "origin_audit"
            origin = origin_task.audit(base, run=run, budget_seconds=max(0, (deadline - time.monotonic() - 5) / 2))
            origin_task.replay(origin, allow_simulated=run is not None)
            nonce = secrets.token_hex(16)
            child = origin_task.native_request(base)
            child.update(_origin_stage="counter", _native_counter={"nonce": nonce}, build_root=str(directory / "build"))
            child["request_id"] = "kb-" + fingerprint(child)
            source, target, csv = directory / "request.json", directory / "worker.json", directory / "counters.csv"
            atomic_json(source, child)
            argv = command(tool["path"], source, target, csv, metrics)
            remaining = deadline - time.monotonic() - 5
            if remaining <= 0:
                raise ValueError("Native counter collection exhausted its original timeout")
            result = runner(argv, remaining, directory / "application.log", environment=environment())
            if (result.get("exit_code") != 0 or result.get("timed_out") is not False
                    or any(not p.is_file() or p.is_symlink() or p.stat().st_size > MAX_BYTES for p in (csv, target))):
                raise ValueError("Native Nsight process failed or left missing/excessive reports")
            if hashlib.sha256(Path(tool["path"]).read_bytes()).hexdigest() != tool["sha256"]:
                raise ValueError("Nsight executable changed during native collection")
            document = read_json(target)
            tool.update(name="ncu", configuration=CONFIGURATION, binary_unchanged=True, exit_code=0, timed_out=False)
            archives = [path for path in (directory / "counters.ncu-rep", directory / "counters.ncu-repz") if path.exists()]
            raw_report = None
            if len(archives) == 1 and archives[0].is_file() and not archives[0].is_symlink() and archives[0].stat().st_size <= MAX_BYTES:
                with archives[0].open("rb") as stream:
                    raw = stream.read(MAX_BYTES + 1)
                if len(raw) <= MAX_BYTES:
                    raw_report = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            value = {"protocol": PROTOCOL, "status": "ok", "purpose": "native_work_counters_diagnostics_only",
                     "origin": origin, "native_run": document, "recipe": payload["_native_counter"]["recipe"], "nonce": nonce,
                     "raw_csv": csv.read_text(encoding="utf-8-sig"), "raw_report": raw_report, "tool": tool, "command": argv,
                     "simulated": run is not None or binary is not None or document.get("counter_simulated") is not False,
                     "scored_wall_times_replaced": False, "formal_qualification_implemented": False}
            value["calls"] = _observations(value, allow_simulated=value["simulated"])
            value["receipt_sha256"] = _receipt(value)
            return validate(value, allow_simulated=value["simulated"])
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, AttributeError, UnicodeError, ImportError) as exc:
            artifacts = {}
            for name in ("counters.csv", "counters.ncu-rep", "counters.ncu-repz", "application.log", "sanitizer-version.txt", "worker.json"):
                path = directory / name
                if path.is_file() and not path.is_symlink():
                    size = path.stat().st_size
                    with path.open("rb") as stream:
                        raw = stream.read(64 * 1024)
                    artifacts[name] = {"bytes": size, "prefix_base64": base64.b64encode(raw).decode(),
                                       "prefix_sha256": hashlib.sha256(raw).hexdigest(), "complete": size == len(raw)}
            return {"status": "not_measured", "purpose": "native_work_counters_diagnostics_only", "reason": str(exc),
                    "origin": origin, "failure_artifacts": artifacts, "scored_wall_times_replaced": False,
                    "formal_qualification_implemented": False}
