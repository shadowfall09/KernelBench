"""Optional NRT observations and actual invocation binding; no device work on import."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..common import ROOT, fingerprint
from ..hardware.neuron_tables import file_digest, integer
from .neuron_origin import _seal, driver_events, validate_compilation
from .work_counters import sha

PROTOCOL = "kernelbench_pro_neuron_native_runtime_v1"
SCOPE = "outermost_preloaded_nrt_load_and_collectives; nrta_execute_schedule; current_process; development_only"
CALL_SCOPE = "outermost_preloaded_nrt_load_and_collectives; nrta_execute_schedule; actual_native_invocations; no_formal_qualification"
NATIVE_CALL_SCOPE = "actual_native_correctness_both_orders_warmups_timing_and_paired; strict_auxiliary_calls_unscoped"
SOURCE = Path(__file__).with_suffix(".cpp")
ENV = "KB_NEURON_RUNTIME_MANIFEST"
COMPILER = "kernelbench_pro_neuron_compiler_job_v1"
SCORED_MODE = "synchronized_wall_continuous_nrt_v1"


def capture_artifacts(directory, manifest, maximum):
    """Transfer original snapshots by content hash, without retaining duplicates."""
    result = {}
    for name, entry in manifest.items():
        digest = entry["sha256"]
        if digest in result:
            continue
        raw = (Path(directory) / name).read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Neuron scoring artifact changed during transfer")
        result[digest] = base64.b64encode(raw).decode()
        if sum(map(len, result.values())) > maximum:
            raise ValueError("Neuron scoring artifacts exceed the configured report limit")
    return result


def _artifact_files(capture, maximum):
    from ..hardware.neuron_processing import DEFAULT_LIMITS, descriptors_valid

    manifest, data = capture["artifact_files"], capture["artifact_data"]
    descriptors_valid(manifest, DEFAULT_LIMITS)
    if (not isinstance(data, dict) or set(data) != {r["sha256"] for r in manifest.values()}
            or any(not isinstance(raw, str) for raw in data.values()) or sum(map(len, data.values())) > maximum):
        raise ValueError("Neuron scoring artifacts are missing or exceed their transfer limit")
    sizes = {}
    for digest, encoded in data.items():
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Neuron scoring artifact bytes changed")
        sizes[digest] = len(raw)
    if any(sizes[row["sha256"]] != row["bytes"] for row in manifest.values()):
        raise ValueError("Neuron scoring artifact size changed")
    return {p: r["sha256"] for p, r in manifest.items()}, {p: r["bytes"] for p, r in manifest.items()}


def invocation_receipt(payload, result, state):
    """Retain actual native calls; this does not grant a source or timing grade."""
    from .origin_task import _document, _scored_timing, native_request

    metadata = result["metadata"]
    runtime = metadata.get("origin_capture", {}).get("native_runtime")
    request = native_request(payload)
    value = _document(request, result, state, "scored")
    value.pop("native_result")
    value.update(native_request=request, timing_sha256=fingerprint(_scored_timing(result)), timing_seed_plan=state.seeds,
                 windows=state.capture.windows if state.capture is not None else [], paired_windows=state.paired_windows,
                 native_runtime_sha256=runtime.get("receipt_sha256") if runtime else None,
                 execution_origin_collected=runtime is not None, simulated=metadata["simulated_execution"],
                 timed_callable_wrapped=False, formal_origin_qualified=False, status="captured", issues=[], scope=NATIVE_CALL_SCOPE)
    if not state.registry.intact():
        value["environment_changes"].append("registered_source")
    _seal(value)
    try:
        validate_invocations(value, payload, result, allow_simulated=True)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        value.update(status="partial", issues=[str(exc)[:300]])
    return _seal(value)


def validate_invocations(value, payload, result, *, allow_simulated=False):
    from ..evaluation.paired_task import call_plan
    from .origin_task import (
        BASELINE_SCOPE,
        NATIVE_PROTOCOL,
        NATIVE_SCOPE,
        _scored_timing,
        native_request,
        validate_call_rows,
        validate_call_timings,
    )

    if (not isinstance(value, dict) or value.get("protocol") != NATIVE_PROTOCOL
            or value.get("scope") != NATIVE_CALL_SCOPE or type(value.get("simulated")) is not bool
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
            or value.get("native_request") != native_request(payload)
            or value.get("binding", {}).get("native_request_sha256") != fingerprint(native_request(payload))
            or value.get("stage") != "scored" or not integer(value.get("pid"), minimum=1)
            or value.get("binding", {}).get("role") != "candidate"
            or value.get("binding", {}).get("scope") != (BASELINE_SCOPE if payload.get("task") == "baseline" else NATIVE_SCOPE)
            or value.get("timing_sha256") != fingerprint(_scored_timing(result))
            or value.get("simulated") is not False and not allow_simulated
            or value.get("timed_callable_wrapped") is not False or value.get("formal_origin_qualified") is not False
            or value.get("status") != "captured" or value.get("issues") != [] or value.get("environment_changes") != []):
        raise ValueError("Neuron invocation receipt is missing, changed, partial or simulated")
    rows = validate_call_rows(value.get("calls"))
    settings, metadata = payload["eval_config"], result["metadata"]
    checks = metadata.get("reliability_checks", [])
    if (not checks or [c["label"] for c in checks] != [c["label"] for c in metadata["correctness_cases"]]
            or any(c.get("passed") is not True or c.get("orders") != ["candidate_first", "reference_first"] for c in checks)):
        raise ValueError("Neuron invocations lack their complete correctness cases")
    if settings.get("use_hidden_tests"):
        from src.reliability import HIDDEN_SEEDS

        if (metadata.get("hidden_completed_checks") != len(checks)
                or {c["seed"] for c in checks} != set(settings.get("hidden_seeds", HIDDEN_SEEDS))):
            raise ValueError("Neuron invocations changed their hidden seed/case coverage")
    elif [(c["label"], c["seed"]) for c in checks] != [
            (f"standard/{i}", settings.get("evaluation_seed", 42) + i) for i in range(settings["num_correct_trials"])]:
        raise ValueError("Neuron invocations changed their public correctness coverage")
    expected = [("correctness", c["seed"], c["label"], order) for c in checks for order in c["orders"]]
    seeds = value["timing_seed_plan"]
    if (not isinstance(seeds, list) or value.get("seeds_consumed") != len(seeds)
            or any(not integer(s) for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Neuron invocation seed plan changed")
    if payload.get("task") == "paired_timing":
        plan = list(call_plan(trials=settings["paired_trials"], warmup=3, seed=settings.get("evaluation_seed", 42)))
        paired = metadata["paired_timing"]
        if (paired.get("num_pairs") != settings["paired_trials"]
                or [(row["seed"], row["order"]) for row in paired["pairs"]] != [(seed, order) for _, seed, order in plan[3:]]):
            raise ValueError("Neuron paired samples changed their seed/order plan")
        expected += [("paired_warmup" if index < 3 else "paired", seed, str(index), order) for index, seed, order in plan]
    else:
        count = 0 if settings.get("use_hidden_tests") else 3 + settings["num_perf_trials"]
        if len(seeds) != count or count and metadata["timing_receipt"]["input_seeds"] != seeds:
            raise ValueError("Neuron invocation timing seed coverage changed")
        expected += [("warmup" if n < 3 else "timing", seed, str(n), None) for n, seed in enumerate(seeds)]
    if [(r["phase"], r["seed"], r["label"], r["order"]) for r in rows] != expected:
        raise ValueError("Neuron invocations omit or change actual correctness/timing calls")
    timings = validate_call_timings(value, result, paired=payload.get("task") == "paired_timing", baseline=payload.get("task") == "baseline")
    runtime = metadata.get("origin_capture", {}).get("native_runtime")
    if (value.get("native_runtime_sha256") != (runtime.get("receipt_sha256") if runtime else None)
            or value.get("execution_origin_collected") is not (runtime is not None)):
        raise ValueError("Neuron invocations differ from their native runtime capture")
    if runtime is not None:
        capture = metadata["origin_capture"]
        files, sizes = _artifact_files(capture, settings.get("isolation_report_mb", 64) * 1024**2)
        validate(runtime, files, sizes, allow_simulated=allow_simulated)
        validate_compilation(capture["compiler_observations"], files, sizes, allow_simulated=allow_simulated)
        if runtime.get("event_source") != "native_invocations" or capture["compiler_output_links"] != compiler_links(capture["compiler_observations"], runtime):
            raise ValueError("Neuron actual-call compiler/runtime association changed")
        calls = runtime["native_calls"]
        windows = value["windows"]
        if (len(windows) != len(rows) or len(calls) != len(rows)
                or [(w["id"], w["start_ns"], w["end_ns"]) for w in windows] != [(c["id"], c["start_ns"], c["end_ns"]) for c in calls]
                or any(w.get("native_call_sha256") != fingerprint(r) or w.get("output_verified") is not True
                       for w, r in zip(windows, rows))):
            raise ValueError("Neuron native NRT scopes differ from the actual verifier calls")
        by_id = {w["id"]: w for w in windows}
        if any(t["wall_ms"] > (by_id[t["id"]]["end_ns"] - by_id[t["id"]]["start_ns"]) / 1e6 for t in timings):
            raise ValueError("Neuron native timer exceeds its own NRT invocation scope")
    elif value["windows"]:
        raise ValueError("Neuron invocation windows lack their native runtime")
    return value


def validate_scored(result, payload=None, *, baseline=False, allow_simulated=False):
    """Replay a complete actual request through the shared scoring interface."""
    from ..evaluation.evaluate_task import validate_sources
    from .origin_task import _scored_contract

    try:
        metadata = result["metadata"]
        record = metadata["native_call_receipt"]
        request = record["native_request"]
        settings = request["eval_config"]
        if (settings.get("device_backend") != "neuron" or settings.get("evidence_level") not in {"L1", "L2"}
                or settings.get("diagnostic_stage") or "_origin_timing_seeds" in request
                or request.get("task") not in ({"baseline"} if baseline else {"evaluate", "paired_timing"})
                or payload is not None and _scored_contract(request) != _scored_contract(payload)):
            raise ValueError("Neuron scored request changed its phase, level or native contract")
        validate_sources(request)
        validate_invocations(record, request, result, allow_simulated=allow_simulated)
        device = metadata["device_fingerprint"]
        sources = {"reference.py": hashlib.sha256(request["ref_arch_src"].encode()).hexdigest()}
        if not baseline:
            sources["candidate.py"] = hashlib.sha256(request["kernel_src"].encode()).hexdigest()
        if (metadata.get("compiler_source_sha256") != sources
                or metadata.get("compiler_source_registration") != "frozen_compile_code_objects"
                or metadata.get("reference_sha256") != fingerprint({"source": request["ref_arch_src"]})
                or not baseline and metadata.get("kernel_sha256") != fingerprint({"source": request["kernel_src"]})
                or metadata.get("precision") != settings["precision"]
                or metadata.get("simulated_execution") is not record["simulated"]
                or record["binding"].get("device_fingerprint") != device
                or record["binding"].get("evaluator_sha256") != settings["evaluator_sha256"]
                or (result.get("baseline_correctness") is not True if baseline else result.get("correctness") is not True)):
            raise ValueError("Neuron actual call source, device, precision or output binding changed")
        if not allow_simulated:
            from ..analysis.calibration import identity

            identity(device)
            if device.get("device_backend") != "neuron" or device.get("cpu_fallback_allowed") is not False:
                raise ValueError("Neuron scoring requires the actual native device without fallback")
        return request, record
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Missing or invalid actual Neuron scored calls") from exc


def timing_observer(result, *, baseline=False, allow_simulated=False):
    validate_scored(result, baseline=baseline, allow_simulated=allow_simulated)
    capture = result["metadata"].get("origin_capture", {})
    if capture.get("mode") != SCORED_MODE or capture.get("native_runtime", {}).get("event_source") != "native_invocations":
        raise ValueError("Actual Neuron timing requires its native NRT invocation observer")
    return SCORED_MODE


def _compile_job(plan, arguments):
    """Self-contained stdlib child; frozen into the preload before candidate code."""
    import hashlib
    import json
    import os
    import subprocess
    from pathlib import Path

    directory, cwd = Path(arguments[0]), Path(arguments[1]).resolve()
    original = arguments[2:]
    command = [plan["path"], *original[1:]]
    def digest(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024**2:
            raise ValueError("Compiler file exceeds its hash envelope")
        with path.open("rb") as stream:
            raw = stream.read(128 * 1024**2 + 1)
        if len(raw) > 128 * 1024**2:
            raise ValueError("Compiler file grew beyond its hash envelope")
        return hashlib.sha256(raw).hexdigest()
    def snapshot(path, name, maximum=128 * 1024**2):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise ValueError("Invalid compiler input/output file envelope")
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("Compiler file grew beyond its envelope")
        (directory / name).write_bytes(raw)
        return {"path": str(path.resolve()), "artifact": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    def constants(parent):
        names = []
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.name.startswith(".") and entry.name.endswith(".npy"):
                    names.append(entry.name)
                    if len(names) > 256:
                        raise ValueError("NKI constant directory exceeds 256 files")
        return sorted(names)
    value = {"protocol": plan["protocol"], "simulated": plan["simulated"], "pid": os.getpid(), "command": command,
             "cwd": str(cwd), "compiler": {"path": plan["path"], "sha256": plan["sha256"]},
             "inputs": [], "constant_directories": {}, "output": None, "issues": [], "exit_code": None}
    try:
        directory.mkdir(parents=True, exist_ok=False)
        if (len(original) < 4 or original[1] != "compile" or original[2] not in {"module.mlir", "sg00/bir.json"} or original.count("--output") != 1
                or original[-2:] != ["--output", "module.neff"]):
            raise ValueError("Unsupported compiler invocation")
        value["inputs"].append(snapshot(cwd / original[2], "input.mlir", 64 * 1024**2))
        index_path = Path(plan["index"])
        if index_path.is_file() and not index_path.is_symlink() and index_path.stat().st_size <= 8 * 1024**2:
            index = json.loads(index_path.read_bytes())
            rows = index.get(value["inputs"][0]["sha256"], [])
            if not isinstance(rows, list) or len(rows) > 256:
                raise ValueError("Invalid observed BIR input set")
            for row in rows:
                if row.get("role") == "bir":
                    parent = str(Path(row["path"]).parent)
                    value["constant_directories"][parent] = sorted({Path(item["path"]).name for item in rows
                        if str(Path(item["path"]).parent) == parent and Path(item["path"]).name.endswith(".npy")
                        and not Path(item["path"]).name.startswith(".")})
            for n, row in enumerate(rows):
                item = snapshot(row["path"], "bir-" + str(n) + ".bin", 16 * 1024**2)
                if item["sha256"] != row["sha256"]:
                    raise ValueError("Observed BIR changed before compiler invocation")
                value["inputs"].append(item)
                if sum(item["bytes"] for item in value["inputs"][1:]) > 64 * 1024**2:
                    raise ValueError("Observed BIR bundle exceeds 64 MiB")
            for parent, expected in value["constant_directories"].items():
                if constants(parent) != expected:
                    raise ValueError("NKI constant directory changed before compiler invocation")
    except Exception as exc:
        value["issues"].append(type(exc).__name__ + ": " + str(exc)[:240])
    if digest(plan["path"]) != plan["sha256"]:
        return 126
    environment = dict(os.environ)
    for key in ("LD_PRELOAD", "KB_NEURON_RUNTIME_DIRECTORY", "KB_NEURON_RUNTIME_MANIFEST", "KB_NEURON_COMPILER_SCRIPT"):
        environment.pop(key, None)
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False, timeout=plan["timeout"])
    value["exit_code"] = completed.returncode
    try:
        if completed.returncode == 0:
            value["output"] = snapshot(cwd / "module.neff", "compiler-" + directory.name + ".neff")
        if digest(plan["path"]) != plan["sha256"] or any(digest(row["path"]) != row["sha256"] for row in value["inputs"]):
            raise ValueError("Compiler tool or inputs changed during invocation")
        for parent, expected in value["constant_directories"].items():
            if constants(parent) != expected:
                raise ValueError("NKI constant directory changed during compiler invocation")
    except Exception as exc:
        value["issues"].append(type(exc).__name__ + ": " + str(exc)[:240])
    value["receipt_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    if directory.is_dir():
        (directory / "receipt.json").write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode


def build(directory, include, compiler, payload_sha256, *, simulated=False, backend_compiler=None, timeout=300):
    """Compile only the observer, using actual SDK declarations, never a model."""
    root, include = Path(directory).resolve(), Path(include).resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    binary = shutil.which(compiler)
    if not binary:
        raise ValueError("NRT observer requires the configured C++ compiler")
    binary = Path(binary).resolve(strict=True)
    headers = {str(include / "nrt" / name): file_digest(include / "nrt" / name)
               for name in ("nrt.h", "nrt_async.h", "nrt_experimental.h")}
    frozen = {str(SOURCE): file_digest(SOURCE), str(binary): file_digest(binary), **headers}
    library = root / "observer.so"
    command = [str(binary), "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
               "-I", str(include), str(SOURCE), "-ldl", "-o", str(library)]
    with (root / "build.log").open("xb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=30, check=True)
    if any(file_digest(path) != digest for path, digest in frozen.items()):
        raise ValueError("NRT observer compiler/source/headers changed during build")
    if library.is_symlink() or not 0 < library.stat().st_size <= 16 * 1024**2:
        raise ValueError("NRT observer library exceeds its regular-file envelope")
    manifest = {"protocol": PROTOCOL, "request_sha256": payload_sha256, "simulated": simulated,
                "source_sha256": frozen[str(SOURCE)], "compiler": str(binary), "inputs": frozen,
                "command": command, "library": str(library), "library_sha256": file_digest(library)}
    if backend_compiler:
        tool = shutil.which(backend_compiler)
        if not tool:
            raise ValueError("NRT compiler observation requires the configured neuronx-cc executable")
        tool = str(Path(tool).resolve(strict=True))
        plan = {"protocol": COMPILER, "path": tool, "sha256": file_digest(tool), "simulated": simulated,
                "index": str(root / "compiler-inputs.json"), "timeout": timeout}
        script = "import sys\nPLAN=" + repr(plan) + "\n" + inspect.getsource(_compile_job) + "\nraise SystemExit(_compile_job(PLAN,sys.argv[1:]))\n"
        script_path = root / "compiler-observer.py"
        script_path.write_text(script)
        manifest["backend_compiler"] = plan
        manifest["compiler_script"] = str(script_path)
        manifest["compiler_script_sha256"] = file_digest(script_path)
        manifest["observer_python"] = sys.executable
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return path


def bootstrap(payload):
    """Re-exec a fresh worker with the observer before Torch/Neuron is imported."""
    settings = payload["eval_config"]
    if not (settings.get("device_backend") == "neuron" and (payload.get("task") == "profile"
            or payload.get("task") in {"evaluate", "baseline", "paired_timing"} and settings.get("evidence_level") in {"L1", "L2"})
            and settings.get("neuron_runtime_origin")):
        return
    if ENV in os.environ:
        return  # Observer construction validates the current request and loaded symbols.
    if "torch_neuronx" in sys.modules:
        raise ValueError("NRT observer must be preloaded before importing the device SDK")
    directory = ROOT / "cache/workflow" / fingerprint(payload) / "neuron_runtime" / uuid.uuid4().hex
    path = build(directory, settings["neuron_runtime_include"], settings["neuron_runtime_compiler"], fingerprint(payload),
                 backend_compiler=settings.get("neuron_backend_compiler", "neuronx-cc"), timeout=settings.get("timeout", 300))
    manifest = json.loads(path.read_text())
    environment = {**os.environ, ENV: str(path), "KB_NEURON_RUNTIME_DIRECTORY": str(path.parent),
                   "LD_PRELOAD": manifest["library"] + (" " + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")}
    if "compiler_script" in manifest:
        environment.update(KB_NEURON_COMPILER_SCRIPT=manifest["compiler_script"], KB_NEURON_COMPILER_PYTHON=manifest["observer_python"])
    os.execve(sys.executable, [sys.executable, "-m", "kb_workflows.evaluation.evaluate_task", *sys.argv[1:]], environment)


def _journal(rows, *, with_calls=False):
    """Replay successful model generations; an async schedule is not completion."""
    if (not isinstance(rows, list) or not 2 <= len(rows) <= 200002 or any(not isinstance(row, dict) for row in rows)
            or rows[0].get("type") != "start" or rows[0].get("protocol") != 1
            or not integer(rows[0].get("pid"), minimum=1)
            or rows[-1] != {"type": "finish", "errors": 0}):
        raise ValueError("NRT journal is incomplete, unsupported or reports collection errors")
    models, loads, executions, load_ids, compiler_ids = {}, {}, {}, set(), set()
    calls, active, timestamp = [], None, 0
    for row in rows[1:-1]:
        if row.get("type") in {"call_begin", "call_end"}:
            ident, stamp = row.get("id"), row.get("timestamp_ns")
            if not integer(ident, minimum=1) or not integer(stamp, minimum=1) or stamp < timestamp:
                raise ValueError("Invalid native NRT invocation clock/identity")
            if row["type"] == "call_begin":
                if active is not None or ident != len(calls) + 1:
                    raise ValueError("Overlapping or repeated native NRT invocation")
                active = {"id": ident, "start_ns": stamp, "end_ns": None, "executions": []}
                calls.append(active)
            else:
                if active is None or active["id"] != ident or stamp <= active["start_ns"]:
                    raise ValueError("Native NRT invocation lacks its matching start")
                active["end_ns"], active = stamp, None
            timestamp = stamp
            continue
        if row.get("type") == "compiler_spawn":
            if (not integer(row.get("id"), minimum=1) or row["id"] > 4096 or not integer(row.get("child"))
                    or type(row.get("status")) is not int or row["id"] in compiler_ids
                    or row["status"] == 0 and not row["child"]):
                raise ValueError("Invalid native compiler spawn")
            compiler_ids.add(row["id"])
            continue
        if not integer(row.get("model")) or type(row.get("status")) is not int or not -(2**31) <= row["status"] < 2**31:
            raise ValueError("Invalid NRT journal model/status")
        if row.get("type") == "load":
            ident, size, name = row.get("id"), row.get("bytes"), row.get("artifact")
            if not integer(ident, minimum=1) or ident > 4096 or ident in load_ids or not integer(size):
                raise ValueError("Invalid NRT load generation")
            load_ids.add(ident)
            if row["status"] == 0:
                if not row["model"] or not 0 < size <= 128 * 1024**2 or name != f"nrt-load-{ident}.neff" or row["model"] in models:
                    raise ValueError("NRT load lacks its immutable binary snapshot")
                models[row["model"]] = ident
                loads[ident] = row
        elif row.get("type") == "unload":
            if row["status"] == 0 and models.pop(row["model"], None) is None:
                raise ValueError("NRT unload has no observed live model")
        elif row.get("type") == "execute":
            if row.get("call", 0) != (active["id"] if active is not None else 0):
                raise ValueError("NRT schedule differs from its active native invocation")
            if active is not None:
                active["executions"].append(row)
            if row["status"] == 0:
                seq, generation = row.get("sequence"), row.get("generation")
                if (not integer(seq) or seq >= 2**64 or not integer(generation, minimum=1)
                        or models.get(row["model"]) != generation):
                    raise ValueError("NRT schedule has no observed live model generation")
                executions.setdefault(seq, []).append(generation)
        else:
            raise ValueError("Unsupported NRT journal operation")
    if active is not None:
        raise ValueError("Native NRT invocation did not close")
    return (loads, executions, calls) if with_calls else (loads, executions)


def _call_events(rows):
    _, _, calls = _journal(rows, with_calls=True)
    events = []
    for call in calls:
        for row in call["executions"]:
            events.append({"index": len(events), "sequence": row["sequence"] if row["status"] == 0 else None,
                           "call_id": call["id"]})
    return calls, events


def _bindings(rows, events, files, sizes):
    loads, executions = _journal(rows)
    result, unresolved, indexes = [], [], set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid NRT device-event record")
        index, seq = event.get("index"), event.get("sequence")
        if not integer(index) or index >= 200000 or index in indexes:
            raise ValueError("Invalid NRT device-event coverage")
        indexes.add(index)
        candidates = executions.get(seq, []) if integer(seq) else []
        if len(candidates) != 1:
            unresolved.append(index)
            continue
        generation = candidates[0]
        load = loads[generation]
        path = "native_runtime/" + load["artifact"]
        if not sha(files.get(path)) or sizes.get(path) != load["bytes"]:
            raise ValueError("NRT loaded binary snapshot missing or changed")
        result.append({"event_index": index, "sequence": seq, "generation": generation,
                       "artifact": path, "neff_sha256": files[path], "neff_bytes": load["bytes"]})
    return result, unresolved


def validate_compiler_job(job, prefix, spawn, plan, files, sizes, *, allow_simulated=False):
    if (not isinstance(job, dict) or job.get("protocol") != COMPILER
            or job.get("simulated") is not False and not allow_simulated
            or job.get("receipt_sha256") != fingerprint({k: v for k, v in job.items() if k != "receipt_sha256"})
            or job.get("pid") != spawn.get("child") or spawn.get("status") != 0
            or job.get("compiler") != {"path": plan["path"], "sha256": plan["sha256"]}
            or job.get("simulated") is not plan.get("simulated")):
        raise ValueError("Invalid observed Neuron compiler job")
    raw = (json.dumps(job, separators=(",", ":")) + "\n").encode()
    if len(raw) > 1024**2 or files.get(prefix + "/receipt.json") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Neuron compiler job differs from its original receipt")
    command, inputs, output = job.get("command"), job.get("inputs"), job.get("output")
    if (not isinstance(command, list) or not 4 <= len(command) <= 256
            or any(not isinstance(s, str) or len(s) > 4096 for s in command)
            or command[0] != plan["path"] or command[1] != "compile"
            or not isinstance(inputs, list) or len(inputs) > 257
            or not isinstance(job.get("issues"), list) or len(job["issues"]) > 32):
        raise ValueError("Invalid Neuron compiler command/input envelope")
    for row in [*inputs, *([output] if output is not None else [])]:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("artifact"), str)
                or Path(row["artifact"]).name != row["artifact"] or row["artifact"] in {"", ".", ".."}
                or not sha(row.get("sha256")) or not integer(row.get("bytes"), minimum=1) or row["bytes"] > 128 * 1024**2
                or files.get(prefix + "/" + row["artifact"]) != row["sha256"]
                or sizes.get(prefix + "/" + row["artifact"]) != row["bytes"]):
            raise ValueError("Neuron compiler input/output snapshot missing or changed")
    if sum(row["bytes"] for row in inputs[1:]) > 64 * 1024**2:
        raise ValueError("Neuron compiler input bundle exceeds its byte envelope")
    directories = job.get("constant_directories", {})
    if not isinstance(directories, dict) or len(directories) > 256:
        raise ValueError("Invalid Neuron constant directory envelope")
    for parent, names in directories.items():
        expected = sorted({Path(row["path"]).name for row in inputs[1:] if str(Path(row["path"]).parent) == parent
                           and Path(row["path"]).name.endswith(".npy") and not Path(row["path"]).name.startswith(".")})
        if not Path(parent).is_absolute() or names != expected:
            raise ValueError("Neuron constant directory differs from its input snapshots")
    return job


def compiler_links(compilation, runtime):
    """Match observed files/tool outputs, without asserting auxiliary-input closure."""
    compilation = compilation if isinstance(compilation, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    jobs = runtime.get("compiler_jobs", [])
    rows = compilation.get("pipelines", [])
    links, unresolved = [], list(runtime.get("unresolved_events", []))
    for binding in runtime.get("bindings", []) if isinstance(runtime, dict) else []:
        matches = []
        for job in jobs:
            if job.get("exit_code") != 0 or job.get("issues") or not job.get("inputs") or not job.get("output"):
                continue
            if job["output"]["sha256"] != binding["neff_sha256"]:
                continue
            for index, row in enumerate(rows):
                native = row.get("native_ast") or {}
                emission = row.get("bir_emission") or {}
                observed = {(item["path"], item["sha256"]) for item in job["inputs"][1:]}
                if (row.get("fresh_lowering_observed") and native.get("all_definition_sources_matched")
                        and (compilation.get("simulated") is True or native.get("native_definition_sources_bound") is True)
                        and native.get("module_handoff_observed") and job["inputs"][0]["sha256"] in row.get("stablehlo_sha256", [])
                        and emission.get("cache_copy_matched") is True and emission.get("effective_files")
                        and str(Path(row.get("bir_path", "")).parent) in job.get("constant_directories", {})
                        and all((item["path"], item["sha256"]) in observed for item in emission["effective_files"])
                        and (row.get("bir_path"), row.get("bir_sha256")) in observed):
                    matches.append({"pipeline": index, "job_receipt_sha256": job["receipt_sha256"],
                                    "root_source_sha256": native.get("root_source_sha256"), "bir_sha256": row["bir_sha256"],
                                    "hlo_sha256": job["inputs"][0]["sha256"], "neff_sha256": binding["neff_sha256"]})
        if matches:
            links.append({"event_index": binding["event_index"], "matches": matches})
        else:
            unresolved.append(binding["event_index"])
    unresolved = sorted(set(unresolved))
    return _seal({"protocol": "kernelbench_pro_neuron_compiler_links_v1", "links": links, "unresolved_events": unresolved,
                  "simulated": bool(runtime.get("simulated")) or bool(compilation.get("simulated")),
                  "scope": "original-emission-matched NKI BIR/per-core/npy/debug bundle, HLO, compiler output and loaded NEFF; other external inputs not proven",
                  "formal_origin_qualified": False, "status": "observed" if links and not unresolved else "partial" if links else "not_measured"})


def validate(value, files, sizes, *, allow_simulated=False):
    native = isinstance(value, dict) and value.get("event_source") == "native_invocations"
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL or value.get("scope") != (CALL_SCOPE if native else SCOPE)
            or value.get("formal_origin_qualified") is not False
            or value.get("simulated") is not False and not allow_simulated
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})):
        raise ValueError("Invalid native NRT observation receipt")
    rows, events = value.get("journal"), value.get("device_events")
    if not isinstance(rows, list) or not isinstance(events, list) or len(events) > 200000 or not isinstance(sizes, dict):
        raise ValueError("Invalid native NRT observation envelope")
    raw = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    if len(raw) > 32 * 1024**2 or files.get("native_runtime/journal.jsonl") != hashlib.sha256(raw).hexdigest():
        raise ValueError("NRT journal bytes differ from their native artifact")
    build_info = value.get("build")
    if (not isinstance(build_info, dict) or build_info.get("protocol") != PROTOCOL
            or build_info.get("source_sha256") != file_digest(SOURCE)
            or not sha(build_info.get("library_sha256"))
            or files.get("native_runtime/observer.so") != build_info["library_sha256"]
            or (value.get("trace_sha256") is not None if native else files.get("framework_trace.json") != value.get("trace_sha256"))):
        raise ValueError("NRT observer build or framework trace is missing or changed")
    if native:
        calls, expected_events = _call_events(rows)
        if value.get("native_calls") != calls or events != expected_events:
            raise ValueError("NRT invocation events differ from their original native scopes")
    manifest_bytes = (json.dumps(build_info, separators=(",", ":")) + "\n").encode()
    if (build_info.get("simulated") is not value.get("simulated") or not sha(build_info.get("request_sha256"))
            or files.get("native_runtime/manifest.json") != hashlib.sha256(manifest_bytes).hexdigest()
            or not 0 < sizes.get("native_runtime/observer.so", 0) <= 16 * 1024**2):
        raise ValueError("NRT observer manifest differs from its retained build artifact")
    result, unresolved = _bindings(rows, events, files, sizes)
    proven = bool(events) and not unresolved and value["simulated"] is False
    if (value.get("bindings") != result or value.get("unresolved_events") != unresolved
            or value.get("loaded_binary_identity_proven") is not proven
            or value.get("status") != ("observed" if events and not unresolved else "partial" if result else "not_measured")):
        raise ValueError("NRT observations differ from replayed native loads and schedules")
    jobs = value.get("compiler_jobs", [])
    spawns = {row["id"]: row for row in rows if row.get("type") == "compiler_spawn"}
    plan = build_info.get("backend_compiler")
    if not isinstance(jobs, list) or len(jobs) > 4096 or jobs and not isinstance(plan, dict):
        raise ValueError("Invalid NRT compiler job list")
    if plan is not None and (not isinstance(plan, dict) or plan.get("protocol") != COMPILER
            or not sha(plan.get("sha256")) or not sha(build_info.get("compiler_script_sha256"))
            or files.get("native_runtime/compiler-observer.py") != build_info["compiler_script_sha256"]):
        raise ValueError("Native compiler observer script/plan changed")
    for job in jobs:
        ident = job.get("job_id")
        if ident not in spawns:
            raise ValueError("NRT compiler receipt lacks its actual spawn")
        validate_compiler_job({k: v for k, v in job.items() if k != "job_id"}, "native_runtime/compiler_jobs/" + str(ident),
                              spawns[ident], plan, files, sizes, allow_simulated=allow_simulated)
    return value


class Observer:
    def __init__(self, payload, *, allow_simulated=False):
        path = Path(os.environ.get(ENV, "")).resolve(strict=True)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024**2:
            raise ValueError("NRT observation requires its current bootstrap manifest")
        self.build = json.loads(path.read_text())
        if (self.build.get("protocol") != PROTOCOL or self.build.get("request_sha256") != fingerprint(payload)
                or self.build.get("simulated") is not False and not allow_simulated
                or self.build.get("source_sha256") != file_digest(SOURCE)):
            raise ValueError("NRT bootstrap does not match this worker request/source")
        self.directory, self.manifest = path.parent, path
        self.frozen = {str(path): file_digest(path), self.build["library"]: self.build["library_sha256"]}
        if self.build.get("compiler_script"):
            self.frozen.update({self.build["compiler_script"]: self.build["compiler_script_sha256"],
                                self.build["backend_compiler"]["path"]: self.build["backend_compiler"]["sha256"]})
        self.handle = ctypes.CDLL(None)
        self.finish = self.handle.kb_neuron_runtime_finish
        self.finish.restype, self.finish.argtypes = ctypes.c_int, []
        self.boundary = self.handle.kb_neuron_runtime_call
        self.boundary.restype, self.boundary.argtypes = ctypes.c_uint64, [ctypes.c_uint64, ctypes.c_int]
        self.windows = []
        if file_digest(self.build["library"]) != self.build["library_sha256"]:
            raise ValueError("NRT preloaded observer changed before candidate import")

    @staticmethod
    def timestamp():
        import time

        return time.monotonic_ns()

    @contextmanager
    def call(self, *, seed, signature, training):
        window = {"id": len(self.windows) + 1, "seed": seed, "input_signature": signature, "training": training,
                  "no_grad": True, "output_verified": False, "start_ns": None, "end_ns": None}
        self.windows.append(window)
        window["start_ns"] = self.boundary(window["id"], 1)
        if not window["start_ns"]:
            raise ValueError("Native NRT invocation could not start")
        try:
            yield window
        finally:
            window["end_ns"] = self.boundary(window["id"], 0)
            if not window["end_ns"]:
                raise ValueError("Native NRT invocation could not close")

    def export(self, directory, *, native_invocations=False):
        """Finish after synchronization; preserve the actual journal and load bytes."""
        if self.finish() != 0 or any(file_digest(p) != h for p, h in self.frozen.items()):
            raise ValueError("NRT observer failed or changed during execution")
        from ..hardware.neuron_processing import DEFAULT_LIMITS, file_manifest

        path = self.directory / "journal.jsonl"
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError("NRT journal exceeds its byte envelope")
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if rows[0].get("pid") != os.getpid():
            raise ValueError("NRT journal belongs to a different process")
        _journal(rows)
        target = Path(directory) / "native_runtime"
        target.mkdir()
        for source in [path, Path(self.build["library"]), self.manifest, *sorted(self.directory.glob("nrt-load-*.neff"))]:
            if source.is_symlink() or not source.is_file():
                raise ValueError("NRT observer artifacts must be regular files")
            shutil.copyfile(source, target / source.name)
            if file_digest(source) != file_digest(target / source.name):
                raise ValueError("NRT observer artifact changed during copying")
        jobs = []
        if self.build.get("compiler_script"):
            source = Path(self.build["compiler_script"])
            shutil.copyfile(source, target / source.name)
        job_root = self.directory / "compiler_jobs"
        if job_root.exists():
            inventory = file_manifest(job_root, DEFAULT_LIMITS)
            for name, info in inventory.items():
                destination = target / "compiler_jobs" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(job_root / name, destination)
                if file_digest(destination) != info["sha256"]:
                    raise ValueError("Native compiler artifacts changed during copying")
                if Path(name).name == "receipt.json":
                    if info["bytes"] > 1024**2:
                        raise ValueError("Native compiler receipt exceeds its envelope")
                    job = json.loads(destination.read_text())
                    job["job_id"] = int(Path(name).parent.name)
                    jobs.append(job)
        manifest = file_manifest(directory, DEFAULT_LIMITS)
        if native_invocations:
            calls, events = _call_events(rows)
            if [(r["id"], r["start_ns"], r["end_ns"]) for r in calls] != [(r["id"], r["start_ns"], r["end_ns"]) for r in self.windows]:
                raise ValueError("NRT scopes differ from the actual verifier boundaries")
        else:
            trace = Path(directory) / "framework_trace.json"
            if trace.stat().st_size > 64 * 1024**2:
                raise ValueError("NRT framework trace exceeds its byte envelope")
            raw_trace = trace.read_bytes()
            if len(raw_trace) > 64 * 1024**2 or hashlib.sha256(raw_trace).hexdigest() != manifest["framework_trace.json"]["sha256"]:
                raise ValueError("NRT framework trace changed while reading")
            trace_events = json.loads(raw_trace)["traceEvents"]
            if not isinstance(trace_events, list) or len(trace_events) > 200000:
                raise ValueError("NRT framework trace lacks bounded events")
            events = [{"index": index, "sequence": args.get("nrt_sequence_id") if valid else None}
                      for index, args, valid in driver_events(trace_events)]
        files, sizes = {k: r["sha256"] for k, r in manifest.items()}, {k: r["bytes"] for k, r in manifest.items()}
        bindings, unresolved = _bindings(rows, events, files, sizes)
        result = {"protocol": PROTOCOL, "scope": CALL_SCOPE if native_invocations else SCOPE, "build": self.build,
                       "simulated": self.build["simulated"], "formal_origin_qualified": False,
                       "trace_sha256": files.get("framework_trace.json") if not native_invocations else None, "journal": rows, "device_events": events,
                       "bindings": bindings, "unresolved_events": unresolved,
                       "compiler_jobs": jobs,
                       "loaded_binary_identity_proven": bool(events) and not unresolved and self.build["simulated"] is False,
                       "status": "observed" if events and not unresolved else "partial" if bindings else "not_measured"}
        if native_invocations:
            result.update(event_source="native_invocations", native_calls=calls)
        _seal(result)
        return validate(result, files, sizes, allow_simulated=self.build["simulated"] is True)
