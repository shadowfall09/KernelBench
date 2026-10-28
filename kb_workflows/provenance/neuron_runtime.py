"""Opt-in development NRT load/execution observation; no device work on import."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from ..common import ROOT, fingerprint
from ..hardware.neuron_tables import file_digest, integer
from .neuron_origin import _seal, driver_events
from .work_counters import sha

PROTOCOL = "kernelbench_pro_neuron_native_runtime_v1"
SCOPE = "outermost_preloaded_nrt_load_and_collectives; nrta_execute_schedule; current_process; development_only"
SOURCE = Path(__file__).with_suffix(".cpp")
ENV = "KB_NEURON_RUNTIME_MANIFEST"


def build(directory, include, compiler, payload_sha256, *, simulated=False):
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
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return path


def bootstrap(payload):
    """Re-exec a fresh worker with the observer before Torch/Neuron is imported."""
    settings = payload["eval_config"]
    if not (settings.get("device_backend") == "neuron" and payload.get("task") == "profile"
            and settings.get("neuron_runtime_origin")):
        return
    if ENV in os.environ:
        return  # Observer construction validates the current request and loaded symbols.
    if "torch_neuronx" in sys.modules:
        raise ValueError("NRT observer must be preloaded before importing the device SDK")
    directory = ROOT / "cache/workflow" / fingerprint(payload) / "neuron_runtime" / uuid.uuid4().hex
    path = build(directory, settings["neuron_runtime_include"], settings["neuron_runtime_compiler"], fingerprint(payload))
    manifest = json.loads(path.read_text())
    environment = {**os.environ, ENV: str(path), "KB_NEURON_RUNTIME_DIRECTORY": str(path.parent),
                   "LD_PRELOAD": manifest["library"] + (" " + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")}
    os.execve(sys.executable, [sys.executable, "-m", "kb_workflows.evaluation.evaluate_task", *sys.argv[1:]], environment)


def _journal(rows):
    """Replay successful model generations; an async schedule is not completion."""
    if (not isinstance(rows, list) or not 2 <= len(rows) <= 200002 or any(not isinstance(row, dict) for row in rows)
            or rows[0].get("type") != "start" or rows[0].get("protocol") != 1
            or not integer(rows[0].get("pid"), minimum=1)
            or rows[-1] != {"type": "finish", "errors": 0}):
        raise ValueError("NRT journal is incomplete, unsupported or reports collection errors")
    models, loads, executions, load_ids = {}, {}, {}, set()
    for row in rows[1:-1]:
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
            if row["status"] == 0:
                seq, generation = row.get("sequence"), row.get("generation")
                if (not integer(seq) or seq >= 2**64 or not integer(generation, minimum=1)
                        or models.get(row["model"]) != generation):
                    raise ValueError("NRT schedule has no observed live model generation")
                executions.setdefault(seq, []).append(generation)
        else:
            raise ValueError("Unsupported NRT journal operation")
    return loads, executions


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


def validate(value, files, sizes, *, allow_simulated=False):
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL or value.get("scope") != SCOPE
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
            or files.get("framework_trace.json") != value.get("trace_sha256")):
        raise ValueError("NRT observer build or framework trace is missing or changed")
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
        self.handle = ctypes.CDLL(None)
        self.finish = self.handle.kb_neuron_runtime_finish
        self.finish.restype, self.finish.argtypes = ctypes.c_int, []
        if file_digest(self.build["library"]) != self.build["library_sha256"]:
            raise ValueError("NRT preloaded observer changed before candidate import")

    def export(self, directory):
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
        manifest = file_manifest(directory, DEFAULT_LIMITS)
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
        result = _seal({"protocol": PROTOCOL, "scope": SCOPE, "build": self.build,
                       "simulated": self.build["simulated"], "formal_origin_qualified": False,
                       "trace_sha256": files["framework_trace.json"], "journal": rows, "device_events": events,
                       "bindings": bindings, "unresolved_events": unresolved,
                       "loaded_binary_identity_proven": bool(events) and not unresolved and self.build["simulated"] is False,
                       "status": "observed" if events and not unresolved else "partial" if bindings else "not_measured"})
        return validate(result, files, sizes, allow_simulated=self.build["simulated"] is True)
