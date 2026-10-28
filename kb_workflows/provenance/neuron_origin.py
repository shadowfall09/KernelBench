"""Neuron compiler observations and runtime cache snapshots, without ownership claims.

Only the development profile uses Python instrumentation. Actual loaded NEFFs,
compiler dependency ownership and formal timing remain separate requirements.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from ..common import fingerprint
from ..hardware.neuron_tables import integer
from .work_counters import sha

RUNTIME_NEFF = "kernelbench_pro_neuron_runtime_neff_v1"
RUNTIME_NEFF_SCOPE = "privateuse1_driver_events_to_current_cache_snapshots; explicit_host_workers_excluded"
COMPILATION = "kernelbench_pro_neuron_compilation_observation_v1"
COMPILATION_SCOPE = "current_python_thread; observed_frontend_to_bir_to_framework_lowering; root_code_only"


def _seal(value):
    value["receipt_sha256"] = fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
    return value


def _compiler_functions():
    from nki.compiler import driver, frontend, ncc_driver
    from torch_neuronx.python_ops.neuron_nki_call_lowering import _lower_custom_nki

    return {"pipeline": driver.compile_to_bir, "frontend": frontend.NKIFrontend.compile,
            "bir": ncc_driver.compile_mlir_to_bir, "config": ncc_driver.NirResult.build_config,
            "lower": _lower_custom_nki}


class CompilationObserver:
    """Observe original SDK frames rather than wrapping public compiler aliases.

    Cache hits may skip these frames. They receive no fresh-compilation credit.
    The retained objects associate each handoff; names and cache keys do not.
    """

    def __init__(self, registry, *, functions=None):
        self.registry, self.injected = registry, functions is not None
        self.functions = functions
        self.rows, self.issues, self.frames, self.nirs, self.configs, self.binaries = [], [], {}, {}, {}, {}
        self.callback, self.previous = self._observe, None
        self.lower_count = 0
        self.entrypoints = {}
        self.value = None

    def __enter__(self):
        self.previous = sys.getprofile()
        try:
            if self.previous is not None:
                raise ValueError("Existing Python profiler; compiler observation not installed")
            self.functions = self.functions if self.injected else _compiler_functions()
            self.codes = {id(fn.__code__): (fn.__code__, name) for name, fn in self.functions.items()}
            if {name for _, name in self.codes.values()} != {"pipeline", "frontend", "bir", "config", "lower"}:
                raise ValueError("Unsupported NKI compiler observation contract")
            self.original_codes = {name: fn.__code__ for name, fn in self.functions.items()}
            self.entrypoints = {name: self.registry.callable_identity(fn) for name, fn in self.functions.items()}
            sys.setprofile(self.callback)
        except Exception as exc:
            self.issues.append(str(exc)[:300])
        return self

    def __exit__(self, *args):
        if self.previous is None and self.functions is not None and hasattr(self, "original_codes"):
            if sys.getprofile() is not self.callback:
                self.issues.append("Python compiler observation callback changed")
            sys.setprofile(self.previous)
            if any(fn.__code__ is not self.original_codes[name] for name, fn in self.functions.items()):
                self.issues.append("Observed SDK compiler code changed")
        self.value = _seal({"protocol": COMPILATION, "scope": COMPILATION_SCOPE,
            "simulated": self.injected, "pipelines": self.rows, "issues": self.issues,
            "compiler_entrypoints": self.entrypoints,
            "status": "partial" if self.issues or self.rows and not all(r["fresh_lowering_observed"] for r in self.rows)
                      else "observed" if self.rows else "not_measured",
            "formal_origin_qualified": False, "loaded_binary_identity_proven": False})
        self.frames.clear()
        self.nirs.clear()
        self.configs.clear()

    def _snapshot(self, path):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("NKI BIR must be a regular file")
            raw = stream.read(16 * 1024**2 + 1)
        if not 0 < len(raw) <= 16 * 1024**2:
            raise ValueError("NKI BIR exceeds the 16 MiB observation envelope")
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in self.binaries and sum(map(len, self.binaries.values())) + len(raw) > 64 * 1024**2:
            raise ValueError("NKI BIR snapshots exceed the 64 MiB envelope")
        self.binaries[digest] = raw
        return digest

    def _observe(self, frame, event, result):
        match = self.codes.get(id(frame.f_code))
        if match is None or match[0] is not frame.f_code or event not in {"call", "return"}:
            return
        try:
            self._event(match[1], frame, event, result)
        except Exception as exc:
            if len(self.issues) < 64:
                self.issues.append(type(exc).__name__ + ": " + str(exc)[:200])

    def _event(self, name, frame, event, result):
        local = frame.f_locals
        if name == "pipeline" and event == "call":
            if len(self.rows) >= 4096:
                raise ValueError("NKI pipeline observations exceed 4096 calls")
            kernel = local["kernel_func"]
            row = {"root": self.registry.callable_identity(getattr(kernel, "func", kernel)),
                   "frontend_observed": False, "bir_lowering_observed": False,
                   "config_observed": False, "fresh_lowering_observed": False,
                   "bir_sha256": None, "cache_keys": [], "stablehlo_sha256": []}
            self.rows.append(row)
            self.frames[frame] = {"row": row}
        if event != "return":
            return
        parent = frame
        while parent is not None and parent not in self.frames:
            parent = parent.f_back
        active = self.frames.get(parent)
        if name == "frontend" and active is not None and result is not None:
            kernel = local["kernel"]
            source = self.registry.callable_identity(kernel.func)
            if source == active["row"]["root"]:
                active["module"] = result.module
                active["row"]["frontend_observed"] = True
        elif name == "bir" and active is not None and result is not None:
            if local["module"] is active.get("module") and active["row"]["frontend_observed"]:
                active["bir"] = result
                active["row"]["bir_sha256"] = self._snapshot(result.descriptor.kernel_json_path)
                active["row"]["bir_lowering_observed"] = True
        elif name == "pipeline":
            active = self.frames.pop(frame)
            if result is not None:
                self.nirs[id(result)] = (result, active["row"], active.get("bir") is result)
        elif name == "config" and result is not None:
            nir = self.nirs.get(id(local["self"]))
            if nir and nir[0] is local["self"] and nir[2]:
                row = nir[1]
                if self._snapshot(nir[0].descriptor.kernel_json_path) == row["bir_sha256"]:
                    config = result.backend_config_b64
                    if type(config) is not bytes or not 0 < len(config) <= 1024**2:
                        raise ValueError("Invalid NKI backend configuration")
                    if len(self.configs) >= 4096 and id(result) not in self.configs:
                        raise ValueError("NKI configuration observations exceed 4096 objects")
                    self.configs[id(result)] = (result, row, config)
                    row["config_observed"] = True
        elif name == "lower" and result is not None:
            self.lower_count += 1
            if self.lower_count > 4096:
                raise ValueError("NKI lowering observations exceed 4096 calls")
            config = self.configs.get(id(local.get("dconfig")))
            if config and config[0] is local["dconfig"]:
                row = config[1]
                if local["backend_config"].encode("ascii") != config[2] or config[0].backend_config_b64 != config[2]:
                    raise ValueError("NKI backend configuration changed before lowering")
                if self._snapshot(local["dconfig"].backend_config["klir_binary"]["binary"]) != row["bir_sha256"]:
                    raise ValueError("NKI BIR changed before framework lowering")
                hlo, key = result.stablehlo_bytes, result.cache_key
                if type(hlo) is not bytes or not 0 < len(hlo) <= 64 * 1024**2 or type(key) is not str or not 0 < len(key) <= 4096:
                    raise ValueError("Invalid NKI StableHLO/key observation")
                digest = hashlib.sha256(hlo).hexdigest()
                if key not in row["cache_keys"]:
                    if len(row["cache_keys"]) >= 4096:
                        raise ValueError("NKI lowering observations exceed 4096 keys")
                    row["cache_keys"].append(key)
                    row["stablehlo_sha256"].append(digest)
                elif row["stablehlo_sha256"][row["cache_keys"].index(key)] != digest:
                    raise ValueError("NKI lowering key reused with different StableHLO")
                row["fresh_lowering_observed"] = True

    def export(self, directory):
        """Preserve observed BIR bytes inside the existing profile artifact tree."""
        if self.binaries:
            root = Path(directory) / "compiler_ir"
            root.mkdir(exist_ok=False)
            for digest, raw in self.binaries.items():
                with (root / (digest + ".bir")).open("xb") as stream:
                    stream.write(raw)
        return self.value


def validate_compilation(value, files, sizes, *, allow_simulated=False):
    if (not isinstance(value, dict) or value.get("protocol") != COMPILATION
            or value.get("scope") != COMPILATION_SCOPE or value.get("formal_origin_qualified") is not False
            or value.get("loaded_binary_identity_proven") is not False
            or value.get("simulated") is not False and not allow_simulated
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})):
        raise ValueError("Invalid Neuron compiler observation receipt")
    rows, issues = value.get("pipelines"), value.get("issues")
    if (not isinstance(rows, list) or len(rows) > 4096 or not isinstance(issues, list) or len(issues) > 64
            or any(not isinstance(issue, str) or len(issue) > 300 for issue in issues)):
        raise ValueError("Invalid Neuron compiler observation envelope")
    entrypoints = value.get("compiler_entrypoints")
    if (not isinstance(entrypoints, dict) or entrypoints and set(entrypoints) != {"pipeline", "frontend", "bir", "config", "lower"}
            or any(not isinstance(item, dict) or not sha(item.get("code_sha256")) for item in entrypoints.values())
            or rows and not entrypoints):
        raise ValueError("Invalid Neuron compiler entrypoint observations")
    artifacts = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("root"), dict)
                or any(type(row.get(k)) is not bool for k in ("frontend_observed", "bir_lowering_observed", "config_observed", "fresh_lowering_observed"))
                or row["bir_lowering_observed"] and not row["frontend_observed"]
                or row["bir_lowering_observed"] and not sha(row.get("bir_sha256"))
                or bool(row.get("bir_sha256")) is not row["bir_lowering_observed"]
                or row["config_observed"] and not row["bir_lowering_observed"]
                or row["fresh_lowering_observed"] and not row["config_observed"]):
            raise ValueError("Inconsistent Neuron compiler stages")
        digest = row.get("bir_sha256")
        root = row["root"]
        if (root.get("kind") not in {"registered_code", "external_code_observation", "builtin_observation"}
                or root["kind"] == "registered_code" and (not sha(root.get("source_sha256")) or not integer(root.get("code_index")))
                or root["kind"] == "external_code_observation" and not sha(root.get("code_sha256"))):
            raise ValueError("Invalid Neuron compiler root code observation")
        if digest is not None:
            path = "compiler_ir/" + str(digest) + ".bir"
            if not sha(digest) or files.get(path) != digest or sizes is None or not 0 < sizes.get(path, 0) <= 16 * 1024**2:
                raise ValueError("Neuron compiler BIR snapshot missing or changed")
            artifacts.add(path)
        keys, hashes = row.get("cache_keys"), row.get("stablehlo_sha256")
        if (not isinstance(keys, list) or not isinstance(hashes, list) or len(keys) != len(hashes)
                or len(keys) > 4096 or any(type(k) is not str or not 0 < len(k) <= 4096 for k in keys)
                or len(keys) != len(set(keys)) or any(not sha(h) for h in hashes)
                or bool(keys) is not row["fresh_lowering_observed"]):
            raise ValueError("Inconsistent Neuron compiler lowering observations")
    if sum(sizes[path] for path in artifacts) > 64 * 1024**2:
        raise ValueError("Neuron compiler snapshots exceed their byte envelope")
    status = "partial" if issues or rows and not all(r["fresh_lowering_observed"] for r in rows) else "observed" if rows else "not_measured"
    if value.get("status") != status:
        raise ValueError("Inconsistent Neuron compiler observation status")
    return value


def runtime_neff_bindings(trace_file, directory, get_neff_info):
    """Bind SDK execution keys to cache byte snapshots, without claiming ownership."""
    from ..hardware.neuron_processing import DEFAULT_LIMITS, file_manifest
    from ..hardware.neuron_tables import file_digest

    root, path = Path(directory).resolve(), Path(trace_file)
    value = {"protocol": RUNTIME_NEFF, "trace_file": "framework_trace.json", "trace_sha256": None,
             "device_event_count": None, "bindings": [], "unresolved_events": [], "issues": [],
             "status": "not_measured", "all_selected_keys_resolved": False,
             "scope": RUNTIME_NEFF_SCOPE,
             "loaded_binary_identity_proven": False, "formal_origin_qualified": False}
    try:
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError("Framework trace must be a regular capture artifact")
        path = path.resolve()
        manifest = file_manifest(directory, DEFAULT_LIMITS)
        value["trace_file"] = path.relative_to(root).as_posix()
        value["trace_sha256"] = manifest[value["trace_file"]]["sha256"]
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Framework trace exceeds the 64 MiB envelope")
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024**2 + 1)
        if len(raw) > 64 * 1024**2 or hashlib.sha256(raw).hexdigest() != value["trace_sha256"]:
            raise ValueError("Framework trace changed or exceeded its byte envelope")
        events = json.loads(raw)["traceEvents"]
        if not isinstance(events, list) or len(events) > 200_000:
            raise ValueError("Framework trace lacks a bounded event list")
        groups, selected = {}, []
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("ph") != "X" or event.get("cat") != "privateuse1_driver":
                continue
            args = event.get("args")
            if isinstance(args, dict) and type(args.get("host_worker_idx")) is int and args["host_worker_idx"] >= 0:
                continue
            selected.append(index)
            keys = tuple(args.get(name) for name in ("neff_cache_key", "persistent_cache_key")) if isinstance(args, dict) else ()
            if (len(keys) != 2 or any(not isinstance(key, str) or not 0 < len(key) <= 4096 for key in keys)
                    or "host_worker_idx" in args
                    or type(event.get("ts")) not in (int, float) or not 0 <= event["ts"] < 2**64
                    or type(event.get("dur")) not in (int, float) or not 0 < event["dur"] < 2**64):
                value["unresolved_events"].append(index)
                continue
            groups.setdefault(keys, []).append(index)
        value["device_event_count"] = len(selected)
        if len(groups) > 4096:
            raise ValueError("Runtime NEFF bindings exceed the cache-key envelope")
        known = {row["sha256"]: root / name for name, row in manifest.items() if name.endswith(".neff")}
        total_bytes, total_files = sum(row["bytes"] for row in manifest.values()), len(manifest)
        for (cache_key, persistent_key), indexes in groups.items():
            try:
                result = get_neff_info(cache_key) if callable(get_neff_info) else None
                if (not isinstance(result, tuple) or len(result) != 2 or result[0] != persistent_key
                        or type(result[1]) is not bytes or not 0 < len(result[1]) <= 128 * 1024**2):
                    raise ValueError("Missing, mismatched or oversized current NEFF cache entry")
                binary = result[1]
                digest = hashlib.sha256(binary).hexdigest()
                artifact = known.get(digest)
                if artifact is None:
                    total_bytes, total_files = total_bytes + len(binary), total_files + 1
                    if total_bytes > DEFAULT_LIMITS["max_bytes"] or total_files > DEFAULT_LIMITS["max_files"]:
                        raise ValueError("Runtime NEFF snapshots exceed the existing artifact envelope")
                    artifact = root / "runtime_neff" / (digest + ".neff")
                    artifact.parent.mkdir(exist_ok=True)
                    with artifact.open("xb") as stream:
                        stream.write(binary)
                    known[digest] = artifact
                if artifact.stat().st_size != len(binary) or file_digest(artifact) != digest:
                    raise ValueError("Runtime NEFF snapshot changed during collection")
                value["bindings"].append({"cache_key": cache_key, "persistent_key": persistent_key,
                    "neff_sha256": digest, "neff_bytes": len(binary), "artifact": artifact.relative_to(root).as_posix(),
                    "event_indexes": indexes})
            except Exception as exc:
                value["unresolved_events"].extend(indexes)
                value["issues"].append({"cache_key": cache_key, "error": type(exc).__name__})
        value["unresolved_events"].sort()
        value["all_selected_keys_resolved"] = bool(selected) and not value["unresolved_events"]
        value["status"] = "observed" if value["all_selected_keys_resolved"] else "partial" if value["bindings"] else "not_measured"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        value.update(bindings=[], device_event_count=None, unresolved_events=[], issues=[{"error": str(exc)[:300]}])
    return _seal(value)


def validate_runtime_neff(value, files, sizes):
    if (not isinstance(value, dict) or value.get("protocol") != RUNTIME_NEFF
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
            or value.get("loaded_binary_identity_proven") is not False or value.get("formal_origin_qualified") is not False
            or value.get("scope") != RUNTIME_NEFF_SCOPE or not isinstance(value.get("trace_file"), str)
            or not sha(value.get("trace_sha256"))
            or files.get(value.get("trace_file")) != value.get("trace_sha256")):
        raise ValueError("Neuron runtime/cache snapshot binding is missing or changed")
    entries, unresolved, count = value.get("bindings"), value.get("unresolved_events"), value.get("device_event_count")
    if not isinstance(entries, list) or len(entries) > 4096 or not isinstance(unresolved, list) or len(unresolved) > 200_000:
        raise ValueError("Neuron runtime/cache snapshot list is invalid")
    indexes, keys = list(unresolved), []
    for entry in entries:
        if (not isinstance(entry, dict) or any(not isinstance(entry.get(k), str) or not 0 < len(entry[k]) <= 4096
                                              for k in ("cache_key", "persistent_key"))
                or not sha(entry.get("neff_sha256")) or not integer(entry.get("neff_bytes"), minimum=1)
                or entry["neff_bytes"] > 128 * 1024**2 or not isinstance(entry.get("artifact"), str)
                or not entry["artifact"].endswith(".neff") or files.get(entry["artifact"]) != entry["neff_sha256"]
                or sizes is None or sizes.get(entry["artifact"]) != entry["neff_bytes"]
                or not isinstance(entry.get("event_indexes"), list) or not 1 <= len(entry["event_indexes"]) <= 200_000):
            raise ValueError("Neuron runtime/cache snapshot differs from its NEFF artifact")
        indexes.extend(entry["event_indexes"])
        keys.append((entry["cache_key"], entry["persistent_key"]))
    if (any(not integer(i) or i >= 200_000 for i in indexes) or len(indexes) != len(set(indexes)) or len(keys) != len(set(keys))
            or count is not None and (not integer(count) or count > 200_000 or len(indexes) != count)
            or count is None and (entries or unresolved)):
        raise ValueError("Neuron runtime/cache snapshot event coverage is inconsistent")
    resolved = bool(count) and not unresolved
    status = "observed" if resolved else "partial" if entries else "not_measured"
    if value.get("all_selected_keys_resolved") is not resolved or value.get("status") != status:
        raise ValueError("Neuron runtime/cache snapshot status differs from its observations")
    return value
