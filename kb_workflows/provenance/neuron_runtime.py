"""Opt-in development NRT load/execution observation; no device work on import."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
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
COMPILER = "kernelbench_pro_neuron_compiler_job_v1"


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
    if not (settings.get("device_backend") == "neuron" and payload.get("task") == "profile"
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


def _journal(rows):
    """Replay successful model generations; an async schedule is not completion."""
    if (not isinstance(rows, list) or not 2 <= len(rows) <= 200002 or any(not isinstance(row, dict) for row in rows)
            or rows[0].get("type") != "start" or rows[0].get("protocol") != 1
            or not integer(rows[0].get("pid"), minimum=1)
            or rows[-1] != {"type": "finish", "errors": 0}):
        raise ValueError("NRT journal is incomplete, unsupported or reports collection errors")
    models, loads, executions, load_ids, compiler_ids = {}, {}, {}, set(), set()
    for row in rows[1:-1]:
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
                       "compiler_jobs": jobs,
                       "loaded_binary_identity_proven": bool(events) and not unresolved and self.build["simulated"] is False,
                       "status": "observed" if events and not unresolved else "partial" if bindings else "not_measured"})
        return validate(result, files, sizes, allow_simulated=self.build["simulated"] is True)
