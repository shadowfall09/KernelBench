"""Observe source-owned Torch CUDA extension builds and their device objects.

An arbitrary loaded .so is not custom provenance. This adapter observes the
supported public loader, literal CUDA inputs, the actual Ninja build, fixed
compiler tools and CUDA object cubins. It grants credit only after a separate
observed CUDA module/function/launch join. Opaque runtime registration and
unsupported compiler/header/link configurations remain unknown.
"""

from __future__ import annotations

import base64
import csv
import functools
import hashlib
import importlib.metadata
import inspect
import io
import os
import re
import shlex
import shutil
import threading
import time
from pathlib import Path

from .common import fingerprint
from .origin_binaries import PROTOCOL as BINARY_PROTOCOL
from .origin_binaries import Catalog, file_digest
from .submission_source import sha256

PROTOCOL = "kernelbench_pro_cuda_extension_v1"
MAX_SOURCE_BYTES = 8 * 1024**2
ENV_KEYS = ("PATH", "CXX", "CC", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH", "PYTORCH_NVCC", "LD_PRELOAD",
            "LD_LIBRARY_PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_CUDA_ARCH_LIST",
            "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES")
HEADERS = {"torch/extension.h", "torch/types.h", "ATen/ATen.h", "cuda.h", "cuda_runtime.h",
           "cuda_runtime_api.h", "cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h", "device_launch_parameters.h",
           "math.h", "cmath", "stdint.h", "cstdint", "stddef.h", "cstddef", "limits", "float.h", "cfloat",
           "vector", "array", "algorithm", "iostream"}
FLAG = re.compile(r"(?:-O[0-3]|-g|-lineinfo|--use_fast_math|--expt-relaxed-constexpr|--expt-extended-lambda|-std=c\+\+(?:14|17|20))")
LINK_FLAGS = {"-lcuda", "-lcudart", "-lcublas", "-lcudnn"}


def strings(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError("Compiler sources must be explicit text or a list of text")
    if sum(len(item.encode()) for item in value) > MAX_SOURCE_BYTES:
        raise ValueError("Compiler input exceeded its text bound")
    return list(value)


def cuda_policy(text):
    if "__global__" not in text:
        return False
    # Preserve literal inputs rather than trusting declarations. This conservative
    # first adapter excludes external implementation headers, preprocessor include
    # tricks and assembly file embedding; unsupported inputs remain unclassified.
    cleaned = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", text)
    if "##" in cleaned or re.search(r"\.(?:incbin|include)\b|#\s*(?:embed|include_next|line)\b", cleaned):
        return False
    for line in cleaned.splitlines():
        if re.match(r"\s*#\s*include\b", line):
            match = re.fullmatch(r'\s*#\s*include\s*[<"]([^>"\n]+)[>"]\s*', line)
            if match is None or match[1] not in HEADERS:
                return False
    return True


def _environment():
    # Never serialize environment values: record only an equality digest.
    return {name: os.environ.get(name) for name in ENV_KEYS}


def _tool(command):
    parts = shlex.split(command)
    if len(parts) != 1:
        raise ValueError("Compiler wrappers need a separate reviewed adapter")
    resolved = shutil.which(parts[0])
    if resolved is None:
        raise ValueError("Compiler tool is unavailable")
    path = str(Path(resolved).resolve())
    digest, size = file_digest(path)
    return {"path": path, "sha256": digest, "bytes": size}


def adapter_source(extension):
    dist = importlib.metadata.distribution("torch")
    record = dist.read_text("RECORD")
    if not record:
        raise RuntimeError("Torch compiler adapter source RECORD is unavailable")
    for row in csv.reader(io.StringIO(record)):
        if len(row) != 3 or row[0] != "torch/utils/cpp_extension.py" or not row[1].startswith("sha256="):
            continue
        path = Path(dist.locate_file(row[0])).resolve()
        expected = base64.b64decode(row[1].removeprefix("sha256=") + "=", altchars=b"-_", validate=True).hex()
        if path != Path(extension.__file__).resolve() or file_digest(path, limit=MAX_SOURCE_BYTES)[0] != expected:
            raise RuntimeError("Actual Torch compiler adapter differs from its installed source record")
        return {"distribution": "torch", "version": "2.12.0", "path": str(path),
                "file_sha256": expected, "record_sha256": sha256(record.encode())}
    raise RuntimeError("Torch compiler adapter is absent from the installed source record")


class Extensions:
    def __init__(self, registry, *, extension=None, version=None, extractor=None, extraction_command="cuobjdump"):
        self.registry = registry
        self.extension = extension
        self.version = version
        self.extractor = extractor
        self.extraction_command = extraction_command
        self.simulated = extension is not None or extractor is not None
        self.patches = []
        self.active = {}
        self.builds = []
        self.environment = None
        self.tools = None
        self.binary_catalog = None
        self.adapter_source = None

    def prepare(self):
        self.environment = _environment()
        try:
            cxx = self.extension.get_cxx_compiler()
            nvcc = Path(self.extension.CUDA_HOME) / "bin/nvcc"
            self.tools = {"cxx": _tool(cxx), "nvcc": _tool(str(nvcc)), "ninja": _tool("ninja")}
        except (OSError, TypeError, ValueError):
            self.tools = None

    def _public(self, api, bound, caller):
        owner = self.registry.source_for_code(caller.f_code)
        if owner is None or self.environment is None or self.tools is None:
            return None
        values = bound.arguments
        if (values.get("extra_include_paths") or any(flag not in LINK_FLAGS for flag in values.get("extra_ldflags") or []) or values.get("with_sycl")
                or values.get("sycl_sources") or values.get("is_standalone")
                or values.get("keep_intermediates", True) is not True):
            return None
        for key in ("extra_cflags", "extra_cuda_cflags"):
            if any(not FLAG.fullmatch(flag) for flag in values.get(key) or []):
                return None
        known = self.registry.literals.get(owner["source_sha256"], set())
        if api == "load_inline":
            cuda = strings(values.get("cuda_sources") or [])
            if not cuda or any(piece not in known for piece in cuda) or not cuda_policy("\n".join(cuda)):
                return None
            expected = "\n".join(([] if values.get("no_implicit_headers") else
                ["#include <torch/types.h>", "#include <cuda.h>", "#include <cuda_runtime.h>"]) + cuda)
            cuda_inputs = {sha256(expected.encode()): [sha256(piece.encode()) for piece in cuda]}
        else:
            cuda_inputs = {}
            for name in strings(values.get("sources") or []):
                if not name.endswith((".cu", ".cuh")):
                    continue
                content = Path(name).read_text()
                if content not in known or not cuda_policy(content):
                    return None
                cuda_inputs[sha256(content.encode())] = [sha256(content.encode())]
            if not cuda_inputs:
                return None
        return {"protocol": PROTOCOL, "source_sha256": owner["source_sha256"], "source_path": owner["path"],
                "public_api": api, "cuda_inputs": cuda_inputs, "objects": [], "build_completed": False,
                "issues": [], "tools": self.tools, "compiler_environment_sha256": fingerprint(self.environment),
                "adapter_source": self.adapter_source,
                "simulated": self.simulated}

    def _write(self, values, frame):
        path = Path(values["path"]).resolve()
        root = Path(frame["source_path"]).parent
        if (not path.is_relative_to(root) or not values.get("with_cuda") or values.get("with_sycl")
                or values.get("cuda_dlink_post_cflags") or values.get("sycl_dlink_post_cflags")):
            frame["issues"].append("Unsupported build path, device linking or backend")
            return
        sources, objects = values["sources"], values["objects"]
        if len(sources) != len(objects):
            frame["issues"].append("Source/object population differs")
            return
        pending = []
        for source, object_ in zip(sources, objects, strict=True):
            source_path = Path(source).resolve()
            if not source_path.is_relative_to(root):
                frame["issues"].append("Compiler source outside frozen build root")
                return
            digest, size = file_digest(source_path, limit=MAX_SOURCE_BYTES)
            if source_path.suffix in {".cu", ".cuh"}:
                if digest not in frame["cuda_inputs"]:
                    frame["issues"].append("Actual CUDA source differs from frozen literal inputs")
                    return
                output = Path(object_)
                output = (output if output.is_absolute() else path.parent / output).resolve()
                if not output.is_relative_to(root):
                    frame["issues"].append("CUDA object outside build root")
                    return
                pending.append({"source_path": str(source_path), "source_file_sha256": digest,
                    "source_bytes": size, "object_path": str(output), "literal_sha256": frame["cuda_inputs"][digest]})
        if not pending:
            frame["issues"].append("No source-owned CUDA compilation units")
        frame.update(ninja_path=str(path), objects=pending)

    def _run(self, directory, frame):
        path = Path(directory).resolve() / "build.ninja"
        if str(path) != frame.get("ninja_path") or _environment() != self.environment:
            raise ValueError("Build request or compiler environment changed")
        text = path.read_text()
        commands = dict(re.findall(r"^(cxx|nvcc)\s*=\s*(.+)$", text, flags=re.MULTILINE))
        if set(commands) != {"cxx", "nvcc"}:
            raise ValueError("Build has no unambiguous compiler assignments")
        for name in commands:
            if _tool(commands[name]) != self.tools[name]:
                raise ValueError("Ninja command differs from the frozen compiler tool")
        if _tool("ninja") != self.tools["ninja"]:
            raise ValueError("Ninja executable changed")
        frame["ninja_sha256"] = file_digest(path, limit=MAX_SOURCE_BYTES)[0]
        return frame["ninja_sha256"]

    def _after(self, frame, before):
        if file_digest(frame["ninja_path"], limit=MAX_SOURCE_BYTES)[0] != before or _environment() != self.environment:
            raise ValueError("Build rules or compiler environment changed during compilation")
        for name, tool in self.tools.items():
            if file_digest(tool["path"]) != (tool["sha256"], tool["bytes"]):
                raise ValueError(f"Compiler tool changed: {name}")
        for item in frame["objects"]:
            if file_digest(item["source_path"], limit=MAX_SOURCE_BYTES) != (item["source_file_sha256"], item["source_bytes"]):
                raise ValueError("CUDA source changed during compilation")
            digest, size = file_digest(item["object_path"])
            item.update(object_sha256=digest, object_bytes=size)
        frame["build_completed"] = True

    def patch(self, name, value):
        self.patches.append((name, getattr(self.extension, name), value))
        setattr(self.extension, name, value)

    def install(self):
        if self.extension is None:
            import torch.utils.cpp_extension as extension

            self.extension = extension
        version = self.version or importlib.metadata.version("torch").split("+")[0]
        if version != "2.12.0":
            raise RuntimeError("CUDA extension observer requires Torch 2.12.0")
        self.adapter_source = ({"distribution": "torch", "version": version, "path": "synthetic",
                                "file_sha256": "0" * 64, "record_sha256": "0" * 64}
                               if self.simulated else adapter_source(self.extension))
        # Fix extraction executable identity before candidate imports. No tool is
        # executed until collect(), and collection remains outside scored timing.
        empty = {"protocol": BINARY_PROTOCOL, "sources": [], "manifest_sha256": fingerprint({"sources": []})}
        self.binary_catalog = Catalog(frozen=empty, command=self.extraction_command, runner=self.extractor)
        for api in ("load_inline", "load"):
            original = getattr(self.extension, api)
            signature = inspect.signature(original)

            def wrapper(*args, _api=api, _original=original, _signature=signature, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                bound.apply_defaults()
                frame = None
                try:
                    frame = self._public(_api, bound, inspect.currentframe().f_back)
                except (OSError, TypeError, ValueError):
                    pass  # Unsupported instrumentation must preserve native loading.
                if frame is None:
                    return _original(*args, **kwargs)
                tid = threading.get_ident()
                previous = self.active.get(tid)
                self.active[tid] = frame
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.builds.append(frame)
                    if previous is None:
                        self.active.pop(tid, None)
                    else:
                        self.active[tid] = previous

            self.patch(api, functools.wraps(original)(wrapper))
        writer, runner = self.extension._write_ninja_file, self.extension._run_ninja_build
        write_signature, run_signature = inspect.signature(writer), inspect.signature(runner)

        @functools.wraps(writer)
        def observed_write(*args, **kwargs):
            result = writer(*args, **kwargs)
            frame = self.active.get(threading.get_ident())
            if frame is not None:
                try:
                    values = write_signature.bind(*args, **kwargs)
                    values.apply_defaults()
                    self._write(values.arguments, frame)
                except (OSError, TypeError, ValueError) as exc:
                    frame["issues"].append(str(exc))
            return result

        @functools.wraps(runner)
        def observed_run(*args, **kwargs):
            frame = self.active.get(threading.get_ident())
            before = None
            if frame is not None and not frame["issues"]:
                try:
                    values = run_signature.bind(*args, **kwargs)
                    values.apply_defaults()
                    before = self._run(values.arguments["build_directory"], frame)
                except (OSError, TypeError, ValueError) as exc:
                    frame["issues"].append(str(exc))
            result = runner(*args, **kwargs)
            if before is not None:
                try:
                    self._after(frame, before)
                except (OSError, TypeError, ValueError) as exc:
                    frame["issues"].append(str(exc))
            return result

        self.patch("_write_ninja_file", observed_write)
        self.patch("_run_ninja_build", observed_run)

    def collect(self, *, timeout=30):
        result = []
        deadline = time.monotonic() + timeout
        for frame in self.builds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not frame["build_completed"] or frame["issues"] or not frame["objects"]:
                continue
            try:
                if file_digest(frame["source_path"], limit=MAX_SOURCE_BYTES)[0] != frame["source_sha256"]:
                    continue
                if any(file_digest(row["source_path"], limit=MAX_SOURCE_BYTES) !=
                       (row["source_file_sha256"], row["source_bytes"]) for row in frame["objects"]):
                    continue
            except (OSError, ValueError):
                continue
            # Reuse bounded CUDA ELF extraction without putting candidate objects
            # into the approved installed-wheel/library ownership namespace.
            sources = [{"id": index + 1, "path": item["object_path"], "file_sha256": item["object_sha256"],
                        "file_bytes": item["object_bytes"]} for index, item in enumerate(frame["objects"])]
            frozen = {"protocol": BINARY_PROTOCOL, "sources": sources, "manifest_sha256": fingerprint({"sources": sources})}
            catalog = Catalog(frozen=frozen, command=self.binary_catalog.command or "cuobjdump", runner=self.extractor)
            if catalog.tool != self.binary_catalog.tool:
                continue
            record = {**frame, "extraction": catalog.collect(paths=[row["path"] for row in sources], timeout=remaining)}
            record["receipt_sha256"] = fingerprint(record)
            result.append(record)
        return result

    @staticmethod
    def validate(document, *, candidate_sha256, binary_sha256, allow_simulated=False):
        try:
            return Extensions._validate(document, candidate_sha256=candidate_sha256,
                                        binary_sha256=binary_sha256, allow_simulated=allow_simulated)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _validate(document, *, candidate_sha256, binary_sha256, allow_simulated=False):
        if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
                or document.get("source_sha256") != candidate_sha256
                or document.get("public_api") not in {"load_inline", "load"}
                or document.get("build_completed") is not True or document.get("issues") != []
                or (document.get("simulated") is not False and not allow_simulated)
                or document.get("receipt_sha256") != fingerprint({k: v for k, v in document.items() if k != "receipt_sha256"})
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(document.get(key))) for key in ("ninja_sha256", "compiler_environment_sha256"))):
            return False
        tools = document.get("tools", {})
        adapter = document.get("adapter_source", {})
        if (adapter.get("distribution") != "torch" or adapter.get("version") != "2.12.0"
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(adapter.get(key))) for key in ("file_sha256", "record_sha256"))):
            return False
        if set(tools) != {"cxx", "nvcc", "ninja"} or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) for row in tools.values()):
            return False
        objects = document.get("objects")
        inputs = document.get("cuda_inputs")
        if not isinstance(objects, list) or not objects or not isinstance(inputs, dict) or not inputs:
            return False
        for row in objects:
            if (row.get("source_file_sha256") not in inputs or row.get("literal_sha256") != inputs[row["source_file_sha256"]]
                    or not row["literal_sha256"] or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in row["literal_sha256"])
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("object_sha256")))):
                return False
        extracted = document.get("extraction", {})
        expected = [{"id": index + 1, "path": row["object_path"], "file_sha256": row["object_sha256"], "file_bytes": row["object_bytes"]}
                    for index, row in enumerate(objects)]
        frozen = extracted.get("manifest", {})
        if (extracted.get("protocol") != BINARY_PROTOCOL or frozen.get("protocol") != BINARY_PROTOCOL
                or frozen.get("sources") != expected or frozen.get("manifest_sha256") != fingerprint({"sources": expected})
                or (extracted.get("simulated") is not False and not allow_simulated)
                or extracted.get("receipt_sha256") != fingerprint({k: v for k, v in extracted.items() if k != "receipt_sha256"})
                or not re.fullmatch(r"[0-9a-f]{64}", str(extracted.get("tool", {}).get("sha256")))):
            return False
        complete = set()
        for row in extracted.get("sources", []):
            if row.get("source") in complete:
                return False
            report = row.get("extraction", {}) or {}
            log = base64.b64decode(report.get("log_base64", ""), validate=True)
            if (row.get("status") == "measured" and report.get("exit_code") == 0 and report.get("timed_out") is False
                    and len(log) <= 4 * 1024**2 and hashlib.sha256(log).hexdigest() == report.get("log_sha256")):
                complete.add(row["source"])
        return any(row.get("source") in complete and row.get("source") in range(1, len(objects) + 1)
                   and row.get("binary_sha256") == binary_sha256 for row in extracted.get("cubins", []))

    def intact(self):
        unchanged = not self.active and all(getattr(self.extension, name) is value for name, _, value in self.patches)
        if unchanged and not self.simulated and self.adapter_source is not None:
            try:
                unchanged = file_digest(self.adapter_source["path"], limit=MAX_SOURCE_BYTES)[0] == self.adapter_source["file_sha256"]
            except (OSError, ValueError):
                unchanged = False
        return unchanged

    def restore(self):
        for name, original, _ in reversed(self.patches):
            setattr(self.extension, name, original)
