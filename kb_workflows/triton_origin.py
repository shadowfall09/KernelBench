"""Observe actual Triton frontend calls and frozen language-primitive sources.

This is compiler/input provenance within the trusted diagnostic worker. It is
not proof of algorithmic usefulness or resistance to arbitrary interpreter or
native-memory compromise. No SDK import occurs until install() is requested.
"""

from __future__ import annotations

import ast
import base64
import builtins
import csv
import dataclasses
import functools
import importlib.metadata
import inspect
import io
import json
import marshal
import math
import re
import textwrap
import types
from contextvars import ContextVar
from pathlib import Path

from .common import fingerprint
from .submission_source import sha256

PROTOCOL = "kernelbench_pro_triton_frontend_v1"
MAX_CALLS = 20_000
_ACTIVE = ContextVar("kernelbench_triton_frontend", default=None)
_BUILD = ContextVar("kernelbench_triton_build", default=None)
EMISSION_PROTOCOL = "kernelbench_pro_triton_emission_v1"


def ast_digest(tree):
    return sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())


def metadata_fingerprint(metadata):
    """The real KernelMetadata restores GPUTarget as a dataclass, not JSON."""
    def canonical(value):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return canonical(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if type(value) in {str, int, float, bool, type(None)}:
            return value
        raise ValueError("Unsupported opaque compiler metadata")

    return fingerprint(canonical(dict(metadata._asdict())))


def expected_jit(text, code):
    block = textwrap.dedent("".join(inspect.getblock(text.splitlines(keepends=True)[code.co_firstlineno - 1:])))
    match = re.search(r"^def\s+\w+\s*\(", block, re.MULTILINE)
    return block[match.start():] if match else None


class LanguageSources:
    def __init__(self, *, distribution=None):
        dist = distribution or importlib.metadata.distribution("triton")
        if dist.version != "3.7.0":
            raise RuntimeError("Frontend provenance requires Triton 3.7.0")
        record = dist.read_text("RECORD")
        if not record:
            raise RuntimeError("Triton source RECORD is unavailable")
        self.records = []
        self.codes = {}
        self.texts = {}
        self.paths = {}
        record_sha = sha256(record.encode())
        root = Path(dist.locate_file("")).resolve()
        for row in csv.reader(io.StringIO(record)):
            if len(row) != 3:
                continue
            relative, digest, _ = row
            language = relative.startswith(("triton/language/", "triton/experimental/gluon/language/"))
            if not ((language or relative == "triton/__init__.py") and relative.endswith(".py") and digest.startswith("sha256=")):
                continue
            path = Path(dist.locate_file(relative)).resolve()
            if not path.is_relative_to(root) or path.stat().st_size > 4 * 1024**2:
                raise ValueError("Invalid language primitive path/size")
            content = path.read_bytes()
            expected = base64.b64decode(digest.removeprefix("sha256=") + "=", altchars=b"-_", validate=True).hex()
            if sha256(content) != expected:
                raise ValueError("Installed language source does not match RECORD")
            source = {"path": str(path), "relative_path": relative, "file_sha256": expected,
                      "distribution": "triton", "version": dist.version, "record_sha256": record_sha}
            self.records.append(source)
            text = content.decode()
            self.paths[str(path)] = source
            self.texts[str(path)] = text
            collected = []

            def visit(code):
                collected.append(code)
                for child in code.co_consts:
                    if isinstance(child, types.CodeType):
                        visit(child)

            visit(compile(text, str(path), "exec", dont_inherit=True))
            self.codes[str(path)] = collected
        self.builtins = {id(value): value for value in [getattr(builtins, name) for name in
            ("bool", "float", "int", "len", "list", "max", "min", "print", "range", "tuple", "getattr", "isinstance", "enumerate", "zip")]
            + [value for value in vars(math).values() if isinstance(value, types.BuiltinFunctionType)]}

    def python_sources(self, fn, seen=None):
        """Validate actual bytecode and callable closure cells, never __wrapped__."""
        seen = set() if seen is None else seen
        if id(fn) in seen:
            return []
        seen.add(id(fn))
        if id(fn) in self.builtins and self.builtins[id(fn)] is fn:
            return [{"role": "builtin", "module": fn.__module__, "function": fn.__qualname__}]
        if isinstance(fn, functools.partial):
            sources = self.python_sources(fn.func, seen)
            if sources is None:
                return None
            for value in (*fn.args, *fn.keywords.values()):
                if callable(value):
                    child = self.python_sources(value, seen)
                    if child is None:
                        return None
                    sources.extend(child)
            return sources
        if isinstance(fn, types.MethodType):
            fn = fn.__func__
        code = getattr(fn, "__code__", None)
        if code is None:
            return None
        source = self.paths.get(code.co_filename)
        if source is None or not any(code == known for known in self.codes[code.co_filename]):
            return None
        if source["relative_path"] == "triton/__init__.py" and code.co_name not in {"cdiv", "next_power_of_2"}:
            return None
        sources = [{**source, "role": "primitive", "function": fn.__qualname__, "code_sha256": sha256(marshal.dumps(code))}]
        for cell in fn.__closure__ or ():
            value = cell.cell_contents
            if callable(value):
                child = self.python_sources(value, seen)
                if child is None:
                    return None
                sources.extend(child)
        return sources

    def python_function(self, fn):
        return self.python_sources(fn) is not None

    def jit(self, fn):
        function = getattr(fn, "fn", None)
        if not self.python_function(function):
            return None
        code = function.__code__
        text = expected_jit(self.texts[code.co_filename], code)
        if text is None or text != fn.src:
            return None
        return {**self.paths[code.co_filename], "function": function.__qualname__, "role": "primitive",
                "jit_source_sha256": sha256(text.encode()), "ast_sha256": ast_digest(ast.parse(text))}

    def intact(self):
        try:
            return all(sha256(Path(row["path"]).read_bytes()) == row["file_sha256"] for row in self.records)
        except OSError:
            return False


class Frontend:
    def __init__(self, registry, *, language=None):
        self.registry = registry
        self.language = language if language is not None else LanguageSources()
        self.patches = []
        self.sources = {}  # Retain real ASTSource instances, not names/cache keys.
        self.compiled = {}

    def emitted(self, binary, metadata, options):
        phase = _BUILD.get()
        if phase is None:
            return
        if (not isinstance(binary, bytes) or not binary or metadata.get("ir_override")
                or getattr(options, "ir_override", None)):
            phase["issues"].append("Missing backend bytes or unsupported IR override")
            return
        phase["emitted_sha256"] = sha256(binary)

    def freeze_compiled(self, kernel):
        phase = _BUILD.get()
        if phase is None or kernel.src is not phase["source"]:
            return
        if (phase["issues"] or not phase.get("emitted_sha256") or not isinstance(kernel.kernel, bytes)
                or sha256(kernel.kernel) != phase["emitted_sha256"]):
            return
        frontend = self.proof_for(kernel.src)
        if frontend is None:
            return
        frozen_frontend = json.loads(json.dumps(frontend))
        document = {"protocol": EMISSION_PROTOCOL, "backend": "cuda", "binary_sha256": phase["emitted_sha256"],
                    "metadata_sha256": metadata_fingerprint(kernel.metadata),
                    "frontend_receipt_sha256": frozen_frontend["receipt_sha256"]}
        self.compiled[id(kernel)] = (kernel, kernel.src, document, frozen_frontend)

    def compiled_for(self, kernel):
        item = self.compiled.get(id(kernel))
        if (item is None or item[0] is not kernel or item[1] is not kernel.src
                or not isinstance(kernel.kernel, bytes) or sha256(kernel.kernel) != item[2]["binary_sha256"]
                or metadata_fingerprint(kernel.metadata) != item[2]["metadata_sha256"]):
            return None
        return item[2]

    def compiled_frontend_for(self, kernel):
        if self.compiled_for(kernel) is None:
            return None
        return json.loads(json.dumps(self.compiled[id(kernel)][3]))

    def describe(self, fn):
        owner = self.registry.jit_source(fn)
        if owner:
            return {**owner, "role": "candidate", "ast_sha256": ast_digest(ast.parse(fn.src))}
        return self.language.jit(fn)

    def parse(self, fn, tree):
        phase = _ACTIVE.get()
        if phase is None:
            return
        owner = self.describe(fn)
        if owner is None or owner["ast_sha256"] != ast_digest(tree):
            phase["issues"].append("Actual parsed AST has no matching frozen candidate/primitive source")
            return
        phase["parsed"][id(fn)] = (fn, owner)

    def call(self, fn, *, jit=False):
        phase = _ACTIVE.get()
        if phase is None:
            return
        if len(phase["calls"]) >= MAX_CALLS:
            phase["issues"].append("Frontend call limit exceeded")
            return
        if jit:
            owner = self.describe(fn)
            phase["calls"].append({"kind": "jit", "function_id": id(fn), "owner": owner})
            if owner is None:
                phase["issues"].append("JIT callee is outside frozen submission and approved language primitives")
        else:
            sources = self.language.python_sources(fn)
            phase["calls"].append({"kind": "python", "primitive": sources is not None, "sources": sources,
                                   "function_name": getattr(fn, "__qualname__", type(fn).__name__),
                                   "module": getattr(fn, "__module__", None)})
            if sources is None:
                phase["issues"].append("Unreviewed Python/compiler builtin or callable closure")

    def phase(self, source, function):
        phase = {"root": function, "calls": [], "parsed": {}, "issues": [], "complete": False}
        token = _ACTIVE.set(phase)
        self.sources[id(source)] = (source, phase)
        return token, phase

    def finish(self, token, phase, *, complete):
        phase["complete"] = complete
        _ACTIVE.reset(token)

    def proof_for(self, source):
        item = self.sources.get(id(source))
        if item is None or item[0] is not source:
            return None
        phase = item[1]
        root = self.describe(phase["root"])
        # Actual root parsing must have been observed. A cached binary or an
        # overridden make_ir/parse method cannot supply a new closure receipt.
        if (not phase["complete"] or phase["issues"] or root is None or root.get("role") != "candidate"
                or id(phase["root"]) not in phase["parsed"]):
            return None
        calls = []
        for row in phase["calls"]:
            if row["kind"] == "jit":
                if row["owner"] is None or row["owner"] not in [owner for _, owner in phase["parsed"].values()]:
                    return None
                calls.append({"kind": "jit", "owner": row["owner"]})
            else:
                calls.append(row)
        document = {"protocol": PROTOCOL, "root": root, "calls": calls,
                    "parsed_sources": [owner for _, owner in phase["parsed"].values()],
                    "complete": True, "language_manifest": {"sources": self.language.records},
                    "language_manifest_sha256": fingerprint({"sources": self.language.records})}
        document["receipt_sha256"] = fingerprint(document)
        return document

    @staticmethod
    def validate(document, *, source_sha256, jit_source_sha256):
        try:
            return Frontend._validate(document, source_sha256=source_sha256, jit_source_sha256=jit_source_sha256)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _validate(document, *, source_sha256, jit_source_sha256):
        if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL or document.get("complete") is not True
                or document.get("root", {}).get("source_sha256") != source_sha256
                or document["root"].get("jit_source_sha256") != jit_source_sha256
                or document["root"].get("role") != "candidate"
                or document.get("receipt_sha256") != fingerprint({k: v for k, v in document.items() if k != "receipt_sha256"})):
            return False
        parsed = document.get("parsed_sources")
        calls = document.get("calls")
        manifest = document.get("language_manifest")
        if not isinstance(manifest, dict) or document.get("language_manifest_sha256") != fingerprint(manifest):
            return False
        sources = {row["path"]: row for row in manifest.get("sources", [])}
        if not isinstance(parsed, list) or document["root"] not in parsed or not isinstance(calls, list) or len(calls) > MAX_CALLS:
            return False
        for row in [*parsed, *(item.get("owner") for item in calls if item.get("kind") == "jit")]:
            if (not isinstance(row, dict) or row.get("role") not in {"candidate", "primitive"}
                    or any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key))) for key in ("jit_source_sha256", "ast_sha256"))):
                return False
            if row["role"] == "candidate" and row.get("source_sha256") != source_sha256:
                return False
            if row["role"] == "primitive" and (row.get("distribution") != "triton" or row.get("version") != "3.7.0"
                    or any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key))) for key in ("file_sha256", "record_sha256"))):
                return False
            if row["role"] == "primitive" and any(row.get(key) != sources.get(row.get("path"), {}).get(key)
                                                  for key in ("file_sha256", "record_sha256", "relative_path")):
                return False
        for item in calls:
            if item.get("kind") == "jit":
                if item.get("owner") not in parsed:
                    return False
                continue
            if item.get("kind") != "python" or item.get("primitive") is not True or not isinstance(item.get("sources"), list):
                return False
            for row in item["sources"]:
                if row.get("role") == "builtin":
                    if row.get("module") not in {"builtins", "math"} or not row.get("function"):
                        return False
                elif (row.get("role") != "primitive" or row.get("path") not in sources
                        or any(row.get(key) != sources[row["path"]].get(key) for key in ("file_sha256", "record_sha256"))
                        or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("code_sha256")))):
                    return False
        return True

    def patch(self, obj, name, value):
        self.patches.append((obj, name, getattr(obj, name), value))
        setattr(obj, name, value)

    def install(self):
        import triton
        import triton.compiler as compiler_package
        import triton.compiler.compiler as compiler_module
        from triton import knobs
        from triton.backends.nvidia.compiler import CUDABackend
        from triton.compiler.code_generator import BoundJITMethod, CodeGenerator
        from triton.compiler.compiler import ASTSource, CompiledKernel
        from triton.runtime.jit import BoundConstexprFunction, ConstexprFunction, JITCallable, JITFunction

        make_ir, parse = ASTSource.make_ir, JITCallable.parse
        call_jit, call_function = CodeGenerator.call_JitFunction, CodeGenerator.call_Function
        compile_, make_cubin, constructor = compiler_module.compile, CUDABackend.make_cubin, CompiledKernel.__init__

        @functools.wraps(compile_)
        def observed_compile(source, *args, **kwargs):
            phase = {"source": source, "issues": [], "emitted_sha256": None}
            if (not isinstance(source, ASTSource) or knobs.compilation.override
                    or (kwargs.get("options") or {}).get("ir_override")
                    or knobs.runtime.add_stages_inspection_hook is not None):
                phase["issues"].append("Unsupported source, override or compilation-stage hook")
            token = _BUILD.set(phase)
            try:
                return compile_(source, *args, **kwargs)
            finally:
                _BUILD.reset(token)

        @functools.wraps(make_cubin)
        def observed_cubin(backend, source, metadata, options, capability):
            binary = make_cubin(backend, source, metadata, options, capability)
            self.emitted(binary, metadata, options)
            return binary

        @functools.wraps(constructor)
        def observed_constructor(kernel, *args, **kwargs):
            constructor(kernel, *args, **kwargs)
            try:
                self.freeze_compiled(kernel)
            except Exception as exc:
                phase = _BUILD.get()
                if phase is not None:
                    phase["issues"].append("Unclassified compiler constructor metadata: " + type(exc).__name__)

        @functools.wraps(make_ir)
        def observed_ir(source, *args, **kwargs):
            token, phase = self.phase(source, source.fn)
            try:
                result = make_ir(source, *args, **kwargs)
            except BaseException:
                self.finish(token, phase, complete=False)
                raise
            self.finish(token, phase, complete=True)
            return result

        @functools.wraps(parse)
        def observed_parse(fn):
            tree = parse(fn)
            self.parse(fn, tree)
            return tree

        @functools.wraps(call_jit)
        def observed_jit(generator, fn, *args, **kwargs):
            self.call(fn, jit=True)
            return call_jit(generator, fn, *args, **kwargs)

        @functools.wraps(call_function)
        def observed_function(generator, node, fn, *args, **kwargs):
            actual = fn.__func__ if type(fn) in {BoundJITMethod, BoundConstexprFunction} else fn
            if type(actual) is ConstexprFunction:
                actual = actual.fn
            if not isinstance(actual, JITFunction):
                self.call(actual)
            return call_function(generator, node, fn, *args, **kwargs)

        self.patch(ASTSource, "make_ir", observed_ir)
        self.patch(JITCallable, "parse", observed_parse)
        self.patch(CodeGenerator, "call_JitFunction", observed_jit)
        self.patch(CodeGenerator, "call_Function", observed_function)
        self.patch(compiler_module, "compile", observed_compile)
        self.patch(compiler_package, "compile", observed_compile)
        self.patch(triton, "compile", observed_compile)
        self.patch(CUDABackend, "make_cubin", observed_cubin)
        self.patch(CompiledKernel, "__init__", observed_constructor)

    def intact(self):
        return all(getattr(obj, name) is value for obj, name, _, value in self.patches) and self.language.intact()

    def restore(self):
        for obj, name, original, _ in reversed(self.patches):
            setattr(obj, name, original)
