"""Frozen Python source identities, without importing a device SDK.

This registry is an instrumentation aid inside a trusted worker, not a Python
sandbox. A matching filename or a newly compiled equal code object is not a
matching registered object.
"""

from __future__ import annotations

import base64
import builtins
import csv
import functools
import hashlib
import importlib.metadata
import inspect
import io
import marshal
import math
import os
import re
import sys
import textwrap
import types
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ACTIVE = ContextVar("kernelbench_submission_sources", default=None)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


class SourceRegistry:
    def __init__(self):
        self.sources = []
        self.texts = {}
        self.codes = {}  # id -> retained object: code equality is insufficient.
        self.jit_text = {}
        self.literals = {}
        self.external_codes = {}

    def register(self, source, path, code):
        item = {"source_sha256": sha256(source.encode()), "path": str(path)}
        self.sources.append(item)
        self.texts[item["source_sha256"]] = source
        lines = source.splitlines(keepends=True)
        literals = self.literals.setdefault(item["source_sha256"], set())
        position = 0

        def visit(node):
            nonlocal position
            self.codes[id(node)] = (node, item, position)
            position += 1
            # Compute the expected JIT text before executing candidate code.
            # Runtime inspect/linecache functions are not source attestations.
            block = "" if node.co_name.startswith("<") else textwrap.dedent("".join(inspect.getblock(lines[node.co_firstlineno - 1:])))
            start = re.search(r"^def\s+\w+\s*\(", block, re.MULTILINE)
            if start:
                self.jit_text[id(node)] = block[start.start():]
            for child in node.co_consts:
                if isinstance(child, types.CodeType):
                    visit(child)
                elif isinstance(child, str):
                    literals.add(child)

        visit(code)

    def source_for(self, function):
        return self.source_for_code(getattr(function, "__code__", None))

    def source_for_code(self, code):
        match = self.codes.get(id(code))
        return match[1] if match and match[0] is code else None

    def callable_identity(self, function):
        """Observe code identity; external code observations are not ownership proofs."""
        if type(function) in (staticmethod, classmethod, types.MethodType):
            function = function.__func__
        if type(function) is types.FunctionType:
            code = function.__code__
            owner = self.codes.get(id(code))
            if owner and owner[0] is code:
                return {"kind": "registered_code", "source_sha256": owner[1]["source_sha256"],
                        "code_index": owner[2], "qualname": code.co_qualname}
            cached = self.external_codes.get(id(code))
            if cached is None or cached[0] is not code:
                def canonical(node):
                    return node.replace(co_filename="", co_firstlineno=0, co_consts=tuple(
                        canonical(value) if isinstance(value, types.CodeType) else value for value in node.co_consts))
                cached = self.external_codes[id(code)] = (code, sha256(marshal.dumps(canonical(code))))
            return {"kind": "external_code_observation", "code_sha256": cached[1], "qualname": code.co_qualname}
        if type(function) in (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType):
            module = getattr(function, "__module__", None) or getattr(getattr(function, "__objclass__", None), "__module__", None)
            return {"kind": "builtin_observation", "module": module, "qualname": function.__qualname__}
        raise ValueError("Unsupported callable descriptor in native state observation")

    def jit_source(self, function):
        """Check the actual JIT text as well as the original Python code object."""
        owner = self.source_for(getattr(function, "fn", None))
        if not owner:
            return None
        expected = self.jit_text.get(id(function.fn.__code__))
        if expected is None or expected != function.src:
            return None
        if sha256(Path(owner["path"]).read_bytes()) != owner["source_sha256"]:
            return None
        return {**owner, "function": function.fn.__qualname__, "jit_source_sha256": sha256(function.src.encode())}

    def intact(self):
        try:
            return all(sha256(Path(row["path"]).read_bytes()) == row["source_sha256"] for row in self.sources)
        except OSError:
            return False


@contextmanager
def record_sources(registry):
    token = _ACTIVE.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE.reset(token)


def compile_source(source, path):
    """Freeze bytes and register the exact code executed by either device loader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL + O_NOFOLLOW avoids silently executing through a reused symlink.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            import stat

            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode) or handle.read(len(source.encode()) + 1) != source.encode():
                raise ValueError("Frozen compiler source file is not the expected regular file")
    else:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source.encode())
    # Only the source's own future imports may change its language semantics.
    code = compile(source, str(path), "exec", dont_inherit=True)
    registry = _ACTIVE.get()
    if registry is not None:
        registry.register(source, path, code)
    return code


def execute(source, context, build_directory):
    """Keep source inspectable for Triton and return definitions in context."""
    path = Path(build_directory).resolve() / ("candidate_" + sha256(source.encode()) + ".py")
    code = compile_source(source, path)
    context["__name__"] = "kernelbench_candidate_" + sha256(str(path).encode())
    context["__file__"] = str(path)
    # inspect.getsource(classes), dataclasses and JIT imports consult sys.modules.
    # Execute in the actual module namespace so imports defined earlier in the
    # same source are already visible to decorators.
    module = types.ModuleType(context["__name__"])
    module.__dict__.update(context)
    previous = sys.modules.get(context["__name__"])
    sys.modules[context["__name__"]] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(context["__name__"], None)
        else:
            sys.modules[context["__name__"]] = previous
        raise
    context.update(module.__dict__)


class WheelSources:
    def __init__(self, name, version, prefixes, *, distribution=None):
        dist = distribution or importlib.metadata.distribution(name)
        if dist.version != version:
            raise RuntimeError("Frontend provenance requires " + name + " " + version)
        record = dist.read_text("RECORD")
        if not record:
            raise RuntimeError(name + " source RECORD is unavailable")
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
            selected = any(relative.startswith(prefix) if prefix.endswith("/") else relative == prefix for prefix in prefixes)
            if not (selected and relative.endswith(".py") and digest.startswith("sha256=")):
                continue
            path = Path(dist.locate_file(relative)).resolve()
            if not path.is_relative_to(root) or path.stat().st_size > 4 * 1024**2:
                raise ValueError("Invalid language primitive path/size")
            content = path.read_bytes()
            expected = base64.b64decode(digest.removeprefix("sha256=") + "=", altchars=b"-_", validate=True).hex()
            if sha256(content) != expected:
                raise ValueError("Installed language source does not match RECORD")
            source = {"path": str(path), "relative_path": relative, "file_sha256": expected,
                      "distribution": name, "version": dist.version, "record_sha256": record_sha}
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
        if not self.accepts(source, code):
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

    def accepts(self, source, code):
        return True

    def intact(self):
        try:
            return all(sha256(Path(row["path"]).read_bytes()) == row["file_sha256"] for row in self.records)
        except OSError:
            return False
