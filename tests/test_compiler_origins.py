"""Source/compiler provenance on CPU; no device, nvcc, ptxas or cuobjdump runs."""

from __future__ import annotations

import ast
import base64
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from kb_workflows.common import fingerprint
from kb_workflows.cpp_origin import Extensions, cuda_policy
from kb_workflows.submission_source import SourceRegistry, execute, record_sources, sha256
from kb_workflows.triton_origin import Frontend, LanguageSources

CUDA = '#include <cuda_runtime.h>\n__global__ void owned(float *x) { x[0] += 1; }\n'
CUBIN = b'\x7fELF\x02\x01' + b'\x00' * 12 + (190).to_bytes(2, 'little') + b'\x00' * 44 + b'synthetic-device-object'


def frontend_ir(frontend, context, *, signature=None):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import ASTSource

    target = GPUTarget("cuda", 80, 32)
    backend = CUDABackend(target)
    options = backend.parse_options({})
    mlir = ir.context()
    ir.load_dialects(mlir)
    backend.load_dialects(mlir)
    source = ASTSource(context["kernel"], signature=signature or {"X": "*fp32", "Y": "*fp32"})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), mlir)
    assert module.verify()
    return source, frontend.proof_for(source)


@pytest.fixture
def real_frontend():
    pytest.importorskip("triton")
    registry = SourceRegistry()
    frontend = Frontend(registry)
    frontend.install()
    try:
        yield registry, frontend
    finally:
        frontend.restore()


@pytest.mark.parametrize("body", ["x + 1", "tl.exp(x)", "helper(x)", "tl.sum(tl.full((16,), 1, tl.float32), 0) + x"])
def test_real_frontend_records_ast_jit_and_primitive_calls_without_tool_or_device(tmp_path, real_frontend, monkeypatch, body):
    registry, frontend = real_frontend
    source = ("import triton\nimport triton.language as tl\n@triton.jit\ndef helper(x):\n    return x + 2\n"
              f"@triton.jit\ndef kernel(X, Y):\n    x = tl.load(X)\n    tl.store(Y, {body})\n")
    context = {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    # Actual frontend/MLIR library only; the test fails on any attempt to spawn
    # a compiler/tool. No driver.active or kernel launcher is used.
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("frontend spawned a tool"))
    src, document = frontend_ir(frontend, context)
    assert document is not None, [row for row in frontend.sources[id(src)][1]["calls"] if row.get("primitive") is False]
    assert Frontend.validate(document, source_sha256=sha256(source.encode()),
                             jit_source_sha256=sha256(context["kernel"].src.encode()))
    assert document["complete"] is True and frontend.intact()
    assert document["root"] in document["parsed_sources"]
    assert json.loads(json.dumps(document)) == document


@pytest.mark.parametrize("spoof_builtin", [False, True])
def test_real_frontend_rejects_outer_candidate_delegating_to_unowned_jit(tmp_path, real_frontend, spoof_builtin):
    registry, frontend = real_frontend
    foreign = {}
    execute("import triton\n@triton.jit\ndef helper(x):\n    return x + 7\n", foreign, tmp_path / "foreign")
    if spoof_builtin:
        foreign["helper"].__triton_builtin__ = True
    source = ("import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(X,Y):\n"
              "    x=tl.load(X)\n    tl.store(Y,helper(x))\n")
    context = {"helper": foreign["helper"]}
    with record_sources(registry):
        execute(source, context, tmp_path / "owned")
    src, document = frontend_ir(frontend, context)
    assert document is None
    assert any("JIT callee" in issue for issue in frontend.sources[id(src)][1]["issues"])


def test_frozen_jit_text_ignores_runtime_inspect_and_linecache_forgery(tmp_path, monkeypatch):
    registry, context = SourceRegistry(), {}
    source = "def kernel(x):\n    return x + 1\n"
    with record_sources(registry):
        execute(source, context, tmp_path)
    monkeypatch.setattr("inspect.getsource", lambda fn: "def kernel(x):\n    return 999\n")
    assert registry.jit_source(NS(fn=context["kernel"], src=source)) is not None
    assert registry.jit_source(NS(fn=context["kernel"], src="def kernel(x):\n    return 999\n")) is None


def test_nested_comprehension_code_is_registered_without_treating_it_as_a_jit_function(tmp_path):
    registry, context = SourceRegistry(), {}
    source = "def function(n):\n    return [[i*j for i in range(n)]\n            for j in range(n)]\n"
    with record_sources(registry):
        execute(source, context, tmp_path)
    assert context["function"](2) == [[0, 0], [0, 1]]
    assert registry.intact()


def test_cached_or_missing_frontend_phase_cannot_supply_new_origin(tmp_path, real_frontend):
    registry, frontend = real_frontend
    source = "def kernel(x):\n    return x + 1\n"
    context = {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    src = NS(fn=NS(fn=context["kernel"], src=source))
    assert frontend.proof_for(src) is None
    token, phase = frontend.phase(src, src.fn)
    frontend.finish(token, phase, complete=True)
    assert frontend.proof_for(src) is None  # Root was never actually parsed.


def test_changed_actual_parse_tree_cannot_inherit_frozen_jit_source(tmp_path, real_frontend):
    registry, frontend = real_frontend
    source = "def kernel(x):\n    return x + 1\n"
    context = {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    fn = NS(fn=context["kernel"], src=source)
    src = NS(fn=fn)
    token, phase = frontend.phase(src, fn)
    frontend.parse(fn, ast.parse("def kernel(x):\n    return 2\n"))
    frontend.finish(token, phase, complete=True)
    assert frontend.proof_for(src) is None and phase["issues"]


def test_language_metadata_and_actual_callable_closures_are_checked():
    pytest.importorskip("triton")
    import triton.language as tl

    language = LanguageSources()
    assert language.python_function(tl.load)
    assert language.python_function(len)
    assert not language.python_function(lambda x: x)
    assert language.records and language.intact()


def test_language_wrapper_cannot_hide_an_unapproved_callable_in_its_closure():
    import types

    pytest.importorskip("triton")
    import triton.language as tl

    language = LanguageSources()
    assert "fn" in tl.load.__code__.co_freevars
    closure = tuple(types.CellType(lambda *args: None) if name == "fn" else cell
                    for name, cell in zip(tl.load.__code__.co_freevars, tl.load.__closure__, strict=True))
    forged = types.FunctionType(tl.load.__code__, tl.load.__globals__, closure=closure)
    assert not language.python_function(forged)


def test_overridden_jit_parse_and_unobserved_cache_reuse_have_no_ast_receipt(tmp_path, real_frontend):
    registry, frontend = real_frontend
    source = ("import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(X,Y):\n"
              "    tl.store(Y,tl.load(X)+1)\n")
    context = {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    context["kernel"].parse = lambda: ast.parse(context["kernel"].src)
    _, document = frontend_ir(frontend, context)
    assert document is None


def test_completed_frontend_receipt_rejects_removed_asts_or_modified_calls(tmp_path, real_frontend):
    registry, frontend = real_frontend
    source = ("import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(X,Y):\n"
              "    tl.store(Y,tl.load(X)+1)\n")
    context = {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    _, document = frontend_ir(frontend, context)
    for mutate in (lambda d: d.update(parsed_sources=[]), lambda d: d.update(complete=False),
                   lambda d: d["calls"][0].update(primitive=False),
                   lambda d: d["calls"][0]["sources"][0].update(file_sha256="1" * 64)):
        changed = json.loads(json.dumps(document))
        mutate(changed)
        changed["receipt_sha256"] = fingerprint({k: v for k, v in changed.items() if k != "receipt_sha256"})
        assert not Frontend.validate(changed, source_sha256=sha256(source.encode()),
                                     jit_source_sha256=sha256(context["kernel"].src.encode()))


def make_extension(tmp_path, monkeypatch):
    from kb_workflows import cpp_origin

    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("ninja", "c++", "nvcc", "cuobjdump"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 99\n")
        path.chmod(0o700)
    cuda = tmp_path / "cuda/bin"
    cuda.mkdir(parents=True)
    (cuda / "nvcc").write_bytes((tools / "nvcc").read_bytes())
    (cuda / "nvcc").chmod(0o700)
    monkeypatch.setenv("PATH", str(tools))
    extension = NS(CUDA_HOME=str(cuda.parent), get_cxx_compiler=lambda: str(tools / "c++"))

    def write(path, sources, objects, with_cuda, with_sycl=False, cuda_dlink_post_cflags=None, sycl_dlink_post_cflags=None):
        Path(path).write_text(f'cxx = {tools / "c++"}\nnvcc = {cuda / "nvcc"}\n# observed fake build\n')

    def run(build_directory, verbose=False, error_prefix=""):
        (Path(build_directory) / "cuda.cuda.o").write_bytes(b"synthetic CUDA host object")

    def inline(name, cpp_sources, cuda_sources=None, build_directory=None, no_implicit_headers=False,
               extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None,
               keep_intermediates=True, with_sycl=False):
        directory = Path(build_directory)
        directory.mkdir(parents=True, exist_ok=True)
        sources = cuda_sources if isinstance(cuda_sources, list) else [cuda_sources]
        headers = [] if no_implicit_headers else ["#include <torch/types.h>", "#include <cuda.h>", "#include <cuda_runtime.h>"]
        (directory / "cuda.cu").write_text("\n".join(headers + sources))
        (directory / "main.cpp").write_text(("" if no_implicit_headers else "#include <torch/extension.h>\n") + cpp_sources)
        extension._write_ninja_file(str(directory / "build.ninja"), sources=[str(directory / "main.cpp"), str(directory / "cuda.cu")],
                                    objects=["main.o", "cuda.cuda.o"], with_cuda=True)
        extension._run_ninja_build(str(directory))
        return NS(loaded=True)

    def load(name, sources, build_directory=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None,
             extra_include_paths=None, keep_intermediates=True, with_sycl=False, is_standalone=False):
        directory = Path(build_directory)
        extension._write_ninja_file(str(directory / "build.ninja"), sources=sources,
                                    objects=["cuda.cuda.o"], with_cuda=True)
        extension._run_ninja_build(str(directory))
        return NS(loaded=True)

    def extract(command, directory, timeout):
        assert command[1:3] == ["--extract-elf", "all"]
        (directory / "device.cubin").write_bytes(CUBIN)
        log = b"synthetic extraction\n"
        return {"exit_code": 0, "timed_out": False, "log_base64": base64.b64encode(log).decode(), "log_sha256": sha256(log)}

    extension._write_ninja_file, extension._run_ninja_build = write, run
    extension.load_inline, extension.load = inline, load
    registry = SourceRegistry()
    observer = cpp_origin.Extensions(registry, extension=extension, version="2.12.0", extractor=extract,
                                     extraction_command=str(tools / "cuobjdump"))
    observer.install()
    observer.prepare()
    return registry, observer, extension


def run_extension(tmp_path, registry, extension, *, cuda=CUDA, extra="", public="load_inline"):
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    if public == "load_inline":
        source = f"cuda_source = {cuda!r}\nmodule = load_inline('owned', '', cuda_sources=cuda_source, build_directory=BUILD_DIRECTORY{extra})\n"
    else:
        source = f"cuda_source = {cuda!r}\nfrom pathlib import Path\npath=Path(BUILD_DIRECTORY)/'cuda.cu'\npath.write_text(cuda_source)\nmodule=load('owned',[str(path)],build_directory=BUILD_DIRECTORY{extra})\n"
    context = {"load_inline": extension.load_inline, "load": extension.load, "BUILD_DIRECTORY": str(build)}
    with record_sources(registry):
        execute(source, context, build)
    return source, context


@pytest.mark.parametrize("public", ["load_inline", "load"])
def test_cuda_extension_observes_literal_source_ninja_tools_and_object_bytes(tmp_path, monkeypatch, public):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        source, context = run_extension(tmp_path, registry, extension, public=public)
        documents = observer.collect()
        assert context["module"].loaded and len(documents) == 1
        document = documents[0]
        assert document["build_completed"] and observer.intact()
        assert not Extensions.validate(document, candidate_sha256=sha256(source.encode()), binary_sha256=sha256(CUBIN))
        assert Extensions.validate(document, candidate_sha256=sha256(source.encode()), binary_sha256=sha256(CUBIN), allow_simulated=True)
        assert json.loads(json.dumps(document)) == document
        assert document["objects"][0]["source_file_sha256"] in document["cuda_inputs"]
    finally:
        observer.restore()


@pytest.mark.parametrize("cuda", ["void host_only() {}", "#include <cub/cub.cuh>\n" + CUDA,
                                "#define H <private.h>\n#include H\n" + CUDA,
                                '#line 1 "other.cu"\n' + CUDA, CUDA + '\nasm(".incbin secret");',
                                "#define X(a,b) a ## b\n" + CUDA])
def test_unsupported_source_dependencies_do_not_gain_custom_credit(cuda):
    assert not cuda_policy(cuda)


@pytest.mark.parametrize("extra", [", extra_include_paths=['/outside']", ", extra_cuda_cflags=['-include foreign.h']",
                                  ", extra_ldflags=['foreign.o']", ", keep_intermediates=False"])
def test_unsupported_extension_options_preserve_native_load_but_supply_no_proof(tmp_path, monkeypatch, extra):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        _, context = run_extension(tmp_path, registry, extension, extra=extra)
        assert context["module"].loaded and observer.collect() == []
    finally:
        observer.restore()


def test_compiler_output_or_source_changes_cannot_reuse_a_build_receipt(tmp_path, monkeypatch):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        run_extension(tmp_path, registry, extension)
        (tmp_path / "build/cuda.cuda.o").write_bytes(b"substituted artifact")
        assert observer.collect()[0]["extraction"]["cubins"] == []
        (tmp_path / "build/cuda.cu").write_text("changed source")
        assert observer.collect() == []
    finally:
        observer.restore()


def test_frozen_extension_compiler_environment_and_ninja_rules_cannot_be_substituted(tmp_path, monkeypatch):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    original_runner = extension._run_ninja_build

    def replace_environment(*args, **kwargs):
        os.environ["NVCC_APPEND_FLAGS"] = "--changed"
        try:
            return original_runner(*args, **kwargs)
        finally:
            os.environ.pop("NVCC_APPEND_FLAGS", None)

    try:
        extension._run_ninja_build = replace_environment
        _, context = run_extension(tmp_path, registry, extension)
        assert context["module"].loaded and observer.collect() == []
        assert any("environment changed" in issue for frame in observer.builds for issue in frame["issues"])
    finally:
        observer.restore()


def test_cpp_receipt_replay_still_requires_observed_device_module_and_function(tmp_path, monkeypatch):
    from kb_workflows.cuda_origin import Capture
    from kb_workflows.kernel_origin import summarize

    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        source, _ = run_extension(tmp_path, registry, extension)
        document = observer.collect()[0]
        # Synthetic serialized parser fixture: remove simulated markers only
        # here to exercise format validation. This is not hardware evidence.
        document["simulated"] = False
        extraction = document["extraction"]
        extraction["simulated"] = False
        extraction["receipt_sha256"] = fingerprint({k: v for k, v in extraction.items() if k != "receipt_sha256"})
        document["receipt_sha256"] = fingerprint({k: v for k, v in document.items() if k != "receipt_sha256"})
        capture = Capture(source)
        capture.raw["modules"] = [{"id": 1, "handle": 10, "context": 20, "binary_sha256": sha256(CUBIN), "binary_bytes": len(CUBIN)}]
        capture.raw["functions"] = [{"id": 1, "handle": 30, "context": 20, "module": 1}]
        capture.bind_extensions([document])
        assert len(capture.raw["proofs"]) == 1
        capture.raw.update(capture_complete=True, source_intact=True, hooks_intact=True, errors=[], simulated=False,
            windows=[{"id": 1, "start_ns": 100, "end_ns": 300, "device": 0, "seed": 42, "input_signature": [],
                      "no_grad": True, "training": True, "training_after": True, "output_verified": True}],
            launches=[{"context_id": 9, "correlation": 10, "function": 1, "call": 1, "exit_success": True}],
            activities=[{"kind": "kernel", "start_ns": 120, "end_ns": 220, "device": 0, "stream": 7,
                         "context_id": 9, "correlation": 10, "graph_id": 0, "graph_node_id": 0}])
        result = summarize(capture.raw)
        assert result["status"] == "passed" and result["formal_qualification_implemented"] is False
        capture.raw["launches"][0]["function"] = None
        assert summarize(capture.raw)["status"] == "not_measured"
    finally:
        observer.restore()


def test_cpp_build_receipt_rejects_tool_input_and_artifact_rebinding(tmp_path, monkeypatch):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        source, _ = run_extension(tmp_path, registry, extension)
        document = observer.collect()[0]
        for mutate in (lambda d: d.update(build_completed=False), lambda d: d["objects"][0].update(source_file_sha256="1" * 64),
                       lambda d: d["tools"]["nvcc"].update(sha256="bad"),
                       lambda d: d["extraction"]["cubins"][0].update(binary_sha256="2" * 64)):
            changed = json.loads(json.dumps(document))
            mutate(changed)
            changed["receipt_sha256"] = fingerprint({k: v for k, v in changed.items() if k != "receipt_sha256"})
            assert not Extensions.validate(changed, candidate_sha256=sha256(source.encode()),
                                           binary_sha256=sha256(CUBIN), allow_simulated=True)
    finally:
        observer.restore()


def test_unregistered_loader_callsite_and_nonliteral_cuda_remain_unknown(tmp_path, monkeypatch):
    registry, observer, extension = make_extension(tmp_path, monkeypatch)
    try:
        extension.load_inline("outside", "", cuda_sources=CUDA, build_directory=str(tmp_path / "outside"))
        assert observer.collect() == []
        build = tmp_path / "owned"
        build.mkdir()
        context = {"load_inline": extension.load_inline, "CUDA": CUDA, "BUILD": str(build)}
        with record_sources(registry):
            execute("module=load_inline('owned','',cuda_sources=CUDA,build_directory=BUILD)\n", context, build)
        assert observer.collect() == []
    finally:
        observer.restore()


def test_frontend_receipt_malformed_or_incomplete_is_not_an_attestation():
    for value in (None, [], {}, {"protocol": "kernelbench_pro_triton_frontend_v1", "root": None}):
        assert not Frontend.validate(value, source_sha256="1" * 64, jit_source_sha256="2" * 64)


@pytest.mark.parametrize("mutation", ["binary", "source", "metadata"])
def test_compiled_kernel_cannot_replace_emitted_bytes_source_or_metadata(real_frontend, mutation, monkeypatch):
    from collections import namedtuple

    from kb_workflows.triton_origin import _BUILD

    _, frontend = real_frontend
    monkeypatch.setattr(frontend, "proof_for", lambda source: {"receipt_sha256": "1" * 64})
    source = NS()
    kernel = NS(src=source, kernel=CUBIN, metadata=namedtuple("Metadata", "shared")(0))
    phase = {"source": source, "issues": [], "emitted_sha256": None}
    token = _BUILD.set(phase)
    try:
        frontend.emitted(CUBIN, {}, NS(ir_override=None))
        frontend.freeze_compiled(kernel)
    finally:
        _BUILD.reset(token)
    assert frontend.compiled_for(kernel) is not None
    if mutation == "binary":
        kernel.kernel = b"foreign substituted bytes"
    elif mutation == "source":
        kernel.src = NS()
    else:
        kernel.metadata = type(kernel.metadata)(1)
    assert frontend.compiled_for(kernel) is None


def test_cache_only_constructor_and_ir_override_cannot_supply_emission_identity(real_frontend):
    from collections import namedtuple

    from kb_workflows.triton_origin import _BUILD

    _, frontend = real_frontend
    source = NS()
    kernel = NS(src=source, kernel=CUBIN, metadata=namedtuple("Metadata", "shared")(0))
    frontend.freeze_compiled(kernel)  # No observed backend generation.
    assert frontend.compiled_for(kernel) is None
    phase = {"source": source, "issues": [], "emitted_sha256": None}
    token = _BUILD.set(phase)
    try:
        frontend.emitted(CUBIN, {"ir_override": "foreign.ptx"}, NS(ir_override=None))
        frontend.freeze_compiled(kernel)
    finally:
        _BUILD.reset(token)
    assert phase["issues"] and frontend.compiled_for(kernel) is None


def test_frontend_compiler_entry_aliases_are_restored(real_frontend):
    import triton
    import triton.compiler as compiler
    import triton.compiler.compiler as module

    _, frontend = real_frontend
    assert triton.compile is compiler.compile is module.compile
    assert triton.compile.__name__ == "compile"
    frontend.restore()
    assert triton.compile is compiler.compile is module.compile


def test_real_target_metadata_is_bound_structurally_without_loading_a_device():
    from collections import namedtuple

    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget

    from kb_workflows.triton_origin import metadata_fingerprint

    metadata = namedtuple("KernelMetadata", "target shared extern_libs")(
        GPUTarget("cuda", 80, 32), 0, [["libdevice", "/frozen/libdevice.bc"]])
    assert metadata_fingerprint(metadata) == fingerprint({"target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "shared": 0, "extern_libs": [["libdevice", "/frozen/libdevice.bc"]]})


def test_real_compiled_kernel_constructor_keeps_emission_identity_without_loading_cuda(tmp_path, real_frontend, monkeypatch):
    from triton.compiler.compiler import CompiledKernel

    from kb_workflows.triton_origin import _BUILD

    registry, frontend = real_frontend
    context = {}
    code = "import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(X,Y):\n    tl.store(Y,tl.load(X)+1)\n"
    with record_sources(registry):
        execute(code, context, tmp_path / "source")
    source, closure = frontend_ir(frontend, context)
    metadata = {"target": {"backend": "cuda", "arch": 80, "warp_size": 32},
                "name": "kernel", "num_warps": 4, "num_ctas": 1, "shared": 0}
    meta, binary = tmp_path / "kernel.json", tmp_path / "kernel.cubin"
    meta.write_text(json.dumps(metadata))
    binary.write_bytes(CUBIN)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("constructor spawned a compiler/tool"))
    phase = {"source": source, "issues": [], "emitted_sha256": None}
    token = _BUILD.set(phase)
    try:
        frontend.emitted(CUBIN, {}, NS(ir_override=None))
        kernel = CompiledKernel(source, {"kernel.json": str(meta), "kernel.cubin": str(binary)}, "synthetic-hash")
    finally:
        _BUILD.reset(token)
    assert kernel.module is None and kernel.function is None
    assert frontend.compiled_for(kernel)["binary_sha256"] == sha256(CUBIN)
    assert frontend.compiled_frontend_for(kernel) == closure
    kernel.kernel = b"replacement"
    assert frontend.compiled_for(kernel) is None


def test_compiler_cannot_replace_bad_dependency_phase_with_a_later_clean_one(real_frontend, monkeypatch):
    from collections import namedtuple

    from kb_workflows.triton_origin import _BUILD

    _, frontend = real_frontend
    source = NS()
    kernel = NS(src=source, kernel=CUBIN, metadata=namedtuple("Metadata", "shared")(0))
    current = {"receipt": None}
    monkeypatch.setattr(frontend, "proof_for", lambda source: current["receipt"])
    phase = {"source": source, "issues": [], "emitted_sha256": None}
    token = _BUILD.set(phase)
    try:
        frontend.emitted(CUBIN, {}, NS(ir_override=None))
        frontend.freeze_compiled(kernel)
    finally:
        _BUILD.reset(token)
    current["receipt"] = {"receipt_sha256": "1" * 64}
    assert frontend.compiled_for(kernel) is None
    token = _BUILD.set(phase)
    try:
        frontend.freeze_compiled(kernel)
    finally:
        _BUILD.reset(token)
    frozen = frontend.compiled_frontend_for(kernel)
    current["receipt"]["receipt_sha256"] = "2" * 64
    assert frontend.compiled_frontend_for(kernel) == frozen
    assert frontend.compiled_for(kernel)["frontend_receipt_sha256"] == "1" * 64
