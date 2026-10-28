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
from kb_workflows.provenance.cpp_origin import Extensions, cuda_policy
from kb_workflows.provenance.submission_source import SourceRegistry, execute, record_sources, sha256
from kb_workflows.provenance.triton_origin import Frontend, LanguageSources
from tests.hardware.test_device_adapters import payload as payload

CUDA = '#include <cuda_runtime.h>\n__global__ void owned(float *x) { x[0] += 1; }\n'
CUBIN = b'\x7fELF\x02\x01' + b'\x00' * 12 + (190).to_bytes(2, 'little') + b'\x00' * 44 + b'synthetic-device-object'


def test_neuron_preloaded_runtime_binds_model_generations_and_device_sequences(tmp_path, monkeypatch):
    """Compile CPU-only ABI stubs; never link or execute the vendor runtime."""
    from kb_workflows.provenance import neuron_runtime

    include = tmp_path / "include" / "nrt"
    include.mkdir(parents=True)
    header = """#pragma once
#include <stddef.h>
#include <stdint.h>
typedef enum {NRT_SUCCESS=0, NRT_FAILURE=1} NRT_STATUS;
typedef struct model {int value;} nrt_model_t;
typedef struct tensors {int value;} nrt_tensor_set_t;
typedef uint64_t nrta_seq_t;
extern "C" {
NRT_STATUS nrt_load(const void*,size_t,int32_t,int32_t,nrt_model_t**);
NRT_STATUS nrt_load_collectives(const void*,size_t,int32_t,int32_t,uint32_t,uint32_t,nrt_model_t**);
NRT_STATUS nrt_unload(nrt_model_t*);
NRT_STATUS nrta_execute_schedule(nrt_model_t*,const nrt_tensor_set_t*,nrt_tensor_set_t*,int,NRT_STATUS*,nrta_seq_t*);
}
"""
    (include / "nrt.h").write_text(header)
    for name in ("nrt_async.h", "nrt_experimental.h"):
        (include / name).write_text('#pragma once\n#include "nrt.h"\n')
    stub = tmp_path / "stub.cpp"
    stub.write_text("""#include <nrt/nrt.h>
#include <spawn.h>
#include <sys/wait.h>
extern "C" int torch_neuronx_posix_spawn_with_stable_environ(pid_t*,const char*,const posix_spawn_file_actions_t*,const posix_spawnattr_t*,char* const[],const char*) __attribute__((weak));
static nrt_model_t model;
static nrta_seq_t next=40;
extern "C" NRT_STATUS nrt_load(const void*,size_t,int32_t,int32_t,nrt_model_t** out) {*out=&model;return NRT_SUCCESS;}
extern "C" NRT_STATUS nrt_load_collectives(const void* b,size_t n,int32_t s,int32_t c,uint32_t,uint32_t,nrt_model_t** o) {return nrt_load(b,n,s,c,o);}
extern "C" NRT_STATUS nrt_unload(nrt_model_t*) {return NRT_SUCCESS;}
extern "C" NRT_STATUS nrta_execute_schedule(nrt_model_t*,const nrt_tensor_set_t*,nrt_tensor_set_t*,int,NRT_STATUS*,nrta_seq_t* seq) {*seq=++next;return NRT_SUCCESS;}
extern "C" int exercise(const char* cwd) {
 posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
 posix_spawn_file_actions_addchdir_np(&actions,cwd);
 char* argv[]={(char*)"neuronx-cc",(char*)"compile",(char*)"module.mlir",(char*)"--output",(char*)"module.neff",nullptr};
 pid_t child;int status;
 int error=torch_neuronx_posix_spawn_with_stable_environ(&child,argv[0],&actions,nullptr,argv,cwd);
 posix_spawn_file_actions_destroy(&actions);
 if(error || waitpid(child,&status,0)!=child || status!=0)return 1;
 nrt_model_t* ptr; NRT_STATUS completion; nrta_seq_t seq; char binary[]="SIMULATED NEFF";
 nrt_load(binary,sizeof(binary),0,1,&ptr);
 nrta_execute_schedule(ptr,nullptr,nullptr,0,&completion,&seq);
 nrt_unload(ptr);
 nrt_load_collectives(binary,sizeof(binary),0,1,0,1,&ptr);
 nrta_execute_schedule(ptr,nullptr,nullptr,0,&completion,&seq);
 nrt_unload(ptr);
 return 0;
}
""")
    library = tmp_path / "libnrt.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-I", str(include.parent), str(stub), "-o", str(library)],
                   check=True, capture_output=True, timeout=30)
    request = {"task": "profile", "eval_config": {"device_backend": "neuron", "neuron_runtime_origin": True,
               "neuron_runtime_include": str(include.parent), "neuron_runtime_compiler": "c++"}}
    work = tmp_path / "compiler-work"
    work.mkdir()
    (work / "module.mlir").write_bytes(b"SIMULATED StableHLO")
    bir_file = work / "input.bir"
    bir_file.write_bytes(b"SIMULATED BIR")
    constant_file = work / "weight.npy"
    constant_file.write_bytes(b"SIMULATED NPY")
    fake_compiler = tmp_path / "fake-neuronx-cc"
    fake_compiler.write_text("#!" + os.sys.executable + "\nfrom pathlib import Path\nPath('module.neff').write_bytes(b'SIMULATED NEFF\\x00')\n"
                            "if Path('inject-constant').exists(): Path('late.npy').write_bytes(b'LATE CONSTANT')\n")
    fake_compiler.chmod(0o755)
    manifest = neuron_runtime.build(tmp_path / "runtime", include.parent, "c++", fingerprint(request), simulated=True,
                                   backend_compiler=str(fake_compiler))
    (manifest.parent / "compiler-inputs.json").write_text(json.dumps({sha256(b"SIMULATED StableHLO"): [
        {"path": str(bir_file), "sha256": sha256(b"SIMULATED BIR"), "role": "bir"},
        {"path": str(constant_file), "sha256": sha256(b"SIMULATED NPY")}]}))
    build_info = json.loads(manifest.read_text())
    capture = tmp_path / "profile"
    capture.mkdir()
    events = [{"ph": "X", "cat": "privateuse1_driver", "ts": 100 + i * 10, "dur": 5,
               "args": {"nrt_sequence_id": seq}} for i, seq in enumerate((41, 42, 99))]
    (capture / "framework_trace.json").write_text(json.dumps({"traceEvents": events}))
    code = f"""
import ctypes,json
from kb_workflows.provenance.neuron_runtime import Observer
observer=Observer({request!r},allow_simulated=True)
from pathlib import Path
import os
script=Path({build_info["compiler_script"]!r})
original=script.read_bytes()
script.write_text("raise RuntimeError('MUTATED compiler wrapper must never execute')")
os.environ['PATH']='/not-a-compiler-path'
handle=ctypes.CDLL({str(library)!r},mode=ctypes.RTLD_LOCAL)
handle.exercise.argtypes=[ctypes.c_char_p]
assert handle.exercise({str(work).encode()!r})==0
script.write_bytes(original)
result=observer.export({str(capture)!r})
print(json.dumps(result))
"""
    child = subprocess.run([os.sys.executable, "-c", code], env={**os.environ,
        "LD_PRELOAD": build_info["library"], neuron_runtime.ENV: str(manifest),
        "KB_NEURON_RUNTIME_DIRECTORY": str(manifest.parent),
        "KB_NEURON_COMPILER_SCRIPT": build_info["compiler_script"], "KB_NEURON_COMPILER_PYTHON": os.sys.executable}, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["simulated"] and result["status"] == "partial"
    assert [b["generation"] for b in result["bindings"]] == [1, 2]
    assert result["bindings"][0]["neff_sha256"] == result["bindings"][1]["neff_sha256"]
    assert result["unresolved_events"] == [2] and not result["loaded_binary_identity_proven"]
    assert not result["formal_origin_qualified"]
    assert len(result["compiler_jobs"]) == 1 and not result["compiler_jobs"][0]["issues"]
    assert result["compiler_jobs"][0]["constant_directories"] == {str(work): ["weight.npy"]}
    compilation = {"simulated": True, "pipelines": [{"fresh_lowering_observed": True,
        "native_ast": {"all_definition_sources_matched": True, "module_handoff_observed": True, "root_source_sha256": "a"*64},
        "stablehlo_sha256": [sha256(b"SIMULATED StableHLO")], "bir_path": str(bir_file), "bir_sha256": sha256(b"SIMULATED BIR"),
        "bir_emission": {"cache_copy_matched": True, "effective_files": [
            {"path": str(bir_file), "sha256": sha256(b"SIMULATED BIR")},
            {"path": str(constant_file), "sha256": sha256(b"SIMULATED NPY")} ]}} ]}
    linked = neuron_runtime.compiler_links(compilation, result)
    assert linked["status"] == "partial" and len(linked["links"]) == 2 and linked["simulated"]
    assert linked["unresolved_events"] == [2]
    assert not linked["formal_origin_qualified"]
    assert neuron_runtime.compiler_links(compilation, {**result, "compiler_jobs": []})["status"] == "not_measured"
    wrong_hlo = json.loads(json.dumps(compilation))
    wrong_hlo["pipelines"][0]["stablehlo_sha256"] = ["b"*64]
    assert neuron_runtime.compiler_links(wrong_hlo, result)["status"] == "not_measured"
    missing_bundle = json.loads(json.dumps(compilation))
    missing_bundle["pipelines"][0]["bir_emission"]["effective_files"][1]["sha256"] = "c"*64
    assert neuron_runtime.compiler_links(missing_bundle, result)["status"] == "not_measured"
    missing_bundle["pipelines"][0]["bir_emission"] = None
    assert neuron_runtime.compiler_links(missing_bundle, result)["status"] == "not_measured"
    from kb_workflows.hardware.neuron_processing import DEFAULT_LIMITS, file_manifest

    files = file_manifest(capture, DEFAULT_LIMITS)
    hashes, sizes = {p: r["sha256"] for p, r in files.items()}, {p: r["bytes"] for p, r in files.items()}
    neuron_runtime.validate(result, hashes, sizes, allow_simulated=True)
    with pytest.raises(ValueError, match="receipt"):
        neuron_runtime.validate(result, hashes, sizes)
    changed = json.loads(json.dumps(result))
    next(r for r in changed["journal"] if r["type"] == "unload")["model"] += 1
    from kb_workflows.provenance.neuron_origin import _seal

    _seal(changed)
    with pytest.raises(ValueError, match="journal bytes"):
        neuron_runtime.validate(changed, hashes, sizes, allow_simulated=True)
    with pytest.raises(ValueError, match="binary snapshot"):
        neuron_runtime.validate(result, {p: h for p, h in hashes.items() if not p.endswith(".neff")}, sizes, allow_simulated=True)
    extra_constant = work / "unexpected.npy"
    extra_constant.write_bytes(b"UNOBSERVED CONSTANT")
    assert neuron_runtime._compile_job(build_info["backend_compiler"], [str(tmp_path / "extra-job"), str(work),
        "neuronx-cc", "compile", "module.mlir", "--output", "module.neff"]) == 0
    extra_receipt = json.loads((tmp_path / "extra-job/receipt.json").read_text())
    assert any("directory changed before" in issue for issue in extra_receipt["issues"])
    extra_constant.unlink()
    constant_file.write_bytes(b"POISONED CONSTANT")
    assert neuron_runtime._compile_job(build_info["backend_compiler"], [str(tmp_path / "changed-job"), str(work),
        "neuronx-cc", "compile", "module.mlir", "--output", "module.neff"]) == 0
    changed_receipt = json.loads((tmp_path / "changed-job/receipt.json").read_text())
    assert any("BIR changed before" in issue for issue in changed_receipt["issues"])
    constant_file.write_bytes(b"SIMULATED NPY")
    (work / "inject-constant").touch()
    assert neuron_runtime._compile_job(build_info["backend_compiler"], [str(tmp_path / "late-job"), str(work),
        "neuronx-cc", "compile", "module.mlir", "--output", "module.neff"]) == 0
    late_receipt = json.loads((tmp_path / "late-job/receipt.json").read_text())
    assert any("directory changed during" in issue for issue in late_receipt["issues"])
    journal = result["journal"]
    with pytest.raises(ValueError, match="live model"):
        neuron_runtime._journal([journal[0], next(r for r in journal if r["type"] == "execute"), journal[-1]])
    duplicate = [*journal[:-1], next(r for r in journal if r["type"] == "load"), journal[-1]]
    with pytest.raises(ValueError, match="generation"):
        neuron_runtime._journal(duplicate)
    monkeypatch.delenv(neuron_runtime.ENV, raising=False)
    monkeypatch.setenv("LD_PRELOAD", "existing-observer.so")
    monkeypatch.setattr(neuron_runtime, "build", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(os.sys, "argv", ["worker", "request.json", "result.json", "--native-worker"])
    executions = []
    monkeypatch.setattr(os, "execve", lambda *args: executions.append(args))
    neuron_runtime.bootstrap(request)
    assert executions[0][1][1:] == ["-m", "kb_workflows.evaluation.evaluate_task", "request.json", "result.json", "--native-worker"]
    assert executions[0][2]["LD_PRELOAD"] == build_info["library"] + " existing-observer.so"
    neuron_runtime.bootstrap({**request, "task": "evaluate"})
    assert len(executions) == 1


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
    from kb_workflows.provenance import cpp_origin

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
    from kb_workflows.provenance.cuda_origin import Capture
    from kb_workflows.provenance.kernel_origin import summarize

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

    from kb_workflows.provenance.triton_origin import _BUILD

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

    from kb_workflows.provenance.triton_origin import _BUILD

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

    from kb_workflows.provenance.triton_origin import metadata_fingerprint

    metadata = namedtuple("KernelMetadata", "target shared extern_libs")(
        GPUTarget("cuda", 80, 32), 0, [["libdevice", "/frozen/libdevice.bc"]])
    assert metadata_fingerprint(metadata) == fingerprint({"target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "shared": 0, "extern_libs": [["libdevice", "/frozen/libdevice.bc"]]})


def test_real_compiled_kernel_constructor_keeps_emission_identity_without_loading_cuda(tmp_path, real_frontend, monkeypatch):
    from triton.compiler.compiler import CompiledKernel

    from kb_workflows.provenance.triton_origin import _BUILD

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

    from kb_workflows.provenance.triton_origin import _BUILD

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


def test_nki_compiler_can_inspect_frozen_source_and_changed_file_is_rejected(payload, monkeypatch, tmp_path):
    import copy
    import importlib.util
    import inspect
    import os
    import py_compile
    import sys
    from types import ModuleType

    from kb_workflows import engine
    from kb_workflows.hardware import neuron_worker
    from kb_workflows.provenance.submission_source import SourceRegistry, record_sources
    from tests.hardware.test_device_adapters import SimulatedNeuron

    config, request = payload
    value = neuron_worker.evaluate(request, adapter=SimulatedNeuron())
    path = neuron_worker.ROOT / "cache/workflow" / fingerprint(request) / "candidate.py"
    registry = SourceRegistry()
    with record_sources(registry):
        namespace = neuron_worker.load_source(request["kernel_src"], "inspection_test", filename=path)
    assert "class ModelNew" in inspect.getsource(namespace["ModelNew"])
    assert value["metadata"]["compiler_source_sha256"]["candidate.py"]
    assert value["metadata"]["compiler_source_registration"] == "frozen_compile_code_objects"
    assert registry.source_for(namespace["ModelNew"].forward)["source_sha256"] == value["metadata"]["compiler_source_sha256"]["candidate.py"]
    copied = {"__name__": "recompiled_neuron_source"}
    exec(compile(request["kernel_src"], str(path), "exec", dont_inherit=True), copied)
    assert registry.source_for(copied["ModelNew"].forward) is None
    original, stamp = path.read_bytes(), path.stat()
    path.write_bytes(original.replace(b"ModelNew", b"XodelNew"))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    path.write_bytes(original)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    spec = importlib.util.spec_from_file_location("stale_neuron_pyc_control", path)
    control = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, control)
    spec.loader.exec_module(control)
    assert "XodelNew" in vars(control) and "ModelNew" not in vars(control)
    refreshed = neuron_worker.load_source(request["kernel_src"], "trusted_compilation", filename=path)
    assert "ModelNew" in refreshed and "XodelNew" not in refreshed
    crlf = copy.deepcopy(request)
    crlf["kernel_src"] = request["kernel_src"].replace("\n", "\r\n")
    assert neuron_worker.evaluate(crlf, adapter=SimulatedNeuron())["correctness"]

    from kb_workflows.hardware.neuron_processing import DEFAULT_LIMITS, file_manifest
    from kb_workflows.provenance.neuron_origin import CompilationObserver, validate_compilation

    ir = tmp_path / "emission" / "observed.bir"
    ir.parent.mkdir()
    ir.write_bytes(b"SIMULATED BIR; no NKI compiler or device executed")
    constant = ir.with_name("weight.npy")
    constant.write_bytes(b"SIMULATED NPY")
    core = ir.with_name("core.bir")
    core.write_bytes(b"SIMULATED PER-CORE BIR")
    Path(str(ir) + ".debuginfo.json").write_bytes(b"SIMULATED DEBUG")
    module = object()
    extra_dependencies = []
    cache_mode = None

    def frontend(kernel_fn):
        kernel = NS(func=kernel_fn)
        assert kernel.func is kernel_fn
        return NS(module=module, dep_files=[kernel_fn.__code__.co_filename, *extra_dependencies])

    def emitter(module):
        return None, NS(kernel_json_path=ir, per_core_json_paths=(str(core),)), "nki", "nki"

    def bir(module):
        work_dir = ir.parent
        assert work_dir.is_dir()
        _, bir_result, _, _ = emitter(module)
        path = ir
        if cache_mode is not None:
            path = tmp_path / ("cache-" + cache_mode) / "cached.bir"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(ir.read_bytes())
            (path.parent / constant.name).write_bytes(constant.read_bytes())
            Path(str(path) + ".debuginfo.json").write_bytes(Path(str(ir) + ".debuginfo.json").read_bytes())
            if cache_mode == "bir":
                path.write_bytes(b"POISONED CACHE BIR")
            elif cache_mode == "constant":
                (path.parent / constant.name).write_bytes(b"POISONED CACHE NPY")
            elif cache_mode == "extra":
                (path.parent / "extra.npy").write_bytes(b"EXTRA")
            elif cache_mode == "missing":
                (path.parent / constant.name).unlink()
        return NS(descriptor=NS(kernel_json_path=path, per_core_json_paths=bir_result.per_core_json_paths))

    def pipeline(kernel_func, *, cached=False, mismatch=False):
        if cached:
            return NS(descriptor=NS(kernel_json_path=ir))
        result = frontend(kernel_func)
        return bir(object() if mismatch else result.module)

    def build_config(self):
        return NS(backend_config_b64=b"SIMULATED backend config",
                               backend_config={"klir_binary": {"binary": str(self.descriptor.kernel_json_path)}})

    def lower(dconfig, *, corrupt=False):
        backend_config = dconfig.backend_config_b64.decode("ascii")
        if corrupt:
            ir.write_bytes(b"CHANGED BIR")
        assert backend_config
        return NS(stablehlo_bytes=b"SIMULATED StableHLO", cache_key="simulated-key")

    functions = {"pipeline": pipeline, "frontend": frontend, "bir": bir, "config": build_config, "lower": lower, "emitter": emitter}
    with CompilationObserver(registry, functions=functions) as observer:
        lower(build_config(pipeline(namespace["ModelNew"].forward)))
    directory = tmp_path / "compiler-capture"
    directory.mkdir()
    observation = observer.export(directory)
    files = file_manifest(directory, DEFAULT_LIMITS)
    hashes, sizes = {k: r["sha256"] for k, r in files.items()}, {k: r["bytes"] for k, r in files.items()}
    assert observation["status"] == "observed" and observation["simulated"]
    assert observation["pipelines"][0]["root"]["source_sha256"] == value["metadata"]["compiler_source_sha256"]["candidate.py"]
    assert observation["pipelines"][0]["cache_keys"] == ["simulated-key"]
    assert not observation["formal_origin_qualified"] and not observation["loaded_binary_identity_proven"]
    validate_compilation(observation, hashes, sizes, allow_simulated=True)
    with pytest.raises(ValueError, match="receipt"):
        validate_compilation(observation, hashes, sizes)
    with pytest.raises(ValueError, match="snapshot"):
        validate_compilation(observation, {p: h for p, h in hashes.items() if not p.endswith(".bir")}, sizes, allow_simulated=True)
    bundle = observation["pipelines"][0]["bir_emission"]
    assert bundle["cache_copy_matched"] and {r["role"] for r in bundle["files"]} == {"bir", "per_core:0", "constant:weight.npy", "debug"}
    with pytest.raises(ValueError, match="bundle snapshot"):
        validate_compilation(observation, {p: h for p, h in hashes.items() if not p.startswith("compiler_inputs/")}, sizes, allow_simulated=True)
    for cache_mode in ("valid", "bir", "constant", "extra", "missing"):
        with CompilationObserver(registry, functions=functions) as cached_bundle:
            lower(build_config(pipeline(namespace["ModelNew"].forward)))
        row = cached_bundle.value["pipelines"][0]
        assert row["fresh_lowering_observed"] is (cache_mode == "valid")
        assert row["bir_emission"]["cache_copy_matched"] is (cache_mode == "valid")
        assert bool(cached_bundle.value["issues"]) is (cache_mode != "valid")
        cached_dir = tmp_path / ("capture-" + cache_mode)
        cached_dir.mkdir()
        cached_bundle.export(cached_dir)
        inventory = file_manifest(cached_dir, DEFAULT_LIMITS)
        validate_compilation(cached_bundle.value, {p: r["sha256"] for p, r in inventory.items()},
                             {p: r["bytes"] for p, r in inventory.items()}, allow_simulated=True)
    cache_mode = None
    with CompilationObserver(registry, functions=functions) as shared_hlo:
        shared_hlo.compiler_index = tmp_path / "shared-inputs.json"
        lower(build_config(pipeline(namespace["ModelNew"].forward)))
        cache_mode = "valid"
        lower(build_config(pipeline(namespace["ModelNew"].forward)))
    merged = json.loads(shared_hlo.compiler_index.read_text())[sha256(b"SIMULATED StableHLO")]
    assert {item["path"] for item in merged if item["role"] == "bir"} == {str(ir), str(tmp_path / "cache-valid/cached.bir")}
    assert {item["path"] for item in merged if item["path"].endswith(".npy")} == {str(constant), str(tmp_path / "cache-valid/weight.npy")}
    cache_mode = None
    with CompilationObserver(registry, functions=functions) as changed_constant:
        config_value = build_config(pipeline(namespace["ModelNew"].forward))
        constant.write_bytes(b"CHANGED AFTER CONFIG")
        lower(config_value)
    assert changed_constant.value["issues"] and not changed_constant.value["pipelines"][0]["fresh_lowering_observed"]
    constant.write_bytes(b"SIMULATED NPY")
    for options in ({"cached": True}, {"mismatch": True}):
        with CompilationObserver(registry, functions=functions) as missing:
            lower(build_config(pipeline(namespace["ModelNew"].forward, **options)))
        assert missing.value["status"] == "partial"
        assert not missing.value["pipelines"][0]["fresh_lowering_observed"]
    with CompilationObserver(registry, functions=functions) as changed:
        lower(build_config(pipeline(namespace["ModelNew"].forward)), corrupt=True)
    assert changed.value["issues"] and not changed.value["pipelines"][0]["fresh_lowering_observed"]
    with CompilationObserver(registry, functions=functions) as disrupted:
        sys.setprofile(None)
    assert disrupted.value["status"] == "partial" and disrupted.value["issues"]
    assert sys.getprofile() is None
    from types import FunctionType

    recompiled = FunctionType(pipeline.__code__.replace(), pipeline.__globals__, closure=pipeline.__closure__)
    with CompilationObserver(registry, functions=functions) as unregistered:
        recompiled(namespace["ModelNew"].forward, cached=True, mismatch=False)
    assert unregistered.value["pipelines"] == []
    assert unregistered.value["status"] == "not_measured"
    def previous(*args):
        return None
    sys.setprofile(previous)
    try:
        with CompilationObserver(registry, functions=functions) as occupied:
            assert sys.getprofile() is previous
        assert sys.getprofile() is previous and occupied.value["issues"]
    finally:
        sys.setprofile(None)
    import base64
    import hashlib

    from kb_workflows.provenance.submission_source import WheelSources, compile_source

    vendor = tmp_path / "vendor"
    primitive_path = vendor / "nki/language/fixture.py"
    primitive_path.parent.mkdir(parents=True)
    primitive_source = "def primitive(x): return x\n"
    primitive_path.write_text(primitive_source)
    lookalike = primitive_path.with_name("fixture.py.other.py")
    lookalike.write_text(primitive_source)
    primitive_namespace = {}
    exec(compile(primitive_source, str(primitive_path), "exec", dont_inherit=True), primitive_namespace)
    digest = base64.urlsafe_b64encode(hashlib.sha256(primitive_source.encode()).digest()).decode().rstrip("=")
    class Distribution:
        version = "0.7.0"
        def locate_file(self, name): return vendor / name
        def read_text(self, name):
            return (f"nki/language/fixture.py,sha256={digest},{len(primitive_source)}\n"
                    f"nki/language/fixture.py.other.py,sha256={digest},{len(primitive_source)}\n")
    language = WheelSources("nki", "0.7.0", ("nki/language/",), distribution=Distribution())
    exact = WheelSources("nki", "0.7.0", ("nki/language/fixture.py",), distribution=Distribution())
    assert set(exact.paths) == {str(primitive_path)}
    sdk = ModuleType("reviewed_nki_fixture")
    sdk.primitive = primitive_namespace["primitive"]
    graph = "def helper(x): return x+1\ndef kernel(x): return sdk.primitive(helper(x))\n"
    graph_namespace = {"sdk": sdk}
    with record_sources(registry):
        exec(compile_source(graph, tmp_path / "graph.py"), graph_namespace)
    extra_dependencies[:] = [str(primitive_path)]
    with CompilationObserver(registry, functions=functions, language=language) as dependencies:
        lower(build_config(pipeline(graph_namespace["kernel"])))
    observed = dependencies.value["pipelines"][0]["source_dependencies"]
    assert observed["status"] == "observed" and observed["issues"] == []
    assert {item["role"] for item in observed["files"]} == {"registered_source", "approved_wheel_source"}
    assert [item["kind"] for item in observed["callables"]] == ["registered_code", "approved_primitive", "registered_code"]
    dependency_dir = tmp_path / "dependencies"
    dependency_dir.mkdir()
    dependencies.export(dependency_dir)
    artifacts = file_manifest(dependency_dir, DEFAULT_LIMITS)
    digests, byte_sizes = {p: r["sha256"] for p, r in artifacts.items()}, {p: r["bytes"] for p, r in artifacts.items()}
    validate_compilation(dependencies.value, digests, byte_sizes, allow_simulated=True)
    with pytest.raises(ValueError, match="source artifact"):
        validate_compilation(dependencies.value, {p: h for p, h in digests.items() if not p.startswith("compiler_sources/")},
                             byte_sizes, allow_simulated=True)
    import msgpack

    from kb_workflows.provenance import neuron_ast

    facade = NS()
    functions_used = [graph_namespace["kernel"], graph_namespace["helper"], sdk.primitive]
    encoded_definitions = []
    for fn in functions_used:
        source_path = Path(fn.__code__.co_filename)
        definition = next(n for n in ast.walk(ast.parse(source_path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__)
        location = [str(source_path), definition.lineno, definition.col_offset + 1, definition.end_lineno, definition.end_col_offset + 1]
        encoded_definitions.append([str(fn.__module__) + "." + fn.__qualname__, [], [[], [], [], None, [], [], None], [], location])
    blob = msgpack.packb([encoded_definitions[0][0], encoded_definitions, [], [], [], [], 3, 0, 1, []], use_bin_type=True)
    class NativeKernel:
        def __init__(self, fn): self.function = fn
        def specialize(self):
            for fn in functions_used:
                inspect.getsourcelines(fn)
        def serialize_python(self): return blob
        def compile(self): return module
    facade.Kernel = NativeKernel
    def parser(kernel_fn):
        arch, arch_sub_version, lnc = 3, 0, 1
        assert arch + arch_sub_version + lnc == 4
        native = facade.Kernel(kernel_fn)
        native.specialize()
        return NS(module=native.compile(), dep_files=[kernel_fn.__code__.co_filename, str(primitive_path)])
    def parsed_frontend(kernel_fn):
        kernel = NS(func=kernel_fn)
        assert kernel.func is kernel_fn
        return parser(kernel_fn)
    def parsed_pipeline(kernel_func):
        return bir(parsed_frontend(kernel_func).module)
    native_functions = {**functions, "pipeline": parsed_pipeline, "frontend": parsed_frontend,
                        "parser": parser, "source": inspect.getsourcelines}
    with CompilationObserver(registry, functions=native_functions, language=language, native_frontend=facade) as native_capture:
        lower(build_config(parsed_pipeline(graph_namespace["kernel"])))
    assert facade.Kernel is NativeKernel
    native_packet = native_capture.value["pipelines"][0]["native_ast"]
    assert native_packet["all_definition_sources_matched"] and native_packet["module_handoff_observed"]
    assert native_packet["simulated"] and not native_packet["native_definition_sources_bound"]
    assert not native_packet["formal_origin_qualified"]
    indirect = neuron_ast.bind(blob, native_packet["reads"], [3, 0, 1])
    assert not indirect["native_definition_sources_bound"] and indirect["unresolved_definitions"]
    native_dir = tmp_path / "native-ast"
    native_dir.mkdir()
    native_capture.export(native_dir)
    native_files = file_manifest(native_dir, DEFAULT_LIMITS)
    native_hashes = {p: r["sha256"] for p, r in native_files.items()}
    native_sizes = {p: r["bytes"] for p, r in native_files.items()}
    validate_compilation(native_capture.value, native_hashes, native_sizes, allow_simulated=True)
    with pytest.raises(ValueError, match="packet"):
        neuron_ast.validate(native_packet, native_hashes, native_sizes)
    with pytest.raises(ValueError, match="artifact"):
        neuron_ast.validate(native_packet, {p: h for p, h in native_hashes.items() if not p.endswith(".msgpack")}, native_sizes, allow_simulated=True)
    missing = neuron_ast.bind(blob, native_packet["reads"][:1], [3, 0, 1], simulated=True)
    assert not missing["all_definition_sources_matched"] and missing["unresolved_definitions"]
    changed_config = neuron_ast.bind(blob, native_packet["reads"], [3, 0, 2], simulated=True)
    assert not changed_config["all_definition_sources_matched"]
    with pytest.raises(ValueError, match="MessagePack"):
        neuron_ast.decode(blob + b"extra")
    # A format failure must preserve the real compiler call and leave source evidence unknown.
    original_serializer = NativeKernel.serialize_python
    NativeKernel.serialize_python = lambda self: b"unsupported"
    with CompilationObserver(registry, functions=native_functions, language=language, native_frontend=facade) as unknown_ast:
        lower(build_config(parsed_pipeline(graph_namespace["kernel"])))
    assert unknown_ast.value["issues"] and unknown_ast.value["pipelines"][0]["fresh_lowering_observed"]
    assert unknown_ast.value["pipelines"][0]["native_ast"] is None
    NativeKernel.serialize_python = original_serializer
    # Equal filenames do not turn separately compiled helper code into ownership.
    unowned = {}
    exec(compile("def helper(x): return x+2", str(tmp_path / "graph.py"), "exec"), unowned)
    graph_namespace["helper"] = unowned["helper"]
    extra_dependencies.append(str(tmp_path / "unapproved-do-not-read.py"))
    with CompilationObserver(registry, functions=functions, language=language) as unsupported:
        lower(build_config(pipeline(graph_namespace["kernel"])))
    incomplete = unsupported.value["pipelines"][0]["source_dependencies"]
    assert incomplete["status"] == "partial"
    assert "unregistered_or_opaque_callable" in incomplete["issues"]
    assert any(issue.startswith("unapproved_gather_file:") for issue in incomplete["issues"])
    path.write_text("# changed compiled source")
    with pytest.raises(ValueError, match="compiler source file"):
        neuron_worker.evaluate(engine.task_payload(config, 19), adapter=SimulatedNeuron())
