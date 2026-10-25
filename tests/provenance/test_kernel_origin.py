"""CPU-only replay/ABI/loader checks; never load CUPTI or launch a kernel."""

from __future__ import annotations

import copy
import ctypes as C
import inspect
import json
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from kb_workflows.common import ROOT, evaluator_digest, fingerprint
from kb_workflows.provenance import cuda_origin, kernel_origin
from kb_workflows.provenance.submission_source import SourceRegistry, execute, record_sources, sha256

SOURCE = "def kernel(x):\n    return x + 1\n"
BINARY = b"\x7fELF\x00\xfffrozen-cubin"


def frontend_receipt(source=SOURCE):
    import ast

    from kb_workflows.provenance.triton_origin import PROTOCOL, ast_digest

    root = {"role": "candidate", "source_sha256": sha256(source.encode()), "jit_source_sha256": sha256(source.encode()),
            "ast_sha256": ast_digest(ast.parse(source)), "function": "kernel"}
    document = {"protocol": PROTOCOL, "root": root, "calls": [], "parsed_sources": [root], "complete": True,
                "language_manifest": {"sources": []}, "language_manifest_sha256": fingerprint({"sources": []})}
    document["receipt_sha256"] = fingerprint(document)
    return document


def raw_capture():
    """Synthetic parser fixture, not a hardware observation or qualification."""
    return {"protocol": kernel_origin.PROTOCOL, "candidate_sha256": sha256(SOURCE.encode()),
            "modules": [{"id": 1, "handle": 100, "context": 1, "binary_sha256": sha256(BINARY), "binary_bytes": len(BINARY)}],
            "functions": [{"id": 1, "handle": 200, "context": 1, "module": 1}],
            "proofs": [{"id": 1, "function": 1, "compiler": "triton-3.7.0", "source_sha256": sha256(SOURCE.encode()),
                        "binary_sha256": sha256(BINARY), "jit_source_sha256": sha256(SOURCE.encode()),
                        "frontend": frontend_receipt(),
                        "emission": {"protocol": "kernelbench_pro_triton_emission_v1", "backend": "cuda", "binary_sha256": sha256(BINARY), "metadata_sha256": "1" * 64,
                                     "frontend_receipt_sha256": frontend_receipt()["receipt_sha256"]},
                        "compiler_metadata_sha256": "1" * 64}],
            "launches": [{"context_id": 9, "correlation": 10, "function": 1, "call": 1, "exit_success": True}],
            "windows": [{"id": 1, "start_ns": 100, "end_ns": 300, "device": 0, "seed": 42,
                         "input_signature": [], "no_grad": True, "training": True, "training_after": True, "output_verified": True}],
            "activities": [{"kind": "kernel", "start_ns": 120, "end_ns": 220, "device": 0, "context_id": 9,
                            "stream": 7, "correlation": 10, "graph_id": 0, "graph_node_id": 0, "name": "any name"}],
            "capture_complete": True, "source_intact": True, "hooks_intact": True, "dropped_records": 0,
            "errors": [], "simulated": False}


def test_replay_has_complete_binary_function_launch_join_and_no_qualification():
    raw = raw_capture()
    result = kernel_origin.summarize(raw)
    assert result["status"] == "passed"
    assert result["calls"][0]["custom_exclusive_fraction_lower_bound"] == 1
    assert result["raw_sha256"] == fingerprint(raw)
    assert result["formal_qualification_implemented"] is False
    assert result["timings_are_diagnostic"] is True
    assert not result["output_causal_use_proven"] and not result["uninstrumented_path_equivalence_proven"]


@pytest.mark.parametrize("change", [
    {"proofs": []}, {"launches": []}, {"capture_complete": False}, {"source_intact": False},
    {"hooks_intact": False}, {"dropped_records": 1}, {"errors": ["incomplete callback"]}, {"simulated": True},
])
def test_incomplete_capture_or_missing_chain_stays_unknown(change):
    raw = raw_capture()
    raw.update(change)
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


@pytest.mark.parametrize("change", [{"graph_id": 4}, {"graph_node_id": 5}, {"kind": "memcpy"}, {"kind": "memset"},
                                    {"correlation": 88}, {"context_id": 99}])
def test_graph_memory_and_unassociated_activities_never_receive_own_credit(change):
    raw = raw_capture()
    raw["activities"][0].update(change)
    result = kernel_origin.summarize(raw)
    assert result["status"] == "not_measured"
    assert result["calls"][0]["custom_exclusive_fraction_lower_bound"] == 0


@pytest.mark.parametrize("change", [{"exit_success": False}, {"call": None}, {"call": 2}, {"function": 99}])
def test_launch_must_succeed_in_this_call_and_have_a_proven_function_generation(change):
    raw = raw_capture()
    raw["launches"][0].update(change)
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


def test_names_and_declarations_cannot_turn_framework_activity_into_own_work():
    raw = raw_capture()
    raw["proofs"] = []
    raw["activities"][0].update(name="handwritten_candidate_kernel", origin="custom", custom=True)
    raw["origin_map"] = {"10": "custom"}
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


def test_concurrent_unknown_work_reduces_exclusive_time_and_keeps_attribution_unknown():
    raw = raw_capture()
    raw["activities"].append({**raw["activities"][0], "start_ns": 170, "end_ns": 270, "correlation": 90})
    result = kernel_origin.summarize(raw)["calls"][0]
    assert result["device_busy_ns"] == 150
    assert result["custom_exclusive_ns"] == 50
    assert result["custom_exclusive_fraction_lower_bound"] == pytest.approx(1 / 3)
    assert result["status"] == "not_measured"


def test_multi_stream_overlap_is_counted_once_and_scope_clips_outside_work():
    raw = raw_capture()
    raw["activities"].extend([{**raw["activities"][0], "start_ns": 190, "end_ns": 250, "stream": 8},
                              {**raw["activities"][0], "start_ns": 310, "end_ns": 350, "correlation": 90},
                              {**raw["activities"][0], "device": 1, "correlation": 91}])
    result = kernel_origin.summarize(raw)["calls"][0]
    assert result["device_busy_ns"] == 130 and result["kernel_count"] == 2
    assert result["custom_exclusive_fraction_lower_bound"] == 1


@pytest.mark.parametrize("change", [{"source_sha256": "2" * 64}, {"binary_sha256": "3" * 64},
                                    {"compiler": "trust-me"}, {"compiler_metadata_sha256": "bad"}])
def test_compiler_source_binary_and_version_mismatches_are_rejected(change):
    raw = raw_capture()
    raw["proofs"][0].update(change)
    with pytest.raises(ValueError, match="Compiler proof"):
        kernel_origin.summarize(raw)


@pytest.mark.parametrize("label", ["modules", "functions", "proofs", "windows", "launches"])
def test_ambiguous_identity_records_are_rejected(label):
    raw = raw_capture()
    raw[label].append(copy.deepcopy(raw[label][0]))
    with pytest.raises(ValueError, match="[Aa]mbiguous|duplicated"):
        kernel_origin.summarize(raw)


@pytest.mark.parametrize("change", [{"output_verified": False}, {"no_grad": False}, {"training": None}, {"training_after": False},
                                    {"start_ns": 300}, {"seed": True}, {"input_signature": None}])
def test_unverified_or_unbound_mode_and_input_windows_are_rejected(change):
    raw = raw_capture()
    raw["windows"][0].update(change)
    with pytest.raises(ValueError, match="candidate call"):
        kernel_origin.summarize(raw)


def test_overlapping_call_windows_and_absent_activity_do_not_pass():
    raw = raw_capture()
    raw["windows"].append({**raw["windows"][0], "id": 2})
    with pytest.raises(ValueError, match="overlapping"):
        kernel_origin.summarize(raw)
    raw["windows"].pop()
    raw["activities"] = []
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


@pytest.mark.parametrize("value", [0, -1, 1.01, float("nan"), float("inf"), True, "0.95"])
def test_invalid_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="fraction"):
        kernel_origin.summarize(raw_capture(), minimum_custom_fraction=value)


def test_inspectable_source_preserves_future_import_globals_and_code_object_identity(tmp_path):
    source = "from __future__ import annotations\nVALUE = 7\ndef kernel(x: Missing):\n    return VALUE + x\n"
    registry, context = SourceRegistry(), {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    fn = context["kernel"]
    assert fn(1) == 8 and fn.__globals__ is sys.modules[fn.__module__].__dict__
    assert context["VALUE"] == fn.__globals__["VALUE"] == 7
    assert fn.__annotations__ == {"x": "Missing"}
    assert inspect.getsource(fn) == "def kernel(x: Missing):\n    return VALUE + x\n"
    assert registry.source_for(fn)["source_sha256"] == sha256(source.encode())
    assert registry.callable_identity(fn)["kind"] == "registered_code"
    assert registry.intact()
    # Code objects compare equal by value. Equal content recompiled under the
    # exact same filename still does not inherit the registered identity.
    other = {}
    exec(compile(source, context["__file__"], "exec"), other)
    assert other["kernel"].__code__ == fn.__code__ and other["kernel"].__code__ is not fn.__code__
    assert registry.source_for(other["kernel"]) is None
    assert registry.callable_identity(other["kernel"])["kind"] == "external_code_observation"


def test_nested_code_is_retained_and_source_mutation_breaks_identity(tmp_path):
    source = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    registry, context = SourceRegistry(), {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    assert registry.source_for(context["outer"]()) is not None
    Path(context["__file__"]).write_text(source + "# changed\n")
    assert not registry.intact()


def test_loader_does_not_inherit_its_own_future_annotation_flags(tmp_path):
    context = {}
    execute("def kernel(x: int) -> int:\n    return x\n", context, tmp_path)
    assert context["kernel"].__annotations__ == {"x": int, "return": int}


def test_jit_text_mutation_is_not_the_registered_source(tmp_path):
    registry, context = SourceRegistry(), {}
    with record_sources(registry):
        execute(SOURCE, context, tmp_path)
    jit = NS(fn=context["kernel"], src=SOURCE)
    assert registry.jit_source(jit)["jit_source_sha256"] == sha256(SOURCE.encode())
    jit.src = "def kernel(x):\n    return 99\n"
    assert registry.jit_source(jit) is None


@pytest.mark.parametrize("kind", ["symlink", "fifo", "different"])
def test_frozen_source_never_follows_or_blocks_on_preexisting_special_files(tmp_path, kind):
    target = tmp_path / ("candidate_" + sha256(SOURCE.encode()) + ".py")
    if kind == "symlink":
        link = tmp_path / "elsewhere.py"
        link.write_text(SOURCE)
        target.symlink_to(link)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.write_text("# wrong source\n")
    with pytest.raises((OSError, ValueError)):
        execute(SOURCE, {}, tmp_path)


def fake_sdk():
    calls = []

    def drops(context, stream, pointer):
        calls.append(("drops", context, stream))
        C.c_size_t.from_address(pointer).value = 0

    sdk = NS(CallbackDomain=NS(DRIVER_API=1, RUNTIME_API=2, RESOURCE=3), CallbackIdResource=NS(MODULE_LOADED=8),
             ApiCallbackSite=NS(API_ENTER=0, API_EXIT=1), ActivityKind=NS(DRIVER=4, RUNTIME=5, CONCURRENT_KERNEL=10, MEMCPY=1, MEMSET=2),
             activity_register_callbacks=lambda *args: calls.append(("register",)),
             subscribe=lambda *args: 99, enable_domain=lambda *args: calls.append(("domain", *args)),
             activity_enable=lambda value: calls.append(("enable", value)),
             activity_disable=lambda value: calls.append(("disable", value)),
             unsubscribe=lambda value: calls.append(("unsubscribe", value)),
             activity_flush_all=lambda value: calls.append(("flush", value)),
             activity_get_num_dropped_records=drops)
    return sdk, calls


def observe_load(capture, module_handle=100, function_handle=200, *, correlation=1, binary=BINARY):
    sdk = capture.cupti
    result = C.c_int(0)
    module = C.c_void_p()
    params = cuda_origin.ModuleLoad(C.pointer(module))
    data = NS(function_name="cuModuleLoadData", callback_site=0, context=123, context_uid=9,
              function_params=C.addressof(params), correlation_id=correlation, function_return_value=C.addressof(result))
    capture.callback(None, sdk.CallbackDomain.DRIVER_API, 0, data)
    blob = C.create_string_buffer(binary)
    resource = cuda_origin.ModuleResource(7, len(binary), C.addressof(blob))
    capture.callback(None, sdk.CallbackDomain.RESOURCE, sdk.CallbackIdResource.MODULE_LOADED,
                     NS(resource_descriptor=C.addressof(resource), context=123))
    module.value = module_handle
    data.callback_site = 1
    capture.callback(None, sdk.CallbackDomain.DRIVER_API, 0, data)
    function = C.c_void_p(function_handle)
    lookup = cuda_origin.ModuleGetFunction(C.pointer(function), module_handle, None)
    data.function_name = "cuModuleGetFunction"
    data.function_params = C.addressof(lookup)
    capture.callback(None, sdk.CallbackDomain.DRIVER_API, 0, data)


def test_cpu_callback_chain_copies_full_binary_and_joins_real_function_handle(tmp_path):
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.frontend = NS(compiled_frontend_for=lambda source: frontend_receipt(), compiled_for=lambda kernel: {
        "protocol": "kernelbench_pro_triton_emission_v1", "backend": "cuda", "binary_sha256": sha256(BINARY),
        "frontend_receipt_sha256": frontend_receipt()["receipt_sha256"],
        "metadata_sha256": fingerprint(dict(kernel.metadata._asdict()))})
    registry, context = capture.registry, {}
    with record_sources(registry):
        execute(SOURCE, context, tmp_path)
    observe_load(capture)
    metadata = namedtuple("Metadata", "shared")(0)
    compiled = NS(src=NS(fn=NS(fn=context["kernel"], src=SOURCE)), kernel=BINARY,
                  module=100, function=200, metadata=metadata)
    capture.bind_triton(compiled)
    assert capture.raw["modules"][0]["binary_sha256"] == sha256(BINARY)
    assert capture.raw["modules"][0]["binary_bytes"] == len(BINARY)
    assert capture.raw["proofs"][0]["function"] == 1
    assert capture.raw["proofs"][0]["source_sha256"] == sha256(SOURCE.encode())
    capture.bind_triton(compiled)
    assert len(capture.raw["proofs"]) == 1
    assert capture.raw["errors"] == []


@pytest.mark.parametrize("mismatch", ["binary", "module", "function", "source"])
def test_compiler_hook_refuses_unobserved_or_mismatched_actual_load(tmp_path, mismatch):
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.frontend = NS(compiled_frontend_for=lambda source: frontend_receipt(), compiled_for=lambda kernel: {
        "protocol": "kernelbench_pro_triton_emission_v1", "backend": "cuda", "binary_sha256": sha256(BINARY),
        "frontend_receipt_sha256": frontend_receipt()["receipt_sha256"],
        "metadata_sha256": fingerprint(dict(kernel.metadata._asdict()))})
    context = {}
    with record_sources(capture.registry):
        execute(SOURCE, context, tmp_path)
    observe_load(capture)
    compiled = NS(src=NS(fn=NS(fn=context["kernel"], src=SOURCE)), kernel=BINARY,
                  module=100, function=200, metadata=namedtuple("Metadata", "shared")(0))
    if mismatch == "source":
        compiled.src.fn.src += "# altered\n"
    else:
        setattr(compiled, {"binary": "kernel"}.get(mismatch, mismatch), b"wrong" if mismatch == "binary" else 900)
    capture.bind_triton(compiled)
    assert capture.raw["proofs"] == []


@pytest.mark.parametrize("api", ["cuLaunchKernel", "cuLaunchKernel_ptsz", "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"])
def test_driver_launch_captures_scoped_function_generation_and_exit_status(api):
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    observe_load(capture)
    capture.active_calls[__import__("threading").get_ident()] = 1
    params = cuda_origin.LaunchKernelEx(None, 200, None, None) if "Ex" in api else cuda_origin.LaunchKernel(function=200)
    result = C.c_int(0)
    data = NS(function_name=api, callback_site=0, context=123, context_uid=9,
              function_params=C.addressof(params), correlation_id=10, function_return_value=C.addressof(result))
    capture.callback(None, 1, 0, data)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    assert capture.raw["launches"] == [{"context_id": 9, "correlation": 10, "runtime_correlation": None, "function": 1, "call": 1,
                                        "exit_success": True, "api": api}]
    assert capture.pending_launches == {}


def test_unload_and_handle_reuse_get_new_generations_and_no_inherited_proof():
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    observe_load(capture)
    result, params = C.c_int(0), cuda_origin.ModuleUnload(100)
    capture.callback(None, 1, 0, NS(function_name="cuModuleUnload", callback_site=1, context=123, context_uid=9,
                     function_params=C.addressof(params), correlation_id=2, function_return_value=C.addressof(result)))
    assert not capture.function_map and not capture.module_map
    observe_load(capture, correlation=3, binary=b"different binary")
    assert capture.function_map[(123, 200)]["id"] == 2
    assert capture.raw["modules"][1]["id"] == 2


def test_missing_module_descriptor_and_oversized_binary_remain_errors():
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.module_loaded(NS(resource_descriptor=0))
    resource = cuda_origin.ModuleResource(7, cuda_origin.MAX_CUBIN_BYTES + 1, 1)
    capture.module_loaded(NS(resource_descriptor=C.addressof(resource)))
    assert len(capture.raw["errors"]) == 2 and not capture.raw["modules"]


def test_stop_queries_only_actual_context_stream_queues_and_cleans_up():
    sdk, calls = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.start()
    capture.contexts.update({9: 123, 10: 456})
    capture.raw["activities"] = [{"context_id": 9, "stream": 7}, {"context_id": 10, "stream": 8}]
    capture.stop()
    assert {(row[1], row[2]) for row in calls if row[0] == "drops"} == {(0, 0), (123, 0), (456, 0), (123, 7), (456, 8)}
    assert calls[-1] == ("unsubscribe", 99)
    assert capture.raw["capture_complete"] is True and capture.raw["simulated"] is True


def test_partial_start_and_failed_flush_still_disable_and_unsubscribe():
    sdk, calls = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.start()

    def fail(*args):
        raise RuntimeError("synthetic flush failure")

    sdk.activity_flush_all = fail
    capture.stop()
    assert calls[-1] == ("unsubscribe", 99)
    assert sum(row[0] == "disable" for row in calls) == 5
    assert not capture.raw["capture_complete"] and capture.raw["errors"]


def test_real_triton_jit_can_inspect_frozen_candidate_without_a_device(tmp_path):
    pytest.importorskip("triton")
    source = "import triton\n@triton.jit\ndef kernel(x):\n    pass\n"
    registry, context = SourceRegistry(), {}
    with record_sources(registry):
        execute(source, context, tmp_path)
    assert registry.jit_source(context["kernel"])["source_sha256"] == sha256(source.encode())


def test_real_triton_hook_tampering_is_detected_and_original_is_restored():
    pytest.importorskip("triton")
    from triton.compiler.compiler import CompiledKernel

    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    original = CompiledKernel._init_handles
    try:
        capture.install_triton()
        assert CompiledKernel._init_handles is not original
        CompiledKernel._init_handles = lambda compiled: None
        capture.stop()
        assert capture.raw["hooks_intact"] is False
        assert CompiledKernel._init_handles is original
    finally:
        CompiledKernel._init_handles = original


def test_activity_callback_copies_vendor_fields_and_incomplete_stream_accounting_is_unknown():
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.buffer_completed([NS(kind=10, start=100, end=120, device_id=0, stream_id=7,
                                  context_id=9, correlation_id=10, graph_id=2, graph_node_id=3, name="test")])
    row = capture.raw["activities"][0]
    assert row["graph_id"] == 2 and row["graph_node_id"] == 3 and row["kind"] == "kernel"
    capture.stop()
    assert "drop accounting" in capture.raw["errors"][0]


def test_start_failure_cleans_up_enabled_collection_and_subscriber(monkeypatch):
    sdk, calls = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)

    def fail():
        raise RuntimeError("synthetic compiler hook failure")

    monkeypatch.setattr(capture, "install_triton", fail)
    with pytest.raises(RuntimeError, match="compiler hook"):
        with capture.recording():
            pytest.fail("failed start entered candidate")
    assert ("unsubscribe", 99) in calls and sum(row[0] == "disable" for row in calls) == 5


def test_submission_loader_handles_build_path_quotes_and_future_annotations(tmp_path):
    from src.eval import load_custom_model

    build = tmp_path / "build's directory"
    context = {}
    model = load_custom_model("from __future__ import annotations\nclass ModelNew:\n    pass\n", context, str(build))
    assert model.__name__ == "ModelNew" and context["BUILD_DIRECTORY"] == str(build)
    assert inspect.getsource(model).startswith("class ModelNew")


def test_frozen_module_exposes_imports_to_decorators_and_does_not_replace_main(tmp_path):
    source = ("from __future__ import annotations\nimport dataclasses\nimport typing\n"
              "@dataclasses.dataclass\nclass Entry:\n    value: int\n    CONST: typing.ClassVar[int] = 2\n")
    main = sys.modules["__main__"]
    context = {"__name__": "__main__"}
    execute(source, context, tmp_path)
    assert sys.modules["__main__"] is main
    assert [field.name for field in __import__("dataclasses").fields(context["Entry"])] == ["value"]
    assert context["Entry"](4).value == 4


def test_lightweight_import_and_cli_help_do_not_import_vendor_sdks():
    code = f'''import sys
sys.path.insert(0, {str(ROOT)!r})
from kb_workflows.provenance import cuda_origin, kernel_origin, origin_task, submission_source
from scripts.audit_kernel_origin import main
assert not any(name in sys.modules for name in ("torch", "triton", "cupti", "harbor", "modal"))
'''
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


def make_config(tmp_path):
    from kb_workflows.common import kernel_name
    from kb_workflows.config import RunConfig

    config = RunConfig(run_name="saved", runs_dir=str(tmp_path), start=2, end=2,
                       evaluation="local", generation="local").validate()
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 2)).write_text(SOURCE)
    path = tmp_path / "config.toml"
    path.write_text(f'[run]\nrun_name = "saved"\nruns_dir = {json.dumps(str(tmp_path))}\nstart = 2\nend = 2\n'
                    'evaluation = "local"\ngeneration = "local"\n')
    return config, path


def test_origin_cli_previews_without_allocating_output_or_dispatching(tmp_path, monkeypatch, capsys):
    from scripts import audit_kernel_origin as cli

    _, path = make_config(tmp_path)
    monkeypatch.setattr(cli, "evaluate_tasks", lambda *args: pytest.fail("preview dispatched execution"))
    output = tmp_path / "new/audit.json"
    assert cli.main(["--config", str(path), "--problem", "2", "--output", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["experiments_started"] is False and preview["formal_qualification_implemented"] is False
    assert not output.parent.exists()


def test_origin_cli_simulated_dispatch_keeps_source_run_and_scores_untouched(tmp_path, monkeypatch):
    from scripts import audit_kernel_origin as cli

    config, path = make_config(tmp_path)
    before = {str(p): p.read_bytes() for p in config.run_dir.iterdir()}
    observed = []

    def dispatch(cfg, payloads):
        observed.append((cfg, payloads))
        return iter([("origin", {"status": "ok", "simulated": True})])

    monkeypatch.setattr(cli, "evaluate_tasks", dispatch)
    output = tmp_path / "audit.json"
    assert cli.main(["--config", str(path), "--problem", "2", "--output", str(output), "--execute"]) == 0
    actual_cfg, rows = observed[0]
    assert actual_cfg.run_dir != config.run_dir and rows[0][1]["task"] == "origin_audit"
    assert rows[0][1]["eval_config"]["measure_performance"] is True
    assert rows[0][1]["eval_config"]["use_hidden_tests"] is False
    assert {str(p): p.read_bytes() for p in config.run_dir.iterdir()} == before
    assert json.loads(output.read_text())["simulated"] is True


def test_origin_prepare_rejects_outside_population_neuron_and_bad_policy(tmp_path):
    from scripts.audit_kernel_origin import prepare

    config, _ = make_config(tmp_path)
    with pytest.raises(ValueError, match="population"):
        prepare(config, 1, 0, minimum_custom_fraction=0.95)
    for threshold in (0, 1.1, True, float("nan")):
        with pytest.raises(ValueError):
            prepare(config, 2, 0, minimum_custom_fraction=threshold)
    config.device_backend = "neuron"
    with pytest.raises(ValueError, match="Neuron"):
        prepare(config, 2, 0, minimum_custom_fraction=0.95)


def test_origin_dispatch_does_not_fall_into_neuron_evaluator(monkeypatch):
    from kb_workflows.evaluation import evaluate_task
    from kb_workflows.provenance import origin_task

    monkeypatch.setattr(origin_task, "audit", lambda payload: {"origin_diagnostic": True})
    from kb_workflows.config import RunConfig
    from kb_workflows.engine import task_payload

    payload = task_payload(RunConfig(start=2, end=2).validate(), 2, task="baseline")
    payload["task"] = "origin_audit"
    payload["eval_config"]["device_backend"] = "neuron"
    assert evaluate_task.dispatch(payload) == {"origin_diagnostic": True}


def test_origin_worker_rejects_incomplete_requests_before_sdk_import():
    from kb_workflows.provenance.origin_task import audit

    for settings in ({"use_hidden_tests": True}, {"device_backend": "neuron"}):
        with pytest.raises(ValueError):
            audit({"eval_config": settings})




def graph_capture():
    sdk, calls = fake_sdk()
    resources = {"GRAPHNODE_CREATED": 13, "GRAPHNODE_CLONED": 20,
                 "GRAPH_NODE_UPDATED": 22, "GRAPH_NODE_SET_PARAMS": 23}
    for name, value in resources.items():
        setattr(sdk.CallbackIdResource, name, value)
    descriptors = {}
    sdk.GraphData = NS(from_ptr=lambda pointer, **kwargs: descriptors[pointer])
    sdk.get_graph_node_id = lambda node: node + 1000
    sdk.get_graph_id = lambda graph: graph + 10000
    sdk.get_graph_exec_id = lambda executable: executable + 20000
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    observe_load(capture)
    return capture, descriptors, calls


def driver_data(name, params, correlation, *, entering=True, return_code=0):
    result = C.c_int(return_code)
    return NS(function_name=name, function_params=C.addressof(params), context=123, context_uid=9,
              correlation_id=correlation, callback_site=0 if entering else 1,
              function_return_value=C.addressof(result), _keepalive=(result, params))


def graph_resource(capture, descriptors, *, cloned=False, graph=77, node=88, original=0, node_type=0):
    pointer = len(descriptors) + 1
    descriptors[pointer] = NS(graph=graph, node=node, node_type=node_type, original_node=original)
    kind = capture.cupti.CallbackIdResource.GRAPHNODE_CLONED if cloned else capture.cupti.CallbackIdResource.GRAPHNODE_CREATED
    capture.callback(None, 3, kind, NS(resource_descriptor=pointer, context=123))


def build_graph(capture, descriptors, *, clone=False, memory_kind=None):
    from kb_workflows.provenance.origin_graphs import Instantiate

    api, params = ("cuLaunchKernel", cuda_origin.LaunchKernel(function=200)) if memory_kind is None else (
        "cuMemcpyDtoDAsync" if memory_kind == "memcpy" else "cuMemsetD8Async", C.c_void_p())
    data = driver_data(api, params, 20)
    capture.callback(None, 1, 0, data)
    graph_resource(capture, descriptors, node_type={None: 0, "memcpy": 1, "memset": 2}[memory_kind])
    assert capture.raw["graph_nodes"][-1]["creation_success"] is False
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    executable = C.c_void_p(99)
    data = driver_data("cuGraphInstantiate_v2", Instantiate(C.pointer(executable), 77), 30)
    capture.callback(None, 1, 0, data)
    if clone:
        graph_resource(capture, descriptors, cloned=True, graph=78, node=89, original=88)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)


def replay_graph(capture, *, correlation=50, runtime_correlation=None, return_code=0, executable=99):
    runtime = None
    if runtime_correlation is not None:
        runtime = driver_data("cudaGraphLaunch", C.c_void_p(executable), runtime_correlation)
        capture.callback(None, 2, 0, runtime)
    capture.active_calls[__import__("threading").get_ident()] = 1
    data = driver_data("cuGraphLaunch", C.c_void_p(executable), correlation, return_code=return_code)
    capture.callback(None, 1, 0, data)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    capture.active_calls.clear()
    if runtime is not None:
        runtime.callback_site = 1
        capture.callback(None, 2, 0, runtime)


def graph_fixture(capture, *, node=1088, correlation=50):
    """Serialized synthetic parser fixture, not a real device receipt."""
    original = raw_capture()
    raw = copy.deepcopy(capture.raw)
    raw.update(windows=original["windows"], proofs=original["proofs"], capture_complete=True,
               hooks_intact=True, source_intact=True, simulated=False)
    raw["activities"] = [{**original["activities"][0], "correlation": correlation,
                           "graph_id": 20099, "graph_node_id": node}]
    return raw


@pytest.mark.parametrize("clone", [False, True])
def test_graph_replay_joins_created_or_cloned_node_to_the_actual_function(clone):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors, clone=clone)
    replay_graph(capture)
    raw = graph_fixture(capture, node=1089 if clone else 1088)
    assert raw["errors"] == [] and capture.graphs.balanced()
    result = kernel_origin.summarize(raw)
    assert result["status"] == "passed" and result["graph_attribution_implemented"] is True
    assert kernel_origin.summarize(json.loads(json.dumps(raw)))["raw_sha256"] == result["raw_sha256"]
    assert result["formal_qualification_implemented"] is False


def test_graph_runtime_parent_correlation_matches_driver_snapshot():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    replay_graph(capture, runtime_correlation=500)
    raw = graph_fixture(capture, correlation=500)
    assert capture.runtime_calls[__import__("threading").get_ident()] == []
    assert kernel_origin.summarize(raw)["status"] == "passed"


def test_ambiguous_runtime_parent_launch_cannot_assign_one_function_or_graph():
    raw = raw_capture()
    raw["launches"][0]["runtime_correlation"] = 500
    raw["launches"].append({**raw["launches"][0], "correlation": 11})
    raw["activities"][0]["correlation"] = 500
    assert kernel_origin.summarize(raw)["status"] == "not_measured"
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    replay_graph(capture, runtime_correlation=500)
    replay_graph(capture, correlation=51, runtime_correlation=500)
    assert kernel_origin.summarize(graph_fixture(capture, correlation=500))["status"] == "not_measured"


def test_destroying_graph_template_does_not_erase_instantiated_execution_origin():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    data = driver_data("cuGraphDestroy", C.c_void_p(77), 35)
    capture.callback(None, 1, 0, data)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    assert not capture.graphs.nodes and capture.graphs.executables
    replay_graph(capture)
    assert kernel_origin.summarize(graph_fixture(capture))["status"] == "passed"


@pytest.mark.parametrize("operation", ["cuGraphExecDestroy", "cuGraphExecNodeSetParams", "cuGraphNodeSetParams", "cuGraphExecUpdate"])
def test_destroyed_or_updated_executable_does_not_reuse_previous_function_origin(operation):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    replay_graph(capture, correlation=49)
    before = copy.deepcopy(capture.raw["graph_launches"][0])
    data = driver_data(operation, C.c_void_p(99), 40)
    capture.callback(None, 1, 0, data)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    replay_graph(capture)
    assert capture.raw["graph_launches"][0] == before
    assert kernel_origin.summarize(graph_fixture(capture))["status"] == "not_measured"


def test_unknown_graph_cloning_source_never_inherits_an_arbitrary_function():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    graph_resource(capture, descriptors, cloned=True, graph=78, node=89, original=999)
    assert capture.raw["graph_nodes"][-1]["function"] is None
    assert capture.raw["graph_nodes"][-1]["creation_success"] is False


@pytest.mark.parametrize("change", [{"graph_id": 999}, {"graph_node_id": 999}, {"context_id": 999}, {"correlation": 999}])
def test_graph_activity_requires_this_observed_graph_node_context_and_launch(change):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    replay_graph(capture)
    raw = graph_fixture(capture)
    raw["activities"][0].update(change)
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


def test_failed_graph_launch_or_unfinished_node_creation_is_unknown():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    replay_graph(capture, return_code=1)
    assert kernel_origin.summarize(graph_fixture(capture))["status"] == "not_measured"
    capture.raw["graph_launches"][0]["exit_success"] = True
    raw = graph_fixture(capture)
    raw["graph_nodes"][0]["creation_success"] = False
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


def test_graph_clone_and_snapshot_tampering_cannot_rebind_a_function_generation():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors, clone=True)
    replay_graph(capture)
    raw = graph_fixture(capture, node=1089)
    raw["graph_nodes"][1]["original"] = 2
    with pytest.raises(ValueError, match="clone"):
        kernel_origin.summarize(raw)
    raw = graph_fixture(capture, node=1089)
    raw["graph_launches"][0]["nodes"]["1089"] = 1
    with pytest.raises(ValueError, match="snapshot"):
        kernel_origin.summarize(raw)


def test_runtime_and_driver_activity_are_enabled_for_vendor_correlation_ids():
    sdk, calls = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.start()
    capture.stop()
    enabled = {row[1] for row in calls if row[0] == "enable"}
    assert sdk.ActivityKind.DRIVER in enabled and sdk.ActivityKind.RUNTIME in enabled
    assert ("domain", 1, 99, sdk.CallbackDomain.RUNTIME_API) in calls


def library_receipt(binary=BINARY, *, distribution="nvidia-cublas", origin="library"):
    """Synthetic serialized schema for pure replay tests; no installed DSO read."""
    import base64

    from kb_workflows.provenance.origin_binaries import PROTOCOL

    source = {"id": 1, "distribution": distribution, "origin": origin, "version": "1.0",
              "file_sha256": "2" * 64, "record_sha256": "3" * 64, "file_bytes": 100,
              "relative_path": "package/lib.so", "path": "/synthetic/package/lib.so"}
    frozen = {"protocol": PROTOCOL, "sources": [source], "manifest_sha256": fingerprint({"sources": [source]})}
    document = {"protocol": PROTOCOL, "manifest": frozen, "tool": {"sha256": "4" * 64}, "simulated": False,
                "sources": [{"source": 1, "status": "measured", "extraction": {"exit_code": 0, "timed_out": False,
                             "log_base64": base64.b64encode(b"extracted").decode(), "log_sha256": sha256(b"extracted")}}],
                "cubins": [{"source": 1, "binary_sha256": sha256(binary), "binary_bytes": len(binary)}], "errors": []}
    document["receipt_sha256"] = fingerprint(document)
    return document


@pytest.mark.parametrize("distribution,origin", [("nvidia-cublas", "library"), ("torch", "framework")])
def test_loaded_binary_ownership_separates_custom_and_library_track_verdicts(distribution, origin):
    raw = raw_capture()
    raw["binary_inventory"] = library_receipt(distribution=distribution, origin=origin)
    result = kernel_origin.summarize(raw)
    assert result["status"] == "failed" and result["track_verdicts"] == {"custom": "failed", "library": "passed"}
    assert result["calls"][0]["by_origin_union_ns"][origin] == 100
    assert result["calls"][0]["custom_exclusive_fraction_lower_bound"] == 0
    assert result["calls"][0]["unknown_activity_count"] == 0
    assert not result["formal_qualification_implemented"]


def test_mixed_real_binary_identities_apply_the_custom_fraction_to_interval_union():
    raw = raw_capture()
    other = b"second source-bound cubin"
    raw["binary_inventory"] = library_receipt(other)
    raw["modules"].append({**raw["modules"][0], "id": 2, "handle": 101, "binary_sha256": sha256(other), "binary_bytes": len(other)})
    raw["functions"].append({**raw["functions"][0], "id": 2, "module": 2, "handle": 201})
    raw["launches"].append({**raw["launches"][0], "correlation": 11, "function": 2})
    raw["activities"][0]["end_ns"] = 210
    raw["activities"].append({**raw["activities"][0], "start_ns": 210, "end_ns": 220, "correlation": 11})
    result = kernel_origin.summarize(raw)
    assert result["status"] == "failed" and result["track_verdicts"]["library"] == "passed"
    assert result["calls"][0]["custom_exclusive_fraction_lower_bound"] == 0.9
    assert kernel_origin.summarize(raw, minimum_custom_fraction=0.85)["status"] == "passed"
    raw["activities"][1].update(start_ns=170, end_ns=270)
    result = kernel_origin.summarize(raw)["calls"][0]
    assert result["device_busy_ns"] == 150
    assert result["by_origin_union_ns"] == {"custom": 90, "library": 100, "framework": 0, "runtime": 0, "unknown": 0}
    assert result["custom_exclusive_ns"] == 50


def test_unverified_wheel_or_simulated_extraction_does_not_assign_library_ownership():
    for change in ({"simulated": True}, {"receipt_sha256": "0" * 64}):
        raw = raw_capture()
        raw["proofs"] = []
        raw["binary_inventory"] = library_receipt()
        raw["binary_inventory"].update(change)
        assert kernel_origin.summarize(raw)["track_verdicts"] == {"custom": "not_measured", "library": "not_measured"}


@pytest.mark.parametrize("api,domain,kind", [("cudaMemcpyAsync", 2, "memcpy"), ("cuMemcpyDtoDAsync", 1, "memcpy"),
                                          ("cudaMemsetAsync", 2, "memset"), ("cuMemsetD32Async", 1, "memset")])
def test_native_memory_api_activity_receives_runtime_credit_and_no_custom_credit(api, domain, kind):
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk)
    capture.active_calls[__import__("threading").get_ident()] = 1
    data = driver_data(api, C.c_void_p(), 60)
    capture.callback(None, domain, 0, data)
    data.callback_site = 1
    capture.callback(None, domain, 0, data)
    raw = raw_capture()
    raw["memory_calls"] = capture.raw["memory_calls"]
    raw["activities"][0].update(kind=kind, correlation=60)
    assert capture.raw["errors"] == []
    result = kernel_origin.summarize(raw)
    assert result["track_verdicts"] == {"custom": "failed", "library": "passed"}
    assert result["calls"][0]["by_origin_union_ns"]["runtime"] == 100
    raw["memory_calls"][0]["exit_success"] = False
    assert kernel_origin.summarize(raw)["status"] == "not_measured"
    raw["memory_calls"][0].update(exit_success=True, call=2)
    assert kernel_origin.summarize(raw)["status"] == "not_measured"


@pytest.mark.parametrize("kind", ["memcpy", "memset"])
def test_graph_memory_node_retains_native_kind_and_never_counts_as_custom(kind):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors, memory_kind=kind)
    replay_graph(capture)
    raw = graph_fixture(capture)
    raw["activities"][0]["kind"] = kind
    assert raw["graph_nodes"][0]["kind"] == kind
    assert raw["graph_nodes"][0]["creation_success"] is True
    result = kernel_origin.summarize(raw)
    assert result["track_verdicts"] == {"custom": "failed", "library": "passed"}
    assert result["calls"][0]["by_origin_union_ns"]["runtime"] == 100


def test_queue_accepts_source_bound_origin_requests_and_requires_a_candidate(tmp_path):
    from kb_workflows.runtime.queue_worker import validate_payload
    from scripts.audit_kernel_origin import prepare

    config, _ = make_config(tmp_path)
    payload = prepare(config, 2, 0, minimum_custom_fraction=0.95)
    validate_payload(payload)
    payload.pop("kernel_src")
    with pytest.raises(ValueError):
        validate_payload(payload)


@pytest.fixture
def native_origin_runner(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import torch

    from kb_workflows.evaluation import evaluate_task, measurement, paired_task
    from kb_workflows.provenance import origin_binaries, origin_task
    from src import eval as native
    from src.input_groups import copy_inputs
    from src.validation import compare_outputs

    class FakeCapture(cuda_origin.Capture):
        def __init__(self, source, **options):
            sdk, _ = fake_sdk()
            ticks = iter(range(1000, 100_000_000, 1000))
            sdk.get_timestamp = lambda: next(ticks)
            super().__init__(source, bindings=sdk, **options)

        @contextmanager
        def recording(self):
            with record_sources(self.registry):
                yield self
            self.raw.update(capture_complete=True, source_intact=self.registry.intact(), hooks_intact=True)

    class Reference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x + self.calls + 1

    hidden = tmp_path / "private_hidden.py"
    hidden.write_text("import torch\ndef get_hidden_inputs():\n    return [[torch.rand(4)]]\n")
    action = {"damage": None, "activities": False, "caps": None}
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_USE_CUDA_DSA", "1")
    monkeypatch.setenv("KB_FP32_TOL", "0.0001")
    monkeypatch.setattr(cuda_origin, "Capture", FakeCapture)
    monkeypatch.setattr(origin_binaries, "Catalog", lambda: NS(collect=lambda **kwargs: {"simulated": True}))
    monkeypatch.setattr(evaluate_task, "validate_sources", lambda payload: "CPU parser fixture, not measured source evidence")
    monkeypatch.setattr(native, "set_seed", lambda seed: torch.random.default_generator.manual_seed(seed))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 16)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "SIMULATED")

    def invoke(payload):
        build = Path(payload["build_root"]) / fingerprint(payload)
        os.environ.update(TORCH_EXTENSIONS_DIR=str(build), TORCH_USE_CUDA_DSA="1", KB_FP32_TOL="0.0001")
        Candidate = native.load_custom_model(payload["kernel_src"], {}, str(build))
        metadata = {"device_fingerprint": action["caps"] or {"available": False, "name": "SIMULATED"},
                    "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
                    "kernel_sha256": fingerprint({"source": payload["kernel_src"]}), "precision": "fp32"}
        with torch.no_grad():
            candidate, reference = Candidate(), Reference()
            if payload["task"] == "paired_timing":
                metadata["paired_timing"] = paired_task.collect_pairs(
                    Reference(), candidate, Reference(), lambda _: [torch.rand(4)], Reference(),
                    device="cpu", trials=10, seed=42, set_seed=native.set_seed)
                return {"compiled": True, "correctness": True, "metadata": metadata}
            outcome = native.run_and_check_correctness(reference, candidate, lambda: [torch.rand(4)], metadata,
                                                       num_correct_trials=1, device="cpu", precision=torch.float32)
            if payload["eval_config"]["use_hidden_tests"] and outcome.correctness:
                outcome.correctness = native.run_hidden_correctness_check(reference, candidate, str(hidden), metadata,
                                                                          device="cpu", precision=torch.float32)
            if payload["eval_config"]["measure_performance"] and outcome.correctness:
                native.set_seed(42)
                memory_inputs = [torch.rand(4)]
                receipt = metadata["timing_receipt"] = {}
                def inputs(trial):
                    seed = measurement.fresh_seed()
                    receipt.setdefault("input_seeds", []).append(seed)
                    native.set_seed(seed)
                    return [torch.rand(4)]
                def verify(args, actual):
                    issue = compare_outputs(reference(*copy_inputs(args)), actual, 1e-4)
                    if issue:
                        raise ValueError(issue)
                times = measurement.synchronized_times(candidate, [torch.zeros(4)], "cpu", num_trials=2, input_factory=inputs,
                                                      verifier=verify, receipt=receipt, origin_candidate=True)
                memory_stats = native.get_memory_stats(candidate, *memory_inputs, device="cpu", verifier=verify,
                                                       receipt=metadata.setdefault("memory_receipt", {}),
                                                       input_seed=42, origin_candidate=True)
                return {"compiled": True, "correctness": outcome.correctness, "metadata": metadata,
                        "peak_memory": memory_stats["peak_bytes"], "memory_stats": memory_stats,
                        "runtime_stats": measurement.timing_stats(times)}
            return {"compiled": True, "correctness": outcome.correctness, "metadata": metadata}

    def runner(command, timeout, log, *, environment):
        payload = json.loads(Path(command[3]).read_text())
        document = origin_task._run_native(payload, lambda: invoke(payload))
        if payload["_origin_stage"] == "profile":
            if action["activities"]:
                # Deliberately unknown overlapping device intervals, never real capture.
                document["raw"]["activities"] = [
                    {"kind": "kernel", "start_ns": window["start_ns"] + left, "end_ns": window["start_ns"] + right,
                     "device": 0, "stream": stream, "context_id": 1, "correlation": window["id"] * 10 + stream,
                     "graph_id": 0, "graph_node_id": 0}
                    for window in document["raw"]["windows"] for left, right, stream in ((100, 700, 1), (400, 900, 2))]
            if action["damage"] == "raw_window":
                document["raw"]["windows"].pop()
            elif action["damage"] == "output":
                document["calls"][-1]["output_content_sha256"] = "f" * 64
            elif action["damage"] == "request":
                document["binding"]["native_request_sha256"] = "f" * 64
        Path(command[4]).write_text(json.dumps(document))
        Path(log).write_text("Explicit CPU process simulation; no device collector loaded")
        return {"exit_code": 0, "timed_out": False, "request_wall_seconds": 0.01}

    def audit(*, target="evaluate", hidden=False, mutation="", damage=None, activities=False, role="candidate", caps=None):
        action.update(damage=damage, activities=activities, caps=caps)
        source = ("import torch\nclass ModelNew(torch.nn.Module):\n"
                  "    def __init__(self):\n        super().__init__()\n        self.calls = 0\n"
                  f"    def forward(self, x):\n        self.calls += 1\n{mutation}"
                  "        return x + self.calls + 1\n")
        payload = {"task": "origin_audit", "level": 1, "problem_id": 2, "kernel_src": source, "ref_arch_src": "reference",
                   "_origin_target": target, "_origin_timing_seeds": [100, 101, 102, 103, 104] if target == "evaluate" and not hidden else [],
                   "eval_config": {"precision": "fp32", "fp32_tolerance": 1e-4, "num_perf_trials": 2,
                                   "num_correct_trials": 1, "timeout": 60, "use_hidden_tests": hidden,
                                   "measure_performance": not hidden, "paired_trials": 10,
                                   "evaluator_sha256": evaluator_digest()}}
        if role == "reference":
            payload["_origin_role"] = role
        return origin_task.audit(payload, run=runner)
    def scored(request):
        payload = origin_task.native_request(request)
        payload.pop("_origin_timing_seeds", None)
        payload["eval_config"]["evidence_level"] = "L1"
        payload["build_root"] = str(tmp_path / "scored-build")
        return origin_task.native_policy(invoke)(payload)

    audit.runner, audit.invoke, audit.scored = runner, invoke, scored
    return audit


def test_native_origin_covers_actual_orders_warmups_hidden_and_paired_calls(native_origin_runner):
    from kb_workflows.analysis.performance import native_timeline
    from kb_workflows.provenance.origin_task import audit, replay, validate_scored

    for target, hidden, expected in (("evaluate", False, 8), ("evaluate", True, 8), ("paired_timing", False, 13)):
        result = native_origin_runner(target=target, hidden=hidden)
        assert result["native_coverage"]["status"] == "passed", result.get("native_coverage")
        assert result["native_coverage"]["candidate_calls"] == len(result["raw"]["windows"]) == expected
        assert result["native_coverage"]["simulated"] and not result["formal_qualification_implemented"]
        assert not result["scored_wall_times_replaced"] and result["summary"]["status"] == "not_measured"
        assert all(row["output_verified"] and row["native_call_sha256"] for row in result["raw"]["windows"])
        assert replay(result, allow_simulated=True) == result["summary"]
        assert result["native_coverage"]["memory_probe_included"] is (target == "evaluate" and not hidden)
        actual = native_origin_runner.scored(result["native_request"])
        frozen, observed = validate_scored(actual, allow_simulated=True)
        assert len(observed["calls"]) == expected
        assert not observed["timed_callable_wrapped"] and not observed["execution_origin_collected"]
        assert "_origin_timing_seeds" not in observed["native_request"]
        frozen.update(task="origin_audit", _origin_target=target, _scored_result=actual)
        linked = audit(frozen, run=native_origin_runner.runner)
        assert linked["native_coverage"]["scored_call_observations_match"]
        assert not linked["native_coverage"]["uninstrumented_path_equivalence_proven"]
        assert not linked["formal_qualification_implemented"] and replay(linked, allow_simulated=True) == linked["summary"]
        timeline = native_timeline(linked, allow_simulated=True)
        times = {row["id"]: row["wall_ms"] for row in observed["native_timings"]}
        assert [row["native_scored_wall_ms"] for row in timeline["calls"]] == [times.get(row["id"]) for row in timeline["calls"]]


def test_scored_call_binding_rejects_wrappers_changed_results_and_claimed_qualification(native_origin_runner, monkeypatch, tmp_path, capsys):
    from kb_workflows.evaluation import evidence
    from kb_workflows.provenance import origin_task

    original, observed = origin_task._Calls.call, []
    def call(state, model, args, **scope):
        execute, row = original(state, model, args, **scope)
        if state.scored:
            observed.append(execute is model)
        return execute, row
    monkeypatch.setattr(origin_task._Calls, "call", call)
    template = native_origin_runner()
    actual = native_origin_runner.scored(template["native_request"])
    frozen, _ = origin_task.validate_scored(actual, allow_simulated=True)
    assert observed and all(observed)
    with pytest.raises(ValueError, match="physical device UUID"):
        origin_task.validate_scored(actual)
    for damage in (
        lambda value: value["metadata"]["native_call_receipt"].update(timed_callable_wrapped=True),
        lambda value: value["metadata"]["native_call_receipt"]["calls"].pop(),
        lambda value: value["metadata"]["native_call_receipt"]["calls"][0].update(training_after=False),
        lambda value: value["metadata"]["native_call_receipt"]["native_timings"][-1].update(wall_ms=999),
        lambda value: value["runtime_stats"].update(samples_ms=[999, 999]),
        lambda value: value["metadata"]["native_call_receipt"]["timing_seed_plan"].reverse(),
    ):
        changed = copy.deepcopy(actual)
        damage(changed)
        receipt = changed["metadata"]["native_call_receipt"]
        receipt["receipt_sha256"] = fingerprint({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        with pytest.raises(ValueError):
            origin_task.validate_scored(changed, allow_simulated=True)
    changed = copy.deepcopy(frozen)
    changed["eval_config"]["fp32_tolerance"] = 1
    with pytest.raises(ValueError, match="observation contract"):
        origin_task.validate_scored(actual, changed, allow_simulated=True)
    passed = evidence.check("passed", reason="Synthetic claim without runtime evidence")
    actual["metadata"]["evidence_receipt"] = evidence.receipt(
        {name: passed for names in evidence.LEVELS.values() for name in names}, requested="L1")
    missing = copy.deepcopy(actual)
    missing["metadata"].pop("native_call_receipt")
    assert evidence.score_issue(missing, {"evidence_level": "L1"}, "standard") == "missing_or_invalid_scored_call_receipt"
    assert evidence.score_issue(actual, {"evidence_level": "L1"}, "standard") == "missing_or_invalid_scored_call_receipt"
    from scripts import audit_kernel_origin as command

    # Only this explicitly simulated CPU fixture bypasses physical identity.
    # The public CLI and score gate above retain the real-device requirement.
    validator = origin_task.validate_scored
    monkeypatch.setattr(origin_task, "validate_scored", lambda result, payload=None, **kwargs:
                        validator(result, payload, allow_simulated=True))
    monkeypatch.setattr(command, "task_payload", lambda *args, **kwargs:
                        copy.deepcopy(actual["metadata"]["native_call_receipt"]["native_request"]))
    monkeypatch.setattr(command, "evaluate_tasks", lambda *args: pytest.fail("Preview cannot dispatch experiments"))
    config = tmp_path / "scored.toml"
    config.write_text('[gpu]\nprofile="H100"\ndevices=[0]\n[run]\nstart=2\nend=2\nevidence_level="L1"\n')
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"2_0": actual}))
    output = tmp_path / "new-preview" / "report.json"
    capsys.readouterr()
    assert command.main(["--config", str(config), "--problem", "2", "--scored-result", str(results),
                         "--output", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["actual_scored_calls_requested"] and preview["timing_seed_count"] == len(frozen["_origin_timing_seeds"])
    assert not preview["experiments_started"] and not output.parent.exists()


def test_native_state_binding_includes_parameters_aliases_and_entrypoints(native_origin_runner, monkeypatch, tmp_path):
    from kb_workflows.evaluation.state_paths import invocation_state
    from kb_workflows.provenance import origin_task

    monkeypatch.setenv("KB_SOFTWARE_STATE", "1")
    template = native_origin_runner()
    request = copy.deepcopy(template["native_request"])
    request["kernel_src"] = request["kernel_src"].replace("        self.calls = 0\n", '''        self.calls = 0
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("view", self.weight.detach()[::2])
        self.plan = {"factor": int(__import__("os").environ["KB_SOFTWARE_STATE"])}
''')
    actual = native_origin_runner.scored(request)
    frozen, observed = origin_task.validate_scored(actual, allow_simulated=True)
    assert all(row["model_state_before"]["tensor_count"] == 2 for row in observed["calls"])
    assert all(row["model_state_before"]["sha256"] != row["model_state_after"]["sha256"] for row in observed["calls"])
    frozen.update(task="origin_audit", _origin_target="evaluate", _scored_result=actual)
    assert origin_task.audit(frozen, run=native_origin_runner.runner)["native_coverage"]["scored_call_observations_match"]
    # Both replay processes agree and every output remains identical, but their
    # ordinary configuration differs from the actual timing process.
    monkeypatch.setenv("KB_SOFTWARE_STATE", "2")
    changed = origin_task.audit(frozen, run=native_origin_runner.runner)
    assert changed["native_coverage"]["status"] == "not_measured"
    assert "state, entrypoints" in changed["native_coverage"]["reason"]
    registry, context = SourceRegistry(), {}
    with record_sources(registry):
        execute(request["kernel_src"], context, tmp_path / "state-source")
    model = context["ModelNew"]()
    inputs = [model.weight.detach()]
    before = invocation_state(model, inputs, callable_identity=registry.callable_identity)
    model.weight.detach().add_(1)
    updated = invocation_state(model, inputs, callable_identity=registry.callable_identity)
    assert before["sha256"] != updated["sha256"]
    model.view = model.view.clone()  # Same values; changes the storage relation.
    after = invocation_state(model, inputs, callable_identity=registry.callable_identity)
    assert updated["sha256"] != after["sha256"]
    model.forward = context["ModelNew"].forward.__get__(context["ModelNew"]())
    rebound = invocation_state(model, inputs, callable_identity=registry.callable_identity)
    assert rebound["sha256"] != after["sha256"]
    isolated = invocation_state(model, [inputs[0].clone()], callable_identity=registry.callable_identity)
    assert isolated["sha256"] != rebound["sha256"]
    # Descriptor discovery must not execute a candidate property.
    class Opaque:
        @property
        def __dict__(self):
            pytest.fail("Snapshot executed an opaque property")
    model.opaque = Opaque()
    with pytest.raises(ValueError, match="custom descriptor"):
        invocation_state(model, inputs, callable_identity=registry.callable_identity)
    class OpaqueClass:
        @property
        def __class__(self):
            pytest.fail("Snapshot executed __class__")
    vars(model)["opaque"] = OpaqueClass()
    with pytest.raises(ValueError, match="opaque"):
        invocation_state(model, inputs, callable_identity=registry.callable_identity)
    legacy = copy.deepcopy(actual)
    record = legacy["metadata"]["native_call_receipt"]
    record["protocol"] = "kernelbench_pro_native_origin_v1"
    record["receipt_sha256"] = fingerprint({k: v for k, v in record.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError):
        origin_task.validate_scored(legacy, allow_simulated=True)


def test_saved_native_replay_rejects_resealed_binding_coverage_and_source_changes(native_origin_runner):
    from kb_workflows.provenance.origin_task import replay

    report = native_origin_runner(activities=True)
    with pytest.raises(ValueError, match="simulated"):
        replay(report)
    damages = (
        lambda value: value["raw"]["windows"].pop(),
        lambda value: value["native_request"]["eval_config"].update(fp32_tolerance=1),
        lambda value: value["native_request"]["eval_config"].update(evaluator_sha256="f" * 64),
        lambda value: value["native_runs"]["profile"]["calls"][-1].update(output_content_sha256="f" * 64),
        lambda value: value["native_runs"]["control"]["native_result"]["metadata"].update(kernel_sha256="f" * 64),
        lambda value: value["raw"].update(candidate_sha256="f" * 64),
        lambda value: value["summary"].update(status="passed"),
        lambda value: value["native_coverage"].update(candidate_calls=1),
        lambda value: value["native_runs"]["profile"]["native_timings"].pop(),
        lambda value: value["native_runs"]["profile"]["native_timings"][-1].update(wall_ms=999),
        lambda value: value["native_runs"]["profile"]["native_result"]["runtime_stats"].update(samples_ms=[1, 2]),
        lambda value: value["native_runs"]["profile"]["native_result"]["metadata"]["memory_receipt"].update(output_verified=False),
        lambda value: value["native_runs"]["control"]["binding"].update(scope="all_calls_including_memory"),
    )
    for damage in damages:
        changed = copy.deepcopy(report)
        damage(changed)
        changed["receipt_sha256"] = fingerprint({k: v for k, v in changed.items() if k != "receipt_sha256"})
        with pytest.raises(ValueError):
            replay(changed, allow_simulated=True)
    missing_device = copy.deepcopy(report)
    missing_device["raw"]["simulated"] = missing_device["native_coverage"]["simulated"] = False
    missing_device["summary"] = kernel_origin.summarize(missing_device["raw"])
    missing_device["receipt_sha256"] = fingerprint({k: v for k, v in missing_device.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="physical device UUID"):
        replay(missing_device)


def test_native_timeline_joins_same_clock_intervals_and_keeps_missing_work_unknown(native_origin_runner, tmp_path):
    from kb_workflows.analysis.performance import native_timeline
    from scripts.analyze_device_trace import main

    report = native_origin_runner(activities=True)
    result = native_timeline(report, allow_simulated=True)
    assert result["status"] == "simulated" and not result["formal_qualification_implemented"]
    assert {key: row["calls"] for key, row in result["by_phase"].items()} == {"correctness": 2, "warmup": 3, "timing": 2, "memory": 1}
    for row, observed in zip(result["calls"], report["native_runs"]["profile"]["calls"], strict=True):
        assert row["native_call_sha256"] == fingerprint(observed)
        assert row["profile_window_wall_ms"] == pytest.approx(0.001)
        assert row["device_busy_ms"] == pytest.approx(0.0008) and row["O"] == pytest.approx(1.25)
        assert row["unknown_activity_count"] == 2 and row["origin_verdicts"]["custom"] == "not_measured"
        assert row["A"] is row["E"] is row["AE"] is None
        for stage in ("control", "profile"):
            if row["phase"] in {"correctness", "memory"}:
                assert row[f"native_{stage}_wall_ms"] is None
            else:
                observed_times = {item["id"]: item["wall_ms"] for item in report["native_runs"][stage]["native_timings"]}
                assert row[f"native_{stage}_wall_ms"] == observed_times[row["id"]]
    assert all(row["O"] == pytest.approx(1.25) for row in result["by_phase"].values())
    empty = native_timeline(native_origin_runner(), allow_simulated=True)
    assert empty["status"] == "not_measured" and all(row["O"] is row["device_busy_ms"] is None for row in empty["calls"])
    assert all(row["O"] is None and not row["complete"] for row in empty["by_phase"].values())
    for target, hidden in (("evaluate", True), ("paired_timing", False)):
        other = native_timeline(native_origin_runner(target=target, hidden=hidden, activities=True), allow_simulated=True)
        assert len(other["calls"]) == (8 if hidden else 13)
    source, output = tmp_path / "origin.json", tmp_path / "offline.json"
    source.write_text(json.dumps(report))
    assert main(["native", str(source), "--output", str(output)]) == 2 and not output.exists()


def test_native_origin_refuses_late_wrong_outputs_mode_changes_and_missing_windows(native_origin_runner):
    mutations = ("        if self.calls > 4: return x + 99\n", "        if self.calls > 5: return x + 99\n", "        self.eval()\n",
                 "        torch.set_grad_enabled(True)\n", "        import os\n        os.environ['KB_FP32_TOL'] = '1'\n",
                 "        return x + self.calls + 1.0005\n")
    for mutation in mutations:
        result = native_origin_runner(mutation=mutation)
        assert result["native_coverage"]["status"] == "not_measured"
        assert result["summary"]["status"] == "not_measured"
    for damage in ("raw_window", "output", "request"):
        result = native_origin_runner(damage=damage)
        assert result.get("native_coverage", {}).get("status", result["status"]) == "not_measured"


def test_streaming_content_signature_preserves_dense_strided_bits_and_typed_structure():
    import hashlib

    import torch

    from kb_workflows.evaluation.determinism import signature

    values = [torch.arange(800_000, dtype=torch.float32), torch.arange(800_000, dtype=torch.float32).bfloat16(),
              torch.arange(1_600_000).reshape(1000, 1600)[:, ::3], torch.empty(0, 3), torch.tensor(3)]
    for value in values:
        expected = hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        assert signature(value)["bytes_sha256"] == expected
    first = {1: torch.tensor([3]), "1": (torch.tensor([5]),)}
    assert signature(first, typed=True) == signature(dict(reversed(list(first.items()))), typed=True)
    assert signature(first, typed=True) != signature({"1": [torch.tensor([5])]}, typed=True)


def test_native_activity_filter_preserves_unknown_overlaps_and_all_drop_accounting_queues():
    sdk, _ = fake_sdk()
    capture = cuda_origin.Capture(SOURCE, bindings=sdk, candidate_windows_only=True)
    capture.raw.update(raw_capture(), candidate_windows_only=True, scope_filtered_activities=0)
    capture.raw["activities"] = []
    def event(start, end, correlation, stream=7):
        return NS(kind=10, start=start, end=end, device_id=0, stream_id=stream, context_id=9,
                  correlation_id=correlation, graph_id=0, graph_node_id=0, name="fixture")
    capture.buffer_completed([event(10, 20, 999, stream=17), event(120, 220, 10),
                              event(170, 270, 99), event(310, 350, 999)])
    assert capture.raw["scope_filtered_activities"] == 2
    assert (9, 17) in capture.activity_queues  # Filtered streams still need drop queries.
    result = kernel_origin.summarize(capture.raw)
    assert result["status"] == "not_measured"
    assert result["calls"][0]["custom_exclusive_fraction_lower_bound"] == pytest.approx(1 / 3)
