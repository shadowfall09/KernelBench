"""CPU C-ABI and serialized graph-update controls; no CUDA/CUPTI execution."""

from __future__ import annotations

import copy
import ctypes as C
import json

import pytest
from test_kernel_origin import (
    build_graph,
    driver_data,
    graph_capture,
    graph_fixture,
    graph_resource,
    replay_graph,
)

from kb_workflows import cuda_origin, kernel_origin
from kb_workflows.origin_graphs import (
    Instantiate,
    KernelParams,
    KernelParamsV2,
    SetExecKernelParams,
    SetKernelParams,
    abi_supported,
)


def new_function(capture, handle=201):
    output = C.c_void_p(handle)
    data = driver_data("cuModuleGetFunction", cuda_origin.ModuleGetFunction(C.pointer(output), 100, None), 31)
    capture.callback(None, 1, 0, data)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    return capture.function_map[(123, handle)]["id"]


def update(
    capture,
    descriptors,
    *,
    scope="executable",
    version="",
    function=201,
    node=88,
    executable=99,
    event=True,
    event_node=None,
    event_exec=None,
    return_code=0,
    correlation=40,
):
    params = (KernelParamsV2 if version == "_v2" else KernelParams)(
        function=function, gx=1, gy=1, gz=1, bx=32, by=1, bz=1
    )
    pointer = C.cast(C.pointer(params), C.POINTER(KernelParams))
    if scope == "executable":
        api = "cuGraphExecKernelNodeSetParams" + version
        arguments = SetExecKernelParams(executable, node, pointer)
    else:
        api = "cuGraphKernelNodeSetParams" + version
        arguments = SetKernelParams(node, pointer)
    data = driver_data(api, arguments, correlation, return_code=return_code)
    capture.callback(None, 1, 0, data)
    if event:
        pointer = len(descriptors) + 1
        from types import SimpleNamespace as NS

        descriptors[pointer] = NS(
            node=node if event_node is None else event_node,
            node_type=0,
            graph=77,
            graph_exec=(executable if scope == "executable" else 0) if event_exec is None else event_exec,
        )
        kind = capture.cupti.CallbackIdResource.GRAPH_NODE_SET_PARAMS
        capture.callback(None, 3, kind, NS(resource_descriptor=pointer, context=123))
    data.callback_site = 1
    capture.callback(None, 1, 0, data)
    return capture.raw["graph_updates"][-1]


def fixture(capture, *, node=1088, correlation=50):
    raw = graph_fixture(capture, node=node, correlation=correlation)
    proof = copy.deepcopy(raw["proofs"][0])
    proof.update(id=2, function=2)
    raw["proofs"].append(proof)
    return raw


def instantiate(capture, descriptors, handle=101, *, clone=False):
    output = C.c_void_p(handle)
    data = driver_data("cuGraphInstantiate_v2", Instantiate(C.pointer(output), 77), 41)
    capture.callback(None, 1, 0, data)
    if clone:
        graph_resource(capture, descriptors, cloned=True, graph=79, node=90, original=88)
    data.callback_site = 1
    capture.callback(None, 1, 0, data)


@pytest.mark.parametrize("scope", ["template", "executable"])
@pytest.mark.parametrize("version", ["", "_v2"])
@pytest.mark.parametrize("clone", [False, True])
def test_successful_updates_bind_new_function_generations_and_preserve_earlier_launches(scope, version, clone):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors, clone=clone)
    assert new_function(capture) == 2
    replay_graph(capture, correlation=49)
    before = copy.deepcopy(capture.raw["graph_launches"][0])
    receipt = update(capture, descriptors, scope=scope, version=version)
    assert receipt["exit_success"] and receipt["resource_complete"]
    assert capture.raw["graph_launches"][0] == before
    if scope == "template":
        assert capture.graphs.executables[99]["nodes"] == before["nodes"]
        instantiate(capture, descriptors, clone=clone)
    replay_graph(capture, executable=101 if scope == "template" else 99)
    raw = fixture(capture, node=(1090 if scope == "template" else 1089) if clone else 1088)
    if scope == "template":
        raw["activities"][0]["graph_id"] = 20101
    assert capture.raw["errors"] == [] and capture.graphs.balanced()
    summary = kernel_origin.summarize(raw)
    assert summary["status"] == "passed"
    uid = str(raw["activities"][0]["graph_node_id"])
    latest = raw["graph_nodes"][raw["graph_launches"][-1]["nodes"][uid] - 1]
    assert latest["function"] == 2
    assert kernel_origin.summarize(json.loads(json.dumps(raw))) == summary


@pytest.mark.parametrize("function,event", [(999, True), (201, False), (None, True)])
def test_unassociated_function_missing_event_or_cukernel_never_retains_old_custom_credit(function, event):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    update(capture, descriptors, function=function, event=event)
    replay_graph(capture)
    result = kernel_origin.summarize(fixture(capture))
    assert result["track_verdicts"] == {"custom": "not_measured", "library": "not_measured"}


@pytest.mark.parametrize("event", [False, True])
def test_failed_update_with_a_resource_event_invalidates_ambiguous_effects(event):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    old = copy.deepcopy(capture.graphs.executables[99])
    receipt = update(capture, descriptors, return_code=1, event=event)
    assert receipt["replacements"] == {} and receipt["after_generation"] is None
    if not event:
        assert capture.graphs.executables[99] == old
    replay_graph(capture)
    assert kernel_origin.summarize(fixture(capture))["status"] == ("not_measured" if event else "passed")


@pytest.mark.parametrize("mismatch", [{"event_node": 777}, {"event_exec": 101}])
def test_unmatched_update_resource_invalidates_all_affected_source_associations(mismatch):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    receipt = update(capture, descriptors, **mismatch)
    assert not receipt["resource_complete"] and not capture.graphs.executables
    replay_graph(capture)
    assert kernel_origin.summarize(fixture(capture))["status"] == "not_measured"


@pytest.mark.parametrize(
    "damage",
    [
        "function",
        "previous",
        "scope",
        "event",
        "event_exec",
        "replaced_id",
        "boundary",
        "target",
        "context",
        "success",
        "creation_success",
        "generation",
        "missing_update",
        "wrong_exec",
        "stale_snapshot",
        "stale_nodes_only",
        "future_update",
    ],
)
def test_serialized_updates_cannot_rebind_nodes_or_move_updates_across_call_boundaries(damage):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    replay_graph(capture, correlation=49)
    update(capture, descriptors)
    replay_graph(capture)
    raw = fixture(capture)
    row, node = raw["graph_updates"][0], raw["graph_nodes"][-1]
    if damage == "function":
        node["function"] = 1
    elif damage == "previous":
        node["previous"] = node["id"]
    elif damage == "scope":
        row["scope"] = "template"
    elif damage == "event":
        row["resource_events"] = []
    elif damage == "event_exec":
        row["resource_events"][0]["exec_handle"] = 777
    elif damage == "replaced_id":
        row["replacements"]["1088"] = 1
    elif damage == "boundary":
        row["launch_boundary"] = 3
    elif damage == "target":
        row["targets"]["1088"] = 999
    elif damage == "context":
        row["context"] = 456
    elif damage == "success":
        row["exit_success"] = False
    elif damage == "creation_success":
        node["creation_success"] = False
    elif damage == "generation":
        row["after_generation"] = row["before_generation"]
    elif damage == "missing_update":
        raw["graph_updates"] = []
    elif damage == "wrong_exec":
        raw["graph_launches"][-1]["exec_handle"] = 100
    elif damage == "stale_snapshot":
        raw["graph_launches"][-1].update(raw["graph_launches"][0], correlation=50)
    elif damage == "stale_nodes_only":
        raw["graph_launches"][-1]["nodes"] = copy.deepcopy(raw["graph_launches"][0]["nodes"])
    elif damage == "future_update":
        raw["graph_launches"][0].update(raw["graph_launches"][-1], correlation=49)
    with pytest.raises(ValueError, match="[Gg]raph|executable"):
        kernel_origin.summarize(raw)


def test_multiple_updates_keep_generation_ancestry_without_rewriting_previous_calls():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors, clone=True)
    new_function(capture)
    update(capture, descriptors)
    replay_graph(capture, correlation=49)
    before = copy.deepcopy(capture.raw["graph_launches"][0])
    update(capture, descriptors, function=200, correlation=42)
    replay_graph(capture)
    assert capture.raw["graph_launches"][0] == before
    assert kernel_origin.summarize(fixture(capture, node=1089))["status"] == "passed"


def test_update_callback_scopes_and_supported_host_abi_are_explicit():
    assert abi_supported()
    assert C.sizeof(SetKernelParams) == 16 and C.sizeof(SetExecKernelParams) == 24
    assert SetKernelParams.params.offset == 8 and SetExecKernelParams.params.offset == 16
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    params = KernelParams(function=200)
    data = driver_data("cuGraphExecKernelNodeSetParams", SetExecKernelParams(99, 88, C.pointer(params)), 40)
    capture.callback(None, 1, 0, data)
    assert not capture.graphs.balanced()


def test_template_update_does_not_modify_an_already_instantiated_executable():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    update(capture, descriptors, scope="template")
    replay_graph(capture)
    raw = fixture(capture)
    assert capture.graphs.nodes[1088]["function"] == 2
    assert raw["graph_launches"][0]["nodes"] == {"1088": 1}
    assert kernel_origin.summarize(raw)["status"] == "passed"


def test_executable_update_preserves_the_template_and_other_executables():
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    instantiate(capture, descriptors)
    before = copy.deepcopy(capture.graphs.executables[101])
    update(capture, descriptors)
    assert capture.graphs.nodes[1088]["function"] == 1
    assert capture.graphs.executables[101] == before
    replay_graph(capture, executable=101)
    raw = fixture(capture)
    raw["activities"][0]["graph_id"] = 20101
    assert kernel_origin.summarize(raw)["status"] == "passed"


@pytest.mark.parametrize("event,return_code", [(False, 0), (True, 1)])
def test_invalidated_executables_cannot_reappear_by_copying_an_old_launch_snapshot(event, return_code):
    capture, descriptors, _ = graph_capture()
    build_graph(capture, descriptors)
    new_function(capture)
    replay_graph(capture, correlation=49)
    update(capture, descriptors, event=event, return_code=return_code)
    replay_graph(capture)
    raw = fixture(capture)
    raw["graph_launches"][-1].update(raw["graph_launches"][0], correlation=50)
    with pytest.raises(ValueError, match="executable"):
        kernel_origin.summarize(raw)
