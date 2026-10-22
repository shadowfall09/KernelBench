"""CUPTI graph-node provenance with per-launch immutable snapshots.

Only observed kernel creation, cloning and successful instantiation establish a
node/function association. Parameter updates invalidate it until rebuilt. CUPTI
ID queries are used in callbacks; no CUDA driver/runtime query is made there.
"""

from __future__ import annotations

import ctypes as C
import threading


class KernelParams(C.Structure):
    _fields_ = [("function", C.c_void_p), ("gx", C.c_uint32), ("gy", C.c_uint32), ("gz", C.c_uint32),
                ("bx", C.c_uint32), ("by", C.c_uint32), ("bz", C.c_uint32), ("shared", C.c_uint32),
                ("params", C.c_void_p), ("extra", C.c_void_p)]


class AddKernelNode(C.Structure):
    _fields_ = [("output", C.c_void_p), ("graph", C.c_void_p), ("dependencies", C.c_void_p),
                ("count", C.c_size_t), ("params", C.POINTER(KernelParams))]


class Instantiate(C.Structure):
    # cuGraphInstantiate{,_v2,WithFlags,WithParams} all begin with these fields.
    _fields_ = [("output", C.POINTER(C.c_void_p)), ("graph", C.c_void_p)]


def abi_supported():
    return (C.sizeof(KernelParams) == 56 and KernelParams.params.offset == 40
            and C.sizeof(AddKernelNode) == 40 and AddKernelNode.params.offset == 32
            and C.sizeof(Instantiate) == 16)


class Graphs:
    def __init__(self, capture):
        self.capture = capture
        self.builders = {}
        self.nodes = {}  # CUPTI globally unique node ID -> generation record.
        self.by_graph = {}
        self.executables = {}
        self.generation = 0
        self.pending_launches = {}

    def driver(self, data):
        name, tid = str(data.function_name), threading.get_ident()
        entering = int(data.callback_site) == int(self.capture.cupti.ApiCallbackSite.API_ENTER)
        pointer = int(data.function_params)
        if not pointer:
            return
        key = (int(data.context_uid), int(data.correlation_id))
        context = int(data.context or 0)
        instantiate = name in {"cuGraphInstantiate", "cuGraphInstantiate_v2", "cuGraphInstantiateWithFlags", "cuGraphInstantiateWithParams"}
        launch = name in {"cuGraphLaunch", "cuGraphLaunch_ptsz"}
        kernel = name in {"cuLaunchKernel", "cuLaunchKernel_ptsz", "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"}
        add = name == "cuGraphAddKernelNode"
        memory = name.startswith(("cuMemcpy", "cuMemset")) or name in {"cuGraphAddMemcpyNode", "cuGraphAddMemsetNode"}
        if entering:
            if kernel or add or instantiate or memory:
                function = None
                if kernel:
                    offset = C.sizeof(C.c_void_p) if "Ex" in name else 0
                    handle = C.c_void_p.from_address(pointer + offset).value
                    function = self.capture.function_map.get((context, handle))
                elif add:
                    parameters = AddKernelNode.from_address(pointer).params
                    if parameters:
                        function = self.capture.function_map.get((context, parameters.contents.function))
                frame = {"key": key, "kind": "instantiate" if instantiate else ("memory" if memory else "kernel"),
                         "function": function["id"] if function else None, "events": [], "clones": []}
                if instantiate:
                    frame["graph"] = Instantiate.from_address(pointer).graph
                self.builders.setdefault(tid, []).append(frame)
            elif launch:
                handle = C.c_void_p.from_address(pointer).value
                executable = self.executables.get(handle)
                row = self.capture.append("graph_launches", {"context_id": key[0], "correlation": key[1],
                    "runtime_correlation": self.capture.runtime_parent(tid), "call": self.capture.active_calls.get(tid),
                    "api": name, "exec_handle": handle, "exit_success": False,
                    "exec_generation": executable["generation"] if executable else None,
                    "graph_ids": executable["graph_ids"] if executable else [],
                    "nodes": dict(executable["nodes"]) if executable else {}})
                if row is not None:
                    self.pending_launches[key] = row
            elif name.startswith("cuGraph") and ("SetParams" in name or "Update" in name):
                # Unknown/update ABIs must never retain a stale source function.
                self.nodes.clear()
                self.by_graph.clear()
                self.executables.clear()
            elif name == "cuGraphExecDestroy":
                self.executables.pop(C.c_void_p.from_address(pointer).value, None)
            elif name in {"cuGraphDestroy", "cuGraphDestroyNode"}:
                # A destroyed template does not change an already instantiated
                # executable's immutable snapshot. Its own provenance is gone.
                self.nodes.clear()
                self.by_graph.clear()
            return
        success = bool(data.function_return_value) and C.c_int.from_address(int(data.function_return_value)).value == 0
        if kernel or add or instantiate or memory:
            stack = self.builders.get(tid, [])
            if not stack or stack[-1]["key"] != key:
                self.capture.error("Unbalanced graph construction callbacks")
                return
            frame = stack.pop()
            for event in frame["events"]:
                event["creation_success"] = success
            if instantiate and success:
                item = Instantiate.from_address(pointer)
                handle = item.output[0] if item.output else None
                if handle:
                    source_id = int(self.capture.cupti.get_graph_id(frame["graph"]))
                    exec_id = int(self.capture.cupti.get_graph_exec_id(handle))
                    # The SDK exposes source and executable IDs. Activity/node
                    # correlation still has to match this exact launch snapshot.
                    nodes = {str(node): row["id"] for node, row in self.nodes.items()
                             if row["graph_handle"] == frame["graph"] or node in frame["clones"]}
                    self.generation += 1
                    self.executables[handle] = {"generation": self.generation,
                        "graph_ids": sorted({source_id, exec_id}), "nodes": nodes}
        elif launch:
            row = self.pending_launches.pop(key, None)
            if row is not None:
                row["exit_success"] = success

    def resource(self, callback_id, data):
        cupti = self.capture.cupti
        names = cupti.CallbackIdResource
        created = callback_id == int(names.GRAPHNODE_CREATED)
        cloned = callback_id == int(names.GRAPHNODE_CLONED)
        invalidating = {int(getattr(names, name, -1)) for name in ("GRAPH_NODE_UPDATED", "GRAPH_NODE_SET_PARAMS")}
        if callback_id in invalidating:
            self.nodes.clear()
            self.by_graph.clear()
            self.executables.clear()
            return
        if not created and not cloned:
            return
        descriptor = int(data.resource_descriptor)
        if not descriptor:
            self.capture.error("Missing graph resource descriptor")
            return
        graph = cupti.GraphData.from_ptr(descriptor, readonly=True)
        node_id = int(cupti.get_graph_node_id(int(graph.node)))
        graph_id = int(cupti.get_graph_id(int(graph.graph)))
        stack = self.builders.get(threading.get_ident(), [])
        frame = stack[-1] if stack else None
        original = None
        function, success = None, False
        kind = {0: "kernel", 1: "memcpy", 2: "memset"}.get(int(graph.node_type), "other")
        if cloned and graph.original_node:
            original_id = int(cupti.get_graph_node_id(int(graph.original_node)))
            original = self.nodes.get(original_id)
            if original is not None:
                function, success = original["function"], original["creation_success"]
            if frame and frame["kind"] == "instantiate":
                frame["clones"].append(node_id)
        elif created and frame and frame["kind"] == "kernel" and kind == "kernel":
            function = frame["function"]
        record = self.capture.append("graph_nodes", {"node_id": node_id, "graph_id": graph_id,
            "graph_handle": int(graph.graph), "kind": kind, "function": function,
            "creation_success": success, "original": original["id"] if original else None,
            "creation": "clone" if cloned else ("driver_memory" if frame and frame["kind"] == "memory" else "driver_kernel")})
        if record is None:
            return
        if created and frame and frame["kind"] in {"kernel", "memory"}:
            frame["events"].append(record)
        self.nodes[node_id] = record
        self.by_graph.setdefault(int(graph.graph), set()).add(node_id)

    def balanced(self):
        return not self.pending_launches and not any(self.builders.values())
