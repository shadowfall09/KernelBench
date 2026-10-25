"""CUPTI graph-node provenance with per-launch immutable snapshots.

Observed creation, cloning, instantiation and supported kernel-parameter updates
establish node/function associations. Updates create new generations and never
rewrite earlier launches. CUPTI ID queries do not call CUDA driver/runtime APIs.
"""

from __future__ import annotations

import ctypes as C
import threading


class KernelParams(C.Structure):
    _fields_ = [("function", C.c_void_p), ("gx", C.c_uint32), ("gy", C.c_uint32), ("gz", C.c_uint32),
                ("bx", C.c_uint32), ("by", C.c_uint32), ("bz", C.c_uint32), ("shared", C.c_uint32),
                ("params", C.c_void_p), ("extra", C.c_void_p)]


class KernelParamsV2(KernelParams):
    _fields_ = [("kernel", C.c_void_p), ("context", C.c_void_p)]


class AddKernelNode(C.Structure):
    _fields_ = [("output", C.c_void_p), ("graph", C.c_void_p), ("dependencies", C.c_void_p),
                ("count", C.c_size_t), ("params", C.POINTER(KernelParams))]


class Instantiate(C.Structure):
    # cuGraphInstantiate{,_v2,WithFlags,WithParams} all begin with these fields.
    _fields_ = [("output", C.POINTER(C.c_void_p)), ("graph", C.c_void_p)]


class SetKernelParams(C.Structure):
    _fields_ = [("node", C.c_void_p), ("params", C.POINTER(KernelParams))]


class SetExecKernelParams(C.Structure):
    _fields_ = [("executable", C.c_void_p), ("node", C.c_void_p), ("params", C.POINTER(KernelParams))]


TEMPLATE_UPDATES = {"cuGraphKernelNodeSetParams", "cuGraphKernelNodeSetParams_v2"}
EXEC_UPDATES = {"cuGraphExecKernelNodeSetParams", "cuGraphExecKernelNodeSetParams_v2"}


def validate_updates(updates, nodes, functions, launches, *, limit):
    """Replay update generations, resource matches and immutable launch ancestry."""
    def positive(value):
        return type(value) is int and value > 0

    def uid(value):
        return isinstance(value, str) and value.isdecimal() and str(int(value)) == value

    after, stale, invalidations = {}, {}, {}
    for row in updates.values():
        scope = row.get("scope")
        targets, replacements, events = row.get("targets"), row.get("replacements"), row.get("resource_events")
        if (scope not in {"template", "executable"}
                or row.get("api") not in (TEMPLATE_UPDATES if scope == "template" else EXEC_UPDATES)
                or not all(positive(row.get(k)) for k in ("context", "context_id", "correlation"))
                or (row.get("node_id") is not None and not positive(row["node_id"]))
                or type(row.get("exit_success")) is not bool or type(row.get("resource_complete")) is not bool
                or not isinstance(events, list) or len(events) > limit
                or type(row.get("launch_boundary")) is not int or not 0 <= row["launch_boundary"] <= len(launches)
                or not isinstance(targets, dict) or not isinstance(replacements, dict)
                or len(targets) > limit or set(replacements) - set(targets)
                or any(not uid(k) or not positive(value) or value not in nodes or nodes[value]["node_id"] != int(k) for k, value in targets.items())
                or any(not uid(k) or not positive(value) or value not in nodes for k, value in replacements.items())):
            raise ValueError("Invalid graph parameter-update receipt")
        function = row.get("function")
        if function is not None and (not positive(function) or function not in functions or functions[function]["context"] != row["context"]):
            raise ValueError("Graph update function belongs to another context")
        if scope == "executable":
            if not positive(row.get("exec_handle")) or (row.get("before_generation") is not None and not positive(row["before_generation"])):
                raise ValueError("Graph update lacks its executable generation")
            generation = row.get("after_generation")
            if generation is not None:
                if (not positive(generation) or not positive(row.get("before_generation"))
                        or generation <= row["before_generation"] or not row["exit_success"] or generation in after):
                    raise ValueError("Graph update reused or reversed an executable generation")
                after[generation] = row
                key = row["exec_handle"], row["before_generation"]
                stale[key] = min(stale.get(key, len(launches) + 1), row["launch_boundary"])
        elif any(row.get(k) is not None for k in ("exec_handle", "before_generation", "after_generation")):
            raise ValueError("Template update cannot modify an executable generation")
        complete = bool(events) and all(isinstance(event, dict) and event.get("callback_id") in {22, 23}
            and str(event.get("node_id")) in targets
            and event.get("context") == row["context"]
            and event.get("exec_handle") == (row["exec_handle"] if scope == "executable" else 0) for event in events)
        if row["resource_complete"] and not complete:
            raise ValueError("Graph update has no matching CUPTI node/executable event")
        if (not row["exit_success"] and replacements) or (replacements and set(replacements) != set(targets)):
            raise ValueError("Failed or partial graph update replaced node provenance")
        invalidation = row.get("invalidation_generation")
        if invalidation is not None:
            if type(invalidation) is not int or invalidation < 0 or replacements or row.get("after_generation") is not None:
                raise ValueError("Invalid graph update invalidation boundary")
            boundary = row["launch_boundary"]
            invalidations[boundary] = max(invalidations.get(boundary, 0), invalidation)
        if ((row["exit_success"] and not row["resource_complete"]) or (not row["exit_success"] and events)) and invalidation is None:
            raise ValueError("Ambiguous graph update retained previously known executable provenance")
        for key, record in replacements.items():
            new, old = nodes[record], nodes[targets[key]]
            if (not positive(new.get("previous")) or not positive(new.get("update"))
                    or old["id"] >= new["id"] or new.get("previous") != old["id"] or new.get("update") != row["id"]
                    or new.get("function") != function or new.get("original") is not None
                    or new.get("creation") != ("exec_kernel_update" if scope == "executable" else "template_kernel_update")
                    or new.get("creation_success") != (row["resource_complete"] and old["kind"] == "kernel" and old["creation_success"])
                    or any(new.get(k) != old.get(k) for k in ("node_id", "graph_id", "graph_handle", "kind"))):
                raise ValueError("Graph parameter update changed its node lineage or actual function")
    for node in nodes.values():
        if node["creation"] in {"template_kernel_update", "exec_kernel_update"}:
            if not positive(node.get("update")):
                raise ValueError("Graph node update has no valid API identity")
            row = updates.get(node.get("update"), {})
            if row.get("replacements", {}).get(str(node["node_id"])) != node["id"]:
                raise ValueError("Graph node update has no matching successful API receipt")
        elif node.get("previous") is not None or node.get("update") is not None:
            raise ValueError("Unobserved graph creation cannot claim parameter-update lineage")
    steps, invalidated_generation = 0, 0
    for ordinal, launch in enumerate(launches):
        ancestors, expected_nodes, generation = set(), {}, launch.get("exec_generation")
        invalidated_generation = max(invalidated_generation, invalidations.get(ordinal, 0))
        if ((generation is not None and not positive(generation))
                or (launch.get("nodes") and (not positive(generation) or not positive(launch.get("exec_handle"))
                    or generation <= invalidated_generation))):
            raise ValueError("Invalid executable graph generation")
        while generation in after:
            steps += 1
            if steps > 4 * limit:
                raise ValueError("Graph update replay exceeds its bounded lineage work envelope")
            update = after[generation]
            if update["exec_handle"] != launch.get("exec_handle") or ordinal < update["launch_boundary"]:
                raise ValueError("Graph launch used an update from another executable or a future call")
            ancestors.add(update["id"])
            for key, rid in update["replacements"].items():
                expected_nodes.setdefault(key, rid)
            generation = update["before_generation"]
        if any(launch["nodes"].get(key) != rid for key, rid in expected_nodes.items()):
            raise ValueError("Graph launch snapshot omitted or rolled back its actual node update")
        if ordinal >= stale.get((launch.get("exec_handle"), launch.get("exec_generation")), len(launches) + 1):
            raise ValueError("Graph launch retained a stale executable generation after update")
        visited = set()
        for rid in launch.get("nodes", {}).values():
            while rid is not None and rid not in visited:
                steps += 1
                if steps > 4 * limit:
                    raise ValueError("Graph update replay exceeds its bounded lineage work envelope")
                visited.add(rid)
                node = nodes[rid]
                if node.get("update") is not None:
                    update = updates[node["update"]]
                    if (ordinal < update["launch_boundary"] or (update["scope"] == "executable" and update["id"] not in ancestors)):
                        raise ValueError("Graph launch copied a node update outside its executable lineage")
                rid = node.get("previous") or node.get("original")


def abi_supported():
    return (C.sizeof(KernelParams) == 56 and KernelParams.params.offset == 40
            and C.sizeof(AddKernelNode) == 40 and AddKernelNode.params.offset == 32
            and C.sizeof(KernelParamsV2) == 72 and KernelParamsV2.kernel.offset == 56 and KernelParamsV2.context.offset == 64
            and C.sizeof(Instantiate) == 16 and C.sizeof(SetKernelParams) == 16
            and SetExecKernelParams.params.offset == 16 and C.sizeof(SetExecKernelParams) == 24)


class Graphs:
    def __init__(self, capture):
        self.capture = capture
        self.builders = {}
        self.nodes = {}  # CUPTI globally unique node ID -> generation record.
        self.by_graph = {}
        self.executables = {}
        self.generation = 0
        self.pending_launches = {}
        self.pending_updates = {}
        self.root_cache = {}

    def root(self, record):
        """The template identity survives instantiation clones and parameter updates."""
        if record["id"] in self.root_cache:
            return self.root_cache[record["id"]]
        ancestors = []
        while record["id"] not in self.root_cache and (record.get("previous") or record.get("original")):
            ancestors.append(record["id"])
            record = self.capture.raw["graph_nodes"][(record.get("previous") or record["original"]) - 1]
        root = self.root_cache.get(record["id"], record["node_id"])
        for rid in ancestors:
            self.root_cache[rid] = root
        return root

    def invalidate(self):
        self.nodes.clear()
        self.by_graph.clear()
        self.executables.clear()
        for stack in self.pending_updates.values():
            for frame in stack:
                frame["events_valid"] = False
                frame["invalidation_generation"] = self.generation

    def begin_update(self, data, key, context):
        name, tid = str(data.function_name), threading.get_ident()
        executable = name in EXEC_UPDATES
        item = (SetExecKernelParams if executable else SetKernelParams).from_address(int(data.function_params))
        node = int(self.capture.cupti.get_graph_node_id(item.node)) if item.node else None
        previous = self.nodes.get(node)
        target = self.executables.get(item.executable) if executable else None
        root = self.root(previous) if previous else node
        targets = ({uid: rid for uid, rid in target["nodes"].items()
                    if self.root(self.capture.raw["graph_nodes"][rid - 1]) == root} if target else {})
        if not executable and previous:
            targets = {str(node): previous["id"]}
        function = self.capture.function_map.get((context, item.params.contents.function)) if item.params else None
        # v2 retains the v1 prefix. A NULL func uses CUkernel/ctx, which this
        # adapter deliberately leaves unassociated rather than dereferencing it.
        frame = {"key": key, "api": name, "context": context, "node_id": node,
                 "exec_handle": item.executable if executable else None,
                 "scope": "executable" if executable else "template", "targets": targets,
                 "before_generation": target["generation"] if target else None,
                 "function": function["id"] if function else None, "resource_events": [], "events_valid": True,
                 "invalidation_generation": None}
        self.pending_updates.setdefault(tid, []).append(frame)

    def finish_update(self, key, success):
        stack = self.pending_updates.get(threading.get_ident(), [])
        if not stack or stack[-1]["key"] != key:
            self.capture.error("Unbalanced graph parameter-update callbacks")
            return
        frame = stack.pop()
        target = self.executables.get(frame["exec_handle"]) if frame["scope"] == "executable" else None
        if target is not None and target["generation"] != frame["before_generation"]:
            frame["events_valid"] = False
        update = self.capture.append("graph_updates", {key: value for key, value in frame.items() if key not in {"key", "events_valid"}} | {
            "context_id": key[0], "correlation": key[1], "exit_success": success,
            "resource_complete": bool(frame["resource_events"]) and frame["events_valid"],
            "launch_boundary": len(self.capture.raw["graph_launches"]), "after_generation": None, "replacements": {}})
        if update is None:
            return
        if not success:
            if frame["resource_events"]:
                update["invalidation_generation"] = self.generation
                self.invalidate()  # A resource event plus API failure is ambiguous.
            return
        if not update["resource_complete"]:
            update["invalidation_generation"] = self.generation
            self.invalidate()
            return
        if frame["scope"] == "executable":
            if target is None:
                return  # Unknown executable remains unknown.
            self.generation += 1
            target["generation"] = self.generation
            update["after_generation"] = self.generation
        for uid, rid in frame["targets"].items():
            previous = self.capture.raw["graph_nodes"][rid - 1]
            record = self.capture.append("graph_nodes", {**{k: v for k, v in previous.items() if k != "id"},
                "function": frame["function"], "original": None, "previous": rid, "update": update["id"],
                "creation": "exec_kernel_update" if target is not None else "template_kernel_update",
                "creation_success": update["resource_complete"] and previous["kind"] == "kernel" and previous["creation_success"]})
            if record is None:
                if target is not None:
                    target["nodes"].pop(uid, None)
                else:
                    self.nodes.pop(int(uid), None)
                continue
            update["replacements"][uid] = record["id"]
            if target is not None:
                target["nodes"][uid] = record["id"]
            else:
                self.nodes[int(uid)] = record

    def update_resource(self, callback_id, data):
        """Bind SDK update events to the active API and its exact node/exec."""
        stack = self.pending_updates.get(threading.get_ident(), [])
        if not stack:
            return False
        frame = stack[-1]
        descriptor = int(data.resource_descriptor)
        if not descriptor:
            frame["events_valid"] = False
            return True
        graph = self.capture.cupti.GraphData.from_ptr(descriptor, readonly=True)
        node = int(self.capture.cupti.get_graph_node_id(int(graph.node)))
        executable = int(getattr(graph, "graph_exec", 0) or 0)
        valid = (int(graph.node_type) == 0 and str(node) in frame["targets"] and int(data.context or 0) == frame["context"]
                 and (executable == frame["exec_handle"] if frame["scope"] == "executable" else executable == 0))
        frame["resource_events"].append({"callback_id": callback_id, "node_id": node, "exec_handle": executable, "context": int(data.context or 0)})
        frame["events_valid"] = frame["events_valid"] and valid
        if not valid:
            self.invalidate()
        return True

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
        add = name in {"cuGraphAddKernelNode", "cuGraphAddKernelNode_v2"}
        memory = name.startswith(("cuMemcpy", "cuMemset")) or name in {"cuGraphAddMemcpyNode", "cuGraphAddMemsetNode"}
        if entering:
            if name in TEMPLATE_UPDATES | EXEC_UPDATES:
                self.begin_update(data, key, context)
            elif kernel or add or instantiate or memory:
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
                self.invalidate()
            elif name == "cuGraphExecDestroy":
                self.executables.pop(C.c_void_p.from_address(pointer).value, None)
            elif name in {"cuGraphDestroy", "cuGraphDestroyNode"}:
                # A destroyed template does not change an already instantiated
                # executable's immutable snapshot. Its own provenance is gone.
                self.nodes.clear()
                self.by_graph.clear()
            return
        success = bool(data.function_return_value) and C.c_int.from_address(int(data.function_return_value)).value == 0
        if name in TEMPLATE_UPDATES | EXEC_UPDATES:
            self.finish_update(key, success)
        elif kernel or add or instantiate or memory:
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
            if self.update_resource(callback_id, data):
                return
            self.invalidate()
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
        return not self.pending_launches and not any(self.builders.values()) and not any(self.pending_updates.values())
