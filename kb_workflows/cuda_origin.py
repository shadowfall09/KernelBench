"""Optional CUPTI 13.3 / Triton 3.7 source-to-launch collector.

No vendor import or device work occurs at module import. This is a separate
diagnostic worker; unsupported launch/compilation paths remain unclassified.
The ctypes layouts below match NVIDIA's generated_cuda_meta.h and
cupti_callbacks.h on 64-bit Linux. SDK and ABI mismatches are not auto-adapted.
"""

from __future__ import annotations

import ctypes as C
import importlib.metadata
import platform
import threading
from contextlib import contextmanager

from .common import fingerprint
from .kernel_origin import MAX_RECORDS, PROTOCOL
from .submission_source import SourceRegistry, record_sources, sha256

MAX_CUBIN_BYTES = 128 * 1024**2


class ModuleResource(C.Structure):
    _fields_ = [("module_id", C.c_uint32), ("size", C.c_size_t), ("cubin", C.c_void_p)]


class ModuleLoad(C.Structure):
    # All supported cuModuleLoad* parameter layouts begin with CUmodule*.
    _fields_ = [("output", C.POINTER(C.c_void_p))]


class ModuleGetFunction(C.Structure):
    _fields_ = [("output", C.POINTER(C.c_void_p)), ("module", C.c_void_p), ("name", C.c_void_p)]


class ModuleUnload(C.Structure):
    _fields_ = [("module", C.c_void_p)]


class LaunchKernel(C.Structure):
    _fields_ = [("function", C.c_void_p), ("gx", C.c_uint32), ("gy", C.c_uint32), ("gz", C.c_uint32),
                ("bx", C.c_uint32), ("by", C.c_uint32), ("bz", C.c_uint32), ("shared", C.c_uint32),
                ("stream", C.c_void_p), ("params", C.c_void_p), ("extra", C.c_void_p)]


class LaunchKernelEx(C.Structure):
    _fields_ = [("config", C.c_void_p), ("function", C.c_void_p), ("params", C.c_void_p), ("extra", C.c_void_p)]


def abi_supported():
    return (platform.system() == "Linux" and C.sizeof(C.c_void_p) == 8
            and C.sizeof(ModuleResource) == 24 and ModuleResource.cubin.offset == 16
            and C.sizeof(LaunchKernel) == 64 and LaunchKernel.stream.offset == 40)


class Capture:
    def __init__(self, candidate_source, *, bindings=None):
        self.registry = SourceRegistry()
        self.cupti = bindings
        self.raw = {"protocol": PROTOCOL, "candidate_sha256": sha256(candidate_source.encode()),
                    "modules": [], "functions": [], "proofs": [], "launches": [], "activities": [], "windows": [],
                    "capture_complete": False, "source_intact": False, "hooks_intact": False,
                    "dropped_records": 0, "errors": [], "simulated": bindings is not None}
        self.lock = threading.RLock()
        self.loads, self.module_map, self.function_map = {}, {}, {}
        self.pending_launches, self.active_calls = {}, {}
        self.contexts = {}
        self.enabled = []
        self.subscriber = None
        self._original_handle_init = None
        self._wrapped_handle_init = None
        self._compiled_class = None
        self._proof_functions = set()

    def error(self, message):
        if len(self.raw["errors"]) < 100:
            self.raw["errors"].append(str(message)[:300])

    def append(self, label, item):
        rows = self.raw[label]
        if len(rows) >= MAX_RECORDS:
            self.error(f"{label} record limit exceeded")
            return None
        if label in {"modules", "functions", "proofs", "windows"}:
            item = {**item, "id": len(rows) + 1}
        rows.append(item)
        return item

    @staticmethod
    def buffer_requested():
        return 8 * 1024**2, 0

    def buffer_completed(self, records):
        try:
            with self.lock:
                for row in records:
                    kind = int(row.kind)
                    kinds = self.cupti.ActivityKind
                    if kind == int(kinds.CONCURRENT_KERNEL):
                        label = "kernel"
                    elif kind == int(kinds.MEMCPY):
                        label = "memcpy"
                    elif kind == int(kinds.MEMSET):
                        label = "memset"
                    else:
                        continue
                    self.append("activities", {"kind": label, "start_ns": int(row.start), "end_ns": int(row.end),
                        "device": int(row.device_id), "stream": int(row.stream_id), "context_id": int(row.context_id),
                        "correlation": int(row.correlation_id), "graph_id": int(getattr(row, "graph_id", 0)),
                        "graph_node_id": int(getattr(row, "graph_node_id", 0)),
                        "name": str(getattr(row, "name", ""))[:2048]})
        except Exception as exc:
            self.error("Activity callback: " + type(exc).__name__)

    def callback(self, unused, domain, callback_id, data):
        # NVIDIA forbids calling CUDA runtime/driver APIs from CUPTI callbacks.
        # Only copy documented callback fields here; no device queries or syncs.
        try:
            with self.lock:
                if int(domain) == int(self.cupti.CallbackDomain.DRIVER_API):
                    self.driver_callback(data)
                elif (int(domain) == int(self.cupti.CallbackDomain.RESOURCE)
                      and int(callback_id) == int(self.cupti.CallbackIdResource.MODULE_LOADED)):
                    self.module_loaded(data)
        except Exception as exc:
            self.error("Driver/resource callback: " + type(exc).__name__)

    def module_loaded(self, data):
        descriptor = int(data.resource_descriptor)
        if not descriptor:
            self.error("Missing module resource descriptor")
            return
        item = ModuleResource.from_address(descriptor)
        if not item.cubin or not 0 < item.size <= MAX_CUBIN_BYTES:
            self.error("Module binary unavailable or oversized")
            return
        # Cython's char* property is a string and cannot safely represent a cubin.
        binary_sha = sha256(C.string_at(item.cubin, item.size))
        pending = self.loads.get(threading.get_ident(), [])
        if not pending:
            return  # Unsupported loader: never invent a CUmodule association.
        pending[-1]["resources"].append({"resource_module_id": int(item.module_id),
            "binary_sha256": binary_sha, "binary_bytes": int(item.size), "context": int(data.context)})

    def driver_callback(self, data):
        name = str(data.function_name)
        tid = threading.get_ident()
        entering = int(data.callback_site) == int(self.cupti.ApiCallbackSite.API_ENTER)
        context, uid = int(data.context or 0), int(data.context_uid)
        if context:
            self.contexts[uid] = context
        params = int(data.function_params)
        if not params:
            return
        loading = name in {"cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx", "cuModuleLoadFatBinary"}
        launching = name in {"cuLaunchKernel", "cuLaunchKernel_ptsz", "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"}
        key = (uid, int(data.correlation_id))
        if entering:
            if loading:
                self.loads.setdefault(tid, []).append({"key": key, "resources": []})
            elif launching:
                struct = LaunchKernelEx if name.startswith("cuLaunchKernelEx") else LaunchKernel
                handle = struct.from_address(params).function
                function = self.function_map.get((context, handle))
                launch = self.append("launches", {"context_id": uid, "correlation": key[1],
                    "function": function["id"] if function else None, "call": self.active_calls.get(tid),
                    "exit_success": False, "api": name})
                if launch is not None:
                    self.pending_launches[key] = launch
            return
        success = bool(data.function_return_value) and C.c_int.from_address(int(data.function_return_value)).value == 0
        if loading:
            pending = self.loads.get(tid, [])
            if not pending or pending[-1]["key"] != key:
                self.error("Unbalanced module load callbacks")
                return
            loaded = pending.pop()
            pointer = ModuleLoad.from_address(params).output
            handle = pointer[0] if success and pointer else None
            resources = loaded["resources"]
            if not handle or len(resources) != 1 or resources[0]["context"] != context:
                return
            module = self.append("modules", {**resources[0], "handle": handle})
            if module:
                self.module_map[(context, handle)] = module
        elif name == "cuModuleGetFunction" and success:
            item = ModuleGetFunction.from_address(params)
            module = self.module_map.get((context, item.module))
            handle = item.output[0] if item.output else None
            if handle and module:
                function = self.append("functions", {"handle": handle, "module": module["id"], "context": context})
                if function:
                    self.function_map[(context, handle)] = function
        elif name == "cuModuleUnload" and success:
            handle = ModuleUnload.from_address(params).module
            module = self.module_map.pop((context, handle), None)
            if module:
                self.function_map = {key: value for key, value in self.function_map.items() if value["module"] != module["id"]}
        elif launching:
            launch = self.pending_launches.pop(key, None)
            if launch is not None:
                launch["exit_success"] = success

    def bind_triton(self, compiled):
        owner = self.registry.jit_source(getattr(getattr(compiled, "src", None), "fn", None))
        if owner is None:
            return
        binary = compiled.kernel
        if not isinstance(binary, bytes) or not binary:
            return
        # A real function lookup and the real loaded bytes must agree with the
        # compiler output; kernel names and paths do not participate in the join.
        with self.lock:
            matches = [value for value in self.function_map.values()
                       if value["handle"] == compiled.function
                       and self.raw["modules"][value["module"] - 1]["handle"] == compiled.module]
            if len(matches) != 1:
                return
            function = matches[0]
            module = self.raw["modules"][function["module"] - 1]
            if module["binary_sha256"] != sha256(binary) or function["id"] in self._proof_functions:
                return
            self._proof_functions.add(function["id"])
            self.append("proofs", {"function": function["id"], "compiler": "triton-3.7.0",
                "source_sha256": owner["source_sha256"], "jit_source_sha256": owner["jit_source_sha256"],
                "binary_sha256": module["binary_sha256"],
                "compiler_metadata_sha256": fingerprint(dict(compiled.metadata._asdict()))})

    def install_triton(self):
        if importlib.metadata.version("triton") != "3.7.0":
            raise RuntimeError("Origin compiler adapter requires Triton 3.7.0")
        from triton.compiler.compiler import CompiledKernel

        self._compiled_class = CompiledKernel
        original = CompiledKernel._init_handles
        self._original_handle_init = original

        def instrumented(compiled):
            output = original(compiled)
            try:
                self.bind_triton(compiled)
            except Exception as exc:
                self.error("Compiler association: " + type(exc).__name__)
            return output

        self._wrapped_handle_init = instrumented
        CompiledKernel._init_handles = instrumented

    def start(self):
        if not abi_supported():
            raise RuntimeError("Origin callback ABI requires 64-bit Linux")
        if self.cupti is None:
            if importlib.metadata.version("cupti-python") != "13.3.0":
                raise RuntimeError("Origin collector requires cupti-python 13.3.0")
            from cupti import cupti

            self.cupti = cupti
        self.cupti.activity_register_callbacks(self.buffer_requested, self.buffer_completed)
        self.subscriber = self.cupti.subscribe(self.callback, None)
        self.cupti.enable_domain(1, self.subscriber, self.cupti.CallbackDomain.DRIVER_API)
        self.cupti.enable_domain(1, self.subscriber, self.cupti.CallbackDomain.RESOURCE)
        for label in ("CONCURRENT_KERNEL", "MEMCPY", "MEMSET"):
            kind = getattr(self.cupti.ActivityKind, label)
            self.cupti.activity_enable(kind)
            self.enabled.append(kind)

    @contextmanager
    def call(self, *, seed, signature, device=0, training=True):
        tid = threading.get_ident()
        if tid in self.active_calls:
            raise RuntimeError("Nested origin call scopes are unsupported")
        row = self.append("windows", {"seed": seed, "input_signature": signature, "device": device,
            "training": training, "no_grad": True, "start_ns": int(self.cupti.get_timestamp()),
            "end_ns": None, "output_verified": False})
        if row is None:
            raise RuntimeError("Origin call limit exceeded")
        self.active_calls[tid] = row["id"]
        try:
            yield row
        finally:
            row["end_ns"] = int(self.cupti.get_timestamp())
            self.active_calls.pop(tid, None)

    def stop(self):
        # Flush before disable; query every observed context/stream queue as well
        # as the global queue. Unknown drop accounting blocks attribution.
        try:
            self.cupti.activity_flush_all(1)
            queues = {(0, 0), *((context, 0) for context in self.contexts.values())}
            for row in self.raw["activities"]:
                context = self.contexts.get(row["context_id"])
                if context is None:
                    self.error("Activity has no observed context handle for drop accounting")
                else:
                    queues.add((context, row["stream"]))
            for context, stream in sorted(queues):
                dropped = C.c_size_t()
                self.cupti.activity_get_num_dropped_records(context, stream, C.addressof(dropped))
                self.raw["dropped_records"] += dropped.value
            self.raw["capture_complete"] = (not self.pending_launches and not self.active_calls
                                            and not any(self.loads.values()))
        except Exception as exc:
            self.error("Flush/drop accounting: " + type(exc).__name__)
        finally:
            for kind in reversed(self.enabled):
                try:
                    self.cupti.activity_disable(kind)
                except Exception as exc:
                    self.error("Disable activity: " + type(exc).__name__)
            if self.subscriber is not None:
                try:
                    self.cupti.unsubscribe(self.subscriber)
                except Exception as exc:
                    self.error("Unsubscribe: " + type(exc).__name__)
            if self._compiled_class is not None:
                self.raw["hooks_intact"] = self._compiled_class._init_handles is self._wrapped_handle_init
                self._compiled_class._init_handles = self._original_handle_init
            self.raw["source_intact"] = bool(self.registry.sources) and self.registry.intact()

    @contextmanager
    def recording(self):
        try:
            self.start()
            self.install_triton()
            with record_sources(self.registry):
                yield self
        finally:
            if self.cupti is not None:
                self.stop()
