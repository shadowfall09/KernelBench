"""Native TorchNeuron 2.14 / NKI adapter for Trainium2 and Trainium3.

Import is lightweight. Construction and methods are explicit hardware execution
paths. XLA-based torch-neuronx and CPU fallback are not interchangeable backends.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from ..analysis.calibration import identity, positive, probe_record, seal
from ..common import fingerprint


def core_ids(value):
    """Parse the documented visible-core list without treating unknown IDs as zero."""
    if not isinstance(value, str) or not value:
        raise ValueError("Neuron visible cores must be a nonempty list or inclusive ranges")
    result = []
    for token in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError("Invalid NEURON_RT_VISIBLE_CORES")
        ends = list(map(int, token.split("-")))
        if len(ends) == 2 and (ends[1] < ends[0] or ends[1] - ends[0] > 4096):
            raise ValueError("Invalid Neuron core range")
        result.extend(range(ends[0], ends[-1] + 1))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate Neuron core IDs")
    return result


def inventory_record(rows, *, core, sysfs=Path("/sys/devices/virtual/neuron_device"), driver=Path("/sys/module/neuron/version")):
    matches = [row for row in rows if core in row.get("neuroncore_ids", [])]
    if len(matches) != 1:
        raise ValueError("Selected physical NeuronCore is absent or ambiguous in neuron-ls")
    row = matches[0]
    directory = Path(sysfs) / f"neuron{row['neuron_device']}" / "info"
    serial = (directory / "serial_number").read_text().strip()
    name = (directory / "architecture/device_name").read_text().strip()
    loaded_driver = Path(driver).read_text().strip()
    if not serial or not name or not loaded_driver or not positive(row.get("memory_size")):
        raise ValueError("Neuron device identity, loaded driver or memory size is unknown")
    return {"serial_number": serial, "physical_core": core, "neuron_device": row["neuron_device"],
            "core_index_within_device": row["neuroncore_ids"].index(core),
            "bdf": row.get("bdf"), "name": name, "total_memory": row["memory_size"],
            "logical_neuroncore_config": row.get("logical_neuroncore_config"), "driver_version": loaded_driver}


def native_versions(torch_version, neuron_version, nki_version):
    from packaging.version import Version

    torch, neuron, nki = map(Version, (torch_version, neuron_version, nki_version))
    if (torch.release[:2] != (2, 14) or neuron.release[:2] != (2, 14)
            or len(neuron.release) < 3 or neuron.release[2] < 3 or not Version("0.6") <= nki < Version("0.8")):
        raise ValueError("Use native torch==2.14.*, torch-neuronx>=2.14.3,<2.15 and nki>=0.6,<0.8; XLA packages are not supported")


class NeuronDevice:
    def __init__(self, device="neuron:0"):
        if os.environ.get("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1") != "1":
            raise ValueError("Neuron CPU fallback must be disabled before loading the SDK")
        os.environ["TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION"] = "1"
        import torch
        import torch_neuronx

        native_versions(torch.__version__, version("torch-neuronx"), version("nki"))
        if not hasattr(torch_neuronx, "nki_op") or not hasattr(torch, "neuron"):
            raise ValueError("Native TorchNeuron/NKI registration is unavailable")
        self.device = torch.device(device)
        if self.device.type != "neuron" or int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1")) != 1:
            raise ValueError("This serial device adapter requires neuron devices with LNC=1")
        self.backend, self.torch = torch_neuronx, torch
        self._sync = torch_neuronx.synchronize
        self._clock = perf_counter
        self._versions = {"torch_neuronx": version("torch-neuronx"), "nki": version("nki"), "neuronx_cc": version("neuronx-cc")}
        if torch_neuronx.device_count() <= (self.device.index or 0):
            raise ValueError("Requested native Neuron device is unavailable")

    def caps(self):
        visible = core_ids(os.environ.get("NEURON_RT_VISIBLE_CORES", ""))
        index = self.device.index or 0
        if index >= len(visible):
            raise ValueError("Neuron device ordinal is outside the explicit core allocation")
        process = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10, check=True)
        physical = inventory_record(json.loads(process.stdout), core=visible[index])
        if physical.get("logical_neuroncore_config") not in {None, 1}:
            raise ValueError("Observed Neuron core topology differs from the required LNC=1 configuration")
        properties = self.backend.get_device_properties(index)
        if not positive(properties.total_memory):
            raise ValueError("Native framework could not establish Neuron device memory")
        result = {"available": True, "device_backend": "neuron", "name": physical["name"],
                  "capability": ["NKI", "LNC1"], "total_memory": min(physical["total_memory"], properties.total_memory),
                  "multiprocessors": 1, "l2_bytes": None, "torch_version": self.torch.__version__,
                  "cuda_version": None, "hip_version": None, "driver_version": physical["driver_version"],
                  "uuid": "neuron:" + fingerprint(physical), "physical_device": physical, "software_versions": self._versions,
                  "float64_arithmetic": False, "cpu_fallback_allowed": False, "native_low_precision_execution_certified": False}
        # Include the compiler/NKI versions in the identity consumed by calibration
        # and scores, without pretending these vendor versions are CUDA versions.
        result["driver_version"] = fingerprint({"loaded_driver": physical["driver_version"], "software": self._versions})
        identity(result)
        return result

    def synchronize(self):
        self._sync(self.device)

    def flush(self, bytes_=None):
        if bytes_ is None:
            raise ValueError("Neuron cache topology is not reported; provide an explicit scratch-byte policy")
        if type(bytes_) is not int or bytes_ < 1:
            raise ValueError("Scratch byte count must be positive")
        if bytes_ > self.caps()["total_memory"] // 10:
            raise ValueError("Flush scratch exceeds the declared device-memory envelope")
        scratch = self.torch.empty(bytes_, dtype=self.torch.uint8, device=self.device)
        scratch.zero_()
        self.synchronize()
        return {"bytes_written": bytes_, "policy": "explicit_device_scratch_write; cache_topology_unknown",
                "complete_cache_eviction_certified": False}

    def time(self, function, *, trials=15, warmup=3):
        from ..evaluation.measurement import consume_output

        if type(trials) is not int or trials < 5 or type(warmup) is not int or warmup < 0:
            raise ValueError("Use at least five trials and nonnegative warmup")
        samples = []
        with self.torch.no_grad():
            for index in range(warmup + trials):
                self.synchronize()
                started = self._clock()
                output = function()
                consumed = consume_output(output)
                self.synchronize()
                elapsed = (self._clock() - started) * 1000
                if not positive(elapsed):
                    raise ValueError("Nonpositive Neuron synchronized-wall time")
                if index >= warmup:
                    samples.append(elapsed)
                del output, consumed
        return samples

    def calibrate(self, *, matrix_sizes=(1024, 2048, 4096), memory_bytes=256 * 1024**2, trials=15):
        from ..evaluation.measurement import consume_output

        torch, caps = self.torch, self.caps()
        if (not matrix_sizes or any(type(n) is not int or n < 128 for n in matrix_sizes)
                or type(memory_bytes) is not int or memory_bytes < 1
                or max(2 * memory_bytes, 3 * max(matrix_sizes)**2 * 4) > caps["total_memory"] // 10):
            raise ValueError("Calibration probes must fit 10% of device memory and use GEMM sizes >=128")
        probes = {}
        with torch.no_grad():
            source = torch.ones(memory_bytes, dtype=torch.uint8, device=self.device)
            target = torch.empty_like(source)
            probes["memory_bytes"] = probe_record(2 * memory_bytes, self.time(lambda dst=target, src=source: dst.copy_(src), trials=trials),
                unit="bytes/s", parameters={"bytes_per_buffer": memory_bytes, "direction": "device_to_device",
                    "traffic": "read_plus_write", "time_method": "synchronized_wall; includes_dispatch_and_output_consumption"})
            del source, target
            for name, dtype in (("fp32_flops", torch.float32), ("fp16_flops", torch.float16), ("bf16_flops", torch.bfloat16)):
                choices = []
                for n in matrix_sizes:
                    a = torch.randn(n, n, dtype=dtype).to(self.device)
                    b = torch.randn(n, n, dtype=dtype).to(self.device)
                    model = torch.compile(lambda a_, b_: torch.mm(a_, b_), backend="neuron", fullgraph=True, dynamic=False)
                    choices.append(probe_record(2 * n**3, self.time(lambda fn=model, x=a, y=b: fn(x, y), trials=trials),
                        unit="operations/s", parameters={"shape": [n, n, n], "dtype": str(dtype),
                            "implementation": "torch_compile_neuron", "time_method": "synchronized_wall"}))
                    del a, b, model
                best = max(range(len(choices)), key=lambda i: choices[i]["rate"])
                probes[name] = copy.deepcopy(choices[best])
                probes[name]["parameters"]["sweep"] = choices[:best] + choices[best + 1:]
            x = torch.zeros(1, device=self.device)
            launch = []
            for index in range(trials + 3):
                self.synchronize()
                started = self._clock()
                output = x.add_(1)
                consumed = consume_output(output)
                self.synchronize()
                if index >= 3:
                    launch.append((self._clock() - started) * 1000)
                del output, consumed
        return seal(caps, probes, launch, settings={"matrix_sizes": list(matrix_sizes), "memory_bytes": memory_bytes,
            "trials": trials, "time_method": "synchronized_wall", "cache_policy": "steady_state; unknown_cache_topology",
            "float64_probe": "not_applicable; native_device_uses_float32", "device_backend": "neuron"})

    def attribute(self, trace, **settings):
        from .neuron_tables import timeline

        return timeline(trace, **settings)

    def profile(self, function, *, directory, verifier=None):
        from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

        torch = self.torch
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Neuron capture must use a fresh artifact directory")
        directory.mkdir(parents=True, exist_ok=True)
        self.synchronize()
        cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")
        options = NeuronConfig(modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME], profile_output_dir=str(directory),
                               **({"neff_cache_dir": cache} if cache else {}))
        exporter = NeuronProfiler(options)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1],
                                    experimental_config=options, on_trace_ready=exporter.export_trace) as capture:
            with torch.profiler.record_function("kernelbench_profile"):
                output = function()
                self.synchronize()
        if verifier is not None:
            verifier(output)
        path = directory / "framework_trace.json"
        capture.export_chrome_trace(str(path))
        del output
        from .neuron_tables import file_digest

        files = {str(item.relative_to(directory)): item for item in directory.rglob("*")
                 if item.is_file() and not item.is_symlink()}
        return {"framework_trace": str(path), "vendor_artifact_directory": str(directory),
                "artifact_files": {name: file_digest(item) for name, item in files.items()},
                "artifact_file_bytes": {name: item.stat().st_size for name, item in files.items()},
                "output_verified": verifier is not None,
                "exporter": "NeuronProfiler.export_trace", "timings_are_diagnostic": True, "hardware_counters_collected": False}
