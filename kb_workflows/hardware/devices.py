"""Vendor adapters and declarative device configuration; no SDK imports at module load."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

PROTOCOL = "kernelbench_pro_device_adapter_v1"
NEURON_TARGETS = {"Trainium2": "trn2", "Trainium3": "trn3"}


@runtime_checkable
class DeviceAdapter(Protocol):
    def caps(self): ...
    def calibrate(self, **settings): ...
    def flush(self, bytes_=None): ...
    def time(self, function, *, trials=15, warmup=3): ...
    def attribute(self, trace, **settings): ...


def device_options(section, *, directory=None):
    from .gpu import gpu_options

    if not isinstance(section, dict) or set(section) - {"backend", "hardware", "target", "devices", "python"}:
        raise ValueError("Unknown [device] fields; use backend, hardware, target, devices and python")
    backend = section.get("backend", "cuda")
    if backend not in {"cuda", "neuron"}:
        raise ValueError("Supported device backends are cuda and neuron")
    gpu = {key: value for key, value in section.items() if key in {"hardware", "devices"}}
    if backend == "cuda":
        if section.get("target"):
            gpu["architecture"] = section["target"]
        result = gpu_options(gpu)
    else:
        hardware = section.get("hardware", "")
        target = section.get("target", NEURON_TARGETS.get(hardware, ""))
        if hardware not in NEURON_TARGETS or target != NEURON_TARGETS[hardware]:
            raise ValueError("Native Neuron requires hardware Trainium2/trn2 or Trainium3/trn3")
        result = gpu_options({"devices": section.get("devices", [0])})
        result.update(hardware=hardware, gpu_arch=target, gpu="")
    result["device_backend"] = backend
    if "python" in section:
        value = section["python"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("device.python must name an explicit Python interpreter")
        path = Path(value).expanduser()
        result["device_python"] = str((path if path.is_absolute() else Path(directory or Path.cwd()) / path).resolve())
    return result


@lru_cache(maxsize=16)
def adapter(backend="cuda", device="0"):
    if backend == "cuda":
        from ..analysis.calibration import CUDADevice

        return CUDADevice(f"cuda:{device}")
    if backend == "neuron":
        from .neuron_device import NeuronDevice

        return NeuronDevice(f"neuron:{device}")
    raise ValueError("Unknown device adapter")


def torch_adapter(device):
    """Use only after an explicit execution request; adapters cache trusted sync functions."""
    name, _, ordinal = str(device).partition(":")
    return adapter(name, ordinal or "0")
