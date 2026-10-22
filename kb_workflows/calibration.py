"""Replayable device calibration records; importing this module never runs a probe."""

from __future__ import annotations

import copy
import math
import statistics
from datetime import datetime, timezone

from .common import evaluator_digest, fingerprint

PROTOCOL = "device_calibration_v1"
IDENTITY_FIELDS = ("name", "capability", "total_memory", "multiprocessors", "l2_bytes",
                   "uuid", "torch_version", "cuda_version", "hip_version", "driver_version")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def identity(device):
    if (not isinstance(device, dict) or device.get("available") is not True or not device.get("uuid") or not device.get("driver_version")
            or any(name not in device for name in IDENTITY_FIELDS)):
        raise ValueError("Calibration requires a physical device UUID and software identity")
    return fingerprint({name: device[name] for name in IDENTITY_FIELDS})


def probe_record(work, samples_ms, *, unit, parameters):
    if (not positive(work) or not isinstance(samples_ms, list) or len(samples_ms) < 5
            or not all(map(positive, samples_ms)) or unit not in {"bytes/s", "operations/s"}
            or not isinstance(parameters, dict)):
        raise ValueError("Calibration needs positive work and at least five positive timing samples")
    return {"work_per_call": work, "samples_ms": samples_ms, "unit": unit, "parameters": parameters,
            "rate": work / (statistics.median(samples_ms) / 1000),
            "rate_policy": "work_over_median_synchronized_wall_time" if parameters.get("time_method", "").startswith("synchronized_wall")
                else "work_over_median_device_event_time"}


def seal(device, probes, launch_samples_ms, *, settings, digest=None):
    if not probes or len(launch_samples_ms) < 5 or not all(map(positive, launch_samples_ms)):
        raise ValueError("Calibration requires probes and measured launch samples")
    result = {"protocol": PROTOCOL, "device": device, "device_identity_sha256": identity(device),
              "evaluator_sha256": digest or evaluator_digest(), "settings": settings,
              "created_at": datetime.now(timezone.utc).isoformat(), "probes": probes,
              "launch_samples_ms": launch_samples_ms, "launch_median_ms": statistics.median(launch_samples_ms),
              "launch_method": "synchronized_wall_in_place_one_element_add",
              "interpretation": "attained_microbenchmark_rates; not_proven_physical_upper_bounds",
              "clock_and_power_state_certified": False, "experiments_run": True}
    result["receipt_sha256"] = fingerprint(result)
    validate(result, device, digest=result["evaluator_sha256"])
    return result


def validate(record, device, *, digest=None):
    if (not isinstance(record, dict) or record.get("protocol") != PROTOCOL or record.get("experiments_run") is not True
            or record.get("device_identity_sha256") != identity(device)
            or record.get("device_identity_sha256") != identity(record.get("device", {}))
            or record.get("evaluator_sha256") != (digest or evaluator_digest())
            or record.get("receipt_sha256") != fingerprint({k: v for k, v in record.items() if k != "receipt_sha256"})):
        raise ValueError("Missing, changed, stale, or different-device calibration")
    probes = record.get("probes", {})
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Calibration probes are missing")
    for item in probes.values():
        if not isinstance(item, dict):
            raise ValueError("Malformed calibration probe")
        expected = probe_record(item["work_per_call"], item["samples_ms"], unit=item["unit"], parameters=item["parameters"])
        if item != expected:
            raise ValueError("Calibration rate does not match its raw samples")
        sweep = item["parameters"].get("sweep", [])
        if not isinstance(sweep, list):
            raise ValueError("Calibration sweep must be a list")
        for other in sweep:
            if (not isinstance(other, dict) or not isinstance(other.get("parameters"), dict)
                    or "sweep" in other["parameters"]):
                raise ValueError("Malformed calibration sweep entry")
            repeated = probe_record(other["work_per_call"], other["samples_ms"], unit=other["unit"], parameters=other["parameters"])
            if repeated != other or other["unit"] != item["unit"] or other["rate"] > item["rate"]:
                raise ValueError("Calibration winner does not match the retained sweep")
    samples = record.get("launch_samples_ms", [])
    if len(samples) < 5 or not all(map(positive, samples)) or record.get("launch_median_ms") != statistics.median(samples):
        raise ValueError("Launch calibration does not match its raw samples")
    return record


class CUDADevice:
    """Trusted probes only. This adapter is not a candidate sandbox."""

    def __init__(self, device="cuda:0"):
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
            raise ValueError("CUDA calibration requires an available NVIDIA device")

    def caps(self):
        from .measurement import device_fingerprint

        return device_fingerprint(self.device)

    def attribute(self, trace, **settings):
        from .performance import timeline

        return timeline(trace, device=self.device.index or 0, **settings)

    def flush(self, bytes_=None):
        import torch

        l2 = self.caps().get("l2_bytes")
        if not l2 and bytes_ is None:
            raise ValueError("Unknown cache size; supply an explicit flush byte count")
        size = bytes_ if bytes_ is not None else max(32 * 1024**2, 4 * l2)
        if type(size) is not int or size <= 0:
            raise ValueError("flush bytes must be positive")
        scratch = torch.empty(size, dtype=torch.uint8, device=self.device)
        scratch.zero_()
        torch.cuda.synchronize(self.device)
        return {"bytes_written": size, "policy": "write_buffer_larger_than_reported_L2",
                "complete_cache_eviction_certified": False}

    def time(self, function, *, trials=15, warmup=3):
        import torch

        if type(trials) is not int or trials < 5 or type(warmup) is not int or warmup < 0:
            raise ValueError("Use at least five trials and nonnegative warmup")
        samples = []
        with torch.cuda.device(self.device), torch.no_grad():
            for index in range(warmup + trials):
                torch.cuda.synchronize(self.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = function()
                end.record()
                end.synchronize()
                if index >= warmup:
                    samples.append(start.elapsed_time(end))
                del output
        return samples

    def calibrate(self, *, matrix_sizes=(1024, 2048, 4096), memory_bytes=256 * 1024**2, trials=15):
        import torch

        from src.validation import preserve_backend_settings

        from .measurement import CLOCK

        caps = self.caps()
        identity(caps)
        if (not positive(caps.get("l2_bytes")) or type(memory_bytes) is not int or memory_bytes <= 2 * caps["l2_bytes"]
                or not matrix_sizes or any(type(n) is not int or n < 128 for n in matrix_sizes)):
            raise ValueError("Memory probe must exceed twice L2; GEMM sizes must be integers >=128")
        # Refuse oversized allocations rather than silently changing the declared probes.
        largest = max(matrix_sizes)
        required = max(2 * memory_bytes, 3 * largest**2 * 8)
        free, _ = torch.cuda.mem_get_info(self.device)
        if required > free * 0.6:
            raise ValueError("Calibration probes exceed 60% of currently free device memory")
        probes = {}
        with torch.cuda.device(self.device), torch.no_grad(), preserve_backend_settings():
            # Use the current precision API when available; never mix both APIs.
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.fp32_precision = "ieee"
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                torch.backends.cudnn.fp32_precision = "ieee"
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            source = torch.empty(memory_bytes, dtype=torch.uint8, device=self.device).fill_(7)
            destination = torch.empty_like(source)
            probes["memory_bytes"] = probe_record(2 * memory_bytes,
                self.time(lambda dst=destination, src=source: dst.copy_(src), trials=trials), unit="bytes/s",
                parameters={"bytes_per_buffer": memory_bytes, "direction": "device_to_device", "traffic": "read_plus_write"})
            del source, destination
            formats = [("fp32_flops", torch.float32), ("fp16_flops", torch.float16), ("fp64_flops", torch.float64)]
            if torch.cuda.is_bf16_supported():
                formats.append(("bf16_flops", torch.bfloat16))
            for name, dtype in formats:
                choices = []
                for n in matrix_sizes:
                    a = torch.randn(n, n, device=self.device, dtype=dtype)
                    b = torch.randn_like(a)
                    c = torch.empty_like(a)
                    choices.append(probe_record(2 * n**3, self.time(lambda a_=a, b_=b, c_=c: torch.mm(a_, b_, out=c_), trials=trials),
                        unit="operations/s", parameters={"shape": [n, n, n], "dtype": str(dtype), "tf32": False,
                        "reduced_precision_reduction": False}))
                    del a, b, c
                best = max(range(len(choices)), key=lambda i: choices[i]["rate"])
                # Retain the entire sweep separately in settings, not just its winner.
                probes[name] = copy.deepcopy(choices[best])
                probes[name]["parameters"]["sweep"] = choices[:best] + choices[best + 1:]
            x = torch.zeros(1, device=self.device)
            launch = []
            for index in range(trials + 3):
                torch.cuda.synchronize(self.device)
                started = CLOCK()
                x.add_(1)
                torch.cuda.synchronize(self.device)
                if index >= 3:
                    launch.append((CLOCK() - started) * 1000)
        return seal(caps, probes, launch, settings={"matrix_sizes": list(matrix_sizes), "memory_bytes": memory_bytes,
                    "trials": trials, "time_method": "CUDA_events_current_stream", "cache_policy": "steady_state"})
