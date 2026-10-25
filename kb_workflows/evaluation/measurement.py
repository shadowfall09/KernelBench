"""Synchronized wall timing, varied-input validation and measured cold costs."""

from __future__ import annotations

import math
import secrets
import statistics
import subprocess
from time import perf_counter

import torch

from src.input_groups import copy_inputs
from src.validation import preserve_backend_settings

CLOCK = perf_counter
_SYNC = torch.cuda.synchronize
_TENSOR = torch.Tensor
_SEED = secrets.randbits


def fresh_seed():
    # Generated on demand, so a candidate cannot precompute a public seed schedule.
    from ..provenance.origin_task import next_seed

    observed = next_seed()
    return _SEED(63) if observed is None else observed


def synchronize(device):
    if torch.device(device).type == "cuda":
        _SYNC(device)
    elif str(device).split(":")[0] == "neuron":
        from ..hardware.devices import torch_adapter

        torch_adapter(device).synchronize()


def consume_output(output):
    """Require materialized tensors, including each member of a structured output."""
    if type(output) is _TENSOR:
        # The same small GPU read is included for reference and candidate.
        if output.numel():
            return output[(0,) * output.ndim].sum()
    elif isinstance(output, dict):
        return [consume_output(value) for value in output.values()]
    elif isinstance(output, (tuple, list)):
        return [consume_output(value) for value in output]
    else:
        raise TypeError(f"Output must be a materialized torch.Tensor, got {type(output).__name__}")


def tensor_bytes(value):
    if isinstance(value, _TENSOR):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def synchronized_times(fn, args, device, num_warmup=3, num_trials=100, receipt=None, input_factory=None, verifier=None,
                       origin_candidate=False):
    """Include host dispatch and completion of work on every stream in this process.

    Input allocation/copy is outside the interval and synchronized before timing.
    Fresh input objects prevent a same-object cache from winning steady-state time.
    """
    times = []
    from contextlib import nullcontext

    from ..provenance.origin_task import candidate_call, confirm, record_elapsed, snapshot_output
    with torch.no_grad():
        for trial in range(num_warmup + num_trials):
            fresh = input_factory(trial) if input_factory else copy_inputs(args)
            verification_args = copy_inputs(fresh) if verifier else None
            rng = torch.get_rng_state() if verifier else None
            cuda_rng = torch.cuda.get_rng_state(device) if verifier and torch.device(device).type == "cuda" else None
            neuron_rng = torch.neuron.get_rng_state(device) if verifier and str(device).split(":")[0] == "neuron" else None
            with preserve_backend_settings():
                synchronize(device)
                scope = candidate_call(fn, fresh, device=device, phase="warmup" if trial < num_warmup else "timing",
                                       label=str(trial)) if origin_candidate else nullcontext((fn, None))
                with scope as (execute, observed):
                    start = CLOCK()
                    output = execute(*fresh)
                    consumed = consume_output(output)
                    synchronize(device)
                    elapsed = (CLOCK() - start) * 1000
            record_elapsed(observed, elapsed)
            snapshot_output(observed, output)
            if verifier:
                after_rng = torch.get_rng_state()
                after_cuda_rng = torch.cuda.get_rng_state(device) if cuda_rng is not None else None
                after_neuron_rng = torch.neuron.get_rng_state(device) if neuron_rng is not None else None
                torch.set_rng_state(rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                if neuron_rng is not None:
                    torch.neuron.set_rng_state(neuron_rng, device)
                try:
                    verifier(verification_args, output)
                finally:
                    torch.set_rng_state(after_rng)
                    if after_cuda_rng is not None:
                        torch.cuda.set_rng_state(after_cuda_rng, device)
                    if after_neuron_rng is not None:
                        torch.neuron.set_rng_state(after_neuron_rng, device)
                confirm(observed)
            if trial == 0 and receipt is not None:
                receipt["first_timed_call_ms"] = elapsed
                receipt["output_bytes"] = tensor_bytes(output)
            if trial >= num_warmup:
                times.append(elapsed)
            elif receipt is not None:
                receipt["warmup_total_ms"] = receipt.get("warmup_total_ms", 0) + elapsed
            del output, consumed, fresh, verification_args
    if receipt is not None:
        receipt.update(timing_method="synchronized_wall", output_consumed=True, fresh_inputs=True,
                       varied_input_contents=input_factory is not None, timed_outputs_verified=verifier is not None)
    return times


def timing_stats(times):
    if not times or any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("Timing samples must be finite and positive")
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    radius = 1.96 * std / math.sqrt(len(times))
    return {
        "mean": mean, "std": std, "min": min(times), "max": max(times),
        "median": statistics.median(times), "num_trials": len(times), "samples_ms": list(times),
        "mean_ci95_ms": [max(0, mean - radius), mean + radius], "ci_method": "normal_approximation",
        "timing_method": "synchronized_wall", "time_unit": "ms",
    }


def input_signature(args):
    return [
        {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)}
        if isinstance(x, torch.Tensor) else {"type": type(x).__name__, "value": x}
        for x in args
    ]


def device_fingerprint(device):
    if str(device).split(":")[0] == "neuron":
        from ..hardware.devices import torch_adapter

        return torch_adapter(device).caps()
    info = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "hip_version": torch.version.hip}
    if not torch.cuda.is_available():
        return {**info, "available": False}
    properties = torch.cuda.get_device_properties(device)
    uuid = str(getattr(properties, "uuid", ""))
    driver = None
    if uuid and not torch.version.hip:
        try:
            query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=2, check=False)
            for line in query.stdout.splitlines() if query.returncode == 0 else []:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and fields[0] == uuid:
                    driver = fields[1]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        **info, "available": True, "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)), "total_memory": properties.total_memory,
        "multiprocessors": properties.multi_processor_count,
        "l2_bytes": getattr(properties, "L2_cache_size", None), "uuid": uuid, "driver_version": driver,
    }


def amortized(steady, cold, calls):
    """Average measured startup and steady work over the declared call budget."""
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (steady, cold)):
        raise ValueError("steady and cold must be finite positive times")
    return (cold + (calls - 1) * steady) / calls
