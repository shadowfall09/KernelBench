"""Development-only profiling. Instrumented durations never enter benchmark scores."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .common import ROOT, fingerprint


def profile(payload):
    import torch

    from src.eval import (
        _process_input_tensor,
        get_torch_dtype_from_string,
        load_custom_model,
        load_original_model_and_inputs,
    )
    from src.utils import set_gpu_arch
    from src.validation import clone_tensors

    from .evaluate_task import configure_tf32, validate_sources

    settings = payload["eval_config"]
    if settings.get("use_hidden_tests"):
        raise ValueError("Development profiling cannot access hidden tests")
    reference = validate_sources(payload)
    device = torch.device("cuda:0")
    dtype = get_torch_dtype_from_string(settings["precision"])
    configure_tf32(torch, False)
    if settings.get("gpu_arch"):
        set_gpu_arch(settings["gpu_arch"].split(","))
    _, init, inputs = load_original_model_and_inputs(reference, {})
    initial = [_process_input_tensor(x, device, dtype) for x in init()]
    args = [_process_input_tensor(x, device, dtype) for x in inputs()]
    build = ROOT / "cache/workflow" / fingerprint(payload)
    build.mkdir(parents=True, exist_ok=True)
    ModelNew = load_custom_model(payload["kernel_src"], {}, str(build))
    model = ModelNew(*initial).to(device=device, dtype=dtype)
    with torch.no_grad():
        model(*clone_tensors(args))
        torch.cuda.synchronize(device)
        profile_args = clone_tensors(args)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as capture:
            with torch.profiler.record_function("kernelbench_profile"):
                output = model(*profile_args)
                torch.cuda.synchronize(device)
        del output
    from .performance import timeline

    with tempfile.TemporaryDirectory(prefix="kb-profile-") as directory:
        trace_path = Path(directory) / "trace.json"
        capture.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
    try:
        activity = timeline(trace, device=0)
        activity.pop("activities")  # Keep the interactive response small.
    except (KeyError, TypeError, ValueError) as exc:
        activity = {"status": "not_measured", "reason": str(exc)}
    events = [{"name": item.key, "calls": item.count, "cpu_self_us": item.self_cpu_time_total,
               "device_self_us": getattr(item, "self_device_time_total", 0)} for item in capture.key_averages()]
    events.sort(key=lambda row: row["device_self_us"], reverse=True)
    return {"status": "ok", "purpose": "development_diagnostics_only", "events": events[:15],
            "device_activity": activity, "trace_sha256": fingerprint(trace),
            "timings_are_diagnostic": True, "hardware_counters_collected": False,
            "hint": "Inspect kernel count and CPU/device time; these are profiler durations, not scored benchmark times"}
