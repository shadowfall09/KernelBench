"""Run exactly one task in a disposable process (also used inside Modal)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, json_safe, problem_paths


def configure_tf32(torch, enabled):
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
        torch.backends.cudnn.fp32_precision = "tf32" if enabled else "ieee"
    else:
        torch.set_float32_matmul_precision("high" if enabled else "highest")
        torch.backends.cudnn.allow_tf32 = enabled


def validate_sources(payload):
    if payload["eval_config"].get("evaluator_sha256") != evaluator_digest():
        raise ValueError("Client and verifier evaluator versions differ; rebuild the verifier image/checkout")
    reference = problem_paths(payload["level"])[payload["problem_id"]].read_text()
    if reference != payload["ref_arch_src"]:
        raise ValueError("Client and verifier reference sources differ; use the same Verified checkout")
    return reference


def evaluate(payload: dict):
    import torch

    from src.eval import (
        check_metadata_serializable_all_types,
        eval_kernel_against_ref,
        get_hidden_test_path,
        get_torch_dtype_from_string,
    )
    from src.utils import set_gpu_arch

    settings = payload["eval_config"]
    hidden = settings.get("use_hidden_tests", False)
    path = get_hidden_test_path(payload["level"], payload["problem_id"]) if hidden else None
    if hidden and path is None:
        raise FileNotFoundError(f"Missing hidden tests for level {payload['level']}, problem {payload['problem_id']}")
    if hidden and hashlib.sha256(Path(path).read_bytes()).hexdigest() != settings.get("hidden_source_sha256"):
        raise ValueError("Client and verifier hidden test sources differ")
    reference = validate_sources(payload)
    os.environ["KB_FP32_TOL"] = str(settings["fp32_tolerance"])
    # Correctness is checked at full FP32 precision; TF32 applies to the timed baseline.
    configure_tf32(torch, False)
    arch = settings.get("gpu_arch", "")
    if arch:
        set_gpu_arch(arch.split(","))
    signature = fingerprint(payload)
    build = Path(payload.get("build_root", ROOT / "cache" / "workflow")) / signature
    build.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    result = eval_kernel_against_ref(
        original_model_src=reference,
        custom_model_src=payload["kernel_src"],
        num_correct_trials=settings["num_correct_trials"],
        num_perf_trials=settings["num_perf_trials"],
        measure_performance=settings.get("measure_performance", not hidden),
        verbose=settings.get("verbose", False),
        build_dir=str(build),
        device=torch.device("cuda:0"),
        precision=get_torch_dtype_from_string(settings["precision"]),
        hidden_test_path=path,
    )
    if result is None:
        return failure("Evaluator returned no result (possible compilation lock failure)")
    runtime = result.runtime if isinstance(result.runtime, (int, float)) and math.isfinite(result.runtime) else -1.0
    result.metadata.update(
        torch_version=torch.__version__, precision=settings["precision"], fp32_tolerance=settings["fp32_tolerance"]
    )
    return json_safe(
        {
            "compiled": result.compiled,
            "correctness": result.correctness,
            "metadata": check_metadata_serializable_all_types(result.metadata),
            "runtime": runtime,
            "runtime_stats": result.runtime_stats,
            "peak_memory": result.peak_memory,
            "memory_stats": result.memory_stats,
        }
    )


def baseline(payload: dict):
    import torch

    from src.eval import (
        _process_input_tensor,
        get_memory_stats,
        get_timing_stats,
        get_torch_dtype_from_string,
        load_original_model_and_inputs,
        set_seed,
        time_execution_with_cuda_event,
    )

    reference = validate_sources(payload)
    settings = payload["eval_config"]
    tf32 = settings["precision"] == "fp32"
    configure_tf32(torch, tf32)
    device = torch.device("cuda:0")
    dtype = get_torch_dtype_from_string(settings["precision"])
    Model, init, inputs = load_original_model_and_inputs(reference, {})
    set_seed(42)
    with torch.no_grad():
        init_inputs = [_process_input_tensor(x, device, dtype) for x in init()]
        set_seed(42)
        model = Model(*init_inputs).to(device=device, dtype=dtype)
        set_seed(42)
        args = [_process_input_tensor(x, device, dtype) for x in inputs()]
        backend = settings.get("compile_backend")
        if backend:
            options = {"backend": backend}
            if backend == "inductor":
                options["mode"] = settings["compile_mode"]
            model = torch.compile(model, **options)
        times = time_execution_with_cuda_event(
            model, *args, num_trials=settings["num_perf_trials"], device=device, verbose=False
        )
        stats = get_timing_stats(times, device=device)
        memory = get_memory_stats(model, *args, device=device)
        stats.update(peak_memory=memory["peak_bytes"], memory_stats=memory)
    return json_safe(
        {
            **stats,
            "precision": settings["precision"],
            "tf32": tf32,
            "torch_version": torch.__version__,
            "timing_method": "cuda_event",
            "time_unit": "ms",
            "compile_backend": backend,
            "compile_mode": settings.get("compile_mode"),
            "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
        }
    )


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = baseline(payload) if payload.get("task") == "baseline" else evaluate(payload)
    except Exception as exc:
        result = failure(str(exc), error_name=type(exc).__name__)
    atomic_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
