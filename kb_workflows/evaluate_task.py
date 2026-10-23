"""Run exactly one task in a disposable process (also used inside Modal)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path

from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, json_safe, problem_paths
from .production import task_input_policy


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
        raise ValueError("Client and verifier reference sources differ; use the same repository checkout")
    return reference


@task_input_policy
def evaluate(payload: dict):
    import torch

    from src.eval import (
        check_metadata_serializable_all_types,
        eval_kernel_against_ref,
        get_hidden_test_path,
        get_torch_dtype_from_string,
    )
    from src.utils import set_gpu_arch

    from .measurement import amortized, device_fingerprint

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
    os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Trusted native compiler scaffolding, before the candidate audit.
    native_changes = []
    if settings.get("strict_validation"):
        from .strict_validation import environment_guard

        context = environment_guard(native_changes)
    else:
        context = nullcontext()
    with context:
        result = eval_kernel_against_ref(
            original_model_src=reference,
            custom_model_src=payload["kernel_src"],
            seed_num=settings.get("evaluation_seed", 42),
            num_correct_trials=settings["num_correct_trials"],
            num_perf_trials=settings["num_perf_trials"],
            measure_performance=settings.get("measure_performance", not hidden),
            verbose=settings.get("verbose", False),
            build_dir=str(build),
            device=torch.device("cuda:0"),
            precision=get_torch_dtype_from_string(settings["precision"]),
            hidden_test_path=path,
            **({"hidden_seeds": settings["hidden_seeds"]} if "hidden_seeds" in settings else {}),
        )
    if result is None:
        return failure("Evaluator returned no result (possible compilation lock failure)")
    runtime = result.runtime if isinstance(result.runtime, (int, float)) and math.isfinite(result.runtime) else -1.0
    result.metadata.update(
        torch_version=torch.__version__, precision=settings["precision"], fp32_tolerance=settings["fp32_tolerance"],
        device_fingerprint=device_fingerprint(torch.device("cuda:0")),
        reference_sha256=fingerprint({"source": reference}), timing_method="synchronized_wall",
        kernel_sha256=fingerprint({"source": payload["kernel_src"]}),
    )
    cold = result.metadata.get("cold_start_ms")
    if runtime > 0 and isinstance(cold, (int, float)) and math.isfinite(cold) and cold > 0:
        result.metadata.update(cold_calls=settings.get("cold_calls", 10000),
                               amortized_runtime_ms=amortized(runtime, cold, settings.get("cold_calls", 10000)))
    from .evidence import LEVELS, accepted, check, from_result

    strict_checks = {}
    if settings.get("strict_validation") and result.compiled and result.correctness:
        from .strict_validation import evaluate_contract

        try:
            strict_checks = evaluate_contract(payload, reference, build)
        except Exception as exc:
            strict_checks = {"precision_contract": check("not_measured", reason=f"Strict verifier error: {type(exc).__name__}")}
        if native_changes:
            strict_checks["environment_integrity"] = check("failed", reason="Native evaluation changed process settings",
                                                           values={"changed_names": sorted(set(native_changes))})
        strict_ok = all(accepted(strict_checks.get(name)) for name in LEVELS["L0"] if name != "correctness")
        if not strict_ok:
            result.correctness = False
            result.metadata["strict_validation_error"] = "Required strict contract evidence failed or is unavailable"
    result.metadata["evidence_receipt"] = from_result(
        {"correctness": result.correctness}, settings, scope="hidden" if hidden else "standard", additional=strict_checks)
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


@task_input_policy
def baseline(payload: dict):
    import torch

    from src.eval import (
        _process_input_tensor,
        get_memory_stats,
        get_timing_stats,
        get_tolerance_for_precision,
        get_torch_dtype_from_string,
        load_original_model_and_inputs,
        move_model,
        set_seed,
    )
    from src.validation import clone_tensors, compare_outputs

    from .baseline_rewrites import rewrite_for
    from .baseline_variants import prepare_baseline
    from .measurement import (
        CLOCK,
        amortized,
        device_fingerprint,
        fresh_seed,
        input_signature,
        synchronize,
        synchronized_times,
    )

    reference = validate_sources(payload)
    settings = payload["eval_config"]
    os.environ["KB_FP32_TOL"] = str(settings["fp32_tolerance"])
    variant = settings.get("baseline_variant", "eager")
    tf32 = settings["precision"] == "fp32" and not settings.get("baseline_suite", False)
    configure_tf32(torch, tf32)
    device = torch.device("cuda:0")
    dtype = get_torch_dtype_from_string(settings["precision"])
    Model, init, inputs = load_original_model_and_inputs(reference, {})
    set_seed(42)
    with torch.no_grad():
        init_inputs = [_process_input_tensor(x, device, dtype) for x in init()]
        set_seed(42)
        args = [_process_input_tensor(x, device, dtype) for x in inputs()]
        set_seed(42)
        oracle = move_model(Model(*init_inputs), device, dtype)
        set_seed(42)
        expected = clone_tensors(oracle(*clone_tensors(args)))
        synchronize(device)
        configure_tf32(torch, tf32)
        set_seed(42)
        synchronize(device)
        started = CLOCK()
        model = move_model(Model(*init_inputs), device, dtype)
        backend = settings.get("compile_backend")
        mode = settings.get("compile_mode", "default")
        model, backend, mode = prepare_baseline(model, args, device, variant, backend=backend, mode=mode,
                                                rewrite=rewrite_for(payload["level"], payload["problem_id"], reference))
        synchronize(device)
        setup_ms = (CLOCK() - started) * 1000
        set_seed(42)
        fresh = clone_tensors(args)
        synchronize(device)
        started = CLOCK()
        actual = model(*fresh)
        synchronize(device)
        first_forward_ms = (CLOCK() - started) * 1000
        issue = compare_outputs(expected, actual, get_tolerance_for_precision(dtype))
        del expected, actual, fresh
        if issue:
            return failure(f"Baseline correctness failed: {issue}", baseline_variant=variant)
        timing_receipt = {}

        def input_factory(trial):
            seed = fresh_seed()
            timing_receipt.setdefault("input_seeds", []).append(seed)
            set_seed(seed)
            return [_process_input_tensor(x, device, dtype) for x in inputs()]

        def verifier(fresh, output):
            expected = clone_tensors(oracle(*fresh))
            synchronize(device)
            issue = compare_outputs(expected, output, get_tolerance_for_precision(dtype))
            if issue:
                raise RuntimeError(f"Baseline varied-input correctness failed: {issue}")

        times = synchronized_times(model, args, device, num_trials=settings["num_perf_trials"],
                                   input_factory=input_factory, verifier=verifier, receipt=timing_receipt)
        stats = get_timing_stats(times, device=device)
        memory = get_memory_stats(model, *args, device=device)
        stats.update(peak_memory=memory["peak_bytes"], memory_stats=memory)

    return json_safe(
        {
            **stats,
            "precision": settings["precision"],
            "tf32": tf32,
            "torch_version": torch.__version__,
            "timing_method": "synchronized_wall",
            "time_unit": "ms",
            "compile_backend": backend,
            "compile_mode": mode,
            "baseline_variant": variant,
            "baseline_correctness": True,
            "device_fingerprint": device_fingerprint(device),
            "input_signature": input_signature(args),
            "cold_start_ms": setup_ms + first_forward_ms + timing_receipt.get("warmup_total_ms", 0),
            "candidate_setup_ms": setup_ms,
            "first_forward_ms": first_forward_ms,
            "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}),
            "timing_receipt": timing_receipt,
            "cold_calls": settings.get("cold_calls", 10000),
            "amortized_mean_ms": amortized(stats["mean"], setup_ms + first_forward_ms
                                           + timing_receipt.get("warmup_total_ms", 0), settings.get("cold_calls", 10000)),
        }
    )


def qualify_small(payload):
    """Reviewed auxiliary correctness before allocating the original performance workload."""
    import torch

    from src.utils import set_gpu_arch

    from .evidence import accepted, from_result
    from .measurement import device_fingerprint
    from .strict_validation import evaluate_contract

    source = validate_sources(payload)
    configure_tf32(torch, False)
    settings = payload["eval_config"]
    if settings.get("gpu_arch"):
        set_gpu_arch(settings["gpu_arch"].split(","))
    build = ROOT / "cache/workflow" / fingerprint(payload)
    build.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    checks = evaluate_contract(payload, source, build)
    result = {"compiled": True, "correctness": all(accepted(item) for item in checks.values()), "runtime": -1,
              "metadata": {"reference_sha256": fingerprint({"source": source}),
                           "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                           "device_fingerprint": device_fingerprint(torch.device("cuda:0")),
                           "qualification_scope": "reviewed_auxiliary_cases; no_original_input_allocation_or_timing"}}
    result["metadata"]["evidence_receipt"] = from_result(result, settings, additional=checks)
    return result


def dispatch(payload):
    if payload.get("task") == "origin_audit":
        from .origin_task import audit

        return audit(payload)
    if payload.get("eval_config", {}).get("device_backend", "cuda") == "neuron":
        from .neuron_worker import evaluate as evaluate_neuron

        return evaluate_neuron(payload)
    if payload.get("task") == "qualify_small":
        return qualify_small(payload)
    if payload.get("task") == "paired_timing":
        from .paired_task import evaluate_paired

        return evaluate_paired(payload)
    if payload.get("task") == "profile":
        from .profile_task import profile

        return profile(payload)
    if payload.get("task", "evaluate") in {"evaluate", "baseline"}:
        return baseline(payload) if payload.get("task") == "baseline" else evaluate(payload)
    raise ValueError("Unknown evaluation task")


def diagnostic_control(payload):
    """Collect requested functional probes in the separate, untimed control run."""
    settings = payload["eval_config"]
    with ExitStack() as stack:
        repeat, state, numerical = None, None, None
        if settings.get("determinism_checks"):
            from .determinism import audit, validate_inline

            validate_inline(settings["determinism_policy"], payload["ref_arch_src"])
            repeat = stack.enter_context(audit(settings["determinism_policy"], settings["determinism_repeats"]))
        if settings.get("state_validation"):
            from .state_audit import audit

            state = stack.enter_context(audit())
        if settings.get("numerical_validation"):
            from .numerical_audit import audit

            numerical = stack.enter_context(audit(payload))
        result = dispatch(payload)
    metadata = result.setdefault("metadata", {})
    if repeat is not None:
        metadata["repeat_audit"] = repeat.document()
    if state is not None:
        metadata["state_audit"] = state.document()
    if numerical is not None:
        metadata["numerical_audit"] = numerical.document()
    return result


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        from .tool_checks import enabled

        native_worker = sys.argv[3:] == ["--native-worker"]
        settings = payload["eval_config"]
        if native_worker and settings.get("diagnostic_stage") == "control" and (
                settings.get("determinism_checks") or settings.get("state_validation") or settings.get("numerical_validation")):
            result = diagnostic_control(payload)
        elif enabled(payload) and not native_worker:
            from .tool_runner import evaluate as evaluate_with_tools

            result = evaluate_with_tools(payload)
        else:
            result = dispatch(payload)
    except Exception as exc:
        result = failure(str(exc), error_name=type(exc).__name__)
    atomic_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
