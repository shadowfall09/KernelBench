"""Separate, public-input source/binary origin diagnostics. Never scored timing."""

from __future__ import annotations

import os

from .common import ROOT, evaluator_digest, fingerprint, json_safe
from .kernel_origin import summarize
from .production import task_input_policy


@task_input_policy
def audit(payload):
    settings = payload["eval_config"]
    if settings.get("use_hidden_tests"):
        raise ValueError("Origin diagnostics cannot expose hidden inputs")
    if settings.get("device_backend", "cuda") != "cuda":
        raise ValueError("This origin collector supports CUDA; Neuron binary association is pending")
    seeds = settings.get("origin_seeds", [42, 43, 44])
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 100 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Origin diagnostic seeds must be 1..100 distinct uint32 values")
    threshold = settings.get("origin_minimum_custom_fraction", 0.95)
    # Validate the policy before importing optional SDKs or allocating inputs.
    import math

    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Origin diagnostic fraction must be in (0, 1]")
    from .evaluate_task import validate_sources

    reference = validate_sources(payload)
    from .cuda_origin import Capture
    from .origin_binaries import Catalog

    capture = Capture(payload["kernel_src"])
    catalog = Catalog()  # Freeze wheel identities before candidate imports.
    with capture.recording():
        import torch

        from src.eval import (
            _process_input_tensor,
            get_tolerance_for_precision,
            get_torch_dtype_from_string,
            load_custom_model,
            load_original_model_and_inputs,
            move_model,
            set_seed,
        )
        from src.utils import set_gpu_arch
        from src.validation import clone_tensors, compare_outputs

        from .evaluate_task import configure_tf32
        from .measurement import consume_output, device_fingerprint, input_signature, synchronize
        from .strict_validation import environment_guard

        device = torch.device("cuda:0")
        dtype = get_torch_dtype_from_string(settings["precision"])
        configure_tf32(torch, False)
        if settings.get("gpu_arch"):
            set_gpu_arch(settings["gpu_arch"].split(","))
        build = ROOT / "cache/workflow/origin" / fingerprint(payload)
        build.mkdir(parents=True, exist_ok=True)
        os.environ.update(KB_FP32_TOL=str(settings["fp32_tolerance"]), TORCH_EXTENSIONS_DIR=str(build), TORCH_USE_CUDA_DSA="1")
        tolerance = get_tolerance_for_precision(dtype)  # Freeze before candidate code.
        Model, init, inputs = load_original_model_and_inputs(reference, {})
        set_seed(seeds[0])
        initial = [_process_input_tensor(value, device, dtype) for value in init()]
        set_seed(seeds[0])
        oracle = move_model(Model(*clone_tensors(initial)), device, dtype)
        set_seed(seeds[0])
        changes = []
        with environment_guard(changes):
            ModelNew = load_custom_model(payload["kernel_src"], {}, str(build))
            model = move_model(ModelNew(*clone_tensors(initial)), device, dtype)
        capture.raw["binding"] = {"request_sha256": fingerprint(payload), "evaluator_sha256": evaluator_digest(),
            "reference_sha256": fingerprint({"source": reference}), "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
            "device_fingerprint": device_fingerprint(device), "precision": settings["precision"],
            "fp32_tolerance": settings["fp32_tolerance"], "torch_version": torch.__version__,
            "compiler": "triton-3.7.0", "collector": "cupti-python-13.3.0", "input_seeds": seeds,
            "scope": "public_input_constructor; separate_instrumented_process; no_hidden_or_native_case_coverage_claim"}
        with torch.no_grad():
            for seed in seeds:
                set_seed(seed)
                args = [_process_input_tensor(value, device, dtype) for value in inputs()]
                expected_args, actual_args = clone_tensors(args), clone_tensors(args)
                signature = input_signature(actual_args)
                set_seed(seed)
                expected = clone_tensors(oracle(*expected_args))
                synchronize(device)
                set_seed(seed)
                with environment_guard(changes):
                    with capture.call(seed=seed, signature=signature, training=bool(model.training)) as window:
                        before_grad = torch.is_grad_enabled()
                        output = model(*actual_args)
                        synchronize(device)
                        window["no_grad"] = not before_grad and not torch.is_grad_enabled()
                        window["training_after"] = bool(model.training)
                if torch.is_grad_enabled():
                    changes.append("grad_enabled")
                    torch.set_grad_enabled(False)
                consume_output(output)
                synchronize(device)
                issue = compare_outputs(expected, output, tolerance)
                window["output_verified"] = issue is None
                if issue:
                    capture.error("Profiled output failed native structural/numerical comparison: " + str(issue))
                del expected, output, args, expected_args, actual_args
                synchronize(device)
        capture.raw["environment_changes"] = sorted(set(changes))
        if changes:
            capture.error("Candidate import/execution changed process settings")
    capture.raw["binary_inventory"] = catalog.collect(timeout=min(60, max(1, settings.get("timeout", 300) / 4)))
    try:
        summary = summarize(capture.raw, minimum_custom_fraction=threshold)
    except ValueError as exc:
        summary = {"status": "not_measured", "reason": str(exc), "timings_are_diagnostic": True,
                   "formal_qualification_implemented": False}
    return json_safe({"status": "ok", "purpose": "execution_origin_diagnostics_only", "raw": capture.raw,
                      "summary": summary, "scored_wall_times_replaced": False})
