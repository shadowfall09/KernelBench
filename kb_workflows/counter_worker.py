"""One public forward inside cudaProfilerStart/Stop, only on explicit execution."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, evaluator_digest, fingerprint, json_safe, read_json
from .input_calibration import task_policy as nominal_policy
from .production import task_input_policy


@contextmanager
def build_environment(directory):
    previous = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = previous


@nominal_policy
@task_input_policy
def run_call(payload, *, adapter=None, loader=None, profiler=None):
    import torch

    from src.eval import (
        _process_input_group,
        get_torch_dtype_from_string,
        load_custom_model,
        load_original_model_and_inputs,
        move_model,
        set_seed,
    )
    from src.input_groups import copy_inputs
    from src.validation import clone_tensors, compare_outputs

    from .calibration import identity
    from .determinism import preserve_rng
    from .devices import adapter as get_adapter
    from .evaluate_task import validate_sources
    from .input_calibration_task import precision_environment, probe
    from .measurement import input_signature
    from .work_counters import binding_valid

    settings = payload["eval_config"]
    if settings.get("use_hidden_tests") or settings.get("device_backend", "cuda") != "cuda":
        raise ValueError("Nsight counters accept one public CUDA call, never hidden or Neuron execution")
    target, seed = settings["counter_target"], settings["counter_seed"]
    if target not in {"reference", "candidate"} or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid counter target or public input seed")
    simulated = adapter is not None or loader is not None or profiler is not None
    adapter = adapter or get_adapter("cuda", "0")
    device, caps = adapter.device, adapter.caps()
    identity(caps)
    profiler = profiler or torch.cuda.profiler
    start, stop = profiler.start, profiler.stop  # Capture before submission import.
    reference = validate_sources(payload)
    dtype = get_torch_dtype_from_string(settings["precision"])
    tolerance = settings["fp32_tolerance"] if settings["precision"] == "fp32" else 1e-2
    Model, init, inputs = (loader or load_original_model_and_inputs)(reference, {})
    changes, numeric = [], None
    policy = {"native_tolerance": tolerance}
    pid = os.getpid()
    with (tempfile.TemporaryDirectory(prefix="kb-counter-call-") as build, build_environment(build),
          torch.no_grad(), preserve_rng(device), precision_environment(changes)):
        set_seed(42)
        initial = _process_input_group(init(), device, dtype)
        set_seed(42)
        oracle = move_model(Model(*copy_inputs(initial)), device, dtype)
        reference_training = bool(oracle.training)
        set_seed(seed)
        args = _process_input_group(inputs(), device, dtype)
        signature, input_probe = input_signature(args), fingerprint({"probe": probe(args)})
        state_probe = fingerprint({"probe": probe(dict(oracle.state_dict()))})
        truth = prepared = None
        if settings.get("numerical_validation"):
            from .numerical_audit import _spec
            from .reference_oracles import prepare

            spec, policy = _spec(payload)
            prepared = prepare(oracle, args, spec=spec, problem_id=payload["problem_id"])
            if prepared.estimated_host_bytes > settings["oracle_memory_mb"] * 1024**2:
                raise ValueError("Counter truth exceeds original-size oracle memory envelope; no resizing")
            set_seed(42)
            truth = prepared.execute()
        set_seed(42)
        expected = clone_tensors(oracle(*copy_inputs(args)))
        from .measurement import synchronize

        synchronize(device)
        ModelNew = Model if target == "reference" else load_custom_model(payload["kernel_src"], {}, build)
        # Warm a separate initial instance; the profiled instance starts at seed 42.
        set_seed(42)
        warm = move_model(ModelNew(*copy_inputs(initial)), device, dtype)
        warm(*copy_inputs(args))
        synchronize(device)
        del warm
        set_seed(42)
        model = move_model(ModelNew(*copy_inputs(initial)), device, dtype)
        actual_args = copy_inputs(args)
        synchronize(device)
        set_seed(42)
        before_mode = model.training
        if before_mode != reference_training or torch.is_grad_enabled():
            raise ValueError("Counter instance changed the frozen reference training/gradient mode")
        start()
        try:
            output = model(*actual_args)
            synchronize(device)
        finally:
            stop()
        if model.training != before_mode or torch.is_grad_enabled():
            changes.append("candidate_mode_or_grad_changed")
        issue = compare_outputs(expected, output, tolerance)
        if settings.get("numerical_validation"):
            from .numerical import PrecisionContract, precision_gate

            def host(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, (tuple, list)):
                    return type(value)(host(item) for item in value)
                if isinstance(value, dict):
                    return {key: host(item) for key, item in value.items()}
                return value

            numeric = precision_gate(host(expected), host(output), truth, PrecisionContract(**policy))
            if not numeric["passed"] or not numeric["reference_qualified"]:
                issue = "Independent counter-call FP64 qualification failed"
        if issue:
            raise ValueError("Counter-call output did not pass its frozen native/numerical policy: " + str(issue))
    binding = {"source_sha256": hashlib.sha256((reference if target == "reference" else payload["kernel_src"]).encode()).hexdigest(),
               "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(), "evaluator_sha256": evaluator_digest(),
               "evaluation_contract_sha256": fingerprint({"settings": settings}), "precision": settings["precision"],
               "input_signature_sha256": fingerprint({"signature": signature}), "input_probe_sha256": input_probe,
               "state_probe_sha256": state_probe, "policy_sha256": fingerprint(policy),
               "device": caps, "device_identity_sha256": identity(caps), "target": target, "seed": seed, "pid": pid,
               "initialization_seed": 42, "forward_seed": 42,
               "training": reference_training, "no_grad": True,
               "output_verified": True, "environment_changes": sorted(set(changes))}
    binding_valid(binding)
    return json_safe({"status": "ok", "binding": binding, "simulated": simulated or bool(getattr(adapter, "simulated", False)),
                      "numerical_gate": numeric, "oracle_receipt": prepared.receipt() if prepared else None,
                      "input_signature": signature, "probe_scope": "up_to_256_positions_per_tensor; not_full_content_or_state_hash"})


def main():
    payload = read_json(Path(sys.argv[1]))
    try:
        result = run_call(payload)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    atomic_json(Path(sys.argv[2]), result)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
