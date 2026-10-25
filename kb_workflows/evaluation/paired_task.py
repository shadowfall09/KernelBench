"""Balanced candidate/baseline/control calls in one fresh verification process."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext

from src.input_groups import copy_inputs
from src.validation import clone_tensors, compare_outputs, preserve_backend_settings

from ..analysis.paired_statistics import paired_summary
from ..production import task_input_policy
from .measurement import CLOCK, consume_output, synchronize
from .strict_validation import environment_guard


def call_plan(*, trials, warmup, seed):
    rng = random.Random(seed)
    choices = ["RCB", "CRB", "BCR", "BRC", "CBR", "RBC"]
    orders = []
    for trial in range(warmup + trials):
        if trial == warmup:
            orders = []
        if not orders:
            orders = list(choices)
            rng.shuffle(orders)
        yield trial, rng.randrange(1 << 31), orders.pop()


def collect_pairs(reference, candidate, control, input_factory, oracle, *, device, trials, seed,
                  reset=None, set_seed=None, tolerance=1e-3, warmup=3, clock=CLOCK, verifier=None, prepare_verifier=None):
    import torch

    if type(trials) is not int or trials < 10:
        raise ValueError("At least 10 paired trials are required")
    from ..provenance.origin_task import candidate_call, confirm, snapshot_output

    models = {"R": reference, "C": candidate, "B": control}
    rows, changes = [], []
    with torch.no_grad():
        for trial, case_seed, order in call_plan(trials=trials, warmup=warmup, seed=seed):
            if set_seed:
                set_seed(case_seed)
            args = input_factory(trial)
            if reset:
                reset("oracle")
            if set_seed:
                set_seed(case_seed)
            prepared = prepare_verifier(args) if prepare_verifier is not None else None
            expected = clone_tensors(oracle(*copy_inputs(args)))
            synchronize(device)
            samples = {}
            for name in order:
                fresh = copy_inputs(args)
                if reset:
                    reset(name)
                if set_seed:
                    set_seed(case_seed)
                context = environment_guard(changes) if name == "C" else preserve_backend_settings()
                with context:
                    synchronize(device)
                    scope = candidate_call(candidate, fresh, device=device, phase="paired_warmup" if trial < warmup else "paired",
                                           seed=case_seed, label=str(trial), order=order) if name == "C" else nullcontext((models[name], None))
                    with scope as (execute, observed):
                        started = clock()
                        output = execute(*fresh)
                        consumed = consume_output(output)
                        synchronize(device)
                        elapsed = (clock() - started) * 1000
                snapshot_output(observed, output)
                issue = compare_outputs(expected, output, tolerance)
                if issue:
                    raise ValueError(f"Paired trial {trial}/{name}: {issue}")
                if verifier:
                    from .determinism import preserve_rng

                    with preserve_rng(device):
                        if set_seed:
                            set_seed(case_seed)
                        if prepare_verifier is None:
                            verifier(args, output)
                        else:
                            verifier(args, output, expected=expected, prepared=prepared)
                confirm(observed)
                samples[name] = elapsed
                del output, consumed, fresh
            if trial >= warmup:
                rows.append({"reference_ms": samples["R"], "candidate_ms": samples["C"],
                             "control_ms": samples["B"], "seed": case_seed, "order": order})
            del args, expected, prepared
    if changes:
        raise ValueError(f"Candidate changed process settings during paired timing: {sorted(set(changes))}")
    return paired_summary(rows)


@task_input_policy
def evaluate_paired(payload):
    import torch

    from src.eval import (
        _process_input_group,
        _process_input_tensor,
        get_tolerance_for_precision,
        get_torch_dtype_from_string,
        load_custom_model,
        load_original_model_and_inputs,
        move_model,
        set_seed,
    )
    from src.validation import clone_tensors

    from ..common import ROOT, fingerprint
    from .baseline_rewrites import plan_for, rewrite_for
    from .baseline_variants import prepare_baseline
    from .evaluate_task import configure_tf32, validate_sources
    from .evidence import check
    from .measurement import device_fingerprint, input_signature
    from .strict_validation import environment_guard

    settings = payload["eval_config"]
    if settings.get("use_hidden_tests"):
        raise ValueError("Paired timing uses the public performance workload, never hidden feedback")
    source = validate_sources(payload)
    configure_tf32(torch, settings["precision"] == "fp32" and not settings.get("baseline_suite", True) and not settings.get("input_profile") and not settings.get("numerical_validation"))
    dtype, device = get_torch_dtype_from_string(settings["precision"]), torch.device("cuda:0")
    seed = settings.get("evaluation_seed", 42)
    variant = settings.get("paired_baseline_variant", "eager")
    Model, init, inputs = load_original_model_and_inputs(source, {})
    set_seed(seed)
    initial = [_process_input_tensor(value, device, dtype) for value in init()]
    models = {}
    for name in ("R", "B", "oracle"):
        set_seed(seed)
        models[name] = move_model(Model(*clone_tensors(initial)), device, dtype)
    from pathlib import Path

    build = Path(payload.get("build_root", ROOT / "cache/workflow")) / fingerprint(payload)
    build.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    if settings.get("gpu_arch"):
        from src.utils import set_gpu_arch

        set_gpu_arch(settings["gpu_arch"].split(","))
    changes = []
    with environment_guard(changes):
        Candidate = load_custom_model(payload["kernel_src"], {}, str(build))
        set_seed(seed)
        models["C"] = move_model(Candidate(*clone_tensors(initial)), device, dtype)
    # Compilation scaffolding assigns this trusted setting before candidate import.
    if changes:
        raise ValueError(f"Candidate import changed process settings: {sorted(set(changes))}")
    snapshots = {name: clone_tensors(model.state_dict()) for name, model in models.items()}
    set_seed(seed)
    nominal = _process_input_group(inputs(), device, dtype)
    rewrite = rewrite_for(payload["level"], payload["problem_id"], source)
    baseline_plan = plan_for(payload["level"], payload["problem_id"], source, variant)
    functions = {}
    for name in ("R", "B"):
        functions[name], _, _ = prepare_baseline(models[name], nominal, device, variant,
                                                backend=settings.get("compile_backend", ""),
                                                mode=settings.get("compile_mode", "default"), rewrite=rewrite, plan=baseline_plan)

    def factory(index):
        return _process_input_group(inputs(), device, dtype)

    def reset(name):
        models[name].load_state_dict(snapshots[name])

    def prepare_verify(args):
        from ..numerics.input_calibration import prepare_output

        return prepare_output(models["oracle"], args)

    def verify(args, output, *, expected, prepared):
        from ..numerics.input_calibration import verify_output

        verify_output(models["oracle"], args, expected, output, prepared=prepared)

    result = collect_pairs(functions["R"], models["C"], functions["B"], factory, models["oracle"],
                           device=device, trials=settings.get("paired_trials", 30), seed=seed,
                           reset=reset, set_seed=set_seed, tolerance=get_tolerance_for_precision(dtype),
                           verifier=verify if settings.get("input_profile") or settings.get("numerical_validation") else None,
                           prepare_verifier=prepare_verify if settings.get("input_profile") or settings.get("numerical_validation") else None)
    result.update(baseline_variant=variant, baseline_plan=baseline_plan, input_signature=input_signature(nominal), output_consumed=True,
                  varied_inputs=True, all_outputs_verified=True)
    return {"compiled": True, "correctness": True, "runtime": result["candidate_mean_ms"],
            "runtime_stats": {}, "peak_memory": -1, "memory_stats": {},
            "metadata": {"paired_timing": result,
                         "paired_checks": {"paired_timing": check("passed", reason="Balanced synchronized calls with verified outputs"),
                                           "noise_floor": check("passed", reason="Reference/control pairs collected", values=result["reference_control_ci95"])},
                         "device_fingerprint": device_fingerprint(device),
                         "reference_sha256": fingerprint({"source": source}),
                         "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                         "input_signature": result["input_signature"], "precision": settings["precision"]}}


def validate_for_score(config, pid, sid, variant, baseline):
    from ..analysis.paper_analysis import current_result, same_device_class
    from ..common import evaluator_digest, read_json
    from ..engine import paired_payload
    from ..runtime.isolation import runtime_issue

    results = read_json(config.run_dir / "paired_results.json", {})
    pairs = []
    for repeat in range(config.final_repeats or 1):
        result = results.get(f"{pid}_{sid}:{repeat}", {})
        if result.get("compiled") is not True or result.get("correctness") is not True:
            return "paired_measurement_failed_or_missing", None
        try:
            expected = paired_payload(config, pid, sid, repeat=repeat if config.final_repeats else None, variant=variant)
        except (ValueError, KeyError, OSError, IndexError):
            return "paired_source_or_manifest_changed", None
        if current_result(result, expected) or expected["eval_config"]["evaluator_sha256"] != evaluator_digest():
            return "paired_measurement_source_mismatch", None
        if runtime_issue(baseline, result, expected["eval_config"]):
            return "paired_measurement_runtime_mismatch", None
        metadata = result.get("metadata", {})
        record = metadata.get("paired_timing", {})
        if config.device_backend == "cuda":
            from .baseline_rewrites import plan_for

            try:
                plan = plan_for(config.level, pid, expected["ref_arch_src"], variant)
            except (ValueError, KeyError, TypeError):
                return "paired_baseline_recipe_changed", None
            if record.get("baseline_plan") != plan:
                return "paired_baseline_recipe_missing_or_changed", None
        if (not same_device_class(baseline.get("device_fingerprint", {}), metadata.get("device_fingerprint", {}))
                or record.get("baseline_variant") != variant or record.get("num_pairs") != config.paired_trials
                or metadata.get("input_signature") != baseline.get("input_signature")
                or not all(record.get(field) is True for field in ("output_consumed", "varied_inputs", "all_outputs_verified"))):
            return "paired_measurement_protocol_mismatch", None
        try:
            recomputed = paired_summary(record["pairs"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return "paired_measurement_samples_invalid", None
        if any(recomputed.get(field) != record.get(field) for field in
               ("protocol", "speedup_ci95", "reference_control_ci95", "reference_mean_ms", "candidate_mean_ms", "significantly_faster")):
            return "paired_measurement_summary_mismatch", None
        pairs.extend({**row, "repeat": repeat} for row in record["pairs"])
    return None, paired_summary(pairs)
