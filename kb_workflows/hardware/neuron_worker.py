"""Native Neuron task runner; independent FP64 reference arithmetic stays on CPU."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
import uuid
from contextlib import contextmanager, nullcontext

from ..common import ROOT, fingerprint
from .devices import adapter as device_adapter


def map_tensors(value, function):
    import torch

    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(item, function) for item in value)
    return value


def load_source(source, name, *, filename=None):
    if filename is not None:
        from ..provenance.submission_source import compile_source

        code = compile_source(source, filename)
        module_name = name + "_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        specification = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        exec(code, vars(module))
        return vars(module)
    namespace = {"__name__": name}
    exec(compile(source, f"<{name}>", "exec"), namespace)
    return namespace


def library_supported(level, pid):
    if level != 5 or pid not in {1, 3, 27, 30}:
        return False
    from ..benchmarks.suites import suite_rows
    from ..common import select_problems

    path = select_problems(5, problem_ids=[pid])[pid]
    reviewed = suite_rows(5, "scored", suite="production41")[pid]
    if hashlib.sha256(path.read_bytes()).hexdigest() != reviewed["reference_sha256"]:
        raise ValueError("Neuron library rewrite reference changed; review equivalence before use")
    return True


def prepare_variant(model, variant, *, level=None, pid=None):
    import torch

    if variant == "neuron_eager":
        return model
    if variant == "neuron_compile":
        return torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)
    if variant == "neuron_library" and library_supported(level, pid):
        import torch.nn.functional as F

        class Library(torch.nn.Module):
            def forward(self, *args):
                if pid == 1:
                    q, k, v, mask = args
                    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask).to(q.dtype)
                if pid == 3:
                    x, weight, bias, eps = args
                    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)
                if pid == 27:
                    x, residual, weight, eps = args
                    return F.rms_norm(x.float() + residual.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)
                q, k, v = args
                return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True).to(q.dtype)

        return torch.compile(Library(), backend="neuron", fullgraph=True, dynamic=False)
    raise ValueError("Neuron baseline variant must be neuron_eager or neuron_compile")


@contextmanager
def profile_source_cache(payload):
    if payload.get("task") != "profile":
        yield
        return
    directory = ROOT / "cache/workflow" / fingerprint(payload) / "neff_cache"
    directory.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")
    os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_NEURONX_NEFF_CACHE_DIR", None)
        else:
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = previous


def evaluate(payload, *, adapter=None):
    import torch

    from ..provenance.neuron_origin import CompilationObserver
    from ..provenance.submission_source import SourceRegistry, record_sources

    registry = SourceRegistry()
    runtime = None
    if payload.get("task") == "profile" and payload["eval_config"].get("neuron_runtime_origin"):
        from ..provenance.neuron_runtime import Observer

        runtime = Observer(payload)
    observer = CompilationObserver(registry) if payload.get("task") == "profile" else nullcontext()
    with torch.no_grad(), profile_source_cache(payload), record_sources(registry), observer:
        result = _evaluate(payload, adapter=adapter, registry=registry)
    if payload.get("task") == "profile":
        from .neuron_processing import DEFAULT_LIMITS, file_manifest

        result["compiler_observations"] = observer.value
        if directory := result.get("vendor_artifact_directory"):
            observer.export(directory)
            if runtime is not None:
                result["native_runtime"] = runtime.export(directory)
            files = file_manifest(directory, DEFAULT_LIMITS)
            result["artifact_files"] = {name: row["sha256"] for name, row in files.items()}
            result["artifact_file_bytes"] = {name: row["bytes"] for name, row in files.items()}
    build = ROOT / "cache/workflow" / fingerprint(payload)
    hashes = {}
    for key, name in (("ref_arch_src", "reference.py"), ("kernel_src", "candidate.py")):
        if key not in payload:
            continue
        path = build / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload[key].encode():
            raise ValueError("NKI compiler source files changed during execution")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if payload["eval_config"].get("use_hidden_tests"):
        path = ROOT / "hidden_tests" / f"level{payload['level']}" / f"{payload['problem_id']}_hidden.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != payload["eval_config"]["hidden_source_sha256"]:
            raise ValueError("Hidden sources changed during Neuron execution")
    result.setdefault("metadata", {})["compiler_source_sha256"] = hashes
    if not registry.intact():
        raise ValueError("Registered Neuron compiler sources changed during execution")
    result["metadata"]["compiler_source_registration"] = "frozen_compile_code_objects"
    if payload.get("task") == "profile":
        from .neuron_work import capture_record

        result["capture_receipt"] = capture_record(payload, result)
    return result


def _evaluate(payload, *, adapter=None, registry=None):
    """Run native stages. Injecting a CPU adapter produces explicitly simulated receipts."""
    import torch

    from src.reliability import HIDDEN_SEEDS, check_paired_calls, stress_cases, stress_enabled
    from src.validation import clone_tensors, compare_outputs, output_comparison_policy

    from ..evaluation.evaluate_task import validate_sources
    from ..evaluation.evidence import LEVELS, accepted, check, from_result
    from ..evaluation.measurement import CLOCK, amortized, fresh_seed, input_signature, synchronized_times, timing_stats
    from ..evaluation.strict_validation import environment_guard, run_checks
    from ..numerics.contract_specs import contract_for
    from ..numerics.numerical import PrecisionContract, precision_gate, signal_audit, to_oracle
    from ..production import instance_plan, production_context

    source = validate_sources(payload)
    settings = payload["eval_config"]
    phase, hidden = payload.get("task", "evaluate"), settings.get("use_hidden_tests", False)
    if settings.get("device_backend") != "neuron" or phase not in {"evaluate", "baseline", "paired_timing", "qualify_small", "profile"}:
        raise ValueError("Invalid Neuron task or backend")
    if phase == "profile" and hidden:
        raise ValueError("Development profiling cannot use hidden tests")
    injected = adapter is not None
    adapter = adapter or device_adapter("neuron", "0")
    caps, device = adapter.caps(), adapter.device
    simulated = injected or bool(getattr(adapter, "simulated", False))
    if caps.get("device_backend") != "neuron" and not simulated:
        raise ValueError("A CUDA or CPU adapter cannot establish Neuron execution")
    if not simulated and caps.get("name", "").lower().replace(" ", "") != {"trn2": "trainium2", "trn3": "trainium3"}.get(settings.get("gpu_arch")):
        raise ValueError("Observed Neuron hardware differs from the requested Trainium target")
    build = ROOT / "cache/workflow" / fingerprint(payload)
    reference = load_source(source, "kernelbench_neuron_reference", filename=build / "reference.py")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[settings["precision"]]
    instance, context, contract = None, nullcontext(), None
    if payload["level"] == 5:
        from ..benchmarks.production_specs import task
        from ..production import PROTOCOL

        options = settings["production"]
        expected = {"protocol": PROTOCOL, "task_sha256": fingerprint(task(payload["problem_id"])),
                    "profile": options["profile"], "memory_bytes": options["memory_bytes"]}
        if "dimensions_override" in options:
            expected["dimensions_override"] = options["dimensions_override"]
        if expected != options:
            raise ValueError("Neuron production input contract changed")
        instance = instance_plan(payload["problem_id"], precision=settings["precision"], profile=options["profile"],
            memory_bytes=options["memory_bytes"], total_memory_bytes=caps["total_memory"],
            dimensions_override=options.get("dimensions_override"))
        if any(row["dtype"] == "fp8_e4m3fn" for row in instance["input_roles"]):
            raise ValueError("No reviewed Neuron float8 storage bridge; dtype downgrades are forbidden")
        context = production_context(precision=settings["precision"], profile=options["profile"],
            memory_bytes=options["memory_bytes"], total_memory_bytes=caps["total_memory"],
            **({"dimensions_override": options["dimensions_override"]} if "dimensions_override" in options else {}))
        contract = PrecisionContract(**instance["output_comparison_contract"])

    def seed(value):
        torch.manual_seed(value)
        if str(device).split(":")[0] == "neuron":
            torch.neuron.manual_seed(value)

    def host(value):
        from src.input_groups import copy_inputs

        return copy_inputs(value, lambda tensor: tensor.detach().cpu())

    def move(value):
        def transfer(tensor):
            target_dtype = dtype if instance is None and tensor.is_floating_point() else tensor.dtype
            copied = tensor.detach().to(device=device, dtype=target_dtype)
            if not tensor.is_contiguous() and all(stride > 0 for stride in tensor.stride()):
                output = torch.empty_strided(tensor.shape, tensor.stride(), dtype=target_dtype, device=device)
                return output.copy_(copied)
            return copied.clone()

        from src.input_groups import copy_inputs

        return copy_inputs(value, transfer, isolate=False)

    from ..numerics.input_profiles import wrap_factory

    nominal_factory = wrap_factory(source, reference["get_inputs"])

    def inputs():
        values = nominal_factory()
        if instance is None:
            values = map_tensors(values, lambda tensor: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor)
        return values

    seed(settings.get("evaluation_seed", 42))
    with context:
        initial = reference["get_init_inputs"]()
        cpu = reference["Model"](*clone_tensors(initial))
        if instance is None:
            cpu = cpu.to(dtype=dtype)
        ref = copy.deepcopy(cpu).to(device=device)
        changes = []
        adapter.synchronize()
        started = CLOCK()
        if phase == "baseline":
            variant = settings.get("baseline_variant", "neuron_eager")
            seed(settings.get("evaluation_seed", 42))
            baseline_cpu = reference["Model"](*clone_tensors(initial))
            if instance is None:
                baseline_cpu = baseline_cpu.to(dtype=dtype)
            model = prepare_variant(baseline_cpu.to(device=device), variant, level=payload["level"], pid=payload["problem_id"])
        else:
            from ..evaluation.cache_audit import candidate_guard as cache_guard
            from ..evaluation.throughput_audit import candidate_guard

            with environment_guard(changes), candidate_guard(), cache_guard():
                candidate = load_source(payload["kernel_src"], "kernelbench_neuron_candidate", filename=build / "candidate.py")
                seed(settings.get("evaluation_seed", 42))
                model = candidate["ModelNew"](*clone_tensors(initial)).to(device=device, **({"dtype": dtype} if instance is None else {}))
            variant = None
        adapter.synchronize()
        setup_ms = (CLOCK() - started) * 1000
        args = inputs() if phase != "qualify_small" and not hidden else []
        metadata = {"device_backend": "neuron", "device_fingerprint": caps, "precision": settings["precision"],
                    "reference_sha256": fingerprint({"source": source}), "timing_method": "synchronized_wall",
                    "runtime_kind": "native_torch_neuron_NKI", "cpu_fallback_allowed": False,
                    "simulated_execution": simulated, "security_isolation_certified": False, "candidate_setup_ms": setup_ms}
        if not simulated:
            from .neuron_device import core_ids

            metadata["neuron_core_allocation"] = core_ids(os.environ.get("NEURON_RT_VISIBLE_CORES", ""))
        if instance:
            metadata["production_instance"] = instance
        if phase != "baseline":
            metadata["kernel_sha256"] = fingerprint({"source": payload["kernel_src"]})
        metadata["input_signature"] = input_signature(args)
        checked, first_ms, strict = [], None, {}
        oracle_cpu = copy.deepcopy(cpu).to(device="cpu", dtype=torch.float64) if contract else None

        def verify(cpu_args, actual, expected=None, prepared=None):
            from ..numerics.input_calibration import prepare_output, verify_output

            def on_device(tensor):
                if tensor.device != torch.device(device):
                    raise ValueError("Submission output is not materialized on its declared Neuron device")
                return tensor

            map_tensors(actual, on_device)
            if expected is None:
                prepared = prepare_output(ref, cpu_args)
            dev_args = cpu_args
            cpu_args = host(cpu_args)
            if expected is None:
                expected = host(copy.deepcopy(cpu)(*clone_tensors(cpu_args))) if contract else host(ref(*move(cpu_args)))
            else:
                expected = host(expected)
            actual = host(actual)
            if prepared is not None:
                verify_output(ref, dev_args, expected, actual, prepared=prepared)
            if issue := compare_outputs(expected, actual, settings["fp32_tolerance"]):
                raise ValueError(f"Neuron output mismatch: {issue}")
            if contract:
                truth = copy.deepcopy(oracle_cpu)(*to_oracle(cpu_args))
                observed = precision_gate(expected, actual, truth, contract)
                if not observed["passed"]:
                    raise ValueError(f"Neuron CPU-FP64 contract failed: {observed['issues']}")
            return expected

        def comparator(a, b, path):
            observed = precision_gate(host(a), host(b), host(a).double(), contract)
            return None if observed["passed"] else f"{path}: production numerical budget violated"

        comparison = output_comparison_policy(comparator) if contract else nullcontext()
        with comparison:
            if phase != "baseline" and (settings.get("strict_validation") or phase == "qualify_small"):
                reviewed = contract_for(payload["level"], payload["problem_id"], source)
                if reviewed:
                    strict = run_checks(ref, model, reviewed, hidden=hidden, seed=settings.get("evaluation_seed", 42),
                        device=device, dtype=dtype, set_seed=seed, oracle_on_cpu=True)
                else:
                    strict = {name: check("not_measured", reason="No reviewed auxiliary contract for this Neuron task") for name in LEVELS["L0"]}
            if phase == "qualify_small":
                passed = bool(strict) and all(accepted(row) for row in strict.values()) and not changes
                metadata["evidence_receipt"] = from_result({"correctness": passed}, settings, additional=strict)
                return {"compiled": True, "correctness": passed, "runtime": -1, "metadata": metadata}
            hidden_path = ROOT / "hidden_tests" / f"level{payload['level']}" / f"{payload['problem_id']}_hidden.py"
            def cases():
                if not hidden:
                    for index in range(settings["num_correct_trials"]):
                        value = settings.get("evaluation_seed", 42) + index
                        seed(value)
                        yield f"standard/{index}", inputs(), value
                    return
                if not hidden_path.is_file() or hashlib.sha256(hidden_path.read_bytes()).hexdigest() != settings.get("hidden_source_sha256"):
                    raise ValueError("Neuron hidden test sources differ")
                hidden_module = load_source(hidden_path.read_text(), "kernelbench_neuron_hidden")
                seeds = settings.get("hidden_seeds", HIDDEN_SEEDS)
                for value in seeds:
                    seed(value)
                    for index, values in enumerate(hidden_module["get_hidden_inputs"]()):
                        yield f"hidden/{value}/{index}", values, value
                        if stress_enabled(hidden_path):
                            for label, changed in stress_cases(values, value):
                                yield f"stress/{value}/{index}/{label}", changed, value
            native_cases = cases
            extra_rows = {}

            def cases():
                yield from native_cases()
                if hidden:
                    from ..numerics.operator_input_policy import extra_cases

                    for value in settings.get("hidden_seeds", HIDDEN_SEEDS):
                        for row in extra_cases(ref, value):
                            extra_rows[row["case_label"]] = row
                            yield row["case_label"], row["inputs"], value

            from ..evaluation.throughput_audit import bind_factory

            bind_factory(lambda: move(inputs()))
            from ..evaluation.cache_audit import bind_factory as bind_cache_factory

            bind_cache_factory(lambda: move(inputs()))
            for label, values, value in cases():
                observed = []
                dev = move(values)
                if instance and input_signature(dev) != input_signature(values):
                    raise ValueError("Device transfer changed a production input's dtype, shape or stride")
                prepared_pair = None

                def prepare_pair():
                    nonlocal prepared_pair
                    from ..evaluation.determinism import preserve_rng
                    from ..numerics.input_calibration import prepare_output

                    with preserve_rng(device):
                        seed(value)
                        prepared_pair = prepare_output(ref, dev, scope="first" if phase in {"baseline", "evaluate"} and first_ms is None else "control")

                def verify_order(dev_inputs, outputs):
                    expected = host(outputs["reference"])
                    if contract:
                        expected = host(copy.deepcopy(cpu)(*host(dev_inputs)))
                    for actual in outputs.values():
                        verify(dev_inputs, actual, expected=expected, prepared=prepared_pair)
                    observed.append(expected)

                with environment_guard(changes):
                    issue = check_paired_calls(ref, model, dev, seed=value, tolerance=settings["fp32_tolerance"],
                        device=device, set_seed=seed, metadata=metadata, label=label,
                        record_first_call=first_ms is None, clock=CLOCK, verify_pair=verify_order,
                        **({"prepare_pair": prepare_pair} if settings.get("input_profile") or settings.get("numerical_validation") else {}))
                if label in extra_rows:
                    from ..numerics.operator_input_policy import completed

                    completed(extra_rows.pop(label), dev, issue is None)
                if issue:
                    raise ValueError(issue)
                first_ms = metadata["first_forward_ms"]
                checked.append({"label": label, "input_signature": input_signature(dev),
                                "reference_signal": signal_audit(observed[-1], absolute_tolerance=settings["fp32_tolerance"])})
            if changes:
                raise ValueError("Candidate changed process settings")
            if phase == "profile":
                from ..evaluation.determinism import preserve_rng
                from ..numerics.input_calibration import prepare_output

                profile_args = move(args)
                profile_seed = settings.get("evaluation_seed", 42)
                from src.input_groups import layout

                from ..evaluation.determinism import signature
                from ..evaluation.state_paths import invocation_state
                from ..provenance.submission_source import SourceRegistry
                from .neuron_work import SCOPE

                registry = registry or SourceRegistry()
                training = bool(ref.training)
                if bool(model.training) != training:
                    raise ValueError("Neuron profile candidate changed the reference training mode")
                with preserve_rng(device):
                    seed(profile_seed)
                    profile_ticket = prepare_output(ref, profile_args, scope="control")
                    expected_profile = clone_tensors(ref(*move(profile_args)))
                    adapter.synchronize()
                seed(profile_seed)
                candidate_profile_args = move(profile_args)
                aliases = layout(candidate_profile_args)
                input_content = fingerprint({"value": signature(candidate_profile_args, typed=True)})

                def candidate_state():
                    try:
                        return invocation_state(model, candidate_profile_args, callable_identity=registry.callable_identity)
                    except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as exc:
                        return {"status": "not_measured", "reason": str(exc)[:300]}

                state_before = candidate_state()
                output_content = None

                def profiled_forward():
                    with environment_guard(changes):
                        output = model(*candidate_profile_args)
                        if bool(model.training) != training:
                            changes.append("candidate_training_mode")
                        return output

                def verify_profile(output):
                    nonlocal output_content
                    verify(profile_args, output, expected=expected_profile, prepared=profile_ticket)
                    if changes:
                        raise ValueError("Neuron profile candidate changed execution settings")
                    output_content = fingerprint({"value": signature(output, typed=True)})

                directory = ROOT / "cache/workflow" / fingerprint(payload) / "neuron_profile" / uuid.uuid4().hex
                report = adapter.profile(profiled_forward, directory=directory, verifier=verify_profile)
                state_after = candidate_state()
                metadata["profile_call"] = {"seed": profile_seed, "input_signature": input_signature(profile_args),
                    "training": training, "no_grad": True, "output_verified": True,
                    "independent_oracle_ticket": profile_ticket is not None,
                    "input_content_sha256": input_content,
                    "input_layout_sha256": fingerprint(aliases) if aliases is not None else None,
                    "output_content_sha256": output_content,
                    "model_state_before": state_before, "model_state_after": state_after,
                    "preceding_cases_sha256": fingerprint({"cases": checked}),
                    "precision_policy": {"native_tolerance": settings["fp32_tolerance"],
                        "production_contract": vars(contract) if contract else None,
                        "oracle_spec_sha256": profile_ticket.get("oracle_spec_sha256") if profile_ticket else None,
                        "independent_numerical": profile_ticket is not None},
                    "state_scope": SCOPE}
                return {"status": "ok", "purpose": "development_diagnostics_only", "metadata": metadata, **report}
            if strict and not all(accepted(row) for row in strict.values()):
                raise ValueError("Required Neuron auxiliary contract failed or is unavailable")
            metadata.update(cold_start_ms=setup_ms + (first_ms or 0), correctness_cases=checked,
                hidden_completed_checks=len(checked) if hidden else 0,
                correctness_scope="paired_orders_and_CPU_oracle_contracts; legacy_reference_stays_on_device")
            if hidden:
                metadata["evidence_receipt"] = from_result({"correctness": True}, settings, scope="hidden", additional=strict)
                return {"compiled": True, "correctness": True, "runtime": -1, "metadata": metadata}
            receipt = {}

            def factory(trial):
                value = fresh_seed()
                receipt.setdefault("input_seeds", []).append(value)
                seed(value)
                return move(inputs())

            if phase == "paired_timing":
                from ..evaluation.paired_task import collect_pairs

                winner = settings["paired_baseline_variant"]
                baseline = prepare_variant(copy.deepcopy(ref), winner, level=payload["level"], pid=payload["problem_id"])
                control = prepare_variant(copy.deepcopy(ref), winner, level=payload["level"], pid=payload["problem_id"])
                from ..numerics.input_calibration import prepare_output

                paired = collect_pairs(baseline, model, control, factory, ref, device=device,
                    trials=settings["paired_trials"], seed=settings.get("evaluation_seed", 42), set_seed=seed,
                    tolerance=settings["fp32_tolerance"], verifier=verify if contract or settings.get("input_profile") or settings.get("numerical_validation") else None,
                    prepare_verifier=(lambda values: prepare_output(ref, values)) if settings.get("numerical_validation") else None)
                metadata["paired_timing"] = {**paired, "baseline_variant": winner,
                    "output_consumed": True, "varied_inputs": True, "all_outputs_verified": True}
                return {"compiled": True, "correctness": True, "runtime": paired["candidate_mean_ms"], "metadata": metadata}
            times = synchronized_times(model, move(args), device, num_trials=settings["num_perf_trials"],
                                       input_factory=factory, verifier=verify, receipt=receipt)
            stats = timing_stats(times)
            metadata.update(timing_receipt=receipt, cold_start_ms=metadata["cold_start_ms"] + receipt.get("warmup_total_ms", 0))
            if phase == "baseline":
                return {**stats, "precision": settings["precision"], "tf32": False, "baseline_variant": variant,
                        "baseline_correctness": True, "reference_sha256": metadata["reference_sha256"],
                        "device_fingerprint": caps, "input_signature": metadata["input_signature"],
                        "cold_start_ms": metadata["cold_start_ms"], "timing_receipt": receipt, "metadata": metadata}
            metadata["evidence_receipt"] = from_result({"correctness": True}, settings, additional=strict)
            return {"compiled": True, "correctness": True, "runtime": stats["mean"], "runtime_stats": stats,
                    "peak_memory": None, "memory_stats": None, "metadata": metadata,
                    "amortized_runtime": amortized(stats["mean"], metadata["cold_start_ms"], settings["cold_calls"])}
