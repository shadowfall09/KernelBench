"""Small baseline portfolio; each variant is evaluated in a separate task process."""

from __future__ import annotations

import math

VARIANTS = ("eager", "compile_default", "compile_max_autotune", "cuda_graph")
EXTENDED_VARIANTS = (*VARIANTS, "compile_default_cuda_graph", "compile_max_autotune_cuda_graph")


def variants_for(config, pid):
    if getattr(config, "device_backend", "cuda") == "neuron":
        from .neuron_worker import library_supported

        return ("neuron_eager", "neuron_compile", "neuron_library") if library_supported(config.level, pid) else ("neuron_eager", "neuron_compile")
    if config.baseline_portfolio == "standard":
        return VARIANTS
    from .baseline_rewrites import extension_options
    from .common import select_problems

    source = select_problems(config.level, problem_ids=[pid])[pid].read_text()
    return (*EXTENDED_VARIANTS, *extension_options(config.level, pid, source))


def prepare_baseline(model, args, device, variant, *, backend="", mode="default", rewrite=None, plan=None):
    import torch

    from .baseline_rewrites import framework_rewrite

    if plan is not None:
        if plan.get("protocol") != "kernelbench_pro_baseline_recipe_v1" or plan.get("variant") != variant:
            raise ValueError("Baseline variant differs from its source-bound recipe")
        rewrite = plan.get("rewrite")
    if variant not in {*EXTENDED_VARIANTS, "library_rewrite", "library_rewrite_cuda_graph"} and plan is None:
        raise ValueError(f"Unknown baseline variant: {variant}")
    if variant.startswith("library_rewrite"):
        if rewrite is None:
            raise ValueError("No reviewed library rewrite for this task")
        model = framework_rewrite(model, rewrite)
    if plan is not None and plan.get("backend_policy"):
        from .baseline_policies import apply

        model = apply(model, plan["backend_policy"])
    if variant.startswith("compile_"):
        backend, mode = "inductor", "max-autotune" if variant.startswith("compile_max_autotune") else "default"
        if variant.endswith("_cuda_graph") and mode == "max-autotune":
            mode = "max-autotune-no-cudagraphs"  # The outer capture owns replay and its input copies.
    if backend:
        options = {"backend": backend}
        if backend == "inductor":
            options["mode"] = mode
        model = torch.compile(model, **options)
    if variant == "cuda_graph" or variant.endswith("_cuda_graph"):
        model = cuda_graph_model(model, args, device)
    return model, backend, mode


def select_baseline(variants, cold_calls=10000):
    valid = {
        name: result for name, result in variants.items()
        if isinstance(result.get("mean"), (int, float)) and math.isfinite(result["mean"])
        and result["mean"] > 0 and result.get("baseline_correctness") is True
    }
    if not valid:
        return {"mean": -1, "metadata": {"error": "No baseline variant passed correctness"}, "variants": variants}
    from .measurement import amortized

    winner = min(valid, key=lambda name: valid[name]["mean"])
    cold = {name: amortized(result["mean"], result["cold_start_ms"], cold_calls)
            for name, result in valid.items()
            if isinstance(result.get("cold_start_ms"), (int, float))
            and math.isfinite(result["cold_start_ms"]) and result["cold_start_ms"] > 0}
    cold_winner = min(cold, key=cold.get) if cold else None
    return {
        **valid[winner], "metadata": dict(valid[winner].get("metadata", {})),
        "winning_baseline": winner, "variants": variants, "baseline_suite": True,
        "cold_calls": cold_calls, "cold_winning_baseline": cold_winner,
        "amortized_mean_ms": cold.get(cold_winner),
    }


def cuda_graph_model(model, args, device):
    import torch

    from src.input_groups import copy_inputs
    from src.validation import clone_tensors

    static = copy_inputs(args)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            model(*static)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        output = model(*static)

    def replay(*fresh):
        for destination, value in zip(static, fresh):
            if isinstance(destination, torch.Tensor):
                destination.copy_(value)
            elif destination != value:
                raise ValueError("CUDA Graph scalar inputs changed")
        graph.replay()
        # Both the input copy and stable output copy are part of timed replay.
        return clone_tensors(output)

    return replay
