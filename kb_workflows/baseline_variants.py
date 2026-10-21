"""Small baseline portfolio; each variant is evaluated in a separate task process."""

from __future__ import annotations

import math

VARIANTS = ("eager", "compile_default", "compile_max_autotune", "cuda_graph")


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

    from src.validation import clone_tensors

    static = clone_tensors(args)
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
