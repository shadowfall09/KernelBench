"""Scoped framework backend preferences and explicit layout conversion costs."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def settings(policy):
    import torch

    allowed = {"cudnn_benchmark", "memory_format", "blas_preference"}
    if (not isinstance(policy, dict) or set(policy) - allowed
            or "cudnn_benchmark" in policy and policy["cudnn_benchmark"] is not True
            or "memory_format" in policy and policy["memory_format"] not in {"channels_last", "channels_last_3d"}
            or "blas_preference" in policy and policy["blas_preference"] != "cublaslt"):
        raise ValueError("Invalid reviewed baseline backend policy")
    benchmark = torch.backends.cudnn.benchmark
    previous = torch.backends.cuda.preferred_blas_library() if policy.get("blas_preference") else None
    try:
        if policy.get("cudnn_benchmark"):
            torch.backends.cudnn.benchmark = True
        if policy.get("blas_preference"):
            torch.backends.cuda.preferred_blas_library(policy["blas_preference"])
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        if previous is not None:
            torch.backends.cuda.preferred_blas_library(previous)


def apply(model, policy):
    import torch

    layout = policy.get("memory_format")
    dimension = 4 if layout == "channels_last" else 5
    memory_format = torch.channels_last if dimension == 4 else torch.channels_last_3d

    def convert(value):
        if layout and isinstance(value, torch.Tensor) and value.ndim == dimension and (value.is_floating_point() or value.is_complex()):
            return value.contiguous(memory_format=memory_format)
        return value

    with settings(policy), torch.no_grad():
        if layout:
            model._apply(convert)

    def forward(*args):
        # Input conversion and preference dispatch stay inside the timed call;
        # model conversion belongs to setup. Graph capture records the same work.
        with settings(policy):
            return model(*(convert(value) for value in args))

    return forward
