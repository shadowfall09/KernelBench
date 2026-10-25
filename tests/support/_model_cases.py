"""Explicit small constructor configurations, never scored original workloads."""

from unittest.mock import patch

import torch

from kb_workflows.common import problem_paths
from src.eval import load_original_model_and_inputs

SMALL_TASKS = [1, 2, 3, 4, 6, 8, 13, 14, 17, 19, 21, 25, 27, *range(28, 51)]


def small_model_case(pid, dtype=torch.float64):
    source = problem_paths(3)[pid].read_text()
    namespace = {}
    Model, _, _ = load_original_model_and_inputs(source, namespace)
    namespace["batch_size"] = 2
    kwargs = {}
    if pid in {1, 2, 3}:
        init, shapes = [4, [6, 5], 3], [(2, 4)]
    elif pid == 4:
        init, shapes = [3], [(2, 1, 32, 32)]
    elif pid == 6:
        init, shapes = [3, 2, 2, 3, 1, 2, 2], [(2, 3, 7, 9)]
    elif pid == 8:
        init, shapes = [3, 4], [(2, 3, 7, 9)]
    elif pid == 13:
        init, shapes = [4, 2], [(2, 4, 8, 10)]
    elif pid == 14:
        init, shapes = [2, 4, 2], [(2, 4, 7, 9)]
    elif pid == 17:
        init, shapes = [4, 2, 3, 4], [(2, 4, 7, 9)]
    elif pid == 19:
        init, shapes = [3, 3, .125], [(2, 3, 224, 224)]
    elif pid == 21:
        init, shapes = [4, 4, 3, 1, 2], [(2, 4, 7, 9)]
    elif pid == 25:
        init, shapes = [6, 12, 3], [(2, 6, 7, 9)]
    elif pid == 27:
        init, shapes = [3, 2, [4, 8], 3], [(2, 3, 9, 11)]
    elif pid == 28:
        init, shapes = [8, 2, 3, 8, 2, 2, 12, 3, 0, 0], [(2, 3, 8, 8)]
    elif pid in {29, 30}:
        init, shapes = [], [(2, 3, 16, 16)]
        kwargs = dict(img_size=16, patch_size=2, num_classes=3, embed_dim=8,
                      depths=[2, 2], num_heads=[2, 4], window_size=2, mlp_ratio=2.)
    elif pid == 31:
        init, shapes = [8, 2], [(2, 8, 3, 5)]
    elif pid == 32:
        init, shapes = [3, 8, 2, 2, 2., 2, 3, 8], [(2, 3, 8, 8)]
    elif pid == 33:
        init, shapes = [3, 4, 2], [(2, 3), (2, 4)]
    elif pid == 34:
        init, shapes = [3, 4, 2], [(5, 2, 3), (2, 4)]
    elif 35 <= pid <= 38:
        init, shapes = [3, 4, 2, 2, 0.], [(2, 5, 3)]
        # Explicit optional states keep the toy FP64 call legal: L3/35's
        # omitted-state factory creates FP32 zeros regardless of x.dtype.
        shapes += [(4 if pid == 38 else 2, 2, 4)] * 2
    elif 39 <= pid <= 42:
        init, shapes = [3, 4, 2], [(5, 2, 3), (4 if pid >= 41 else 2, 2, 4)]
    elif pid in {43, 44}:
        init, shapes = [8, 2, 0., 0., 8], [(2, 5, 8)]
    elif pid == 45:
        init, shapes = [2, 3, 2], [(2, 2, 16, 16)]
    elif pid in {46, 47}:
        init, shapes = [3, 4, 2 if pid == 46 else 0], [(2, 5, 4)]
    elif pid in {48, 49}:
        init, shapes = [2, 8, 2, 3, 4, 4], [(2, 8, 2, 3)]
    elif pid == 50:
        init, shapes = [8, 2, 8], [(2, 5, 8)]
    else:
        raise ValueError("No small model fixture")
    return source, Model(*init, **kwargs).to(dtype=dtype), [torch.randn(shape, dtype=dtype) for shape in shapes]


def original_meta(level, pid, *, nominal=True):
    original = torch.linspace

    def constructor_constants(*args, **kwargs):
        # Swin stores at most 24 scalar stochastic-depth probabilities via
        # .item(). Only this bounded constant vector is materialized on CPU;
        # all model parameters, buffers, inputs and outputs remain on meta.
        assert len(args) >= 3 and args[2] <= 4096
        return original(*args, **{**kwargs, "device": "cpu"})

    source = problem_paths(level)[pid].read_text()
    with torch.device("meta"), patch.object(torch, "linspace", constructor_constants):
        Model, init, inputs = load_original_model_and_inputs(source, {})
        return source, Model(*init()), inputs() if nominal else None
