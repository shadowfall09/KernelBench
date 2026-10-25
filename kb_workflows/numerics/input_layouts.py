"""Reviewed alias cases and SDK-light replay of their actual delivered layouts."""

from __future__ import annotations

import math

LABELS = ("storage_offset", "shared_storage")
PROTOCOL = "kernelbench_pro_input_layout_v1"


def strides(shape):
    result, step = [], 1
    for size in reversed(shape):
        result.append(step)
        step *= size
    return list(reversed(result))


def estimate(label, dimensions):
    shapes = [shape for shape in dimensions if shape is not None]
    count = sum(math.prod(shape) for shape in shapes)
    if label not in LABELS:
        return 16 * count
    guard = sum(8 + 3 * index for index in range(len(shapes))) if label == "storage_offset" else 8
    return 32 * (count + guard)


def expected(label, signature):
    if label not in LABELS:
        return None
    tensors = [item for item in signature if "shape" in item]
    if label == "shared_storage" and (len(tensors) < 2 or len({item["dtype"] for item in tensors}) != 1
                                     or tensors[0]["dtype"] not in {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}):
        raise ValueError("Shared layout receipt needs multiple uniform floating inputs")
    return {"protocol": PROTOCOL, "tensors": [{"object": index, "storage": index if label == "storage_offset" else 0,
                "offset": 7 + 3 * index if label == "storage_offset" else 7,
                "shape": item["shape"], "stride": strides(item["shape"]), "dtype": item["dtype"]}
            for index, item in enumerate(tensors)]}


def matches(actual, target):
    """Receipt integers are strict: bool/float IDs cannot masquerade as ints."""
    if (type(actual) is not dict or set(actual) != set(target)
            or type(actual.get("tensors")) is not list or len(actual["tensors"]) != len(target["tensors"])):
        return False
    for row, wanted in zip(actual["tensors"], target["tensors"]):
        if (type(row) is not dict or set(row) != set(wanted)
                or any(type(row.get(key)) is not int for key in ("object", "storage", "offset"))
                or any(type(row.get(key)) is not list or any(type(n) is not int for n in row[key])
                       for key in ("shape", "stride"))):
            return False
    return actual == target


def apply(values, label, *, generator, allocation_device):
    import torch

    if label not in LABELS:
        return values
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if label == "shared_storage" and (len(tensors) < 2 or len({value.dtype for value in tensors}) != 1
                                     or not all(value.is_floating_point() for value in tensors)):
        raise ValueError("Shared input storage requires reviewed uniform floating roles")
    common = (torch.empty(max(value.numel() for value in tensors) + 8, dtype=tensors[0].dtype, device=allocation_device)
              if label == "shared_storage" else None)
    result, index = [], 0
    for value in values:
        if not isinstance(value, torch.Tensor):
            result.append(value)
            continue
        offset = 7 + 3 * index if common is None else 7
        backing = common if common is not None else torch.empty(value.numel() + offset + 1, dtype=value.dtype, device=allocation_device)
        if common is None or index == 0:
            backing.fill_(3)  # Initialized guards expose kernels using the storage base as data_ptr.
        view = backing.as_strided(value.shape, strides(value.shape), offset)
        view.copy_(value)
        result.append(view)
        index += 1
    return result
