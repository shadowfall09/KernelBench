"""Input-only snapshots and transfers that retain dense storage/view relations.

Output snapshots keep their existing validation path. Unsupported storage APIs
are explicit in layout receipts; they cannot qualify a required alias case.
"""

from __future__ import annotations

import torch

PROTOCOL = "kernelbench_pro_input_layout_v1"


def _leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)


def layout(value):
    """Address-free description; None means this storage backend is unsupported."""
    objects, stores, rows = {}, {}, []
    try:
        for tensor in _leaves(value):
            if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided
                    or tensor.is_quantized or tensor.is_conj() or tensor.is_neg()):
                return None
            key = str(tensor.device), tensor.untyped_storage()._cdata
            rows.append({"object": objects.setdefault(id(tensor), len(objects)), "storage": stores.setdefault(key, len(stores)),
                         "offset": tensor.storage_offset(), "shape": list(tensor.shape), "stride": list(tensor.stride()),
                         "dtype": str(tensor.dtype)})
    except (RuntimeError, TypeError, NotImplementedError):
        return None
    return {"protocol": PROTOCOL, "tensors": rows}


def copy_inputs(value, transform=None, *, isolate=True):
    """Apply an elementwise device/dtype transfer to each full aliased backing.

    Independent zero-offset tensors use the legacy logical clone path. Full
    backing copies are only required for offsets or relationships across leaves.
    Autograd history is detached, as with the existing native input snapshots.
    """
    from .validation import clone_tensors

    description = layout(value)
    leaves = list(_leaves(value))
    groups, memo = {}, {}
    if description is not None:
        for tensor, record in zip(leaves, description["tensors"]):
            groups.setdefault(record["storage"], {})[id(tensor)] = tensor
        for members in groups.values():
            tensors = list(members.values())
            special = len(tensors) > 1 or any(tensor.storage_offset() or not tensor.numel()
                          or any(stride == 0 for stride in tensor.stride()) for tensor in tensors)
            if not special:
                continue
            if len({tensor.dtype for tensor in tensors}) != 1:
                raise ValueError("Cross-dtype storage aliases need a reviewed input adapter")
            first = tensors[0]
            if any(stride < 0 for tensor in tensors for stride in tensor.stride()):
                raise ValueError("Negative input strides need a reviewed input adapter")
            storage = first.untyped_storage()
            if storage.nbytes() % first.element_size():
                raise ValueError("Input storage does not contain a whole typed backing")
            backing = torch.empty(0, dtype=first.dtype, device=first.device).set_(
                storage, 0, (storage.nbytes() // first.element_size(),), (1,))
            backing = transform(backing) if transform is not None else backing
            if backing.ndim != 1 or backing.numel() != storage.nbytes() // first.element_size():
                raise ValueError("Input transfer changed its backing geometry")
            if isolate:
                backing = backing.detach().clone()
            for tensor in tensors:
                memo[id(tensor)] = backing.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())

    def rebuild(item):
        if isinstance(item, torch.Tensor):
            if id(item) not in memo:
                result = transform(item) if transform is not None else item
                memo[id(item)] = clone_tensors(result) if isolate else result
            return memo[id(item)]
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(rebuild(child) for child in item)
        return item

    result = rebuild(value)
    if description is not None:
        after = layout(result)
        if after is None or len(after["tensors"]) != len(description["tensors"]):
            raise ValueError("Input copy lost its supported layout description")
        for before, copied in zip(description["tensors"], after["tensors"]):
            if any(before[key] != copied[key] for key in ("shape", "stride", "offset", "object", "storage")):
                raise ValueError("Input copy changed its shape, stride, offset or alias relationships")
    return result
