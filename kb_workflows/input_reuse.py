"""Isolated flat input groups that can be updated without replacing tensors.

The initial clone preserves storage/view aliasing with Torch's storage-aware
deepcopy. Full backing storage and logical hashing costs must be admitted first.
"""

from __future__ import annotations

import copy
import math


def tensors(inputs):
    import torch

    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("Reusable inputs require the declared nonempty flat argument list")
    result = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor or value.layout != torch.strided or value.is_quantized or value.__dict__:
                raise ValueError("Unsupported reusable input tensor representation")
            if value.is_conj() or value.is_neg() or any(step < 0 for step in value.stride()):
                raise ValueError("Unsupported reusable input view flags")
            result.append(value)
        elif type(value) not in (int, float, bool, str, type(None)) or type(value) is float and not math.isfinite(value):
            raise ValueError("Reusable input arguments must be tensors or declared finite scalars")
    if not result:
        raise ValueError("Reusable input groups need at least one tensor")
    return result


def description(inputs):
    """Normalize object/storage aliases without recording addresses."""
    values = tensors(inputs)
    objects, stores, result = {}, {}, []
    for value in values:
        storage = value.untyped_storage()
        key = str(value.device), storage._cdata
        result.append({"object": objects.setdefault(id(value), len(objects)), "storage": stores.setdefault(key, len(stores)),
            "offset": value.storage_offset(), "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)})
    return result


def storage_bytes(inputs):
    values = tensors(inputs)
    stores = {}
    for value in values:
        for item in (value, value.grad if value.is_leaf else None):
            if item is not None:
                storage = item.untyped_storage()
                stores[str(item.device), storage._cdata] = storage.nbytes()
    return sum(stores.values())


def logical_bytes(inputs):
    return sum(value.numel() * value.element_size() for value in tensors(inputs))


def isolated(inputs):
    """Detach graphs first; retain requires_grad and repeated tensor references."""
    import torch

    tensors(inputs)
    prepared, memo = [], {}
    with torch.no_grad():
        for item in inputs:
            if isinstance(item, torch.Tensor):
                if id(item) not in memo:
                    value = item.detach().requires_grad_(item.requires_grad)
                    if item.is_leaf and item.grad is not None:
                        value.grad = item.grad.detach()
                    memo[id(item)] = value
                prepared.append(memo[id(item)])
            else:
                prepared.append(item)
        output = copy.deepcopy(prepared)
    if description(output) != description(inputs):
        raise ValueError("Input clone did not retain its storage/view aliases")
    source_stores = {(str(value.device), value.untyped_storage()._cdata) for value in tensors(inputs)}
    if any((str(value.device), value.untyped_storage()._cdata) in source_stores for value in tensors(output)):
        raise ValueError("Input clone retained original storage")
    return output


def _nonoverlap_copy(target, source):
    """Zero-stride broadcast axes have only one independently stored value."""
    for axis in reversed(range(target.ndim)):
        if target.stride(axis) == 0 and target.shape[axis]:
            target, source = target.select(axis, 0), source.select(axis, 0)
    span = 1
    for stride, size in sorted(zip(target.stride(), target.shape)):
        if size > 1:
            if stride < span:
                raise ValueError("Complex overlapping input writes need a reviewed adapter")
            span += (size - 1) * stride
    target.copy_(source)


class MutableInputs:
    def __init__(self, inputs):
        self.spec = description(inputs)
        self.values = isolated(inputs)
        self.views = {id(value): value.detach() for value in tensors(self.values)}
        self.identities = [id(value) for value in tensors(self.values)]

    def update(self, new_values):
        import torch

        if description(new_values) != self.spec or len(new_values) != len(self.values):
            raise ValueError("Input update changed argument geometry, dtype, device or aliases")
        restored, same_storage = 0, True
        seen = set()
        with torch.no_grad():
            for index, (target, source) in enumerate(zip(self.values, new_values)):
                if not isinstance(target, torch.Tensor):
                    if type(target) is not type(source):
                        raise ValueError("Input update changed scalar role")
                    self.values[index] = source
                    continue
                if id(target) in seen:
                    continue
                seen.add(id(target))
                original = self.views[id(target)]
                needed = ((original.storage_offset() + 1 + sum((n - 1) * step for n, step in zip(original.shape, original.stride())))
                          * original.element_size() if original.numel() else 0)
                if original.untyped_storage().nbytes() < needed:
                    raise ValueError("Forward destroyed the retained input storage")
                before = target.untyped_storage()._cdata
                if (target.dtype, target.device, target.shape, target.stride(), target.storage_offset(), before) != (
                        original.dtype, original.device, original.shape, original.stride(), original.storage_offset(), original.untyped_storage()._cdata):
                    target.data = original
                    restored += 1
                same_storage = same_storage and before == target.untyped_storage()._cdata
                _nonoverlap_copy(target, source)
        unchanged = [id(value) for value in tensors(self.values)] == self.identities
        if not unchanged:
            raise ValueError("Input update replaced a tensor object")
        return {"same_tensor_objects": True, "storage_preserved_during_update": same_storage,
                "geometry_restorations": restored}
