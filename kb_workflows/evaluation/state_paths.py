"""Names for ordinary tensor state without evaluating candidate properties.

Plain paths start with @; dictionary keys retain their type and value. Walking
only ordinary module namespaces and builtin containers keeps lookup declarative.
"""

from __future__ import annotations

import json
import types
from collections import OrderedDict

MAX_STATES = 100_000
MAX_DEPTH = 64


def plain_tensors(model, *, allow_meta=False):
    import torch

    found, visited = {}, 0
    registered, modules = set(), set()

    def registered_objects(module):
        if id(module) in modules:
            return
        modules.add(id(module))
        namespace = vars(module)
        for name in ("_parameters", "_buffers"):
            registered.update(id(value) for value in namespace[name].values() if value is not None)
        for child in namespace["_modules"].values():
            if child is not None:
                registered_objects(child)

    registered_objects(model)

    def namespace(value):
        if any("__slots__" in vars(cls) for cls in type(value).__mro__):
            raise ValueError("Slotted state objects need an explicit state adapter")
        for cls in type(value).__mro__:
            descriptor = vars(cls).get("__dict__")
            if descriptor is not None:
                if type(descriptor) not in (types.GetSetDescriptorType, types.MemberDescriptorType):
                    raise ValueError("State object __dict__ must not be a custom descriptor")
                result = object.__getattribute__(value, "__dict__")
                if type(result) is not dict:
                    raise ValueError("State object namespace must be an ordinary dictionary")
                return result
        return None

    def walk(value, path, ancestors, depth):
        nonlocal visited
        visited += 1
        if visited > MAX_STATES or depth > MAX_DEPTH:
            raise ValueError("Plain state exceeds its traversal envelope")
        if isinstance(value, torch.Tensor):
            # RNN/LSTM/GRU keep ordinary lists pointing to the same registered
            # Parameters. They are already audited and are not extra state.
            if id(value) in registered:
                return
            if (type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided
                    or value.is_quantized or value.device.type == "meta" and not allow_meta):
                raise ValueError("Plain state must be a materialized dense tensor")
            found[path] = value
            return
        if value is None or type(value) in (bool, int, float, complex, str, bytes, torch.dtype, torch.device):
            return
        if isinstance(value, (dict, list, tuple)) and type(value) not in (dict, OrderedDict, list, tuple):
            raise ValueError("Container subclasses need an explicit state adapter")
        attributes = namespace(value) if not isinstance(value, torch.nn.Module) and type(value) not in (dict, OrderedDict, list, tuple) else None
        if not isinstance(value, torch.nn.Module) and type(value) not in (dict, OrderedDict, list, tuple) and attributes is None:
            return
        if id(value) in ancestors:
            raise ValueError("Cyclic plain state needs an explicit task adapter")
        ancestors = ancestors | {id(value)}
        if isinstance(value, torch.nn.Module):
            if id(value) not in modules:
                raise ValueError("Unregistered child modules need an explicit state adapter")
            for name, item in vars(value).items():
                if name not in {"_parameters", "_buffers", "_modules"}:
                    if type(name) is not str or not name.isidentifier():
                        raise ValueError("Plain module attribute names must be identifiers")
                    walk(item, path + ("." if path != "@" else "") + name, ancestors, depth + 1)
            for name, child in vars(value)["_modules"].items():
                if child is not None:
                    walk(child, path + ("." if path != "@" else "") + name, ancestors, depth + 1)
        elif type(value) in (dict, OrderedDict):
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise ValueError("Plain tensor-state dictionaries require string or integer keys")
                walk(item, path + "[" + json.dumps(key, ensure_ascii=True) + "]", ancestors, depth + 1)
        elif type(value) in (list, tuple):
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]", ancestors, depth + 1)
        else:
            for name, item in attributes.items():
                if type(name) is not str or not name.isidentifier():
                    raise ValueError("Ordinary state attribute names must be identifiers")
                walk(item, path + "." + name, ancestors, depth + 1)

    walk(model, "@", set(), 0)
    return found
