"""Names for ordinary tensor state without evaluating candidate properties.

Plain paths start with @; dictionary keys retain their type and value. Walking
only ordinary module namespaces and builtin containers keeps lookup declarative.
"""

from __future__ import annotations

import inspect
import json
import types
from collections import OrderedDict

MAX_STATES = 100_000
MAX_DEPTH = 64


def namespace(value):
    """Read ordinary storage without calling a candidate property or iterator."""
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


def invocation_state(model, inputs, *, callable_identity):
    """Hash actual state and entrypoints, retaining input/state storage aliases.

    This observes module data and callable identities, not arbitrary global
    dependencies, host control flow or execution ownership.
    """
    import torch

    from src.input_groups import layout

    from ..common import fingerprint
    from .determinism import signature

    tensors, seen, classes, declarations, visited = [], {}, {}, [], 0

    def walk(value, depth=0):
        nonlocal visited
        visited += 1
        if visited > MAX_STATES or depth > MAX_DEPTH:
            raise ValueError("Native model state exceeds its traversal envelope")
        if value is None or type(value) in (bool, int, float, complex, str, bytes, torch.dtype, torch.device):
            if type(value) in (str, bytes) and len(value) > 1024**2:
                raise ValueError("Native model state contains an excessive literal")
            if type(value) is bytes:
                return {"type": "bytes", "hex": value.hex()}
            if type(value) in (torch.dtype, torch.device):
                return {"type": type(value).__name__, "value": str(value)}
            return signature(value, typed=True)
        if id(value) in seen:
            return {"ref": seen[id(value)]}
        identity = seen[id(value)] = len(seen)
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            if value.device.type == "meta":
                raise ValueError("Native model state must be materialized")
            tensors.append(value)
            return {"id": identity, "kind": type(value).__name__, "requires_grad": value.requires_grad,
                    "value": signature(value.detach(), typed=True)}
        cls = type(value)
        if type(cls) is not type:
            raise ValueError("Custom metaclasses need an explicit native state adapter")
        if issubclass(cls, torch.Tensor):
            raise ValueError("Native model state cannot observe tensor subclasses")
        if type(value) in (types.FunctionType, types.MethodType, staticmethod, classmethod):
            function = value.__func__ if type(value) is not types.FunctionType else value
            return {"id": identity, "callable": callable_identity(value), "defaults": walk(function.__defaults__, depth + 1),
                    "kwdefaults": walk(function.__kwdefaults__, depth + 1),
                    "bound_to": walk(value.__self__, depth + 1) if type(value) is types.MethodType else None,
                    "closure": [walk(cell.cell_contents, depth + 1) for cell in function.__closure__ or ()]}
        if type(value) in (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType):
            bound = getattr(value, "__self__", None)
            return {"id": identity, "callable": callable_identity(value),
                    "bound_to": walk(bound, depth + 1) if type(bound) in (torch.Tensor, torch.nn.Parameter) else None}
        if type(value) is type:
            if id(value) not in classes:
                raise ValueError("Class-valued state needs an observed class declaration")
            return {"id": identity, "class_ref": classes[id(value)]}
        if type(value) in (dict, OrderedDict):
            if any(type(key) not in (str, int) for key in value):
                raise ValueError("Native state dictionary keys must be strings or integers")
            return {"id": identity, "kind": type(value).__name__, "items": [
                [walk(key, depth + 1), walk(item, depth + 1)] for key, item in value.items()]}
        if type(value) in (tuple, list):
            return {"id": identity, "kind": type(value).__name__, "items": [walk(item, depth + 1) for item in value]}
        if type(value) in (set, frozenset):
            if any(type(item) is not str for item in value):
                raise ValueError("Native state sets require plain string entries")
            return {"id": identity, "kind": type(value).__name__, "items": list(value)}
        if issubclass(cls, (dict, list, tuple, set, frozenset)):
            raise ValueError("Container subclasses need an explicit native state adapter")
        attributes = namespace(value)
        if attributes is None or issubclass(cls, types.ModuleType):
            raise ValueError("Unsupported opaque native model state")
        result = {"id": identity}
        if id(cls) not in classes:
            index = classes[id(cls)] = len(declarations)
            declarations.append(None)
            excluded = {"__dict__", "__weakref__", "__module__", "__doc__", "__qualname__", "__annotations__"}
            declarations[index] = {"qualname": cls.__qualname__, "attributes": walk(
                {name: item for name, item in vars(cls).items() if name not in excluded}, depth + 1)}
        result["class_ref"] = classes[id(cls)]
        if issubclass(cls, torch.nn.Module):
            result["entrypoints"] = {
                name: walk(item, depth + 1) if type(item) in (types.FunctionType, types.MethodType, staticmethod, classmethod)
                else callable_identity(item) if item is not None else None
                for name in ("forward", "__call__", "__getattribute__", "__getattr__", "_call_impl", "_compiled_call_impl")
                for item in [inspect.getattr_static(value, name, None)]}
        result["attributes"] = walk(attributes, depth + 1)
        return result

    data = walk(model)
    aliases = layout([inputs, tensors])
    if aliases is None:
        raise ValueError("Native model state storage layout is unsupported")
    return {"status": "observed", "sha256": fingerprint({"model": data, "classes": declarations, "input_state_layout": aliases}),
            "tensor_count": len(tensors), "scope": "module_data_entrypoints_and_input_state_aliases; no_host_control_proof"}
