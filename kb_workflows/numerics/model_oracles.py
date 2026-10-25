"""Source-bound L3 plans, meta admission and independent CPU FP64 algorithms.

The reference's meta forward is only a shape/layout observer. Numerical truth
uses the reviewed FX layer interpreter or a separate explicit model algorithm.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ..common import fingerprint
from ..evaluation.state_contract import _plain_tensors
from . import composite_math, model_layers, model_math
from .composite_oracles import NATIVE_DROPOUT, meta_copy
from .composite_oracles import Prepared as CompositePrepared
from .operator_oracles import _host

PROTOCOL = "kernelbench_pro_model_oracle_v1"
EXPLICIT_TASKS = {29, 30, *range(33, 45), *range(46, 51)}


def _constant(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite model constructor attribute")
        return value
    if type(value) in (tuple, list):
        return {"type": type(value).__name__, "items": [_constant(item) for item in value]}
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError("Not a scalar constructor attribute")


def architecture(model):
    """Freeze the entire module tree, scalar attributes and state geometry.

    Training mode and floating storage dtype are runtime contracts; changing
    precision or declared train/eval mode does not create a different topology.
    """
    modules = {}
    for name, layer in model.named_modules(remove_duplicate=False):
        attributes = {}
        for key, value in vars(layer).items():
            if key.startswith("_") or key == "training":
                continue
            try:
                attributes[key] = _constant(value)
            except TypeError:
                continue
        if type(layer) is torch.nn.TransformerEncoderLayer:
            if layer.activation not in {torch.nn.functional.relu, torch.nn.functional.gelu}:
                raise ValueError("Unreviewed encoder activation")
            attributes["activation"] = "relu" if layer.activation is torch.nn.functional.relu else "gelu"
        modules[name] = {"class": type(layer).__name__, "attributes": attributes}
    state = {**dict(model.named_parameters(remove_duplicate=False)), **dict(model.named_buffers(remove_duplicate=False)), **_plain_tensors(model)}
    return {"modules": modules, "state": {name: {"shape": list(value.shape),
                "kind": "floating" if value.is_floating_point() else str(value.dtype)} for name, value in sorted(state.items())}}


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class Prepared:
    def __init__(self, model, inputs, *, problem_id, expected_program=None):
        if type(problem_id) is not int or not 1 <= problem_id <= 50:
            raise ValueError("Independent L3 oracle requires an original model task")
        self.model, self.inputs, self.problem_id = model, inputs, problem_id
        self.input_device = next(_tensors(inputs)).device
        self.inner = None
        if problem_id not in EXPLICIT_TASKS:
            self.inner = CompositePrepared(model, inputs, extra_modules=model_layers.MODULES, extra_module_call=model_layers.module)
        self.description = {"protocol": PROTOCOL, "problem_id": problem_id, "architecture": architecture(model),
            "algorithm": "reviewed_explicit_model" if self.inner is None else "reviewed_fx_layers",
            "fx_program_sha256": self.inner.program_sha256 if self.inner is not None else None,
            "fx_node_count": len(self.inner.description["nodes"]) if self.inner is not None else 0,
            "fx_dropout_paths": ({node.name: str(node.target) for node in self.inner.graph.graph.nodes
                if node.op == "call_module" and type(self.inner.graph.get_submodule(node.target)) is torch.nn.Dropout}
                if self.inner is not None else {})}
        self.events, self.masks, self.dropout_states = [], [], []
        total = sum(value.numel() for value in _tensors(inputs))
        total += sum(math.prod(row["shape"]) for row in self.description["architecture"]["state"].values())
        clone = meta_copy(model)
        if problem_id == 44:
            # deepcopy retains function closures. Rebind the source's mlpf
            # closure to the isolated meta ModuleDict for shape observation.
            clone.mlpf = lambda x: clone.mlp.dropout(clone.mlp.c_proj(clone.mlp.act(clone.mlp.c_fc(x))))
        handles = []
        peak_result = 0

        def dropout(name):
            def observe(layer, args, output):
                value = args[0]
                self.events.append({"node": name, "shape": list(value.shape), "input_stride": list(value.stride()),
                    "dtype": str(value.dtype), "training": layer.training, "probability": layer.p, "inplace": layer.inplace})
            return observe

        def attention_workspace(layer, args, kwargs, output):
            nonlocal total
            query = args[0] if args else kwargs["query"]
            key = args[1] if len(args) > 1 else kwargs["key"]
            batch_axis, length_axis = (0, 1) if layer.batch_first else (1, 0)
            total += query.shape[batch_axis] * layer.num_heads * query.shape[length_axis] * key.shape[length_axis]

        class SizeObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal peak_result
                result = func(*args, **(kwargs or {}))
                peak_result = max(peak_result, sum(value.numel() for value in _tensors(result)))
                return result

        for name, layer in clone.named_modules():
            if type(layer) is torch.nn.Dropout:
                handles.append(layer.register_forward_hook(dropout(name)))
            if type(layer) is torch.nn.MultiheadAttention:
                handles.append(layer.register_forward_hook(attention_workspace, with_kwargs=True))
        try:
            values = [torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
                      if isinstance(value, torch.Tensor) else value for value in inputs]
            if 35 <= problem_id <= 42:
                layer = clone.lstm if problem_id <= 38 else clone.gru
                x = values[0]
                batch, length = x.shape[:2] if layer.batch_first else x.shape[1::-1]
                if x.ndim != 3 or x.shape[-1] != layer.input_size or layer.dropout != 0:
                    raise ValueError("Recurrent input geometry or dropout differs from its reviewed contract")
                directions = 2 if layer.bidirectional else 1
                for state in values[1:]:
                    if state is not None and tuple(state.shape) != (layer.num_layers * directions, batch, layer.hidden_size):
                        raise ValueError("Recurrent initial-state geometry differs from its reviewed contract")
                peak_result = batch * length * layer.hidden_size * directions * 8
            else:
                with torch.no_grad(), SizeObserver():
                    clone(*values)
        finally:
            for handle in handles:
                handle.remove()
        self.estimated_host_bytes = max(1, 64 * (total + peak_result), self.inner.estimated_host_bytes if self.inner is not None else 0)
        self.description["dropout_order"] = [event["node"] for event in self.events]
        self.program_sha256 = fingerprint(self.description)
        if expected_program is not None and self.program_sha256 != expected_program:
            raise ValueError("Independent model constructor/arithmetic changed from its reviewed source plan")
        self.next_event = 0

    def call(self, path, *args):
        layer = self.model.get_submodule(path)
        if type(layer) is torch.nn.Dropout:
            event = self.events[self.next_event] if self.next_event < len(self.events) else None
            self.next_event += 1
            if (event is None or event["node"] != path or event["shape"] != list(args[0].shape)
                    or event["probability"] != layer.p or event["training"] is not layer.training):
                raise ValueError("Independent model dropout call differs from the observed meta plan")
            replayed = layer.training and 0 < layer.p < 1
            self.dropout_states.append({"node": path, "training": layer.training, "probability": layer.p,
                                        "inplace": layer.inplace, "native_mask_replayed": replayed})

            def sample(value, probability):
                dtype = getattr(torch, event["dtype"].removeprefix("torch."))
                ones = torch.empty_strided(value.shape, event["input_stride"], dtype=dtype, device=self.input_device).fill_(1)
                sampled = NATIVE_DROPOUT(ones, probability, training=True)
                if not torch.isfinite(sampled).all():
                    raise ValueError("Model native random mask is unavailable")
                mask = (sampled > 0).cpu()
                self.masks.append({"node": path, "shape": list(mask.shape), "dtype": event["dtype"],
                    "device": str(self.input_device), "input_stride": event["input_stride"], "probability": probability,
                    "elements": mask.numel(), "kept": int(mask.sum()), "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                    "scope": "native_random_mask_only; independent_FP64_arithmetic"})
                return mask
        else:
            sample = None
        return composite_math.module(layer, args, {}, dropout=sample)

    def execute(self):
        self.masks, self.dropout_states, self.next_event = [], [], 0
        if self.inner is not None:
            value = self.inner.execute()
            paths = {node.name: str(node.target) for node in self.inner.graph.graph.nodes if node.op == "call_module"}
            self.masks = [{**row, "node": paths[row["node"]]} for row in self.inner.masks]
            self.dropout_states = [{key: event[key] for key in ("node", "training", "probability", "inplace")}
                | {"native_mask_replayed": event["training"] and 0 < event["probability"] < 1} for event in self.events]
            if [row["node"] for row in self.masks] != [row["node"] for row in self.dropout_states if row["native_mask_replayed"]]:
                raise ValueError("Independent FX arithmetic did not replay every model random mask")
            return value
        result = model_math.execute(self, self.problem_id, _host(self.inputs))
        if self.next_event != len(self.events):
            raise ValueError("Independent model did not consume every declared dropout call")
        return result

    def receipt(self):
        return {"protocol": PROTOCOL, "program_sha256": self.program_sha256, "problem_id": self.problem_id,
            "algorithm": self.description["algorithm"], "reference_numerical_forward_called": False,
            "shape_propagation_device": "meta", "dropout_states": self.dropout_states, "masks": self.masks,
            "meta_dropout_calls": [{key: event[key] for key in ("node", "shape", "input_stride", "dtype", "training", "probability", "inplace")} for event in self.events],
            "inner_receipt": self.inner.receipt() if self.inner is not None else None}
