"""Source-bound FX plans interpreted with independent FP64 layer arithmetic.

Tracing and shape propagation execute Python and meta kernels, not reference
numerical forward computation. Arithmetic never calls the graph's forward or
native fused layer forward. Native dropout is used only to replay its mask.
"""

from __future__ import annotations

import copy
import hashlib
import math

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from . import composite_math as arithmetic
from .common import fingerprint
from .operator_oracles import _host
from .state_contract import _plain_tensors, _registered

PROTOCOL = "kernelbench_pro_composite_oracle_v1"
NATIVE_DROPOUT = torch.nn.functional.dropout


def meta_copy(model):
    """Copy only state metadata, avoiding a transient real-device deep copy."""
    registered, _ = _registered(model) if next(iter(model.parameters()), torch.empty(0)).device.type != "meta" else ({}, set())
    tensors = {**{name: tensor for name, (_, tensor) in registered.items()}, **_plain_tensors(model)}
    if not registered:
        tensors.update({f"parameter/{name}": tensor for name, tensor in model.named_parameters(remove_duplicate=False)})
        tensors.update({f"buffer/{name}": tensor for name, tensor in model.named_buffers(remove_duplicate=False)})
    memo = {}
    for tensor in tensors.values():
        if id(tensor) not in memo:
            value = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="meta")
            memo[id(tensor)] = torch.nn.Parameter(value, requires_grad=tensor.requires_grad) if isinstance(tensor, torch.nn.Parameter) else value
    return copy.deepcopy(model, memo)


class Tracer(torch.fx.Tracer):
    def is_leaf_module(self, module, name):
        if type(module) in arithmetic.MODULES:
            return True
        return super().is_leaf_module(module, name)


def trace(model):
    clone = meta_copy(model)
    graph = torch.fx.GraphModule(clone, Tracer().trace(clone))
    for node in graph.graph.nodes:
        if node.op == "call_module" and type(graph.get_submodule(node.target)) not in arithmetic.MODULES:
            raise ValueError(f"No reviewed module arithmetic: {type(graph.get_submodule(node.target)).__name__}")
        if node.op == "call_function" and node.target not in arithmetic.FUNCTIONS:
            raise ValueError(f"No reviewed function arithmetic: {node.target}")
        if node.op == "call_method" and node.target not in arithmetic.METHODS:
            raise ValueError(f"No reviewed method arithmetic: {node.target}")
    return graph


def describe(graph):
    def encode(value):
        if isinstance(value, torch.fx.Node):
            return {"node": value.name}
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "items": [encode(item) for item in value]}
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in sorted(value.items())}
        if isinstance(value, slice):
            return {"slice": [encode(value.start), encode(value.stop), encode(value.step)]}
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError(f"Unreviewed graph constant: {type(value).__name__}")

    rows = []
    constants, modules, state_shapes = {}, {}, {}
    for node in graph.graph.nodes:
        if node.op == "call_module":
            layer = graph.get_submodule(node.target)
            target = f"{type(layer).__name__}:{node.target}"
            fields = ("in_features", "out_features", "in_channels", "out_channels", "groups", "kernel_size", "stride", "padding", "dilation", "output_padding", "padding_mode", "num_features", "num_groups", "normalized_shape", "eps", "momentum", "track_running_stats", "affine", "elementwise_affine", "output_size", "return_indices", "ceil_mode", "count_include_pad", "divisor_override", "negative_slope", "approximate", "min_val", "max_val", "beta", "threshold", "dim", "p", "inplace")
            attributes = {field: encode(getattr(layer, field)) for field in fields if hasattr(layer, field)}
            for field in ("weight", "bias"):
                if hasattr(layer, field):
                    value = getattr(layer, field)
                    attributes[field + "_shape"] = list(value.shape) if isinstance(value, torch.Tensor) else None
            modules[node.target] = {"class": type(layer).__name__, "attributes": attributes}
        elif node.op == "call_function":
            target = arithmetic.FUNCTIONS[node.target]
        else:
            target = str(node.target)
        if node.op == "get_attr":
            state = graph
            for part in node.target.split("."):
                state = getattr(state, part)
            if isinstance(state, torch.Tensor):
                state_shapes[node.target] = list(state.shape)
        if node.op == "get_attr" and str(node.target).startswith("_tensor_constant"):
            value = graph
            for part in node.target.split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor) or value.device.type == "meta":
                raise ValueError("A folded numerical graph constant has no materialized source value")
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            constants[node.target] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
        rows.append({"name": node.name, "op": node.op, "target": target, "args": encode(node.args), "kwargs": encode(node.kwargs)})
    return {"protocol": PROTOCOL, "nodes": rows, "constants": constants, "modules": modules, "state_shapes": state_shapes,
            "arithmetic": "explicit_cpu_fp64_layers_and_reviewed_tensor_primitives",
            "stochastic_semantics": "native_value_independent_dropout_mask_replayed_at_original_dtype_device_and_rng"}


def _tensor_metadata(value):
    if isinstance(value, TensorMetadata):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_metadata(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_metadata(item)


class Prepared:
    def __init__(self, model, inputs, *, expected_program=None):
        self.model, self.inputs = model, inputs
        self.graph = trace(model)
        self.description = describe(self.graph)
        self.program_sha256 = fingerprint(self.description)
        if expected_program is not None and expected_program != self.program_sha256:
            raise ValueError("Independent composite graph changed from its reviewed source plan")
        meta_inputs = [torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
                       if isinstance(value, torch.Tensor) else value for value in inputs]
        ShapeProp(self.graph).propagate(*meta_inputs)
        self.input_device = next(value.device for value in inputs if isinstance(value, torch.Tensor))
        total = sum(value.numel() for value in inputs if isinstance(value, torch.Tensor))
        total += sum(value.numel() for value in [*model.parameters(), *model.buffers()])
        total += sum(math.prod(meta.shape) for node in self.graph.graph.nodes for meta in _tensor_metadata(node.meta.get("tensor_meta")))
        self.estimated_host_bytes = max(1, 64 * total)
        self.masks, self.dropout_states = [], []

    def execute(self):
        owner = self

        class Interpreter(torch.fx.Interpreter):
            def run_node(self, node):
                self.current = node
                return super().run_node(node)

            def get_attr(self, target, args, kwargs):
                value = owner.model
                try:
                    for part in target.split("."):
                        value = getattr(value, part)
                except AttributeError:
                    value = super().get_attr(target, args, kwargs)
                return _host(value)

            def call_function(self, target, args, kwargs):
                if arithmetic.FUNCTIONS[target] == "tensor":
                    observed = self.current.meta.get("tensor_meta")
                    if not isinstance(observed, TensorMetadata):
                        raise ValueError("Graph tensor literal has no observed native dtype")
                    kwargs = {**kwargs, "dtype": observed.dtype}
                return arithmetic.function(target, args, kwargs)

            def call_method(self, target, args, kwargs):
                return arithmetic.method(target, args, kwargs)

            def call_module(self, target, args, kwargs):
                layer = owner.model.get_submodule(target)
                if type(layer) is torch.nn.Dropout:
                    owner.dropout_states.append({"node": self.current.name, "training": layer.training,
                        "probability": layer.p, "inplace": layer.inplace,
                        "native_mask_replayed": layer.training and 0 < layer.p < 1})

                def dropout(value, probability):
                    node = self.current.args[0]
                    if not isinstance(node, torch.fx.Node) or not isinstance(node.meta.get("tensor_meta"), TensorMetadata):
                        raise ValueError("Native dropout input storage dtype was not observed in the meta plan")
                    dtype = node.meta["tensor_meta"].dtype
                    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
                        raise ValueError("No native dropout mask replay for this storage dtype")
                    ones = torch.empty_strided(value.shape, node.meta["tensor_meta"].stride, dtype=dtype, device=owner.input_device).fill_(1)
                    sampled = NATIVE_DROPOUT(ones, probability, training=True)
                    if torch.isnan(sampled).any():
                        raise ValueError("Native dropout mask could not be observed independently of numerical values")
                    mask = (sampled > 0).cpu()
                    owner.masks.append({"node": self.current.name, "shape": list(mask.shape), "dtype": str(dtype),
                        "device": str(owner.input_device), "input_stride": list(node.meta["tensor_meta"].stride),
                        "probability": probability, "elements": mask.numel(),
                        "kept": int(mask.sum()), "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                        "scope": "native_random_mask_only; independent_FP64_arithmetic"})
                    return mask

                return arithmetic.module(layer, args, kwargs, dropout=dropout)

        return Interpreter(self.graph).run(*_host(self.inputs))

    def receipt(self):
        return {"protocol": PROTOCOL, "program_sha256": self.program_sha256,
                "node_count": len(self.description["nodes"]), "masks": self.masks, "dropout_states": self.dropout_states,
                "reference_numerical_forward_called": False,
                "source_python_symbolically_traced": True, "shape_propagation_device": "meta"}
