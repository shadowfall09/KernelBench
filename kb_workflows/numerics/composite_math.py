"""Explicit CPU FP64 arithmetic for reviewed composite reference graphs.

Library modules' numerical forward methods are never used by this interpreter.
Tensor arithmetic, indexing and reductions remain the trusted CPU primitives.
"""

from __future__ import annotations

import itertools
import math
import operator

import torch
import torch.nn.functional as F

from .operator_oracles import _affine, _convolution, _host, _normalize, _pool, _tuple


def _softmax(x, dim, logarithmic=False):
    shifted = x - x.amax(dim, keepdim=True)
    logarithm = shifted - shifted.exp().sum(dim, keepdim=True).log()
    return logarithm if logarithmic else logarithm.exp()


def _activation(name, x, *, inplace=False, **options):
    if name == "relu":
        value = x.clamp_min(0)
    elif name == "relu6":
        value = x.clamp(0, 6)
    elif name == "leaky_relu":
        value = torch.where(x >= 0, x, x * options.get("negative_slope", 0.01))
    elif name == "sigmoid":
        value = torch.exp(-torch.logaddexp(torch.zeros_like(x), -x))
    elif name == "tanh":
        value = x.tanh()
    elif name == "gelu":
        approximate = options.get("approximate", "none")
        if approximate == "none":
            value = x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        elif approximate == "tanh":
            value = x * 0.5 * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())
        else:
            raise ValueError("Unknown GELU approximation")
    elif name == "hardtanh":
        value = x.clamp(options.get("min_val", -1), options.get("max_val", 1))
    elif name == "hardswish":
        value = x * (x + 3).clamp(0, 6) / 6
    elif name in {"softplus", "mish"}:
        beta, threshold = options.get("beta", 1), options.get("threshold", 20)
        softplus = torch.where(x * beta > threshold, x, torch.logaddexp(x * beta, torch.zeros_like(x)) / beta)
        value = x * softplus.tanh() if name == "mish" else softplus
    else:
        raise ValueError(f"No independent activation adapter: {name}")
    if inplace:
        x.copy_(value)
        return x
    return value


def _adaptive_mean(x, output_size):
    rank = x.ndim - 2
    sizes = _tuple(output_size, rank)
    sizes = tuple(x.shape[2 + axis] if size is None else size for axis, size in enumerate(sizes))
    if len(sizes) != rank or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Invalid adaptive pooling dimensions")
    result = torch.empty((*x.shape[:2], *sizes), dtype=torch.float64)
    for position in itertools.product(*(range(size) for size in sizes)):
        windows = tuple(slice(index * x.shape[2 + axis] // sizes[axis],
                              math.ceil((index + 1) * x.shape[2 + axis] / sizes[axis]))
                        for axis, index in enumerate(position))
        result[(slice(None), slice(None), *position)] = x[(slice(None), slice(None), *windows)].mean(tuple(range(2, x.ndim)))
    return result


MODULES = {getattr(torch.nn, name) for name in (
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "GroupNorm", "LayerNorm", "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "ReLU", "ReLU6", "LeakyReLU", "Sigmoid", "Tanh",
    "GELU", "Hardtanh", "Hardswish", "Mish", "Softplus", "Softmax", "LogSoftmax", "Identity", "Dropout")}


def module(layer, args, kwargs, *, dropout):
    if type(layer) not in MODULES or kwargs:
        raise ValueError(f"No reviewed independent module adapter: {type(layer).__name__}")
    x = args[0]
    name = type(layer).__name__
    if name == "Linear":
        result = x @ _host(layer.weight).t()
        return result if layer.bias is None else result + _host(layer.bias)
    if "Conv" in name:
        return _convolution(x, layer)
    if name.startswith("BatchNorm") or name.startswith("InstanceNorm"):
        batch = name.startswith("BatchNorm")
        shape = (1, -1, *([1] * (x.ndim - 2)))
        if layer.training or not layer.track_running_stats:
            axes = (0, *range(2, x.ndim)) if batch else tuple(range(2, x.ndim))
            result = _normalize(x, axes, layer.eps)
        else:
            result = (x - _host(layer.running_mean).reshape(shape)) / (_host(layer.running_var).reshape(shape) + layer.eps).sqrt()
        return _affine(result, layer, shape) if layer.affine else result
    if name == "GroupNorm":
        result = _normalize(x.reshape(x.shape[0], layer.num_groups, -1), (2,), layer.eps).reshape(x.shape)
        return _affine(result, layer, (1, -1, *([1] * (x.ndim - 2)))) if layer.affine else result
    if name == "LayerNorm":
        axes = tuple(range(x.ndim - len(layer.normalized_shape), x.ndim))
        result = _normalize(x, axes, layer.eps)
        return _affine(result, layer, (*([1] * (x.ndim - len(axes))), *layer.normalized_shape)) if layer.elementwise_affine else result
    if name.startswith("AdaptiveAvgPool"):
        return _adaptive_mean(x, layer.output_size)
    if name.startswith(("MaxPool", "AvgPool")):
        return _pool(x, layer, maximum=name.startswith("MaxPool"))
    if name in {"Softmax", "LogSoftmax"}:
        return _softmax(x, layer.dim, logarithmic=name == "LogSoftmax")
    if name == "Identity":
        return x
    if name == "Dropout":
        if not layer.training or layer.p == 0:
            return x
        if layer.p == 1:
            return x.zero_() if layer.inplace else torch.zeros_like(x)
        value = x * dropout(x, layer.p) / (1 - layer.p)
        if layer.inplace:
            x.copy_(value)
            return x
        return value
    names = {"ReLU": "relu", "ReLU6": "relu6", "LeakyReLU": "leaky_relu", "Sigmoid": "sigmoid", "Tanh": "tanh",
             "GELU": "gelu", "Hardtanh": "hardtanh", "Hardswish": "hardswish", "Mish": "mish", "Softplus": "softplus"}
    options = {field: getattr(layer, field) for field in ("inplace", "negative_slope", "approximate", "min_val", "max_val", "beta", "threshold") if hasattr(layer, field)}
    return _activation(names[name], x, **options)


# Use explicit function identities, not a fallback to arbitrary callable targets.
FUNCTIONS = {operator.add: "add", operator.sub: "sub", operator.mul: "mul", operator.truediv: "div",
             operator.floordiv: "floordiv", operator.getitem: "getitem", getattr: "getattr",
             torch.cat: "cat", torch.flatten: "flatten", torch.sum: "sum", torch.mean: "mean", torch.min: "min",
             torch.max: "max", torch.matmul: "matmul", torch.multiply: "mul", torch.clamp: "clamp",
             torch.softmax: "softmax", F.softmax: "functional_softmax", torch.logsumexp: "logsumexp",
             torch.relu: "relu", F.relu: "relu", torch.sigmoid: "sigmoid", torch.tanh: "tanh", F.gelu: "gelu",
             F.mish: "mish", F.hardswish: "hardswish", F.hardtanh: "hardtanh", F.leaky_relu: "leaky_relu",
             F.softplus: "softplus", F.adaptive_avg_pool2d: "adaptive_mean", F.max_pool2d: "max_pool2d",
             F.max_pool1d: "max_pool1d", F.max_pool3d: "max_pool3d", torch.tensor: "tensor"}
METHODS = frozenset(("view", "size", "transpose", "contiguous", "squeeze", "unsqueeze", "mean", "clone", "detach",
                     "unfold", "expand", "permute", "reshape", "flatten"))


def function(target, args, kwargs):
    if target not in FUNCTIONS:
        raise ValueError(f"No independent function adapter: {target}")
    name = FUNCTIONS[target]
    if name == "functional_softmax":
        return _softmax(args[0], kwargs.get("dim", args[1] if len(args) > 1 else None))
    if name == "softmax":
        return _softmax(args[0], kwargs.get("dim", args[1] if len(args) > 1 else None))
    if name == "logsumexp":
        x = args[0]
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        keep = kwargs.get("keepdim", False)
        maximum = x.amax(dim, keepdim=True)
        result = (x - maximum).exp().sum(dim, keepdim=True).log() + maximum
        if keep:
            return result
        dims = tuple(range(x.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
        for axis in sorted((index % x.ndim for index in dims), reverse=True):
            result = result.squeeze(axis)
        return result
    if name in {"relu", "sigmoid", "tanh", "gelu", "mish", "hardswish", "hardtanh", "leaky_relu", "softplus"}:
        options = dict(kwargs)
        positional = {"gelu": ("approximate",), "hardtanh": ("min_val", "max_val", "inplace"),
                      "leaky_relu": ("negative_slope", "inplace"), "softplus": ("beta", "threshold"),
                      "relu": ("inplace",), "hardswish": ("inplace",), "mish": ("inplace",)}
        for field, value in zip(positional.get(name, ()), args[1:]):
            options[field] = value
        return _activation(name, args[0], **options)
    if name == "adaptive_mean":
        return _adaptive_mean(args[0], kwargs.get("output_size", args[1] if len(args) > 1 else None))
    if name.startswith("max_pool"):
        rank = int(name[-2])
        values = dict(kernel_size=args[1] if len(args) > 1 else kwargs.get("kernel_size"), stride=None,
                      padding=0, dilation=1, ceil_mode=False, return_indices=False)
        for field, value in zip(("stride", "padding", "dilation", "ceil_mode", "return_indices"), args[2:]):
            values[field] = value
        values.update(kwargs)
        layer = getattr(torch.nn, f"MaxPool{rank}d")(**values)
        return _pool(args[0], layer, maximum=True)
    if name == "tensor":
        if "dtype" not in kwargs:
            raise ValueError("A graph tensor literal needs its observed native storage dtype")
        return _host(torch.tensor(*args, **{**kwargs, "device": "cpu"}))
    if name == "getattr":
        if args[1] not in {"shape", "ndim", "dtype", "device", "T", "mT", "values", "indices"}:
            raise ValueError("Independent graph accessed an unreviewed attribute")
        if args[1] in {"values", "indices"} and not isinstance(args[0], tuple):
            raise ValueError("Named reduction fields require a reviewed reduction tuple")
        return getattr(*args)
    if name in {"sum", "mean", "min", "max", "cat", "flatten", "matmul", "clamp"}:
        return getattr(torch, name)(*args, **kwargs)
    return {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
            "floordiv": operator.floordiv, "getitem": operator.getitem}[name](*args, **kwargs)


def method(name, args, kwargs):
    if name not in METHODS:
        raise ValueError(f"No independent tensor-method adapter: {name}")
    return getattr(args[0], name)(*args[1:], **kwargs)
