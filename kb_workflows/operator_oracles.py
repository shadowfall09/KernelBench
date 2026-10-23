"""Independent CPU FP64 formulas for the frozen original L1 operators.

No reference/candidate forward is called here. These are numerical references,
not proofs that the installed CPU tensor primitives are fault-free.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from .numerical import to_oracle


def estimate_bytes(pid, model, inputs):
    """Conservative incremental host-memory estimate; never resize a task.

    This is admission control, not an operating-system RSS guarantee. The
    independent worker and transport still own timeouts and resource isolation.
    """
    x = inputs[0]
    shape = tuple(x.shape)
    padded = 0
    if pid in {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18}:
        a, b = shape, tuple(inputs[1].shape)
        if pid in {16, 18}:
            a = a[::-1]
        if pid in {17, 18}:
            b = b[::-1]
        if len(b) == 1:
            shape = a[:-1]
        else:
            shape = (*torch.broadcast_shapes(a[:-2], b[:-2]), a[-2], b[-1])
    elif pid == 12:
        shape = tuple(inputs[1].shape)
    elif pid in {41, 42, 43, 44, 45, 46}:
        module = model.maxpool if pid < 44 else model.avg_pool
        rank = len(shape) - 2
        kernel, stride, pad = [_tuple(value, rank) for value in (module.kernel_size, module.stride or module.kernel_size, module.padding)]
        dilation = _tuple(module.dilation, rank) if pid < 44 else (1,) * rank
        output = []
        for d in range(rank):
            numerator = shape[2 + d] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1
            ceil = getattr(module, "ceil_mode", False)
            size = (numerator + (stride[d] - 1 if ceil else 0)) // stride[d] + 1
            if ceil and (size - 1) * stride[d] >= shape[2 + d] + pad[d]:
                size -= 1
            output.append(size)
        shape = (*shape[:2], *output)
    elif pid == 50 or 54 <= pid <= 87:
        names = ("depthwise", "pointwise") if pid == 86 else (next(iter(model._modules)),)
        for name in names:
            module = getattr(model, name)
            output = []
            for d, kernel in enumerate(module.kernel_size):
                if module.transposed:
                    size = (shape[2 + d] - 1) * module.stride[d] - 2 * module.padding[d] + module.dilation[d] * (kernel - 1) + module.output_padding[d] + 1
                else:
                    size = (shape[2 + d] + 2 * module.padding[d] - module.dilation[d] * (kernel - 1) - 1) // module.stride[d] + 1
                output.append(size)
            padded += math.prod((*shape[:2], *(shape[2 + d] + 2 * module.padding[d] for d in range(len(output))))) * 8
            shape = (shape[0], module.out_channels, *output)
            padded += math.prod(shape) * 8
    elif pid in {47, 48, 49, 51, 52, 53}:
        shape = tuple(1 if pid == 47 else size for index, size in enumerate(shape) if index != model.dim % len(shape) or pid == 47)
        if pid == 47:
            shape = tuple(1 if index == model.dim % x.ndim else size for index, size in enumerate(x.shape))
    elif 94 <= pid <= 100 and pid != 97:
        shape = ()
    elif pid == 97:
        shape = (*x.shape[:-1], inputs[2].shape[-1])
        padded += math.prod((*x.shape[:-1], inputs[1].shape[-2])) * 8 * 8
    if any(size <= 0 for size in shape):
        raise ValueError("Independent oracle has an empty or illegal output shape")
    inputs_bytes = sum(value.numel() * max(8, value.element_size()) for value in inputs if isinstance(value, torch.Tensor))
    state_bytes = sum(value.numel() * max(8, value.element_size()) for value in [*model.parameters(), *model.buffers()])
    return max(1, 4 * inputs_bytes + 4 * state_bytes + 32 * math.prod(shape) * 8 + 4 * padded)


def _host(value):
    if isinstance(value, torch.Tensor):
        return to_oracle(value.detach().cpu())
    if isinstance(value, dict):
        return {key: _host(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_host(item) for item in value)
    return value


def _affine(value, module, shape):
    if module.weight is not None:
        value = value * _host(module.weight).reshape(shape)
    if module.bias is not None:
        value = value + _host(module.bias).reshape(shape)
    return value


def _normalize(x, axes, eps):
    mean = x.mean(axes, keepdim=True)
    centered = x - mean
    return centered / (centered.square().mean(axes, keepdim=True) + eps).sqrt()


def _convolution(x, module):
    """Direct spatial contractions, independent of nn/functional convolution."""
    weight, bias = _host(module.weight), _host(module.bias)
    rank, groups = x.ndim - 2, module.groups
    stride, pad, dilation = module.stride, module.padding, module.dilation
    kernel = weight.shape[2:]
    if isinstance(pad, str) or module.padding_mode != "zeros":
        raise ValueError("Oracle requires explicit numeric zero padding")
    cin, cout = module.in_channels // groups, module.out_channels // groups
    if not module.transposed:
        output = tuple((x.shape[2 + d] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1) // stride[d] + 1
                       for d in range(rank))
        padded = F.pad(x, tuple(value for width in reversed(pad) for value in (width, width)))
        result = torch.zeros((x.shape[0], groups, cout, *output), dtype=torch.float64)
        shaped_weight = weight.reshape(groups, cout, cin, *kernel)
        for offset in itertools.product(*(range(size) for size in kernel)):
            selection = tuple(slice(offset[d] * dilation[d], offset[d] * dilation[d] + output[d] * stride[d], stride[d])
                              for d in range(rank))
            window = padded[(slice(None), slice(None), *selection)].reshape(x.shape[0], groups, cin, *output)
            result.add_(torch.einsum("goc,bgc...->bgo...", shaped_weight[(slice(None),) * 3 + offset], window))
    else:
        output = tuple((x.shape[2 + d] - 1) * stride[d] - 2 * pad[d] + dilation[d] * (kernel[d] - 1)
                       + module.output_padding[d] + 1 for d in range(rank))
        result = torch.zeros((x.shape[0], groups, cout, *output), dtype=torch.float64)
        shaped_input = x.reshape(x.shape[0], groups, cin, *x.shape[2:])
        shaped_weight = weight.reshape(groups, cin, cout, *kernel)
        for offset in itertools.product(*(range(size) for size in kernel)):
            origin = [offset[d] * dilation[d] - pad[d] for d in range(rank)]
            low = [max(0, -origin[d] // stride[d] + int((-origin[d]) % stride[d] != 0)) for d in range(rank)]
            high = [min(x.shape[2 + d], (output[d] - 1 - origin[d]) // stride[d] + 1) for d in range(rank)]
            if any(a >= b for a, b in zip(low, high)):
                continue
            selection = tuple(slice(low[d], high[d]) for d in range(rank))
            destination = tuple(slice(origin[d] + low[d] * stride[d], origin[d] + (high[d] - 1) * stride[d] + 1, stride[d])
                                for d in range(rank))
            window = shaped_input[(slice(None),) * 3 + selection]
            result[(slice(None),) * 3 + destination].add_(torch.einsum(
                "gic,bgi...->bgc...", shaped_weight[(slice(None),) * 3 + offset], window))
    result = result.reshape(x.shape[0], module.out_channels, *output)
    return result if bias is None else result + bias.reshape(1, -1, *([1] * rank))


def _tuple(value, rank):
    return (value,) * rank if isinstance(value, int) else tuple(value)


def _pool(x, module, *, maximum):
    """Explicit kernel windows and input indices, including ceil-mode edges."""
    rank = x.ndim - 2
    kernel = _tuple(module.kernel_size, rank)
    stride = _tuple(module.stride or module.kernel_size, rank)
    pad = _tuple(module.padding, rank)
    dilation = _tuple(module.dilation, rank) if maximum else (1,) * rank
    ceil = getattr(module, "ceil_mode", False)
    out = []
    for d in range(rank):
        numerator = x.shape[2 + d] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1
        size = (numerator + (stride[d] - 1 if ceil else 0)) // stride[d] + 1
        if ceil and (size - 1) * stride[d] >= x.shape[2 + d] + pad[d]:
            size -= 1
        if size <= 0:
            raise ValueError("Illegal pooling output shape")
        out.append(size)
    result = torch.full((*x.shape[:2], *out), -math.inf if maximum else 0.0, dtype=torch.float64)
    indices = torch.full(result.shape, -1, dtype=torch.int64) if maximum else None
    denominator = torch.zeros(out, dtype=torch.float64) if not maximum else None
    for offset in itertools.product(*(range(size) for size in kernel)):
        coordinates = [torch.arange(out[d]) * stride[d] - pad[d] + offset[d] * dilation[d] for d in range(rank)]
        grid = torch.meshgrid(*coordinates, indexing="ij")
        valid = torch.ones(out, dtype=torch.bool)
        padded_valid = torch.ones(out, dtype=torch.bool)
        linear = torch.zeros(out, dtype=torch.int64)
        for d, position in enumerate(grid):
            valid &= (position >= 0) & (position < x.shape[2 + d])
            padded_valid &= (position >= -pad[d]) & (position < x.shape[2 + d] + pad[d])
            linear = linear * x.shape[2 + d] + position
        clamped = tuple(position.clamp(0, x.shape[2 + d] - 1) for d, position in enumerate(grid))
        value = x[(slice(None), slice(None), *clamped)]
        if maximum:
            take = valid & ((indices < 0) | (value > result) | torch.isnan(value))
            result = torch.where(take, value, result)
            indices = torch.where(take, linear, indices)
        else:
            result.add_(torch.where(valid, value, 0.0))
            denominator.add_(padded_valid if module.count_include_pad else valid)
    if maximum:
        return (result, indices) if module.return_indices else result
    divisor = getattr(module, "divisor_override", None)
    return result / (divisor if divisor is not None else denominator)


def execute(pid, model, inputs):
    """Compute the mathematical task, without rounding the FP64 truth to output."""
    if type(pid) is not int or not 1 <= pid <= 100:
        raise ValueError("No independent L1 operator oracle for this task")
    args = _host(inputs)
    x = args[0]
    if pid in {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13}:
        return torch.matmul(x, args[1])
    if pid == 5:
        return x * args[1]
    if pid == 12:
        return x[:, None] * args[1]
    if pid in {14, 15}:
        value = x @ args[1]
        return value.triu() if pid == 14 else value.tril()
    if pid in {16, 17, 18}:
        return (x.t() if pid in {16, 18} else x) @ (args[1].t() if pid in {17, 18} else args[1])
    if pid == 19:
        return x.clamp_min(0)
    if pid == 20:
        return torch.where(x >= 0, x, x * model.negative_slope)
    if pid in {21, 25}:
        sigmoid = torch.exp(-torch.logaddexp(torch.zeros_like(x), -x))
        return sigmoid if pid == 21 else x * sigmoid
    if pid == 22:
        return x.tanh()
    if pid in {23, 24}:
        dim = 1 if pid == 23 else model.dim
        shifted = x - x.amax(dim, keepdim=True)
        logarithm = shifted - shifted.exp().sum(dim, keepdim=True).log()
        return logarithm.exp() if pid == 23 else logarithm
    if pid == 26:
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    if pid == 27:
        return 1.0507009873554805 * torch.where(x > 0, x, 1.6732632423543772 * torch.expm1(x))
    if pid == 28:
        return ((x + 3) / 6).clamp(0, 1)
    if pid == 29:
        return torch.logaddexp(x, torch.zeros_like(x))
    if pid == 30:
        return x / (1 + x.abs())
    if pid == 31:
        return torch.where(x > 0, x, model.alpha * torch.expm1(x))
    if pid == 32:
        return x.clamp(-1, 1)
    if pid == 33:
        norm = model.bn
        axes = (0, *range(2, x.ndim))
        shape = (1, -1, *([1] * (x.ndim - 2)))
        if norm.training or not norm.track_running_stats:
            value = _normalize(x, axes, norm.eps)
        else:
            value = (x - _host(norm.running_mean).reshape(shape)) / (_host(norm.running_var).reshape(shape) + norm.eps).sqrt()
        return _affine(value, norm, shape) if norm.affine else value
    if pid == 34:
        norm = model.inorm
        shape = (1, -1, *([1] * (x.ndim - 2)))
        if norm.training or not norm.track_running_stats:
            value = _normalize(x, tuple(range(2, x.ndim)), norm.eps)
        else:
            value = (x - _host(norm.running_mean).reshape(shape)) / (_host(norm.running_var).reshape(shape) + norm.eps).sqrt()
        return _affine(value, norm, shape) if norm.affine else value
    if pid == 35:
        norm = model.gn
        value = _normalize(x.reshape(x.shape[0], norm.num_groups, -1), (2,), norm.eps).reshape(x.shape)
        return _affine(value, norm, (1, -1, *([1] * (x.ndim - 2)))) if norm.affine else value
    if pid == 36:
        return x / (x.square().mean(1, keepdim=True) + model.eps).sqrt()
    if pid == 37:
        return x / x.square().sum().sqrt()
    if pid == 38:
        return x / x.abs().mean(1, keepdim=True)
    if pid == 39:
        return x / x.square().sum(1, keepdim=True).sqrt()
    if pid == 40:
        norm = model.ln
        axes = tuple(range(x.ndim - len(norm.normalized_shape), x.ndim))
        value = _normalize(x, axes, norm.eps)
        return _affine(value, norm, (*([1] * (x.ndim - len(axes))), *norm.normalized_shape)) if norm.elementwise_affine else value
    if pid in {41, 42, 43, 44, 45, 46}:
        return _pool(x, model.maxpool if pid < 44 else model.avg_pool, maximum=pid < 44)
    if pid in {47, 48, 49, 51, 52, 53}:
        dim = model.dim
        if pid == 47:
            return x.sum(dim, keepdim=True)
        if pid == 48:
            return x.mean(dim)
        if pid == 49:
            return x.amax(dim)
        if pid == 53:
            return x.amin(dim)
        return x.argmax(dim) if pid == 51 else x.argmin(dim)
    if pid == 50 or 54 <= pid <= 87:
        names = ("depthwise", "pointwise") if pid == 86 else (next(iter(model._modules)),)
        for name in names:
            x = _convolution(x, getattr(model, name))
        return x
    if pid == 88:
        return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())
    if pid in {89, 90, 91, 92, 93}:
        dim = model.dim
        if pid == 90:
            return x.cumprod(dim)
        if pid == 91:
            return x.flip((dim,)).cumsum(dim).flip((dim,))
        if pid == 92:
            # Exclusive means shift on dim, not the source's unconditional axis-0 slice.
            return torch.cat((torch.zeros_like(x.narrow(dim, 0, 1)), x.narrow(dim, 0, x.shape[dim] - 1)), dim).cumsum(dim)
        return (x * args[1] if pid == 93 else x).cumsum(dim)
    if pid in {94, 96}:
        delta = (x - args[1]).abs()
        return delta.square().mean() if pid == 94 else torch.where(delta < 1, 0.5 * delta.square(), delta - 0.5).mean()
    if pid == 95:
        target = args[1]
        values = torch.logsumexp(x, dim=1) - x.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
        return values[target != -100].mean()
    if pid == 97:
        scores = (x @ args[1].transpose(-2, -1)) / math.sqrt(x.shape[-1])
        shifted = scores - scores.amax(-1, keepdim=True)
        weights = shifted.exp()
        return (weights / weights.sum(-1, keepdim=True)) @ args[2]
    if pid == 98:
        target = args[1]
        return (torch.xlogy(target, target) - target * x.log()).sum() / x.shape[0]
    if pid == 99:
        loss = model.loss_fn
        positive = ((x - args[1] + loss.eps).abs().pow(loss.p).sum(1)).pow(1 / loss.p)
        negative = ((x - args[2] + loss.eps).abs().pow(loss.p).sum(1)).pow(1 / loss.p)
        if loss.swap:
            alternate = ((args[1] - args[2] + loss.eps).abs().pow(loss.p).sum(1)).pow(1 / loss.p)
            negative = torch.minimum(negative, alternate)
        values = (positive - negative + loss.margin).clamp_min(0)
        return values.mean() if loss.reduction == "mean" else values.sum() if loss.reduction == "sum" else values
    if pid == 100:
        return (1 - x * args[1]).clamp_min(0).mean()
    raise AssertionError("Missing L1 operator oracle")
