"""Reviewed L1 correctness-only inputs, using the reference's actual parameters."""

from __future__ import annotations

import math

PROTOCOL = "kernelbench_pro_operator_cases_v1"
NONFINITE_TASKS = (19, 20, 21, 22, 28)
LABELS = ("odd", "alternative_shape", "strided", "near_constant", "dynamic_range", "boundary")


def labels(pid):
    if type(pid) is not int or not 1 <= pid <= 100:
        raise ValueError("Reviewed operator cases cover original L1 only")
    result = list(LABELS)
    if pid in {41, 42, 43, 49, 51, 52, 53}:
        result.append("ties")
    if pid in {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 16, 17, 18, 47, 48}:
        result.append("cancellation")
    if pid in NONFINITE_TASKS:
        result.extend(("nonfinite_mixed", "nonfinite_only"))
    if 19 <= pid <= 32 or pid == 88:
        result.extend(("activation_extremes", "format_extremes"))
    if pid in {26, 88}:
        result.append("gelu_discriminator")
    if pid in {41, 42, 43, 49, 51, 52, 53}:
        result.append("isolated_extrema")
    result.append("storage_offset")
    if pid in {*range(1, 19), 94, 96, 97, 98, 99} - {5}:
        result.append("shared_storage")
    return tuple(result)


def shapes(pid, model, *, alternate=False):
    """No constructor replacement, nominal get_inputs call or device allocation."""
    labels(pid)
    n, m, k, width = (31, 5, 11, 19) if alternate else (17, 3, 7, 13)
    if pid in {1, 13, 14, 15}:
        return [(n, n), (n, n)]
    if pid in {2, 8, 9}:
        return [(n, k), (k, n if pid == 9 else width)]
    if pid == 6:
        return [(m, 257 if not alternate else 513), (257 if not alternate else 513, width)]
    if pid == 7:
        return [(n, m), (m, width)]
    if pid == 3:
        return [(m, n, k), (m, k, width)]
    if pid == 4:
        return [(n, k), (k, 1)]
    if pid == 5:
        return [(n, k), None]
    if pid in {10, 11}:
        return [(*((m, n) if pid == 10 else (2, m, n)), k), (k, width)]
    if pid == 12:
        return [(k,), (k, width)]
    if pid in {16, 17, 18}:
        return [(k, n) if pid != 17 else (n, k), (width, k) if pid != 16 else (k, width)]
    if 19 <= pid <= 32 or pid == 88:
        return [(2, 7, 5) if alternate else (3, 17)]
    if pid in {33, 34, 35, 36}:
        layer = getattr(model, {33: "bn", 34: "inorm", 35: "gn"}.get(pid, "num_features"))
        channels = layer if pid == 36 else layer.num_channels if pid == 35 else layer.num_features
        return [(1 if alternate else 3, channels, 5 if alternate else 3, 7 if alternate else 5)]
    if pid in {37, 38, 39}:
        return [(m, n)]
    if pid == 40:
        return [(1 if alternate else 3, *model.ln.normalized_shape)]
    if 41 <= pid <= 46:
        rank = (pid - 41) % 3 + 1
        layer = getattr(model, "maxpool" if pid < 44 else "avg_pool")
        def tup(value):
            return (value,) * rank if isinstance(value, int) else tuple(value)
        kernel, stride, dilation = tup(layer.kernel_size), tup(layer.stride), tup(getattr(layer, "dilation", 1))
        spatial = [d * (size - 1) + 1 + 2 * step + 2 * axis + (5 if alternate else 1)
                   for axis, (size, step, d) in enumerate(zip(kernel, stride, dilation))]
        spatial = [size if size % 2 else size + 1 for size in spatial]
        return [(1 if alternate else 3, 3, *spatial)]
    if pid == 50 or 54 <= pid <= 87:
        import torch

        layer = next(child for child in model.modules() if isinstance(child, torch.nn.modules.conv._ConvNd))
        rank = len(layer.kernel_size)
        spatial = []
        for axis in range(rank):
            effective = layer.dilation[axis] * (layer.kernel_size[axis] - 1) + 1
            if layer.transposed:
                minimum = max(1, math.ceil((2 * layer.padding[axis] - effective - layer.output_padding[axis] + 1) / layer.stride[axis]) + 1)
            else:
                minimum = max(1, effective - 2 * layer.padding[axis])
            size = max(7, minimum + 2 * layer.stride[axis]) + (4 if alternate else 0)
            spatial.append(size if size % 2 else size + 1)
        # Review the frozen source's square-input versus asymmetric-input family.
        asymmetric = pid in {57, 59, 61, 62, 64, 68, 69, 70, 73, 74, 75, 78, 79, 81, 84, 85}
        spatial = [size + 2 * axis for axis, size in enumerate(spatial)] if asymmetric else [max(spatial)] * rank
        return [(1 if alternate else 3, layer.in_channels, *spatial)]
    if pid in {47, 48, 49, 51, 52, 53} or 89 <= pid <= 93:
        axis = model.dim
        rank = max(3, axis + 1 if axis >= 0 else -axis)
        shape = [3] * rank
        shape[axis % rank] = 31 if alternate else 17
        shape[-1] = 7 if alternate and axis % rank != rank - 1 else shape[-1]
        return [tuple(shape)] * (2 if pid == 93 else 1)
    if pid in {94, 96, 98}:
        return [(m, n), (m, n)]
    if pid == 95:
        return [(m, n), (m,)]
    if pid == 97:
        return [(1, 1, 1, 7), (1, 1, 17, 7), (1, 1, 17, 3)] if alternate else [(2, 3, 7, 5), (2, 3, 11, 5), (2, 3, 11, 9)]
    if pid == 99:
        return [(m, n)] * 3
    if pid == 100:
        # Retain the frozen source's 1-D target broadcasting, including its
        # square prediction/label coupling. Do not silently repair the formula.
        return [(n, n), (n,)]
    raise ValueError("No reviewed input shape plan")


def _strided(value):
    import torch

    backing = torch.empty((*value.shape[:-1], 2 * value.shape[-1]), dtype=value.dtype, device=value.device)
    result = backing[..., ::2]
    return result.copy_(value)


def cases(pid, model, *, seed, memory_mb=256, allocation_device="cpu"):
    """Yield cases lazily. The allocation device is injectable for meta-only tests."""
    import torch

    from .input_layouts import apply as alias_inputs
    from .input_layouts import estimate as storage_estimate

    if type(seed) is not int or type(memory_mb) is not int or not 1 <= memory_mb <= 1048576:
        raise ValueError("Operator cases require an integer seed and a positive memory envelope")
    if allocation_device not in {"cpu", "meta"}:
        raise ValueError("Reviewed operator values are generated on CPU; meta is reserved for software checks")
    generator = torch.Generator().manual_seed(seed)
    for label in labels(pid):
        dimensions = shapes(pid, model, alternate=label == "alternative_shape")
        # Includes generation temporaries and strided backing. This is a
        # conservative input-storage estimate, not an OS/process memory limit.
        estimate = storage_estimate(label, dimensions)
        if estimate > memory_mb * 1024**2:
            raise ValueError("Reviewed original-parameter input case exceeds its memory envelope; no resizing")
        values = []
        for index, shape in enumerate(dimensions):
            if shape is None:
                values.append(-1.75 if label == "alternative_shape" else 0.0 if label == "boundary" else 3.14)
            elif pid == 93 and index == 1:
                values.append(torch.rand(shape, generator=generator, device=allocation_device, dtype=torch.float32) > 0.5)
            elif pid == 95 and index == 1:
                values.append(torch.randint(dimensions[0][1], shape, generator=generator, device=allocation_device))
            elif pid == 100 and index == 1:
                values.append((torch.randint(0, 2, shape, generator=generator, device=allocation_device) * 2 - 1).float())
            else:
                values.append(torch.randn(shape, generator=generator, device=allocation_device, dtype=torch.float32))
        for index, value in enumerate(values):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or pid == 100 and index == 1:
                continue
            if label == "near_constant":
                value.fill_(1.0)
                value.reshape(-1)[1::2] += 1 / 128
            elif label == "dynamic_range":
                value.reshape(-1)[::2] *= 8
                value.reshape(-1)[1::2] /= 128
            elif label == "ties":
                value.fill_(1)
            elif label == "cancellation":
                value.fill_(1)
                if index == 0:
                    axis = model.dim if pid in {47, 48} else 0 if pid in {16, 18} else value.ndim - 1
                    view = value.movedim(axis, -1)
                    view[..., ::2] = 8
                    view[..., 1::2] = -8
                    if view.shape[-1] % 2:
                        view[..., -1] = 1 / 1024
            elif label == "boundary":
                value.zero_()
                value.reshape(-1)[-1] = -8 if pid == 52 else 8
                if pid in {38, 39}:
                    value[:, 0] = 1
            elif label == "nonfinite_mixed":
                value.reshape(-1)[::5] = float("nan")
                value.reshape(-1)[1::5] = float("inf")
                value.reshape(-1)[2::5] = -float("inf")
            elif label == "nonfinite_only":
                value.fill_(float("nan"))
            elif label in {"activation_extremes", "format_extremes"}:
                amplitude = 32 if label == "activation_extremes" else 32752
                value.fill_(-amplitude)
                value.reshape(-1)[1::2] = amplitude
            elif label == "gelu_discriminator":
                value.fill_(-2.699)
                value.reshape(-1)[1::2] = 2.699
            elif label == "isolated_extrema":
                value.fill_(64 if pid in {52, 53} else -64)
                # Odd spatial tails may lie outside a floor-pooled window;
                # the first border is in its first declared window.
                value.reshape(-1)[0 if pid in {41, 42, 43} else -1] = 32 if pid in {52, 53} else -32
        if pid == 13:
            values = [(value + value.t()) / 2 for value in values]
        elif pid in {14, 15}:
            values = [value.triu() if pid == 14 else value.tril() for value in values]
        elif pid == 90:
            # Exactly represented signed dyadic prefixes keep every long
            # product informative; no nominal input profile is consulted.
            shape, axis = values[0].shape, model.dim
            exponent = torch.randint(-3, 4, shape, generator=generator, device=allocation_device)
            previous = exponent.roll(1, axis)
            previous.select(axis, 0).zero_()
            sign = torch.randint(0, 2, shape, generator=generator, device=allocation_device) * 2 - 1
            previous_sign = sign.roll(1, axis)
            previous_sign.select(axis, 0).fill_(1)
            values[0].copy_(torch.pow(2.0, (exponent - previous).float()) * sign * previous_sign)
        elif pid == 93 and label == "boundary":
            values[1][0].zero_()
            values[1][-1].fill_(True)
        elif pid == 95 and label == "boundary":
            values[1][0] = -100
            values[1][-1] = dimensions[0][1] - 1
        elif pid == 98:
            # Positive predictions, normalized nonnegative target; zero target
            # entries are legal and must not be turned into NaN by log(target).
            values = [value.softmax(-1) for value in values]
            if label == "boundary":
                values[1].zero_()
                values[1][:, -1] = 1
        elif pid == 99:
            anchor = values[0]
            if label == "boundary":
                values = [anchor, anchor.clone(), anchor + 2]
            else:
                values = [anchor, anchor + 2, anchor + values[2] / 128]
        if label == "strided":
            values = [_strided(value) if isinstance(value, torch.Tensor) else value for value in values]
        values = alias_inputs(values, label, generator=generator, allocation_device=allocation_device)
        yield {"protocol": PROTOCOL, "label": label, "inputs": values, "estimated_input_bytes": estimate,
               "nominal_timing_changed": False, "reference_parameters_replaced": False}
