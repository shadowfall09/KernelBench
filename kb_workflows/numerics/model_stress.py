"""Value-only model counterexamples with optional actual-weight cancellation."""

from __future__ import annotations

import hashlib
import itertools


def apply(values, label, spec, *, reference, allocation_device):
    import torch

    policy = spec["stress_policy"]
    if label in {"signed_feature_pairs", "activation_extremes", "isolated_extrema", "coordinate_asymmetry"}:
        for index, value in enumerate(values):
            axis = policy["feature_axes"][index]
            if label == "signed_feature_pairs":
                value.fill_(0)
                view = value.movedim(axis, -1)
                view[..., ::2] = 32
                view[..., 1::2] = -32
                if view.shape[-1] % 2:
                    view[..., -1] = 2**-10
            elif label == "activation_extremes":
                value.fill_(-policy["activation_amplitude"])
                value.movedim(axis, -1)[..., ::2] = policy["activation_amplitude"]
            elif label == "isolated_extrema":
                value.fill_(-1)
                value.reshape(-1)[0] = -64
                value.reshape(-1)[-1] = 64
            else:
                value.zero_()
                for dim, size in enumerate(value.shape):
                    shape = [1] * value.ndim
                    shape[dim] = size
                    ramp = torch.arange(size, dtype=value.dtype, device=allocation_device).reshape(shape)
                    value.add_((ramp.remainder(17) - 8) * (dim + 1) / 32)
        return {"value_recipe": label, "reference_parameters_replaced": False}
    if label != "weight_cancellation":
        return {}
    main = policy.get("direct_main")
    if main is None or reference is None and allocation_device != "meta":
        raise ValueError("Weight cancellation needs the reviewed actual reference module")
    layer = reference.get_submodule(main["attribute"]) if reference is not None else None
    primary = values[0]
    for value in values:
        value.zero_()
    linear = main["kind"] in {"linear", "recurrent", "recurrent_cell"}
    if linear:
        target = primary.reshape(primary.shape[0], -1) if main.get("flatten") else primary
        locations = [(index,) for index in range(min(3, target.shape[-1]))]
        coefficient_locations = [(0, index) for index in range(len(locations))]
    else:
        if layer is None:
            # Shape-only planning never reads model weights or invents values.
            return {"value_recipe": label, "coefficient_source": "meta_unmeasured", "reference_parameters_replaced": False}
        locations, coefficient_locations = [], []
        for channel in range(layer.weight.shape[1]):
            for kernel in itertools.product(*(range(size) for size in layer.kernel_size)):
                position = tuple(k * dilation - padding for k, dilation, padding in zip(kernel, layer.dilation, layer.padding))
                if all(0 <= point < size for point, size in zip(position, primary.shape[2:])):
                    locations.append((channel, *position))
                    coefficient_locations.append((0, channel, *kernel))
                    if len(locations) == 3:
                        break
            if len(locations) == 3:
                break
        target = primary
    if len(locations) < 2:
        raise ValueError("Reviewed cancellation geometry has fewer than two actual products")
    if allocation_device == "meta":
        coefficient = torch.empty(len(locations), device="meta")
        receipt = {"coefficient_source": "meta_unmeasured"}
    else:
        weight = (layer.weight_ih_l0 if main["kind"] == "recurrent" else layer.weight_ih
                  if main["kind"] == "recurrent_cell" else layer.weight)
        coefficient = torch.stack([weight.detach()[position] for position in coefficient_locations]).to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(coefficient).all():
            raise ValueError("Reference cancellation coefficients are not finite")
        if abs(float(coefficient[0])) > abs(float(coefficient[1])):
            order = [1, 0, *range(2, len(locations))]
            coefficient = coefficient[order]
            locations = [locations[index] for index in order]
            coefficient_locations = [coefficient_locations[index] for index in order]
        receipt = {"coefficient_source": "actual_reference_weight", "coefficient_sha256": hashlib.sha256(coefficient.numpy().tobytes()).hexdigest(),
                   "two_nonzero_products": bool((coefficient[:2] != 0).all())}
    if allocation_device != "meta" and coefficient[1] == 0:
        first, second = 0., 0.
    else:
        first, second = 32., -coefficient[0] / coefficient[1] * 32
    if linear:
        target[..., locations[0][0]] = first
        target[..., locations[1][0]] = second
        if len(locations) > 2:
            target[..., 2] = 2**-12
    else:
        target[(slice(None), *locations[0])] = first
        target[(slice(None), *locations[1])] = second
        if len(locations) > 2:
            target[(slice(None), *locations[2])] = 2**-12
    return {"value_recipe": label, "reference_parameters_replaced": False,
            "construction": "bounded_ratio_with_larger_denominator; FP32_rounding_retained",
            "weight_positions": [list(item) for item in coefficient_locations],
            "input_positions": [list(item) for item in locations], **receipt}
