"""Source-bound, value-only cache probe policies; no SDK imports on lookup."""

from __future__ import annotations

import hashlib

PROTOCOL = "kernelbench_pro_cache_input_policy_v1"


def plan(level, pid, source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    mutable = None
    if level in {1, 2, 3}:
        from .operator_input_policy import plan as input_plan

        bound = input_plan(level, pid, source, 256)
        if bound["reference_sha256"] != digest:
            raise ValueError("Cache recipe source differs from its reviewed input domain")
    elif level == 5:
        from .production_specs import task
        from .suites import load_manifest

        row = next((row for row in load_manifest("production41")["problems"] if row["problem_id"] == pid), None)
        if row is None or row["reference_sha256"] != digest:
            raise ValueError("Cache recipe source differs from its production role contract")
        spec = task(pid)
        domain = spec["roles"][0][3]
        if domain not in {"signed", "distinct"}:
            raise ValueError("The first production cache input needs a reviewed value policy")
        mutable = [[index, role[3]] for index, role in enumerate(spec["roles"]) if role[3] in {"signed", "weight", "distinct"}]
        if spec["gradient_indices"]:
            mutable.append([len(spec["roles"]), "signed"])
    else:
        raise ValueError("No reviewed cache recipe for this task level")
    recipe = ({13: "symmetric", 14: "upper_triangular", 15: "lower_triangular", 90: "bounded_product",
               93: "masked_scan", 95: "class_logits", 97: "attention", 98: "probability",
               99: "triplet", 100: "binary_hinge", 5: "matrix_scalar"}.get(pid, "finite_activation")
              if level == 1 else "distinct" if level == 5 and domain == "distinct" else "finite_activation")
    return {"protocol": PROTOCOL, "reference_sha256": digest, "recipe": recipe, "probe_count": 3,
            "production_mutable_roles": mutable,
            "public_values": "trusted_nominal_factory", "hidden_values": "reviewed_value_only_recipe",
            "shape_dtype_parameter_policy": "unchanged; preserve_labels_indices_masks_scales_and_knobs_except_declared_mask_probe"}


def change(inputs, policy, *, probe, reference):
    """Input group is isolated already. Do not touch targets or scale/index roles."""
    import torch

    if probe not in {1, 2} or not isinstance(inputs[0], torch.Tensor) or not inputs[0].is_floating_point():
        raise ValueError("Reviewed cache recipes require a floating primary input")
    value, recipe = inputs[0], policy["recipe"]
    with torch.no_grad():
        if probe == 1:
            if recipe in {"probability", "distinct"}:
                value.copy_(value.roll(1, -1))
            else:
                # Float8 storage need not implement arithmetic directly.
                value.copy_((-value.float()).to(value.dtype))
            return inputs
        if recipe == "probability":
            classes = value.shape[-1]
            if not classes:
                raise ValueError("Probability probe needs nonempty class rows")
            value.fill_(0.5 / classes)
            value[..., -1] = 0.5 + 0.5 / classes
            inputs[1].fill_(1 / classes)
        elif recipe == "distinct":
            value.copy_(value.flip([-1]))
        elif recipe in {"symmetric", "upper_triangular", "lower_triangular"}:
            for index, tensor in enumerate(inputs):
                tensor.zero_()
                tensor.diagonal(dim1=-2, dim2=-1).fill_(index + 1)
        elif recipe == "bounded_product":
            view = value.movedim(reference.dim, -1)
            view[..., ::2] = 0.5
            view[..., 1::2] = 2
        elif recipe == "binary_hinge":
            value.copy_(inputs[1])  # Preserve the frozen 1-D label broadcasting.
            value.mul_(2)
        else:
            work = torch.empty_like(value, dtype=torch.float32)
            work[..., ::2] = 0.5
            work[..., 1::2] = 1
            value.copy_(work.to(value.dtype))
            if recipe == "matrix_scalar":
                inputs[1] = 1.0
            elif recipe == "masked_scan":
                inputs[1].logical_not_()
            elif recipe == "triplet":
                inputs[1].copy_(value).add_(2)
                inputs[2].copy_(value).add_(0.125)
            elif recipe == "attention":
                inputs[2].fill_(0.75)
            elif recipe == "class_logits":
                value[..., 1::2] = -0.5
            if policy["production_mutable_roles"] is not None:
                for index, domain in policy["production_mutable_roles"]:
                    if index:
                        tensor = inputs[index]
                        constant = (1.0 if str(tensor.dtype).startswith("torch.float8_") else
                                    0.5 / max(1, tensor.shape[-1]) ** 0.5) if domain == "weight" else 0.25 * (index + 1)
                        tensor.fill_(constant)
            elif recipe in {"finite_activation", "matrix_scalar"}:
                for tensor in inputs[1:]:
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                        tensor.fill_(0.5)
    return inputs
