"""Source-bound nominal value portfolios for original L2/L3; SDK-light."""

from __future__ import annotations

import json

from .model_input_cases import MANIFEST, MANIFEST_PROTOCOL, reviewed_spec


def policy(level, pid, source=None):
    if level not in {2, 3} or type(pid) is not int or not 1 <= pid <= (100 if level == 2 else 50):
        raise ValueError("Model nominal profiles require an original L2/L3 task")
    if source is not None:
        spec = reviewed_spec(level, pid, source)
    else:
        document = json.loads(MANIFEST.read_text())
        spec = document.get("tasks", {}).get(f"{level}/{pid}")
        if document.get("protocol") != MANIFEST_PROTOCOL or not isinstance(spec, dict):
            raise ValueError("No reviewed nominal model input roles")
    family = spec["stress_policy"]["family"]
    menus = {
        "convolution_fusion": ("identity", "normal_unit", "symmetric_gain4", "positive_gain4", "symmetric_gain16"),
        "linear_fusion": ("identity", "normal_unit", "symmetric_gain4", "positive_gain4", "symmetric_gain16"),
        "vision_or_mlp": ("identity", "normal_unit", "symmetric_gain4", "positive_gain4", "symmetric_gain16"),
        "recurrent": ("identity", "normal_small", "normal_unit", "symmetric_gain4"),
        "attention": ("identity", "normal_small", "normal_unit", "symmetric_gain4"),
        "descriptor": ("identity", "normal_unit", "symmetric_gain4"),
        "state_space": ("identity", "normal_small", "symmetric_unit", "positive_small"),
    }
    if family not in menus:
        raise ValueError("No reviewed nominal profile for this model family")
    return {"family": family, "reference_sha256": spec["reference_sha256"], "arguments": len(spec["nominal_shapes"]),
            "recipes": list(menus[family]), "minimum_fraction": .1 if level == 2 else .4 if family in {"vision_or_mlp", "attention"} else .9,
            "roles": "finite_floating_features_and_explicit_initial_states; internal_masks_parameters_and_implicit_states_unchanged"}


def transform(inputs, level, pid, recipe):
    import torch

    from src.input_groups import copy_inputs

    spec = policy(level, pid)
    if (recipe not in spec["recipes"] or not isinstance(inputs, list) or len(inputs) != spec["arguments"]
            or any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in inputs)):
        raise ValueError("Model nominal profile changed reviewed floating input roles")

    def draw(value):
        if recipe == "identity":
            return value
        sampled = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
        if recipe.startswith("normal_"):
            return sampled.normal_(0, .125 if recipe == "normal_small" else 1.)
        low, high = {"symmetric_unit": (-1., 1.), "symmetric_gain4": (-4., 4.),
                     "symmetric_gain16": (-16., 16.), "positive_gain4": (.5, 8.), "positive_small": (.0625, .25)}[recipe]
        return sampled.uniform_(low, high)

    # A shared backing is sampled once; views keep offsets, strides and aliases.
    # The old factory is still called for its declared shapes and RNG consumption.
    return copy_inputs(inputs, draw)
