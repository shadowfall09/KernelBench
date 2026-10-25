"""Frozen, source-bound L2/L3 input roles and legal shape families.

The manifest records metadata of the original input factories and reviewed
couplings. It contains no replacement model, numerical oracle or timing case.
Import and receipt replay do not require a tensor SDK.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common import fingerprint
from .input_layouts import LABELS as LAYOUT_LABELS
from .input_layouts import estimate
from .operator_cases import PROTOCOL

MANIFEST = Path(__file__).with_name("model_input_families.json")
MANIFEST_PROTOCOL = "kernelbench_pro_model_input_families_v1"
VALUE_LABELS = ("strided", "near_constant", "dynamic_range", "boundary")


def reviewed_spec(level, pid, source):
    document = json.loads(MANIFEST.read_text())
    row = document.get("tasks", {}).get(f"{level}/{pid}")
    if document.get("protocol") != MANIFEST_PROTOCOL or not isinstance(row, dict):
        raise ValueError("No reviewed model input-family contract")
    if row.get("reference_sha256") != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Model input-family reference changed; review its roles and coupled dimensions")
    nominal = row.get("nominal_shapes")
    variants = row.get("shape_variants")
    fixed = row.get("fixed_shape_reason")
    if (not isinstance(nominal, list) or not nominal or not isinstance(variants, dict)
            or list(variants) != (["fixed_shape"] if fixed else ["odd", "alternative_shape"])
            or not isinstance(row.get("shape_constraints"), str) or not row["shape_constraints"]
            or fixed is not None and (not isinstance(fixed, str) or not fixed)):
        raise ValueError("Invalid reviewed model input roles or shape constraints")
    for shapes in [nominal, *variants.values()]:
        if (not isinstance(shapes, list) or len(shapes) != len(nominal)
                or any(not isinstance(shape, list) or not shape
                       or any(type(n) is not int or n <= 0 for n in shape) for shape in shapes)
                or any(len(shape) != len(original) for shape, original in zip(shapes, nominal))):
            raise ValueError("Invalid reviewed model input shape family")
    if fixed and variants["fixed_shape"] != nominal:
        raise ValueError("A fixed-shape contract cannot replace nominal dimensions")
    if not fixed and variants["odd"] == variants["alternative_shape"]:
        raise ValueError("Variable model input families require distinct legal shapes")
    stress = row.get("stress_policy")
    if (not isinstance(stress, dict) or stress.get("family") not in {"convolution_fusion", "linear_fusion", "vision_or_mlp", "recurrent", "attention", "descriptor", "state_space"}
            or not isinstance(stress.get("feature_axes"), list) or len(stress["feature_axes"]) != len(nominal)
            or any(type(axis) is not int or not 0 <= axis < len(shape) for axis, shape in zip(stress["feature_axes"], nominal))
            or type(stress.get("activation_amplitude")) is not int or stress["activation_amplitude"] != 32):
        raise ValueError("Invalid reviewed model stress roles or amplitudes")
    main = stress.get("direct_main")
    if main is not None and (not isinstance(main, dict) or main.get("kind") not in {"linear", "convolution", "recurrent", "recurrent_cell"}
            or not isinstance(main.get("attribute"), str) or not main["attribute"]
            or type(main.get("flatten", False)) is not bool):
        raise ValueError("Invalid reviewed direct cancellation module")
    required = ["signed_feature_pairs", "activation_extremes", "isolated_extrema", "coordinate_asymmetry", "storage_offset"]
    if len(nominal) > 1:
        required.append("shared_storage")
    if main:
        required.append("weight_cancellation")
    if stress.get("labels") != required:
        raise ValueError("Incomplete reviewed model stress labels")
    return row


def labels(spec):
    return (*spec["shape_variants"], *VALUE_LABELS, *spec["stress_policy"]["labels"])


def shapes(spec, label):
    if label not in labels(spec):
        raise ValueError("Unknown reviewed model input family")
    return spec["shape_variants"][label if label in spec["shape_variants"] else next(iter(spec["shape_variants"]))]


def plan(level, pid, source, memory_mb):
    spec = reviewed_spec(level, pid, source)
    return {"protocol": PROTOCOL, "reference_sha256": spec["reference_sha256"],
            "family": "composite" if level == 2 else "model_block", "labels": list(labels(spec)),
            "memory_mb": memory_mb, "nonfinite_domain_extension": False,
            "model_family_sha256": fingerprint(spec), "fixed_shape_reason": spec["fixed_shape_reason"],
            "scope": "additional_hidden_correctness_only; actual_reference_initialization; nominal_timing_unchanged"}


def cases(spec, *, seed, memory_mb=256, allocation_device="cpu", reference=None):
    """Generate lazily after admission; never resize or select by output quality."""
    import torch

    from .operator_cases import _strided

    if (type(seed) is not int or seed < 0 or type(memory_mb) is not int or not 1 <= memory_mb <= 1048576
            or allocation_device not in {"cpu", "meta"}):
        raise ValueError("Model input families require a nonnegative seed, memory envelope and CPU/meta generation")
    generator = torch.Generator().manual_seed(seed)
    for label in labels(spec):
        dimensions = shapes(spec, label)
        storage = estimate(label, dimensions)
        if storage > memory_mb * 1024**2:
            raise ValueError("Reviewed original-parameter input case exceeds its memory envelope; no resizing")
        values = [torch.randn(shape, generator=generator, dtype=torch.float32, device=allocation_device)
                  for shape in dimensions]
        for value in values:
            if label == "near_constant":
                value.fill_(1)
                value.reshape(-1)[1::2] += 1 / 128
            elif label == "dynamic_range":
                value.reshape(-1)[::2] *= 8
                value.reshape(-1)[1::2] /= 128
            elif label == "boundary":
                value.zero_()
                value.reshape(-1)[-1] = 8
        if label == "strided":
            values = [_strided(value) for value in values]
        from .input_layouts import apply as alias_inputs
        from .model_stress import apply as stress_inputs

        details = stress_inputs(values, label, spec, reference=reference, allocation_device=allocation_device)
        if label in LAYOUT_LABELS:
            values = alias_inputs(values, label, generator=generator, allocation_device=allocation_device)
        yield {"label": label, "inputs": values, "estimated_input_bytes": storage, "stress_details": details,
               "reference_parameters_replaced": False, "nominal_timing_changed": False}


def signature_matches(spec, label, signature, precision):
    """Replay geometry and native precision independently of execution receipts."""
    dtype = {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}.get(precision)
    return (dtype is not None and len(signature) == len(shapes(spec, label))
            and all(item.get("shape") == shape and item.get("dtype") == dtype
                    for item, shape in zip(signature, shapes(spec, label))))
