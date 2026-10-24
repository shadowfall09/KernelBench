"""SDK-light replay of reviewed L3 architecture and independent-math receipts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .common import fingerprint

PROTOCOL = "kernelbench_pro_model_oracle_v1"
PROGRAMS = Path(__file__).with_name("model_programs.json")


def program_for(spec):
    document = json.loads(PROGRAMS.read_text())
    program = document.get("programs", {}).get(spec.get("program_sha256"))
    if (document.get("protocol") != "kernelbench_pro_model_programs_v1" or not isinstance(program, dict)
            or program.get("protocol") != PROTOCOL or spec.get("model_protocol") != PROTOCOL
            or fingerprint(program) != spec.get("program_sha256") or program.get("problem_id") != spec.get("problem_id")
            or not isinstance(program.get("architecture", {}).get("modules"), dict)
            or len(program["architecture"]["modules"]) != spec.get("program_modules")
            or not isinstance(program.get("dropout_order"), list)):
        raise ValueError("Independent model architecture/arithmetic plan is missing or changed")
    return program


def receipt_valid(receipt, spec, *, precision, source_device):
    program = program_for(spec)
    if (precision not in {"fp32", "fp16", "bf16"} or not isinstance(receipt, dict)
            or receipt.get("protocol") != PROTOCOL or receipt.get("program_sha256") != spec["program_sha256"]
            or type(receipt.get("problem_id")) is not int or receipt["problem_id"] != spec["problem_id"]
            or receipt.get("algorithm") != program["algorithm"] or receipt.get("reference_numerical_forward_called") is not False
            or receipt.get("shape_propagation_device") != "meta"):
        return False
    inner = receipt.get("inner_receipt")
    if program["algorithm"] == "reviewed_fx_layers":
        if (not isinstance(receipt.get("dropout_states"), list)
                or not isinstance(inner, dict) or inner.get("protocol") != "kernelbench_pro_composite_oracle_v1"
                or inner.get("program_sha256") != program["fx_program_sha256"]
                or type(inner.get("node_count")) is not int or inner["node_count"] != program["fx_node_count"]
                or inner.get("reference_numerical_forward_called") is not False
                or inner.get("source_python_symbolically_traced") is not True or inner.get("shape_propagation_device") != "meta"):
            return False
        if any(not isinstance(inner.get(key), list) or any(not isinstance(row, dict) for row in inner[key])
               for key in ("masks", "dropout_states")):
            return False
        paths = program["fx_dropout_paths"]
        if any(row.get("node") not in paths for key in ("masks", "dropout_states") for row in inner[key]):
            return False
        if [{**row, "node": paths[row["node"]]} for row in inner["masks"]] != receipt.get("masks"):
            return False
        if ([{**row, "node": paths[row["node"]]} for row in inner["dropout_states"]]
                != [row for row in receipt.get("dropout_states", []) if isinstance(row, dict) and row.get("node") in paths.values()]):
            return False
    elif inner is not None:
        return False
    states, masks, events = receipt.get("dropout_states"), receipt.get("masks"), receipt.get("meta_dropout_calls")
    if (any(not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) for rows in (states, masks, events))
            or [row.get("node") for row in states] != program["dropout_order"]
            or [row.get("node") for row in events] != program["dropout_order"]):
        return False
    dtype = {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}[precision]
    expected = []
    for state, event in zip(states, events):
        layer = program["architecture"]["modules"][state["node"]]
        policy = layer["attributes"]
        shape, stride = event.get("shape"), event.get("input_stride")
        if (layer["class"] != "Dropout" or type(state.get("training")) is not bool
                or type(state.get("probability")) not in (int, float) or state.get("probability") != policy["p"]
                or state.get("inplace") is not policy["inplace"]
                or type(state.get("native_mask_replayed")) is not bool
                or state["native_mask_replayed"] != (state["training"] and 0 < policy["p"] < 1)
                or type(event.get("training")) is not bool or type(event.get("inplace")) is not bool
                or type(event.get("probability")) not in (int, float)
                or any(event.get(key) != state[key] for key in ("training", "probability", "inplace"))
                or not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(stride, list) or len(stride) != len(shape) or any(type(n) is not int or n < 0 for n in stride)
                or event.get("dtype") != dtype):
            return False
        if state["native_mask_replayed"]:
            expected.append(event)
    if len(masks) != len(expected):
        return False
    for mask, event in zip(masks, expected):
        if (any(mask.get(key) != event[key] for key in ("node", "shape", "input_stride", "dtype", "probability"))
                or mask.get("device") != source_device or type(mask.get("elements")) is not int
                or mask["elements"] != math.prod(event["shape"]) or type(mask.get("kept")) is not int
                or not 0 <= mask["kept"] <= mask["elements"] or not isinstance(mask.get("mask_sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", mask["mask_sha256"])
                or mask.get("scope") != "native_random_mask_only; independent_FP64_arithmetic"):
            return False
    return True
