"""Replayable composite arithmetic plans; importing this module loads no SDK."""

import json
import math
import re
from pathlib import Path

from .common import fingerprint

PROTOCOL = "kernelbench_pro_composite_oracle_v1"
PROGRAMS = Path(__file__).with_name("composite_programs.json")


def program_for(spec):
    document = json.loads(PROGRAMS.read_text())
    if document.get("protocol") != "kernelbench_pro_composite_programs_v1":
        raise ValueError("Unknown composite program registry")
    program = document["programs"].get(spec["program_sha256"])
    if (not isinstance(program, dict) or fingerprint(program) != spec["program_sha256"]
            or program.get("protocol") != PROTOCOL or spec.get("composite_protocol") != PROTOCOL
            or not isinstance(program.get("nodes"), list) or len(program["nodes"]) != spec["program_nodes"]):
        raise ValueError("Composite arithmetic plan is missing or changed")
    return program


def receipt_valid(receipt, spec, *, precision, source_device):
    program = program_for(spec)
    if precision not in {"fp32", "fp16", "bf16"}:
        return False
    if (not isinstance(receipt, dict) or receipt.get("protocol") != PROTOCOL
            or receipt.get("program_sha256") != spec["program_sha256"] or receipt.get("node_count") != spec["program_nodes"]
            or receipt.get("reference_numerical_forward_called") is not False
            or receipt.get("source_python_symbolically_traced") is not True or receipt.get("shape_propagation_device") != "meta"):
        return False
    stochastic = [row for row in program["nodes"] if row["op"] == "call_module" and row["target"].startswith("Dropout:")]
    if [row["name"] for row in stochastic] != spec["stochastic_nodes"]:
        return False
    states, masks = receipt.get("dropout_states"), receipt.get("masks")
    if (not isinstance(states, list) or not isinstance(masks, list)
            or any(not isinstance(row, dict) for row in states + masks)
            or [row.get("node") for row in states] != spec["stochastic_nodes"]):
        return False
    for state, node in zip(states, stochastic):
        contract = program["modules"][node["target"].split(":", 1)[1]]["attributes"]
        p = contract["p"]
        if (type(state.get("training")) is not bool or state.get("probability") != p
                or type(state.get("inplace")) is not bool or state.get("inplace") != contract["inplace"]
                or type(state.get("native_mask_replayed")) is not bool
                or state["native_mask_replayed"] != (state["training"] and 0 < p < 1)):
            return False
    expected = [row for row in states if row["native_mask_replayed"]]
    if [row.get("node") for row in masks] != [row["node"] for row in expected]:
        return False
    for mask, state in zip(masks, expected):
        shape, stride = mask.get("shape"), mask.get("input_stride")
        if (not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(stride, list) or len(stride) != len(shape) or any(type(n) is not int or n < 0 for n in stride)
                or mask.get("dtype") != {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}.get(precision)
                or mask.get("device") != source_device or mask.get("probability") != state["probability"]
                or type(mask.get("elements")) is not int or mask["elements"] != math.prod(shape)
                or type(mask.get("kept")) is not int or not 0 <= mask["kept"] <= mask["elements"]
                or not isinstance(mask.get("mask_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", mask["mask_sha256"])
                or mask.get("scope") != "native_random_mask_only; independent_FP64_arithmetic"):
            return False
    return True
