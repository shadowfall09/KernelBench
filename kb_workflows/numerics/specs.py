"""Reviewed numerical contracts, oracle selection and receipt validation.

Catalog lookup remains SDK-light; tensor implementations load only when used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from ..common import fingerprint

CONTRACTS = Path(__file__).with_name("verification_contracts.json")


def contract_for(level, pid, reference_source):
    contract = json.loads(CONTRACTS.read_text())["tasks"].get(f"{level}/{pid}")
    if contract is not None and contract["reference_sha256"] != hashlib.sha256(reference_source.encode()).hexdigest():
        raise ValueError("Strict contract reference snapshot changed; review its domain and oracle")
    return contract


def coverage(level, selected):
    available = json.loads(CONTRACTS.read_text())["tasks"]
    supported, unsupported = [], []
    for pid, path in selected.items():
        contract = available.get(f"{level}/{pid}")
        if contract is None:
            unsupported.append(pid)
        else:
            if contract["reference_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Strict contract reference snapshot changed for {level}/{pid}")
            supported.append(pid)
    return {"supported": supported, "unsupported": unsupported,
            "precision_scope": "reviewed_auxiliary_cases; original_timing_inputs_keep_native_comparator"}


OPERATOR_PROTOCOL = "kernelbench_pro_operator_oracles_v1"
OPERATOR_MANIFEST = Path(__file__).with_name("operator_oracles.json")


def spec_for(level, pid, source):
    document = json.loads(OPERATOR_MANIFEST.read_text())
    if document.get("protocol") != OPERATOR_PROTOCOL:
        raise ValueError("Unknown independent operator oracle manifest")
    result = document["tasks"].get(f"{level}/{pid}")
    if result is None:
        raise ValueError("No independent operator oracle adapter for this task")
    if result["reference_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Independent operator oracle reference changed; review the formula and domain")
    if result.get("family") == "composite":
        composite_program(result)
    elif result.get("family") == "model_block":
        model_program(result)
    return {**{key: document[key] for key in ("oracle", "precision", "scope", "native_input_roles_preserved")}, **result}


COMPOSITE_PROTOCOL = "kernelbench_pro_composite_oracle_v1"
COMPOSITE_PROGRAMS = Path(__file__).with_name("composite_programs.json")


def composite_program(spec):
    document = json.loads(COMPOSITE_PROGRAMS.read_text())
    if document.get("protocol") != "kernelbench_pro_composite_programs_v1":
        raise ValueError("Unknown composite program registry")
    program = document["programs"].get(spec["program_sha256"])
    if (not isinstance(program, dict) or fingerprint(program) != spec["program_sha256"]
            or program.get("protocol") != COMPOSITE_PROTOCOL or spec.get("composite_protocol") != COMPOSITE_PROTOCOL
            or not isinstance(program.get("nodes"), list) or len(program["nodes"]) != spec["program_nodes"]):
        raise ValueError("Composite arithmetic plan is missing or changed")
    return program


def validate_composite_receipt(receipt, spec, *, precision, source_device):
    program = composite_program(spec)
    if precision not in {"fp32", "fp16", "bf16"}:
        return False
    if (not isinstance(receipt, dict) or receipt.get("protocol") != COMPOSITE_PROTOCOL
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


MODEL_PROTOCOL = "kernelbench_pro_model_oracle_v1"
MODEL_PROGRAMS = Path(__file__).with_name("model_programs.json")


def model_program(spec):
    document = json.loads(MODEL_PROGRAMS.read_text())
    program = document.get("programs", {}).get(spec.get("program_sha256"))
    if (document.get("protocol") != "kernelbench_pro_model_programs_v1" or not isinstance(program, dict)
            or program.get("protocol") != MODEL_PROTOCOL or spec.get("model_protocol") != MODEL_PROTOCOL
            or fingerprint(program) != spec.get("program_sha256") or program.get("problem_id") != spec.get("problem_id")
            or not isinstance(program.get("architecture", {}).get("modules"), dict)
            or len(program["architecture"]["modules"]) != spec.get("program_modules")
            or not isinstance(program.get("dropout_order"), list)):
        raise ValueError("Independent model architecture/arithmetic plan is missing or changed")
    return program


def validate_model_receipt(receipt, spec, *, precision, source_device):
    program = model_program(spec)
    if (precision not in {"fp32", "fp16", "bf16"} or not isinstance(receipt, dict)
            or receipt.get("protocol") != MODEL_PROTOCOL or receipt.get("program_sha256") != spec["program_sha256"]
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


def prepare_oracle(reference, inputs, *, spec, problem_id):
    if spec.get("family") == "composite":
        from .composite_oracles import Prepared

        return Prepared(reference, inputs, expected_program=spec["program_sha256"])
    if spec.get("family") == "model_block":
        from .model_oracles import Prepared

        return Prepared(reference, inputs, problem_id=problem_id, expected_program=spec["program_sha256"])
    from .operator_oracles import estimate_bytes, execute

    class Operator:
        estimated_host_bytes = estimate_bytes(problem_id, reference, inputs)

        def execute(self):
            return execute(problem_id, reference, inputs)

        def receipt(self):
            return None

    return Operator()
