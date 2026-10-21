"""Role-preserving production instances and bounded, deterministic memory scaling."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from .common import fingerprint
from .production_specs import comparison_contract, task

PROTOCOL = "production_families_v1"
SETTINGS = ContextVar("production_instance_settings", default={})
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8_e4m3fn": 1, "int64": 8, "uint8": 1, "bool": 1}


def dimensions(profile):
    if profile == "small":
        return dict(B=1, H=2, Hkv=1, S=4, D=4, M=3, N=5, K=8, V=11, E=3, Top=2, G=4, Block=3, Pages=2, R=3, L=3)
    if profile not in {"standard", "large", "hidden_odd", "hidden_strided", "hidden_extreme", "hidden_constant"}:
        raise ValueError("Unknown production instance profile")
    if profile.startswith("hidden_"):
        return dict(B=2, H=4, Hkv=2, S=19, D=12, M=17, N=23, K=24, V=37, E=5, Top=2, G=8, Block=5, Pages=3, R=7, L=9)
    dims = dict(B=2, H=4, Hkv=2, S=256, D=64, M=256, N=384, K=512, V=8192, E=8, Top=2, G=32, Block=16, Pages=16, R=32, L=129)
    if profile == "large":
        for name in ("S", "M", "N", "K", "V", "Pages"):
            dims[name] *= 4
    return dims


def shape(expression, dims):
    def number(word):
        return int(word) if word.isdecimal() else dims[word]
    values = []
    for term in expression.split():
        if "/" in term:
            numerator, denominator = map(number, term.split("/"))
            if numerator % denominator:
                raise ValueError("Packed/scaling dimensions must be divisible")
            values.append(numerator // denominator)
        else:
            values.append(number(term))
    return tuple(values)


def roles(spec, dims, primary):
    return [{"name": name, "shape": list(shape(expression, dims)), "dtype": primary if storage == "activation" else storage,
             "domain": domain} for name, expression, storage, domain in spec["roles"]]


def estimated_peak(spec, dims, primary):
    storage = sum(math.prod(row["shape"]) * DTYPE_BYTES[row["dtype"]]
                  for row in roles(spec, dims, primary) if row["dtype"] != "scalar")
    b, h, s, d = [dims[name] for name in ("B", "H", "S", "D")]
    family = spec["family"]
    temporary = 0
    if family == "attention":
        temporary = b * h * s * s * 32
    elif family == "cross_entropy":
        temporary = dims["M"] * dims["V"] * 32
    elif family in {"moe", "quantized_moe"}:
        temporary = dims["M"] * dims["N"] * dims["K"] * 32
    elif family == "attention_linear":
        temporary = b * h * s * d * d * 32
    elif family == "state_space":
        temporary = b * h * s * dims["R"] * d * 32
    elif family == "fp4_gemm":
        temporary = (dims["M"] + dims["N"]) * dims["K"] * 8 * 8
    return max(1, storage * 12 + temporary)  # Clones, oracle, gradients and selected-reference temporaries.


def instance_plan(pid, *, precision="fp32", profile="standard", memory_bytes=512 * 1024**2, total_memory_bytes=None):
    spec = task(pid)
    if precision not in {"fp32", "bf16", "fp16"} or type(memory_bytes) is not int or memory_bytes < 1024:
        raise ValueError("Invalid production dtype or memory budget")
    primary = spec["fixed_activation_dtype"] or precision
    budget = memory_bytes
    if total_memory_bytes is not None:
        if type(total_memory_bytes) is not int or total_memory_bytes < 10240:
            raise ValueError("Invalid device memory capacity")
        budget = min(budget, total_memory_bytes // 10)
    dims = dimensions(profile)
    if profile in {"standard", "large"}:
        multiplier = 4 if profile == "large" else 1
        overrides = {4: {"M": 32, "N": 131072}, 25: {"M": 4096, "N": 1024},
                     10: {"M": 128, "K": 8192, "N": 2048}, 22: {"M": 32, "K": 4096, "N": 4096},
                     24: {"B": 16, "H": 32, "Hkv": 8, "D": 128, "Pages": 8},
                     12: {"M": 16, "K": 32768, "N": 64}}
        for name, value in overrides.get(spec["base_id"], {}).items():
            dims[name] = value * multiplier if name in {"M", "N", "K"} else value
    requested = dict(dims)
    while True:
        dims["P"] = dims["B"] * dims["Pages"] + 1
        peak = estimated_peak(spec, dims, primary)
        if peak <= budget:
            break
        reduced = False
        for name, floor, multiple in (("M", 1, 1), ("N", 2, 1), ("S", 2, 1), ("K", dims["G"], dims["G"]),
                                      ("V", 2, 1), ("D", 2, 2), ("R", 1, 1), ("L", 2, 1), ("Pages", 1, 1)):
            value = max(floor, (dims[name] // 2 // multiple) * multiple)
            if value < dims[name]:
                dims[name], reduced = value, True
        if not reduced:
            raise ValueError("Production instance cannot fit the declared memory envelope")
    input_roles = roles(spec, dims, primary)
    if spec["upstream_role"]:
        upstream = spec["upstream_role"]
        input_roles.append({"name": "upstream", "shape": list(output_shape(spec, dims)), "domain": "signed",
                            "dtype": primary if upstream["dtype"] == "activation" else upstream["dtype"]})
    result = {"protocol": PROTOCOL, "problem_id": pid, "name": spec["name"], "track": spec["track"],
              "precision_family": primary, "profile": profile, "dimensions": dims, "requested_dimensions": requested,
              "input_roles": input_roles, "estimated_peak_bytes": peak, "memory_budget_bytes": budget,
              "memory_policy": "min(explicit_budget,device_capacity/10); model_estimate_not_an_OOM_guarantee",
              "contract_kind": spec["contract_kind"], "precision_contract": spec["precision_contract"],
              "output_comparison_contract": comparison_contract(spec, primary),
              "comparison_reference": "declared_reference_output; independent_FP64_self_qualification_not_certified",
              "numerical_reference": "FP64_arithmetic_rounded_to_output; prescribed_quantization_precedes_oracle" if spec["precision_contract"] else None,
              "representation": "E2M1_software_quantization" if pid == 17 else
                  ("float8_e4m3fn_storage" if pid == 21 else "explicit_role_storage"),
              "native_fp4_required": False, "native_low_precision_execution_certified": False}
    result["instance_sha256"] = fingerprint(result)
    return result


def output_shape(spec, dims):
    family = spec["family"]
    expression = {"attention": "B H S D", "attention_linear": "B H S D", "state_space": "B H S D",
                  "cross_entropy": "", "normalization": "M D", "recurrent": "B S D", "convolution": "B D S",
                  "moe": "M N", "int4_gemm": "M N", "paged": "B H D", "matmul": "M N", "fp4_gemm": "M N",
                  "fixed_gemm": "M N", "fp8_gemm": "M N", "grouped_gemm": "E M N", "swiglu": "M N",
                  "rmsnorm": "M D", "rotary": "B H S D", "quantized_moe": "M N", "gqa": "B H 1 D"}[family]
    return shape(expression, dims)


def make_inputs(pid, *, case=None, precision=None, profile=None, memory_bytes=None, seed=None):
    import torch

    settings = SETTINGS.get()
    plan = instance_plan(pid, precision=precision or settings.get("precision", "fp32"),
        profile=case or profile or settings.get("profile", "standard"), memory_bytes=memory_bytes if memory_bytes is not None else settings.get("memory_bytes", 512 * 1024**2),
        total_memory_bytes=settings.get("total_memory_bytes"))
    dims = plan["dimensions"]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    types = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "fp8_e4m3fn": torch.float8_e4m3fn,
             "uint8": torch.uint8, "int64": torch.int64, "bool": torch.bool}
    angles = torch.randn(dims["S"], dims["D"] // 2, generator=generator)
    args = []
    for row in plan["input_roles"]:
        domain, size, dtype = row["domain"], row["shape"], row["dtype"]
        if dtype == "scalar":
            args.append({"epsilon": 1e-5 if not case else 3e-4, "topk_count": min(3 if not case else 4, dims["N"]),
                         "group_size": dims["G"]}[domain])
            continue
        if domain == "causal_mask":
            value = torch.ones(size, dtype=torch.bool).tril()
            if case == "hidden_extreme":
                value &= torch.arange(dims["S"])[None, :] % 2 == 0
                value |= torch.eye(dims["S"], dtype=torch.bool)
        elif domain in {"class_index", "expert_index", "page_index", "sequence_length", "packed_int4"}:
            lower, upper = {"class_index": (0, dims["V"]), "expert_index": (0, dims["E"]),
                            "page_index": (0, dims["P"]), "sequence_length": (1, dims["Pages"] * dims["Block"] + 1),
                            "packed_int4": (0, 256)}[domain]
            value = torch.randint(lower, upper, size, generator=generator)
        elif domain in {"cosine", "sine"}:
            value = angles.cos() if domain == "cosine" else angles.sin()
        else:
            value = torch.randn(size, generator=generator)
            if domain == "weight":
                value /= math.sqrt(max(1, size[-1]))
            elif domain == "unit_weight":
                value = 1 + 0.1 * value
            elif domain in {"scale", "positive_step", "negative_decay"}:
                value = 0.01 + value.sigmoid() * (0.05 if domain == "scale" else 0.2)
                if domain == "negative_decay":
                    value.neg_()
            elif domain == "probabilities":
                value = value.softmax(-1)
            elif domain == "distinct":
                value = torch.randperm(math.prod(size), generator=generator).reshape(size).float() / math.prod(size)
            if case == "hidden_extreme" and domain == "signed":
                value *= 16
                value.flatten()[::3] *= 0.0001
            if case == "hidden_constant" and row["name"] == "x":
                value.fill_(64)
                value.flatten()[::2] += 0.001
        if dtype == "fp8_e4m3fn":
            value = (value * 32).clamp(-448, 448)
        value = value.to(dtype=types[dtype])
        if case == "hidden_strided" and value.ndim:
            backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
            strided = backing[..., ::2]
            strided.copy_(value)
            value = strided
        args.append(value)
    return args


def execute(pid, *args):
    from . import production_ops

    spec = task(pid)
    function = getattr(production_ops, spec["operation"])
    if spec["gradient_indices"]:
        return production_ops.vjp(function, args[:-1], spec["gradient_indices"], args[-1])
    return function(*args)


@contextmanager
def production_context(**settings):
    token = SETTINGS.set(settings)
    try:
        yield
    finally:
        SETTINGS.reset(token)


def task_input_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        if payload.get("level") != 5:
            return function(payload, *args, **kwargs)
        import torch

        from src.eval import input_dtype_policy
        from src.validation import output_comparison_policy

        options = payload["eval_config"]["production"]
        expected = {"protocol": PROTOCOL, "task_sha256": fingerprint(task(payload["problem_id"])),
                    "profile": options["profile"], "memory_bytes": options["memory_bytes"]}
        if options != expected:
            raise ValueError("Production task contract changed")
        if options["profile"] not in {"small", "standard", "large"}:
            raise ValueError("Private profiles cannot be selected as a public task")
        capacity = torch.cuda.get_device_properties(0).total_memory
        settings = {"precision": payload["eval_config"]["precision"], "profile": options["profile"],
                    "memory_bytes": options["memory_bytes"], "total_memory_bytes": capacity}
        plan = instance_plan(payload["problem_id"], **settings)
        from .numerical import PrecisionContract, precision_gate

        contract = PrecisionContract(**plan["output_comparison_contract"])
        def comparator(expected, actual, path):
            result = precision_gate(expected, actual, expected.double(), contract)
            return None if result["passed"] else f"{path}: production numerical budget violated"
        with production_context(**settings), input_dtype_policy(preserve=True), output_comparison_policy(comparator):
            result = function(payload, *args, **kwargs)
        result.setdefault("metadata", {})["production_instance"] = plan
        result["metadata"]["production_instance_scope"] = "public_nominal_plan; hidden_validation_uses_private_case_profiles"
        return result
    return wrapped
