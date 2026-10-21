"""KernelBench Pro production reference. Public contract:
{
  "name": "quantized_moe",
  "track": "implementation",
  "roles": [
    [
      "x",
      "M K",
      "activation",
      "signed"
    ],
    [
      "packed",
      "E N K/2",
      "uint8",
      "packed_int4"
    ],
    [
      "scale",
      "E N K/G",
      "fp32",
      "scale"
    ],
    [
      "experts",
      "M Top",
      "int64",
      "expert_index"
    ],
    [
      "gates",
      "M Top",
      "fp32",
      "probabilities"
    ],
    [
      "group_size",
      "",
      "scalar",
      "group_size"
    ]
  ],
  "upstream_role": null,
  "contract_kind": "fixed_format",
  "precision_contract": null,
  "fixed_format_comparison": {
    "norm_rtol_by_activation_dtype": {
      "fp32": 0.001,
      "fp16": 0.002,
      "bf16": 0.01
    },
    "norm_atol": 1e-08,
    "element_atol": 1e-08,
    "element_rtol_multiplier": 2,
    "element_global_rms_multiplier": 1
  },
  "gradient_indices": null,
  "dtype_families": [
    "bf16"
  ],
  "dtype_aliases": {
    "fp32": "torch.float32",
    "bf16": "torch.bfloat16",
    "fp16": "torch.float16",
    "fp8_e4m3fn": "torch.float8_e4m3fn"
  }
}
"""

import torch
import torch.nn as nn


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def unpack_int4(packed, scale, group_size):
    low, high = (packed & 15).to(torch.int16), (packed >> 4).to(torch.int16)
    signed = torch.stack((low, high), -1).flatten(-2).to(scale.dtype) - 8
    return signed * scale.repeat_interleave(group_size, -1)[..., :signed.shape[-1]]

def grouped_moe(x, weight, experts, gates):
    x, weight, gates = acc(x), acc(weight), acc(gates)
    output = torch.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
    for route in range(experts.shape[1]):
        selected = weight[experts[:, route]]
        output = output + torch.einsum("td,tod->to", x, selected) * gates[:, route, None]
    return output

def quantized_moe(x, packed, scale, experts, gates, group_size):
    weights = unpack_int4(packed, acc(scale), group_size)
    return grouped_moe(x, weights, experts, gates).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return quantized_moe(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(29)
