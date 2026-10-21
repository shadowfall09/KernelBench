"""KernelBench Pro production reference. Public contract:
{
  "name": "dequantization_gemm",
  "track": "design",
  "roles": [
    [
      "x",
      "M K",
      "activation",
      "signed"
    ],
    [
      "packed",
      "N K/2",
      "uint8",
      "packed_int4"
    ],
    [
      "scale",
      "N K/G",
      "fp32",
      "scale"
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
    "fp32",
    "bf16",
    "fp16"
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

def dequant_gemm(x, packed, scale, group_size):
    weight = unpack_int4(packed, acc(scale), group_size)
    return (acc(x) @ weight.T).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return dequant_gemm(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(10)
