"""KernelBench Pro production reference. Public contract:
{
  "name": "fp8_scaled_gemm",
  "track": "implementation",
  "roles": [
    [
      "x",
      "M K",
      "fp8_e4m3fn",
      "signed"
    ],
    [
      "weight",
      "N K",
      "fp8_e4m3fn",
      "weight"
    ],
    [
      "x_scale",
      "M 1",
      "fp32",
      "scale"
    ],
    [
      "weight_scale",
      "N 1",
      "fp32",
      "scale"
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

def fp8_scaled_gemm(x, weight, x_scale, weight_scale):
    return ((x.float() * x_scale) @ (weight.float() * weight_scale).T).to(torch.bfloat16)


class Model(nn.Module):
    def forward(self, *inputs):
        return fp8_scaled_gemm(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(21)
