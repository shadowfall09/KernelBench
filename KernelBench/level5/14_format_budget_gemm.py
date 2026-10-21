"""KernelBench Pro production reference. Public contract:
{
  "name": "format_budget_gemm",
  "track": "numerical",
  "roles": [
    [
      "x",
      "M K",
      "activation",
      "signed"
    ],
    [
      "weight",
      "K N",
      "activation",
      "weight"
    ]
  ],
  "upstream_role": null,
  "contract_kind": "error_budget",
  "precision_contract": {
    "norm_rtol": 0.01,
    "norm_atol": 1e-07,
    "element_rtol": 0.01,
    "element_atol": 1e-07,
    "element_scale_rtol": 0.01
  },
  "fixed_format_comparison": null,
  "gradient_indices": null,
  "dtype_families": [
    "fp32"
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

def accurate_gemm(x, weight):
    return (x.double() @ weight.double()).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return accurate_gemm(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(14)
