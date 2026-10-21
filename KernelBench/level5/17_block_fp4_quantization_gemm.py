"""KernelBench Pro production reference. Public contract:
{
  "name": "block_fp4_quantization_gemm",
  "track": "numerical",
  "roles": [
    [
      "x",
      "M K",
      "fp32",
      "signed"
    ],
    [
      "weight",
      "N K",
      "fp32",
      "weight"
    ],
    [
      "block",
      "",
      "scalar",
      "group_size"
    ]
  ],
  "upstream_role": null,
  "contract_kind": "error_budget",
  "precision_contract": {
    "norm_rtol": 0.0002,
    "norm_atol": 1e-07,
    "element_rtol": 0.0002,
    "element_atol": 1e-07,
    "element_scale_rtol": 0.0002
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

def block_fp4_gemm(x, weight, block):
    # E2M1 magnitudes; quantization uses nearest value, ties to an even magnitude code.
    def quantize(value):
        dtype = value.dtype
        blocks = value.double().reshape(*value.shape[:-1], -1, block)
        scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 6
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device, dtype=torch.float64)
        normalized = blocks / scale
        distances = (normalized.abs()[..., None] - levels).abs()
        nearest = distances.argmin(-1)
        codes = torch.arange(8, device=value.device)
        tied_even = (distances == distances.amin(-1, keepdim=True)) & (codes % 2 == 0)
        even = torch.where(tied_even, codes, 8).amin(-1)
        index = torch.where(even < 8, even, nearest)
        return (normalized.sign() * levels[index] * scale).reshape(value.shape).to(dtype)
    return (quantize(x).double() @ quantize(weight).double().T).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return block_fp4_gemm(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(17)
