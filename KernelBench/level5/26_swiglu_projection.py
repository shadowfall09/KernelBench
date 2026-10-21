"""KernelBench Pro production reference. Public contract:
{
  "name": "swiglu_projection",
  "track": "implementation",
  "roles": [
    [
      "x",
      "M K",
      "activation",
      "signed"
    ],
    [
      "up",
      "N K",
      "activation",
      "weight"
    ],
    [
      "gate",
      "N K",
      "activation",
      "weight"
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
import torch.nn.functional as F


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def swiglu(x, up, gate):
    return ((acc(x) @ acc(up).T) * F.silu(acc(x) @ acc(gate).T)).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return swiglu(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(26)
