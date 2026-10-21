"""KernelBench Pro production reference. Public contract:
{
  "name": "fft_long_convolution",
  "track": "design",
  "roles": [
    [
      "x",
      "B D S",
      "activation",
      "signed"
    ],
    [
      "kernel",
      "D L",
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

def long_convolution(x, kernel):
    # Causal depthwise convolution. A direct reference permits FFT implementations.
    x, kernel = acc(x), acc(kernel)
    return F.conv1d(F.pad(x, (kernel.shape[-1] - 1, 0)), kernel.flip(-1)[:, None], groups=x.shape[1])


class Model(nn.Module):
    def forward(self, *inputs):
        return long_convolution(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(8)
