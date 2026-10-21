"""KernelBench Pro production reference. Public contract:
{
  "name": "token_fp8_quantization_attention",
  "track": "numerical",
  "roles": [
    [
      "q",
      "B H S D",
      "activation",
      "signed"
    ],
    [
      "k",
      "B H S D",
      "activation",
      "signed"
    ],
    [
      "v",
      "B H S D",
      "activation",
      "signed"
    ],
    [
      "mask",
      "S S",
      "bool",
      "causal_mask"
    ]
  ],
  "upstream_role": null,
  "contract_kind": "error_budget",
  "precision_contract": {
    "norm_rtol": 0.002,
    "norm_atol": 1e-07,
    "element_rtol": 0.002,
    "element_atol": 1e-07,
    "element_scale_rtol": 0.002
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

import math

import torch
import torch.nn as nn


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def attention(q, k, v, mask):
    logits = acc(q) @ acc(k).transpose(-1, -2) / math.sqrt(q.shape[-1])
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    return (probabilities @ acc(v)).to(q.dtype)

def token_fp8_attention(q, k, v, mask):
    def quantize(x):
        scale = acc(x).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return ((acc(x) / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    return attention(quantize(q).double(), quantize(k).double(), quantize(v).double(), mask).to(q.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return token_fp8_attention(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(18)
