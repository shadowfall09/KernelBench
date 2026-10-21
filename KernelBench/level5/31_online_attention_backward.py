"""KernelBench Pro production reference. Public contract:
{
  "name": "online_attention_backward",
  "track": "backward",
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
  "upstream_role": {
    "name": "upstream",
    "shape": "forward_output_shape",
    "domain": "signed",
    "dtype": "activation"
  },
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
  "gradient_indices": [
    0,
    1,
    2
  ],
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

import math

import torch
import torch.nn as nn


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def attention(q, k, v, mask):
    logits = acc(q) @ acc(k).transpose(-1, -2) / math.sqrt(q.shape[-1])
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    return (probabilities @ acc(v)).to(q.dtype)

def vjp(function, args, gradient_indices, upstream):
    # Backward is exposed as a forward callable; candidate kernels return these VJPs.
    with torch.enable_grad():
        fresh = [value.detach().clone().requires_grad_(True) if i in gradient_indices else value
                 for i, value in enumerate(args)]
        output = function(*fresh)
        return tuple(torch.autograd.grad(output, [fresh[i] for i in gradient_indices], grad_outputs=upstream,
                                         create_graph=False, allow_unused=False))


class Model(nn.Module):
    def forward(self, *inputs):
        return vjp(attention, inputs[:-1], [0, 1, 2], inputs[-1])


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(31)
