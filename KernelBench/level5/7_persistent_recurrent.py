"""KernelBench Pro production reference. Public contract:
{
  "name": "persistent_recurrent",
  "track": "design",
  "roles": [
    [
      "x",
      "B S D",
      "activation",
      "signed"
    ],
    [
      "h",
      "B D",
      "activation",
      "signed"
    ],
    [
      "input_weight",
      "D D",
      "activation",
      "weight"
    ],
    [
      "hidden_weight",
      "D D",
      "activation",
      "weight"
    ],
    [
      "bias",
      "D",
      "fp32",
      "signed"
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

def recurrent(x, h, input_weight, hidden_weight, bias):
    x, h, input_weight, hidden_weight, bias = [acc(t) for t in (x, h, input_weight, hidden_weight, bias)]
    outputs = []
    for t in range(x.shape[1]):
        h = (x[:, t] @ input_weight.T + h @ hidden_weight.T + bias).tanh()
        outputs.append(h)
    return torch.stack(outputs, 1)


class Model(nn.Module):
    def forward(self, *inputs):
        return recurrent(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(7)
