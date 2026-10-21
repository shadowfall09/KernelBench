"""KernelBench Pro production reference. Public contract:
{
  "name": "format_budget_moe",
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
      "E N K",
      "activation",
      "weight"
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
    ]
  ],
  "upstream_role": null,
  "contract_kind": "error_budget",
  "precision_contract": {
    "norm_rtol": 0.005,
    "norm_atol": 1e-07,
    "element_rtol": 0.005,
    "element_atol": 1e-07,
    "element_scale_rtol": 0.005
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

def grouped_moe(x, weight, experts, gates):
    x, weight, gates = acc(x), acc(weight), acc(gates)
    output = torch.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
    for route in range(experts.shape[1]):
        selected = weight[experts[:, route]]
        output = output + torch.einsum("td,tod->to", x, selected) * gates[:, route, None]
    return output

def budget_moe(x, weight, experts, gates):
    return grouped_moe(x.double(), weight.double(), experts, gates.double()).to(x.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return budget_moe(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(16)
