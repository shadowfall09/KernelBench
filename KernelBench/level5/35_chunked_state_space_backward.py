"""KernelBench Pro production reference. Public contract:
{
  "name": "chunked_state_space_backward",
  "track": "backward",
  "roles": [
    [
      "u",
      "B H S D",
      "activation",
      "signed"
    ],
    [
      "delta",
      "B H S",
      "fp32",
      "positive_step"
    ],
    [
      "decay",
      "H",
      "fp32",
      "negative_decay"
    ],
    [
      "b",
      "B H S R",
      "activation",
      "signed"
    ],
    [
      "c",
      "B H S R",
      "activation",
      "signed"
    ]
  ],
  "upstream_role": {
    "name": "upstream",
    "shape": "forward_output_shape",
    "domain": "signed",
    "dtype": "fp32"
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
    2,
    3,
    4
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

import torch
import torch.nn as nn


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def state_space(u, delta, decay, b, c):
    # Scalar decay per head with vector state: the SSD recurrence, not a whole Mamba model.
    u, delta, decay, b, c = [acc(x) for x in (u, delta, decay, b, c)]
    state = torch.zeros((*u.shape[:2], b.shape[-1], u.shape[-1]), dtype=u.dtype, device=u.device)
    outputs = []
    for t in range(u.shape[2]):
        dt = delta[..., t]
        alpha = (dt * decay).exp()[..., None, None]
        state = alpha * state + dt[..., None, None] * b[..., t, :, None] * u[..., t, None, :]
        outputs.append(torch.einsum("bhs,bhsd->bhd", c[..., t, :], state))
    return torch.stack(outputs, 2)

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
        return vjp(state_space, inputs[:-1], [0, 1, 2, 3, 4], inputs[-1])


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(35)
