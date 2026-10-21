"""KernelBench Pro production reference. Public contract:
{
  "name": "paged_attention",
  "track": "design",
  "roles": [
    [
      "q",
      "B H D",
      "activation",
      "signed"
    ],
    [
      "k_cache",
      "P Hkv Block D",
      "activation",
      "signed"
    ],
    [
      "v_cache",
      "P Hkv Block D",
      "activation",
      "signed"
    ],
    [
      "block_table",
      "B Pages",
      "int64",
      "page_index"
    ],
    [
      "lengths",
      "B",
      "int64",
      "sequence_length"
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

import math

import torch
import torch.nn as nn


def acc(x):
    return x if x.dtype == torch.float64 else x.float()

def paged_attention(q, k_cache, v_cache, block_table, lengths):
    outputs = []
    for batch in range(q.shape[0]):
        # Shape is static; no length.item() or host synchronization is needed.
        keys = k_cache[block_table[batch]].transpose(0, 1).flatten(1, 2)
        values = v_cache[block_table[batch]].transpose(0, 1).flatten(1, 2)
        repeat = q.shape[1] // keys.shape[0]
        keys, values = keys.repeat_interleave(repeat, 0), values.repeat_interleave(repeat, 0)
        mask = torch.arange(keys.shape[1], device=q.device) < lengths[batch]
        logits = torch.einsum("hd,hnd->hn", acc(q[batch]), acc(keys)) / math.sqrt(q.shape[-1])
        probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        outputs.append(torch.einsum("hn,hnd->hd", probabilities, acc(values)))
    return torch.stack(outputs).to(q.dtype)


class Model(nn.Module):
    def forward(self, *inputs):
        return paged_attention(*inputs)


def get_init_inputs():
    return []


def get_inputs():
    from kb_workflows.production import make_inputs
    return make_inputs(11)
