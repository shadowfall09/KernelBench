"""Transparent mathematical references, independent of any submitted kernel.

Functions use FP32 accumulation for low-precision storage and preserve FP64 for
oracle/gradient checks. Packed formats specify representation, not native throughput.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def acc(x):
    return x if x.dtype == torch.float64 else x.float()


def attention(q, k, v, mask):
    logits = acc(q) @ acc(k).transpose(-1, -2) / math.sqrt(q.shape[-1])
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    return (probabilities @ acc(v)).to(q.dtype)


def linear_cross_entropy(x, weight, targets):
    logits = acc(x) @ acc(weight).T
    return (logits.logsumexp(-1) - logits.gather(-1, targets[:, None]).squeeze(-1)).mean()


def layer_norm(x, weight, bias, eps):
    y = acc(x)
    mean = y.mean(-1, keepdim=True)
    variance = (y - mean).square().mean(-1, keepdim=True)
    return ((y - mean) * torch.rsqrt(variance + eps) * acc(weight) + acc(bias)).to(x.dtype)


def partial_topk(x, k):
    # Ties use the lowest original index, making the discrete contract unambiguous.
    indices = torch.argsort(x, dim=-1, descending=True, stable=True)[..., :k]
    return x.gather(-1, indices), indices


def linear_attention(q, k, v):
    q, k, v = acc(q), acc(k), acc(v)
    state = torch.zeros((*q.shape[:-2], k.shape[-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    outputs = []
    for t in range(q.shape[-2]):
        state = state + k[..., t, :, None] * v[..., t, None, :]
        outputs.append(torch.einsum("...d,...de->...e", q[..., t, :], state))
    return torch.stack(outputs, -2)


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


def recurrent(x, h, input_weight, hidden_weight, bias):
    x, h, input_weight, hidden_weight, bias = [acc(t) for t in (x, h, input_weight, hidden_weight, bias)]
    outputs = []
    for t in range(x.shape[1]):
        h = (x[:, t] @ input_weight.T + h @ hidden_weight.T + bias).tanh()
        outputs.append(h)
    return torch.stack(outputs, 1)


def long_convolution(x, kernel):
    # Causal depthwise convolution. A direct reference permits FFT implementations.
    x, kernel = acc(x), acc(kernel)
    return F.conv1d(F.pad(x, (kernel.shape[-1] - 1, 0)), kernel.flip(-1)[:, None], groups=x.shape[1])


def grouped_moe(x, weight, experts, gates):
    x, weight, gates = acc(x), acc(weight), acc(gates)
    output = torch.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
    for route in range(experts.shape[1]):
        selected = weight[experts[:, route]]
        output = output + torch.einsum("td,tod->to", x, selected) * gates[:, route, None]
    return output


def unpack_int4(packed, scale, group_size):
    low, high = (packed & 15).to(torch.int16), (packed >> 4).to(torch.int16)
    signed = torch.stack((low, high), -1).flatten(-2).to(scale.dtype) - 8
    return signed * scale.repeat_interleave(group_size, -1)[..., :signed.shape[-1]]


def dequant_gemm(x, packed, scale, group_size):
    weight = unpack_int4(packed, acc(scale), group_size)
    return (acc(x) @ weight.T).to(x.dtype)


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


def split_k_gemm(x, weight):
    return (acc(x) @ acc(weight)).to(x.dtype)


def accurate_gemm(x, weight):
    return (x.double() @ weight.double()).to(x.dtype)


def budget_attention(q, k, v, mask):
    return attention(q.double(), k.double(), v.double(), mask).to(q.dtype)


def budget_moe(x, weight, experts, gates):
    return grouped_moe(x.double(), weight.double(), experts, gates.double()).to(x.dtype)


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


def token_fp8_attention(q, k, v, mask):
    def quantize(x):
        scale = acc(x).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return ((acc(x) / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    return attention(quantize(q).double(), quantize(k).double(), quantize(v).double(), mask).to(q.dtype)


def fixed_gemm(x, weight):
    return (acc(x) @ acc(weight).T).to(x.dtype)


def fp8_scaled_gemm(x, weight, x_scale, weight_scale):
    return ((x.float() * x_scale) @ (weight.float() * weight_scale).T).to(torch.bfloat16)


def grouped_gemm(x, weight):
    return torch.bmm(acc(x), acc(weight).transpose(1, 2)).to(x.dtype)


def swiglu(x, up, gate):
    return ((acc(x) @ acc(up).T) * F.silu(acc(x) @ acc(gate).T)).to(x.dtype)


def residual_rmsnorm(x, residual, weight, eps):
    y = acc(x) + acc(residual)
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps) * acc(weight)).to(x.dtype)


def rotary(x, cosine, sine):
    even, odd = acc(x[..., 0::2]), acc(x[..., 1::2])
    return torch.stack((even * acc(cosine) - odd * acc(sine), even * acc(sine) + odd * acc(cosine)), -1).flatten(-2).to(x.dtype)


def quantized_moe(x, packed, scale, experts, gates, group_size):
    weights = unpack_int4(packed, acc(scale), group_size)
    return grouped_moe(x, weights, experts, gates).to(x.dtype)


def gqa_decode(q, k, v):
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    logits = (acc(q) @ acc(k).transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return (logits.softmax(-1) @ acc(v)).to(q.dtype)


def vjp(function, args, gradient_indices, upstream):
    # Backward is exposed as a forward callable; candidate kernels return these VJPs.
    with torch.enable_grad():
        fresh = [value.detach().clone().requires_grad_(True) if i in gradient_indices else value
                 for i, value in enumerate(args)]
        output = function(*fresh)
        return tuple(torch.autograd.grad(output, [fresh[i] for i in gradient_indices], grad_outputs=upstream,
                                         create_graph=False, allow_unused=False))
