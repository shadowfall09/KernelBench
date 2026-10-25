"""Independent FP64 attention/encoder layers; no native fused numerical calls."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .composite_math import _activation, _softmax
from .operator_oracles import _affine, _host, _normalize

MODULES = {torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer, torch.nn.TransformerEncoder}


def linear(x, weight, bias=None):
    y = x @ _host(weight).t()
    return y if bias is None else y + _host(bias)


def norm(x, layer):
    axes = tuple(range(x.ndim - len(layer.normalized_shape), x.ndim))
    value = _normalize(x, axes, layer.eps)
    return _affine(value, layer, (*([1] * (x.ndim - len(axes))), *layer.normalized_shape)) if layer.elementwise_affine else value


def attention(layer, query, key, value, *, key_padding_mask=None, need_weights=True, attn_mask=None,
              average_attn_weights=True, is_causal=False):
    if (layer.dropout != 0 or layer.bias_k is not None or layer.bias_v is not None or layer.add_zero_attn
            or is_causal and attn_mask is None or query.ndim != 3):
        raise ValueError("No reviewed independent attention contract for these options")
    if layer.batch_first:
        query, key, value = (x.transpose(0, 1) for x in (query, key, value))
    length, batch, width = query.shape
    source = key.shape[0]
    if layer._qkv_same_embed_dim:
        weights = _host(layer.in_proj_weight).chunk(3)
    else:
        weights = [_host(getattr(layer, name + "_proj_weight")) for name in ("q", "k", "v")]
    biases = [None] * 3 if layer.in_proj_bias is None else _host(layer.in_proj_bias).chunk(3)
    q, k, v = [x @ w.t() + (b if b is not None else 0) for x, w, b in zip((query, key, value), weights, biases)]
    heads, dim = layer.num_heads, width // layer.num_heads
    q, k, v = [x.reshape(-1, batch, heads, dim).permute(1, 2, 0, 3) for x in (q, k, v)]
    scores = (q / math.sqrt(dim)) @ k.transpose(-1, -2)
    if attn_mask is not None:
        mask = attn_mask.reshape(batch, heads, length, source) if attn_mask.ndim == 3 else attn_mask
        scores = scores.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :]
        scores = scores.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    probabilities = _softmax(scores, -1)
    output = (probabilities @ v).permute(2, 0, 1, 3).reshape(length, batch, width)
    output = linear(output, layer.out_proj.weight, layer.out_proj.bias)
    if layer.batch_first:
        output = output.transpose(0, 1)
    weights = probabilities.mean(1) if average_attn_weights else probabilities
    return output, weights if need_weights else None


def encoder_layer(layer, x, *, src_mask=None, src_key_padding_mask=None, is_causal=False):
    if any(value.p != 0 for value in (layer.dropout, layer.dropout1, layer.dropout2)):
        raise ValueError("Encoder dropout requires a separate reviewed native sampler contract")

    def self_attention(value):
        return attention(layer.self_attn, value, value, value, attn_mask=src_mask,
                         key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0]

    def feedforward(value):
        value = linear(value, layer.linear1.weight, layer.linear1.bias)
        if layer.activation is F.relu:
            value = _activation("relu", value)
        elif layer.activation is F.gelu:
            value = _activation("gelu", value)
        else:
            raise ValueError("Unreviewed transformer activation")
        return linear(value, layer.linear2.weight, layer.linear2.bias)

    if layer.norm_first:
        x = x + self_attention(norm(x, layer.norm1))
        return x + feedforward(norm(x, layer.norm2))
    x = norm(x + self_attention(x), layer.norm1)
    return norm(x + feedforward(x), layer.norm2)


def module(layer, args, kwargs):
    if type(layer) is torch.nn.MultiheadAttention:
        return attention(layer, *args, **kwargs)
    if type(layer) is torch.nn.TransformerEncoderLayer:
        return encoder_layer(layer, *args, **kwargs)
    if type(layer) is torch.nn.TransformerEncoder:
        x = args[0]
        names = ("mask", "src_key_padding_mask", "is_causal")
        options = {**dict(zip(names, args[1:])), **kwargs}
        for block in layer.layers:
            x = encoder_layer(block, x, src_mask=options.get("mask"),
                              src_key_padding_mask=options.get("src_key_padding_mask"), is_causal=options.get("is_causal") or False)
        return x if layer.norm is None else norm(x, layer.norm)
    raise ValueError("No reviewed independent transformer layer")
