"""Source-reviewed L3 algorithms using CPU FP64 tensor arithmetic only."""

from __future__ import annotations

import math

import torch

from .composite_math import _activation, _softmax
from .model_layers import linear
from .operator_oracles import _host


def unit(x, dim=1, eps=1e-12):
    return x / x.square().sum(dim, keepdim=True).sqrt().clamp_min(eps)


def recurrent(layer, x, initial):
    if layer.dropout != 0 or getattr(layer, "proj_size", 0):
        raise ValueError("Recurrent dropout/projection needs a separately reviewed arithmetic contract")
    lstm = type(layer) is torch.nn.LSTM
    if not lstm and type(layer) is not torch.nn.GRU:
        raise ValueError("Unsupported recurrent cell")
    if layer.batch_first:
        x = x.transpose(0, 1)
    directions = 2 if layer.bidirectional else 1
    batch, hidden = x.shape[1], layer.hidden_size
    h0, c0 = initial if lstm else (initial, None)
    if h0 is None:
        h0 = torch.zeros(layer.num_layers * directions, batch, hidden, dtype=torch.float64)
    if lstm and c0 is None:
        c0 = torch.zeros_like(h0)
    final_h, final_c = [], []
    for level in range(layer.num_layers):
        both = []
        for direction in range(directions):
            suffix = f"_l{level}" + ("_reverse" if direction else "")
            wi, wh = (_host(getattr(layer, "weight_" + name + suffix)) for name in ("ih", "hh"))
            bi, bh = ([_host(getattr(layer, "bias_" + name + suffix)) for name in ("ih", "hh")]
                      if layer.bias else (0, 0))
            state = level * directions + direction
            h, c = h0[state], c0[state] if lstm else None
            outputs = [None] * len(x)
            for t in (range(len(x) - 1, -1, -1) if direction else range(len(x))):
                inputs, previous = x[t] @ wi.t() + bi, h @ wh.t() + bh
                if lstm:
                    i, f, g, o = (inputs + previous).chunk(4, -1)
                    c = _activation("sigmoid", f) * c + _activation("sigmoid", i) * g.tanh()
                    h = _activation("sigmoid", o) * c.tanh()
                else:
                    ir, iz, inn = inputs.chunk(3, -1)
                    hr, hz, hn = previous.chunk(3, -1)
                    reset = _activation("sigmoid", ir + hr)
                    update = _activation("sigmoid", iz + hz)
                    new = (inn + reset * hn).tanh()
                    h = (1 - update) * new + update * h
                outputs[t] = h
            both.append(torch.stack(outputs))
            final_h.append(h)
            if lstm:
                final_c.append(c)
        x = torch.cat(both, -1)
    output = x.transpose(0, 1) if layer.batch_first else x
    return output, torch.stack(final_h), torch.stack(final_c) if lstm else None


def recurrent_model(context, pid, args):
    model, x = context.model, args[0]
    if pid == 33:
        h = args[1].to(model.hidden.dtype).double() if len(args) > 1 and args[1] is not None else _host(model.hidden)
        return context.call("h2o", context.call("tanh", context.call("i2h", torch.cat((x, h), 1))))
    if pid == 34:
        h, outputs = args[1], []
        for step in x:
            h = context.call("tanh", context.call("i2h", torch.cat((step, h), 1)))
            outputs.append(context.call("h2o", h))
        return torch.stack(outputs)
    if 35 <= pid <= 38:
        initial = (args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
        output, h, c = recurrent(model.lstm, x, initial)
        # 36/37 compute the unused fc output before returning the native state.
        projected = context.call("fc", output[:, -1])
        return h if pid == 36 else c if pid == 37 else projected
    output, h, _ = recurrent(model.gru, x, args[1])
    return h if pid in {40, 42} else output


def causal(context, path, x, *, relu=False):
    layer = context.model.get_submodule(path)
    batch, length, width = x.shape
    q, k, v = context.call(path + ".c_attn" if path else "c_attn", x).chunk(3, -1)
    heads = layer.n_head
    q, k, v = [value.reshape(batch, length, heads, width // heads).transpose(1, 2) for value in (q, k, v)]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
    scores = scores.masked_fill(_host(layer.bias)[:, :, :length, :length] == 0, -torch.inf)
    probabilities = scores.clamp_min(0) if relu else _softmax(scores, -1)
    if not relu:
        probabilities = context.call((path + "." if path else "") + "attn_dropout", probabilities)
    result = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
    if relu:
        return result  # L3/50 declares c_proj but does not use it.
    prefix = path + "." if path else ""
    return context.call(prefix + "resid_dropout", context.call(prefix + "c_proj", result))


def gpt(context, pid, x):
    if pid != 44:
        return causal(context, "", x, relu=pid == 50)
    x = x + causal(context, "attn", context.call("ln_1", x))
    hidden = context.call("mlp.c_fc", context.call("ln_2", x))
    hidden = _activation("gelu", hidden, approximate="tanh")
    return x + context.call("mlp.dropout", context.call("mlp.c_proj", hidden))


def vlad(context, x):
    model = context.model
    batch, samples, features = x.shape
    flattened = x.reshape(-1, features)
    assignment = context.call("batch_norm", flattened @ _host(model.clusters))
    assignment = _softmax(assignment, 1)[:, :model.cluster_size].reshape(batch, samples, model.cluster_size)
    residual = assignment.sum(1, keepdim=True) * _host(model.clusters2)
    value = (assignment.transpose(1, 2) @ x).transpose(1, 2) - residual
    return unit(unit(value).reshape(batch, -1))


def _segments(x):
    prefix = x.cumsum(-1)
    difference = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    # j <= i includes the empty diagonal segment, exactly as the task declares.
    mask = torch.arange(x.shape[-1])[:, None] >= torch.arange(x.shape[-1])[None, :]
    return difference.masked_fill(~mask, -torch.inf)


def mamba(context, pid, args):
    model, value = context.model, args[0]
    batch, length, heads, width = value.shape
    block = model.block_len
    chunks = length // block
    value = value.reshape(batch, chunks, block, heads, width)
    a = _host(model.A).reshape(batch, chunks, block, heads).permute(0, 3, 1, 2)
    b = _host(model.B).reshape(batch, chunks, block, heads, -1)
    c = _host(model.C).reshape(batch, chunks, block, heads, -1)
    prefix = a.cumsum(-1)
    local_state = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, (prefix[..., -1:] - prefix).exp(), value)
    initial = args[1] if len(args) > 1 and args[1] is not None else torch.zeros_like(local_state[:, :1])
    history = torch.cat((initial, local_state), 1)
    ends = torch.cat((torch.zeros_like(prefix[..., :1, -1]), prefix[..., -1]), -1)
    states = torch.einsum("bhzc,bchpn->bzhpn", _segments(ends).exp(), history)
    if pid == 49:
        return states[:, -1]
    diagonal = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, _segments(a).exp(), value)
    off_diagonal = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states[:, :-1], prefix.exp())
    return (diagonal + off_diagonal).reshape(batch, length, heads, width)


def windows(x, size):
    batch, height, width, channels = x.shape
    return x.reshape(batch, height // size, size, width // size, size, channels).permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, channels)


def unwindow(x, size, height, width):
    batch = x.shape[0] // ((height // size) * (width // size))
    return x.reshape(batch, height // size, width // size, size, size, -1).permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


def mlp(context, path, x):
    x = context.call(path + ".act", context.call(path + ".fc1", x))
    x = context.call(path + ".drop", x)
    return context.call(path + ".drop", context.call(path + ".fc2", x))


def window_attention(context, path, x, mask):
    layer = context.model.get_submodule(path)
    batch, length, channels = x.shape
    bias = None if layer.q_bias is None else torch.cat((_host(layer.q_bias), torch.zeros_like(_host(layer.v_bias)), _host(layer.v_bias)))
    projected = linear(x, layer.qkv.weight, bias)
    q, k, v = projected.reshape(batch, length, 3, layer.num_heads, -1).permute(2, 0, 3, 1, 4)
    # The source's scalar torch.log(tensor(100.)) is rounded at default storage
    # precision before promotion; preserve that literal, not a higher precision one.
    cap = torch.tensor(100.).log().item()
    scores = (unit(q, -1) @ unit(k, -1).transpose(-1, -2)) * _host(layer.logit_scale).clamp(max=cap).exp()
    table = context.call(path + ".cpb_mlp.2", context.call(path + ".cpb_mlp.1", context.call(path + ".cpb_mlp.0", _host(layer.relative_coords_table))))
    table = table.reshape(-1, layer.num_heads)
    relative = table[_host(layer.relative_position_index).reshape(-1)].reshape(length, length, -1).permute(2, 0, 1)
    scores = scores + 16 * _activation("sigmoid", relative).unsqueeze(0)
    if mask is not None:
        count = mask.shape[0]
        scores = (scores.reshape(batch // count, count, layer.num_heads, length, length) + _host(mask)[None, :, None]).reshape(-1, layer.num_heads, length, length)
    probability = context.call(path + ".attn_drop", context.call(path + ".softmax", scores))
    result = (probability @ v).transpose(1, 2).reshape(batch, length, channels)
    return context.call(path + ".proj_drop", context.call(path + ".proj", result))


def swin(context, pid, x):
    model = context.model
    x = context.call("patch_embed.proj", x).flatten(2).transpose(1, 2)
    if model.patch_embed.norm is not None:
        x = context.call("patch_embed.norm", x)
    x = context.call("pos_drop", x)
    for i, stage in enumerate(model.layers):
        for j, block in enumerate(stage.blocks):
            path = f"layers.{i}.blocks.{j}"
            height, width = block.input_resolution
            batch, _, channels = x.shape
            shortcut = x
            if pid == 29:
                x = context.call(path + ".norm1", x)
            grid = x.reshape(batch, height, width, channels)
            if block.shift_size:
                if pid == 29:
                    left, right, top, bottom = block.padding
                    shifted = torch.zeros(batch, height + top + bottom, width + left + right, channels, dtype=torch.float64)
                    shifted[:, top:top + height, left:left + width] = grid
                    grid = shifted
                else:
                    grid = grid.roll((-block.shift_size, -block.shift_size), (1, 2))
            h, w = grid.shape[1:3]
            partition = windows(grid, block.window_size).reshape(-1, block.window_size**2, channels)
            if pid == 29:
                partition = partition.reshape(-1, block.window_size**2, block.num_heads, channels // block.num_heads).transpose(1, 2)
                partition = context.call(path + ".spatial_mlp", partition.reshape(-1, block.num_heads * block.window_size**2, channels // block.num_heads))
                partition = partition.reshape(-1, block.num_heads, block.window_size**2, channels // block.num_heads).transpose(1, 2).reshape(-1, block.window_size, block.window_size, channels)
            else:
                partition = window_attention(context, path + ".attn", partition, block.attn_mask).reshape(-1, block.window_size, block.window_size, channels)
            grid = unwindow(partition, block.window_size, h, w)
            if block.shift_size:
                grid = (grid[:, top:top + height, left:left + width] if pid == 29 else grid.roll((block.shift_size, block.shift_size), (1, 2)))
            x = grid.reshape(batch, height * width, channels)
            if pid == 30:
                x = context.call(path + ".norm1", x)
            x = shortcut + context.call(path + ".drop_path", x)
            feedforward = mlp(context, path + ".mlp", context.call(path + ".norm2", x)) if pid == 29 else context.call(path + ".norm2", mlp(context, path + ".mlp", x))
            x = x + context.call(path + ".drop_path", feedforward)
        if stage.downsample is not None:
            path = f"layers.{i}.downsample"
            height, width = stage.downsample.input_resolution
            batch, _, channels = x.shape
            grid = x.reshape(batch, height, width, channels)
            x = torch.cat([grid[:, di::2, dj::2] for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1))], -1).reshape(batch, -1, 4 * channels)
            first, second = ("norm", "reduction") if pid == 29 else ("reduction", "norm")
            x = context.call(path + "." + second, context.call(path + "." + first, x))
    x = context.call("norm", x).mean(1)
    return context.call("head", x)


def execute(context, pid, args):
    if 33 <= pid <= 42:
        return recurrent_model(context, pid, args)
    if pid in {43, 44, 50}:
        return gpt(context, pid, args[0])
    if pid in {46, 47}:
        return vlad(context, args[0])
    if pid in {48, 49}:
        return mamba(context, pid, args)
    if pid in {29, 30}:
        return swin(context, pid, args[0])
    raise ValueError("No reviewed model algorithm for this task")
