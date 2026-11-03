#!/usr/bin/env python3
"""Capture real post-RoPE GQA queries for Llama and Qwen2-family models.

Only for post-hoc teacher-forced evaluation; never used by routing or fusion.
For every selected layer, reconstruct future post-RoPE K from that layer's
k_proj output and captured RoPE cos/sin and compare it against the model cache.
This validates the actual positions, RoPE convention, head grouping and dtype.
"""
from __future__ import annotations
import torch

SUPPORTED_MODEL_TYPES = frozenset(('llama', 'qwen2'))


def require_supported_model_type(model_type):
    model_type = str(model_type).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise RuntimeError(
            f'Post-RoPE Query capture supports Llama and Qwen2-family (Qwen2.5); '
            f'got model_type={model_type!r}'
        )
    return model_type


def rotate_half(x):
    half = x.shape[-1] // 2
    if x.shape[-1] % 2:
        raise ValueError('RoPE head_dim must be even')
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotate_queries(projected, cos, sin, n_heads, head_dim):
    """Rotate q_proj OR k_proj using captured model RoPE: [heads,T,head_dim]."""
    if projected.ndim != 3 or projected.shape[0] != 1:
        raise RuntimeError('Require single sequence [1,T,H*D]')
    t = projected.shape[1]
    if projected.shape[-1] != n_heads * head_dim:
        raise RuntimeError(
            f'Projection has width {projected.shape[-1]}, expected {n_heads}*{head_dim}'
        )
    x = projected.reshape(1, t, n_heads, head_dim).transpose(1, 2)
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    if cos.ndim == 2:
        cos = cos[None]
    if sin.ndim == 2:
        sin = sin[None]
    expected = (1, t, head_dim)
    if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
        raise RuntimeError(
            f'Unexpected cos/sin shapes {tuple(cos.shape)}, {tuple(sin.shape)}; '
            f'expected {expected}'
        )
    return (x * cos[:, None] + rotate_half(x) * sin[:, None])[0].detach().float().cpu().contiguous()


def verify_post_rope_key(k_projection, cos, sin, cached_suffix, kv_heads, head_dim, layer):
    """Fail closed if captured position embeddings disagree with model's KV cache."""
    computed = rotate_queries(k_projection, cos, sin, kv_heads, head_dim)
    if cached_suffix.ndim != 4 or cached_suffix.shape[0] != 1:
        raise RuntimeError(f'Layer {layer}: unsupported KV cache layout {tuple(cached_suffix.shape)}')
    t = computed.shape[1]
    expected_shape = (kv_heads, t, head_dim)
    if tuple(cached_suffix.shape[1:]) != expected_shape:
        raise RuntimeError(f'Layer {layer}: cached future K shape {tuple(cached_suffix.shape[1:])} != {expected_shape}')
    actual = cached_suffix[0].detach().float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(computed).all():
        raise RuntimeError(f'Layer {layer}: nonfinite post-RoPE K')
    # The model applies RoPE in its projection dtype (usually bf16/fp16).
    # Compare with the *cached* post-RoPE K, allowing low precision rounding.
    dtype = k_projection.dtype
    if dtype == torch.bfloat16:
        atol, rtol = 3e-2, 2e-2
    elif dtype == torch.float16:
        atol, rtol = 6e-3, 6e-3
    else:
        atol, rtol = 1e-4, 1e-4
    if not torch.allclose(computed, actual, atol=atol, rtol=rtol):
        largest = float((computed - actual).abs().max().item())
        raise RuntimeError(
            f'Layer {layer}: post-RoPE capture disagrees with real future KV cache '
            f'(max_abs_diff={largest:g}, atol={atol:g}, rtol={rtol:g}). '
            'Refusing unvalidated real-Query attention replay.'
        )


@torch.no_grad()
def capture_real_future_queries(model, future_tokens, past, device, model_config, selected_layers):
    """Teacher-forced continuation; returns {layer: post-RoPE Q [Hq,T,D]}, cache.

    Reconstructs and validates future post-RoPE K before releasing Q. Captures
    global RoPE (transformers 4.46+) and per-attention RoPE (legacy layouts).
    No future Q enters compression decisions.
    """
    from core_model_utils import forward_cache_chunk, transformer_layers, cache_key_value_layers
    require_supported_model_type(model_config.model_type)
    layers = transformer_layers(model)
    q_outputs, k_outputs, rotary_global, rotary_by_layer = {}, {}, {}, {}
    handles = []
    try:
        for idx in selected_layers:
            attn = layers[idx].self_attn
            def projection_hook(store, layer):
                def hook(module, inputs, output):
                    if layer in store:
                        raise RuntimeError(f'Projection executed twice for layer {layer}')
                    store[layer] = output.detach().clone()
                return hook
            handles.append(attn.q_proj.register_forward_hook(projection_hook(q_outputs, idx)))
            handles.append(attn.k_proj.register_forward_hook(projection_hook(k_outputs, idx)))

        roots = [model, getattr(model, 'model', None)]
        global_modules = list(dict.fromkeys(
            root.rotary_emb for root in roots
            if root is not None and getattr(root, 'rotary_emb', None) is not None
        ))
        def global_rope_hook(module, inputs, output):
            if isinstance(output, (tuple, list)) and len(output) == 2:
                rotary_global['cos'], rotary_global['sin'] = (
                    output[0].detach().clone(), output[1].detach().clone()
                )
        for module in global_modules:
            handles.append(module.register_forward_hook(global_rope_hook))

        # Old Qwen2/Llama versions may compute RoPE in each attention layer.
        # Use layer-specific embeddings rather than silently reusing last layer.
        for idx in selected_layers:
            rotary = getattr(layers[idx].self_attn, 'rotary_emb', None)
            if rotary is None:
                continue
            def layer_rope_hook(layer):
                def hook(module, inputs, output):
                    if isinstance(output, (tuple, list)) and len(output) == 2:
                        rotary_by_layer[layer] = (output[0].detach().clone(), output[1].detach().clone())
                return hook
            handles.append(rotary.register_forward_hook(layer_rope_hook(idx)))

        continuation = forward_cache_chunk(model, future_tokens, past, device)
        new_past = continuation.past_key_values
        del continuation
    finally:
        for handle in handles:
            handle.remove()

    n_q = int(model_config.num_attention_heads)
    n_kv = int(model_config.num_key_value_heads)
    head_dim = int(getattr(model_config, 'head_dim', 0) or model_config.hidden_size // n_q)
    if n_q % n_kv:
        raise RuntimeError(f'Noninteger GQA mapping nQ={n_q}, nKV={n_kv}')
    cached_keys, _ = cache_key_value_layers(new_past)
    t = int(future_tokens.shape[1])
    queries = {}
    for idx in selected_layers:
        if idx not in q_outputs or idx not in k_outputs:
            raise RuntimeError(f'Missing Q/K projection output for layer {idx}')
        if idx in rotary_by_layer:
            cos, sin = rotary_by_layer[idx]
        elif 'cos' in rotary_global:
            cos, sin = rotary_global['cos'], rotary_global['sin']
        else:
            raise RuntimeError(
                f'Layer {idx}: real RoPE cos/sin were not captured; '
                'refusing to replay unrotated Queries.'
            )
        if cached_keys[idx].shape[-2] < t:
            raise RuntimeError(f'Layer {idx}: continuation KV cache shorter than {t}')
        verify_post_rope_key(
            k_outputs[idx], cos, sin, cached_keys[idx][..., -t:, :],
            n_kv, head_dim, idx
        )
        queries[idx] = rotate_queries(q_outputs[idx], cos, sin, n_q, head_dim)
    return queries, new_past


@torch.no_grad()
def replay_prefix_attention(queries, prefix_k, prefix_v, counts, future_k, future_v, device, eps):
    """Causal replay: compressed original prefix + uncompressed future K/V.

    Queries: [Hq,T,D], prefix KV [N,D], future KV [T,D]. Future token t
    attends to all prefix tokens and continuation positions <= t.
    Count represented by log(C), not repeated K/V.
    """
    q = queries.to(device=device, dtype=torch.float32)
    pk = torch.as_tensor(prefix_k, dtype=torch.float32, device=device)
    pv = torch.as_tensor(prefix_v, dtype=torch.float32, device=device)
    c = torch.as_tensor(counts, dtype=torch.float32, device=device)
    fk = torch.as_tensor(future_k, dtype=torch.float32, device=device)
    fv = torch.as_tensor(future_v, dtype=torch.float32, device=device)
    if (c <= 0).any():
        raise ValueError('Counts must be >0')
    t = q.shape[1]
    if int(fk.shape[0]) != t:
        raise ValueError('Future KV length not equal to Query length')
    scale = float(q.shape[-1]) ** -.5
    left = (q @ pk.T) * scale + torch.log(c)[None, None, :]
    right = (q @ fk.T) * scale
    future_mask = torch.ones((t, t), device=device, dtype=torch.bool).tril()
    right = right.masked_fill(~future_mask[None, :, :], float('-inf'))
    probs = torch.cat((left, right), dim=-1).softmax(dim=-1)
    values = torch.cat((pv, fv), dim=0)
    return probs @ values


def output_errors(approx, exact, eps=1e-8):
    diff = approx - exact
    abs_err = torch.linalg.vector_norm(diff, dim=-1)
    ref_norm = torch.linalg.vector_norm(exact, dim=-1).clamp_min(float(eps))
    rel = (abs_err / ref_norm).float()
    stats = dict(output_rel_error=float(rel.mean().item()),
        output_rel_p90=float(torch.quantile(rel.flatten(), .9).item()),
        output_abs_error=float(abs_err.mean().item()),
        output_rmse=float(torch.sqrt((diff * diff).mean()).item()),
        reference_output_norm=float(ref_norm.mean().item()))
    per_offset = rel.mean(dim=0).detach().cpu().numpy().astype('float64')
    return stats, per_offset
