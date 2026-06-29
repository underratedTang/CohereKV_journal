"""
GroupLogitKV eager-only for LLaMA-family models.

Method
------
GroupLogitKV compresses KV cache by group-wise assignment and count-weighted
centroid merging.

For each retained slot m and assigned merge slots n:

    C_new = C_m + sum_n C_n
    K_new = (C_m K_m + sum_n C_n K_n) / C_new
    V_new = (C_m V_m + sum_n C_n V_n) / C_new

During decoding, GroupLogitKV applies logit-space count calibration:

    attention_logits += log(C)

This file only implements:
  1. batched GroupLogitKV merge;
  2. group count C maintenance;
  3. precomputed query-head log(C) bias metadata;
  4. enable/disable state for the eager attention patch.

It does NOT monkey-patch LLaMA attention forward. Use it together with:

    patch_transformers_llama_attention_grouplogitkv_eager_inplace_clean.py

No SDPA flag is enabled here.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


_EPS = 1e-8


def _as_legacy_tuple(past_key_values):
    """Return a legacy tuple cache: tuple((K, V), ...)."""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def disable_grouplogitkv_sdpa(model) -> None:
    """
    Disable GroupLogitKV state for all layers.

    Name kept for compatibility with existing eval imports.
    """
    model._grouplogitkv_counts = None
    model._grouplogitkv_log_bias_q = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = layer.self_attn
            attn._grouplogitkv_eager_enabled = False
            attn._grouplogitkv_sdpa_enabled = False
            attn._grouplogitkv_log_bias_q = None


def disable_grouplogitkv_eager(model) -> None:
    """Alias for eager-only naming."""
    disable_grouplogitkv_sdpa(model)


def _enable_grouplogitkv_eager(model, new_counts, new_log_bias_q) -> None:
    """Enable GroupLogitKV eager-bias state for compressed decoding."""
    model._grouplogitkv_counts = new_counts
    model._grouplogitkv_log_bias_q = new_log_bias_q

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._grouplogitkv_eager_enabled = True
        attn._grouplogitkv_sdpa_enabled = False
        attn._grouplogitkv_log_bias_q = new_log_bias_q[layer_idx]


def _make_log_bias_q(counts: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Build query-head log(C) bias.

    Args:
        counts: [B, H_kv, L]
        num_key_value_groups: H_q / H_kv

    Returns:
        [B, H_q, 1, L]
    """
    log_bias_kv = torch.log(counts.clamp_min(1.0))

    if num_key_value_groups != 1:
        bsz, h_kv, seqlen = log_bias_kv.shape
        log_bias_q = log_bias_kv[:, :, None, :].expand(
            bsz, h_kv, num_key_value_groups, seqlen
        ).reshape(bsz, h_kv * num_key_value_groups, seqlen)
    else:
        log_bias_q = log_bias_kv

    return log_bias_q.unsqueeze(-2).contiguous()


def _get_layer_cache(key_value_cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cache = _as_legacy_tuple(key_value_cache)
    k, v = cache[layer_idx]
    return k, v


def _token_scores_for_kv_heads(layer_attention: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """
    Convert query-head attention to per-KV-head token scores.

    Args:
        layer_attention: [B, H_q, Q, K]
        num_kv_heads: H_kv

    Returns:
        [B, H_kv, K]
    """
    if layer_attention is None:
        raise ValueError("Missing attention tensor")
    if layer_attention.dim() != 4:
        raise ValueError(f"Expected attention [B,H,Q,K], got {tuple(layer_attention.shape)}")

    bsz, q_heads, q_len, key_len = layer_attention.shape

    if q_heads == num_kv_heads:
        return layer_attention.sum(dim=2)

    if q_heads % num_kv_heads == 0:
        groups = q_heads // num_kv_heads
        attn = layer_attention.view(bsz, num_kv_heads, groups, q_len, key_len)
        return attn.mean(dim=2).sum(dim=2)

    return layer_attention[:, :num_kv_heads].sum(dim=2)


def _gather_seq(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along sequence dimension."""
    if x.dim() == 3:
        return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if x.dim() == 2:
        return torch.gather(x, dim=1, index=idx)
    raise ValueError(f"Unsupported gather source shape: {tuple(x.shape)}")


def _batched_group_merge_layer(
    K_layer: torch.Tensor,
    V_layer: torch.Tensor,
    C_layer: torch.Tensor,
    layer_scores: torch.Tensor,
    ratio: float,
    sim_thr: float,
    fan_in: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched GroupLogitKV merge for one layer.

    Args:
        K_layer, V_layer: [B, H_kv, L, D]
        C_layer:          [B, H_kv, L]
        layer_scores:     [B, H_kv, L]
        ratio:            fraction of tokens to merge
        sim_thr:          default -1.0, no similarity-threshold filtering
        fan_in:           default 0, no fan-in cap

    Returns:
        K_new, V_new, C_new with fixed length L - int(L * ratio)
    """
    if sim_thr > -1.0:
        raise NotImplementedError(
            "GroupLogitKV currently uses sim_thr=-1.0. "
            "Similarity-threshold filtering needs a separate fixed-length policy."
        )
    if fan_in is not None and fan_in > 0:
        raise NotImplementedError(
            "GroupLogitKV currently uses fan_in=0. "
            "Fan-in constrained assignment needs a separate batched policy."
        )

    B, H_kv, L, D = K_layer.shape

    if L <= 1 or ratio <= 0:
        return K_layer.contiguous(), V_layer.contiguous(), C_layer.contiguous()

    num_merge = int(L * ratio)
    num_merge = min(max(num_merge, 0), L - 1)
    if num_merge == 0:
        return K_layer.contiguous(), V_layer.contiguous(), C_layer.contiguous()

    M = L - num_merge
    X = B * H_kv

    K_flat = K_layer.reshape(X, L, D)
    V_flat = V_layer.reshape(X, L, D)
    C_flat = C_layer.reshape(X, L)
    score_flat = layer_scores.reshape(X, L)

    merge_idx = torch.topk(-score_flat, k=num_merge, dim=-1).indices

    keep_mask = torch.ones((X, L), dtype=torch.bool, device=K_layer.device)
    keep_mask.scatter_(1, merge_idx, False)
    keep_idx = keep_mask.nonzero(as_tuple=False)[:, 1].view(X, M)

    K_m = _gather_seq(K_flat, keep_idx)
    V_m = _gather_seq(V_flat, keep_idx)
    C_m = _gather_seq(C_flat, keep_idx)

    K_n = _gather_seq(K_flat, merge_idx)
    V_n = _gather_seq(V_flat, merge_idx)
    C_n = _gather_seq(C_flat, merge_idx)

    sim = torch.bmm(
        F.normalize(K_n, dim=-1, eps=_EPS),
        F.normalize(K_m, dim=-1, eps=_EPS).transpose(1, 2),
    )
    best_idx = sim.argmax(dim=-1)

    add_C = torch.zeros((X, M), device=K_layer.device, dtype=K_layer.dtype)
    add_C.scatter_add_(dim=1, index=best_idx, src=C_n)
    C_new = C_m + add_C

    scatter_index = best_idx.unsqueeze(-1).expand(-1, -1, D)
    add_K_num = torch.zeros((X, M, D), device=K_layer.device, dtype=K_layer.dtype)
    add_V_num = torch.zeros((X, M, D), device=K_layer.device, dtype=K_layer.dtype)

    add_K_num.scatter_add_(dim=1, index=scatter_index, src=C_n.unsqueeze(-1) * K_n)
    add_V_num.scatter_add_(dim=1, index=scatter_index, src=C_n.unsqueeze(-1) * V_n)

    denom = C_new.clamp_min(_EPS).unsqueeze(-1)
    K_new = (C_m.unsqueeze(-1) * K_m + add_K_num) / denom
    V_new = (C_m.unsqueeze(-1) * V_m + add_V_num) / denom

    return (
        K_new.reshape(B, H_kv, M, D).contiguous(),
        V_new.reshape(B, H_kv, M, D).contiguous(),
        C_new.reshape(B, H_kv, M).contiguous(),
    )


def _initial_or_existing_counts(model, layer_idx: int, K_layer: torch.Tensor) -> torch.Tensor:
    """
    Get existing counts if present; otherwise initialize all counts to one.

    If a future compression round sees generated suffix tokens beyond the stored
    compressed prefix, those suffix tokens are assigned count 1.
    """
    B, H_kv, L, _ = K_layer.shape
    device = K_layer.device
    dtype = K_layer.dtype

    counts_all = getattr(model, "_grouplogitkv_counts", None)
    if counts_all is None or layer_idx >= len(counts_all) or counts_all[layer_idx] is None:
        return torch.ones((B, H_kv, L), device=device, dtype=dtype)

    c = counts_all[layer_idx].to(device=device, dtype=dtype)

    if c.dim() != 3 or c.shape[0] != B or c.shape[1] != H_kv:
        raise ValueError(
            f"Existing GroupLogitKV counts are incompatible at layer {layer_idx}: "
            f"C={tuple(c.shape)}, K={tuple(K_layer.shape)}"
        )

    if c.shape[-1] < L:
        add_len = L - c.shape[-1]
        ones = torch.ones((B, H_kv, add_len), device=device, dtype=dtype)
        c = torch.cat([c, ones], dim=-1)
    elif c.shape[-1] > L:
        c = c[..., :L]

    return c


def Repair(
    model,
    key_value_cache,
    attention,
    hidden_states: Optional[List[torch.Tensor]],
    ratio: float,
    sim_thr: float = -1.0,
    fan_in: int = 0,
):
    """
    Compress KV cache with GroupLogitKV.

    Args:
        model: LLaMA model whose eager attention has been patched for
               GroupLogitKV in-place log-bias.
        key_value_cache: HF past_key_values.
        attention: prefill/decode attentions, one tensor per layer.
        hidden_states: unused; kept for evaluation-harness compatibility.
        ratio: compression ratio.
        sim_thr: default -1.0, no similarity-threshold filtering.
        fan_in: default 0, no fan-in cap.

    Returns:
        tuple((K_new, V_new), ...), compatible with HF generation utilities.
    """
    cache = _as_legacy_tuple(key_value_cache)
    num_layers = len(cache)

    new_past = []
    new_counts = []
    new_log_bias_q = []

    for layer_idx in range(num_layers):
        K_layer, V_layer = _get_layer_cache(cache, layer_idx)
        B, H_kv, L, D = K_layer.shape

        if attention is None or layer_idx >= len(attention) or attention[layer_idx] is None:
            raise ValueError(f"Missing attention for layer {layer_idx}")

        layer_scores = _token_scores_for_kv_heads(attention[layer_idx], H_kv).to(
            device=K_layer.device,
            dtype=K_layer.dtype,
        )
        if tuple(layer_scores.shape) != (B, H_kv, L):
            raise ValueError(
                f"Layer score shape mismatch at layer {layer_idx}: "
                f"expected {(B, H_kv, L)}, got {tuple(layer_scores.shape)}"
            )

        C_layer = _initial_or_existing_counts(model, layer_idx, K_layer)

        K_new_layer, V_new_layer, C_new_layer = _batched_group_merge_layer(
            K_layer=K_layer,
            V_layer=V_layer,
            C_layer=C_layer,
            layer_scores=layer_scores,
            ratio=ratio,
            sim_thr=sim_thr,
            fan_in=fan_in,
        )

        attn_mod = model.model.layers[layer_idx].self_attn
        num_key_value_groups = getattr(attn_mod, "num_key_value_groups", None)
        if num_key_value_groups is None:
            num_heads = getattr(attn_mod, "num_heads")
            num_kv_heads = getattr(attn_mod, "num_key_value_heads")
            num_key_value_groups = num_heads // num_kv_heads

        log_bias_q_layer = _make_log_bias_q(
            C_new_layer,
            num_key_value_groups=num_key_value_groups,
        ).to(device=K_new_layer.device, dtype=K_new_layer.dtype)

        new_past.append((K_new_layer, V_new_layer))
        new_counts.append(C_new_layer)
        new_log_bias_q.append(log_bias_q_layer)

    _enable_grouplogitkv_eager(model, new_counts, new_log_bias_q)

    return tuple(new_past)
