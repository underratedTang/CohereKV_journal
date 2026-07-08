"""
GroupLogitKV calibrated-QK support for Qwen-family models.

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

This file implements:
  1. batched GroupLogitKV merge;
  2. similarity-threshold filtering for assigned merge slots;
  3. group count C maintenance across repeated Qwen re-compressions;
  4. precomputed query-head log(C) bias metadata;
  5. enable/disable state for the Qwen runtime attention patch.

Default behavior uses sim_thr=-1.0, which disables similarity-threshold
filtering and merges every selected evicted token into its nearest kept token.
When sim_thr > -1.0, selected evicted tokens below the threshold are removed
from the explicit cache but do not contribute to any kept slot.

Expected cache format:
    past_key_values[layer] = (K, V)
    K, V shape = [B, H_kv, L, D]

Expected attention format:
    attention[layer] shape = [B, H_q, Q, K]
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
    """Disable GroupLogitKV state for all Qwen layers."""
    model._grouplogitkv_counts = None
    model._grouplogitkv_log_bias_q = None
    model._keepkv_counts = None
    model._keepkv_log_bias_q = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = layer.self_attn
            attn._grouplogitkv_sdpa_enabled = False
            attn._grouplogitkv_eager_enabled = False
            attn._grouplogitkv_log_bias_q = None


def disable_grouplogitkv_eager(model) -> None:
    """Alias for naming consistency with the LLaMA implementation."""
    disable_grouplogitkv_sdpa(model)


def _enable_grouplogitkv_sdpa(model, new_counts, new_log_bias_q) -> None:
    """Enable GroupLogitKV state for compressed Qwen decoding."""
    model._grouplogitkv_counts = new_counts
    model._grouplogitkv_log_bias_q = new_log_bias_q

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._grouplogitkv_sdpa_enabled = True
        attn._grouplogitkv_eager_enabled = True
        attn._grouplogitkv_log_bias_q = new_log_bias_q[layer_idx]


def _infer_num_key_value_groups(attn_mod, num_kv_heads: int, layer_attention: Optional[torch.Tensor] = None) -> int:
    """Infer H_q / H_kv for Qwen variants."""
    value = getattr(attn_mod, "num_key_value_groups", None)
    if value is not None:
        return int(value)

    if layer_attention is not None and layer_attention.dim() == 4:
        h_q = int(layer_attention.shape[1])
        if h_q >= num_kv_heads and h_q % num_kv_heads == 0:
            return h_q // int(num_kv_heads)

    for name in ("num_heads", "num_attention_heads", "n_head"):
        h_q = getattr(attn_mod, name, None)
        if h_q is not None:
            h_q = int(h_q)
            if h_q >= num_kv_heads and h_q % num_kv_heads == 0:
                return h_q // int(num_kv_heads)

    raise ValueError(f"Cannot infer num_key_value_groups for num_kv_heads={num_kv_heads}")


def _make_log_bias_q(counts: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Build query-head log(C) bias for the Qwen runtime patch.

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
    if layer_attention.dim() == 3:
        # Pre-reduced token scores [B,H_kv,K] captured by the runtime patch.
        if layer_attention.shape[1] != num_kv_heads:
            raise ValueError(f"Expected pre-reduced scores with H_kv={num_kv_heads}, got {tuple(layer_attention.shape)}")
        return layer_attention
    if layer_attention.dim() != 4:
        raise ValueError(f"Expected attention [B,H,Q,K] or score [B,H_kv,K], got {tuple(layer_attention.shape)}")

    bsz, q_heads, q_len, key_len = layer_attention.shape

    if q_heads == num_kv_heads:
        return layer_attention.sum(dim=2)

    if q_heads % num_kv_heads == 0:
        groups = q_heads // num_kv_heads
        attn = layer_attention.view(bsz, num_kv_heads, groups, q_len, key_len)
        return attn.mean(dim=2).sum(dim=2)

    raise ValueError(
        f"q_heads={q_heads} is not divisible by num_kv_heads={num_kv_heads}; "
        "cannot map query-head attentions to KV heads."
    )


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched GroupLogitKV merge for one layer.

    Args:
        K_layer, V_layer: [B, H_kv, L, D]
        C_layer:          [B, H_kv, L]
        layer_scores:     [B, H_kv, L]
        ratio:            fraction of tokens to evict from the explicit cache
        sim_thr:          default -1.0 disables threshold filtering. If > -1.0,
                          only evicted tokens with nearest-kept cosine similarity
                          >= sim_thr are absorbed into kept slots.

    Returns:
        K_new, V_new, C_new with fixed length L - int(L * ratio)
    """
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
    best_sim, best_idx = sim.max(dim=-1)

    if sim_thr <= -1.0:
        merge_mask = torch.ones_like(best_sim, dtype=torch.bool)
    else:
        merge_mask = best_sim >= sim_thr

    C_n_eff = C_n * merge_mask.to(dtype=C_n.dtype)

    add_C = torch.zeros((X, M), device=K_layer.device, dtype=K_layer.dtype)
    add_C.scatter_add_(dim=1, index=best_idx, src=C_n_eff)
    C_new = C_m + add_C

    scatter_index = best_idx.unsqueeze(-1).expand(-1, -1, D)

    add_K_num = torch.zeros((X, M, D), device=K_layer.device, dtype=K_layer.dtype)
    add_V_num = torch.zeros((X, M, D), device=K_layer.device, dtype=K_layer.dtype)

    add_K_num.scatter_add_(dim=1, index=scatter_index, src=C_n_eff.unsqueeze(-1) * K_n)
    add_V_num.scatter_add_(dim=1, index=scatter_index, src=C_n_eff.unsqueeze(-1) * V_n)

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

    If the current KV cache is longer than stored counts, generated suffix tokens
    are assigned count 1.
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

    return c.contiguous()


def Repair(
    model,
    key_value_cache,
    attention,
    hidden_states: Optional[List[torch.Tensor]],
    ratio: float,
    sim_thr: float = -1.0,
):
    """
    Compress Qwen KV cache with GroupLogitKV.

    Args:
        model: Qwen model whose attention has been patched by
            grouplogitkv_runtime_patch_qwen.py.
        key_value_cache: HF past_key_values.
        attention: attentions returned by the model/generation path, one tensor per layer.
        hidden_states: unused; kept for eval-harness compatibility.
        ratio: compression ratio.
        sim_thr: default -1.0, no similarity-threshold filtering.

    Returns:
        tuple((K_new, V_new), ...), compatible with HF generation utilities.
    """
    cache = _as_legacy_tuple(key_value_cache)
    if cache is None:
        raise ValueError("key_value_cache is None")

    if attention is None:
        raise ValueError("attention is None. GroupLogitKV requires attention scores for token ranking.")

    num_layers = len(cache)
    new_past = []
    new_counts = []
    new_log_bias_q = []

    for layer_idx in range(num_layers):
        K_layer, V_layer = _get_layer_cache(cache, layer_idx)
        B, H_kv, L, D = K_layer.shape

        if layer_idx >= len(attention) or attention[layer_idx] is None:
            raise ValueError(f"Missing attention for layer {layer_idx}")

        layer_scores = _token_scores_for_kv_heads(attention[layer_idx], H_kv).to(
            device=K_layer.device,
            dtype=K_layer.dtype,
        )

        if tuple(layer_scores.shape) != (B, H_kv, L):
            raise ValueError(
                f"Layer score shape mismatch at layer {layer_idx}: "
                f"expected {(B, H_kv, L)}, got {tuple(layer_scores.shape)}. "
                "This usually means attention length and KV-cache length are not aligned."
            )

        C_layer = _initial_or_existing_counts(model, layer_idx, K_layer)

        K_new_layer, V_new_layer, C_new_layer = _batched_group_merge_layer(
            K_layer=K_layer,
            V_layer=V_layer,
            C_layer=C_layer,
            layer_scores=layer_scores,
            ratio=ratio,
            sim_thr=sim_thr,
        )

        attn_mod = model.model.layers[layer_idx].self_attn
        num_key_value_groups = _infer_num_key_value_groups(
            attn_mod=attn_mod,
            num_kv_heads=H_kv,
            layer_attention=attention[layer_idx],
        )

        log_bias_q_layer = _make_log_bias_q(
            C_new_layer,
            num_key_value_groups=num_key_value_groups,
        ).to(device=K_new_layer.device, dtype=K_new_layer.dtype)

        if log_bias_q_layer.shape[0] != B or log_bias_q_layer.shape[-1] != K_new_layer.shape[-2]:
            raise RuntimeError(
                f"Invalid GroupLogitKV log-bias shape at layer {layer_idx}: "
                f"bias={tuple(log_bias_q_layer.shape)}, K_new={tuple(K_new_layer.shape)}"
            )

        new_past.append((K_new_layer, V_new_layer))
        new_counts.append(C_new_layer)
        new_log_bias_q.append(log_bias_q_layer)

    _enable_grouplogitkv_sdpa(model, new_counts, new_log_bias_q)

    return tuple(new_past)


def Repair_qwen(
    model,
    key_value_cache,
    attention,
    hidden_states: Optional[List[torch.Tensor]],
    ratio: float = 0.2,
    sim_thr: float = -1.0,
):
    """Compatibility alias for older Qwen evaluation code."""
    return Repair(
        model=model,
        key_value_cache=key_value_cache,
        attention=attention,
        hidden_states=hidden_states,
        ratio=ratio,
        sim_thr=sim_thr,
    )
