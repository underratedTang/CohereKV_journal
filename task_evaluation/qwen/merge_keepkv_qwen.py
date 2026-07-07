"""
KeepKV cache merging for Qwen2-family models.

This implementation follows the practical KeepKV mechanism used in the official
codebase:
  * Electoral Votes / cluster counts are maintained per physical KV slot.
  * Decoding attention receives a log(count) bias through the existing eager
    GroupLogitKV runtime patch used by this task_evaluation harness.
  * Evicted slots are assigned to the most cosine-similar retained slot.
  * K/V are merged with count- and attention-score-weighted estimates, matching
    the official implementation's simulated KV_update path.

The public function signatures are kept compatible with the original evaluator:
    KeepKV(model, key_value_cache, attention, hidden_states, ratio, sim_thr=-1)
    Repair_keepkv(...)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return past_key_values
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _num_kv_groups(attn_mod, h_kv: int) -> int:
    groups = getattr(attn_mod, "num_key_value_groups", None)
    if groups is not None:
        return int(groups)
    h_q = getattr(attn_mod, "num_heads", None)
    if h_q is None:
        h_q = getattr(attn_mod, "num_attention_heads", None)
    if h_q is None:
        raise ValueError("Cannot infer number of query heads for KeepKV log-bias expansion")
    if int(h_q) % int(h_kv) != 0:
        raise ValueError(f"num_heads={h_q} is not divisible by num_kv_heads={h_kv}")
    return int(h_q) // int(h_kv)


def _make_log_bias_q(counts: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Return [B, H_q, 1, L] log-count bias used by the runtime attention patch."""
    log_bias_kv = torch.log(counts.float().clamp_min(1.0))
    if num_key_value_groups != 1:
        bsz, h_kv, seqlen = log_bias_kv.shape
        log_bias_q = log_bias_kv[:, :, None, :].expand(
            bsz, h_kv, num_key_value_groups, seqlen
        ).reshape(bsz, h_kv * num_key_value_groups, seqlen)
    else:
        log_bias_q = log_bias_kv
    return log_bias_q.unsqueeze(-2).contiguous()


def _enable_keepkv_runtime_state(model, counts_by_layer: Sequence[torch.Tensor], log_bias_by_layer: Sequence[torch.Tensor]) -> None:
    """Expose KeepKV Electoral Votes to the task_evaluation eager-attention patch."""
    model._keepkv_counts = list(counts_by_layer)
    model._keepkv_log_bias_q = list(log_bias_by_layer)

    # The supplied runtime patch reads the GroupLogitKV attribute names.  KeepKV
    # deliberately populates the same runtime metadata because the mathematical
    # operation is the same Electoral Votes calibration: logits += log(count).
    model._grouplogitkv_counts = list(counts_by_layer)
    model._grouplogitkv_log_bias_q = list(log_bias_by_layer)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            attn._keepkv_eager_enabled = True
            attn._keepkv_log_bias_q = log_bias_by_layer[layer_idx]
            attn._grouplogitkv_eager_enabled = True
            attn._grouplogitkv_sdpa_enabled = True    # Qwen patch checks this flag.
            attn._grouplogitkv_log_bias_q = log_bias_by_layer[layer_idx]


def _initial_or_existing_counts(model, layer_idx: int, K_layer: torch.Tensor) -> torch.Tensor:
    """Load previous Electoral Votes; generated suffix tokens get vote 1."""
    B, H_kv, L, _ = K_layer.shape
    device = K_layer.device
    dtype = torch.float32

    counts_all = getattr(model, "_keepkv_counts", None)
    if counts_all is None:
        counts_all = getattr(model, "_grouplogitkv_counts", None)

    if counts_all is None or layer_idx >= len(counts_all) or counts_all[layer_idx] is None:
        return torch.ones((B, H_kv, L), device=device, dtype=dtype)

    counts = counts_all[layer_idx].to(device=device, dtype=dtype)
    if counts.dim() != 3 or counts.shape[0] != B or counts.shape[1] != H_kv:
        raise ValueError(
            f"Existing KeepKV counts incompatible at layer {layer_idx}: "
            f"counts={tuple(counts.shape)}, K={tuple(K_layer.shape)}"
        )
    if counts.shape[-1] < L:
        pad = torch.ones((B, H_kv, L - counts.shape[-1]), device=device, dtype=dtype)
        counts = torch.cat([counts, pad], dim=-1)
    elif counts.shape[-1] > L:
        counts = counts[..., :L]
    return counts.contiguous()


def _token_scores_for_kv_heads(layer_attention: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Aggregate [B,H_q,Q,K] attention probabilities to [B,H_kv,K]."""
    if layer_attention is None:
        raise ValueError("KeepKV requires output_attentions=True")
    if layer_attention.dim() != 4:
        raise ValueError(f"Expected attention [B,H_q,Q,K], got {tuple(layer_attention.shape)}")

    bsz, h_q, q_len, k_len = layer_attention.shape
    if h_q == num_kv_heads:
        attn_kv = layer_attention
    else:
        if h_q % num_kv_heads != 0:
            raise ValueError(f"H_q={h_q} is not divisible by H_kv={num_kv_heads}")
        groups = h_q // num_kv_heads
        attn_kv = layer_attention.view(bsz, num_kv_heads, groups, q_len, k_len).mean(dim=2)

    # Official KeepKV uses recent attention-score history.  In this post-hoc
    # evaluator the current forward's attention is the available recent window.
    return attn_kv.float().mean(dim=2).clamp_min(_EPS)


def _gather_seq_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _gather_seq_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, dim=1, index=idx)


def _merge_layer_keepkv(
    K_layer: torch.Tensor,
    V_layer: torch.Tensor,
    C_layer: torch.Tensor,
    score_layer: torch.Tensor,
    ratio: float,
    sim_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge one layer using KeepKV votes + score-weighted similarity merging."""
    B, H_kv, L, D = K_layer.shape
    if L <= 1 or ratio <= 0:
        return K_layer.contiguous(), V_layer.contiguous(), C_layer.contiguous()

    num_merge = int(L * float(ratio))
    num_merge = min(max(num_merge, 0), L - 1)
    if num_merge == 0:
        return K_layer.contiguous(), V_layer.contiguous(), C_layer.contiguous()

    M = L - num_merge
    X = B * H_kv
    K = K_layer.reshape(X, L, D)
    V = V_layer.reshape(X, L, D)
    C = C_layer.reshape(X, L).float()
    S = score_layer.reshape(X, L).float().clamp_min(_EPS)

    # Remove the lowest recent-attention slots from explicit storage.
    merge_idx = torch.topk(-S, k=num_merge, dim=-1).indices
    keep_mask = torch.ones((X, L), dtype=torch.bool, device=K_layer.device)
    keep_mask.scatter_(1, merge_idx, False)
    keep_idx = keep_mask.nonzero(as_tuple=False)[:, 1].view(X, M)

    K_keep = _gather_seq_3d(K, keep_idx).float()
    V_keep = _gather_seq_3d(V, keep_idx).float()
    C_keep = _gather_seq_2d(C, keep_idx)
    S_keep = _gather_seq_2d(S, keep_idx)

    K_evict = _gather_seq_3d(K, merge_idx).float()
    V_evict = _gather_seq_3d(V, merge_idx).float()
    C_evict = _gather_seq_2d(C, merge_idx)
    S_evict = _gather_seq_2d(S, merge_idx)

    sim = torch.bmm(
        F.normalize(K_evict, dim=-1, eps=_EPS),
        F.normalize(K_keep, dim=-1, eps=_EPS).transpose(1, 2),
    )
    best_sim, target = sim.max(dim=-1)
    if sim_thr <= -1.0:
        merge_mask = torch.ones_like(best_sim, dtype=torch.bool)
    else:
        merge_mask = best_sim >= float(sim_thr)

    # Electoral Votes: rejected slots are evicted rather than merged; accepted
    # slots transfer their vote mass to the target retained slot.
    C_evict_eff = C_evict * merge_mask.to(dtype=C_evict.dtype)

    # Official implementation weights merged K/V by estimated attention mass
    # times cluster count.  This is the practical ZIP-Merging estimator used in
    # the released code's KV_update path.
    W_keep = (C_keep * S_keep).clamp_min(_EPS)
    W_evict = (C_evict_eff * S_evict).clamp_min(0.0)

    scatter_index = target.unsqueeze(-1).expand(-1, -1, D)
    add_W = torch.zeros((X, M), device=K_layer.device, dtype=torch.float32)
    add_W.scatter_add_(dim=1, index=target, src=W_evict)

    add_C = torch.zeros((X, M), device=K_layer.device, dtype=torch.float32)
    add_C.scatter_add_(dim=1, index=target, src=C_evict_eff)

    add_K = torch.zeros((X, M, D), device=K_layer.device, dtype=torch.float32)
    add_V = torch.zeros((X, M, D), device=K_layer.device, dtype=torch.float32)
    add_K.scatter_add_(dim=1, index=scatter_index, src=W_evict.unsqueeze(-1) * K_evict)
    add_V.scatter_add_(dim=1, index=scatter_index, src=W_evict.unsqueeze(-1) * V_evict)

    W_total = (W_keep + add_W).clamp_min(_EPS)
    K_new = (W_keep.unsqueeze(-1) * K_keep + add_K) / W_total.unsqueeze(-1)
    V_new = (W_keep.unsqueeze(-1) * V_keep + add_V) / W_total.unsqueeze(-1)
    C_new = (C_keep + add_C).clamp_min(1.0)

    return (
        K_new.to(dtype=K_layer.dtype).reshape(B, H_kv, M, D).contiguous(),
        V_new.to(dtype=V_layer.dtype).reshape(B, H_kv, M, D).contiguous(),
        C_new.reshape(B, H_kv, M).contiguous(),
    )


def KeepKV(model, key_value_cache, attention, hidden_states: Optional[List[torch.Tensor]], ratio: float, sim_thr: float = -1.0):
    """Compress a legacy HF KV cache with KeepKV."""
    del hidden_states  # Attention tensors already provide the recent score estimate.

    cache = _as_legacy_tuple(key_value_cache)
    if cache is None:
        raise ValueError("key_value_cache is None")
    if attention is None:
        raise ValueError("KeepKV requires attention tensors; call the model with output_attentions=True")

    new_past = []
    new_counts = []
    new_log_bias_q = []

    for layer_idx, (K_layer, V_layer) in enumerate(cache):
        B, H_kv, L, D = K_layer.shape
        if layer_idx >= len(attention) or attention[layer_idx] is None:
            raise ValueError(f"Missing attention for layer {layer_idx}")
        scores = _token_scores_for_kv_heads(attention[layer_idx], H_kv).to(device=K_layer.device)
        if scores.shape[-1] != L:
            # This can occur with some HF versions returning attentions after a
            # cache update.  Align to the physical cache length conservatively.
            if scores.shape[-1] > L:
                scores = scores[..., :L]
            else:
                pad = torch.full((*scores.shape[:-1], L - scores.shape[-1]), _EPS, device=K_layer.device)
                scores = torch.cat([scores, pad], dim=-1)
        C_layer = _initial_or_existing_counts(model, layer_idx, K_layer)
        K_new, V_new, C_new = _merge_layer_keepkv(K_layer, V_layer, C_layer, scores, ratio, sim_thr)

        attn_mod = model.model.layers[layer_idx].self_attn
        log_bias = _make_log_bias_q(C_new, _num_kv_groups(attn_mod, H_kv)).to(
            device=K_new.device, dtype=K_new.dtype
        )
        new_past.append((K_new, V_new))
        new_counts.append(C_new)
        new_log_bias_q.append(log_bias)

    _enable_keepkv_runtime_state(model, new_counts, new_log_bias_q)
    return tuple(new_past)


def Repair_keepkv(model, key_value_cache, attention, hidden_states, ratio: float = 0.2, sim_thr: float = -1.0):
    return KeepKV(model, key_value_cache, attention, hidden_states, ratio, sim_thr)
