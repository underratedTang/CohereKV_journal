"""Vectorized SimCalKV cache compression with MHA/GQA/MQA support.

The token selection, cosine gate, key merge, and many-to-one value merge follow
SimCalKV's official LLaMA implementation. Qwen uses the same complete,
vectorized many-to-one formulation. GQA/MQA support is the minimal extension
required by shared KV heads:

* attention scores for query heads sharing one KV head are averaged before
  cumulative token scoring;
* the official exponential value weight is evaluated independently for every
  query head, then aggregated across the group.

``ratio`` is the compression ratio: ratio=0.4 removes 40% and retains 60%.
The similarity rule is ``max_similarity > sim_thr`` and defaults to 0.0.

Engineering optimizations are mathematically equivalent to the head-wise code:
all KV heads are processed in one batched tensor path, and value weights use a
per-merge-group common logit offset to prevent FP16/BF16 overflow without
changing the normalized weighted average.
"""

import math
from typing import List

import torch
import torch.nn.functional as F


def _aggregate_attention_scores_to_kv_heads(
    layer_attention: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Return cumulative token scores shaped ``(B, H_kv, K)``."""
    if layer_attention.dim() == 3:
        if layer_attention.shape[1] != num_kv_heads:
            raise ValueError(
                f"Expected pre-reduced scores with H_kv={num_kv_heads}, "
                f"got {tuple(layer_attention.shape)}"
            )
        return layer_attention

    if layer_attention.dim() != 4:
        raise ValueError(
            "attention must be (B,H_q,Q,K) or (B,H_kv,K), "
            f"got {tuple(layer_attention.shape)}"
        )

    batch_size, num_q_heads, query_len, key_len = layer_attention.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"H_q={num_q_heads} is not divisible by H_kv={num_kv_heads}"
        )

    group_size = num_q_heads // num_kv_heads
    attention_by_kv = layer_attention.reshape(
        batch_size,
        num_kv_heads,
        group_size,
        query_len,
        key_len,
    ).mean(dim=2)

    # Official SimCalKV accumulates attention over all query positions.
    return attention_by_kv.sum(dim=2)


def _project_query(attn, hidden_state: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Project queries for modern LLaMA/Qwen and older fused-QKV Qwen modules."""
    if hasattr(attn, "q_proj"):
        return attn.q_proj(hidden_state)

    if hasattr(attn, "qkv_proj"):
        qkv = attn.qkv_proj(hidden_state)
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(attn, "hidden_size", None)
            if hidden_size is None or hidden_size % num_q_heads != 0:
                raise ValueError("Cannot infer head_dim for fused qkv_proj")
            head_dim = hidden_size // num_q_heads
        query_size = num_q_heads * int(head_dim)
        if qkv.shape[-1] < query_size:
            raise ValueError(
                f"qkv_proj output ({qkv.shape[-1]}) is smaller than "
                f"the expected query block ({query_size})"
            )
        return qkv[..., :query_size]

    raise AttributeError(
        f"{type(attn).__name__} exposes neither q_proj nor qkv_proj"
    )


def _get_query_groups_for_kv_heads(
    layer_idx: int,
    model,
    hidden_states: List[torch.Tensor],
    num_kv_heads: int,
) -> torch.Tensor:
    """Return last-position queries shaped ``(B, H_kv, G, D)``."""
    attn = model.model.layers[layer_idx].self_attn
    num_q_heads = getattr(
        model.config,
        "num_attention_heads",
        getattr(attn, "num_heads", None),
    )
    if num_q_heads is None:
        raise ValueError("Cannot infer number of query heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"H_q={num_q_heads} is not divisible by H_kv={num_kv_heads}"
        )

    query = _project_query(attn, hidden_states[layer_idx], num_q_heads)
    batch_size, seq_len, query_width = query.shape
    if query_width % num_q_heads != 0:
        raise ValueError(
            f"Projected query width {query_width} is not divisible by "
            f"H_q={num_q_heads}"
        )

    head_dim = query_width // num_q_heads
    group_size = num_q_heads // num_kv_heads
    query = query.reshape(batch_size, seq_len, num_q_heads, head_dim)
    query = query.transpose(1, 2)[:, :, -1, :]
    return query.reshape(batch_size, num_kv_heads, group_size, head_dim)


def _batched_gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather sequence positions independently for every head.

    ``tokens`` is ``(H,L,D)`` and ``indices`` is ``(H,N)``.
    """
    return torch.gather(
        tokens,
        1,
        indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


def _merge_all_heads(
    query_groups: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    sim_thr: float,
):
    """Vectorized equivalent of official LLaMA head-wise many-to-one merge.

    Shapes:
      query_groups: ``(H,G,D)``
      keys/values:  ``(H,L,D)``
      scores:       ``(H,L)``
    """
    num_heads, seq_len, head_dim = keys.shape
    if seq_len <= 1:
        return keys, values
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError(f"compress ratio must satisfy 0 <= ratio < 1, got {ratio}")

    # Official LLaMA behavior: at least one token is selected whenever Repair
    # is called. The evaluator itself disables Repair when ratio == 0.
    num_evict = min(max(1, int(seq_len * float(ratio))), seq_len - 1)
    num_keep = seq_len - num_evict

    evict_idx = torch.topk(-scores, num_evict, dim=1).indices
    keep_mask = torch.ones(
        (num_heads, seq_len), dtype=torch.bool, device=keys.device
    )
    keep_mask.scatter_(1, evict_idx, False)
    # nonzero is row-major, so reshaping preserves ascending original token
    # order independently within every head, matching boolean-mask indexing.
    keep_idx = keep_mask.nonzero(as_tuple=False)[:, 1].reshape(num_heads, num_keep)

    evicted_keys = _batched_gather(keys, evict_idx)
    evicted_values = _batched_gather(values, evict_idx)
    kept_keys = _batched_gather(keys, keep_idx)
    kept_values = _batched_gather(values, keep_idx)

    similarity = torch.bmm(
        F.normalize(evicted_keys, dim=2),
        F.normalize(kept_keys, dim=2).transpose(1, 2),
    )
    max_similarity, target = similarity.max(dim=2)
    merge_mask = max_similarity > float(sim_thr)

    # Flatten (head, kept-token) into one scatter domain. Adding a per-head
    # offset prevents any cross-head interaction.
    head_offset = (
        torch.arange(num_heads, device=keys.device, dtype=target.dtype)[:, None]
        * num_keep
    )
    global_target = target + head_offset
    flat_valid = merge_mask.reshape(-1)
    flat_target = global_target.reshape(-1)[flat_valid]

    # Key merge. Invalid/gated-out evicted tokens contribute nothing and remain
    # removed, exactly as in the official implementation.
    flat_evicted_keys = evicted_keys.reshape(-1, head_dim)[flat_valid]
    sum_evicted_keys = torch.zeros(
        (num_heads * num_keep, head_dim),
        device=keys.device,
        dtype=keys.dtype,
    )
    if flat_target.numel() > 0:
        sum_evicted_keys.scatter_add_(
            0, flat_target[:, None].expand(-1, head_dim), flat_evicted_keys
        )

    count = torch.zeros(
        num_heads * num_keep, device=keys.device, dtype=keys.dtype
    )
    if flat_target.numel() > 0:
        count.scatter_add_(
            0, flat_target, torch.ones_like(flat_target, dtype=keys.dtype)
        )
    active = count > 0

    flat_kept_keys = kept_keys.reshape(-1, head_dim)
    merged_keys_flat = flat_kept_keys.clone()
    if active.any():
        grouped_evicted = sum_evicted_keys[active] / count[active, None]
        merged_keys_flat[active] = (
            0.5 * (grouped_evicted + flat_kept_keys[active]) + math.log(2.0)
        )
    merged_keys = merged_keys_flat.reshape(num_heads, num_keep, head_dim)

    if flat_target.numel() == 0:
        return merged_keys, kept_values

    # GQA value weights. logsumexp evaluates sum_g exp(q_g @ k / sqrt(D))
    # without overflow. Dividing by G (mean rather than sum) is unnecessary:
    # the same constant cancels in every normalized merge for a given KV head.
    q_fp32 = query_groups.float()
    evicted_logits = torch.matmul(
        q_fp32, evicted_keys.float().transpose(1, 2)
    ) / math.sqrt(head_dim)  # (H,G,E)
    kept_logits = torch.matmul(
        q_fp32, kept_keys.float().transpose(1, 2)
    ) / math.sqrt(head_dim)  # (H,G,K)
    log_w_evicted = torch.logsumexp(evicted_logits, dim=1)  # (H,E)
    log_w_kept = torch.logsumexp(kept_logits, dim=1)  # (H,K)

    flat_log_w_evicted = log_w_evicted.reshape(-1)[flat_valid]
    flat_log_w_kept = log_w_kept.reshape(-1)

    # One common offset per actual many-to-one merge group. This is exactly
    # equivalent to the official normalized exponential weighting, while
    # guaranteeing that at least one weight in each group equals 1.
    group_max = flat_log_w_kept.clone()
    group_max.scatter_reduce_(
        0,
        flat_target,
        flat_log_w_evicted,
        reduce="amax",
        include_self=True,
    )
    evicted_weights = torch.exp(flat_log_w_evicted - group_max[flat_target])
    kept_weights = torch.exp(flat_log_w_kept - group_max)

    sum_evicted_weights = torch.zeros(
        num_heads * num_keep, device=keys.device, dtype=torch.float32
    )
    sum_evicted_weights.scatter_add_(0, flat_target, evicted_weights)

    flat_evicted_values = evicted_values.reshape(-1, head_dim)[flat_valid].float()
    sum_weighted_evicted_values = torch.zeros(
        (num_heads * num_keep, head_dim),
        device=values.device,
        dtype=torch.float32,
    )
    sum_weighted_evicted_values.scatter_add_(
        0,
        flat_target[:, None].expand(-1, head_dim),
        evicted_weights[:, None] * flat_evicted_values,
    )

    flat_kept_values = kept_values.reshape(-1, head_dim).float()
    denominator = sum_evicted_weights + kept_weights
    merged_values_flat = (
        sum_weighted_evicted_values + kept_weights[:, None] * flat_kept_values
    ) / denominator[:, None]
    merged_values = merged_values_flat.reshape(
        num_heads, num_keep, head_dim
    ).to(values.dtype)

    return merged_keys, merged_values


def Repair(
    model,
    key_value_cache,
    attention,
    hidden_states,
    ratio: float,
    sim_thr: float = 0.0,
):
    """Compress every layer of a legacy tuple KV cache."""
    if sim_thr is None:
        sim_thr = 0.0

    cache = tuple(key_value_cache)
    output = []
    for layer_idx, (layer_keys, layer_values) in enumerate(cache):
        batch_size, num_kv_heads, _, _ = layer_keys.shape
        if batch_size != 1:
            raise ValueError("This evaluator currently expects batch_size=1")

        scores = _aggregate_attention_scores_to_kv_heads(
            attention[layer_idx], num_kv_heads
        ).to(layer_keys.device)
        query_groups = _get_query_groups_for_kv_heads(
            layer_idx, model, hidden_states, num_kv_heads
        ).to(device=layer_keys.device, dtype=layer_keys.dtype)

        compressed_keys, compressed_values = _merge_all_heads(
            query_groups[0],
            layer_keys[0],
            layer_values[0],
            scores[0],
            ratio,
            sim_thr,
        )
        output.append(
            (compressed_keys.unsqueeze(0), compressed_values.unsqueeze(0))
        )

    return tuple(output)
