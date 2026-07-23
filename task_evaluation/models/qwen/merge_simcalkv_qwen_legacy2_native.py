"""SimCalKV cache compression with MHA/GQA/MQA support.

The token selection, cosine gate, key merge, and many-to-one value merge follow
SimCalKV's official LLaMA implementation. GQA/MQA support is the minimal
extension required by shared KV heads:

* attention scores belonging to the same KV head are averaged across their
  query-head group before cumulative token scoring;
* the official per-query-head exponential value weight is evaluated for every
  query head in the group, then averaged across the group.

``ratio`` is the compression ratio: ratio=0.4 removes 40% and retains 60%.
The official similarity rule is ``max_similarity > sim_thr`` and the default
threshold is 0.0.
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


def _aggregate_attention_scores_to_kv_heads(
    layer_attention: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Return cumulative token scores shaped ``(B, H_kv, K)``.

    Accepted inputs:
      * full attention probabilities ``(B, H_q, Q, K)``;
      * runtime-preaggregated scores ``(B, H_kv, K)``.
    """
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
    """Project queries for LLaMA/Qwen attention modules.

    Modern LLaMA and Qwen2 modules expose ``q_proj``. Some older Qwen modules
    expose a fused ``qkv_proj``; their query block is the first
    ``num_q_heads * head_dim`` output features.
    """
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
    query = query.reshape(
        batch_size,
        seq_len,
        num_q_heads,
        head_dim,
    ).transpose(1, 2)
    query = query[:, :, -1, :]
    return query.reshape(
        batch_size,
        num_kv_heads,
        group_size,
        head_dim,
    )


def _merge_by_token_simcal(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    """Official SimCalKV key merge equation."""
    return 0.5 * (key1 + key2) + math.log(2.0)


def _get_group_w_pair(
    query_group: torch.Tensor,
    evicted_keys: torch.Tensor,
    kept_keys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable GQA extension of the official SimCalKV value weights.

    Every query head applies the official score ``exp(q @ k.T / sqrt(D))``.
    A single common logit offset is subtracted across the complete query group
    and both key sets. This only multiplies every weight by one common positive
    scalar, so the normalized value merge is mathematically unchanged while
    avoiding FP16/BF16 overflow. All calculations and later accumulations use
    FP32.
    """
    head_dim = query_group.shape[-1]
    all_keys = torch.cat((evicted_keys, kept_keys), dim=0)
    logits = torch.matmul(
        query_group.float(),
        all_keys.float().transpose(0, 1),
    ) / math.sqrt(head_dim)

    # One global offset is essential. Per-head or per-key normalization would
    # change the relative contribution of different GQA query heads/keys.
    offset = logits.max()
    weights = torch.exp(logits - offset).clamp_min(EPS)
    group_weights = weights.mean(dim=0)

    num_evicted = evicted_keys.shape[0]
    return group_weights[:num_evicted], group_weights[num_evicted:]


def _merge_one_head(
    query_group: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    sim_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len, head_dim = keys.shape
    if seq_len <= 1:
        return keys, values
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError(f"compress ratio must satisfy 0 <= ratio < 1, got {ratio}")

    # Official LLaMA code selects at least one token whenever compression runs.
    num_evict = min(max(1, int(seq_len * float(ratio))), seq_len - 1)

    _, evict_idx = torch.topk(-scores, num_evict)
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=keys.device)
    keep_mask[evict_idx] = False
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]

    evicted_keys = keys[evict_idx]
    evicted_values = values[evict_idx]
    kept_keys = keys[keep_idx]
    kept_values = values[keep_idx]

    similarity = F.normalize(evicted_keys, dim=1) @ F.normalize(
        kept_keys, dim=1
    ).transpose(0, 1)
    max_similarity, target = similarity.max(dim=1)
    merge_mask = max_similarity > float(sim_thr)

    # Tokens selected for compression are removed even when they fail the gate,
    # matching the official LLaMA implementation.
    if not merge_mask.any():
        return kept_keys, kept_values

    evicted_keys = evicted_keys[merge_mask]
    evicted_values = evicted_values[merge_mask]
    target = target[merge_mask]
    num_keep = kept_keys.shape[0]

    # Official LLaMA many-to-one key merge: average all evicted keys assigned
    # to a retained key, then apply the SimCalKV pair merge once.
    sum_evicted_keys = torch.zeros_like(kept_keys)
    sum_evicted_keys.scatter_add_(
        0,
        target[:, None].expand(-1, head_dim),
        evicted_keys,
    )
    count = torch.zeros(num_keep, device=keys.device, dtype=keys.dtype)
    count.scatter_add_(0, target, torch.ones_like(target, dtype=keys.dtype))
    active = count > 0

    grouped_evicted_keys = torch.zeros_like(kept_keys)
    grouped_evicted_keys[active] = (
        sum_evicted_keys[active] / count[active, None]
    )
    merged_keys = kept_keys.clone()
    merged_keys[active] = _merge_by_token_simcal(
        grouped_evicted_keys[active],
        kept_keys[active],
    )

    # Official LLaMA many-to-one value merge, with only the necessary GQA
    # extension in _get_group_w.
    evicted_weights, kept_weights = _get_group_w_pair(
        query_group, evicted_keys, kept_keys
    )

    # Accumulate in FP32. Accumulating the stabilized weights or weighted
    # values in FP16 can still underflow and produce a zero denominator.
    sum_evicted_weights = torch.zeros(
        num_keep, device=keys.device, dtype=torch.float32
    )
    sum_evicted_weights.scatter_add_(0, target, evicted_weights)

    sum_weighted_evicted_values = torch.zeros(
        (num_keep, head_dim), device=values.device, dtype=torch.float32
    )
    sum_weighted_evicted_values.scatter_add_(
        0,
        target[:, None].expand(-1, head_dim),
        evicted_weights[:, None] * evicted_values.float(),
    )

    denominator = (sum_evicted_weights + kept_weights).clamp_min(EPS)
    merged_values_fp32 = (
        sum_weighted_evicted_values
        + kept_weights[:, None] * kept_values.float()
    ) / denominator[:, None]
    merged_values = merged_values_fp32.to(values.dtype)

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
            attention[layer_idx],
            num_kv_heads,
        ).to(layer_keys.device)
        query_groups = _get_query_groups_for_kv_heads(
            layer_idx,
            model,
            hidden_states,
            num_kv_heads,
        ).to(device=layer_keys.device, dtype=layer_keys.dtype)

        compressed_keys = []
        compressed_values = []
        for head_idx in range(num_kv_heads):
            new_keys, new_values = _merge_one_head(
                query_groups[0, head_idx],
                layer_keys[0, head_idx],
                layer_values[0, head_idx],
                scores[0, head_idx],
                ratio,
                sim_thr,
            )
            compressed_keys.append(new_keys)
            compressed_values.append(new_values)

        output.append(
            (
                torch.stack(compressed_keys, dim=0).unsqueeze(0),
                torch.stack(compressed_values, dim=0).unsqueeze(0),
            )
        )

    return tuple(output)
