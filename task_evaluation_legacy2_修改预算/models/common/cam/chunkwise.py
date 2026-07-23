"""Chunk-wise adapter of the official CaM KV-cache merge rule.

The merge rule follows generate_task/streaming_llm/kv_cache_cam.py:
  * keep a start region and a recent region;
  * evict the token immediately before the recent region;
  * compute a Bernoulli merge mask from accumulated attention;
  * distribute the evicted VALUE uniformly to the next ``merge_size`` tokens;
  * crop KEY directly, and crop VALUE after merging.

The only evaluation-protocol adaptation is that attentions from one input chunk are
summed over the query dimension before compression is performed once at chunk end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

LayerKV = Tuple[torch.Tensor, torch.Tensor]
LegacyCache = Tuple[LayerKV, ...]


@dataclass(frozen=True)
class CamChunkwiseConfig:
    start_size: int
    recent_size: int
    merge_size: int = 16

    @property
    def cache_size(self) -> int:
        return self.start_size + self.recent_size

    def validate(self) -> None:
        if self.start_size < 0:
            raise ValueError(f"CaM start_size must be >= 0, got {self.start_size}")
        if self.recent_size <= 0:
            raise ValueError(f"CaM recent_size must be > 0, got {self.recent_size}")
        if self.merge_size <= 0:
            raise ValueError(f"CaM merge_size must be > 0, got {self.merge_size}")
        if self.recent_size < self.merge_size:
            raise ValueError(
                "CaM recent_size must be >= merge_size because the official rule "
                f"merges into the next {self.merge_size} recent tokens; "
                f"got recent_size={self.recent_size}."
            )


@dataclass
class _CamState:
    accumulated_attention: list[Optional[torch.Tensor]]


def reset_cam_state(model) -> None:
    """Clear sequence-local accumulated attention between PPL windows."""
    if hasattr(model, "_cam_chunkwise_state"):
        delattr(model, "_cam_chunkwise_state")


def _get_state(model, num_layers: int) -> _CamState:
    state = getattr(model, "_cam_chunkwise_state", None)
    if state is None or len(state.accumulated_attention) != num_layers:
        state = _CamState(accumulated_attention=[None] * num_layers)
        model._cam_chunkwise_state = state
    return state


def _reduce_query_heads_to_kv_heads(
    scores: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Map [B, query_heads, K] attention scores to [B, kv_heads, K]."""
    if scores.ndim != 3:
        raise ValueError(f"Expected [B,H,K] attention scores, got {tuple(scores.shape)}")
    batch, num_query_heads, kv_len = scores.shape
    if num_query_heads == num_kv_heads:
        return scores
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "Query-head count must be divisible by KV-head count for GQA mapping: "
            f"query_heads={num_query_heads}, kv_heads={num_kv_heads}."
        )
    groups = num_query_heads // num_kv_heads
    return scores.reshape(batch, num_kv_heads, groups, kv_len).mean(dim=2)


def _accumulate_chunk_attention(
    previous: Optional[torch.Tensor],
    layer_attention: torch.Tensor,
    num_kv_heads: int,
    kv_len: int,
) -> torch.Tensor:
    """Batch the official per-query accumulation over one chunk.

    ``layer_attention`` is [B, query_heads, Q, K]. Summing over Q is equivalent
    to adding every query's attention contribution to the running score, while
    preserving the evaluator's once-per-chunk compression protocol.
    """
    if layer_attention is None:
        raise ValueError("CaM requires output_attentions=True.")
    if layer_attention.ndim != 4:
        raise ValueError(
            f"Expected attention shape [B,H,Q,K], got {tuple(layer_attention.shape)}"
        )
    if layer_attention.shape[-1] != kv_len:
        raise ValueError(
            "Attention key length and KV-cache length disagree: "
            f"attention={layer_attention.shape[-1]}, cache={kv_len}."
        )

    chunk_scores = layer_attention.float().sum(dim=-2)
    chunk_scores = _reduce_query_heads_to_kv_heads(chunk_scores, num_kv_heads)

    if previous is None:
        return chunk_scores

    previous = previous.to(device=chunk_scores.device, dtype=chunk_scores.dtype)
    old_len = previous.shape[-1]
    if old_len > kv_len:
        raise ValueError(
            f"Stored CaM attention length {old_len} exceeds current KV length {kv_len}."
        )
    return torch.cat(
        [previous + chunk_scores[..., :old_len], chunk_scores[..., old_len:]],
        dim=-1,
    )


def _merge_and_evict_once(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    accumulated_attention: torch.Tensor,
    config: CamChunkwiseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one official CaM merge/eviction step."""
    seq_len = key_states.shape[-2]
    recent_start = seq_len - config.recent_size
    evicted_index = recent_start - 1
    receiver_end = recent_start + config.merge_size

    if evicted_index < config.start_size:
        raise RuntimeError(
            "CaM cannot evict without entering the protected start region: "
            f"seq_len={seq_len}, start_size={config.start_size}, "
            f"recent_size={config.recent_size}."
        )
    if receiver_end > seq_len:
        raise RuntimeError(
            "CaM receiver window exceeds cache length: "
            f"recent_start={recent_start}, merge_size={config.merge_size}, seq_len={seq_len}."
        )

    denominator = accumulated_attention[..., recent_start:receiver_end].mean(dim=-1)
    merge_prob = accumulated_attention[..., evicted_index] / denominator
    # Official code only replaces NaNs, then clamps. Preserve that behavior while
    # making +inf/-inf deterministic under clamp.
    merge_prob = torch.nan_to_num(merge_prob, nan=0.0, posinf=1.0, neginf=0.0)
    merge_mask = torch.bernoulli(merge_prob.clamp(min=0.0, max=1.0))

    merged_values = value_states.clone()
    contribution = (
        value_states[..., evicted_index, :]
        * merge_mask.to(value_states.dtype).unsqueeze(-1)
        / config.merge_size
    )
    merged_values[..., recent_start:receiver_end, :] += contribution.unsqueeze(-2)

    keep = torch.cat(
        [
            torch.arange(evicted_index, device=key_states.device),
            torch.arange(evicted_index + 1, seq_len, device=key_states.device),
        ]
    )
    key_states = key_states.index_select(-2, keep)
    merged_values = merged_values.index_select(-2, keep)
    accumulated_attention = torch.cat(
        [
            accumulated_attention[..., :evicted_index],
            accumulated_attention[..., evicted_index + 1 :],
        ],
        dim=-1,
    )
    return key_states, merged_values, accumulated_attention


def _resolve_config(model, compress_ratio: float, current_len: int) -> CamChunkwiseConfig:
    cfg = getattr(model, "_cam_eval_config", {}) or {}
    logical_length = int(cfg.get("logical_length", current_len))
    merge_size = int(cfg.get("merge_size", 16))
    start_fraction = float(cfg.get("start_fraction", 0.1))

    retained = max(1, int(round(logical_length * (1.0 - float(compress_ratio)))))
    start_size = int(round(retained * start_fraction))
    start_size = min(max(start_size, 0), max(retained - 1, 0))
    recent_size = retained - start_size

    config = CamChunkwiseConfig(start_size, recent_size, merge_size)
    config.validate()
    return config


@torch.no_grad()
def repair_cam_chunkwise(
    model,
    past_key_values: Sequence[LayerKV],
    attentions: Sequence[torch.Tensor],
    compress_ratio: float,
) -> LegacyCache:
    """Compress a complete chunk's cache using the official CaM merge rule."""
    legacy = tuple((kv[0], kv[1]) for kv in past_key_values)
    if not legacy:
        return legacy
    if attentions is None or len(attentions) != len(legacy):
        raise ValueError(
            "CaM requires one attention tensor per KV layer; "
            f"got attentions={None if attentions is None else len(attentions)}, layers={len(legacy)}."
        )

    current_len = int(legacy[0][0].shape[-2])
    config = _resolve_config(model, compress_ratio, current_len)
    state = _get_state(model, len(legacy))

    output: list[LayerKV] = []
    for layer_idx, ((key_states, value_states), layer_attention) in enumerate(zip(legacy, attentions)):
        if key_states.shape[-2] != current_len or value_states.shape[-2] != current_len:
            raise ValueError("All CaM layers must have the same KV sequence length.")

        accumulated = _accumulate_chunk_attention(
            state.accumulated_attention[layer_idx],
            layer_attention,
            num_kv_heads=int(value_states.shape[1]),
            kv_len=current_len,
        )

        layer_k, layer_v = key_states, value_states
        while layer_k.shape[-2] > config.cache_size:
            layer_k, layer_v, accumulated = _merge_and_evict_once(
                layer_k, layer_v, accumulated, config
            )

        state.accumulated_attention[layer_idx] = accumulated.detach()
        output.append((layer_k, layer_v))

    return tuple(output)
