"""
CaM cache compressor for LLaMA-style Hugging Face models.

This implementation follows the CaM paper / official code semantics used with
StreamingLLM-style caches:
  * keep a start window and a recent window;
  * evict the middle cache slots;
  * merge only VALUE cache, never KEY cache;
  * for each evicted slot i, merge V_i / m into the following local window of
    m slots with Bernoulli(clamp(A_i / avg(A_local), 0, 1)).

Integration note for this task_evaluation package:
  `compress_ratio` is the evicted fraction for all methods. Therefore CaM keeps
  approximately (1 - compress_ratio) of the cache, split evenly into start and
  recent windows unless explicit start/recent sizes or ratios are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn


def _as_legacy_cache(past_key_values: Any):
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()

        def restore(x):
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache.from_legacy_cache(tuple(x))
            except Exception:
                return tuple(x)

        return legacy, restore

    return past_key_values, lambda x: tuple(x)


def _get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot find decoder layers at model.model.layers or model.layers")


def _layer_item(items: Optional[Sequence[torch.Tensor]], layer_idx: int):
    if items is None:
        return None
    return items[layer_idx]


def _aggregate_attention_to_kv_heads(attentions: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Map query-head attentions to KV-head attentions for MHA/GQA/MQA."""
    if attentions.dim() != 4:
        raise ValueError(
            f"attentions must have shape (batch, q_heads, q_len, kv_len); got {tuple(attentions.shape)}"
        )
    bsz, num_q_heads, q_len, kv_len = attentions.shape
    if num_q_heads == num_kv_heads:
        return attentions
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_q_heads} is not divisible by num_kv_heads={num_kv_heads}"
        )
    groups = num_q_heads // num_kv_heads
    return attentions.view(bsz, num_kv_heads, groups, q_len, kv_len).mean(dim=2)


@dataclass
class CaMPress:
    """Official-CaM-style start+recent value-cache merging.

    In this evaluation package, `compression_ratio` is the evicted fraction, not
    the retained budget. Example: compression_ratio=0.4 keeps about 60% of the
    current KV slots.
    """

    compression_ratio: float = 0.4
    start_ratio: Optional[float] = None
    recent_ratio: Optional[float] = None
    start_size: Optional[int] = None
    recent_size: Optional[int] = None
    eps: float = 1e-8
    stochastic: bool = True

    def _budgets(self, seq_len: int) -> Tuple[int, int]:
        if seq_len <= 1:
            return seq_len, 0

        if self.start_size is not None or self.recent_size is not None:
            start = int(self.start_size or 0)
            recent = int(self.recent_size or 0)
        elif self.start_ratio is not None or self.recent_ratio is not None:
            start = int(torch.ceil(torch.tensor(seq_len * float(self.start_ratio or 0.0))).item())
            recent = int(torch.ceil(torch.tensor(seq_len * float(self.recent_ratio or 0.0))).item())
        else:
            # Match the package-wide ratio semantics: remove floor(ratio * L),
            # keep the rest, then split evenly into start + recent windows.
            evict = int(seq_len * float(self.compression_ratio))
            keep_total = seq_len - evict
            keep_total = max(1, min(seq_len, keep_total))
            start = keep_total // 2
            recent = keep_total - start

        start = max(0, min(int(start), seq_len))
        recent = max(0, min(int(recent), seq_len - start))
        if start + recent >= seq_len:
            return seq_len, 0
        if recent <= 0:
            # CaM needs a local merge target. If only one slot can be kept,
            # make it a recent slot.
            recent = 1
            start = min(start, seq_len - recent - 1) if seq_len > 1 else 0
        return start, recent

    def compress(
        self,
        module: nn.Module,
        hidden_states: Optional[torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attentions is None:
            raise ValueError("CaMPress requires attention weights from output_attentions=True")
        if keys.dim() != 4 or values.dim() != 4:
            raise ValueError("keys and values must have shape (batch, kv_heads, seq_len, head_dim)")
        if keys.shape != values.shape:
            raise ValueError(f"keys and values must have identical shape; got {keys.shape} vs {values.shape}")

        bsz, kv_heads, seq_len, head_dim = keys.shape
        start_size, recent_size = self._budgets(seq_len)
        if recent_size == 0 or start_size + recent_size >= seq_len:
            return keys, values

        attn = _aggregate_attention_to_kv_heads(attentions.to(dtype=torch.float32), kv_heads)
        if attn.shape[0] != bsz or attn.shape[-1] != seq_len:
            raise ValueError(
                f"attention shape {tuple(attn.shape)} is incompatible with cache shape {tuple(keys.shape)}"
            )

        # Abar proxy available in this chunked PPL setting: mean attention over
        # query positions returned by the current forward chunk.
        score = attn.mean(dim=2)  # (batch, kv_heads, seq_len)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        merged_values = values.clone()
        last_evict = seq_len - recent_size
        for evict_idx in range(start_size, last_evict):
            # CaM merges into the following local tokens, k = j, ..., j + m.
            local_start = evict_idx + 1
            local_end = min(evict_idx + 1 + recent_size, seq_len)
            if local_start >= local_end:
                continue
            local_score = score[:, :, local_start:local_end]
            denom = local_score.mean(dim=-1).clamp_min(self.eps)
            prob = (score[:, :, evict_idx] / denom).clamp(0.0, 1.0)
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
            mask = torch.bernoulli(prob) if self.stochastic else prob

            m = float(local_end - local_start)
            delta = merged_values[:, :, evict_idx, :] * mask.unsqueeze(-1) / m
            merged_values[:, :, local_start:local_end, :] += delta.unsqueeze(2)

        new_keys = torch.cat(
            [keys[:, :, :start_size, :], keys[:, :, seq_len - recent_size : seq_len, :]],
            dim=2,
        )
        new_values = torch.cat(
            [merged_values[:, :, :start_size, :], merged_values[:, :, seq_len - recent_size : seq_len, :]],
            dim=2,
        )
        return new_keys, new_values


def Repair(
    model: nn.Module,
    past_key_values,
    attention,
    hidden_states=None,
    compress_ratio: float = 0.4,
    sim_thr: float = -1.0,
    *,
    start_ratio: Optional[float] = None,
    recent_ratio: Optional[float] = None,
    start_size: Optional[int] = None,
    recent_size: Optional[int] = None,
    stochastic: bool = True,
):
    """Compress a legacy or DynamicCache KV cache using CaM.

    `sim_thr` is ignored; it is kept only for evaluator API compatibility.
    """
    legacy_cache, restore = _as_legacy_cache(past_key_values)
    layers = _get_layers(model)
    compressor = CaMPress(
        compression_ratio=compress_ratio,
        start_ratio=start_ratio,
        recent_ratio=recent_ratio,
        start_size=start_size,
        recent_size=recent_size,
        stochastic=stochastic,
    )

    new_past_key_values = []
    for layer_idx, layer_cache in enumerate(legacy_cache):
        layer_keys, layer_values = layer_cache[:2]
        module = layers[layer_idx].self_attn if hasattr(layers[layer_idx], "self_attn") else layers[layer_idx]
        layer_attn = _layer_item(attention, layer_idx)
        layer_hidden = _layer_item(hidden_states, layer_idx)
        new_k, new_v = compressor.compress(
            module=module,
            hidden_states=layer_hidden,
            keys=layer_keys,
            values=layer_values,
            attentions=layer_attn,
            kwargs={},
        )
        if len(layer_cache) > 2:
            new_past_key_values.append((new_k, new_v, *layer_cache[2:]))
        else:
            new_past_key_values.append((new_k, new_v))
    return restore(new_past_key_values)
