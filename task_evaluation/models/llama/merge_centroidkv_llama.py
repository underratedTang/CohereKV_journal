"""CentroidKV cache compression adapter for LLaMA models."""

from __future__ import annotations

from typing import Optional

from models.common.centroidkv_common import (
    CentroidKVConfig,
    as_legacy_tuple,
    compress_layer,
    initial_or_existing_counts,
    reset_centroidkv_state,
)


def Repair(
    model,
    key_value_cache,
    attention,
    hidden_states,
    ratio: float,
    target_length: Optional[int] = None,
    *,
    sink_budget: int = 16,
    local_budget: int = 64,
    chunk_size: int = 256,
    merge_ratio: float = 0.8,
    decrease_rate: float = 0.2,
):
    del attention, hidden_states
    cache = as_legacy_tuple(key_value_cache)
    if cache is None:
        raise ValueError("key_value_cache is None")
    cfg = CentroidKVConfig(
        sink_budget=int(sink_budget),
        local_budget=int(local_budget),
        chunk_size=int(chunk_size),
        merge_ratio=float(merge_ratio),
        decrease_rate=float(decrease_rate),
    ).validate()

    new_past = []
    new_counts = []
    for layer_idx, (keys, values) in enumerate(cache):
        length = int(keys.shape[-2])
        target = (
            min(max(int(target_length), 1), length)
            if target_length is not None
            else max(1, int(length * (1.0 - float(ratio))))
        )
        counts = initial_or_existing_counts(model, layer_idx, keys)
        knew, vnew, cnew = compress_layer(keys, values, counts, target, cfg)
        new_past.append((knew, vnew))
        new_counts.append(cnew)
    model._centroidkv_counts = new_counts
    return tuple(new_past)


__all__ = ["Repair", "reset_centroidkv_state"]
