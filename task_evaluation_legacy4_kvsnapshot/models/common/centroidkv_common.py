# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CentroidKV core adapted from the official ``centroidkv_press.py``.

The official press is a one-shot prefill compressor.  This module preserves its
bipartite soft matching, protected sink/local regions, iterative merge schedule,
and count-weighted centroid update while exposing the function-style interface
used by this evaluation package.  For repeated PPL compression, counts are kept
across rounds solely to preserve the official centroid weighting semantics.
Counts are not added to attention logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_SINK_BUDGET = 16
DEFAULT_LOCAL_BUDGET = 64
DEFAULT_CHUNK_SIZE = 256
DEFAULT_MERGE_RATIO = 0.8
DEFAULT_DECREASE_RATE = 0.2
DEFAULT_MAX_ITERATIONS = 10
_COUNT_DTYPE = torch.int32


@dataclass(frozen=True)
class CentroidKVConfig:
    sink_budget: int = DEFAULT_SINK_BUDGET
    local_budget: int = DEFAULT_LOCAL_BUDGET
    chunk_size: int = DEFAULT_CHUNK_SIZE
    merge_ratio: float = DEFAULT_MERGE_RATIO
    decrease_rate: float = DEFAULT_DECREASE_RATE
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def validate(self) -> "CentroidKVConfig":
        if self.sink_budget < 0:
            raise ValueError("CentroidKV sink_budget must be non-negative")
        if self.local_budget < 0:
            raise ValueError("CentroidKV local_budget must be non-negative")
        if self.chunk_size < 2 or self.chunk_size % 2 != 0:
            raise ValueError("CentroidKV chunk_size must be a positive even integer >= 2")
        if not 0.0 <= self.merge_ratio <= 1.0:
            raise ValueError("CentroidKV merge_ratio must be in [0,1]")
        if self.decrease_rate < 0.0:
            raise ValueError("CentroidKV decrease_rate must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("CentroidKV max_iterations must be positive")
        return self


def as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def bipartite_soft_matching(
    key: torch.Tensor,
    n_ex: int,
    n_chunks: int,
    chunk_size: int,
    r: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official CentroidKV bipartite soft matching.

    ``key`` has shape ``[B,H,L,D]``. Even-indexed nodes are sources and
    odd-indexed nodes are destinations inside each chunk. Each source selects
    its most similar destination; the globally strongest ``r`` source edges are
    merged. The remaining source nodes and every destination node are retained.
    """
    bsz, n_head, _, head_dim = key.shape
    key = F.normalize(key.float(), p=2.0, dim=-1, eps=1e-12)

    if n_ex == 0:
        key_chunks = key.view(bsz, n_head, n_chunks, chunk_size, head_dim)
        ex_node_max = None
        ex_node_idx = None
    else:
        key_chunks = key[:, :, :-n_ex, :].view(
            bsz, n_head, n_chunks, chunk_size, head_dim
        )
        key_ex = key[:, :, -n_ex:, :]
        ex_a, ex_b = key_ex[:, :, ::2, :], key_ex[:, :, 1::2, :]
        scores_ex = ex_a @ ex_b.transpose(-1, -2)
        ex_node_max, ex_node_idx = scores_ex.max(dim=-1)

    a = key_chunks[:, :, :, ::2, :]
    b = key_chunks[:, :, :, 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    sort_score = node_max.reshape(bsz, n_head, -1)
    if n_ex != 0:
        sort_score = torch.cat((sort_score, ex_node_max), dim=2)

    # Match the official implementation's global descending ordering.
    edge_idx = torch.argsort(sort_score, dim=-1, descending=True).unsqueeze(-1)
    unm_idx = edge_idx[:, :, r:, :]
    src_idx = edge_idx[:, :, :r, :]

    chunk_offset = src_idx // (chunk_size // 2)
    local_idx = node_idx.reshape(bsz, n_head, -1, 1)
    if n_ex != 0:
        local_idx = torch.cat((local_idx, ex_node_idx.unsqueeze(-1)), dim=2)
    local_offset = local_idx.gather(dim=-2, index=src_idx)
    dst_idx = chunk_offset * (chunk_size // 2) + local_offset
    return unm_idx, src_idx, dst_idx


def _split_regions(
    tensor: torch.Tensor,
    n_sink: int,
    n_local: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sink = tensor[:, :, :n_sink]
    if n_local > 0:
        local = tensor[:, :, -n_local:]
        middle = tensor[:, :, n_sink:-n_local]
    else:
        local = tensor[:, :, :0]
        middle = tensor[:, :, n_sink:]
    return sink, middle, local


def _bsmerge_step(
    key: torch.Tensor,
    value: torch.Tensor,
    count: torch.Tensor,
    *,
    n_local: int,
    n_sink: int,
    chunk_size: int,
    max_merge: int,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """One official-style CentroidKV merge step."""
    bsz, n_head, n_total, head_dim = key.shape
    n_sink = min(max(int(n_sink), 0), n_total)
    n_local = min(max(int(n_local), 0), max(n_total - n_sink, 0))
    middle_len = n_total - n_sink - n_local
    if middle_len < 2 or max_merge <= 0:
        return key, value, count, False

    n_ex = middle_len % chunk_size
    if n_ex % 2 != 0:
        n_ex -= 1
        n_local += 1
        middle_len -= 1
    if middle_len < 2:
        return key, value, count, False

    n_chunks = middle_len // chunk_size
    effective_chunk = chunk_size
    if n_chunks == 0:
        effective_chunk = middle_len
        n_chunks = 1
        n_ex = 0
    if effective_chunk < 2 or effective_chunk % 2 != 0:
        return key, value, count, False

    sink_key, merge_key, local_key = _split_regions(key, n_sink, n_local)
    sink_value, merge_value, local_value = _split_regions(value, n_sink, n_local)
    sink_count, merge_count, local_count = _split_regions(count, n_sink, n_local)

    n_merge = min(int(max_merge), int(merge_key.shape[-2] * float(ratio) / 2.0))
    unm_idx, src_idx, dst_idx = bipartite_soft_matching(
        merge_key, n_ex, n_chunks, effective_chunk, n_merge
    )
    a_key = merge_key[:, :, ::2, :]
    a_value = merge_value[:, :, ::2, :]
    a_count = merge_count[:, :, ::2]
    b_key = merge_key[:, :, 1::2, :]
    b_value = merge_value[:, :, 1::2, :]
    b_count = merge_count[:, :, 1::2]

    vector_unm = unm_idx.expand(*unm_idx.shape[:-1], head_dim)
    vector_src = src_idx.expand(*src_idx.shape[:-1], head_dim)
    preserved_key = a_key.gather(dim=-2, index=vector_unm)
    preserved_value = a_value.gather(dim=-2, index=vector_unm)
    preserved_count = a_count.gather(dim=-1, index=unm_idx[..., 0])

    src_key = a_key.gather(dim=-2, index=vector_src).float()
    src_value = a_value.gather(dim=-2, index=vector_src).float()
    src_count = a_count.gather(dim=-1, index=src_idx[..., 0]).to(_COUNT_DTYPE)
    dst_count = b_count.to(_COUNT_DTYPE).clone()
    dst_count.scatter_add_(-1, dst_idx[..., 0], src_count)

    dst_key_num = b_key.float() * b_count.float().unsqueeze(-1)
    dst_value_num = b_value.float() * b_count.float().unsqueeze(-1)
    dst_key_num.scatter_add_(
        -2, dst_idx.expand(*dst_idx.shape[:-1], head_dim),
        src_key * src_count.float().unsqueeze(-1),
    )
    dst_value_num.scatter_add_(
        -2, dst_idx.expand(*dst_idx.shape[:-1], head_dim),
        src_value * src_count.float().unsqueeze(-1),
    )
    denom = dst_count.float().clamp_min(1.0).unsqueeze(-1)
    dst_key = (dst_key_num / denom).to(key.dtype)
    dst_value = (dst_value_num / denom).to(value.dtype)

    new_count = torch.cat((sink_count, preserved_count, dst_count, local_count), dim=-1)
    new_key = torch.cat((sink_key, preserved_key, dst_key, local_key), dim=-2)
    new_value = torch.cat((sink_value, preserved_value, dst_value, local_value), dim=-2)
    return (
        new_key.contiguous(), new_value.contiguous(), new_count.contiguous(),
        bool(n_merge > 0),
    )


def compress_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    count: torch.Tensor,
    target_length: int,
    config: CentroidKVConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively compress one layer toward ``target_length``."""
    config.validate()
    current_key = key
    current_value = value
    current_count = count.to(_COUNT_DTYPE)
    current_length = int(key.shape[-2])
    target_length = min(max(int(target_length), 1), current_length)
    # The official method never merges the protected sink/local regions, so
    # their combined size is a structural lower bound on the physical cache.
    # Clamp once here to avoid futile merge iterations for an impossible target.
    protected_floor = min(
        current_length, int(config.sink_budget) + int(config.local_budget)
    )
    target_length = max(target_length, protected_floor)
    if current_length <= target_length:
        return current_key.contiguous(), current_value.contiguous(), current_count.contiguous()

    for merge_step in range(config.max_iterations):
        if current_length <= target_length:
            break
        if config.decrease_rate == 0:
            effective_step = merge_step
        else:
            max_linear_steps = max(
                0, math.floor((config.merge_ratio - 1e-6) / config.decrease_rate)
            )
            effective_step = min(merge_step, max_linear_steps)
        merge_ratio = max(
            0.0, config.merge_ratio - effective_step * config.decrease_rate
        )
        current_key, current_value, current_count, progressed = _bsmerge_step(
            current_key,
            current_value,
            current_count,
            n_local=config.local_budget,
            n_sink=config.sink_budget,
            chunk_size=config.chunk_size,
            max_merge=current_length - target_length,
            ratio=merge_ratio,
        )
        new_length = int(current_key.shape[-2])
        if not progressed or new_length >= current_length:
            break
        current_length = new_length

    return current_key.contiguous(), current_value.contiguous(), current_count.contiguous()


def initial_or_existing_counts(model, layer_idx: int, keys: torch.Tensor) -> torch.Tensor:
    """Align persistent counts with a cache extended by newly generated tokens."""
    bsz, h_kv, length, _ = keys.shape
    states = getattr(model, "_centroidkv_counts", None)
    old = None if states is None or layer_idx >= len(states) else states[layer_idx]
    if old is None or old.shape[:2] != (bsz, h_kv) or old.shape[-1] > length:
        return torch.ones((bsz, h_kv, length), device=keys.device, dtype=_COUNT_DTYPE)
    old = old.to(device=keys.device, dtype=_COUNT_DTYPE)
    if old.shape[-1] == length:
        return old
    suffix = torch.ones(
        (bsz, h_kv, length - old.shape[-1]), device=keys.device, dtype=_COUNT_DTYPE
    )
    return torch.cat((old, suffix), dim=-1)


def reset_centroidkv_state(model) -> None:
    model._centroidkv_counts = None
    # CentroidKV has no decode-time count bias. Clear flags owned by the generic
    # compressed-cache eager patch so state cannot leak from another method.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = layer.self_attn
            for name in (
                "_grouplogitkv_eager_enabled",
                "_grouplogitkv_sdpa_enabled",
                "_keepkv_eager_enabled",
                "_mymethod_eager_enabled",
            ):
                setattr(attn, name, False)
            attn._grouplogitkv_log_bias_q = None
            attn._keepkv_log_bias_q = None
            attn._mymethod_counts_kv = None
            attn._kvmerge_capture_attention_scores = False
