"""Exact eager-attention query tiling helpers.

This module changes only the workspace schedule: the query axis is processed in
small blocks, while every block attends to the same complete key/value prefix
and uses the same causal/additive mask and fp32 softmax as Hugging Face eager
attention.  No KV compression, selector, score capture, or method-specific
state lives here.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _slice_additive_mask(
    attention_mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    key_len: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    # HF eager attention normally receives a 4-D additive causal mask.  Keep
    # support for a singleton query axis for cached single-token style masks.
    if attention_mask.ndim != 4:
        raise ValueError(
            f"Exact eager query tiling expects a 4-D additive attention mask; "
            f"got shape={tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[-2] == 1:
        return attention_mask[:, :, :, :key_len]
    return attention_mask[:, :, q_start:q_end, :key_len]


def _apply_implicit_causal_mask(
    attn_weights: torch.Tensor,
    q_start: int,
    q_end: int,
    key_len: int,
    full_q_len: int,
) -> torch.Tensor:
    """Apply causal masking only when the model did not provide an additive mask."""
    if full_q_len <= 1:
        return attn_weights
    past_len = key_len - full_q_len
    if past_len < 0:
        raise ValueError(
            f"key_len={key_len} is smaller than full_q_len={full_q_len}"
        )
    q_abs = torch.arange(q_start, q_end, device=attn_weights.device) + past_len
    k_abs = torch.arange(key_len, device=attn_weights.device)
    future = k_abs.view(1, -1) > q_abs.view(-1, 1)
    if future.any():
        attn_weights = attn_weights.masked_fill(
            future.view(1, 1, q_end - q_start, key_len),
            torch.finfo(attn_weights.dtype).min,
        )
    return attn_weights


def exact_eager_attention_output(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    head_dim: int,
    attention_mask: Optional[torch.Tensor],
    chunk_size: int,
    attention_dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Return eager attention output with bounded query-axis workspace.

    Shapes follow HF eager attention after GQA KV repetition:
      query_states: [B, H_q, Q, D]
      key_states:   [B, H_q, K, D]
      value_states: [B, H_q, K, D]

    The returned tensor is [B, H_q, Q, D].  In eval mode this is mathematically
    identical to materializing the complete [B,H_q,Q,K] eager score tensor.
    """
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value states must all be 4-D")
    q_len = int(query_states.shape[-2])
    key_len = int(key_states.shape[-2])
    if key_states.shape != value_states.shape:
        raise ValueError(
            f"key/value shape mismatch: {tuple(key_states.shape)} vs {tuple(value_states.shape)}"
        )
    if query_states.shape[:2] != key_states.shape[:2]:
        raise ValueError(
            f"query/key batch-head mismatch: {tuple(query_states.shape)} vs {tuple(key_states.shape)}"
        )
    if int(chunk_size) <= 0:
        chunk_size = q_len
    chunk_size = max(1, min(int(chunk_size), q_len))

    outputs = []
    key_t = key_states.transpose(2, 3)
    for q_start in range(0, q_len, chunk_size):
        q_end = min(q_start + chunk_size, q_len)
        q_chunk = query_states[:, :, q_start:q_end, :]
        attn_weights = torch.matmul(q_chunk, key_t) / math.sqrt(head_dim)
        mask = _slice_additive_mask(attention_mask, q_start, q_end, key_len)
        if mask is not None:
            attn_weights = attn_weights + mask
        else:
            attn_weights = _apply_implicit_causal_mask(
                attn_weights, q_start, q_end, key_len, q_len
            )
        # Match HF eager: fp32 softmax, cast back to query dtype before dropout/matmul.
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_probs = F.dropout(attn_probs, p=float(attention_dropout), training=bool(training))
        outputs.append(torch.matmul(attn_probs, value_states))
        del q_chunk, attn_weights, attn_probs
    return torch.cat(outputs, dim=2)


__all__ = ["exact_eager_attention_output"]
