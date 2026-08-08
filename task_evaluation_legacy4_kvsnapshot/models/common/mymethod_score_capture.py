"""Numerically stable recent base-score capture shared by LLaMA and Qwen."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch


def score_window(attn_module, q_len: int) -> int:
    value = getattr(attn_module, "_mymethod_score_window", 128)
    try:
        value = int(value)
    except Exception:
        value = 128
    return max(0, min(value, int(q_len)))


def slice_attention_mask(
    attention_mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    key_len: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.shape[-2] == 1:
        return attention_mask[:, :, :, :key_len]
    return attention_mask[:, :, q_start:q_end, :key_len]


def visibility_from_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    """Convert an HF boolean/additive mask into a broadcastable visible-pair mask."""
    if mask is None:
        return torch.ones(
            (reference.shape[0], 1, reference.shape[-2], reference.shape[-1]),
            device=reference.device,
            dtype=torch.bool,
        )
    if mask.dtype == torch.bool:
        return mask.to(device=reference.device)
    # HF additive masks use either -inf or the finite dtype minimum.  Values below
    # -1e4 are unambiguously masked for the model families supported here.
    return mask.to(device=reference.device).float() > -1.0e4


def fallback_causal_visibility(
    reference: torch.Tensor,
    q_start: int,
    q_end: int,
    key_len: int,
    full_q_len: int,
) -> torch.Tensor:
    past_len = key_len - full_q_len
    if full_q_len <= 1 or past_len < 0:
        return torch.ones(
            (reference.shape[0], 1, q_end - q_start, key_len),
            device=reference.device,
            dtype=torch.bool,
        )
    q_abs = torch.arange(q_start, q_end, device=reference.device) + past_len
    k_abs = torch.arange(key_len, device=reference.device)
    visible = k_abs.view(1, -1) <= q_abs.view(-1, 1)
    return visible.view(1, 1, q_end - q_start, key_len).expand(
        reference.shape[0], -1, -1, -1
    )


def physical_chunk_visibility(
    reference: torch.Tensor,
    q_len: int,
    key_len: int,
) -> torch.Tensor:
    """Visibility when the compressed prefix is past and current suffix is causal."""
    past_len = key_len - q_len
    if q_len <= 1 or past_len < 0:
        return torch.ones(
            (reference.shape[0], 1, q_len, key_len),
            device=reference.device,
            dtype=torch.bool,
        )
    q = torch.arange(q_len, device=reference.device).view(-1, 1)
    k = torch.arange(key_len, device=reference.device).view(1, -1)
    visible = k < past_len
    visible = visible | (k <= past_len + q)
    return visible.view(1, 1, q_len, key_len).expand(reference.shape[0], -1, -1, -1)


def scaled_base_score_rows(
    base_logits: torch.Tensor,
    visible: torch.Tensor,
    num_kv_heads: int,
) -> Tuple[torch.Tensor, float]:
    """Capture GQA-reduced ``exp(base_logits)`` under one shared log scale."""
    if base_logits.dim() != 4:
        raise ValueError(f"Expected logits [B,Hq,Q,K], got {tuple(base_logits.shape)}")
    masked = base_logits.float().masked_fill(~visible, float("-inf"))
    max_logit = masked.amax()
    if not bool(torch.isfinite(max_logit)):
        scale = 0.0
        scores = torch.zeros_like(masked)
    else:
        scale = float(max_logit.item())
        scores = torch.exp(masked - max_logit)
        scores.masked_fill_(~visible, 0.0)

    bsz, h_q, q_len, k_len = scores.shape
    if h_q == num_kv_heads:
        reduced = scores
    else:
        if h_q % num_kv_heads != 0:
            raise ValueError(f"H_q={h_q} is not divisible by H_kv={num_kv_heads}")
        groups = h_q // num_kv_heads
        reduced = scores.view(bsz, num_kv_heads, groups, q_len, k_len).mean(dim=2)
    return reduced.contiguous(), scale


def align_score_chunks(
    chunks: Iterable[torch.Tensor],
    log_scales: Iterable[float],
    empty_shape,
    device,
) -> Tuple[torch.Tensor, float]:
    chunks = list(chunks)
    scales = [float(x) for x in log_scales]
    if not chunks:
        return torch.empty(empty_shape, device=device, dtype=torch.float32), 0.0
    common = max(scales)
    aligned = []
    for tensor, scale in zip(chunks, scales):
        factor = math.exp(scale - common)
        aligned.append(tensor if factor == 1.0 else tensor * factor)
    return torch.cat(aligned, dim=-2).contiguous(), common
