"""Shared runtime helpers for KeepKV across Transformers versions.

KeepKV score capture order is strict:
  1. scaled QK logits
  2. causal / attention mask
  3. exp(masked float32 logits) capture
  4. log(cluster_count) electoral-vote bias
  5. softmax

Captured scores are immediately reduced from query heads to KV heads. This is
lossless with respect to the official GQA adaptation used by the external
KeepKV core and avoids retaining a full [B, Hq, Q, K] float32 tensor per layer.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable
import math

import torch
import os
from torch import nn
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _num_kv_groups(module: nn.Module, query_heads: int, count_heads: int | None = None) -> int:
    groups = getattr(module, "num_key_value_groups", None)
    if groups is not None:
        return int(groups)
    config = getattr(module, "config", None)
    if config is not None:
        n_heads = int(getattr(config, "num_attention_heads", query_heads))
        n_kv_heads = int(getattr(config, "num_key_value_heads", count_heads or n_heads))
        if n_kv_heads > 0 and n_heads % n_kv_heads == 0:
            return n_heads // n_kv_heads
    if count_heads and query_heads % count_heads == 0:
        return query_heads // count_heads
    raise RuntimeError(
        f"Cannot infer GQA group count: query_heads={query_heads}, count_heads={count_heads}."
    )


def _num_kv_heads(module: nn.Module, query_heads: int) -> int:
    value = getattr(module, "num_key_value_heads", None)
    if value is not None:
        return int(value)
    config = getattr(module, "config", None)
    if config is not None:
        value = getattr(config, "num_key_value_heads", None)
        if value is not None:
            return int(value)
    groups = _num_kv_groups(module, query_heads)
    if query_heads % groups != 0:
        raise RuntimeError(f"query_heads={query_heads} is not divisible by groups={groups}")
    return query_heads // groups


def _reduce_query_heads_to_kv_heads(module: nn.Module, scores: torch.Tensor) -> torch.Tensor:
    """Mean GQA query-head groups, yielding [B,Hkv,Q,K]."""
    bsz, hq, q_len, k_len = scores.shape
    hkv = _num_kv_heads(module, hq)
    if hq % hkv != 0:
        raise RuntimeError(f"Query heads {hq} are not divisible by KV heads {hkv}.")
    groups = hq // hkv
    if groups == 1:
        return scores
    return scores.view(bsz, hkv, groups, q_len, k_len).mean(dim=2)


def count_bias_for_logits(module: nn.Module, logits: torch.Tensor) -> torch.Tensor | None:
    """Build broadcastable log(cluster_count) bias for [B,Hq,Q,K] logits."""
    if not getattr(module, "_keepkv_enabled", False):
        return None
    if getattr(module, "_keepkv_disable_vote_bias", False):
        return None
    counts = getattr(module, "_keepkv_counts", None)
    if counts is None or logits.ndim != 4:
        return None

    counts = counts.to(device=logits.device, dtype=torch.float32)
    key_len = logits.shape[-1]
    prefix_len = counts.shape[-1]
    if prefix_len > key_len:
        raise RuntimeError(
            f"KeepKV count length {prefix_len} exceeds attention key length {key_len}."
        )
    if prefix_len < key_len:
        counts = torch.cat(
            (
                counts,
                torch.ones(
                    (*counts.shape[:-1], key_len - prefix_len),
                    device=logits.device,
                    dtype=torch.float32,
                ),
            ),
            dim=-1,
        )

    query_heads = logits.shape[1]
    count_heads = counts.shape[1]
    groups = _num_kv_groups(module, query_heads, count_heads)
    counts = counts[:, :, None, :].expand(counts.shape[0], count_heads, groups, key_len)
    counts = counts.reshape(counts.shape[0], count_heads * groups, key_len)
    if counts.shape[1] != query_heads:
        raise RuntimeError(
            f"Expanded KeepKV count heads {counts.shape[1]} do not match query heads {query_heads}."
        )
    return torch.log(counts.clamp_min(1.0)).unsqueeze(-2)


def enforce_physical_chunk_causal_mask(logits: torch.Tensor) -> torch.Tensor:
    """Apply causal masking for a compressed physical-cache layout.

    The external evaluator stores keys as::

        [compressed historical prefix][tokens from the current forward]

    Every current query may attend to all historical physical slots. Inside the
    current chunk, query ``i`` may attend only to chunk keys ``0..i``. This mask
    is applied immediately before KeepKV score capture and the actual softmax,
    so it is independent of Transformers' version-specific 4D-mask handling.
    Existing padding or user masks are preserved because only the strict upper
    triangle of the final ``q_len`` key columns is overwritten.
    """
    if logits.ndim != 4:
        return logits
    q_len = int(logits.shape[-2])
    k_len = int(logits.shape[-1])
    if q_len <= 1:
        return logits
    if k_len < q_len:
        raise RuntimeError(
            f"Cannot apply physical causal mask: key length {k_len} < query length {q_len}."
        )

    future = torch.triu(
        torch.ones((q_len, q_len), device=logits.device, dtype=torch.bool),
        diagonal=1,
    ).view(1, 1, q_len, q_len)
    local = logits[..., k_len - q_len : k_len]
    # Use -inf as in the mathematical causal mask. exp(-inf) is exactly zero,
    # which is required by KeepKV's official masked exponential-score capture.
    local = local.masked_fill(future, float("-inf"))
    return torch.cat((logits[..., : k_len - q_len], local), dim=-1)


def _assert_no_future_token_leakage(module: nn.Module, masked_logits: torch.Tensor) -> None:
    """Verify that the current chunk's strict upper triangle is causally masked.

    The current chunk occupies the final q_len key columns in the evaluator's
    physical-cache layout. A masked position may be -inf or a large negative
    finite value such as torch.finfo(dtype).min.
    """
    if not getattr(module, "_keepkv_assert_causal", False):
        return
    if masked_logits.ndim != 4 or masked_logits.shape[-2] <= 1:
        return

    q_len = int(masked_logits.shape[-2])
    k_len = int(masked_logits.shape[-1])
    if k_len < q_len:
        raise RuntimeError(f"KeepKV causal check: key length {k_len} < query length {q_len}.")

    local = masked_logits[..., k_len - q_len : k_len]
    future = torch.triu(
        torch.ones((q_len, q_len), device=local.device, dtype=torch.bool),
        diagonal=1,
    ).view(1, 1, q_len, q_len)
    values = local.masked_select(future.expand(local.shape[0], local.shape[1], -1, -1))
    if values.numel() == 0:
        return

    # Values above -1e4 are definitely not causally masked for the dtypes used
    # here. This accepts both -inf and finite dtype-min masks.
    leaked = torch.isfinite(values) & (values > -1.0e4)
    if leaked.any():
        maximum = float(values[leaked].max().item())
        raise RuntimeError(
            "KeepKV causal leakage detected before score capture: "
            f"q_len={q_len}, k_len={k_len}, max_future_logit={maximum}."
        )




def _capture_keepkv_prefill_summary(module: nn.Module, masked_logits: torch.Tensor) -> None:
    """Accumulate one-shot prefill sufficient statistics without storing QxK."""
    q_len = int(masked_logits.shape[-2])
    k_len = int(masked_logits.shape[-1])
    chunk_start = int(getattr(module, "_keepkv_summary_chunk_start", 0))
    prompt_len = int(getattr(module, "_keepkv_summary_prompt_length"))
    recent = int(getattr(module, "_keepkv_summary_recent"))
    recent_start = max(0, prompt_len - recent)
    global_pos = torch.arange(chunk_start, chunk_start + q_len, device=masked_logits.device)
    selected = global_pos >= recent_start
    if not bool(selected.any()):
        return

    selected_logits = masked_logits[..., selected, :]
    finite = selected_logits[torch.isfinite(selected_logits)]
    if finite.numel() == 0:
        raise RuntimeError("KeepKV summary capture received no finite logits")
    local_scale = float(finite.max().item())
    bsz, hq, qsel, _ = selected_logits.shape
    hkv = _num_kv_heads(module, hq)
    groups = hq // hkv
    reduced = torch.empty((bsz, hkv, qsel, k_len), device=masked_logits.device, dtype=torch.float32)
    # Process one KV-head group at a time, avoiding a persistent Hq-sized exp tensor.
    for h in range(hkv):
        block = selected_logits[:, h*groups:(h+1)*groups].float()
        reduced[:, h] = torch.exp(block - local_scale).mean(dim=1)

    selected_pos = global_pos[selected]
    rank = (selected_pos - recent_start).to(torch.float32)
    if recent <= 1:
        weights = torch.ones_like(rank)
    else:
        distance = float(getattr(module, "_keepkv_summary_distance_weight", 1.2))
        weights = 1.0 + (distance - 1.0) * rank / float(recent - 1)
    local_sum = reduced.sum(dim=-2)
    local_weighted = (reduced * weights.view(1,1,-1,1)).sum(dim=-2)

    old_sum = getattr(module, "_keepkv_summary_sum", None)
    old_weighted = getattr(module, "_keepkv_summary_weighted_sum", None)
    old_scale = getattr(module, "_keepkv_summary_log_scale", None)
    if old_sum is None:
        module._keepkv_summary_sum = local_sum
        module._keepkv_summary_weighted_sum = local_weighted
        module._keepkv_summary_log_scale = local_scale
        return

    common_scale = max(float(old_scale), local_scale)
    old_factor = math.exp(float(old_scale) - common_scale)
    new_factor = math.exp(local_scale - common_scale)
    if old_sum.shape[-1] < k_len:
        pad = k_len - old_sum.shape[-1]
        old_sum = F.pad(old_sum, (0, pad))
        old_weighted = F.pad(old_weighted, (0, pad))
    module._keepkv_summary_sum = old_sum * old_factor + local_sum * new_factor
    module._keepkv_summary_weighted_sum = old_weighted * old_factor + local_weighted * new_factor
    module._keepkv_summary_log_scale = common_scale

def capture_keepkv_scores(module: nn.Module, masked_logits: torch.Tensor) -> None:
    """Capture official KeepKV scores: exp(masked float32 logits), before vote bias.

    No per-query max subtraction is performed. The tensor is detached and reduced
    to KV heads immediately to control peak memory.
    """
    if masked_logits.ndim != 4:
        return
    if getattr(module, "_keepkv_capture_disabled", False):
        return
    _assert_no_future_token_leakage(module, masked_logits)
    if getattr(module, "_keepkv_summary_mode", False):
        _capture_keepkv_prefill_summary(module, masked_logits)
        module._keepkv_masked_exp_scores = None
        module._keepkv_score_log_scale = None
        return
    # The KeepKV patch is installed only for KeepKV runs. Capture the first
    # forward unconditionally: no cluster-count state exists before the first
    # compression, so gating capture on _keepkv_enabled creates a bootstrap
    # deadlock. Electoral-vote bias remains gated separately by counts.
    module._keepkv_enabled = True
    with torch.no_grad():
        logits_fp64 = masked_logits.detach().to(torch.float32)
        finite_logits = logits_fp64[torch.isfinite(logits_fp64)]
        if finite_logits.numel() == 0:
            raise RuntimeError("KeepKV score capture received no finite attention logits.")

        # Store exp(logits) in a numerically exact scaled representation:
        #   scaled_scores = exp(logits - log_scale)
        # with one scalar log_scale shared by the whole layer/forward.  A
        # common positive factor does not change official KeepKV top-k, weighted
        # sums, means, or merge ratios.  Unlike per-query max subtraction, this
        # preserves relative magnitudes across recent queries.
        log_scale = float(finite_logits.max().item())
        scores = torch.exp(logits_fp64 - log_scale)
        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "KeepKV scaled exponential score contains NaN/Inf: "
                f"log_scale={log_scale}."
            )
        if getattr(module, "_keepkv_assert_causal", False) and scores.shape[-2] > 1:
            q_len = int(scores.shape[-2])
            k_len = int(scores.shape[-1])
            local_scores = scores[..., k_len - q_len : k_len]
            future = torch.triu(
                torch.ones((q_len, q_len), device=scores.device, dtype=torch.bool),
                diagonal=1,
            ).view(1, 1, q_len, q_len)
            leaked_scores = local_scores.masked_select(
                future.expand(local_scores.shape[0], local_scores.shape[1], -1, -1)
            )
            if leaked_scores.numel() and torch.any(leaked_scores != 0):
                raise RuntimeError(
                    "Future tokens entered KeepKV importance: "
                    f"max_future_score={float(leaked_scores.max().item())}."
                )
        scores = _reduce_query_heads_to_kv_heads(module, scores)
        module._keepkv_masked_exp_scores = scores
        module._keepkv_score_log_scale = log_scale


def keepkv_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Transformers versions exposing module-level eager_attention_forward."""
    groups = _num_kv_groups(module, query.shape[1], key.shape[1])
    key_states = repeat_kv(key, groups)
    value_states = repeat_kv(value, groups)
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # Official order: mask -> score capture -> electoral-vote bias -> softmax.
    if attention_mask is not None:
        logits = logits + attention_mask
    logits = enforce_physical_chunk_causal_mask(logits)
    capture_keepkv_scores(module, logits)
    bias = count_bias_for_logits(module, logits)
    if bias is not None:
        logits = logits + bias.to(logits.dtype)

    attn_weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@contextmanager
def intercept_attention_softmax(module: nn.Module):
    """Intercept softmax in older class-based eager attention implementations.

    In these implementations the softmax input already contains the causal /
    attention mask. Capture therefore occurs directly on that input, before the
    electoral-vote bias is added.
    """
    original_softmax = F.softmax

    def patched_softmax(input: torch.Tensor, *args, **kwargs):
        candidate = input
        if (
            torch.is_tensor(candidate)
            and candidate.ndim == 4
        ):
            candidate = enforce_physical_chunk_causal_mask(candidate)
            capture_keepkv_scores(module, candidate)
            bias = count_bias_for_logits(module, candidate)
            if bias is not None:
                candidate = candidate + bias.to(candidate.dtype)
        return original_softmax(candidate, *args, **kwargs)

    F.softmax = patched_softmax
    try:
        yield
    finally:
        F.softmax = original_softmax


def make_class_forward_wrapper(original_forward: Callable) -> Callable:
    def wrapped_forward(self, *args, **kwargs):
        # This wrapper is installed only for a KeepKV evaluation process. The
        # first forward must be intercepted before any KeepKV state exists.
        self._keepkv_enabled = True
        with intercept_attention_softmax(self):
            return original_forward(self, *args, **kwargs)

    wrapped_forward.__name__ = getattr(original_forward, "__name__", "forward")
    wrapped_forward.__doc__ = getattr(original_forward, "__doc__", None)
    wrapped_forward._keepkv_original_forward = original_forward
    return wrapped_forward
