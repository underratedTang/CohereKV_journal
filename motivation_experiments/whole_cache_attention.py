#!/usr/bin/env python3
from __future__ import annotations

"""Whole-cache attention evaluators used by motivation experiments.

Two equivalent views are provided:

1) ``WholePrefixAttentionEvaluator`` evaluates from explicit post-RoPE queries.
2) ``WholePrefixAttentionFromProbsEvaluator`` evaluates directly from the real
   post-softmax attention probabilities produced by the frozen long-context
   probe.  The second form is convenient for the focused v24 experiment because
   it keeps the exact same natural probe queries already used by the group-local
   analysis and does not require another query-capture pass.

For a count-weighted merged state, q^T(mean K) is the count-weighted mean of the
member logits.  Since ``log p_i = logit_i - log Z`` for one fixed query, the
query-specific ``log Z`` cancels inside that mean.  Therefore the compressed
attention score can be reconstructed exactly from the original probabilities:

    log w_G = log(C_G) + sum_i c_i log p_i / C_G.

Rejected candidates are evicted.  Prefix states that are not compressed (e.g.
probe-token KV after the frozen prefix) remain unchanged singleton states.
"""

from typing import Dict
import math
import torch
import torch.nn.functional as F


def qfloat(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().flatten(), float(q)).item())


def describe_error(err: torch.Tensor, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean_error": float(err.float().mean().item()),
        f"{prefix}_p90_error": qfloat(err, 0.90),
        f"{prefix}_p99_error": qfloat(err, 0.99),
    }


@torch.inference_mode()
def compose_whole_layer_attention_output(head_outputs, *, num_probes: int, o_proj) -> torch.Tensor:
    """Concatenate all query-head outputs and apply the actual attention o_proj.

    ``head_outputs`` contains one tensor per KV head, each laid out as
    ``[Hq_per_kv * P, D]`` in query-head-major order.  The returned tensor is
    ``[P, hidden_size]``: the attention sublayer output before residual addition.
    """
    if not head_outputs:
        raise ValueError("head_outputs is empty")
    p = int(num_probes)
    if p <= 0:
        raise ValueError("num_probes must be positive")
    blocks = []
    value_dim = None
    for out in head_outputs:
        if out is None or out.ndim != 2:
            raise ValueError("every head output must be a [Hq_per_kv*P,D] tensor")
        if int(out.shape[0]) % p != 0:
            raise ValueError("head-output rows must be divisible by num_probes")
        if value_dim is None:
            value_dim = int(out.shape[-1])
        elif int(out.shape[-1]) != value_dim:
            raise ValueError("head output dimensions differ")
        hq_per_kv = int(out.shape[0]) // p
        blocks.append(out.detach().float().reshape(hq_per_kv, p, value_dim))
    concat = torch.cat(blocks, dim=0).permute(1, 0, 2).contiguous().reshape(p, -1)
    weight = o_proj.weight.detach().float()
    bias = o_proj.bias.detach().float() if getattr(o_proj, "bias", None) is not None else None
    if int(concat.shape[-1]) != int(weight.shape[-1]):
        raise ValueError(
            f"concatenated attention width {concat.shape[-1]} != o_proj input width {weight.shape[-1]}"
        )
    concat = concat.to(weight.device)
    if bias is not None:
        bias = bias.to(weight.device)
    return F.linear(concat, weight, bias)


@torch.inference_mode()
def whole_layer_attention_relative_error(*, full_head_outputs, compressed_head_outputs,
                                         num_probes: int, o_proj,
                                         output_norm_eps: float = 1e-20) -> float:
    """One relative error for the complete attention layer after o_proj."""
    full = compose_whole_layer_attention_output(full_head_outputs, num_probes=num_probes, o_proj=o_proj)
    comp = compose_whole_layer_attention_output(compressed_head_outputs, num_probes=num_probes, o_proj=o_proj)
    denom = torch.linalg.vector_norm(full, dim=-1).clamp_min(float(output_norm_eps))
    rel = torch.linalg.vector_norm(comp - full, dim=-1) / denom
    return float(rel.float().mean().item())


class WholePrefixAttentionEvaluator:
    """Evaluate compression against the full, uncompressed prefix attention.

    queries: [Hq_per_kv, P, D]
    keys:    [L, D]
    values:  [L, Dv]
    """

    def __init__(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 output_norm_eps: float = 1e-20):
        if queries.ndim != 3 or keys.ndim != 2 or values.ndim != 2:
            raise ValueError("Expected queries [Hq,P,D], keys [L,D], values [L,Dv]")
        if int(keys.shape[0]) != int(values.shape[0]):
            raise ValueError("K/V sequence lengths differ")
        if int(queries.shape[-1]) != int(keys.shape[-1]):
            raise ValueError("Query/key head dimensions differ")
        self.q = queries.detach().float().reshape(-1, int(queries.shape[-1])).contiguous()
        self.k = keys.detach().float().contiguous()
        self.v = values.detach().float().contiguous()
        self.scale = 1.0 / math.sqrt(float(self.k.shape[-1]))
        self.output_norm_eps = float(output_norm_eps)

        logits = torch.mm(self.q, self.k.t()) * self.scale
        probs = torch.softmax(logits, dim=-1)
        self.full_output = torch.mm(probs, self.v)
        self.full_norm = torch.linalg.vector_norm(self.full_output, dim=-1).clamp_min(self.output_norm_eps)

    @torch.inference_mode()
    def compressed_output_from_mapping(self, mapping) -> torch.Tensor:
        """Return compressed pre-o_proj head outputs for an original-to-state mapping.

        This is the output-valued counterpart of ``compressed_error_from_mapping``.
        It reconstructs the current compressed states as count-weighted K/V means
        and applies the same ``log(count)`` multiplicity correction, but returns
        the attention output itself so callers can compose all query heads and
        evaluate the real whole-layer output after ``o_proj``.
        """
        owner = torch.as_tensor(mapping, dtype=torch.long, device=self.k.device).flatten()
        if owner.numel() != int(self.k.shape[0]):
            raise ValueError("mapping must cover the original frozen prefix")
        live = owner >= 0
        if not bool(live.any()):
            raise RuntimeError("mapping evicted the entire prefix")
        live_pos = live.nonzero(as_tuple=False).flatten()
        live_owner = owner.index_select(0, live_pos)
        groups = int(live_owner.max().item()) + 1

        counts = torch.zeros(groups, dtype=torch.float32, device=self.k.device)
        counts.index_add_(0, live_owner, torch.ones_like(live_owner, dtype=torch.float32))
        ksum = torch.zeros((groups, int(self.k.shape[-1])), dtype=torch.float32, device=self.k.device)
        vsum = torch.zeros((groups, int(self.v.shape[-1])), dtype=torch.float32, device=self.v.device)
        ksum.index_add_(0, live_owner, self.k.index_select(0, live_pos))
        vsum.index_add_(0, live_owner.to(self.v.device), self.v.index_select(0, live_pos.to(self.v.device)))
        mean_k = ksum / counts.clamp_min(1.0).unsqueeze(-1)
        mean_v = vsum / counts.to(self.v.device).clamp_min(1.0).unsqueeze(-1)

        logits = torch.mm(self.q, mean_k.t()) * self.scale
        logits = logits + torch.log(counts.clamp_min(1.0)).unsqueeze(0)
        return torch.mm(torch.softmax(logits, dim=-1), mean_v)

    @torch.inference_mode()
    def compressed_error_from_mapping(self, mapping) -> torch.Tensor:
        """Evaluate a multi-round partition of the original frozen prefix.

        ``mapping[t]`` is the current compressed-state id representing original
        prefix token ``t`` or -1 if token ``t`` has been permanently evicted.
        Current states are reconstructed as count-weighted arithmetic K/V means
        of their original members and receive a ``log(count)`` multiplicity bias.
        No probe-token KV is present because this evaluator is built only from
        the frozen prefix and independent future Q vectors.
        """
        owner = torch.as_tensor(mapping, dtype=torch.long, device=self.k.device).flatten()
        if owner.numel() != int(self.k.shape[0]):
            raise ValueError("mapping must cover the original frozen prefix")
        live = owner >= 0
        if not bool(live.any()):
            raise RuntimeError("mapping evicted the entire prefix")
        live_pos = live.nonzero(as_tuple=False).flatten()
        live_owner = owner.index_select(0, live_pos)
        groups = int(live_owner.max().item()) + 1

        counts = torch.zeros(groups, dtype=torch.float32, device=self.k.device)
        counts.index_add_(0, live_owner, torch.ones_like(live_owner, dtype=torch.float32))
        ksum = torch.zeros((groups, int(self.k.shape[-1])), dtype=torch.float32, device=self.k.device)
        vsum = torch.zeros((groups, int(self.v.shape[-1])), dtype=torch.float32, device=self.v.device)
        ksum.index_add_(0, live_owner, self.k.index_select(0, live_pos))
        vsum.index_add_(0, live_owner.to(self.v.device), self.v.index_select(0, live_pos.to(self.v.device)))
        mean_k = ksum / counts.clamp_min(1.0).unsqueeze(-1)
        mean_v = vsum / counts.to(self.v.device).clamp_min(1.0).unsqueeze(-1)

        logits = torch.mm(self.q, mean_k.t()) * self.scale
        logits = logits + torch.log(counts.clamp_min(1.0)).unsqueeze(0)
        out = torch.mm(torch.softmax(logits, dim=-1), mean_v)
        return (torch.linalg.vector_norm(out - self.full_output, dim=-1) / self.full_norm).flatten()

    @torch.inference_mode()
    def eviction_error(self, *, keep_idx: torch.Tensor, counts_1d: torch.Tensor) -> torch.Tensor:
        keep = keep_idx[0].to(dtype=torch.long)
        counts = counts_1d.index_select(0, keep).float().clamp_min(1.0)
        k = self.k.index_select(0, keep)
        v = self.v.index_select(0, keep)
        logits = torch.mm(self.q, k.t()) * self.scale + torch.log(counts).unsqueeze(0)
        out = torch.mm(torch.softmax(logits, dim=-1), v)
        return (torch.linalg.vector_norm(out - self.full_output, dim=-1) / self.full_norm).flatten()

    @torch.inference_mode()
    def compressed_output(self, *, keep_idx: torch.Tensor, evict_idx: torch.Tensor,
                          best_idx: torch.Tensor, accepted_mask: torch.Tensor,
                          counts_1d: torch.Tensor) -> torch.Tensor:
        """Return the compressed pre-o_proj attention-head outputs.

        Shape matches ``self.full_output``: ``[Hq_per_kv * P, Dv]``.  Figure 2
        uses this to compose all KV/query heads first and measure one whole-layer
        error after the real attention ``o_proj`` instead of averaging per-head
        relative errors.
        """
        keep = keep_idx[0].to(dtype=torch.long)
        evict = evict_idx[0].to(dtype=torch.long)
        best = best_idx.to(dtype=torch.long).flatten()
        accepted = accepted_mask.bool().flatten()
        if int(best.numel()) != int(evict.numel()) or int(accepted.numel()) != int(evict.numel()):
            raise ValueError("best_idx / accepted_mask size must match evicted states")

        counts_keep = counts_1d.index_select(0, keep).float().clamp_min(1.0)
        counts_evict = counts_1d.index_select(0, evict).float().clamp_min(1.0)
        total = counts_keep.clone()
        ksum = self.k.index_select(0, keep) * counts_keep.unsqueeze(-1)
        vsum = self.v.index_select(0, keep) * counts_keep.unsqueeze(-1)

        chosen = accepted.nonzero(as_tuple=False).flatten()
        if chosen.numel():
            pos = evict.index_select(0, chosen)
            dest = best.index_select(0, chosen)
            w = counts_evict.index_select(0, chosen)
            total.index_add_(0, dest, w)
            ksum.index_add_(0, dest, self.k.index_select(0, pos) * w.unsqueeze(-1))
            vsum.index_add_(0, dest, self.v.index_select(0, pos) * w.unsqueeze(-1))

        denom = total.clamp_min(1.0).unsqueeze(-1)
        merged_k = ksum / denom
        merged_v = vsum / denom
        logits = torch.mm(self.q, merged_k.t()) * self.scale + torch.log(total.clamp_min(1.0)).unsqueeze(0)
        return torch.mm(torch.softmax(logits, dim=-1), merged_v)

    @torch.inference_mode()
    def compressed_error(self, *, keep_idx: torch.Tensor, evict_idx: torch.Tensor,
                         best_idx: torch.Tensor, accepted_mask: torch.Tensor,
                         counts_1d: torch.Tensor) -> torch.Tensor:
        keep = keep_idx[0].to(dtype=torch.long)
        evict = evict_idx[0].to(dtype=torch.long)
        best = best_idx.to(dtype=torch.long).flatten()
        accepted = accepted_mask.bool().flatten()
        if int(best.numel()) != int(evict.numel()) or int(accepted.numel()) != int(evict.numel()):
            raise ValueError("best_idx / accepted_mask size must match evicted states")

        counts_keep = counts_1d.index_select(0, keep).float().clamp_min(1.0)
        counts_evict = counts_1d.index_select(0, evict).float().clamp_min(1.0)
        total = counts_keep.clone()
        ksum = self.k.index_select(0, keep) * counts_keep.unsqueeze(-1)
        vsum = self.v.index_select(0, keep) * counts_keep.unsqueeze(-1)

        chosen = accepted.nonzero(as_tuple=False).flatten()
        if chosen.numel():
            pos = evict.index_select(0, chosen)
            dest = best.index_select(0, chosen)
            w = counts_evict.index_select(0, chosen)
            total.index_add_(0, dest, w)
            ksum.index_add_(0, dest, self.k.index_select(0, pos) * w.unsqueeze(-1))
            vsum.index_add_(0, dest, self.v.index_select(0, pos) * w.unsqueeze(-1))

        denom = total.clamp_min(1.0).unsqueeze(-1)
        merged_k = ksum / denom
        merged_v = vsum / denom
        logits = torch.mm(self.q, merged_k.t()) * self.scale + torch.log(total.clamp_min(1.0)).unsqueeze(0)
        out = torch.mm(torch.softmax(logits, dim=-1), merged_v)
        return (torch.linalg.vector_norm(out - self.full_output, dim=-1) / self.full_norm).flatten()


class WholePrefixAttentionFromProbsEvaluator:
    """Exact current-query whole-cache counterfactual from real attention probs.

    ``attention_probs`` is [N,T] or [Hq,Q,T] and ``values`` is [T,D].  Only the
    first ``prefix_len`` positions are subject to keep/evict/merge.  Positions
    after the frozen prefix remain singleton states, which is important when the
    natural probe contains multiple autoregressive tokens.
    """

    def __init__(self, attention_probs: torch.Tensor, values: torch.Tensor, *,
                 prefix_len: int, prob_eps: float = 1e-30,
                 output_norm_eps: float = 1e-20):
        if attention_probs.ndim not in (2, 3):
            raise ValueError("attention_probs must be [N,T] or [Hq,Q,T]")
        if values.ndim != 2:
            raise ValueError("values must be [T,D]")
        a = attention_probs.detach().float().reshape(-1, int(attention_probs.shape[-1])).contiguous()
        v = values.detach().float().contiguous()
        if int(a.shape[-1]) != int(v.shape[0]):
            raise ValueError("attention/value sequence lengths differ")
        if not 1 <= int(prefix_len) <= int(v.shape[0]):
            raise ValueError("bad prefix_len")
        self.a = a
        self.v = v
        self.prefix_len = int(prefix_len)
        self.prob_eps = float(prob_eps)
        self.output_norm_eps = float(output_norm_eps)
        self.full_output = torch.mm(self.a, self.v)
        self.full_norm = torch.linalg.vector_norm(self.full_output, dim=-1).clamp_min(self.output_norm_eps)

    @torch.inference_mode()
    def compressed_error(self, *, keep_idx: torch.Tensor, evict_idx: torch.Tensor,
                         best_idx: torch.Tensor, accepted_mask: torch.Tensor,
                         counts_1d: torch.Tensor) -> torch.Tensor:
        keep = keep_idx[0].long().to(self.a.device)
        evict = evict_idx[0].long().to(self.a.device)
        best = best_idx.long().flatten().to(self.a.device)
        accepted = accepted_mask.bool().flatten().to(self.a.device)
        counts = counts_1d.detach().float().flatten().to(self.a.device)
        if counts.numel() != self.prefix_len:
            raise ValueError("counts_1d must cover exactly the frozen prefix")
        if keep.numel() + evict.numel() != self.prefix_len:
            raise ValueError("keep_idx + evict_idx must partition the frozen prefix")
        if best.numel() != evict.numel() or accepted.numel() != evict.numel():
            raise ValueError("best_idx / accepted_mask size must match evicted states")

        c_keep = counts.index_select(0, keep).clamp_min(1.0)
        c_evict = counts.index_select(0, evict).clamp_min(1.0)
        total = c_keep.clone()

        # Prefix compressed-state pseudo logits from the exact full-attention probs.
        logp = torch.log(self.a.clamp_min(self.prob_eps))
        logp_num = logp.index_select(1, keep) * c_keep.unsqueeze(0)
        v_keep = self.v.index_select(0, keep.to(self.v.device))
        vsum = v_keep * c_keep.to(self.v.device).unsqueeze(-1)

        chosen = accepted.nonzero(as_tuple=False).flatten()
        if chosen.numel():
            pos = evict.index_select(0, chosen)
            dest = best.index_select(0, chosen)
            w = c_evict.index_select(0, chosen)
            total.index_add_(0, dest, w)
            logp_num.index_add_(1, dest, logp.index_select(1, pos) * w.unsqueeze(0))
            vsum.index_add_(0, dest.to(vsum.device),
                            self.v.index_select(0, pos.to(self.v.device)) * w.to(self.v.device).unsqueeze(-1))

        merged_logw = logp_num / total.clamp_min(1.0).unsqueeze(0)
        merged_logw = merged_logw + torch.log(total.clamp_min(1.0)).unsqueeze(0)
        merged_v = vsum / total.to(vsum.device).clamp_min(1.0).unsqueeze(-1)

        # Probe-token KV after the frozen prefix is never compressed.
        if self.prefix_len < int(self.v.shape[0]):
            tail = torch.arange(self.prefix_len, int(self.v.shape[0]), device=self.a.device)
            all_logw = torch.cat([merged_logw, logp.index_select(1, tail)], dim=1)
            all_v = torch.cat([merged_v, self.v[self.prefix_len:]], dim=0)
        else:
            all_logw = merged_logw
            all_v = merged_v

        out = torch.mm(torch.softmax(all_logw, dim=-1), all_v)
        return (torch.linalg.vector_norm(out - self.full_output, dim=-1) / self.full_norm).flatten()
