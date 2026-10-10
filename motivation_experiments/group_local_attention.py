#!/usr/bin/env python3
from __future__ import annotations

"""候选组局部 counterfactual attention 评估。

设计目标不是替代端到端 PPL，而是把长序列中的单个真实候选组从其余 cache 中
隔离出来，避免 whole-cache 平均把组级 failure mode 稀释。

给定真实 future-query attention probabilities p_i 与 value v_i，某个原始候选组 G 的
reference contribution 为

    Z_G = sum_i p_i,
    N_G = sum_i p_i v_i.

若把选中的 current states 用 count-weighted mean K/V 合并成一个 state，则 mean-K 的
logit 可由原始 log p_i 的 count-weighted mean 精确恢复（共同的 softmax normalizer 抵消）：

    Z'_G = C * exp(sum_i c_i log p_i / C).

因此无需显式 Q，也能精确评估当前轮 mean-K/mean-V+log(C) 的 group-local
counterfactual。第一轮实验中 c_i=1；接口保留 counts 以便做合成测试。
"""

from dataclasses import dataclass
from typing import Dict
import math

import torch


@dataclass
class GroupErrors:
    numerator_rel: torch.Tensor
    conditional_rel: torch.Tensor
    mass_rel: torch.Tensor
    mass_log_abs: torch.Tensor
    global_rel: torch.Tensor
    global_rel_over_group_mass: torch.Tensor
    reference_group_mass: torch.Tensor

    def mean_dict(self, prefix: str) -> Dict[str, float]:
        def m(x: torch.Tensor) -> float:
            return float(x.float().mean().item()) if x.numel() else float("nan")

        return {
            f"{prefix}_numerator_rel_error": m(self.numerator_rel),
            f"{prefix}_conditional_rel_error": m(self.conditional_rel),
            f"{prefix}_mass_rel_error": m(self.mass_rel),
            f"{prefix}_mass_log_abs_error": m(self.mass_log_abs),
            f"{prefix}_global_rel_error": m(self.global_rel),
            f"{prefix}_global_rel_over_group_mass": m(self.global_rel_over_group_mass),
            f"{prefix}_reference_group_mass": m(self.reference_group_mass),
        }


class GroupLocalAttentionEvaluator:
    """Evaluate one candidate group under real long-context probe attentions.

    attention_probs: [R,Q,T] or [N,T], already post-softmax.
    values:          [T,Dv].

    The full sequence remains the reference; only the selected prefix-token group is
    replaced.  ``numerator_rel`` is normalized by ``sum_i p_i ||v_i||`` (rather than ||N_G||)
    to avoid cancellation-driven outliers; it and ``conditional_rel`` are group-intrinsic
    and do not vanish merely because the group has small global attention mass. ``global_rel``
    is retained as a consequence check and is expected to be attention-mass diluted.
    """

    @classmethod
    def from_queries(
        cls,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        prob_eps: float = 1e-30,
        norm_eps: float = 1e-20,
    ):
        """Build the evaluator from explicit post-RoPE future queries.

        ``queries`` is [Hq_per_kv, P, D] and ``keys``/``values`` contain only
        the frozen prefix.  Thus the P probe tokens are query sources only: they
        are never appended to the measured context and never become context for
        later probes.
        """
        if queries.ndim != 3 or keys.ndim != 2 or values.ndim != 2:
            raise ValueError("queries/keys/values must be [Hq,P,D], [T,D], [T,Dv]")
        if int(keys.shape[0]) != int(values.shape[0]):
            raise ValueError("K/V sequence lengths differ")
        if int(queries.shape[-1]) != int(keys.shape[-1]):
            raise ValueError("query/key head dimensions differ")
        q = queries.detach().float().reshape(-1, int(queries.shape[-1]))
        k = keys.detach().float()
        logits = torch.mm(q, k.t()) / math.sqrt(float(k.shape[-1]))
        probs = torch.softmax(logits, dim=-1)
        return cls(probs, values, prob_eps=prob_eps, norm_eps=norm_eps)

    def __init__(
        self,
        attention_probs: torch.Tensor,
        values: torch.Tensor,
        *,
        prob_eps: float = 1e-30,
        norm_eps: float = 1e-20,
    ):
        if attention_probs.ndim not in (2, 3):
            raise ValueError("attention_probs must be [N,T] or [R,Q,T]")
        if values.ndim != 2:
            raise ValueError("values must be [T,D]")
        a = attention_probs.detach().float()
        self.a = a.reshape(-1, int(a.shape[-1])).contiguous()
        self.v = values.detach().float().contiguous()
        if int(self.a.shape[-1]) != int(self.v.shape[0]):
            raise ValueError("attention/value sequence lengths differ")
        self.prob_eps = float(prob_eps)
        self.norm_eps = float(norm_eps)
        self.full_output = torch.mm(self.a, self.v)
        self.full_norm = torch.linalg.vector_norm(self.full_output, dim=-1).clamp_min(self.norm_eps)

    def _group_reference(self, positions: torch.Tensor):
        pos = positions.long().flatten().to(self.a.device)
        p = self.a.index_select(1, pos).clamp_min(self.prob_eps)  # [N,G]
        vv = self.v.index_select(0, pos.to(self.v.device))        # [G,D]
        mass = p.sum(dim=-1)                                      # [N]
        num = torch.mm(p, vv)                                     # [N,D]
        cond = num / mass.clamp_min(self.prob_eps).unsqueeze(-1)
        return p, vv, mass, num, cond

    @torch.inference_mode()
    def evaluate_merge(
        self,
        *,
        group_positions: torch.Tensor,
        accepted_mask: torch.Tensor,
        counts: torch.Tensor | None = None,
    ) -> GroupErrors:
        """Replace group by one merged state over anchor + accepted candidates.

        ``group_positions[0]`` is the retained anchor.  ``accepted_mask`` corresponds
        only to candidates ``group_positions[1:]``.  Rejected candidates are evicted.
        """
        pos = group_positions.long().flatten()
        if pos.numel() < 1:
            raise ValueError("group must contain an anchor")
        accepted = accepted_mask.bool().flatten()
        if accepted.numel() != pos.numel() - 1:
            raise ValueError("accepted_mask must match candidate count")
        if counts is None:
            c = torch.ones(pos.numel(), dtype=torch.float32, device=pos.device)
        else:
            c = counts.detach().float().flatten().to(pos.device)
            if c.numel() != pos.numel():
                raise ValueError("counts must match group size")
            c = c.clamp_min(1.0)

        p, vv, ref_mass, ref_num, ref_cond = self._group_reference(pos)
        select = torch.cat([
            torch.ones(1, dtype=torch.bool, device=accepted.device), accepted
        ])
        sel_idx = select.nonzero(as_tuple=False).flatten()
        if sel_idx.numel() < 1:
            raise RuntimeError("anchor must remain selected")

        csel = c.index_select(0, sel_idx.to(c.device))
        ctot = csel.sum().clamp_min(1.0)
        # q^T(mean K) = count-weighted mean(q^T K).  log p differs from logit by
        # one query-specific constant, which cancels under a convex mean.
        logp_sel = torch.log(p.index_select(1, sel_idx.to(p.device)).clamp_min(self.prob_eps))
        merged_logp = (logp_sel * csel.to(logp_sel.device).unsqueeze(0)).sum(dim=-1) / ctot
        merged_mass = ctot.to(merged_logp.device) * torch.exp(merged_logp)

        vsel = vv.index_select(0, sel_idx.to(vv.device))
        merged_v = (vsel * csel.to(vsel.device).unsqueeze(-1)).sum(dim=0) / ctot.to(vsel.device)
        merged_num = merged_mass.unsqueeze(-1) * merged_v.unsqueeze(0)
        merged_cond = merged_v.unsqueeze(0).expand_as(ref_cond)

        # Stable group-local normalization.  Using ||N_G|| directly can explode when
        # different value directions cancel even though the group itself carries normal
        # attention mass.  sum_i p_i ||v_i|| is a reference-only contribution scale,
        # shared by Eviction/Merge-All/Star-Resolved and independent of sequence length.
        value_norm = torch.linalg.vector_norm(vv, dim=-1)
        num_scale = (p * value_norm.unsqueeze(0)).sum(dim=-1).clamp_min(self.norm_eps)
        cond_norm = torch.linalg.vector_norm(ref_cond, dim=-1).clamp_min(self.norm_eps)
        numerator_rel = torch.linalg.vector_norm(merged_num - ref_num, dim=-1) / num_scale
        conditional_rel = torch.linalg.vector_norm(merged_cond - ref_cond, dim=-1) / cond_norm
        mass_rel = (merged_mass - ref_mass).abs() / ref_mass.clamp_min(self.prob_eps)
        mass_log_abs = (torch.log(merged_mass.clamp_min(self.prob_eps)) - torch.log(ref_mass.clamp_min(self.prob_eps))).abs()

        # Exact single-group replacement in the full attention output.  The original
        # softmax denominator is one because ``a`` is post-softmax.
        new_num = self.full_output - ref_num + merged_num
        new_denom = 1.0 - ref_mass + merged_mass
        new_out = new_num / new_denom.clamp_min(self.prob_eps).unsqueeze(-1)
        global_rel = torch.linalg.vector_norm(new_out - self.full_output, dim=-1) / self.full_norm
        global_rel_over_mass = global_rel / ref_mass.clamp_min(1e-6)

        return GroupErrors(
            numerator_rel=numerator_rel,
            conditional_rel=conditional_rel,
            mass_rel=mass_rel,
            mass_log_abs=mass_log_abs,
            global_rel=global_rel,
            global_rel_over_group_mass=global_rel_over_mass,
            reference_group_mass=ref_mass,
        )


    @torch.inference_mode()
    def route_prefix_numerator_errors(
        self,
        *,
        group_positions: torch.Tensor,
        candidate_order: torch.Tensor,
        counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized numerator error for every scalar-routing prefix.

        ``group_positions`` is ``[anchor, all owner candidates...]`` and
        ``candidate_order`` gives candidate indices (0..n-1) sorted from nearest
        to farthest anchor distance.  Prefix k means keep/merge the anchor plus
        the first k candidates and evict all remaining owner candidates.

        Returns a CPU/working-device tensor of length n+1 containing the mean
        group-numerator relative error for k=0..n.  This avoids O(n) repeated
        calls to ``evaluate_merge`` and makes the exact per-group route oracle
        practical on 4K/8K contexts.
        """
        pos = group_positions.long().flatten()
        n = int(pos.numel()) - 1
        order = candidate_order.long().flatten()
        if n < 0 or order.numel() != n:
            raise ValueError("candidate_order must cover all candidates")
        if n and sorted(order.detach().cpu().tolist()) != list(range(n)):
            raise ValueError("candidate_order must be a permutation of 0..n-1")

        if counts is None:
            c = torch.ones(pos.numel(), dtype=torch.float32, device=pos.device)
        else:
            c = counts.detach().float().flatten().to(pos.device).clamp_min(1.0)
            if c.numel() != pos.numel():
                raise ValueError("counts must match group size")

        p, vv, _, ref_num, _ = self._group_reference(pos)
        # Convert candidate-local [0..n-1] to group-local [1..n], with anchor first.
        group_order = torch.cat([
            torch.zeros(1, dtype=torch.long, device=order.device),
            order + 1,
        ])
        group_order_p = group_order.to(p.device)
        group_order_v = group_order.to(vv.device)
        group_order_c = group_order.to(c.device)

        c_ord = c.index_select(0, group_order_c)
        c_cum = torch.cumsum(c_ord, dim=0).clamp_min(1.0)                 # [K]
        logp = torch.log(p.index_select(1, group_order_p).clamp_min(self.prob_eps))
        weighted_logp = logp * c_ord.to(logp.device).unsqueeze(0)
        mean_logp = torch.cumsum(weighted_logp, dim=1) / c_cum.to(logp.device).unsqueeze(0)
        merged_mass = c_cum.to(logp.device).unsqueeze(0) * torch.exp(mean_logp)  # [N,K]

        vv_ord = vv.index_select(0, group_order_v)
        v_cum = torch.cumsum(vv_ord * c_ord.to(vv.device).unsqueeze(-1), dim=0)
        mean_v = v_cum / c_cum.to(vv.device).unsqueeze(-1)                # [K,D]
        merged_num = merged_mass.unsqueeze(-1) * mean_v.unsqueeze(0)      # [N,K,D]
        value_norm = torch.linalg.vector_norm(vv, dim=-1)
        ref_scale = (p * value_norm.unsqueeze(0)).sum(dim=-1).clamp_min(self.norm_eps)
        err = torch.linalg.vector_norm(merged_num - ref_num.unsqueeze(1), dim=-1) / ref_scale.unsqueeze(1)
        return err.mean(dim=0)

    @torch.inference_mode()
    def evaluate_pair_merge(self, pos_u: int, pos_v: int) -> GroupErrors:
        pos = torch.tensor([int(pos_u), int(pos_v)], dtype=torch.long, device=self.a.device)
        return self.evaluate_merge(
            group_positions=pos,
            accepted_mask=torch.tensor([True], dtype=torch.bool, device=self.a.device),
        )
