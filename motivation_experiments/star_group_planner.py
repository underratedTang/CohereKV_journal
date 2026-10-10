#!/usr/bin/env python3
from __future__ import annotations

"""Routing and current-round K-only Star-conflict helpers.

The preliminary experiments keep routing and conflict deliberately separate:

* routing assigns each evicted source to one retained anchor;
* conflict is current-round source-source normalized Euclidean K distance;
* Modified CohereKV keeps a deterministic parallel-MIS priority
  ``(-degree, -anchor_dK, -stable_hash, -position)``.

The performance-critical path is fully tensorized.  Candidate pools from all
anchors in a row batch are packed into one padded tensor, pairwise K distances
are computed with batched GEMM, and every MIS round executes over all pools in
parallel.  There are no per-node/per-neighbor Python loops and no ``.item()`` /
``float(cuda_tensor)`` calls inside MIS rounds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EVAL_ROOT = PROJECT_ROOT / "task_evaluation"
if str(TASK_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_EVAL_ROOT))

from models.common.mymethod_common import select_keep_indices  # noqa: E402

_EPS = 1e-12
_I64_MIN = -(1 << 63)


@dataclass
class RoutingScores:
    keep_idx: torch.Tensor
    evict_idx: torch.Tensor
    euclidean_best_idx: torch.Tensor
    euclidean_best_distance: torch.Tensor
    euclidean_best_cosine: torch.Tensor
    cosine_best_idx: torch.Tensor
    cosine_best_distance: torch.Tensor
    cosine_best_cosine: torch.Tensor


@dataclass
class RouteView:
    best_idx: torch.Tensor
    compatible: torch.Tensor
    assigned_distance: torch.Tensor
    assigned_cosine: torch.Tensor


@dataclass
class MISBatchResult:
    accepted: torch.Tensor
    conflict_edges_per_anchor: torch.Tensor
    candidate_pool_sizes: torch.Tensor
    selected_counts: torch.Tensor
    mis_rounds_per_anchor: torch.Tensor


def _complement_indices(keep_idx: torch.Tensor, length: int) -> torch.Tensor:
    mask = torch.ones((keep_idx.shape[0], length), dtype=torch.bool, device=keep_idx.device)
    mask.scatter_(1, keep_idx, False)
    return mask.nonzero(as_tuple=False)[:, 1].view(keep_idx.shape[0], length - keep_idx.shape[1])


def _gather_seq(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if x.dim() == 2:
        return torch.gather(x, 1, idx)
    raise ValueError(tuple(x.shape))


def fixed_partition(*, importance: torch.Tensor, positions: torch.Tensor, keep_len: int,
                    sink_size: int, recent_window: int):
    keep_idx = select_keep_indices(
        importance.float(), positions, int(keep_len),
        sink_size=int(sink_size), recent_window=int(recent_window),
    )
    evict_idx = _complement_indices(keep_idx, int(importance.shape[1]))
    return keep_idx, evict_idx


@torch.inference_mode()
def precompute_routing_scores_batched(*, keys: torch.Tensor, keep_idx: torch.Tensor,
                                      evict_idx: torch.Tensor,
                                      chunk_size: int = 1024) -> RoutingScores:
    """Batched Euclidean/cosine routing scores for ``keys=[X,L,D]``."""
    if keys.dim() != 3:
        raise ValueError("keys must be [X,L,D]")
    if keep_idx.dim() != 2 or evict_idx.dim() != 2:
        raise ValueError("keep_idx/evict_idx must be [X,N]")
    if int(keys.shape[0]) != int(keep_idx.shape[0]) or int(keys.shape[0]) != int(evict_idx.shape[0]):
        raise ValueError("row dimensions must match")

    retained = _gather_seq(keys, keep_idx).float()
    evicted = _gather_seq(keys, evict_idx).float()
    x = int(keys.shape[0])
    n = int(evicted.shape[1])
    if n == 0:
        zf = torch.empty((x, 0), dtype=torch.float32, device=keys.device)
        zl = torch.empty((x, 0), dtype=torch.long, device=keys.device)
        return RoutingScores(keep_idx, evict_idx, zl, zf, zf, zl, zf, zf)

    chunk = n if int(chunk_size) <= 0 else min(max(1, int(chunk_size)), n)
    r_norm2 = (retained * retained).sum(dim=-1).unsqueeze(1)  # [X,1,K]
    r_norm = torch.sqrt(r_norm2.clamp_min(_EPS))
    parts = {k: [] for k in ("ei", "ed", "ec", "ci", "cd", "cc")}
    rt = retained.transpose(1, 2)
    for start in range(0, n, chunk):
        e = evicted[:, start:start + chunk]
        dot = torch.bmm(e, rt)
        e_norm2 = (e * e).sum(dim=-1, keepdim=True)
        dist2 = (e_norm2 + r_norm2 - 2.0 * dot).clamp_min_(0.0)
        e_norm = torch.sqrt(e_norm2.clamp_min(_EPS))
        cos = (dot / (e_norm * r_norm).clamp_min(_EPS)).clamp_(-1.0, 1.0)

        best_d2, idx_e = dist2.min(dim=2)
        dist_e = torch.sqrt(best_d2)
        cos_e = cos.gather(2, idx_e.unsqueeze(2)).squeeze(2)
        cos_c, idx_c = cos.max(dim=2)
        dist_c = torch.sqrt(dist2.gather(2, idx_c.unsqueeze(2)).squeeze(2))

        parts["ei"].append(idx_e); parts["ed"].append(dist_e); parts["ec"].append(cos_e)
        parts["ci"].append(idx_c); parts["cd"].append(dist_c); parts["cc"].append(cos_c)

    return RoutingScores(
        keep_idx=keep_idx,
        evict_idx=evict_idx,
        euclidean_best_idx=torch.cat(parts["ei"], dim=1),
        euclidean_best_distance=torch.cat(parts["ed"], dim=1),
        euclidean_best_cosine=torch.cat(parts["ec"], dim=1),
        cosine_best_idx=torch.cat(parts["ci"], dim=1),
        cosine_best_distance=torch.cat(parts["cd"], dim=1),
        cosine_best_cosine=torch.cat(parts["cc"], dim=1),
    )


@torch.inference_mode()
def precompute_routing_scores(*, keys: torch.Tensor, keep_idx: torch.Tensor,
                              evict_idx: torch.Tensor, chunk_size: int = 1024) -> RoutingScores:
    if keys.dim() != 3 or int(keys.shape[0]) != 1:
        raise ValueError("keys must be [1,L,D]; use precompute_routing_scores_batched for X>1")
    return precompute_routing_scores_batched(
        keys=keys, keep_idx=keep_idx, evict_idx=evict_idx, chunk_size=chunk_size
    )


def route_view_batched(*, scores: RoutingScores, metric: str, threshold) -> RouteView:
    metric = str(metric).lower()
    if torch.is_tensor(threshold):
        thr = threshold.to(scores.keep_idx.device, dtype=torch.float32)
        if thr.ndim == 1:
            thr = thr.unsqueeze(1)
    else:
        thr = float(threshold)
    if metric == "euclidean":
        return RouteView(
            best_idx=scores.euclidean_best_idx,
            compatible=scores.euclidean_best_distance <= thr,
            assigned_distance=scores.euclidean_best_distance,
            assigned_cosine=scores.euclidean_best_cosine,
        )
    if metric == "cosine":
        return RouteView(
            best_idx=scores.cosine_best_idx,
            compatible=scores.cosine_best_cosine >= thr,
            assigned_distance=scores.cosine_best_distance,
            assigned_cosine=scores.cosine_best_cosine,
        )
    raise ValueError(metric)


def route_view(*, scores: RoutingScores, metric: str, threshold: float) -> RouteView:
    v = route_view_batched(scores=scores, metric=metric, threshold=threshold)
    if int(v.best_idx.shape[0]) != 1:
        raise ValueError("route_view expects one row; use route_view_batched for X>1")
    return RouteView(
        best_idx=v.best_idx[0], compatible=v.compatible[0],
        assigned_distance=v.assigned_distance[0], assigned_cosine=v.assigned_cosine[0],
    )


def route_threshold(metric: str, param: float, reference_scale: float) -> float:
    metric = str(metric).lower()
    if metric == "euclidean":
        return float(param) * float(reference_scale)
    if metric == "cosine":
        return float(param)
    raise ValueError(metric)


@torch.inference_mode()
def enumerate_candidate_groups(*, view: RouteView, min_candidates: int = 2) -> List[Tuple[int, torch.Tensor]]:
    idx = view.compatible.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    anchors = view.best_idx.index_select(0, idx)
    order = torch.argsort(anchors)
    idx = idx.index_select(0, order)
    anchors = anchors.index_select(0, order)
    uniq, counts = torch.unique_consecutive(anchors, return_counts=True)
    groups: List[Tuple[int, torch.Tensor]] = []
    start = 0
    need = max(1, int(min_candidates))
    # This helper is only used by diagnostic motivation/plotting code; performance-critical
    # Modified CohereKV uses select_conflict_free_routed_batched below.
    for anchor_t, count_t in zip(uniq.detach().cpu().tolist(), counts.detach().cpu().tolist()):
        count = int(count_t)
        if count >= need:
            groups.append((int(anchor_t), idx[start:start + count]))
        start += count
    return groups


@torch.inference_mode()
def pairwise_candidate_distances(candidate_keys: torch.Tensor) -> torch.Tensor:
    if candidate_keys.ndim != 2:
        raise ValueError("candidate_keys must be [N,D]")
    if int(candidate_keys.shape[0]) == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=candidate_keys.device)
    x = candidate_keys.float()
    norm2 = (x * x).sum(dim=-1, keepdim=True)
    d2 = (norm2 + norm2.t() - 2.0 * torch.mm(x, x.t())).clamp_min_(0.0)
    return torch.sqrt(d2)


@torch.inference_mode()
def pairwise_candidate_distances_batched(candidate_keys: torch.Tensor) -> torch.Tensor:
    if candidate_keys.ndim != 3:
        raise ValueError("candidate_keys must be [G,N,D]")
    if int(candidate_keys.shape[1]) == 0:
        g = int(candidate_keys.shape[0])
        return torch.empty((g, 0, 0), dtype=torch.float32, device=candidate_keys.device)
    x = candidate_keys.float()
    norm2 = (x * x).sum(dim=-1, keepdim=True)
    d2 = (norm2 + norm2.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))).clamp_min_(0.0)
    return torch.sqrt(d2)


def normalized_k_conflict_score(*, k_pairwise_dist: torch.Tensor, k_scale: float) -> torch.Tensor:
    """Current-round, query-independent source-source K disagreement."""
    score = k_pairwise_dist.detach().float() / max(float(k_scale), _EPS)
    if score.ndim == 2 and score.numel():
        score = score.clone(); score.fill_diagonal_(0.0)
    return score


@torch.inference_mode()
def route_stats(view: RouteView, num_retained: int, num_evicted: int):
    routed = view.compatible.nonzero(as_tuple=False).flatten()
    routed_n = int(routed.numel())
    if routed_n:
        owners = view.best_idx.index_select(0, routed)
        counts = torch.bincount(owners, minlength=int(num_retained))
        nz = counts[counts > 0]
        group_count = int(nz.numel())
        multi_count = int((nz >= 2).sum().item())
        mean_group = float(nz.float().mean().item()) if nz.numel() else 0.0
        max_group = int(nz.max().item()) if nz.numel() else 0
    else:
        group_count = multi_count = max_group = 0
        mean_group = 0.0
    return {
        "candidate_admission": float(routed_n / max(int(num_evicted), 1)),
        "routed_candidate_count": float(routed_n),
        "total_evicted_count": float(num_evicted),
        "candidate_group_count": float(group_count),
        "multi_candidate_group_count": float(multi_count),
        "mean_candidate_group_size": float(mean_group),
        "max_candidate_group_size": float(max_group),
    }


def _logical_rshift_i64(x: torch.Tensor, bits: int) -> torch.Tensor:
    # Arithmetic right shift + low-bit mask emulates unsigned logical shift.
    return (x >> int(bits)) & ((1 << (64 - int(bits))) - 1)


def _signed_u64_constant(value: int) -> int:
    value &= (1 << 64) - 1
    return value if value < (1 << 63) else value - (1 << 64)


def _stable_hash_order_key(positions: torch.Tensor) -> torch.Tensor:
    """Exact vectorized SplitMix64 used by the original Python implementation.

    Returned signed int64 keys preserve *unsigned* uint64 ordering by flipping the
    sign bit.  Thus ordinary signed ``<`` is equivalent to uint64 ``<``.
    """
    z = positions.to(torch.int64)
    z = z + _signed_u64_constant(0x9E3779B97F4A7C15)
    z = (z ^ _logical_rshift_i64(z, 30)) * _signed_u64_constant(0xBF58476D1CE4E5B9)
    z = (z ^ _logical_rshift_i64(z, 27)) * _signed_u64_constant(0x94D049BB133111EB)
    h = z ^ _logical_rshift_i64(z, 31)
    return h ^ _I64_MIN


@torch.inference_mode()
def _parallel_mis_packed(*, conflict: torch.Tensor, valid: torch.Tensor,
                         anchor_gamma: torch.Tensor, physical_positions: torch.Tensor):
    """Parallel MIS over padded pools, all groups at once.

    Inputs are ``[G,M,M]`` / ``[G,M]``.  Priority is deterministic and contains
    no attention-importance term: degree -> anchor distance -> stable hash -> position.
    """
    if conflict.ndim != 3 or valid.ndim != 2:
        raise ValueError("packed conflict/valid rank mismatch")
    g, m, m2 = conflict.shape
    if m != m2 or tuple(valid.shape) != (g, m):
        raise ValueError("packed shapes mismatch")
    if m == 0:
        return torch.zeros_like(valid), torch.zeros_like(valid, dtype=torch.int32)

    degree = conflict.sum(dim=-1).to(torch.int32)
    gamma = anchor_gamma.float()
    pos = physical_positions.long()
    hash_key = _stable_hash_order_key(pos)

    di, dj = degree[:, :, None], degree[:, None, :]
    gi, gj = gamma[:, :, None], gamma[:, None, :]
    hi, hj = hash_key[:, :, None], hash_key[:, None, :]
    pi, pj = pos[:, :, None], pos[:, None, :]

    degree_eq = dj == di
    gamma_eq = gj == gi
    hash_eq = hj == hi
    j_better = (
        (dj < di)
        | (degree_eq & (gj < gi))
        | (degree_eq & gamma_eq & (hj < hi))
        | (degree_eq & gamma_eq & hash_eq & (pj < pi))
    ) & valid[:, :, None] & valid[:, None, :]

    active = valid.clone()
    selected = torch.zeros_like(active)
    selected_round = torch.zeros((g, m), dtype=torch.int32, device=conflict.device)
    # Fixed M rounds deliberately avoid a GPU->CPU convergence test every round.
    for round_no in range(1, m + 1):
        blocked = (conflict & active[:, None, :] & j_better).any(dim=-1)
        winners = active & ~blocked
        selected |= winners
        selected_round = torch.where(
            winners, torch.full_like(selected_round, round_no), selected_round
        )
        remove = winners | (conflict & winners[:, None, :]).any(dim=-1)
        active &= ~remove
    return selected, selected_round


@torch.inference_mode()
def parallel_mis_by_priority(*, conflict: torch.Tensor, anchor_gamma: torch.Tensor,
                             physical_positions: torch.Tensor) -> torch.Tensor:
    """Tensorized deterministic MIS with degree -> anchor distance -> hash -> position."""
    if conflict.ndim != 2 or conflict.shape[0] != conflict.shape[1]:
        raise ValueError("conflict must be square")
    n = int(conflict.shape[0])
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=conflict.device)
    gamma = anchor_gamma.detach().float().flatten().to(conflict.device)
    pos = physical_positions.detach().long().flatten().to(conflict.device)
    if gamma.numel() != n or pos.numel() != n:
        raise ValueError("priority inputs must match conflict size")
    conf = conflict.bool().unsqueeze(0)
    valid = torch.ones((1, n), dtype=torch.bool, device=conflict.device)
    selected, _ = _parallel_mis_packed(
        conflict=conf, valid=valid,
        anchor_gamma=gamma.unsqueeze(0), physical_positions=pos.unsqueeze(0),
    )
    return selected[0]


def _scatter_group_stat(group_ids: torch.Tensor, values: torch.Tensor,
                        total_groups: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(total_groups, dtype=dtype, device=group_ids.device)
    if group_ids.numel():
        out.scatter_add_(0, group_ids, values.to(dtype))
    return out


@torch.inference_mode()
def select_conflict_free_routed_batched(*, evicted_keys: torch.Tensor,
                                        best_idx: torch.Tensor,
                                        route_compatible: torch.Tensor,
                                        anchor_gamma: torch.Tensor,
                                        physical_positions: torch.Tensor,
                                        keep_len: int,
                                        k_scale: torch.Tensor,
                                        conflict_lambda: float) -> MISBatchResult:
    """Build K-only conflicts + parallel MIS for all rows/anchors in one batch.

    ``evicted_keys`` is ``[X,N,D]``.  Pools are bucketed by ``(row, anchor)`` and
    padded only to the largest pool.  This is the performance-critical Modified
    CohereKV selector used by both single-round and PPL experiments.
    """
    if evicted_keys.ndim != 3:
        raise ValueError("evicted_keys must be [X,N,D]")
    x, n, dim = evicted_keys.shape
    if tuple(best_idx.shape) != (x, n) or tuple(route_compatible.shape) != (x, n):
        raise ValueError("routing shapes mismatch")
    if tuple(anchor_gamma.shape) != (x, n):
        raise ValueError("priority shapes mismatch")
    if tuple(physical_positions.shape) != (x, n):
        raise ValueError("physical_positions shape mismatch")
    keep_len = int(keep_len)
    total_groups = int(x * keep_len)
    device = evicted_keys.device
    accepted = torch.zeros((x, n), dtype=torch.bool, device=device)
    zero_i = torch.zeros((x, keep_len), dtype=torch.int32, device=device)
    if n == 0:
        return MISBatchResult(accepted, zero_i, zero_i.clone(), zero_i.clone(), zero_i.clone())

    row_ids = torch.arange(x, device=device, dtype=torch.long).unsqueeze(1).expand(x, n)
    group_all = row_ids * keep_len + best_idx.long()
    valid_flat = route_compatible.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid_flat.numel() == 0:
        return MISBatchResult(accepted, zero_i, zero_i.clone(), zero_i.clone(), zero_i.clone())

    flat_group = group_all.reshape(-1).index_select(0, valid_flat)
    flat_local = (
        torch.arange(n, device=device, dtype=torch.long).view(1, n).expand(x, n)
        .reshape(-1).index_select(0, valid_flat)
    )
    order = torch.argsort(flat_group * (n + 1) + flat_local)
    sorted_group = flat_group.index_select(0, order)
    sorted_flat = valid_flat.index_select(0, order)
    unique_group, counts = torch.unique_consecutive(sorted_group, return_counts=True)
    g = int(unique_group.numel())
    max_pool = int(counts.max().item())  # one allocation sync per batched selector call
    starts = torch.cumsum(counts, dim=0) - counts
    group_slot = torch.repeat_interleave(torch.arange(g, device=device), counts)
    rank = torch.arange(sorted_flat.numel(), device=device) - torch.repeat_interleave(starts, counts)
    packed_flat = torch.full((g, max_pool), -1, device=device, dtype=torch.long)
    packed_flat[group_slot, rank] = sorted_flat
    valid = packed_flat >= 0
    safe_flat = packed_flat.clamp_min(0)

    flat_keys = evicted_keys.reshape(x * n, dim)
    flat_gamma = anchor_gamma.reshape(-1)
    flat_pos = physical_positions.reshape(-1)
    gather = safe_flat.reshape(-1)
    packed_keys = flat_keys.index_select(0, gather).reshape(g, max_pool, dim)
    packed_gamma = flat_gamma.index_select(0, gather).reshape(g, max_pool)
    packed_pos = flat_pos.index_select(0, gather).reshape(g, max_pool)

    d = pairwise_candidate_distances_batched(packed_keys)
    group_rows = torch.div(unique_group, keep_len, rounding_mode="floor")
    scale = k_scale.float().flatten().to(device)
    if scale.numel() == 1 and x > 1:
        scale = scale.expand(x)
    if scale.numel() != x:
        raise ValueError("k_scale must be scalar or [X]")
    group_scale = scale.index_select(0, group_rows).clamp_min(_EPS)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    conflict = pair_valid & (d / group_scale[:, None, None] > float(conflict_lambda))
    eye = torch.eye(max_pool, dtype=torch.bool, device=device).unsqueeze(0)
    conflict &= ~eye
    conflict = conflict | conflict.transpose(1, 2)

    selected, selected_round = _parallel_mis_packed(
        conflict=conflict, valid=valid,
        anchor_gamma=packed_gamma, physical_positions=packed_pos,
    )
    sel_flat = packed_flat[selected]
    sel_flat = sel_flat[sel_flat >= 0]
    if sel_flat.numel():
        accepted.reshape(-1).index_fill_(0, sel_flat, True)

    pool_flat = _scatter_group_stat(unique_group, counts.to(torch.int32), total_groups, torch.int32)
    sel_counts = selected.sum(dim=1).to(torch.int32)
    selected_flat = _scatter_group_stat(unique_group, sel_counts, total_groups, torch.int32)
    edge_counts = torch.triu(conflict, diagonal=1).sum(dim=(1, 2)).to(torch.int32)
    edge_flat = _scatter_group_stat(unique_group, edge_counts, total_groups, torch.int32)
    rounds = selected_round.amax(dim=1)
    round_flat = _scatter_group_stat(unique_group, rounds, total_groups, torch.int32)

    return MISBatchResult(
        accepted=accepted,
        conflict_edges_per_anchor=edge_flat.view(x, keep_len),
        candidate_pool_sizes=pool_flat.view(x, keep_len),
        selected_counts=selected_flat.view(x, keep_len),
        mis_rounds_per_anchor=round_flat.view(x, keep_len),
    )
