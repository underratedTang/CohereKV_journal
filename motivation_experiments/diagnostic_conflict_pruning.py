#!/usr/bin/env python3
from __future__ import annotations

"""Star-conflict diagnostics used only by Motivation 1/2.

The v9 primary intervention is a *counterfactual potential* test, not the
Modified CohereKV MIS.  K-space alone identifies the current most-severe
conflict pair.  The two legal filtering outcomes (evict either endpoint) are
then evaluated and the lower-error branch is followed before recomputing the
remaining conflicts.  Legacy geometry-only helpers are retained for unit tests
and auxiliary diagnostics but are not used by the v9 primary Figure 1/2 path.
"""

from typing import Callable, List, Optional, Tuple

import torch


def _active_graph(conflict: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    g = conflict.bool() & keep[:, None] & keep[None, :]
    if g.numel():
        g = g.clone()
        g.fill_diagonal_(False)
        g = g | g.t()
    return g


def conflict_stats(conflict: torch.Tensor, keep: torch.Tensor | None = None) -> Tuple[int, int]:
    """Return (#undirected edges, #vertices incident to >=1 active edge)."""
    if conflict.ndim != 2 or conflict.shape[0] != conflict.shape[1]:
        raise ValueError("conflict must be square")
    n = int(conflict.shape[0])
    if keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=conflict.device)
    keep = keep.bool().flatten().to(conflict.device)
    if keep.numel() != n:
        raise ValueError("keep size mismatch")
    g = _active_graph(conflict, keep)
    edges = int(torch.triu(g, diagonal=1).sum().item())
    vertices = int((g.sum(dim=1) > 0).sum().item()) if n else 0
    return edges, vertices


def conflict_mask_from_score(score: torch.Tensor, threshold: float) -> torch.Tensor:
    """Build a symmetric, diagonal-free conflict mask from a square score matrix."""
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score must be square")
    out = score.float() > float(threshold)
    if out.numel():
        out = out.clone()
        out.fill_diagonal_(False)
        out = out | out.t()
    return out


def most_severe_conflict_edge(
    score: torch.Tensor,
    threshold: float,
    *,
    keep: torch.Tensor | None = None,
) -> Optional[Tuple[int, int]]:
    """Return the active conflict edge with largest K-distance score.

    Retained for diagnostics/tests.  The v8 Conflict-Reduced intervention does
    not use this edge-first rule; it uses conflict degree first.
    """
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score must be square")
    n = int(score.shape[0])
    if keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=score.device)
    keep = keep.bool().flatten().to(score.device)
    if keep.numel() != n:
        raise ValueError("keep size mismatch")

    conf = conflict_mask_from_score(score, threshold)
    active = _active_graph(conf, keep)
    upper = torch.triu(active, diagonal=1)
    idx = upper.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return None
    vals = score.float()[idx[:, 0], idx[:, 1]]
    vmax = vals.max()
    tied = idx[torch.isclose(vals, vmax, rtol=0.0, atol=1e-12)]
    u, v = tied[0].tolist()
    return int(u), int(v)


def endpoint_drop_masks(candidate_count: int, edge: Tuple[int, int], *, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(candidate_count)
    u, v = int(edge[0]), int(edge[1])
    if not (0 <= u < n and 0 <= v < n and u != v):
        raise ValueError("bad edge")
    ku = torch.ones(n, dtype=torch.bool, device=device)
    kv = torch.ones(n, dtype=torch.bool, device=device)
    ku[u] = False
    kv[v] = False
    return ku, kv


def counterfactual_conflict_reduction_mask(
    *,
    score: torch.Tensor,
    threshold: float,
    max_removals: int,
    evaluate_keep: Callable[[torch.Tensor], float],
    initial_keep: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[int], List[dict]]:
    """Resolve up to K severe conflicts by a two-branch counterfactual.

    This helper is *motivation-only*.  The Star conflict itself is identified
    only from ``score`` and ``threshold``.  For the current most-severe active
    conflict edge ``(u, v)``, both legal filtering outcomes are evaluated:

      * evict ``u`` and keep ``v`` eligible for Merge-All;
      * evict ``v`` and keep ``u`` eligible for Merge-All.

    The lower-error branch becomes the state for the next step.  The procedure
    is repeated for at most ``max_removals`` sources or until no conflict edge
    remains.  It is a diagnostic counterfactual measuring whether a harmful
    conflict can be resolved by filtering; it is not the Modified CohereKV
    selection algorithm.
    """
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score must be square")
    n = int(score.shape[0])
    if initial_keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=score.device)
    else:
        keep = initial_keep.detach().bool().flatten().to(score.device).clone()
        if keep.numel() != n:
            raise ValueError("initial_keep size mismatch")

    removed: List[int] = []
    trace: List[dict] = []
    for step in range(max(0, int(max_removals))):
        edge = most_severe_conflict_edge(score, float(threshold), keep=keep)
        if edge is None:
            break
        u, v = int(edge[0]), int(edge[1])
        keep_u = keep.clone(); keep_u[u] = False
        keep_v = keep.clone(); keep_v[v] = False
        err_u = float(evaluate_keep(keep_u))
        err_v = float(evaluate_keep(keep_v))
        if err_u <= err_v:
            chosen, chosen_err, keep = u, err_u, keep_u
        else:
            chosen, chosen_err, keep = v, err_v, keep_v
        removed.append(int(chosen))
        trace.append({
            "step": int(step + 1),
            "u": u,
            "v": v,
            "edge_score": float(score[u, v].item()),
            "error_drop_u": err_u,
            "error_drop_v": err_v,
            "chosen": int(chosen),
            "chosen_error": float(chosen_err),
        })
    return keep, removed, trace


def select_high_conflict_source(
    *,
    score: torch.Tensor,
    threshold: float,
    route_badness: torch.Tensor,
    keep: torch.Tensor | None = None,
    physical_positions: torch.Tensor | None = None,
) -> Optional[Tuple[int, int, float, int]]:
    """Select one active conflict source by a method-free geometry rule.

    Priority is exactly:
        higher active conflict degree
        -> worse anchor compatibility (larger ``route_badness``)
        -> later physical position
        -> larger local index.

    Returns ``(local_index, degree, route_badness, position)`` or ``None`` if no
    active conflict edge remains.  No attention output, importance, or MIS state
    is used.
    """
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score must be square")
    n = int(score.shape[0])
    rb = route_badness.detach().float().flatten().to(score.device)
    if rb.numel() != n:
        raise ValueError("route_badness size mismatch")
    if keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=score.device)
    else:
        keep = keep.detach().bool().flatten().to(score.device)
        if keep.numel() != n:
            raise ValueError("keep size mismatch")
    pos = None
    if physical_positions is not None:
        pos = physical_positions.detach().long().flatten().to(score.device)
        if pos.numel() != n:
            raise ValueError("physical_positions size mismatch")

    conf = conflict_mask_from_score(score, threshold)
    active = _active_graph(conf, keep)
    if active.numel() == 0:
        return None
    degree = active.sum(dim=1).long()
    max_degree = int(degree.max().item())
    if max_degree <= 0:
        return None

    candidate_idx = (degree == max_degree).nonzero(as_tuple=False).flatten()
    # K is tiny in the diagnostic intervention.  Move only the tied candidates
    # to Python once; there is no CUDA synchronization inside the production MIS.
    best = None
    for idx_t in candidate_idx:
        i = int(idx_t.item())
        b = float(rb[i].item())
        p = int(pos[i].item()) if pos is not None else i
        key = (max_degree, b, p, i)
        if best is None or key > best[0]:
            best = (key, i, b, p)
    assert best is not None
    _, i, b, p = best
    return int(i), int(max_degree), float(b), int(p)


def geometry_conflict_reduction_mask(
    *,
    score: torch.Tensor,
    threshold: float,
    route_badness: torch.Tensor,
    max_removals: int,
    physical_positions: torch.Tensor | None = None,
    initial_keep: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Remove up to ``max_removals`` high-conflict sources using geometry only."""
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score must be square")
    n = int(score.shape[0])
    rb = route_badness.detach().float().flatten().to(score.device)
    if rb.numel() != n:
        raise ValueError("route_badness size mismatch")
    pos = None
    if physical_positions is not None:
        pos = physical_positions.detach().long().flatten().to(score.device)
        if pos.numel() != n:
            raise ValueError("physical_positions size mismatch")
    if initial_keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=score.device)
    else:
        keep = initial_keep.detach().bool().flatten().to(score.device).clone()
        if keep.numel() != n:
            raise ValueError("initial_keep size mismatch")

    removed: List[int] = []
    for _ in range(max(0, int(max_removals))):
        selected = select_high_conflict_source(
            score=score,
            threshold=float(threshold),
            route_badness=rb,
            keep=keep,
            physical_positions=pos,
        )
        if selected is None:
            break
        endpoint = int(selected[0])
        keep[endpoint] = False
        removed.append(endpoint)
    return keep, removed
