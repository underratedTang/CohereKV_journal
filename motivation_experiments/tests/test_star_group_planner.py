#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from star_group_planner import (  # noqa: E402
    RouteView,
    enumerate_candidate_groups,
    normalized_k_conflict_score,
    pairwise_candidate_distances,
    parallel_mis_by_priority,
    precompute_routing_scores,
    route_threshold,
    route_view,
)


def test_k_only_conflict_score_is_normalized_euclidean_distance():
    keys = torch.tensor([[0., 0.], [3., 4.], [0., 8.]], dtype=torch.float32)
    dk = pairwise_candidate_distances(keys)
    s = normalized_k_conflict_score(k_pairwise_dist=dk, k_scale=5.0)
    assert torch.allclose(s[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(torch.diag(s), torch.zeros(3))
    assert torch.allclose(s, s.t())


def test_euclidean_routing_assigns_nearest_retained_anchor():
    keys = torch.tensor([[[0., 0.], [10., 0.], [1., 0.], [8., 0.]]])
    keep = torch.tensor([[0, 1]], dtype=torch.long)
    evict = torch.tensor([[2, 3]], dtype=torch.long)
    scores = precompute_routing_scores(keys=keys, keep_idx=keep, evict_idx=evict)
    view = route_view(scores=scores, metric="euclidean", threshold=route_threshold("euclidean", 1.0, 3.0))
    assert view.best_idx.tolist() == [0, 1]
    assert view.compatible.tolist() == [True, True]


def test_priority_prefers_lower_degree_first():
    # Star: center has degree=3; leaves degree=1.
    conflict = torch.tensor([
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,0],
        [1,0,0,0],
    ], dtype=torch.bool)
    selected = parallel_mis_by_priority(
        conflict=conflict,
        anchor_gamma=torch.tensor([.1, .2, .2, .2]),
        physical_positions=torch.tensor([0,1,2,3]),
    )
    assert selected.tolist() == [False, True, True, True]


def test_priority_uses_anchor_distance_after_degree():
    edge = torch.tensor([[0,1],[1,0]], dtype=torch.bool)
    # Equal degree: smaller normalized anchor K distance wins directly.
    selected = parallel_mis_by_priority(
        conflict=edge,
        anchor_gamma=torch.tensor([.8, .1]),
        physical_positions=torch.tensor([10,11]),
    )
    assert selected.tolist() == [False, True]


def test_parallel_mis_is_independent_and_maximal():
    conflict = torch.tensor([
        [0,1,1,0],
        [1,0,0,0],
        [1,0,0,1],
        [0,0,1,0],
    ], dtype=torch.bool)
    selected = parallel_mis_by_priority(
        conflict=conflict,
        anchor_gamma=torch.tensor([.2,.3,.4,.5]),
        physical_positions=torch.arange(4),
    )
    idx = selected.nonzero(as_tuple=False).flatten()
    if idx.numel() > 1:
        assert not bool(conflict.index_select(0, idx).index_select(1, idx).any())
    for r in (~selected).nonzero(as_tuple=False).flatten().tolist():
        assert bool(conflict[r].index_select(0, idx).any())


def test_group_enumeration_uses_only_compatible_routed_sources():
    view = RouteView(
        best_idx=torch.tensor([0,0,2,2,2,4]),
        compatible=torch.tensor([1,1,1,0,1,1], dtype=torch.bool),
        assigned_distance=torch.zeros(6),
        assigned_cosine=torch.ones(6),
    )
    groups = enumerate_candidate_groups(view=view, min_candidates=2)
    assert [(a, idx.tolist()) for a, idx in groups] == [(0,[0,1]), (2,[2,4])]


def test_cosine_routing_assigns_max_cosine_retained_anchor():
    keys = torch.tensor([[[1., 0.], [0., 1.], [2., .1], [.1, 2.]]])
    keep = torch.tensor([[0, 1]], dtype=torch.long)
    evict = torch.tensor([[2, 3]], dtype=torch.long)
    scores = precompute_routing_scores(keys=keys, keep_idx=keep, evict_idx=evict)
    view = route_view(scores=scores, metric="cosine", threshold=route_threshold("cosine", 0.9, 123.0))
    assert view.best_idx.tolist() == [0, 1]
    assert view.compatible.tolist() == [True, True]


def _reference_python_mis(conflict, gamma, positions):
    n = int(conflict.shape[0])
    deg = conflict.sum(dim=1).float()
    mask = (1 << 64) - 1

    def stable_hash(x: int) -> int:
        z = (int(x) + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return (z ^ (z >> 31)) & mask

    def better(i, j):
        fi = (-float(deg[i]), -float(gamma[i]))
        fj = (-float(deg[j]), -float(gamma[j]))
        if fi != fj:
            return fi > fj
        hi, hj = stable_hash(int(positions[i])), stable_hash(int(positions[j]))
        if hi != hj:
            return hi < hj
        return int(positions[i]) < int(positions[j])

    active = [True] * n
    selected = [False] * n
    for _ in range(n):
        ids = [i for i, a in enumerate(active) if a]
        if not ids:
            break
        winners = []
        for i in ids:
            neigh = [j for j in ids if bool(conflict[i, j])]
            if all(not better(j, i) for j in neigh):
                winners.append(i)
        assert winners
        for i in winners:
            selected[i] = True
        reject = set(winners)
        for i in winners:
            reject.update(j for j in ids if bool(conflict[i, j]))
        for j in reject:
            active[j] = False
    return torch.tensor(selected, dtype=torch.bool)


def test_tensorized_mis_matches_reference_python_priority_exactly():
    gen = torch.Generator().manual_seed(7)
    for n in (2, 3, 5, 9, 13):
        a = torch.rand((n, n), generator=gen) < 0.35
        conflict = torch.triu(a, diagonal=1)
        conflict = conflict | conflict.t()
        # Quantized anchor distances intentionally create ties so the exact
        # SplitMix64 hash/position tie-breaker is exercised.
        gamma = torch.randint(0, 3, (n,), generator=gen).float() / 2
        positions = torch.randperm(4 * n, generator=gen)[:n]
        got = parallel_mis_by_priority(
            conflict=conflict, anchor_gamma=gamma, physical_positions=positions,
        ).cpu()
        ref = _reference_python_mis(conflict, gamma, positions)
        assert torch.equal(got, ref)


def test_batched_routing_matches_single_row_routing():
    from star_group_planner import precompute_routing_scores_batched
    gen = torch.Generator().manual_seed(11)
    keys = torch.randn((3, 9, 4), generator=gen)
    keep = torch.tensor([[0,3,7],[1,4,8],[0,2,5]], dtype=torch.long)
    all_idx = torch.arange(9).expand(3, 9)
    mask = torch.ones((3,9), dtype=torch.bool); mask.scatter_(1, keep, False)
    evict = all_idx[mask].view(3,6)
    bat = precompute_routing_scores_batched(keys=keys, keep_idx=keep, evict_idx=evict, chunk_size=2)
    for r in range(3):
        one = precompute_routing_scores(keys=keys[r:r+1], keep_idx=keep[r:r+1], evict_idx=evict[r:r+1], chunk_size=2)
        assert torch.equal(bat.euclidean_best_idx[r], one.euclidean_best_idx[0])
        assert torch.allclose(bat.euclidean_best_distance[r], one.euclidean_best_distance[0], atol=1e-6)
        assert torch.equal(bat.cosine_best_idx[r], one.cosine_best_idx[0])
        assert torch.allclose(bat.cosine_best_cosine[r], one.cosine_best_cosine[0], atol=1e-6)


def test_batched_conflict_selector_matches_per_group_mis():
    from star_group_planner import select_conflict_free_routed_batched
    gen = torch.Generator().manual_seed(19)
    x, n, d, keep_len = 2, 7, 5, 3
    keys = torch.randn((x, n, d), generator=gen)
    best = torch.tensor([[0,0,1,1,1,2,2],[0,0,0,1,2,2,2]], dtype=torch.long)
    compatible = torch.tensor([[1,1,1,1,0,1,1],[1,1,1,1,1,0,1]], dtype=torch.bool)
    gamma = torch.rand((x,n),generator=gen)
    positions = torch.arange(x*n).view(x,n) + 10
    scale = torch.tensor([2.0, 1.5])
    lam = 0.7
    got = select_conflict_free_routed_batched(
        evicted_keys=keys, best_idx=best, route_compatible=compatible,
        anchor_gamma=gamma, physical_positions=positions,
        keep_len=keep_len, k_scale=scale, conflict_lambda=lam,
    ).accepted
    ref = torch.zeros_like(got)
    for r in range(x):
        for a in range(keep_len):
            idx = (compatible[r] & (best[r] == a)).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() == 1:
                ref[r, idx] = True
                continue
            kk = keys[r].index_select(0, idx)
            dist = pairwise_candidate_distances(kk)
            conf = dist / scale[r] > lam
            conf.fill_diagonal_(False)
            sel = parallel_mis_by_priority(
                conflict=conf,
                anchor_gamma=gamma[r].index_select(0, idx),
                physical_positions=positions[r].index_select(0, idx),
            )
            ref[r, idx] = sel
    assert torch.equal(got, ref)
