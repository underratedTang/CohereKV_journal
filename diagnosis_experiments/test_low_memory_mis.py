#!/usr/bin/env python3
"""Compare size-bucketed CPU MIS against the unmodified CohereKV reference.

CPU-only: no pretrained weights, CUDA, transformers, or downloads needed.
"""
import time
import torch
from cosine_mis_plan import make_plan, native_ee_mis


def _compare(keys, keep, evict, best, route, dist, cutoff):
    n = evict.shape[1]
    cosine = torch.ones_like(dist)
    ref = torch.full((keys.shape[0], 1), 1., dtype=torch.float32)
    baseline = make_plan(keys, keep, evict, best, cosine, route, dist, ref, cutoff,
                         graph_backend='torch')
    bounded = make_plan(keys, keep, evict, best, cosine, route, dist, ref, cutoff,
                        graph_backend='cpu_batched')
    for name in ('accepted', 'candidate_pool_sizes', 'selected_per_anchor',
                 'conflict_edges_per_anchor', 'mis_rounds_per_anchor', 'ee_delta'):
        a, b = getattr(baseline, name), getattr(bounded, name)
        assert torch.equal(a, b), (name, (a != b).sum().item())
    assert int(bounded.candidate_pool_sizes.sum()) == int(route.sum())
    assert int(bounded.accepted.sum()) == int(bounded.selected_per_anchor.sum())
    assert not bool((bounded.accepted & ~route).any())
    return bounded


def test_parity_random_multirow():
    for x, seq_len, keep_len, dim in ((1, 96, 10, 16), (1, 250, 45, 32),
                                      (2, 512, 20, 24)):
        torch.manual_seed(100 + seq_len)
        keys = torch.randn((x, seq_len, dim), dtype=torch.float32)
        keep = torch.arange(keep_len).repeat(x, 1)
        evict = torch.arange(keep_len, seq_len).repeat(x, 1)
        n = evict.shape[1]
        best = torch.randint(0, keep_len, (x, n))
        valid = torch.rand((x, n)) > 0.07
        # Include empty groups and size-skewed groups.
        best[:, :min(n, 30)] = 0
        dist = (keys.gather(1, evict[:, :, None].expand(-1, -1, dim))
                - keys.gather(1, best[:, :, None].expand(-1, -1, dim))).norm(dim=-1)
        for cutoff in (0.1, 1., 4., 12.):
            _compare(keys, keep, evict, best, valid, dist, cutoff)


def test_empty_and_singletons():
    torch.manual_seed(3)
    keys = torch.randn(1, 14, 8)
    keep = torch.tensor([[0, 1, 2, 3]])
    evict = torch.arange(4, 14)[None]
    best = torch.tensor([[0, 0, 0, 2, 2, 2, 2, 2, 3, 3]])
    for valid in (torch.zeros((1, 10), dtype=torch.bool),
                  torch.ones((1, 10), dtype=torch.bool)):
        dist = torch.randn((1, 10)).abs()
        _compare(keys, keep, evict, best, valid, dist, 0.5)


def test_8192_uneven_pools():
    """Stress-test large and heavily imbalanced routing; no global pad cube."""
    torch.manual_seed(89)
    n, k, d = 8192, 1638, 32
    keys = torch.randn((1, n, d))
    keep, evict = torch.arange(k)[None], torch.arange(k, n)[None]
    routed_n = evict.shape[1]
    best = (torch.arange(routed_n) % 800)[None]
    best[0, :400] = 0
    routed = torch.ones((1, routed_n), dtype=torch.bool)
    dist = (keys[0, evict[0]] - keys[0, best[0]]).norm(dim=-1)[None]
    started = time.monotonic()
    a, pools, counts, edges, rounds, _ = native_ee_mis(
        keys, keep, evict, best, routed, dist, torch.tensor([[16.]]),
        0.5, graph_backend='cpu_batched')
    assert int(pools.sum()) == routed_n
    assert int(counts.sum()) == int(a.sum())
    assert int(counts.max()) >= 1
    assert int(edges.sum()) >= 0
    assert int(rounds.max()) >= 1
    print('[STRESS] prefix=8192, keep=1638, routed=6554, uneven groups; '
          f'elapsed={time.monotonic()-started:.2f}s')


if __name__ == '__main__':
    test_parity_random_multirow()
    test_empty_and_singletons()
    test_8192_uneven_pools()
    print('[PASS] cpu_batched MIS matches original torch MIS: membership, '
          'pool counts, edge counts and rounds; 8192-token stress test')
