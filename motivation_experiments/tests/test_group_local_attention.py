#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from group_local_attention import GroupLocalAttentionEvaluator


def test_identical_keys_merge_preserves_group_contribution():
    # Construct post-softmax probabilities consistent with identical logits for 0,1.
    a = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float32)
    v = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    ev = GroupLocalAttentionEvaluator(a, v)
    e = ev.evaluate_merge(
        group_positions=torch.tensor([0, 1]),
        accepted_mask=torch.tensor([True]),
    )
    assert float(e.numerator_rel.max()) < 1e-6
    assert float(e.mass_rel.max()) < 1e-6
    assert float(e.global_rel.max()) < 1e-6


def test_k_difference_v_same_has_zero_conditional_but_mass_error():
    # Different probabilities imply different logits/K response; identical V means
    # conditional returned information is exact, but mean-K can still distort group mass.
    a = torch.tensor([[0.7, 0.1, 0.2]], dtype=torch.float32)
    v = torch.tensor([[2.0, 1.0], [2.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    ev = GroupLocalAttentionEvaluator(a, v)
    e = ev.evaluate_pair_merge(0, 1)
    assert float(e.conditional_rel.max()) < 1e-6
    assert float(e.mass_rel.max()) > 0.01


def test_merge_all_can_be_worse_than_eviction_group_locally():
    a = torch.tensor([[0.60, 0.05, 0.35]], dtype=torch.float32)
    v = torch.tensor([[1.0, 0.0], [-8.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ev = GroupLocalAttentionEvaluator(a, v)
    pos = torch.tensor([0, 1])
    eo = ev.evaluate_merge(group_positions=pos, accepted_mask=torch.tensor([False]))
    ma = ev.evaluate_merge(group_positions=pos, accepted_mask=torch.tensor([True]))
    assert float(ma.numerator_rel.mean()) > float(eo.numerator_rel.mean())


def test_route_prefix_vectorization_matches_repeated_evaluate_merge():
    torch.manual_seed(3)
    # 4 positions: anchor + 3 owner candidates, two real queries.
    logits = torch.randn(2, 4)
    attn = torch.softmax(logits, dim=-1)
    values = torch.randn(4, 5)
    ev = GroupLocalAttentionEvaluator(attn, values)
    group = torch.tensor([0, 1, 2, 3])
    order = torch.tensor([2, 0, 1])  # candidate-local order
    got = ev.route_prefix_numerator_errors(group_positions=group, candidate_order=order)
    exp = []
    for k in range(4):
        mask = torch.zeros(3, dtype=torch.bool)
        if k:
            mask[order[:k]] = True
        exp.append(ev.evaluate_merge(group_positions=group, accepted_mask=mask).numerator_rel.mean())
    exp = torch.stack(exp)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6)


def test_from_queries_uses_only_frozen_prefix_keys_values():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1 head, 2 probes, 2 dim]
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ev = GroupLocalAttentionEvaluator.from_queries(q, k, v)
    # Only three frozen-prefix positions exist; no probe-token KV is appended.
    assert ev.a.shape == (2, 3)
    assert torch.allclose(ev.a.sum(dim=-1), torch.ones(2), atol=1e-6)
