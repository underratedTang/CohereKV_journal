#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from whole_cache_attention import WholePrefixAttentionEvaluator


def test_exact_merge_for_identical_keys_with_multiplicity():
    # tokens 0 and 1 have identical K; merging 1 into retained 0 with count=2
    # must preserve their combined attention contribution exactly.
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # [2,1,2]
    k = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 0.0], [0.5, 4.0]])
    ev = WholePrefixAttentionEvaluator(q, k, v)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    evict = torch.tensor([[1]], dtype=torch.long)
    best = torch.tensor([0], dtype=torch.long)
    accepted = torch.tensor([True])
    counts = torch.ones(3, dtype=torch.int32)
    err = ev.compressed_error(keep_idx=keep, evict_idx=evict, best_idx=best,
                              accepted_mask=accepted, counts_1d=counts)
    assert float(err.max()) < 1e-6, err


def test_eviction_is_not_forced_equal_to_merge():
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0]])
    ev = WholePrefixAttentionEvaluator(q, k, v)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    counts = torch.ones(3, dtype=torch.int32)
    e = ev.eviction_error(keep_idx=keep, counts_1d=counts)
    assert float(e.item()) > 0.0


if __name__ == "__main__":
    test_exact_merge_for_identical_keys_with_multiplicity()
    test_eviction_is_not_forced_equal_to_merge()
    print("ok")


def test_probability_evaluator_matches_direct_query_evaluator_without_tail():
    from whole_cache_attention import WholePrefixAttentionFromProbsEvaluator
    q = torch.tensor([[[1.0, 0.2]], [[-0.3, 0.8]]])
    k = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [-0.2, 0.7]])
    v = torch.tensor([[1.0, 0.0], [2.0, 0.5], [0.0, 1.0], [0.3, 1.5]])
    direct = WholePrefixAttentionEvaluator(q, k, v)
    qflat = q.reshape(-1, 2).float()
    probs = torch.softmax(qflat @ k.float().t() / (2.0 ** 0.5), dim=-1)
    from_probs = WholePrefixAttentionFromProbsEvaluator(probs, v, prefix_len=4)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    evict = torch.tensor([[1, 3]], dtype=torch.long)
    best = torch.tensor([0, 1], dtype=torch.long)
    accepted = torch.tensor([True, False])
    counts = torch.ones(4, dtype=torch.int32)
    e1 = direct.compressed_error(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=accepted, counts_1d=counts,
    )
    e2 = from_probs.compressed_error(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=accepted, counts_1d=counts,
    )
    assert torch.allclose(e1, e2, atol=1e-6, rtol=1e-6), (e1, e2)


def test_probability_evaluator_keeps_probe_tail_uncompressed():
    from whole_cache_attention import WholePrefixAttentionFromProbsEvaluator
    # Prefix positions 0..2; position 3 is an untouched probe-tail state.
    probs = torch.tensor([[0.25, 0.25, 0.20, 0.30]], dtype=torch.float32)
    v = torch.tensor([[1., 0.], [3., 0.], [0., 1.], [0., 4.]])
    ev = WholePrefixAttentionFromProbsEvaluator(probs, v, prefix_len=3)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    evict = torch.tensor([[1]], dtype=torch.long)
    best = torch.tensor([0], dtype=torch.long)
    counts = torch.ones(3, dtype=torch.int32)
    # p0 == p1, so merging identical-logit states with multiplicity is exact for
    # their score; the untouched tail must remain present in normalization/output.
    err = ev.compressed_error(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=torch.tensor([True]), counts_1d=counts,
    )
    # Values differ, but count-weighted value merge is exact here because p0==p1.
    assert float(err.max()) < 1e-6, err


def test_mapping_evaluator_matches_direct_single_round_merge():
    q = torch.tensor([[[1.0, 0.2]], [[-0.3, 0.8]]])
    k = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [-0.2, 0.7]])
    v = torch.tensor([[1.0, 0.0], [2.0, 0.5], [0.0, 1.0], [0.3, 1.5]])
    ev = WholePrefixAttentionEvaluator(q, k, v)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    evict = torch.tensor([[1, 3]], dtype=torch.long)
    best = torch.tensor([0, 1], dtype=torch.long)
    accepted = torch.tensor([True, False])
    counts = torch.ones(4, dtype=torch.int32)
    direct = ev.compressed_error(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=accepted, counts_1d=counts,
    )
    # state 0 represents original tokens {0,1}; state 1 represents {2}; token 3 evicted.
    mapped = ev.compressed_error_from_mapping([0, 0, 1, -1])
    assert torch.allclose(direct, mapped, atol=1e-6, rtol=1e-6), (direct, mapped)


def test_compressed_output_matches_compressed_error_definition():
    q = torch.tensor([[[1.0, 0.2]], [[-0.3, 0.8]]])
    k = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [-0.2, 0.7]])
    v = torch.tensor([[1.0, 0.0], [2.0, 0.5], [0.0, 1.0], [0.3, 1.5]])
    ev = WholePrefixAttentionEvaluator(q, k, v)
    keep = torch.tensor([[0, 2]], dtype=torch.long)
    evict = torch.tensor([[1, 3]], dtype=torch.long)
    best = torch.tensor([0, 1], dtype=torch.long)
    accepted = torch.tensor([True, False])
    counts = torch.ones(4, dtype=torch.int32)
    out = ev.compressed_output(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=accepted, counts_1d=counts,
    )
    manual = torch.linalg.vector_norm(out - ev.full_output, dim=-1) / ev.full_norm
    err = ev.compressed_error(
        keep_idx=keep, evict_idx=evict, best_idx=best,
        accepted_mask=accepted, counts_1d=counts,
    )
    assert torch.allclose(manual.flatten(), err, atol=1e-7, rtol=1e-7)


def test_whole_layer_post_o_proj_error_uses_joint_head_output():
    from whole_cache_attention import whole_layer_attention_relative_error
    # Two KV heads, two query heads per KV head, one probe, head_dim=2.
    # Identity o_proj means the layer output is exactly the concatenated 8-vector.
    full_heads = [
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
    ]
    comp_heads = [
        torch.tensor([[1.5, 0.0], [0.0, 2.0]]),
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    ]
    proj = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(8))
    got = whole_layer_attention_relative_error(
        full_head_outputs=full_heads,
        compressed_head_outputs=comp_heads,
        num_probes=1,
        o_proj=proj,
    )
    full = torch.tensor([1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0])
    comp = torch.tensor([1.5, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0])
    expected = float(torch.linalg.vector_norm(comp - full) / torch.linalg.vector_norm(full))
    assert abs(got - expected) < 1e-7, (got, expected)


def test_mapping_compressed_output_matches_mapping_error_definition():
    q = torch.tensor([[[1.0, 0.2]], [[-0.3, 0.8]]])
    k = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [-0.2, 0.7]])
    v = torch.tensor([[1.0, 0.0], [2.0, 0.5], [0.0, 1.0], [0.3, 1.5]])
    ev = WholePrefixAttentionEvaluator(q, k, v)
    mapping = [0, 0, 1, -1]
    out = ev.compressed_output_from_mapping(mapping)
    manual = torch.linalg.vector_norm(out - ev.full_output, dim=-1) / ev.full_norm
    err = ev.compressed_error_from_mapping(mapping)
    assert torch.allclose(manual.flatten(), err, atol=1e-7, rtol=1e-7)
