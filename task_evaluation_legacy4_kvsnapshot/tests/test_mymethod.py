from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.mymethod_common import (  # noqa: E402
    append_score_rows,
    base_score_rows_for_kv_heads,
    build_merge_plan,
    _nearest_feasible_routes,
    effective_radius,
    grouplogitkv_importance_for_kv_heads,
    initial_or_existing_state,
    merge_layer,
    recent_window_importance,
    select_keep_indices,
    select_keep_indices_grouplogitkv,
)
from models.common.mymethod_score_capture import (  # noqa: E402
    align_score_chunks,
    physical_chunk_visibility,
    scaled_base_score_rows,
)
from models.llama.merge_mymethod_llama import Repair as LlamaRepair  # noqa: E402
from models.qwen.merge_mymethod_qwen import Repair as QwenRepair  # noqa: E402


class DummyAttention:
    pass


class DummyLayer:
    def __init__(self):
        self.self_attn = DummyAttention()


class DummyModel:
    def __init__(self, layers=1):
        self.model = SimpleNamespace(layers=[DummyLayer() for _ in range(layers)])


def _cache(keys, values=None):
    keys = torch.tensor(keys, dtype=torch.float32).view(1, 1, -1, 1)
    values = keys.clone() if values is None else torch.tensor(
        values, dtype=torch.float32
    ).view(1, 1, -1, 1)
    return ((keys, values),)


class SelectorAndHistoryTests(unittest.TestCase):
    def test_ring_history_and_linear_recency_weights(self):
        history = torch.zeros((1, 1, 4, 3), dtype=torch.float32)
        rows = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]]
        )
        history, ptr, scale = append_score_rows(history, 0, None, rows, 0.0)
        self.assertEqual(ptr, 3)
        self.assertEqual(scale, 0.0)
        importance = recent_window_importance(
            history.reshape(1, 4, 3), ptr, beta=2.0
        )
        # Conceptual oldest row is the untouched zero row, then the three inputs.
        weights = torch.linspace(1.0, 2.0, 4)
        expected = torch.tensor(
            [[weights[1], 2.0 * weights[2], 3.0 * weights[3]]]
        )
        torch.testing.assert_close(importance, expected)

        # Five new rows replace the whole W=4 ring with their last four rows.
        newer = torch.arange(1, 16, dtype=torch.float32).view(1, 1, 5, 3)
        history, ptr, _ = append_score_rows(history, ptr, scale, newer, 0.0)
        self.assertEqual(ptr, 0)
        torch.testing.assert_close(history, newer[..., -4:, :])

    def test_score_scale_alignment_is_common_rescaling(self):
        history = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        rows = torch.ones((1, 1, 1, 2), dtype=torch.float32)
        history, ptr, common = append_score_rows(history, 0, 0.0, rows, 2.0)
        self.assertEqual(common, 2.0)
        # Old rows are multiplied by exp(-2); current row remains one.
        self.assertAlmostEqual(float(history[0, 0, 1, 0]), float(torch.exp(torch.tensor(-2.0))), places=6)
        self.assertAlmostEqual(float(history[0, 0, 0, 0]), 1.0, places=6)
        # ptr=1 means row1 is oldest; row0 was overwritten by the new row.
        self.assertEqual(ptr, 1)

    def test_sink_recent_heavy_selector(self):
        importance = torch.tensor([[0.0, 0.0, 0.0, 9.0, 1.0, 2.0, 3.0, 4.0]])
        positions = torch.arange(8).view(1, 8)
        keep = select_keep_indices(
            importance, positions, keep_len=5, sink_size=2, recent_window=2
        )
        self.assertEqual(keep.tolist(), [[0, 1, 3, 6, 7]])

    def test_grouplogitkv_selector_matches_bottom_k_eviction(self):
        scores = torch.tensor([[0.9, 0.1, 0.4, 0.2, 0.8]])
        keep = select_keep_indices_grouplogitkv(scores, keep_len=3)
        # Bottom-K evicts indices 1 and 3; retained physical order is preserved.
        self.assertEqual(keep.tolist(), [[0, 2, 4]])

    def test_grouplogitkv_attention_reduction_matches_definition(self):
        attn = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]],
                              [[0.5, 0.5], [0.7, 0.3]]]])
        got = grouplogitkv_importance_for_kv_heads(attn, num_kv_heads=1)
        expected = attn.view(1, 1, 2, 2, 2).mean(dim=2).sum(dim=2)
        torch.testing.assert_close(got, expected)

    def test_gqa_base_score_rows_preserve_query_axis(self):
        scores = torch.tensor(
            [[
                [[1.0, 3.0], [2.0, 4.0]],
                [[3.0, 5.0], [4.0, 6.0]],
            ]]
        )
        reduced = base_score_rows_for_kv_heads(scores, num_kv_heads=1)
        expected = torch.tensor([[[[2.0, 4.0], [3.0, 5.0]]]])
        torch.testing.assert_close(reduced, expected)

    def test_scaled_base_scores_exclude_masked_future_and_log_count(self):
        logits = torch.tensor(
            [[[[0.0, 1.0, 9.0], [0.0, 1.0, 2.0]]]], dtype=torch.float32
        )
        visible = torch.tensor(
            [[[[True, True, False], [True, True, True]]]]
        )
        scores, scale = scaled_base_score_rows(logits, visible, num_kv_heads=1)
        self.assertEqual(scale, 2.0)
        self.assertEqual(float(scores[0, 0, 0, 2]), 0.0)
        torch.testing.assert_close(
            scores[0, 0, 1], torch.exp(torch.tensor([-2.0, -1.0, 0.0]))
        )

    def test_attention_score_chunk_alignment_matches_single_capture(self):
        logits = torch.tensor(
            [[[[0.0, 1.0], [2.0, -1.0], [0.5, 3.0], [1.5, 0.0]]]]
        )
        visible = torch.ones_like(logits, dtype=torch.bool)
        whole, whole_scale = scaled_base_score_rows(logits, visible, 1)
        chunks, scales = [], []
        for start, end in ((0, 2), (2, 4)):
            chunk, scale = scaled_base_score_rows(
                logits[:, :, start:end], visible[:, :, start:end], 1
            )
            chunks.append(chunk)
            scales.append(scale)
        joined, joined_scale = align_score_chunks(
            chunks, scales, (1, 1, 0, 2), logits.device
        )
        self.assertEqual(joined_scale, whole_scale)
        torch.testing.assert_close(joined, whole)

    def test_physical_chunk_visibility(self):
        ref = torch.zeros((1, 2, 3, 7))
        visible = physical_chunk_visibility(ref, q_len=3, key_len=7)[0, 0]
        expected = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(visible, expected)


class CertificateCoreTests(unittest.TestCase):
    def test_effective_radius_tightening(self):
        counts = torch.tensor([[1.0, 4.0]])
        radii = torch.tensor([[5.0, 5.0]])
        diameters = torch.tensor([[7.0, 2.0]])
        got = effective_radius(counts, radii, diameters)
        torch.testing.assert_close(got, torch.tensor([[0.0, 1.5]]))

    def test_nearest_infeasible_anchor_falls_back_to_feasible_anchor(self):
        keys = torch.tensor([[[10.0], [10.01], [10.20], [10.21]]])
        counts = torch.ones((1, 4))
        radii = torch.zeros((1, 4))
        diameters = torch.zeros((1, 4))
        # Anchor position 0 is selected as sink but is internally infeasible.
        diameters[0, 0] = 2.0
        importance = torch.zeros((1, 4))
        positions = torch.arange(4).view(1, 4)
        plan = build_merge_plan(
            keys, counts, radii, diameters, importance, positions,
            keep_len=2, tau=0.9, sink_size=1, recent_window=1,
            distance_chunk_size=1,
        )
        self.assertEqual(plan.keep_idx.tolist(), [[0, 3]])
        evicted = plan.evict_idx[0].tolist()
        slot = evicted.index(1)
        self.assertTrue(bool(plan.accepted[0, slot]))
        self.assertEqual(int(plan.best_idx[0, slot]), 1)

    def test_nearest_then_certify_discards_when_alternate_anchor_is_feasible(self):
        keys = torch.tensor([[[10.0], [10.01], [10.20], [10.21]]])
        counts = torch.ones((1, 4))
        radii = torch.zeros((1, 4))
        diameters = torch.zeros((1, 4))
        diameters[0, 0] = 2.0
        importance = torch.zeros((1, 4))
        positions = torch.arange(4).view(1, 4)
        nearest_feasible = build_merge_plan(
            keys, counts, radii, diameters, importance, positions,
            keep_len=2, tau=0.9, sink_size=1, recent_window=1,
            route_mode="nearest_feasible",
        )
        nearest_then = build_merge_plan(
            keys, counts, radii, diameters, importance, positions,
            keep_len=2, tau=0.9, sink_size=1, recent_window=1,
            route_mode="nearest_then_certify",
        )
        slot = nearest_feasible.evict_idx[0].tolist().index(1)
        self.assertTrue(bool(nearest_feasible.accepted[0, slot]))
        self.assertFalse(bool(nearest_then.accepted[0, slot]))

    def test_sink_absorption_ablation(self):
        keys = torch.tensor([[[100.0], [100.01], [100.20], [100.21]]])
        ones = torch.ones((1, 4))
        zeros = torch.zeros((1, 4))
        positions = torch.arange(4).view(1, 4)
        allowed = build_merge_plan(
            keys, ones, zeros, zeros, zeros, positions,
            keep_len=2, tau=0.5, sink_size=1, recent_window=1,
            sink_can_absorb=True,
        )
        forbidden = build_merge_plan(
            keys, ones, zeros, zeros, zeros, positions,
            keep_len=2, tau=0.5, sink_size=1, recent_window=1,
            sink_can_absorb=False,
        )
        slot = allowed.evict_idx[0].tolist().index(1)
        self.assertEqual(int(allowed.best_idx[0, slot]), 0)
        self.assertEqual(int(forbidden.best_idx[0, slot]), 1)

    def test_squared_distance_feasibility_matches_dense_sqrt_reference(self):
        torch.manual_seed(31)
        x, ne, nr, dim = 2, 7, 5, 6
        evicted = torch.randn((x, ne, dim))
        retained = torch.randn((x, nr, dim))
        eri = 0.03 * torch.rand((x, ne))
        rrj = 0.03 * torch.rand((x, nr))
        edi = 0.02 * torch.rand((x, ne))
        rdj = 0.02 * torch.rand((x, nr))
        delta = 4.0 * torch.ones((x, 1))
        allowed = torch.ones((x, nr), dtype=torch.bool)

        idx, dist, accepted = _nearest_feasible_routes(
            evicted, retained, eri, rrj, edi, rdj, delta, allowed,
            chunk_size=3, certify_groups=True,
        )

        dense_distance = torch.cdist(evicted.float(), retained.float())
        threshold = torch.minimum(delta * 0.5, delta - rrj).clamp_min(0.0)
        feasible = (
            (rdj <= delta)[:, None, :]
            & (edi <= delta).unsqueeze(-1)
            & allowed[:, None, :]
            & ((eri.unsqueeze(-1) + dense_distance) <= threshold[:, None, :])
        )
        masked = dense_distance.masked_fill(~feasible, float("inf"))
        ref_dist, ref_idx = masked.min(dim=-1)
        ref_accepted = torch.isfinite(ref_dist)
        ref_dist = torch.where(ref_accepted, ref_dist, torch.zeros_like(ref_dist))

        torch.testing.assert_close(idx, ref_idx)
        torch.testing.assert_close(accepted, ref_accepted)
        torch.testing.assert_close(dist, ref_dist, rtol=1e-5, atol=1e-6)

    def test_distance_tiling_does_not_change_routes(self):
        torch.manual_seed(13)
        keys = torch.randn((2, 9, 5))
        counts = torch.ones((2, 9), dtype=torch.int32)
        zeros = torch.zeros((2, 9))
        importance = torch.randn((2, 9))
        positions = torch.arange(9).view(1, 9).expand(2, -1)
        kwargs = dict(
            keys=keys, counts=counts, radii=zeros, diameters=zeros,
            importance=importance, positions=positions, keep_len=5,
            tau=0.6, sink_size=1, recent_window=2, certify_groups=True,
        )
        dense = build_merge_plan(distance_chunk_size=0, **kwargs)
        tiled = build_merge_plan(distance_chunk_size=2, **kwargs)
        for name in ("keep_idx", "evict_idx", "best_idx", "accepted"):
            torch.testing.assert_close(getattr(dense, name), getattr(tiled, name))
        torch.testing.assert_close(dense.assigned_distance, tiled.assigned_distance)

    def test_count_weighted_kv_history_and_certificates(self):
        keys = torch.tensor([10.0, 10.1, 13.0, 13.1]).view(1, 1, 4, 1)
        values = torch.tensor([0.0, 1.0, 30.0, 31.0]).view(1, 1, 4, 1)
        counts = torch.tensor([1.0, 2.0, 1.0, 3.0]).view(1, 1, 4)
        radii = torch.zeros_like(counts)
        diameters = torch.zeros_like(counts)
        history = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]]]
        )
        positions = torch.arange(4).view(1, 1, 4)

        knew, vnew, cnew, rnew, dnew, hnew, pnew = merge_layer(
            keys, values, counts, radii, diameters, history, 0, positions,
            ratio=0.5, tau=0.5, sink_size=1, recent_window=2,
            distance_chunk_size=1,
        )
        torch.testing.assert_close(cnew, torch.tensor([[[3, 4]]], dtype=torch.int32))
        torch.testing.assert_close(
            knew[..., 0], torch.tensor([[[10.0 + 0.2 / 3.0, 13.075]]]),
            atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            vnew[..., 0], torch.tensor([[[2.0 / 3.0, 30.75]]]),
            atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(rnew, torch.tensor([[[2.0 / 30.0, 0.075]]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dnew, torch.tensor([[[0.1, 0.1]]]), atol=1e-5, rtol=1e-5)
        expected_h0 = torch.tensor([[[(1.0 + 2 * 2.0) / 3.0, (4.0 * 3 + 3.0) / 4.0]]])
        torch.testing.assert_close(hnew[:, :, 0, :], expected_h0)
        self.assertEqual(pnew.tolist(), [[[0, 3]]])

    def test_optional_diagnostics_report_coverage_and_certificate_ratios(self):
        keys = torch.tensor([0.0, 0.05, 1.0, 1.05]).view(1, 1, 4, 1)
        values = keys.clone()
        counts = torch.ones((1, 1, 4), dtype=torch.int32)
        radii = torch.zeros((1, 1, 4), dtype=torch.float32)
        diameters = torch.zeros_like(radii)
        history = torch.zeros((1, 1, 0, 4), dtype=torch.float32)
        positions = torch.arange(4, dtype=torch.int32).view(1, 1, 4)
        importance = torch.tensor([[[4.0, 1.0, 3.0, 2.0]]])
        rows = []
        merge_layer(
            keys, values, counts, radii, diameters, history, 0, positions,
            ratio=0.5, target_length=2, tau=0.5, sink_size=0, recent_window=0,
            distance_chunk_size=0, selector="grouplogitkv",
            current_importance=importance, diagnostics_sink=rows,
            diagnostics_context={"layer_idx": 7},
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["layer_idx"], 7)
        self.assertEqual(row["evicted_entries"], 2)
        self.assertAlmostEqual(
            row["accepted_count_mass"] + row["rejected_count_mass"],
            row["evicted_count_mass"],
        )
        self.assertGreaterEqual(row["accepted_entry_ratio"], 0.0)
        self.assertLessEqual(row["accepted_entry_ratio"], 1.0)
        self.assertGreaterEqual(row["post_diameter_over_delta_max"], 0.0)

    def test_randomized_first_round_complete_diameter_invariant(self):
        torch.manual_seed(7)
        keys = 10.0 + 0.05 * torch.randn((1, 2, 12, 4))
        values = torch.randn_like(keys)
        counts = torch.ones((1, 2, 12))
        radii = torch.zeros_like(counts)
        diameters = torch.zeros_like(counts)
        history = torch.rand((1, 2, 3, 12))
        positions = torch.arange(12).view(1, 1, 12).expand(1, 2, -1).clone()
        keep_len = 7

        flat_keys = keys.reshape(2, 12, 4)
        importance = recent_window_importance(history.reshape(2, 3, 12), 0, 1.0)
        plan = build_merge_plan(
            flat_keys, counts.reshape(2, 12), radii.reshape(2, 12),
            diameters.reshape(2, 12), importance, positions.reshape(2, 12),
            keep_len=keep_len, tau=0.9, sink_size=2, recent_window=3,
            distance_chunk_size=3,
        )
        knew, _, cnew, rnew, dnew, _, _ = merge_layer(
            keys, values, counts, radii, diameters, history, 0, positions,
            ratio=0.0, target_length=keep_len, tau=0.9,
            sink_size=2, recent_window=3, distance_chunk_size=3,
        )
        for row in range(2):
            original = flat_keys[row]
            for anchor_slot in range(keep_len):
                members = [int(plan.keep_idx[row, anchor_slot])]
                for evict_slot in range(plan.evict_idx.shape[1]):
                    if bool(plan.accepted[row, evict_slot]) and int(plan.best_idx[row, evict_slot]) == anchor_slot:
                        members.append(int(plan.evict_idx[row, evict_slot]))
                member_keys = original[torch.tensor(members)]
                expected_centroid = member_keys.mean(dim=0)
                got = knew.reshape(2, keep_len, 4)[row, anchor_slot]
                torch.testing.assert_close(got, expected_centroid, atol=1e-5, rtol=1e-5)
                self.assertEqual(int(cnew.reshape(2, keep_len)[row, anchor_slot]), len(members))
                true_radius = torch.linalg.vector_norm(member_keys - got, dim=-1).max()
                true_diameter = torch.cdist(member_keys, member_keys).max()
                self.assertLessEqual(float(true_radius), float(rnew.reshape(2, keep_len)[row, anchor_slot]) + 1e-5)
                self.assertLessEqual(float(true_diameter), float(dnew.reshape(2, keep_len)[row, anchor_slot]) + 1e-5)
                if len(members) > 1:
                    self.assertLessEqual(float(true_diameter), float(plan.delta[row, 0]) + 1e-5)

    def test_state_suffix_initialization(self):
        model = DummyModel()
        keys = torch.zeros((1, 1, 2, 4))
        c, r, d, h, ptr, p, next_p, scale = initial_or_existing_state(model, 0, keys, 3)
        model._mymethod_counts = [c]
        model._mymethod_radii = [r]
        model._mymethod_diameters = [d]
        model._mymethod_score_history = [h]
        model._mymethod_score_history_ptrs = [ptr]
        model._mymethod_score_log_scales = [scale]
        model._mymethod_positions = [p]
        model._mymethod_next_positions = [next_p]
        longer = torch.zeros((1, 1, 4, 4))
        c2, r2, d2, h2, _, p2, next_p2, _ = initial_or_existing_state(model, 0, longer, 3)
        self.assertEqual(p2.tolist(), [[[0, 1, 2, 3]]])
        self.assertEqual(next_p2.tolist(), [[[4]]])
        torch.testing.assert_close(c2[..., 2:], torch.ones((1, 1, 2), dtype=torch.int32))
        torch.testing.assert_close(r2[..., 2:], torch.zeros((1, 1, 2)))
        torch.testing.assert_close(d2[..., 2:], torch.zeros((1, 1, 2)))
        torch.testing.assert_close(h2[..., 2:], torch.zeros((1, 1, 3, 2)))


    def test_logical_positions_do_not_collapse_after_latest_entry_is_evicted(self):
        model = DummyModel()
        keys = torch.zeros((1, 1, 3, 2))
        c, r, d, h, ptr, p, next_p, scale = initial_or_existing_state(model, 0, keys, 2)
        # Simulate a selector that retained original positions 0 and 1 but dropped 2.
        model._mymethod_counts = [c[..., :2]]
        model._mymethod_radii = [r[..., :2]]
        model._mymethod_diameters = [d[..., :2]]
        model._mymethod_score_history = [h[..., :2]]
        model._mymethod_score_history_ptrs = [ptr]
        model._mymethod_score_log_scales = [scale]
        model._mymethod_positions = [p[..., :2]]
        model._mymethod_next_positions = [next_p]
        incoming = torch.zeros((1, 1, 3, 2))
        *_, positions, next_position, _ = initial_or_existing_state(model, 0, incoming, 2)
        self.assertEqual(positions.tolist(), [[[0, 1, 3]]])
        self.assertEqual(next_position.tolist(), [[[4]]])


class ArchitectureWrapperTests(unittest.TestCase):
    def test_llama_and_qwen_wrappers_match(self):
        cache = _cache([10.0, 10.1, 13.0, 13.1], [0.0, 1.0, 30.0, 31.0])
        score_rows = (torch.tensor([[[[1.0, 0.5, 0.25, 0.1]]]]),)
        llama = DummyModel()
        qwen = DummyModel()
        out_l = LlamaRepair(
            llama, cache, score_rows, None, 0.5, 0.5,
            sink_size=1, recent_window=1, distance_chunk_size=1,
        )
        out_q = QwenRepair(
            qwen, cache, score_rows, None, 0.5, 0.5,
            sink_size=1, recent_window=1, distance_chunk_size=1,
        )
        torch.testing.assert_close(out_l[0][0], out_q[0][0])
        torch.testing.assert_close(out_l[0][1], out_q[0][1])
        for name in (
            "_mymethod_counts", "_mymethod_radii", "_mymethod_diameters",
            "_mymethod_score_history", "_mymethod_positions",
            "_mymethod_next_positions",
        ):
            torch.testing.assert_close(getattr(llama, name)[0], getattr(qwen, name)[0])

    def test_llama_and_qwen_grouplogit_selector_match_without_history(self):
        cache = _cache([10.0, 10.1, 13.0, 13.1], [0.0, 1.0, 30.0, 31.0])
        # Already reduced GroupLogitKV current post-softmax attention scores.
        scores = (torch.tensor([[[0.9, 0.1, 0.8, 0.2]]]),)
        llama = DummyModel()
        qwen = DummyModel()
        out_l = LlamaRepair(
            llama, cache, scores, None, 0.5, 0.5,
            distance_chunk_size=2, selector="grouplogitkv",
        )
        out_q = QwenRepair(
            qwen, cache, scores, None, 0.5, 0.5,
            distance_chunk_size=2, selector="grouplogitkv",
        )
        torch.testing.assert_close(out_l[0][0], out_q[0][0])
        torch.testing.assert_close(out_l[0][1], out_q[0][1])
        self.assertEqual(llama._mymethod_score_history[0].shape[-2], 0)
        self.assertEqual(qwen._mymethod_score_history[0].shape[-2], 0)
        self.assertEqual(llama.model.layers[0].self_attn._mymethod_selector, "grouplogitkv")

    def test_two_round_certificate_state_is_finite(self):
        model = DummyModel()
        cache = _cache([10.0, 10.001, 10.002, 10.003])
        scores = (torch.ones((1, 1, 2, 4)),)
        first = LlamaRepair(
            model, cache, scores, None, 0.5, 0.5,
            sink_size=1, recent_window=2, distance_chunk_size=1,
        )
        self.assertAlmostEqual(float(model._mymethod_counts[0].sum()), 4.0)

        key2 = torch.cat([first[0][0], torch.tensor([[[[10.0015]]]])], dim=2)
        val2 = torch.cat([first[0][1], torch.tensor([[[[10.0015]]]])], dim=2)
        scores2 = (torch.ones((1, 1, 1, 3)),)
        second = LlamaRepair(
            model, ((key2, val2),), scores2, None, 1.0 / 3.0, 0.5,
            sink_size=1, recent_window=2, distance_chunk_size=1,
        )
        self.assertEqual(second[0][0].shape[-2], 2)
        self.assertAlmostEqual(float(model._mymethod_counts[0].sum()), 5.0)
        self.assertTrue(bool(torch.isfinite(model._mymethod_radii[0]).all()))
        self.assertTrue(bool(torch.isfinite(model._mymethod_diameters[0]).all()))
        self.assertTrue(bool((model._mymethod_radii[0] <= model._mymethod_diameters[0] + 1e-6).all()))


if __name__ == "__main__":
    unittest.main()
