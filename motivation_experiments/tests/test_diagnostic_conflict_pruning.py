#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagnostic_conflict_pruning import (  # noqa: E402
    counterfactual_conflict_reduction_mask,
    conflict_mask_from_score,
    conflict_stats,
    endpoint_drop_masks,
    geometry_conflict_reduction_mask,
    most_severe_conflict_edge,
    select_high_conflict_source,
)


def test_conflict_mask_uses_threshold_and_symmetry():
    s = torch.tensor([[0., .8, .4], [.8, 0., .9], [.4, .9, 0.]])
    c = conflict_mask_from_score(s, .75)
    assert c.tolist() == [[False, True, False], [True, False, True], [False, True, False]]


def test_most_severe_edge_utility_is_still_deterministic():
    s = torch.tensor([[0., .8, .4], [.8, 0., .95], [.4, .95, 0.]])
    assert most_severe_conflict_edge(s, .75) == (1, 2)


def test_endpoint_drop_masks_remove_exactly_one_endpoint():
    ku, kv = endpoint_drop_masks(4, (1, 3))
    assert ku.tolist() == [True, False, True, True]
    assert kv.tolist() == [True, True, True, False]


def test_conflict_stats_counts_undirected_edges_once():
    c = torch.tensor([[0,1,1],[1,0,0],[1,0,0]], dtype=torch.bool)
    e, v = conflict_stats(c)
    assert e == 2 and v == 3


def test_degree_first_selection_beats_single_edge_severity():
    # Vertex 0 has degree 3; edge (2,3) is numerically most severe but its
    # endpoints have degree 2. The diagnostic must pick vertex 0.
    score = torch.tensor([
        [0.0, 0.70, 0.65, 0.60],
        [0.70, 0.0, 0.10, 0.10],
        [0.65, 0.10, 0.0, 0.95],
        [0.60, 0.10, 0.95, 0.0],
    ])
    badness = torch.tensor([0.2, 0.9, 0.8, 0.7])
    selected = select_high_conflict_source(
        score=score, threshold=0.5, route_badness=badness,
        physical_positions=torch.tensor([10,20,30,40]),
    )
    assert selected is not None
    assert selected[0] == 0 and selected[1] == 3


def test_degree_tie_uses_worse_anchor_compatibility():
    # Path 0-1-2 gives degrees [1,2,1]; instead use two disjoint edges so all
    # four vertices tie at degree 1 and route badness decides.
    score = torch.tensor([
        [0.0, 0.9, 0.1, 0.1],
        [0.9, 0.0, 0.1, 0.1],
        [0.1, 0.1, 0.0, 0.8],
        [0.1, 0.1, 0.8, 0.0],
    ])
    badness = torch.tensor([0.2, 0.7, 0.4, 0.3])
    selected = select_high_conflict_source(
        score=score, threshold=0.5, route_badness=badness,
        physical_positions=torch.tensor([10,20,30,40]),
    )
    assert selected is not None and selected[0] == 1


def test_geometry_conflict_reduction_supports_configurable_multiple_removals():
    # Star centered at 0: remove center first because of degree 3. Then the
    # remaining edge (1,2) is resolved by badness, removing 2.
    score = torch.tensor([
        [0.0, 0.90, 0.85, 0.80],
        [0.90, 0.0, 0.70, 0.10],
        [0.85, 0.70, 0.0, 0.10],
        [0.80, 0.10, 0.10, 0.0],
    ])
    badness = torch.tensor([0.1, 0.2, 0.8, 0.4])
    keep, removed = geometry_conflict_reduction_mask(
        score=score, threshold=0.5, route_badness=badness,
        max_removals=2, physical_positions=torch.tensor([0,1,2,3]),
    )
    assert removed == [0, 2]
    assert keep.tolist() == [False, True, False, True]


def test_selection_respects_active_keep_mask():
    score = torch.tensor([
        [0.0, 0.9, 0.9],
        [0.9, 0.0, 0.8],
        [0.9, 0.8, 0.0],
    ])
    badness = torch.tensor([0.1, 0.5, 0.6])
    selected = select_high_conflict_source(
        score=score, threshold=0.5, route_badness=badness,
        keep=torch.tensor([False, True, True]),
    )
    assert selected is not None and selected[0] == 2


def test_counterfactual_conflict_reduction_chooses_better_endpoint_and_supports_k():
    score = torch.tensor([
        [0.0, 0.90, 0.20],
        [0.90, 0.0, 0.40],
        [0.20, 0.40, 0.0],
    ])

    # Prefer keeping source 0, then source 2.  The helper must first diagnose
    # edge (0,1), choose dropping 1, then no active conflict remains.
    def fake_error(keep):
        weights = torch.tensor([0.1, 1.0, 0.2])
        return float((~keep).float().dot(weights).item())

    keep, removed, trace = counterfactual_conflict_reduction_mask(
        score=score, threshold=0.5, max_removals=2, evaluate_keep=fake_error,
    )
    assert removed == [0] or removed == [1]
    # fake_error penalizes dropping 1 more heavily, so the better branch drops 0.
    assert removed[0] == 0
    assert len(trace) == 1
    assert keep.tolist() == [False, True, True]
