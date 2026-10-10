from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from motivation_controls import matched_budget_route_view  # noqa: E402
from star_group_planner import precompute_routing_scores, route_threshold, route_view  # noqa: E402


def _scores():
    # anchors 0,1; evicted 2..5 have distinct Euclidean and cosine qualities
    keys = torch.tensor([[[1.,0.],[0.,1.],[.95,.05],[.8,.2],[.3,.7],[.1,.9]]])
    keep = torch.tensor([[0,1]], dtype=torch.long)
    evict = torch.tensor([[2,3,4,5]], dtype=torch.long)
    return precompute_routing_scores(keys=keys, keep_idx=keep, evict_idx=evict)


def test_matched_budget_euclidean_changes_only_scalar_threshold():
    scores = _scores()
    fixed = route_view(scores=scores, metric='euclidean', threshold=route_threshold('euclidean', 1.0, 1.0))
    assert int(fixed.compatible.sum()) == 4
    matched, param, removed = matched_budget_route_view(
        scores=scores, metric='euclidean', fixed_param=1.0, k_scale=1.0, remove_count=2
    )
    assert removed == 2
    assert int(matched.compatible.sum()) == 2
    expected = scores.euclidean_best_distance[0] <= route_threshold('euclidean', param, 1.0)
    assert torch.equal(matched.compatible, expected)


def test_matched_budget_cosine_changes_only_scalar_threshold():
    scores = _scores()
    matched, param, removed = matched_budget_route_view(
        scores=scores, metric='cosine', fixed_param=0.0, k_scale=99.0, remove_count=1
    )
    assert removed == 1
    assert int(matched.compatible.sum()) == 3
    expected = scores.cosine_best_cosine[0] >= param
    assert torch.equal(matched.compatible, expected)
