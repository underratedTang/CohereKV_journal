from __future__ import annotations

"""Pure routing-only controls used by Motivation Figure 2."""

import torch

from star_group_planner import route_threshold, route_view


@torch.inference_mode()
def matched_budget_route_view(*, scores, metric: str, fixed_param: float,
                              k_scale: float, remove_count: int):
    """Tighten one scalar routing threshold to remove about ``remove_count`` sources.

    The control never observes conflict identities when choosing sources. It only
    changes the routing cutoff, so the kept set is the prefix induced by routing
    score under the selected metric.
    """
    fixed = route_view(
        scores=scores, metric=metric,
        threshold=route_threshold(metric, fixed_param, k_scale),
    )
    fixed_n = int(fixed.compatible.sum().item())
    target = max(0, fixed_n - max(0, int(remove_count)))
    if target >= fixed_n:
        return fixed, float(fixed_param), 0

    active = fixed.compatible
    metric = str(metric).lower()
    if metric == "euclidean":
        vals = (fixed.assigned_distance[active] / max(float(k_scale), 1e-12)).float()
        if target <= 0:
            param = -1e-7
        else:
            param = float(torch.sort(vals).values[target - 1].item())
    elif metric == "cosine":
        vals = fixed.assigned_cosine[active].float()
        if target <= 0:
            param = 1.0 + 1e-7
        else:
            param = float(torch.sort(vals, descending=True).values[target - 1].item())
    else:
        raise ValueError(metric)

    matched = route_view(
        scores=scores, metric=metric,
        threshold=route_threshold(metric, param, k_scale),
    )
    actual_removed = fixed_n - int(matched.compatible.sum().item())
    return matched, param, actual_removed
