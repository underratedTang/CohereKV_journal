"""Single-Delta protocol for one-round CohereKV motivation controls.

Euclidean routing and Euclidean candidate conflicts use the SAME scalar
Delta_(layer,head) = lambda*S_calib at a given operating point.
Only cosine routing requires a separate, dimensionless tau_cos.
"""
from __future__ import annotations

import math


def shared_lambda_at_point(metric: str, route_param: float, lambda_value: float) -> float:
    """Return the Euclidean conflict coefficient at a Figure-2 route point.

    For Euclidean routing its sweep coordinate IS lambda. For cosine routing the
    sweep coordinate is cosine similarity tau, independent of the fixed lambda.
    """
    metric = str(metric)
    route_param = float(route_param)
    lambda_value = float(lambda_value)
    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda must be finite and positive")
    if metric == "euclidean":
        if not math.isfinite(route_param) or route_param <= 0:
            raise ValueError("Euclidean lambda must be finite and positive")
        return route_param
    if metric == "cosine":
        if not math.isfinite(route_param) or not 0 < route_param <= 1:
            raise ValueError("cosine similarity tau must lie in (0,1]")
        return lambda_value
    raise ValueError(f"unsupported routing metric: {metric}")
