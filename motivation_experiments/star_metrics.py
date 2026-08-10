from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class GroupMetrics:
    anchor_local_index: int
    anchor_position: int
    candidate_count: int
    min_leaf_cosine: float
    max_leaf_angle: float
    normalized_max_angle: float
    unsafe: bool
    pair_count: int
    conflict_pair_count: int
    conflict_pair_fraction: float
    min_anchor_cosine: float
    mean_anchor_cosine: float


def parse_float_csv(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one floating-point value.")
    return values


def parse_int_csv(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def resolve_selection(text: str, total: int) -> List[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(total))
    if ":" in text and "," not in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice selection: {text}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else total
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        values = list(range(start, stop, step))
    else:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Selection {text!r} resolved to no indices.")
    bad = [i for i in values if i < 0 or i >= total]
    if bad:
        raise ValueError(f"Selection {text!r} contains out-of-range indices {bad}; total={total}.")
    return values


def deterministic_anchor_partition(length: int, budget: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if length < 3:
        raise ValueError("Sequence must contain at least 3 tokens.")
    if not 0.0 < budget < 1.0:
        raise ValueError(f"Anchor budget must be in (0,1), got {budget}.")
    n_anchor = int(np.floor(float(budget) * length))
    n_anchor = max(1, min(length - 1, n_anchor))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(length)
    anchors = np.sort(perm[:n_anchor].astype(np.int64, copy=False))
    candidates = np.sort(perm[n_anchor:].astype(np.int64, copy=False))
    return anchors, candidates


def compute_group_metrics(
    normalized_keys: np.ndarray,
    candidate_positions: np.ndarray,
    assigned_anchor_local: np.ndarray,
    assigned_anchor_cosine: np.ndarray,
    anchor_positions: np.ndarray,
    tau: float,
    numeric_eps: float = 1e-6,
) -> List[GroupMetrics]:
    """Compute group-wise star statistics after thresholding nearest-anchor assignments.

    Each candidate is already assigned to exactly one nearest anchor. A candidate is
    admitted only if its anchor cosine is >= tau. Only groups with >=2 admitted
    candidates are eligible because a star needs at least two leaves.
    """
    if not -1.0 < tau < 1.0:
        raise ValueError(f"tau must be in (-1,1), got {tau}.")
    theta_tau = float(np.arccos(np.clip(tau, -1.0, 1.0)))
    if theta_tau <= 0.0:
        raise ValueError("tau is too close to 1 for a stable angular normalization.")

    admitted = assigned_anchor_cosine >= float(tau)
    if not np.any(admitted):
        return []

    cand = candidate_positions[admitted]
    assn = assigned_anchor_local[admitted]
    acos = assigned_anchor_cosine[admitted]

    order = np.argsort(assn, kind="stable")
    cand = cand[order]
    assn = assn[order]
    acos = acos[order]

    uniq, starts, counts = np.unique(assn, return_index=True, return_counts=True)
    rows: List[GroupMetrics] = []
    for anchor_local, start, count in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        if count < 2:
            continue
        positions = cand[start : start + count]
        leaf = normalized_keys[positions].astype(np.float32, copy=False)
        sim = leaf @ leaf.T
        tri = np.triu_indices(count, k=1)
        pair_sims = np.clip(sim[tri], -1.0, 1.0)
        pair_count = int(pair_sims.size)
        if pair_count == 0:
            continue
        min_leaf_cosine = float(pair_sims.min())
        max_leaf_angle = float(np.arccos(min_leaf_cosine))
        normalized_max_angle = float(max_leaf_angle / theta_tau)
        conflict_pair_count = int(np.count_nonzero(pair_sims < float(tau) - float(numeric_eps)))
        unsafe = bool(conflict_pair_count > 0)
        anchor_cos = acos[start : start + count]
        rows.append(
            GroupMetrics(
                anchor_local_index=int(anchor_local),
                anchor_position=int(anchor_positions[int(anchor_local)]),
                candidate_count=int(count),
                min_leaf_cosine=min_leaf_cosine,
                max_leaf_angle=max_leaf_angle,
                normalized_max_angle=normalized_max_angle,
                unsafe=unsafe,
                pair_count=pair_count,
                conflict_pair_count=conflict_pair_count,
                conflict_pair_fraction=float(conflict_pair_count / pair_count),
                min_anchor_cosine=float(anchor_cos.min()),
                mean_anchor_cosine=float(anchor_cos.mean()),
            )
        )
    return rows


def aggregate_group_metrics(rows: Sequence[GroupMetrics]) -> Dict[str, float]:
    eligible = len(rows)
    if eligible == 0:
        return {
            "eligible_groups": 0,
            "unsafe_groups": 0,
            "unsafe_group_rate": float("nan"),
            "candidate_pairs": 0,
            "conflict_pairs": 0,
            "conflict_pair_rate": float("nan"),
            "mean_candidate_count": float("nan"),
            "max_candidate_count": 0,
            "mean_normalized_max_angle": float("nan"),
            "median_normalized_max_angle": float("nan"),
        }
    unsafe = sum(int(r.unsafe) for r in rows)
    pairs = sum(r.pair_count for r in rows)
    conflicts = sum(r.conflict_pair_count for r in rows)
    z = np.asarray([r.normalized_max_angle for r in rows], dtype=np.float64)
    counts = np.asarray([r.candidate_count for r in rows], dtype=np.float64)
    return {
        "eligible_groups": int(eligible),
        "unsafe_groups": int(unsafe),
        "unsafe_group_rate": float(unsafe / eligible),
        "candidate_pairs": int(pairs),
        "conflict_pairs": int(conflicts),
        "conflict_pair_rate": float(conflicts / pairs) if pairs else float("nan"),
        "mean_candidate_count": float(counts.mean()),
        "max_candidate_count": int(counts.max()),
        "mean_normalized_max_angle": float(z.mean()),
        "median_normalized_max_angle": float(np.median(z)),
    }
