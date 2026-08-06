"""Common implementation of the updated ``mymethod`` KV-cache compressor.

This module follows ``method_motivation_method_theory_zh_keepkv_selector.md``:

* sink-protected KeepKV-style recent-window selector;
* score history from masked ``exp(q @ k.T / sqrt(d))`` without ``log(C)``;
* count-weighted centroid RMS scale;
* effective radius ``min(R, (C-1)/C * D)``;
* pair-wise feasibility followed by nearest-feasible routing;
* count-weighted K/V/history reductions;
* cross-round radius and complete-diameter certificate propagation.

The certificate core stores one count and two FP32 certificates per physical KV
entry.  The default selector additionally stores original positions and a
fixed-size ring buffer of recent base-score rows.  Decode-time ``log(C)`` is
computed by the architecture runtime patches and is never persisted separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


DEFAULT_TAU = 0.7
DEFAULT_SINK_SIZE = 4
DEFAULT_RECENT_WINDOW = 128
DEFAULT_RECENCY_BETA = 1.0
DEFAULT_DISTANCE_CHUNK_SIZE = 256
DEFAULT_SINK_CAN_ABSORB = True
DEFAULT_SELECTOR = "keepkv"
VALID_SELECTORS = ("keepkv", "grouplogitkv")
DEFAULT_ROUTE_MODE = "nearest_feasible"
VALID_ROUTE_MODES = ("nearest_feasible", "nearest_then_certify")
_SCALE_EPS = 1e-6
_REDUCTION_EPS = 1e-8
_POSITION_DTYPE = torch.int32
_COUNT_DTYPE = torch.int32
_RECENCY_WEIGHT_CACHE = {}


@dataclass
class MergePlan:
    """Retained split and deterministic nearest-feasible routing."""

    keep_idx: torch.Tensor           # [X, M]
    evict_idx: torch.Tensor          # [X, N]
    best_idx: torch.Tensor           # [X, N], local index in keep_idx
    accepted: torch.Tensor           # [X, N] bool
    assigned_distance: torch.Tensor  # [X, N] FP32; zero for rejected entries
    candidate_cover: torch.Tensor    # [X, N] = Rhat_i + assigned_distance
    delta: torch.Tensor              # [X, 1]
    scale: torch.Tensor              # [X, 1]


def as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def gather_seq(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather flattened state along sequence dimension 1."""
    if x.dim() == 3:
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if x.dim() == 2:
        return torch.gather(x, 1, idx)
    raise ValueError(f"Unsupported gather source shape: {tuple(x.shape)}")


def gather_history(history: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather score history ``[X,W,L]`` along its last dimension."""
    if history.dim() != 3:
        raise ValueError(f"Expected history [X,W,L], got {tuple(history.shape)}")
    return torch.gather(
        history,
        2,
        idx.unsqueeze(1).expand(-1, history.shape[1], -1),
    )


def base_score_rows_for_kv_heads(score_rows: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Return selector base-score rows as ``[B,H_kv,Q,K]`` FP32.

    Runtime patches already reduce GQA query heads.  The query-head form remains
    accepted for tests and compatibility, and is averaged over query heads that
    share one physical KV head.  A legacy ``[B,H_kv,K]`` tensor is treated as one
    query row rather than summed into an all-history statistic.
    """
    if score_rows is None:
        raise ValueError("Missing mymethod base-score rows")
    if score_rows.dim() == 3:
        if score_rows.shape[1] != num_kv_heads:
            raise ValueError(
                f"Expected H_kv={num_kv_heads}, got {tuple(score_rows.shape)}"
            )
        return score_rows.float().unsqueeze(-2)
    if score_rows.dim() != 4:
        raise ValueError(
            f"Expected score rows [B,H,Q,K] or [B,H_kv,K], got {tuple(score_rows.shape)}"
        )
    bsz, heads, q_len, k_len = score_rows.shape
    if heads == num_kv_heads:
        return score_rows.float()
    if heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads={heads} is not divisible by KV heads={num_kv_heads}"
        )
    groups = heads // num_kv_heads
    return score_rows.float().view(
        bsz, num_kv_heads, groups, q_len, k_len
    ).mean(dim=2)


# Backward-compatible name used by older local utilities.  The updated method
# preserves the query axis; callers must not interpret the result as cumulative
# attention mass.
def token_scores_for_kv_heads(score_rows: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    return base_score_rows_for_kv_heads(score_rows, num_kv_heads)


def grouplogitkv_importance_for_kv_heads(
    attention_scores: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Match GroupLogitKV retained/evicted importance exactly.

    Accepted inputs are either already reduced ``[B,H_kv,K]`` post-softmax
    attention sums captured by the runtime patch, or full
    ``[B,H_q,Q,K]`` post-softmax attention probabilities. Query heads sharing
    one KV head are averaged and query positions are summed.
    """
    if attention_scores is None:
        raise ValueError("Missing GroupLogitKV selector attention scores")
    if attention_scores.dim() == 3:
        if attention_scores.shape[1] != num_kv_heads:
            raise ValueError(
                f"Expected H_kv={num_kv_heads}, got {tuple(attention_scores.shape)}"
            )
        return attention_scores.float()
    if attention_scores.dim() != 4:
        raise ValueError(
            f"Expected attention [B,H,Q,K] or reduced [B,H_kv,K], got "
            f"{tuple(attention_scores.shape)}"
        )
    bsz, heads, q_len, k_len = attention_scores.shape
    if heads == num_kv_heads:
        return attention_scores.float().sum(dim=2)
    if heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads={heads} is not divisible by KV heads={num_kv_heads}"
        )
    groups = heads // num_kv_heads
    return attention_scores.float().view(
        bsz, num_kv_heads, groups, q_len, k_len
    ).mean(dim=2).sum(dim=2)


def normalize_route_mode(route_mode: str) -> str:
    value = str(route_mode).strip().lower()
    if value not in VALID_ROUTE_MODES:
        raise ValueError(
            f"Unknown mymethod route mode {route_mode!r}; choose one of {VALID_ROUTE_MODES}"
        )
    return value


def normalize_selector(selector: str) -> str:
    value = str(selector).strip().lower()
    if value not in VALID_SELECTORS:
        raise ValueError(
            f"Unknown mymethod selector {selector!r}; choose one of {VALID_SELECTORS}"
        )
    return value


def _ring_order_indices(window: int, ptr: int, device) -> torch.Tensor:
    if window <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return (int(ptr) + torch.arange(window, device=device)) % window


def _resize_ring_history(
    history: torch.Tensor,
    ptr: int,
    new_window: int,
) -> Tuple[torch.Tensor, int]:
    """Resize a ring while preserving the newest available rows."""
    new_window = max(int(new_window), 0)
    old_window = int(history.shape[-2])
    if new_window == old_window:
        return history, int(ptr) % max(old_window, 1) if old_window else 0
    new_shape = (*history.shape[:-2], new_window, history.shape[-1])
    if new_window == 0:
        return history.new_zeros(new_shape), 0
    out = history.new_zeros(new_shape)
    if old_window:
        ordered = history.index_select(-2, _ring_order_indices(old_window, ptr, history.device))
        copy_rows = min(old_window, new_window)
        out[..., -copy_rows:, :] = ordered[..., -copy_rows:, :]
    # Physical row zero is now the oldest conceptual row and the next write.
    return out, 0


def append_score_rows(
    history: torch.Tensor,
    ptr: int,
    history_log_scale: Optional[float],
    current_rows: torch.Tensor,
    current_log_scale: float,
) -> Tuple[torch.Tensor, int, Optional[float]]:
    """Append chronologically ordered score rows to a fixed-size ring.

    Runtime capture stores ``exp(base_logit-current_log_scale)``.  Existing and
    current rows are re-expressed under one common scalar log scale.  This is an
    exact common positive rescaling of the mathematical ``P`` tensor, so TopK,
    weighted sums, and count-weighted history reductions are unchanged while
    avoiding FP32 overflow.
    """
    if history.dim() != 4 or current_rows.dim() != 4:
        raise ValueError(
            f"Expected history/current [B,H,W/L,K], got {tuple(history.shape)} and "
            f"{tuple(current_rows.shape)}"
        )
    if history.shape[:2] != current_rows.shape[:2] or history.shape[-1] != current_rows.shape[-1]:
        raise ValueError(
            f"History/current shape mismatch: {tuple(history.shape)} vs {tuple(current_rows.shape)}"
        )
    window = int(history.shape[-2])
    q_len = int(current_rows.shape[-2])
    if window == 0 or q_len == 0:
        return history, 0 if window == 0 else int(ptr), history_log_scale

    current_log_scale = float(current_log_scale)
    if history_log_scale is None:
        common_scale = current_log_scale
    else:
        common_scale = max(float(history_log_scale), current_log_scale)
        old_factor = math.exp(float(history_log_scale) - common_scale)
        if old_factor != 1.0:
            history.mul_(old_factor)
    current_factor = math.exp(current_log_scale - common_scale)
    rows = current_rows.float()
    if current_factor != 1.0:
        rows = rows * current_factor

    if q_len >= window:
        history.copy_(rows[..., -window:, :])
        return history, 0, common_scale

    ptr = int(ptr) % window
    first = min(q_len, window - ptr)
    history[..., ptr:ptr + first, :] = rows[..., :first, :]
    remaining = q_len - first
    if remaining:
        history[..., :remaining, :] = rows[..., first:, :]
    return history, (ptr + q_len) % window, common_scale


def recent_window_importance(
    history: torch.Tensor,
    ptr: int,
    beta: float = DEFAULT_RECENCY_BETA,
) -> torch.Tensor:
    """Compute linearly time-weighted recent-window importance ``J``."""
    if history.dim() != 3:
        raise ValueError(f"Expected flattened history [X,W,L], got {tuple(history.shape)}")
    beta = float(beta)
    if beta < 1.0:
        raise ValueError(f"mymethod recency beta must be >=1, got {beta}")
    window = int(history.shape[1])
    if window == 0:
        return torch.zeros(
            (history.shape[0], history.shape[-1]),
            device=history.device,
            dtype=torch.float32,
        )
    if window == 1:
        return history[:, 0, :].float()
    # All layers of one model use the same W/beta/ptr in a compression round.
    # Cache this tiny vector to avoid repeatedly allocating linspace/index/scatter
    # tensors for every layer.  ``ptr`` points to the oldest physical row, so a
    # right roll maps chronological weights onto ring-buffer storage order.
    device_index = history.device.index if history.device.index is not None else -1
    cache_key = (history.device.type, device_index, window, int(ptr) % window, beta)
    physical_weights = _RECENCY_WEIGHT_CACHE.get(cache_key)
    if physical_weights is None or physical_weights.device != history.device:
        chronological_weights = torch.linspace(
            1.0, beta, window, device=history.device, dtype=torch.float32
        )
        physical_weights = torch.roll(
            chronological_weights, shifts=int(ptr) % window, dims=0
        )
        _RECENCY_WEIGHT_CACHE[cache_key] = physical_weights
    return (history.float() * physical_weights.view(1, window, 1)).sum(dim=1)


def _select_keep_indices_fallback(
    importance: torch.Tensor,
    positions: torch.Tensor,
    keep_len: int,
    sink_size: int,
    recent_window: int,
) -> torch.Tensor:
    rows = []
    for row in range(importance.shape[0]):
        pos = positions[row]
        imp = importance[row]
        sink_idx = torch.nonzero(pos < sink_size, as_tuple=False).flatten()
        if sink_idx.numel() > keep_len:
            raise ValueError(
                f"mymethod retained budget {keep_len} cannot hold "
                f"{sink_idx.numel()} mandatory sink entries"
            )
        selected = torch.zeros_like(pos, dtype=torch.bool)
        selected[sink_idx] = True
        rec_k = min(max(int(recent_window), 0), keep_len - int(sink_idx.numel()))
        if rec_k:
            rec_idx = torch.topk(
                pos.float().masked_fill(selected, float("-inf")),
                k=rec_k,
                largest=True,
                sorted=False,
            ).indices
            selected[rec_idx] = True
        hh_k = keep_len - int(selected.sum().item())
        if hh_k:
            hh_idx = torch.topk(
                imp.float().masked_fill(selected, float("-inf")),
                k=hh_k,
                largest=True,
                sorted=False,
            ).indices
            selected[hh_idx] = True
        rows.append(torch.nonzero(selected, as_tuple=False).flatten())
    return torch.stack(rows, dim=0)


def select_keep_indices(
    importance: torch.Tensor,
    positions: torch.Tensor,
    keep_len: int,
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> torch.Tensor:
    """Sink, recent, then recent-window-heavy retained-set selector."""
    if importance.dim() != 2 or positions.dim() != 2 or importance.shape != positions.shape:
        raise ValueError(
            f"importance/positions must share [X,L], got "
            f"{tuple(importance.shape)} and {tuple(positions.shape)}"
        )
    _, length = importance.shape
    keep_len = int(keep_len)
    if not 1 <= keep_len <= length:
        raise ValueError(f"Invalid keep_len={keep_len} for length={length}")
    sink_size = max(int(sink_size), 0)
    recent_window = max(int(recent_window), 0)

    sink_mask = positions < sink_size
    # Under the method invariant, sink positions [0, sink_size) are mandatory
    # retained entries and selector parameters do not change mid-sequence.  The
    # sink count is therefore known from integer metadata; deriving it through
    # ``Tensor.item()/bool()`` would serialize every CUDA layer on the host.
    sink_count = min(sink_size, length)
    if sink_count > keep_len:
        raise ValueError(
            f"mymethod retained budget {keep_len} cannot hold {sink_count} mandatory "
            "sink entries; increase the retained budget or reduce --mymethod_sink_size"
        )
    # Preserve the fully defensive behavior for CPU/unit-test callers that may
    # provide arbitrary non-method position metadata, without adding GPU syncs
    # to the production path.
    if positions.device.type == "cpu":
        sink_counts = sink_mask.sum(dim=1)
        if not bool(torch.all(sink_counts == sink_count)):
            return _select_keep_indices_fallback(
                importance, positions, keep_len, sink_size, recent_window
            )
    selected = sink_mask.clone()
    rec_k = min(recent_window, keep_len - sink_count)
    if rec_k:
        rec_idx = torch.topk(
            positions.float().masked_fill(selected, float("-inf")),
            k=rec_k,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        selected.scatter_(1, rec_idx, True)
    hh_k = keep_len - sink_count - rec_k
    if hh_k:
        hh_idx = torch.topk(
            importance.float().masked_fill(selected, float("-inf")),
            k=hh_k,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        selected.scatter_(1, hh_idx, True)

    # nonzero visits columns in ascending order, preserving physical cache order.
    return selected.nonzero(as_tuple=False)[:, 1].view(importance.shape[0], keep_len)


def select_keep_indices_grouplogitkv(
    importance: torch.Tensor,
    keep_len: int,
) -> torch.Tensor:
    """Use the exact GroupLogitKV Bottom-K eviction split.

    Lowest current post-softmax attention-score entries are evicted globally;
    there is no mandatory sink or recent region. Retained entries are emitted in
    ascending physical-cache order, matching GroupLogitKV.
    """
    if importance.dim() != 2:
        raise ValueError(f"importance must be [X,L], got {tuple(importance.shape)}")
    x, length = importance.shape
    keep_len = int(keep_len)
    if not 1 <= keep_len <= length:
        raise ValueError(f"Invalid keep_len={keep_len} for length={length}")
    num_evict = length - keep_len
    if num_evict == 0:
        return torch.arange(length, device=importance.device).view(1, length).expand(x, -1)
    evict_idx = torch.topk(-importance.float(), k=num_evict, dim=-1).indices
    keep_mask = torch.ones((x, length), dtype=torch.bool, device=importance.device)
    keep_mask.scatter_(1, evict_idx, False)
    return keep_mask.nonzero(as_tuple=False)[:, 1].view(x, keep_len)


def _complement_indices(keep_idx: torch.Tensor, length: int) -> torch.Tensor:
    mask = torch.ones(
        (keep_idx.shape[0], length), dtype=torch.bool, device=keep_idx.device
    )
    mask.scatter_(1, keep_idx, False)
    return mask.nonzero(as_tuple=False)[:, 1].view(
        keep_idx.shape[0], length - keep_idx.shape[1]
    )


def effective_radius(
    counts: torch.Tensor,
    radii: torch.Tensor,
    diameters: torch.Tensor,
) -> torch.Tensor:
    """Tightened valid radius ``min(R, (C-1)/C * D)``."""
    counts_f = counts.float().clamp_min(1.0)
    diameter_bound = ((counts_f - 1.0) / counts_f) * diameters.float()
    return torch.minimum(radii.float(), diameter_bound)


def _count_weighted_centroid_scale(
    keys: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    numerator = (
        counts.float() * keys.float().square().sum(dim=-1)
    ).sum(dim=-1, keepdim=True)
    denominator = counts.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (numerator / denominator).sqrt().add(_SCALE_EPS)


def _nearest_feasible_routes(
    evicted_keys: torch.Tensor,
    retained_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    retained_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    retained_diameter: torch.Tensor,
    delta: torch.Tensor,
    anchor_allowed: torch.Tensor,
    chunk_size: int,
    certify_groups: bool,
    route_mode: str = DEFAULT_ROUTE_MODE,
    diagnostics_sink: Optional[list] = None,
    diagnostics_context: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiled distance/feasibility/routing without pair-wise square roots.

    Feasibility ``Rhat_i + d_ij <= T_j`` is evaluated equivalently as
    ``d_ij^2 <= (T_j-Rhat_i)^2`` with a non-negative margin.  The square root is
    taken only for the selected anchor of each accepted candidate, reducing
    square-root work from ``O(N_e N_r)`` to ``O(N_e)`` without changing the
    mathematical routing rule.
    """
    route_mode = normalize_route_mode(route_mode)
    x, n, _ = evicted_keys.shape
    m = retained_keys.shape[1]
    if n == 0:
        return (
            torch.empty((x, 0), dtype=torch.long, device=evicted_keys.device),
            torch.empty((x, 0), dtype=torch.float32, device=evicted_keys.device),
            torch.empty((x, 0), dtype=torch.bool, device=evicted_keys.device),
        )

    tile_e = n if int(chunk_size) <= 0 else min(int(chunk_size), n)
    retained_f = retained_keys.float()
    # Translation-invariant centering reduces cancellation in the GEMM identity.
    distance_origin = retained_f.mean(dim=1, keepdim=True)
    retained_distance = retained_f - distance_origin
    evicted_distance = evicted_keys.float() - distance_origin
    retained_sq = retained_distance.square().sum(dim=-1)

    if certify_groups:
        retained_threshold = torch.minimum(delta * 0.5, delta - retained_rhat).clamp_min_(0.0)
        retained_cert_ok = (retained_diameter <= delta) & anchor_allowed
    else:
        retained_threshold = None
        retained_cert_ok = anchor_allowed

    all_best_idx = []
    all_best_dist = []
    all_accepted = []
    for e_start in range(0, n, tile_e):
        e_end = min(e_start + tile_e, n)
        ekey = evicted_distance[:, e_start:e_end, :]
        e_sq = ekey.square().sum(dim=-1, keepdim=True)
        dist2 = (
            e_sq
            + retained_sq[:, None, :]
            - 2.0 * torch.bmm(ekey, retained_distance.transpose(1, 2))
        ).clamp_min_(0.0)

        if certify_groups:
            e_rhat = evicted_rhat[:, e_start:e_end]
            e_diam = evicted_diameter[:, e_start:e_end]
            margin = retained_threshold[:, None, :] - e_rhat.unsqueeze(-1)
            feasible = (
                (e_diam.unsqueeze(-1) <= delta.unsqueeze(-1))
                & retained_cert_ok[:, None, :]
                & (margin >= 0.0)
                & (dist2 <= margin.square())
            )
        else:
            feasible = retained_cert_ok[:, None, :].expand_as(dist2)

        if route_mode == "nearest_feasible":
            masked_dist2 = dist2.masked_fill(~feasible, float("inf"))
            best_dist2, best_idx = masked_dist2.min(dim=-1)
            accepted = torch.isfinite(best_dist2)
        else:
            # Ablation: choose the absolute nearest allowed anchor first, then
            # test only that pair. A candidate is discarded even when another
            # slightly farther anchor would satisfy the complete-group certificate.
            allowed_dist2 = dist2.masked_fill(
                ~anchor_allowed[:, None, :], float("inf")
            )
            best_dist2, best_idx = allowed_dist2.min(dim=-1)
            if certify_groups:
                selected_feasible = torch.gather(
                    feasible, 2, best_idx.unsqueeze(-1)
                ).squeeze(-1)
                accepted = torch.isfinite(best_dist2) & selected_feasible
            else:
                accepted = torch.isfinite(best_dist2)
        best_distance = torch.sqrt(torch.where(
            accepted, best_dist2, torch.zeros_like(best_dist2)
        ))
        all_best_idx.append(best_idx)
        all_best_dist.append(best_distance)
        all_accepted.append(accepted)

    return (
        torch.cat(all_best_idx, dim=1),
        torch.cat(all_best_dist, dim=1),
        torch.cat(all_accepted, dim=1),
    )

def build_merge_plan(
    keys: torch.Tensor,
    counts: torch.Tensor,
    radii: torch.Tensor,
    diameters: torch.Tensor,
    importance: torch.Tensor,
    positions: torch.Tensor,
    keep_len: int,
    tau: float = DEFAULT_TAU,
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    certify_groups: bool = True,
    sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    selector: str = DEFAULT_SELECTOR,
    route_mode: str = DEFAULT_ROUTE_MODE,
) -> MergePlan:
    """Build the split and choose the nearest feasible anchor for each evictee."""
    if not 0.0 < float(tau) < 1.0:
        raise ValueError(f"mymethod tau must be in (0,1), got {tau}")
    x, length, _ = keys.shape
    for name, state in {
        "counts": counts,
        "radii": radii,
        "diameters": diameters,
        "importance": importance,
        "positions": positions,
    }.items():
        if tuple(state.shape) != (x, length):
            raise ValueError(f"{name} must be {(x, length)}, got {tuple(state.shape)}")

    selector = normalize_selector(selector)
    route_mode = normalize_route_mode(route_mode)
    if selector == "grouplogitkv":
        keep_idx = select_keep_indices_grouplogitkv(importance, keep_len)
    else:
        keep_idx = select_keep_indices(
            importance, positions, keep_len,
            sink_size=sink_size, recent_window=recent_window,
        )
    evict_idx = _complement_indices(keep_idx, length)
    retained_keys = gather_seq(keys, keep_idx)
    evicted_keys = gather_seq(keys, evict_idx)

    rhat = effective_radius(counts, radii, diameters)
    retained_rhat = gather_seq(rhat, keep_idx)
    evicted_rhat = gather_seq(rhat, evict_idx)
    retained_d = gather_seq(diameters.float(), keep_idx)
    evicted_d = gather_seq(diameters.float(), evict_idx)

    scale = _count_weighted_centroid_scale(keys, counts)
    delta = scale * (1.0 - float(tau))
    retained_positions = gather_seq(positions, keep_idx)
    if selector == "grouplogitkv" or sink_can_absorb:
        anchor_allowed = torch.ones_like(retained_positions, dtype=torch.bool)
    else:
        anchor_allowed = retained_positions >= max(int(sink_size), 0)

    best_idx, assigned_distance, accepted = _nearest_feasible_routes(
        evicted_keys=evicted_keys,
        retained_keys=retained_keys,
        evicted_rhat=evicted_rhat,
        retained_rhat=retained_rhat,
        evicted_diameter=evicted_d,
        retained_diameter=retained_d,
        delta=delta,
        anchor_allowed=anchor_allowed,
        chunk_size=distance_chunk_size,
        certify_groups=certify_groups,
        route_mode=route_mode,
    )
    candidate_cover = torch.where(
        accepted,
        evicted_rhat + assigned_distance,
        torch.zeros_like(assigned_distance),
    )
    return MergePlan(
        keep_idx=keep_idx,
        evict_idx=evict_idx,
        best_idx=best_idx,
        accepted=accepted,
        assigned_distance=assigned_distance,
        candidate_cover=candidate_cover,
        delta=delta,
        scale=scale,
    )


def _segmented_count_sum(index: torch.Tensor, src: torch.Tensor, out_size: int) -> torch.Tensor:
    out = torch.zeros((src.shape[0], out_size), device=src.device, dtype=_COUNT_DTYPE)
    if src.shape[1]:
        out.scatter_add_(1, index, src.to(_COUNT_DTYPE))
    return out


def _require_scatter_reduce() -> None:
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("mymethod requires torch.Tensor.scatter_reduce_ (PyTorch >= 1.12)")


def _segmented_max_2d(
    index: torch.Tensor,
    src: torch.Tensor,
    valid: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    _require_scatter_reduce()
    out = torch.full(
        (src.shape[0], out_size), float("-inf"),
        device=src.device, dtype=torch.float32,
    )
    if src.shape[1]:
        out.scatter_reduce_(
            1,
            index,
            src.float().masked_fill(~valid, float("-inf")),
            reduce="amax",
            include_self=True,
        )
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _segmented_top2_2d(
    index: torch.Tensor,
    src: torch.Tensor,
    valid: torch.Tensor,
    out_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Segmented largest and second-largest values, counting tied entries."""
    _require_scatter_reduce()
    top1 = _segmented_max_2d(index, src, valid, out_size)
    if src.shape[1] == 0:
        return top1, torch.zeros_like(top1)

    gathered_top1 = torch.gather(top1, 1, index)
    is_top = valid & (src.float() == gathered_top1)
    n = src.shape[1]
    candidate_order = torch.arange(n, device=src.device, dtype=torch.long).view(1, n)
    candidate_order = candidate_order.expand(src.shape[0], -1)
    winner = torch.full(
        (src.shape[0], out_size), n,
        device=src.device, dtype=torch.long,
    )
    winner.scatter_reduce_(
        1,
        index,
        candidate_order.masked_fill(~is_top, n),
        reduce="amin",
        include_self=True,
    )
    gathered_winner = torch.gather(winner, 1, index)
    second_valid = valid & (candidate_order != gathered_winner)
    top2 = _segmented_max_2d(index, src, second_valid, out_size)
    return top1, top2


def _segmented_weighted_kv_sum(
    index: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    out_size: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate K and V in one segmented scatter-add launch per chunk."""
    x, n, dim = keys.shape
    if values.shape != keys.shape:
        raise ValueError(f"K/V reduction shape mismatch: {keys.shape} vs {values.shape}")
    out = torch.zeros((x, out_size, dim * 2), device=keys.device, dtype=torch.float32)
    if n == 0:
        return out[..., :dim], out[..., dim:]
    chunk = n if int(chunk_size) <= 0 else min(int(chunk_size), n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = index[:, start:end].unsqueeze(-1).expand(-1, -1, dim * 2)
        vectors = torch.cat(
            (keys[:, start:end].float(), values[:, start:end].float()), dim=-1
        )
        src = weights[:, start:end].unsqueeze(-1) * vectors
        out.scatter_add_(1, idx, src)
    return out[..., :dim], out[..., dim:]

def _segmented_weighted_history_sum(
    index: torch.Tensor,
    weights: torch.Tensor,
    history: torch.Tensor,
    out_size: int,
    chunk_size: int,
) -> torch.Tensor:
    x, window, n = history.shape
    out = torch.zeros((x, window, out_size), device=history.device, dtype=torch.float32)
    if n == 0 or window == 0:
        return out
    chunk = n if int(chunk_size) <= 0 else min(int(chunk_size), n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = index[:, start:end].unsqueeze(1).expand(-1, window, -1)
        src = history[:, :, start:end].float() * weights[:, None, start:end]
        out.scatter_add_(2, idx, src)
    return out


def merge_layer(
    keys: torch.Tensor,
    values: torch.Tensor,
    counts: torch.Tensor,
    radii: torch.Tensor,
    diameters: torch.Tensor,
    score_history: torch.Tensor,
    history_ptr: int,
    positions: torch.Tensor,
    ratio: float,
    tau: float = DEFAULT_TAU,
    target_length: Optional[int] = None,
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    recency_beta: float = DEFAULT_RECENCY_BETA,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    certify_groups: bool = True,
    sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    selector: str = DEFAULT_SELECTOR,
    current_importance: Optional[torch.Tensor] = None,
    route_mode: str = DEFAULT_ROUTE_MODE,
    diagnostics_sink: Optional[list] = None,
    diagnostics_context: Optional[dict] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Compress one layer and propagate all certificate and selector states."""
    bsz, h_kv, length, dim = keys.shape
    if values.shape != keys.shape:
        raise ValueError(f"K/V shape mismatch: {tuple(keys.shape)} vs {tuple(values.shape)}")
    if score_history.shape[:2] != (bsz, h_kv) or score_history.shape[-1] != length:
        raise ValueError(
            f"score_history must be [B,H,W,L] aligned to K, got {tuple(score_history.shape)}"
        )
    selector = normalize_selector(selector)
    route_mode = normalize_route_mode(route_mode)
    expected_window = max(int(recent_window), 0) if selector == "keepkv" else 0
    if int(score_history.shape[-2]) != expected_window:
        raise ValueError(
            f"score history window {score_history.shape[-2]} != expected {expected_window} "
            f"for selector={selector}"
        )

    if target_length is not None:
        keep_len = min(max(int(target_length), 1), length)
    else:
        num_evict = min(max(int(length * float(ratio)), 0), max(length - 1, 0))
        keep_len = length - num_evict

    if length <= 1 or keep_len >= length:
        return (
            keys.contiguous(), values.contiguous(), counts.to(_COUNT_DTYPE).contiguous(),
            radii.float().contiguous(), diameters.float().contiguous(),
            score_history.float().contiguous(), positions.to(_POSITION_DTYPE).contiguous(),
        )

    x = bsz * h_kv
    kf = keys.reshape(x, length, dim)
    vf = values.reshape(x, length, dim)
    cf_int = counts.reshape(x, length).to(_COUNT_DTYPE)
    cf = cf_int.float()
    rf = radii.reshape(x, length).float()
    df = diameters.reshape(x, length).float()
    hf = score_history.reshape(x, score_history.shape[-2], length).float()
    pf = positions.reshape(x, length).to(_POSITION_DTYPE)
    if selector == "grouplogitkv":
        if current_importance is None:
            raise ValueError("current_importance is required for selector=grouplogitkv")
        if tuple(current_importance.shape) != (bsz, h_kv, length):
            raise ValueError(
                f"current_importance must be {(bsz, h_kv, length)}, got "
                f"{tuple(current_importance.shape)}"
            )
        importance = current_importance.reshape(x, length).float()
    else:
        importance = recent_window_importance(hf, history_ptr, recency_beta)

    plan = build_merge_plan(
        keys=kf,
        counts=cf,
        radii=rf,
        diameters=df,
        importance=importance,
        positions=pf,
        keep_len=keep_len,
        tau=tau,
        sink_size=sink_size,
        recent_window=recent_window,
        distance_chunk_size=distance_chunk_size,
        certify_groups=certify_groups,
        sink_can_absorb=sink_can_absorb,
        selector=selector,
        route_mode=route_mode,
    )

    km = gather_seq(kf, plan.keep_idx)
    vm = gather_seq(vf, plan.keep_idx)
    cm_int = gather_seq(cf_int, plan.keep_idx).to(_COUNT_DTYPE)
    cm = cm_int.float()
    rm = gather_seq(rf, plan.keep_idx).float()
    dm = gather_seq(df, plan.keep_idx).float()
    hm = gather_history(hf, plan.keep_idx).float()
    pm = gather_seq(pf, plan.keep_idx).to(_POSITION_DTYPE)

    kn = gather_seq(kf, plan.evict_idx)
    vn = gather_seq(vf, plan.evict_idx)
    cn_int = gather_seq(cf_int, plan.evict_idx).to(_COUNT_DTYPE)
    cn = cn_int.float()
    rn = gather_seq(rf, plan.evict_idx).float()
    dn = gather_seq(df, plan.evict_idx).float()
    hn = gather_history(hf, plan.evict_idx).float()

    rhat_all = effective_radius(cf, rf, df)
    rhat_m = gather_seq(rhat_all, plan.keep_idx)
    rhat_n = gather_seq(rhat_all, plan.evict_idx)

    accepted_f = plan.accepted.float()
    accepted_counts_int = cn_int * plan.accepted.to(_COUNT_DTYPE)
    add_counts_int = _segmented_count_sum(plan.best_idx, accepted_counts_int, keep_len)
    cnew_int = cm_int + add_counts_int
    cnew = cnew_int.float()
    accepted_counts = cn * accepted_f
    denom = cnew.clamp_min(_REDUCTION_EPS).unsqueeze(-1)

    add_k, add_v = _segmented_weighted_kv_sum(
        plan.best_idx, accepted_counts, kn, vn, keep_len, distance_chunk_size
    )
    knew_f = (cm.unsqueeze(-1) * km.float() + add_k) / denom
    knew = knew_f.to(keys.dtype)
    # Certificates are relative to the actual cached key after dtype rounding,
    # not the ideal FP32 centroid numerator.
    knew_cert = knew.float()
    vnew = ((cm.unsqueeze(-1) * vm.float() + add_v) / denom).to(values.dtype)
    del add_k, add_v

    add_h = _segmented_weighted_history_sum(
        plan.best_idx, accepted_counts, hn, keep_len, distance_chunk_size
    )
    hnew = (
        cm.unsqueeze(1) * hm + add_h
    ) / cnew.clamp_min(_REDUCTION_EPS).unsqueeze(1)
    del add_h

    # Complete-diameter certificate: internal anchor/candidate diameters,
    # anchor-candidate extent, and the two largest candidate extents.
    candidate_dmax = _segmented_max_2d(
        plan.best_idx, dn, plan.accepted, keep_len
    )
    e1, e2 = _segmented_top2_2d(
        plan.best_idx, plan.candidate_cover, plan.accepted, keep_len
    )
    dnew = torch.maximum(dm, candidate_dmax)
    dnew = torch.maximum(dnew, rhat_m + e1)
    dnew = torch.maximum(dnew, e1 + e2)

    # Radius certificate around the new centroid.
    rnew_tilde = rhat_m + (km.float() - knew_cert).square().sum(dim=-1).sqrt()
    n = kn.shape[1]
    if n:
        _require_scatter_reduce()
        chunk = n if int(distance_chunk_size) <= 0 else min(int(distance_chunk_size), n)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            idx = plan.best_idx[:, start:end]
            centers = torch.gather(
                knew_cert, 1, idx.unsqueeze(-1).expand(-1, -1, dim)
            )
            candidate_radius = rhat_n[:, start:end] + (kn[:, start:end, :].float() - centers).square().sum(dim=-1).sqrt()
            candidate_radius = candidate_radius.masked_fill(
                ~plan.accepted[:, start:end], float("-inf")
            )
            rnew_tilde.scatter_reduce_(
                1, idx, candidate_radius, reduce="amax", include_self=True
            )
    rnew = torch.minimum(rnew_tilde, dnew)

    if diagnostics_sink is not None:
        accepted_entries = int(plan.accepted.sum().item())
        evicted_entries = int(plan.accepted.numel())
        accepted_mass = float((cn * plan.accepted.float()).sum().item())
        evicted_mass = float(cn.sum().item())
        per_anchor_candidates = torch.zeros(
            (plan.best_idx.shape[0], keep_len), device=keys.device, dtype=torch.int32
        )
        if evicted_entries:
            per_anchor_candidates.scatter_add_(
                1, plan.best_idx, plan.accepted.to(torch.int32)
            )
        nonempty = per_anchor_candidates > 0
        row = dict(diagnostics_context or {})
        row.update({
            "selector": selector,
            "route_mode": route_mode,
            "certify_groups": bool(certify_groups),
            "input_length": int(length),
            "retained_length": int(keep_len),
            "evicted_entries": evicted_entries,
            "accepted_entries": accepted_entries,
            "accepted_entry_ratio": accepted_entries / max(evicted_entries, 1),
            "evicted_count_mass": evicted_mass,
            "accepted_count_mass": accepted_mass,
            "accepted_count_mass_ratio": accepted_mass / max(evicted_mass, 1e-12),
            "rejected_count_mass": evicted_mass - accepted_mass,
            "nonempty_merged_groups": int(nonempty.sum().item()),
            "max_candidates_per_group": int(per_anchor_candidates.max().item()) if per_anchor_candidates.numel() else 0,
            "mean_candidates_per_nonempty_group": (
                float(per_anchor_candidates[nonempty].float().mean().item())
                if bool(nonempty.any().item()) else 0.0
            ),
            "scale_mean": float(plan.scale.mean().item()),
            "delta_mean": float(plan.delta.mean().item()),
            # The round-relative Delta guarantee applies only to anchors that
            # accepted at least one candidate in this round. Unchanged historical
            # entries retain valid R/D certificates but need not satisfy the new Delta.
            "post_diameter_over_delta_max": (
                float((dnew / plan.delta.clamp_min(1e-12))[nonempty].max().item())
                if bool(nonempty.any().item()) else 0.0
            ),
            "post_radius_over_delta_max": (
                float((rnew / plan.delta.clamp_min(1e-12))[nonempty].max().item())
                if bool(nonempty.any().item()) else 0.0
            ),
            "all_entry_diameter_over_delta_max": float(
                (dnew / plan.delta.clamp_min(1e-12)).max().item()
            ),
            "all_entry_radius_over_delta_max": float(
                (rnew / plan.delta.clamp_min(1e-12)).max().item()
            ),
        })
        diagnostics_sink.append(row)

    return (
        knew.reshape(bsz, h_kv, keep_len, dim).contiguous(),
        vnew.reshape(bsz, h_kv, keep_len, dim).contiguous(),
        cnew_int.reshape(bsz, h_kv, keep_len).contiguous(),
        rnew.reshape(bsz, h_kv, keep_len).contiguous(),
        dnew.reshape(bsz, h_kv, keep_len).contiguous(),
        hnew.reshape(bsz, h_kv, score_history.shape[-2], keep_len).contiguous(),
        pm.reshape(bsz, h_kv, keep_len).contiguous(),
    )


def _get_layer_state(model, name: str, layer_idx: int):
    values = getattr(model, name, None)
    if values is None or layer_idx >= len(values):
        return None
    return values[layer_idx]


def initial_or_existing_state(
    model,
    layer_idx: int,
    keys: torch.Tensor,
    score_window: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    int, torch.Tensor, torch.Tensor, Optional[float],
]:
    """Align persistent ``C,R,D,P,p`` state with the incoming physical cache."""
    bsz, h_kv, length, _ = keys.shape
    device = keys.device
    score_window = max(int(score_window), 0)

    count = _get_layer_state(model, "_mymethod_counts", layer_idx)
    radius = _get_layer_state(model, "_mymethod_radii", layer_idx)
    diameter = _get_layer_state(model, "_mymethod_diameters", layer_idx)
    history = _get_layer_state(model, "_mymethod_score_history", layer_idx)
    positions = _get_layer_state(model, "_mymethod_positions", layer_idx)
    ptr = _get_layer_state(model, "_mymethod_score_history_ptrs", layer_idx)
    log_scale = _get_layer_state(model, "_mymethod_score_log_scales", layer_idx)
    next_position = _get_layer_state(model, "_mymethod_next_positions", layer_idx)

    if count is None:
        count = torch.ones((bsz, h_kv, length), device=device, dtype=_COUNT_DTYPE)
        radius = torch.zeros_like(count)
        diameter = torch.zeros_like(count)
        history = torch.zeros(
            (bsz, h_kv, score_window, length), device=device, dtype=torch.float32
        )
        positions = torch.arange(
            length, device=device, dtype=_POSITION_DTYPE
        ).view(1, 1, length).expand(bsz, h_kv, -1).clone()
        next_position = torch.full(
            (bsz, h_kv, 1), length, device=device, dtype=_POSITION_DTYPE
        )
        return count, radius, diameter, history, 0, positions, next_position, None

    scalar_states = {
        "count": count,
        "radius": radius,
        "diameter": diameter,
        "positions": positions,
    }
    for state_name, state in scalar_states.items():
        if state is None or state.dim() != 3 or state.shape[:2] != (bsz, h_kv):
            shape = None if state is None else tuple(state.shape)
            raise ValueError(
                f"Incompatible mymethod {state_name} at layer {layer_idx}: "
                f"state={shape}, K={tuple(keys.shape)}"
            )
    old_len = int(count.shape[-1])
    if any(int(state.shape[-1]) != old_len for state in scalar_states.values()):
        raise ValueError(f"mymethod scalar state lengths disagree at layer {layer_idx}")
    if history is None or history.dim() != 4 or history.shape[:2] != (bsz, h_kv) or history.shape[-1] != old_len:
        shape = None if history is None else tuple(history.shape)
        raise ValueError(
            f"Incompatible mymethod score history at layer {layer_idx}: {shape}"
        )

    count = count.to(device=device, dtype=_COUNT_DTYPE)
    radius = radius.to(device=device, dtype=torch.float32)
    diameter = diameter.to(device=device, dtype=torch.float32)
    history = history.to(device=device, dtype=torch.float32)
    positions = positions.to(device=device, dtype=_POSITION_DTYPE)
    ptr = int(ptr or 0)
    history, ptr = _resize_ring_history(history, ptr, score_window)
    if next_position is None:
        # Compatibility with states produced by the first implementation.
        next_position = positions.amax(dim=-1, keepdim=True) + 1 if old_len else torch.zeros(
            (bsz, h_kv, 1), device=device, dtype=_POSITION_DTYPE
        )
    else:
        next_position = next_position.to(device=device, dtype=_POSITION_DTYPE)
        if next_position.shape != (bsz, h_kv, 1):
            raise ValueError(
                f"Incompatible mymethod next-position state at layer {layer_idx}: "
                f"{tuple(next_position.shape)}"
            )

    if old_len < length:
        add_len = length - old_len
        suffix_shape = (bsz, h_kv, add_len)
        count = torch.cat([
            count, torch.ones(suffix_shape, device=device, dtype=_COUNT_DTYPE)
        ], dim=-1)
        radius = torch.cat([radius, torch.zeros(suffix_shape, device=device)], dim=-1)
        diameter = torch.cat([diameter, torch.zeros(suffix_shape, device=device)], dim=-1)
        history = torch.cat(
            [
                history,
                torch.zeros(
                    (bsz, h_kv, score_window, add_len),
                    device=device,
                    dtype=torch.float32,
                ),
            ],
            dim=-1,
        )
        start = next_position
        suffix = torch.arange(
            add_len, device=device, dtype=_POSITION_DTYPE
        ).view(1, 1, add_len)
        positions = torch.cat([positions, start + suffix], dim=-1)
        next_position = next_position + add_len
    elif old_len > length:
        count = count[..., :length]
        radius = radius[..., :length]
        diameter = diameter[..., :length]
        history = history[..., :length]
        positions = positions[..., :length]

    return (
        count.contiguous(),
        radius.contiguous(),
        diameter.contiguous(),
        history.contiguous(),
        ptr,
        positions.contiguous(),
        next_position.contiguous(),
        None if log_scale is None else float(log_scale),
    )
