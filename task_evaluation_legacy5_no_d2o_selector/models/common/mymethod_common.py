"""Common implementation of graph-MIS ``mymethod`` KV-cache compression.

This module follows
``method_motivation_method_theory_zh_graph_mis_literature_motivation.md``:

* selector-agnostic retained/evicted partition (KeepKV-style or current-attention);
* request-local frozen RMS pair-distance reference scale ``S_ref``;
* normalized complete-diameter budget ``Delta = lambda * S_ref``;
* entry-pair certificates ``Gamma_uv`` and nearest-compatible anchor routing;
* per-anchor evicted-only conflict graphs;
* deterministic degree-priority parallel maximal independent set (MIS);
* count-weighted K/V/history reductions and cross-round ``C,R,D`` certificates.

The core persistent per-entry side state remains ``C`` (INT32), ``R`` (FP32),
and ``D`` (FP32).  ``S_ref`` is one frozen FP32 scalar per request/layer/KV head.
The KeepKV-style selector additionally stores positions and a recent-score ring.
Decode-time ``log(C)`` is computed by architecture runtime patches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    from mymethod_cuda import graph_mis_select as _cuda_graph_mis_select
    from mymethod_cuda import compatible_route as _cuda_compatible_route
    from mymethod_cuda import is_available as _mymethod_cuda_available
except Exception:
    _cuda_graph_mis_select = None
    _cuda_compatible_route = None
    _mymethod_cuda_available = lambda: False


DEFAULT_LAMBDA = 0.5
DEFAULT_SINK_SIZE = 4
DEFAULT_RECENT_WINDOW = 128
DEFAULT_RECENCY_BETA = 1.0
DEFAULT_DISTANCE_CHUNK_SIZE = 256
DEFAULT_SINK_CAN_ABSORB = True
DEFAULT_SELECTOR = "keepkv"
VALID_SELECTORS = ("keepkv", "attention")
DEFAULT_GROUP_MODE = "full"
DEFAULT_GRAPH_BACKEND = "auto"
DEFAULT_ROUTE_BACKEND = "torch"
VALID_GRAPH_BACKENDS = ("auto", "cuda", "torch")
VALID_ROUTE_BACKENDS = ("auto", "cuda", "torch")
VALID_GROUP_MODES = ("full", "anchor_only", "eviction_only")
_SCALE_EPS = 1e-6
_REDUCTION_EPS = 1e-8
_CERT_MARGIN_REL = 8.0 * torch.finfo(torch.float32).eps
_POSITION_DTYPE = torch.int32
_COUNT_DTYPE = torch.int32
_RECENCY_WEIGHT_CACHE = {}


@dataclass
class MergePlan:
    """Retained split, compatible routing, graph-MIS membership, and certificates."""

    keep_idx: torch.Tensor                 # [X, M]
    evict_idx: torch.Tensor                # [X, N]
    best_idx: torch.Tensor                 # [X, N], local retained-anchor index
    route_compatible: torch.Tensor         # [X, N] pair certificate passed
    accepted: torch.Tensor                 # [X, N] selected by MIS / group mode
    assigned_distance: torch.Tensor        # [X, N] FP32; zero when incompatible
    anchor_gamma: torch.Tensor             # [X, N] Gamma_{i,j*}; zero if incompatible
    group_gamma_max: torch.Tensor          # [X, M], exact Step-6 entry-pair bound max
    candidate_pool_sizes: torch.Tensor     # [X, M] INT32 before MIS
    selected_per_anchor: torch.Tensor      # [X, M] INT32 after MIS
    conflict_edges_per_anchor: torch.Tensor # [X, M] INT32 undirected edges
    mis_rounds_per_anchor: torch.Tensor    # [X, M] INT32 (0 for empty pools)
    delta: torch.Tensor                    # [X, 1] frozen lambda*S_ref
    scale: torch.Tensor                    # [X, 1] frozen S_ref


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


def attention_importance_for_kv_heads(
    attention_scores: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Current-attention selector importance for physical KV heads.

    Accepted inputs are either already reduced ``[B,H_kv,K]`` post-softmax
    attention sums captured by the mymethod runtime patch, or full
    ``[B,H_q,Q,K]`` post-softmax attention probabilities. Query heads sharing
    one KV head are averaged and query positions are summed. This selector is
    an internal mymethod option and has no GroupLogitKV dependency.
    """
    if attention_scores is None:
        raise ValueError("Missing current-attention selector scores")
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


def select_keep_indices_attention(
    importance: torch.Tensor,
    keep_len: int,
) -> torch.Tensor:
    """Global current-attention selector.

    Lowest current post-softmax attention-mass entries are evicted globally;
    there is no mandatory sink or recent region. Retained entries are emitted in
    ascending physical-cache order. This is a standalone mymethod selector, not
    an invocation of any comparison method.
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


def reference_scale_from_original_keys(keys: torch.Tensor, eps: float = _SCALE_EPS) -> torch.Tensor:
    """Compute frozen ``S_ref`` from an unmerged first-compression cache.

    This vectorized one-pass computation is algebraically identical to the
    Welford/M2 definition in the method document:

        S_ref = sqrt(2*M2/(n-1) + eps^2).

    At the first compression all physical entries are still original tokens, so
    computing the same statistic once here avoids adding Welford updates to the
    attention hot path.  The result is then frozen for the request.
    """
    if keys.dim() != 4:
        raise ValueError(f"Expected keys [B,H,L,D], got {tuple(keys.shape)}")
    n = int(keys.shape[-2])
    if n < 2:
        return torch.full(
            (*keys.shape[:2], 1), float(eps), device=keys.device, dtype=torch.float32
        )
    k = keys.float()
    mean = k.mean(dim=-2, keepdim=True)
    m2 = (k - mean).square().sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    return torch.sqrt((2.0 * m2 / float(n - 1)).clamp_min(0.0) + float(eps) ** 2)


def normalize_group_mode(group_mode: str) -> str:
    value = str(group_mode).strip().lower()
    if value not in VALID_GROUP_MODES:
        raise ValueError(
            f"Unknown mymethod group mode {group_mode!r}; choose one of {VALID_GROUP_MODES}"
        )
    return value


def normalize_graph_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in VALID_GRAPH_BACKENDS:
        raise ValueError(
            f"Unknown mymethod graph backend {backend!r}; choose one of {VALID_GRAPH_BACKENDS}"
        )
    return value


def normalize_route_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in VALID_ROUTE_BACKENDS:
        raise ValueError(
            f"Unknown mymethod route backend {backend!r}; choose one of {VALID_ROUTE_BACKENDS}"
        )
    return value


def _stable_u32_hash(index: torch.Tensor) -> torch.Tensor:
    """Stateless collision-free affine hash for practical 32-bit token indices."""
    x = index.to(torch.int64) & 0xFFFFFFFF
    return (x * 2654435761 + 1013904223) & 0xFFFFFFFF


def _pair_distance2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched squared Euclidean distances with translation centering."""
    if a.dim() != 3 or b.dim() != 3 or a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Bad pair-distance shapes: {tuple(a.shape)}, {tuple(b.shape)}")
    af = a.float()
    bf = b.float()
    origin = bf.mean(dim=1, keepdim=True) if b.shape[1] else af.mean(dim=1, keepdim=True)
    ac = af - origin
    bc = bf - origin
    a2 = ac.square().sum(dim=-1, keepdim=True)
    b2 = bc.square().sum(dim=-1).unsqueeze(1)
    return (a2 + b2 - 2.0 * torch.bmm(ac, bc.transpose(1, 2))).clamp_min_(0.0)


def _nearest_compatible_routes_torch(
    evicted_keys: torch.Tensor,
    retained_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    retained_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    retained_diameter: torch.Tensor,
    delta: torch.Tensor,
    anchor_allowed: torch.Tensor,
    chunk_size: int,
    pair_certificate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 2: nearest retained anchor whose entry-pair certificate is compatible.

    ``Gamma_ij <= Delta`` is checked without materializing pairwise square roots:
    both old-entry diameters must fit and
    ``d_ij^2 <= (Delta-Rhat_i-Rhat_j)^2`` with non-negative margin.
    """
    x, n, _ = evicted_keys.shape
    m = retained_keys.shape[1]
    if n == 0:
        empty_i = torch.empty((x, 0), dtype=torch.long, device=evicted_keys.device)
        empty_f = torch.empty((x, 0), dtype=torch.float32, device=evicted_keys.device)
        empty_b = torch.empty((x, 0), dtype=torch.bool, device=evicted_keys.device)
        return empty_i, empty_f, empty_f, empty_b
    tile_e = n if int(chunk_size) <= 0 else min(max(int(chunk_size), 1), n)
    retained_f = retained_keys.float()
    origin = retained_f.mean(dim=1, keepdim=True)
    retained_c = retained_f - origin
    retained_sq = retained_c.square().sum(dim=-1)
    retained_ok = anchor_allowed
    if pair_certificate:
        retained_ok = retained_ok & (retained_diameter <= delta)

    best_idx_chunks = []
    best_dist_chunks = []
    best_gamma_chunks = []
    compatible_chunks = []
    for start in range(0, n, tile_e):
        end = min(start + tile_e, n)
        e = evicted_keys[:, start:end, :].float() - origin
        e2 = e.square().sum(dim=-1, keepdim=True)
        dist2 = (
            e2 + retained_sq[:, None, :]
            - 2.0 * torch.bmm(e, retained_c.transpose(1, 2))
        ).clamp_min_(0.0)
        if pair_certificate:
            eri = evicted_rhat[:, start:end]
            edi = evicted_diameter[:, start:end]
            margin = delta.unsqueeze(1) - eri.unsqueeze(-1) - retained_rhat.unsqueeze(1)
            # Conservative inward margin for finite-precision boundary comparisons.
            tol = _CERT_MARGIN_REL * torch.maximum(delta.abs(), torch.ones_like(delta))
            safe_margin = margin - tol.unsqueeze(1)
            feasible = (
                (edi.unsqueeze(-1) <= delta.unsqueeze(1))
                & retained_ok.unsqueeze(1)
                & (safe_margin >= 0.0)
                & (dist2 <= safe_margin.clamp_min(0.0).square())
            )
        else:
            feasible = retained_ok.unsqueeze(1).expand_as(dist2)
        masked = dist2.masked_fill(~feasible, float("inf"))
        best_dist2, best_idx = masked.min(dim=-1)
        compatible = torch.isfinite(best_dist2)
        best_dist = torch.sqrt(torch.where(compatible, best_dist2, torch.zeros_like(best_dist2)))
        rr = torch.gather(retained_rhat, 1, best_idx)
        rd = torch.gather(retained_diameter, 1, best_idx)
        ed = evicted_diameter[:, start:end]
        er = evicted_rhat[:, start:end]
        gamma = torch.maximum(torch.maximum(ed, rd), er + best_dist + rr)
        gamma = torch.where(compatible, gamma, torch.zeros_like(gamma))
        best_idx_chunks.append(best_idx)
        best_dist_chunks.append(best_dist)
        best_gamma_chunks.append(gamma)
        compatible_chunks.append(compatible)
    return (
        torch.cat(best_idx_chunks, dim=1),
        torch.cat(best_dist_chunks, dim=1),
        torch.cat(best_gamma_chunks, dim=1),
        torch.cat(compatible_chunks, dim=1),
    )


def _nearest_compatible_routes_cuda(
    evicted_keys: torch.Tensor,
    retained_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    retained_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    retained_diameter: torch.Tensor,
    delta: torch.Tensor,
    anchor_allowed: torch.Tensor,
):
    if _cuda_compatible_route is None or not _mymethod_cuda_available():
        raise RuntimeError("mymethod CUDA route backend requested but mymethod_cuda is unavailable")
    return tuple(_cuda_compatible_route(
        evicted_keys, retained_keys, evicted_rhat, retained_rhat,
        evicted_diameter, retained_diameter, delta, anchor_allowed
    ))


def _nearest_compatible_routes(
    evicted_keys: torch.Tensor,
    retained_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    retained_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    retained_diameter: torch.Tensor,
    delta: torch.Tensor,
    anchor_allowed: torch.Tensor,
    chunk_size: int,
    pair_certificate: bool = True,
    route_backend: str = DEFAULT_ROUTE_BACKEND,
):
    backend = normalize_route_backend(route_backend)
    use_cuda = (
        pair_certificate
        and backend in {"auto", "cuda"}
        and evicted_keys.is_cuda
        and _cuda_compatible_route is not None
        and _mymethod_cuda_available()
    )
    if use_cuda:
        return _nearest_compatible_routes_cuda(
            evicted_keys, retained_keys, evicted_rhat, retained_rhat,
            evicted_diameter, retained_diameter, delta, anchor_allowed
        )
    if backend == "cuda":
        if not pair_certificate:
            raise RuntimeError("CUDA route backend only implements certified routing")
        raise RuntimeError(
            "--mymethod_route_backend cuda requires an installed mymethod_cuda extension and CUDA tensors"
        )
    return _nearest_compatible_routes_torch(
        evicted_keys, retained_keys, evicted_rhat, retained_rhat,
        evicted_diameter, retained_diameter, delta, anchor_allowed,
        chunk_size, pair_certificate
    )


def _scatter_group_stat(
    group_ids: torch.Tensor,
    values: torch.Tensor,
    total_groups: int,
    reduce: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros((total_groups,), device=values.device, dtype=dtype)
    if values.numel() == 0:
        return out
    if reduce == "sum":
        out.scatter_add_(0, group_ids, values.to(dtype))
    elif reduce == "amax":
        if not hasattr(out, "scatter_reduce_"):
            raise RuntimeError("mymethod requires torch.Tensor.scatter_reduce_")
        out.scatter_reduce_(0, group_ids, values.to(dtype), reduce="amax", include_self=True)
    else:
        raise ValueError(reduce)
    return out


def _parallel_mis_membership_torch(
    evicted_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    evicted_physical_idx: torch.Tensor,
    best_idx: torch.Tensor,
    route_compatible: torch.Tensor,
    anchor_gamma: torch.Tensor,
    keep_len: int,
    delta: torch.Tensor,
    group_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 3: local conflict graphs and deterministic degree-priority parallel MIS.

    Candidate pools are bucketed by ``(row, retained_anchor)``.  A single padded
    batch of local pools is used for batched candidate-candidate GEMM.  The only
    host synchronization is obtaining the largest local pool size for dynamic
    workspace allocation; the graph computation and every MIS round run across
    all non-empty anchors in parallel.
    """
    group_mode = normalize_group_mode(group_mode)
    x, n, dim = evicted_keys.shape
    total_groups = x * keep_len
    device = evicted_keys.device
    accepted = torch.zeros((x, n), dtype=torch.bool, device=device)
    zero_f = torch.zeros((x, keep_len), dtype=torch.float32, device=device)
    zero_i = torch.zeros((x, keep_len), dtype=torch.int32, device=device)
    if n == 0 or group_mode == "eviction_only":
        return accepted, zero_f, zero_i, zero_i.clone(), zero_i.clone(), zero_i.clone()

    row_ids = torch.arange(x, device=device, dtype=torch.long).unsqueeze(1).expand(x, n)
    group_all = row_ids * keep_len + best_idx
    valid_flat = route_compatible.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid_flat.numel() == 0:
        return accepted, zero_f, zero_i, zero_i.clone(), zero_i.clone(), zero_i.clone()
    flat_group = group_all.reshape(-1).index_select(0, valid_flat)
    flat_local = (
        torch.arange(n, device=device, dtype=torch.long).view(1, n).expand(x, n)
        .reshape(-1).index_select(0, valid_flat)
    )
    # Stable deterministic bucket ordering.
    order = torch.argsort(flat_group * (n + 1) + flat_local)
    sorted_group = flat_group.index_select(0, order)
    sorted_flat = valid_flat.index_select(0, order)
    unique_group, counts = torch.unique_consecutive(sorted_group, return_counts=True)
    g = int(unique_group.numel())
    max_pool = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts
    group_slot = torch.repeat_interleave(torch.arange(g, device=device), counts)
    rank = torch.arange(sorted_flat.numel(), device=device) - torch.repeat_interleave(starts, counts)
    packed_flat = torch.full((g, max_pool), -1, device=device, dtype=torch.long)
    packed_flat[group_slot, rank] = sorted_flat
    valid = packed_flat >= 0
    safe_flat = packed_flat.clamp_min(0)

    flat_keys = evicted_keys.reshape(x * n, dim)
    flat_rhat = evicted_rhat.reshape(-1)
    flat_diam = evicted_diameter.reshape(-1)
    flat_gamma = anchor_gamma.reshape(-1)
    flat_phys = evicted_physical_idx.reshape(-1)
    packed_keys = flat_keys.index_select(0, safe_flat.reshape(-1)).reshape(g, max_pool, dim)
    packed_r = flat_rhat.index_select(0, safe_flat.reshape(-1)).reshape(g, max_pool)
    packed_d = flat_diam.index_select(0, safe_flat.reshape(-1)).reshape(g, max_pool)
    packed_anchor_gamma = flat_gamma.index_select(0, safe_flat.reshape(-1)).reshape(g, max_pool)
    packed_phys = flat_phys.index_select(0, safe_flat.reshape(-1)).reshape(g, max_pool)
    group_rows = torch.div(unique_group, keep_len, rounding_mode="floor")
    group_delta = delta.index_select(0, group_rows).squeeze(-1)

    pool_sizes_flat = _scatter_group_stat(
        unique_group, counts.to(torch.int32), total_groups, "sum", torch.int32
    )

    # Candidate-candidate compatibility inside each routed pool.  Anchor-only
    # ablation still computes these bounds so its propagated D certificate stays
    # valid even though it intentionally ignores conflicts for membership.
    dist2 = _pair_distance2(packed_keys, packed_keys)
    margin = group_delta[:, None, None] - packed_r[:, :, None] - packed_r[:, None, :]
    tol = _CERT_MARGIN_REL * torch.maximum(group_delta.abs(), torch.ones_like(group_delta))
    safe_margin = margin - tol[:, None, None]
    pair_valid = valid[:, :, None] & valid[:, None, :]
    pair_compatible = (
        pair_valid
        & (packed_d[:, :, None] <= group_delta[:, None, None])
        & (packed_d[:, None, :] <= group_delta[:, None, None])
        & (safe_margin >= 0.0)
        & (dist2 <= safe_margin.clamp_min(0.0).square())
    )
    conflict = pair_valid & ~pair_compatible
    eye = torch.eye(max_pool, device=device, dtype=torch.bool).unsqueeze(0)
    conflict &= ~eye
    # Numerical symmetry guard.
    conflict = conflict | conflict.transpose(1, 2)

    degree = conflict.sum(dim=-1).to(torch.int32)
    if group_mode == "anchor_only":
        selected = valid.clone()
        selected_round = valid.to(torch.int32)
    else:
        hashv = _stable_u32_hash(packed_phys)
        # j_better[i,j] means candidate j has higher lexicographic priority than i:
        # lower degree, then lower anchor Gamma, then lower stable hash.
        di = degree[:, :, None]
        dj = degree[:, None, :]
        gi = packed_anchor_gamma[:, :, None]
        gj = packed_anchor_gamma[:, None, :]
        hi = hashv[:, :, None]
        hj = hashv[:, None, :]
        j_better = (
            (dj < di)
            | ((dj == di) & (gj < gi))
            | ((dj == di) & (gj == gi) & (hj < hi))
        ) & pair_valid

        active = valid.clone()
        selected = torch.zeros_like(active)
        selected_round = torch.zeros((g, max_pool), dtype=torch.int32, device=device)
        # Fixed max_pool iterations avoid per-round GPU->CPU convergence checks.
        for round_no in range(1, max_pool + 1):
            blocked = (conflict & active[:, None, :] & j_better).any(dim=-1)
            winners = active & ~blocked
            selected |= winners
            selected_round = torch.where(
                winners, torch.full_like(selected_round, round_no), selected_round
            )
            remove = winners | (conflict & winners[:, None, :]).any(dim=-1)
            active &= ~remove

    selected_flat_indices = packed_flat[selected]
    selected_flat_indices = selected_flat_indices[selected_flat_indices >= 0]
    if selected_flat_indices.numel():
        accepted.reshape(-1).index_fill_(0, selected_flat_indices, True)

    # Exact Step-6 bound among old entries.  Square roots are only needed here,
    # after the MIS, and local pool sizes are normally small.
    dist = torch.sqrt(dist2)
    b_pair = packed_r[:, :, None] + dist + packed_r[:, None, :]
    gamma_pair = torch.maximum(
        torch.maximum(packed_d[:, :, None], packed_d[:, None, :]), b_pair
    )
    selected_pairs = selected[:, :, None] & selected[:, None, :] & ~eye
    pair_gamma_max = gamma_pair.masked_fill(~selected_pairs, 0.0).amax(dim=(1, 2))
    anchor_gamma_max = packed_anchor_gamma.masked_fill(~selected, 0.0).amax(dim=1)
    local_group_gamma = torch.maximum(pair_gamma_max, anchor_gamma_max)
    group_gamma_flat = _scatter_group_stat(
        unique_group, local_group_gamma, total_groups, "amax", torch.float32
    )
    selected_counts = selected.sum(dim=1).to(torch.int32)
    selected_flat_group = _scatter_group_stat(
        unique_group, selected_counts, total_groups, "sum", torch.int32
    )
    conflict_edges = torch.triu(conflict, diagonal=1).sum(dim=(1, 2)).to(torch.int32)
    conflict_flat = _scatter_group_stat(
        unique_group, conflict_edges, total_groups, "sum", torch.int32
    )
    rounds = selected_round.amax(dim=1)
    rounds_flat = _scatter_group_stat(
        unique_group, rounds, total_groups, "amax", torch.int32
    )
    return (
        accepted,
        group_gamma_flat.view(x, keep_len),
        pool_sizes_flat.view(x, keep_len),
        selected_flat_group.view(x, keep_len),
        conflict_flat.view(x, keep_len),
        rounds_flat.view(x, keep_len),
    )


def _parallel_mis_membership_cuda(
    evicted_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    evicted_physical_idx: torch.Tensor,
    best_idx: torch.Tensor,
    route_compatible: torch.Tensor,
    anchor_gamma: torch.Tensor,
    keep_len: int,
    delta: torch.Tensor,
    group_mode: str,
):
    if _cuda_graph_mis_select is None or not _mymethod_cuda_available():
        raise RuntimeError(
            "mymethod CUDA graph backend requested but mymethod_cuda is unavailable"
        )
    return tuple(_cuda_graph_mis_select(
        evicted_keys, evicted_rhat, evicted_diameter, evicted_physical_idx,
        best_idx, route_compatible, anchor_gamma, int(keep_len), delta, group_mode
    ))


def _parallel_mis_membership(
    evicted_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    evicted_physical_idx: torch.Tensor,
    best_idx: torch.Tensor,
    route_compatible: torch.Tensor,
    anchor_gamma: torch.Tensor,
    keep_len: int,
    delta: torch.Tensor,
    group_mode: str,
    graph_backend: str = DEFAULT_GRAPH_BACKEND,
):
    """Dispatch exact Graph-MIS membership to CUDA or PyTorch reference.

    ``auto`` uses the independent ``mymethod_cuda`` extension only for CUDA
    tensors; otherwise it falls back to the reference implementation. ``cuda``
    is strict and raises if the extension is not installed.
    """
    backend = normalize_graph_backend(graph_backend)
    if group_mode == "eviction_only":
        return _parallel_mis_membership_torch(
            evicted_keys, evicted_rhat, evicted_diameter, evicted_physical_idx,
            best_idx, route_compatible, anchor_gamma, keep_len, delta, group_mode
        )
    use_cuda = (
        backend in {"auto", "cuda"}
        and evicted_keys.is_cuda
        and _cuda_graph_mis_select is not None
        and _mymethod_cuda_available()
    )
    if use_cuda:
        return _parallel_mis_membership_cuda(
            evicted_keys, evicted_rhat, evicted_diameter, evicted_physical_idx,
            best_idx, route_compatible, anchor_gamma, keep_len, delta, group_mode
        )
    if backend == "cuda":
        raise RuntimeError(
            "--mymethod_graph_backend cuda requires an installed mymethod_cuda extension and CUDA tensors"
        )
    return _parallel_mis_membership_torch(
        evicted_keys, evicted_rhat, evicted_diameter, evicted_physical_idx,
        best_idx, route_compatible, anchor_gamma, keep_len, delta, group_mode
    )


def build_merge_plan(
    keys: torch.Tensor,
    counts: torch.Tensor,
    radii: torch.Tensor,
    diameters: torch.Tensor,
    importance: torch.Tensor,
    positions: torch.Tensor,
    keep_len: int,
    reference_scale: torch.Tensor,
    lambda_mymethod: float = DEFAULT_LAMBDA,
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    selector: str = DEFAULT_SELECTOR,
    group_mode: str = DEFAULT_GROUP_MODE,
    graph_backend: str = DEFAULT_GRAPH_BACKEND,
    route_backend: str = DEFAULT_ROUTE_BACKEND,
) -> MergePlan:
    """Build retained split, compatible routes, local conflict graphs, and MIS."""
    lam = float(lambda_mymethod)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"mymethod lambda must be finite and >0, got {lambda_mymethod}")
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
    if reference_scale.dim() == 3:
        reference_scale = reference_scale.reshape(x, 1)
    if tuple(reference_scale.shape) != (x, 1):
        raise ValueError(f"reference_scale must be {(x,1)}, got {tuple(reference_scale.shape)}")
    selector = normalize_selector(selector)
    group_mode = normalize_group_mode(group_mode)
    graph_backend = normalize_graph_backend(graph_backend)
    route_backend = normalize_route_backend(route_backend)
    if selector == "attention":
        keep_idx = select_keep_indices_attention(importance, keep_len)
    else:
        keep_idx = select_keep_indices(
            importance, positions, keep_len,
            sink_size=sink_size, recent_window=recent_window,
        )
    evict_idx = _complement_indices(keep_idx, length)
    retained_keys = gather_seq(keys, keep_idx)
    evicted_keys = gather_seq(keys, evict_idx)
    rhat = effective_radius(counts, radii, diameters)
    rr = gather_seq(rhat, keep_idx)
    er = gather_seq(rhat, evict_idx)
    rd = gather_seq(diameters.float(), keep_idx)
    ed = gather_seq(diameters.float(), evict_idx)
    scale = reference_scale.float().clamp_min(_SCALE_EPS)
    delta = scale * lam
    retained_positions = gather_seq(positions, keep_idx)
    if group_mode == "eviction_only":
        n = evict_idx.shape[1]
        m = keep_idx.shape[1]
        best_idx = torch.zeros((x, n), dtype=torch.long, device=keys.device)
        assigned_distance = torch.zeros((x, n), dtype=torch.float32, device=keys.device)
        anchor_gamma = torch.zeros_like(assigned_distance)
        route_compatible = torch.zeros((x, n), dtype=torch.bool, device=keys.device)
        zeros_i = torch.zeros((x, m), dtype=torch.int32, device=keys.device)
        return MergePlan(
            keep_idx=keep_idx, evict_idx=evict_idx, best_idx=best_idx,
            route_compatible=route_compatible, accepted=route_compatible.clone(),
            assigned_distance=assigned_distance, anchor_gamma=anchor_gamma,
            group_gamma_max=torch.zeros((x,m),dtype=torch.float32,device=keys.device),
            candidate_pool_sizes=zeros_i, selected_per_anchor=zeros_i.clone(),
            conflict_edges_per_anchor=zeros_i.clone(), mis_rounds_per_anchor=zeros_i.clone(),
            delta=delta, scale=scale,
        )
    if selector == "attention" or sink_can_absorb:
        anchor_allowed = torch.ones_like(retained_positions, dtype=torch.bool)
    else:
        anchor_allowed = retained_positions >= max(int(sink_size), 0)
    best_idx, assigned_distance, anchor_gamma, route_compatible = _nearest_compatible_routes(
        evicted_keys, retained_keys, er, rr, ed, rd, delta,
        anchor_allowed, distance_chunk_size, pair_certificate=True,
        route_backend=route_backend,
    )
    accepted, group_gamma_max, pool_sizes, selected_counts, conflict_edges, mis_rounds = (
        _parallel_mis_membership(
            evicted_keys=evicted_keys,
            evicted_rhat=er,
            evicted_diameter=ed,
            evicted_physical_idx=evict_idx,
            best_idx=best_idx,
            route_compatible=route_compatible,
            anchor_gamma=anchor_gamma,
            keep_len=keep_len,
            delta=delta,
            group_mode=group_mode,
            graph_backend=graph_backend,
        )
    )
    return MergePlan(
        keep_idx=keep_idx,
        evict_idx=evict_idx,
        best_idx=best_idx,
        route_compatible=route_compatible,
        accepted=accepted,
        assigned_distance=assigned_distance,
        anchor_gamma=anchor_gamma,
        group_gamma_max=group_gamma_max,
        candidate_pool_sizes=pool_sizes,
        selected_per_anchor=selected_counts,
        conflict_edges_per_anchor=conflict_edges,
        mis_rounds_per_anchor=mis_rounds,
        delta=delta,
        scale=scale,
    )


def _segmented_count_sum(index: torch.Tensor, src: torch.Tensor, out_size: int) -> torch.Tensor:
    out = torch.zeros((src.shape[0], out_size), device=src.device, dtype=_COUNT_DTYPE)
    if src.shape[1]:
        out.scatter_add_(1, index, src.to(_COUNT_DTYPE))
    return out


def _segmented_weighted_kv_sum(
    index: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    out_size: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, n, d = keys.shape
    out = torch.zeros((x, out_size, 2 * d), device=keys.device, dtype=torch.float32)
    if n == 0:
        return out[..., :d], out[..., d:]
    chunk = n if int(chunk_size) <= 0 else min(max(int(chunk_size), 1), n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = index[:, start:end]
        w = weights[:, start:end].float().unsqueeze(-1)
        src = torch.cat((keys[:, start:end].float(), values[:, start:end].float()), dim=-1) * w
        out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, 2 * d), src)
    return out[..., :d], out[..., d:]


def _segmented_weighted_history_sum(
    index: torch.Tensor,
    weights: torch.Tensor,
    history: torch.Tensor,
    out_size: int,
    chunk_size: int,
) -> torch.Tensor:
    x, w, n = history.shape
    out = torch.zeros((x, w, out_size), device=history.device, dtype=torch.float32)
    if n == 0 or w == 0:
        return out
    chunk = n if int(chunk_size) <= 0 else min(max(int(chunk_size), 1), n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = index[:, start:end]
        src = history[:, :, start:end].float() * weights[:, start:end].float().unsqueeze(1)
        out.scatter_add_(2, idx.unsqueeze(1).expand(-1, w, -1), src)
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
    reference_scale: torch.Tensor,
    lambda_mymethod: float = DEFAULT_LAMBDA,
    target_length: Optional[int] = None,
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    recency_beta: float = DEFAULT_RECENCY_BETA,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    selector: str = DEFAULT_SELECTOR,
    current_importance: Optional[torch.Tensor] = None,
    group_mode: str = DEFAULT_GROUP_MODE,
    graph_backend: str = DEFAULT_GRAPH_BACKEND,
    route_backend: str = DEFAULT_ROUTE_BACKEND,
    diagnostics_sink: Optional[list] = None,
    diagnostics_context: Optional[dict] = None,
):
    """Compress one physical layer and propagate all method/selector state."""
    bsz, h_kv, length, dim = keys.shape
    if values.shape != keys.shape:
        raise ValueError(f"K/V shape mismatch: {tuple(keys.shape)} vs {tuple(values.shape)}")
    selector = normalize_selector(selector)
    group_mode = normalize_group_mode(group_mode)
    expected_window = max(int(recent_window), 0) if selector == "keepkv" else 0
    if score_history.shape[:2] != (bsz, h_kv) or score_history.shape[-1] != length:
        raise ValueError(f"score_history must align to K, got {tuple(score_history.shape)}")
    if int(score_history.shape[-2]) != expected_window:
        raise ValueError(
            f"score history window {score_history.shape[-2]} != expected {expected_window}"
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
    if selector == "attention":
        if current_importance is None:
            raise ValueError("current_importance is required for selector=attention")
        importance = current_importance.reshape(x, length).float()
    else:
        importance = recent_window_importance(hf, history_ptr, recency_beta)

    plan = build_merge_plan(
        keys=kf, counts=cf, radii=rf, diameters=df,
        importance=importance, positions=pf, keep_len=keep_len,
        reference_scale=reference_scale.reshape(x, 1),
        lambda_mymethod=lambda_mymethod,
        sink_size=sink_size, recent_window=recent_window,
        distance_chunk_size=distance_chunk_size,
        sink_can_absorb=sink_can_absorb, selector=selector, group_mode=group_mode,
        graph_backend=graph_backend,
        route_backend=route_backend,
    )

    km = gather_seq(kf, plan.keep_idx)
    vm = gather_seq(vf, plan.keep_idx)
    cm_int = gather_seq(cf_int, plan.keep_idx).to(_COUNT_DTYPE)
    cm = cm_int.float()
    dm = gather_seq(df, plan.keep_idx).float()
    hm = gather_history(hf, plan.keep_idx).float()
    pm = gather_seq(pf, plan.keep_idx).to(_POSITION_DTYPE)
    kn = gather_seq(kf, plan.evict_idx)
    vn = gather_seq(vf, plan.evict_idx)
    cn_int = gather_seq(cf_int, plan.evict_idx).to(_COUNT_DTYPE)
    cn = cn_int.float()
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
    knew_cert = knew.float()
    vnew = ((cm.unsqueeze(-1) * vm.float() + add_v) / denom).to(values.dtype)
    del add_k, add_v
    add_h = _segmented_weighted_history_sum(
        plan.best_idx, accepted_counts, hn, keep_len, distance_chunk_size
    )
    hnew = (cm.unsqueeze(1) * hm + add_h) / cnew.clamp_min(_REDUCTION_EPS).unsqueeze(1)
    del add_h

    # Step 6 D certificate: group_gamma_max is the maximum Gamma over every
    # anchor-candidate and selected candidate-candidate old-entry pair.
    dnew = torch.maximum(dm, plan.group_gamma_max)
    # Step 6 R certificate around the actual dtype-rounded cached centroid.
    rnew_tilde = rhat_m + (km.float() - knew_cert).square().sum(dim=-1).sqrt()
    n = kn.shape[1]
    if n:
        if not hasattr(rnew_tilde, "scatter_reduce_"):
            raise RuntimeError("mymethod requires torch.Tensor.scatter_reduce_")
        chunk = n if int(distance_chunk_size) <= 0 else min(max(int(distance_chunk_size), 1), n)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            idx = plan.best_idx[:, start:end]
            centers = torch.gather(knew_cert, 1, idx.unsqueeze(-1).expand(-1, -1, dim))
            candidate_radius = rhat_n[:, start:end] + (
                kn[:, start:end].float() - centers
            ).square().sum(dim=-1).sqrt()
            candidate_radius = candidate_radius.masked_fill(
                ~plan.accepted[:, start:end], float("-inf")
            )
            rnew_tilde.scatter_reduce_(1, idx, candidate_radius, reduce="amax", include_self=True)
    rnew = torch.minimum(rnew_tilde, dnew)

    if diagnostics_sink is not None:
        # Diagnostics intentionally synchronize; never enable for timing runs.
        accepted_entries = int(plan.accepted.sum().item())
        routed_entries = int(plan.route_compatible.sum().item())
        evicted_entries = int(plan.accepted.numel())
        accepted_mass = float((cn * plan.accepted.float()).sum().item())
        routed_mass = float((cn * plan.route_compatible.float()).sum().item())
        evicted_mass = float(cn.sum().item())
        nonempty = plan.selected_per_anchor > 0
        pools = plan.candidate_pool_sizes
        row = dict(diagnostics_context or {})
        row.update({
            "selector": selector,
            "group_mode": group_mode,
            "graph_backend": normalize_graph_backend(graph_backend),
            "route_backend": normalize_route_backend(route_backend),
            "lambda_mymethod": float(lambda_mymethod),
            "input_length": int(length),
            "retained_length": int(keep_len),
            "evicted_entries": evicted_entries,
            "routed_compatible_entries": routed_entries,
            "accepted_entries": accepted_entries,
            "route_compatible_ratio": routed_entries / max(evicted_entries, 1),
            "accepted_entry_ratio": accepted_entries / max(evicted_entries, 1),
            "mis_accept_given_routed_ratio": accepted_entries / max(routed_entries, 1),
            "evicted_count_mass": evicted_mass,
            "routed_count_mass": routed_mass,
            "accepted_count_mass": accepted_mass,
            "accepted_count_mass_ratio": accepted_mass / max(evicted_mass, 1e-12),
            "rejected_count_mass": evicted_mass - accepted_mass,
            "reference_scale_mean": float(plan.scale.mean().item()),
            "delta_mean": float(plan.delta.mean().item()),
            "nonempty_candidate_pools": int((pools > 0).sum().item()),
            "nonempty_merged_groups": int(nonempty.sum().item()),
            "max_candidate_pool_size": int(pools.max().item()) if pools.numel() else 0,
            "max_selected_candidates_per_group": int(plan.selected_per_anchor.max().item()) if plan.selected_per_anchor.numel() else 0,
            "total_conflict_edges": int(plan.conflict_edges_per_anchor.sum().item()),
            "max_mis_rounds": int(plan.mis_rounds_per_anchor.max().item()) if plan.mis_rounds_per_anchor.numel() else 0,
            "post_diameter_over_delta_max": (
                float((dnew / plan.delta.clamp_min(1e-12))[nonempty].max().item())
                if bool(nonempty.any().item()) else 0.0
            ),
            "post_radius_over_delta_max": (
                float((rnew / plan.delta.clamp_min(1e-12))[nonempty].max().item())
                if bool(nonempty.any().item()) else 0.0
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


def reference_scale_or_initialize(model, layer_idx: int, keys: torch.Tensor) -> torch.Tensor:
    """Return the frozen per-request/layer/KV-head ``S_ref``."""
    scale = _get_layer_state(model, "_mymethod_reference_scales", layer_idx)
    if scale is None:
        return reference_scale_from_original_keys(keys).contiguous()
    scale = scale.to(device=keys.device, dtype=torch.float32)
    expected = (*keys.shape[:2], 1)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"Incompatible mymethod reference scale at layer {layer_idx}: "
            f"state={tuple(scale.shape)}, expected={expected}"
        )
    return scale.contiguous()


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
        radius = torch.zeros((bsz, h_kv, length), device=device, dtype=torch.float32)
        diameter = torch.zeros((bsz, h_kv, length), device=device, dtype=torch.float32)
        history = torch.zeros((bsz, h_kv, score_window, length), device=device, dtype=torch.float32)
        positions = torch.arange(length, device=device, dtype=_POSITION_DTYPE).view(1,1,length).expand(bsz,h_kv,-1).clone()
        next_position = torch.full((bsz,h_kv,1), length, device=device, dtype=_POSITION_DTYPE)
        return count, radius, diameter, history, 0, positions, next_position, None
    scalar_states = {"count":count,"radius":radius,"diameter":diameter,"positions":positions}
    for state_name,state in scalar_states.items():
        if state is None or state.dim()!=3 or state.shape[:2]!=(bsz,h_kv):
            raise ValueError(f"Incompatible mymethod {state_name} at layer {layer_idx}")
    old_len=int(count.shape[-1])
    if any(int(state.shape[-1])!=old_len for state in scalar_states.values()):
        raise ValueError(f"mymethod scalar state lengths disagree at layer {layer_idx}")
    if history is None or history.dim()!=4 or history.shape[:2]!=(bsz,h_kv) or history.shape[-1]!=old_len:
        raise ValueError(f"Incompatible mymethod score history at layer {layer_idx}")
    count=count.to(device=device,dtype=_COUNT_DTYPE)
    radius=radius.to(device=device,dtype=torch.float32)
    diameter=diameter.to(device=device,dtype=torch.float32)
    history=history.to(device=device,dtype=torch.float32)
    positions=positions.to(device=device,dtype=_POSITION_DTYPE)
    ptr=int(ptr or 0)
    history,ptr=_resize_ring_history(history,ptr,score_window)
    if next_position is None:
        next_position=positions.amax(dim=-1,keepdim=True)+1 if old_len else torch.zeros((bsz,h_kv,1),device=device,dtype=_POSITION_DTYPE)
    else:
        next_position=next_position.to(device=device,dtype=_POSITION_DTYPE)
    if old_len < length:
        add_len=length-old_len
        suffix=(bsz,h_kv,add_len)
        count=torch.cat([count,torch.ones(suffix,device=device,dtype=_COUNT_DTYPE)],dim=-1)
        radius=torch.cat([radius,torch.zeros(suffix,device=device,dtype=torch.float32)],dim=-1)
        diameter=torch.cat([diameter,torch.zeros(suffix,device=device,dtype=torch.float32)],dim=-1)
        history=torch.cat([history,torch.zeros((bsz,h_kv,score_window,add_len),device=device,dtype=torch.float32)],dim=-1)
        offsets=torch.arange(add_len,device=device,dtype=_POSITION_DTYPE).view(1,1,add_len)
        positions=torch.cat([positions,next_position+offsets],dim=-1)
        next_position=next_position+add_len
    elif old_len > length:
        count=count[...,:length]; radius=radius[...,:length]; diameter=diameter[...,:length]
        history=history[...,:length]; positions=positions[...,:length]
    return (
        count.contiguous(), radius.contiguous(), diameter.contiguous(), history.contiguous(),
        ptr, positions.contiguous(), next_position.contiguous(),
        None if log_scale is None else float(log_scale),
    )
