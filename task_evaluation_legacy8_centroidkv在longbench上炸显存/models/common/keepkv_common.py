"""KeepKV core adapted to an external DynamicCache repair interface.

The selection, score history, merge weights, similarity gating, electoral votes,
and pyramid allocation follow the official KeepKV implementation.  The only
added policy is a global physical-cache budget specified by ``compress_ratio``.

Tensor convention: K/V [B, H_kv, S, D], score history [B, H_kv, R, S].
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class KeepKVConfig:
    # Official defaults from run_helm.py.
    epsilon_times: float = 1.0
    similarity_threshold: float = 0.8
    prefill_similarity_threshold: float = 0.8
    pyramid_recent_ratio: float = 0.2
    # Optional official pyramid layer-wise allocation. When disabled, all layers
    # use the same heavy/recent ratios while the global physical budget remains exact.
    use_pyramid_budget: bool = False
    pyramid_decay_strategy: str = "cosine"
    pyramid_decay_ratio: float = 0.8
    distance_weight: float = 1.2
    sink_len: int = 4
    # Fair-comparison switch.  Official KeepKV protects the first sink_len
    # tokens; the unified evaluator defaults to allowing them to be selected,
    # discarded, and merged like other historical tokens.
    protect_sink: bool = False
    # Unified evaluation length L. Must be supplied by the evaluator.
    logical_length: Optional[int] = None

    @property
    def epsilon(self) -> float:
        return float(torch.finfo(torch.float16).tiny) * float(self.epsilon_times)


@dataclass(frozen=True)
class LayerBudget:
    layer_idx: int
    total: int
    sink: int
    heavy: int  # Heavy hitters excluding sink.
    recent: int
    raw_weight: float

    @property
    def official_heavy_budget(self) -> int:
        """Official variable includes sink slots."""
        return self.sink + self.heavy


@dataclass
class KeepKVLayerState:
    previous_recent_scores: Optional[torch.Tensor] = None
    cluster_cnt: Optional[torch.Tensor] = None
    last_keep_idx: Optional[torch.Tensor] = None
    last_rejected_vote_count: Optional[torch.Tensor] = None
    initialized: bool = False
    # previous_recent_scores stores exp(logits - score_log_scale).  The scalar
    # is shared across every retained recent-query row in this layer.
    score_log_scale: Optional[float] = None


def _get_layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError("Expected model.model.layers (or model.layers).")
    return layers


def _ensure_state_list(model, n_layers: int):
    states = getattr(model, "_keepkv_layer_states", None)
    if not isinstance(states, list) or len(states) != n_layers:
        states = [KeepKVLayerState() for _ in range(n_layers)]
        model._keepkv_layer_states = states
    return states


def reset_keepkv_state(model) -> None:
    layers = _get_layers(model)
    model._keepkv_layer_states = [KeepKVLayerState() for _ in layers]
    model._keepkv_layer_budgets = None
    model._keepkv_counts = [None] * len(layers)
    for layer in layers:
        attn = layer.self_attn
        attn._keepkv_enabled = True
        attn._keepkv_counts = None
        attn._keepkv_masked_exp_scores = None
        attn._keepkv_score_log_scale = None
        attn._keepkv_capture_disabled = False
        attn._keepkv_summary_mode = False
        attn._keepkv_summary_sum = None
        attn._keepkv_summary_weighted_sum = None
        attn._keepkv_summary_log_scale = None


def enable_keepkv_runtime(model) -> None:
    layers = _get_layers(model)
    states = _ensure_state_list(model, len(layers))
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        attn._keepkv_enabled = True
        attn._keepkv_counts = states[idx].cluster_cnt


def _raw_pyramid_components(n_layers: int, cfg: KeepKVConfig):
    """Return official heavy_ratio_layers and recent_ratio_layers.

    When ``use_pyramid_budget`` is false, the layer decay is disabled but the
    official heavy/recent base split is retained for every layer.
    """
    recent = []
    heavy = []
    for idx in range(n_layers):
        recent_component = float(cfg.pyramid_recent_ratio)
        if not cfg.use_pyramid_budget:
            schedule = 1.0
        elif cfg.pyramid_decay_strategy == "linear":
            schedule = (1.0 - cfg.pyramid_decay_ratio) * (idx / n_layers) + cfg.pyramid_decay_ratio
        elif cfg.pyramid_decay_strategy == "cosine":
            schedule = (
                (1.0 - cfg.pyramid_decay_ratio)
                * (math.cos(math.pi * idx / n_layers) + 1.0)
                / 2.0
                + cfg.pyramid_decay_ratio
            )
        else:
            schedule = (cfg.pyramid_decay_ratio + 1.0) / 2.0
        heavy_component = schedule * (1.0 - cfg.pyramid_recent_ratio)
        recent.append(recent_component)
        heavy.append(heavy_component)
    return heavy, recent


def _largest_remainder_capped(values, target_sum: int, capacities, minimums=None):
    """Largest-remainder allocation with exact sum and per-layer capacities."""
    capacities = [int(x) for x in capacities]
    minimums = list(minimums or [0] * len(values))
    minimums = [min(int(m), capacities[i]) for i, m in enumerate(minimums)]
    if target_sum < sum(minimums):
        minimums = [0] * len(values)
    if target_sum > sum(capacities):
        raise ValueError("Global KeepKV target exceeds available physical KV slots.")

    alloc = [min(capacities[i], max(minimums[i], int(math.floor(values[i])))) for i in range(len(values))]
    current = sum(alloc)

    if current > target_sum:
        order = sorted(
            range(len(values)),
            key=lambda i: (values[i] - math.floor(values[i]), values[i], i),
        )
        while current > target_sum:
            changed = False
            for i in order:
                if alloc[i] > minimums[i]:
                    alloc[i] -= 1
                    current -= 1
                    changed = True
                    if current == target_sum:
                        break
            if not changed:
                raise RuntimeError("Could not reduce layer budgets to the exact global target.")
    elif current < target_sum:
        order = sorted(
            range(len(values)),
            key=lambda i: (values[i] - math.floor(values[i]), values[i], -i),
            reverse=True,
        )
        while current < target_sum:
            changed = False
            for i in order:
                if alloc[i] < capacities[i]:
                    alloc[i] += 1
                    current += 1
                    changed = True
                    if current == target_sum:
                        break
            if not changed:
                raise RuntimeError("Could not expand layer budgets to the exact global target.")
    return alloc


def build_pyramid_budgets(
    layer_lengths: Sequence[int],
    logical_length: int,
    compress_ratio: float,
    cfg: KeepKVConfig,
):
    """Allocate the exact global physical budget across layers.

    Let N_l be the current physical KV length and let the official ratios be
    heavy_ratio_l and recent_ratio_l.  The raw layer weight is

        w_l = N_l * (heavy_ratio_l + recent_ratio_l).

    Budgets are B_l = B_global * w_l / sum_j(w_j), rounded with the
    largest-remainder method.  The nominal global target is the unified
    logical-length budget ``n_layers * logical_length * (1-compress_ratio)``;
    before enough tokens exist it is capped by the available physical slots.
    """
    layer_lengths = [int(n) for n in layer_lengths]
    n_layers = len(layer_lengths)
    if n_layers <= 0:
        return []
    if not (0.0 <= compress_ratio < 1.0):
        raise ValueError("compress_ratio must satisfy 0 <= ratio < 1.")
    if logical_length <= 0:
        raise ValueError("logical_length must be positive.")
    if any(n < 0 for n in layer_lengths):
        raise ValueError("Layer KV lengths must be non-negative.")

    nominal_global = int(round(n_layers * logical_length * (1.0 - compress_ratio)))
    target_total = min(sum(layer_lengths), nominal_global)

    raw_heavy, raw_recent = _raw_pyramid_components(n_layers, cfg)
    raw_weights = [
        layer_lengths[i] * (raw_heavy[i] + raw_recent[i])
        for i in range(n_layers)
    ]
    weight_sum = sum(raw_weights)
    if weight_sum <= 0:
        real_layer_budgets = [0.0] * n_layers
    else:
        real_layer_budgets = [target_total * w / weight_sum for w in raw_weights]

    minimums = [min(2, n) for n in layer_lengths]
    totals = _largest_remainder_capped(
        real_layer_budgets,
        target_total,
        capacities=layer_lengths,
        minimums=minimums,
    )

    budgets = []
    for idx, total in enumerate(totals):
        ratio_sum = raw_heavy[idx] + raw_recent[idx]
        if total <= 0 or ratio_sum <= 0:
            official_heavy = 0
            recent = 0
        else:
            official_heavy = int(math.floor(total * raw_heavy[idx] / ratio_sum + 0.5))
            official_heavy = max(0, min(official_heavy, total))
            recent = total - official_heavy
            # Keep the official split formula, with only the minimum validity
            # clamp needed by KeepKV's historical/recent update.
            if total >= 2:
                if recent == 0:
                    recent, official_heavy = 1, total - 1
                elif official_heavy == 0:
                    official_heavy, recent = 1, total - 1

        sink = (
            min(max(0, int(cfg.sink_len)), official_heavy)
            if bool(cfg.protect_sink)
            else 0
        )
        heavy = official_heavy - sink
        budgets.append(
            LayerBudget(
                layer_idx=idx,
                total=total,
                sink=sink,
                heavy=heavy,
                recent=recent,
                raw_weight=raw_weights[idx],
            )
        )

    assert sum(b.total for b in budgets) == target_total
    assert all(b.sink + b.heavy + b.recent == b.total for b in budgets)
    return budgets


def _get_or_build_budgets(model, past_key_values, compress_ratio: float, cfg: KeepKVConfig):
    layers = _get_layers(model)
    logical_length = cfg.logical_length
    if logical_length is None:
        logical_length = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0))
    layer_lengths = tuple(int(kv[0].shape[-2]) for kv in past_key_values)
    key = (
        len(layers), int(logical_length), layer_lengths, float(compress_ratio),
        cfg.pyramid_recent_ratio, bool(cfg.use_pyramid_budget),
        cfg.pyramid_decay_strategy, cfg.pyramid_decay_ratio,
        cfg.sink_len, bool(cfg.protect_sink),
    )
    cached = getattr(model, "_keepkv_layer_budgets", None)
    if not isinstance(cached, dict) or cached.get("key") != key:
        cached = {
            "key": key,
            "budgets": build_pyramid_budgets(
                layer_lengths, int(logical_length), compress_ratio, cfg
            ),
        }
        model._keepkv_layer_budgets = cached
    return cached["budgets"]

def _aggregate_query_heads(scores: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """GQA adaptation: average query heads belonging to the same KV head."""
    if scores.ndim != 4:
        raise ValueError(f"Expected [B,Hq,Q,K], got {tuple(scores.shape)}")
    bsz, hq, q_len, k_len = scores.shape
    if hq % num_kv_heads != 0:
        raise ValueError(f"Query heads {hq} are not divisible by KV heads {num_kv_heads}.")
    groups = hq // num_kv_heads
    return scores.float().view(bsz, num_kv_heads, groups, q_len, k_len).mean(dim=2)


def _captured_scores(model, layer_idx: int, fallback, n_kv_heads: int, kv_len: int):
    attn = _get_layers(model)[layer_idx].self_attn
    scores = getattr(attn, "_keepkv_masked_exp_scores", None)
    if scores is None:
        # output_attentions is post-softmax and is not official-equivalent. Fail
        # rather than silently using the old approximation.
        raise RuntimeError(
            "KeepKV did not capture masked pre-electoral exp(logits). Ensure the "
            "KeepKV eager runtime patch is installed and attn_implementation='eager'."
        )
    if scores.shape[-1] != kv_len:
        raise RuntimeError(
            f"Captured KeepKV score length {scores.shape[-1]} != physical KV length {kv_len}."
        )
    log_scale = getattr(attn, "_keepkv_score_log_scale", None)
    if log_scale is None:
        raise RuntimeError("KeepKV captured scores are missing their shared log scale.")
    return _aggregate_query_heads(scores, n_kv_heads), float(log_scale)


def _roll_recent_scores(
    state: KeepKVLayerState,
    current_scores: torch.Tensor,
    current_log_scale: float,
    old_physical_len: int,
    recent_budget: int,
):
    """Official score-history update generalized from one token to a chunk.

    Rows are consumed chronologically. Masked future columns are never copied
    into the active prefix of a row.
    """
    bsz, heads, q_len, full_k_len = current_scores.shape
    if full_k_len != old_physical_len + q_len:
        raise RuntimeError(
            f"Expected K={old_physical_len}+Q={q_len}, got K={full_k_len}."
        )

    was_initialized = state.initialized
    if not was_initialized:
        # Official prefill: keep the last recent_budget query rows.
        state.previous_recent_scores = current_scores[:, :, -recent_budget:, :].detach().clone()
        state.cluster_cnt = torch.ones(
            (bsz, heads, full_k_len), device=current_scores.device, dtype=torch.float32
        )
        state.score_log_scale = float(current_log_scale)
        state.initialized = True
        return True

    prev = state.previous_recent_scores
    cnt = state.cluster_cnt
    if state.score_log_scale is None:
        raise RuntimeError("KeepKV initialized state is missing score_log_scale.")

    # Re-express old and current scores under one shared scale.  This is exactly
    # equivalent to the unscaled exp(logits) representation up to one common
    # positive multiplier, while avoiding overflow for Qwen-family logits.
    common_scale = max(float(state.score_log_scale), float(current_log_scale))
    prev_factor = math.exp(float(state.score_log_scale) - common_scale)
    current_factor = math.exp(float(current_log_scale) - common_scale)
    if prev_factor != 1.0:
        prev = prev * prev_factor
    if current_factor != 1.0:
        current_scores = current_scores * current_factor
    state.score_log_scale = common_scale
    if prev is None or cnt is None or prev.shape[-1] != old_physical_len or cnt.shape[-1] != old_physical_len:
        raise RuntimeError("KeepKV state length is not synchronized with the incoming physical cache.")

    # Sequentially reproduce the official decode update for every query in a chunk.
    for j in range(q_len):
        prev = torch.cat(
            [prev, torch.zeros((*prev.shape[:-1], 1), device=prev.device, dtype=torch.float32)], dim=-1
        )
        active_len = old_physical_len + j + 1
        row = current_scores[:, :, j:j + 1, :active_len]
        if row.shape[-1] < prev.shape[-1]:
            row = torch.cat(
                [row, torch.zeros((*row.shape[:-1], prev.shape[-1] - row.shape[-1]), device=row.device, dtype=row.dtype)],
                dim=-1,
            )
        prev = torch.cat([prev, row], dim=-2)[:, :, -recent_budget:, :]
        cnt = torch.cat(
            [cnt, torch.ones((*cnt.shape[:-1], 1), device=cnt.device, dtype=torch.float32)], dim=-1
        )
    state.previous_recent_scores = prev
    state.cluster_cnt = cnt
    return False


def _gather_3d(x: torch.Tensor, idx: torch.Tensor):
    return torch.gather(x, -1, idx)


def _gather_scores(x: torch.Tensor, idx: torch.Tensor):
    return torch.gather(x, -1, idx.unsqueeze(-2).expand(*x.shape[:-1], idx.shape[-1]))


def _gather_kv(x: torch.Tensor, idx: torch.Tensor):
    return torch.gather(x, 2, idx.unsqueeze(-1).expand(*idx.shape, x.shape[-1]))


def _official_compress_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    state: KeepKVLayerState,
    budget: LayerBudget,
    threshold: float,
    cfg: KeepKVConfig,
):
    """Vectorized official KV_update for B batches and GQA KV heads."""
    bsz, heads, seq_len, head_dim = key.shape
    scores = state.previous_recent_scores
    counts = state.cluster_cnt
    if scores is None or counts is None:
        raise RuntimeError("KeepKV state was not initialized.")
    if seq_len <= budget.total:
        return key, value

    r = budget.recent
    time_weights = torch.linspace(
        1.0, cfg.distance_weight, scores.shape[-2], device=scores.device, dtype=scores.dtype
    ).view(1, 1, -1, 1)
    selected_scores_sum = (scores * time_weights).sum(dim=-2)
    candidate_start = budget.sink
    candidate_end = seq_len - r
    heavy_k = budget.heavy
    if heavy_k > max(0, candidate_end - candidate_start):
        raise RuntimeError("Layer budget requests more heavy slots than available candidates.")

    pieces = []
    if budget.sink:
        pieces.append(torch.arange(budget.sink, device=key.device).view(1, 1, -1).expand(bsz, heads, -1))
    if heavy_k:
        candidate = selected_scores_sum[..., candidate_start:candidate_end]
        topk = torch.topk(candidate, heavy_k, dim=-1).indices.sort(dim=-1).values + candidate_start
        pieces.append(topk)
    recent_idx = torch.arange(seq_len - r, seq_len, device=key.device).view(1, 1, -1).expand(bsz, heads, -1)
    pieces.append(recent_idx)
    keep_idx = torch.cat(pieces, dim=-1)
    if keep_idx.shape[-1] != budget.total:
        raise AssertionError("sink + heavy + recent does not equal layer budget.")

    keep_mask = torch.zeros((bsz, heads, seq_len), device=key.device, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)
    all_idx = torch.arange(seq_len, device=key.device).view(1, 1, -1).expand(bsz, heads, -1)
    discard_idx = all_idx.masked_select(~keep_mask).view(bsz, heads, seq_len - budget.total)

    new_key = _gather_kv(key.float(), keep_idx)
    new_value = _gather_kv(value.float(), keep_idx)
    new_cnt = _gather_3d(counts, keep_idx)
    discard_key = _gather_kv(key.float(), discard_idx)
    discard_value = _gather_kv(value.float(), discard_idx)
    discard_scores = _gather_scores(scores, discard_idx)
    discard_cnt_original = _gather_3d(counts, discard_idx)
    kept_scores = _gather_scores(scores, keep_idx)

    cosine = torch.matmul(
        F.normalize(discard_key, p=2, dim=-1).half(),
        F.normalize(new_key, p=2, dim=-1).half().transpose(-1, -2),
    ).float()
    max_sim, max_sim_idx = cosine.max(dim=-1)
    max_sim_mask = max_sim >= float(threshold)

    # Official zero-inference-perturbation merge weights.
    if not torch.isfinite(kept_scores).all() or not torch.isfinite(discard_scores).all():
        raise RuntimeError("KeepKV recent-score state contains NaN/Inf before merge.")
    w_keep_est = kept_scores.mean(dim=-2)
    correction = (
        r / torch.arange(r, 0, -1, device=key.device, dtype=scores.dtype)
    ).view(1, 1, r)
    w_keep_est[..., -r:] *= correction
    # The official epsilon is added in the unscaled exp-score domain.  Under the
    # shared log scale used to prevent Qwen overflow, epsilon * exp(-scale) can
    # underflow to exact zero (e.g. scale ~= 3552).  A common positive floor is
    # scale-invariant for the weighted average and preserves the official role
    # of epsilon: preventing zero merge denominators without changing any
    # representable non-zero score ratio.
    score_dtype = w_keep_est.dtype
    finfo = torch.finfo(score_dtype)
    scale_value = float(state.score_log_scale or 0.0)
    if scale_value < 700.0:
        epsilon_scaled_value = cfg.epsilon * math.exp(-scale_value)
    else:
        epsilon_scaled_value = 0.0
    epsilon_scaled = max(float(epsilon_scaled_value), float(finfo.tiny))
    w_keep_est = w_keep_est.clamp_min(epsilon_scaled)
    w_keep = new_cnt * w_keep_est

    gated_discard_cnt = torch.where(max_sim_mask, discard_cnt_original, torch.zeros_like(discard_cnt_original))
    w_discard_est = discard_scores.mean(dim=-2).clamp_min(epsilon_scaled)
    w_discard = gated_discard_cnt * w_discard_est

    weighted_key = new_key * w_keep.unsqueeze(-1)
    weighted_value = new_value * w_keep.unsqueeze(-1)
    scatter_idx = max_sim_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    weighted_key.scatter_add_(2, scatter_idx, discard_key * w_discard.unsqueeze(-1))
    weighted_value.scatter_add_(2, scatter_idx, discard_value * w_discard.unsqueeze(-1))
    w_total = w_keep.scatter_add(2, max_sim_idx, w_discard)
    if not torch.isfinite(w_total).all() or torch.any(w_total <= 0):
        raise RuntimeError(
            "KeepKV merge denominator is non-finite or non-positive after "
            "scale-safe epsilon flooring: "
            f"min={float(w_total.min().item())}, max={float(w_total.max().item())}, "
            f"score_log_scale={scale_value}, epsilon_scaled={epsilon_scaled}."
        )
    new_key = weighted_key / w_total.unsqueeze(-1)
    new_value = weighted_value / w_total.unsqueeze(-1)
    if not torch.isfinite(new_key).all() or not torch.isfinite(new_value).all():
        raise RuntimeError("KeepKV produced non-finite merged key/value tensors.")

    cnt_total = new_cnt.scatter_add(2, max_sim_idx, gated_discard_cnt)
    new_recent_scores = kept_scores * new_cnt.unsqueeze(-2)
    weighted_discard_scores = discard_scores * gated_discard_cnt.unsqueeze(-2)
    new_recent_scores.scatter_add_(
        3,
        max_sim_idx.unsqueeze(-2).expand(-1, -1, new_recent_scores.shape[-2], -1),
        weighted_discard_scores,
    )
    new_recent_scores = new_recent_scores / cnt_total.unsqueeze(-2)

    # Official triangular masking after score merging.
    rows = new_recent_scores.shape[-2]
    tri = torch.tril(
        torch.ones((rows, budget.total), device=key.device, dtype=new_recent_scores.dtype),
        diagonal=budget.total - rows,
    )
    new_recent_scores *= tri.view(1, 1, rows, budget.total)

    state.previous_recent_scores = new_recent_scores
    state.cluster_cnt = cnt_total
    state.last_keep_idx = keep_idx.detach()
    state.last_rejected_vote_count = torch.where(
        max_sim_mask, torch.zeros_like(discard_cnt_original), discard_cnt_original
    ).sum(dim=-1)
    return new_key.to(key.dtype), new_value.to(value.dtype)


def compress_keepkv(
    model,
    past_key_values: Sequence,
    attentions,
    compress_ratio: float,
    sim_thr: Optional[float] = None,
    config: Optional[KeepKVConfig] = None,
):
    """Apply official KeepKV update to every layer after one forward call."""
    del attentions  # Official-equivalent scores come only from the runtime capture.
    cfg = config or KeepKVConfig()
    layers = _get_layers(model)
    if len(past_key_values) != len(layers):
        raise ValueError(f"Cache has {len(past_key_values)} layers; model has {len(layers)}.")
    budgets = _get_or_build_budgets(model, past_key_values, compress_ratio, cfg)
    states = _ensure_state_list(model, len(layers))
    compressed = []
    count_list = []

    for layer_idx, ((key, value), budget, state) in enumerate(zip(past_key_values, budgets, states)):
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError(f"Layer {layer_idx}: expected matching [B,H,S,D] K/V tensors.")
        bsz, n_kv_heads, seq_len, _ = key.shape
        current_scores, current_log_scale = _captured_scores(
            model, layer_idx, None, n_kv_heads, seq_len
        )
        current_scores = current_scores.to(key.device)
        q_len = current_scores.shape[-2]
        old_physical_len = seq_len - q_len
        is_prefill = _roll_recent_scores(
            state, current_scores, current_log_scale, old_physical_len, budget.recent
        )

        # Deprecated unified threshold override is accepted only for backward
        # compatibility. By default use the two official thresholds.
        threshold = (
            float(sim_thr) if sim_thr is not None
            else (cfg.prefill_similarity_threshold if is_prefill else cfg.similarity_threshold)
        )
        new_key, new_value = _official_compress_layer(key, value, state, budget, threshold, cfg)
        compressed.append((new_key, new_value))
        count_list.append(state.cluster_cnt)

        attn = layers[layer_idx].self_attn
        attn._keepkv_enabled = True
        attn._keepkv_counts = state.cluster_cnt
        attn._keepkv_masked_exp_scores = None
        attn._keepkv_score_log_scale = None

    model._keepkv_counts = count_list
    return tuple(compressed)


def get_keepkv_debug_state(model):
    """Return detached state useful for validation without mutating the model."""
    budgets_obj = getattr(model, "_keepkv_layer_budgets", None)
    budgets = budgets_obj.get("budgets") if isinstance(budgets_obj, dict) else None
    states = getattr(model, "_keepkv_layer_states", None) or []
    return {
        "budgets": budgets,
        "layers": [
            {
                "previous_recent_scores": None if s.previous_recent_scores is None else s.previous_recent_scores.detach(),
                "cluster_cnt": None if s.cluster_cnt is None else s.cluster_cnt.detach(),
                "last_keep_idx": None if s.last_keep_idx is None else s.last_keep_idx.detach(),
                "last_rejected_vote_count": None if s.last_rejected_vote_count is None else s.last_rejected_vote_count.detach(),
                "score_log_scale": s.score_log_scale,
            }
            for s in states
        ],
    }

# ---------------------------------------------------------------------------
# Memory-efficient one-shot prompt compression for LongBench-style evaluation
# ---------------------------------------------------------------------------

def prepare_keepkv_oneshot_prefill(
    model,
    prompt_length: int,
    compress_ratio: float,
    config: Optional[KeepKVConfig] = None,
    similarity_chunk_size: int = 256,
):
    """Prepare exact sufficient-statistic capture for one prompt compression.

    This mode is intended for LongBench: the complete prompt is prefetched in
    chunks, compressed once, and then decoded without further KeepKV repairs.
    It retains only the two statistics used by the official prefill update:
    the weighted recent-query sum for heavy selection and the unweighted mean
    numerator for merge weights.  It does not retain [R,K] score history.
    """
    cfg = config or KeepKVConfig()
    prompt_length = int(prompt_length)
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    # In one-shot prompt compression the unified ratio is defined against the
    # current prompt, not the model's maximum context length.
    cfg = KeepKVConfig(**{
        **{name: getattr(cfg, name) for name in KeepKVConfig.__dataclass_fields__},
        "logical_length": prompt_length,
    })
    layers = _get_layers(model)
    layer_lengths = [prompt_length] * len(layers)
    budgets = build_pyramid_budgets(layer_lengths, prompt_length, compress_ratio, cfg)
    model._keepkv_oneshot_config = cfg
    model._keepkv_oneshot_budgets = budgets
    model._keepkv_similarity_chunk_size = max(1, int(similarity_chunk_size))
    model._keepkv_layer_states = [KeepKVLayerState() for _ in layers]
    model._keepkv_counts = [None] * len(layers)

    for layer, budget in zip(layers, budgets):
        attn = layer.self_attn
        attn._keepkv_enabled = True
        attn._keepkv_capture_disabled = False
        attn._keepkv_summary_mode = True
        attn._keepkv_summary_prompt_length = prompt_length
        attn._keepkv_summary_recent = int(budget.recent)
        attn._keepkv_summary_distance_weight = float(cfg.distance_weight)
        attn._keepkv_summary_chunk_start = 0
        attn._keepkv_summary_sum = None
        attn._keepkv_summary_weighted_sum = None
        attn._keepkv_summary_log_scale = None
        attn._keepkv_counts = None
        attn._keepkv_masked_exp_scores = None
        attn._keepkv_score_log_scale = None
    return budgets


def set_keepkv_oneshot_prefill_chunk(model, chunk_start: int) -> None:
    """Set the logical query offset for the next chunked prompt forward."""
    for layer in _get_layers(model):
        layer.self_attn._keepkv_summary_chunk_start = int(chunk_start)


def _oneshot_similarity_merge(
    key: torch.Tensor,
    value: torch.Tensor,
    keep_idx: torch.Tensor,
    discard_idx: torch.Tensor,
    mean_scores: torch.Tensor,
    counts: torch.Tensor,
    recent_budget: int,
    threshold: float,
    epsilon_scaled: float,
    chunk_size: int,
):
    """Memory-bounded equivalent of the official top-1 similarity merge."""
    bsz, heads, _, head_dim = key.shape
    kept_key_src = _gather_kv(key, keep_idx)
    kept_value_src = _gather_kv(value, keep_idx)
    kept_scores = _gather_3d(mean_scores, keep_idx)
    kept_cnt = _gather_3d(counts, keep_idx)

    if recent_budget > 0:
        correction = (
            recent_budget
            / torch.arange(
                recent_budget, 0, -1, device=key.device, dtype=kept_scores.dtype
            )
        ).view(1, 1, recent_budget)
        kept_scores[..., -recent_budget:] *= correction
    kept_scores = kept_scores.clamp_min(float(epsilon_scaled))
    kept_weight = kept_cnt * kept_scores

    # Accumulate in float32; source K/V remain in model dtype until each chunk
    # is consumed, avoiding full float32 copies of all discarded slots.
    weighted_key = kept_key_src.float() * kept_weight.unsqueeze(-1)
    weighted_value = kept_value_src.float() * kept_weight.unsqueeze(-1)
    total_weight = kept_weight.clone()
    total_cnt = kept_cnt.clone()
    kept_norm = F.normalize(kept_key_src.float(), p=2, dim=-1).to(torch.float16)

    rejected_votes = torch.zeros((bsz, heads), device=key.device, dtype=counts.dtype)
    n_discard = int(discard_idx.shape[-1])
    for start in range(0, n_discard, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), n_discard)
        idx = discard_idx[..., start:end]
        dkey = _gather_kv(key, idx)
        dvalue = _gather_kv(value, idx)
        dscore = _gather_3d(mean_scores, idx).clamp_min(float(epsilon_scaled))
        dcnt = _gather_3d(counts, idx)

        cosine = torch.matmul(
            F.normalize(dkey.float(), p=2, dim=-1).to(torch.float16),
            kept_norm.transpose(-1, -2),
        ).float()
        max_sim, target = cosine.max(dim=-1)
        accepted = max_sim >= float(threshold)
        gated_cnt = torch.where(accepted, dcnt, torch.zeros_like(dcnt))
        rejected_votes += torch.where(accepted, torch.zeros_like(dcnt), dcnt).sum(dim=-1)
        dweight = gated_cnt * dscore

        scatter_idx = target.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        weighted_key.scatter_add_(2, scatter_idx, dkey.float() * dweight.unsqueeze(-1))
        weighted_value.scatter_add_(2, scatter_idx, dvalue.float() * dweight.unsqueeze(-1))
        total_weight.scatter_add_(2, target, dweight)
        total_cnt.scatter_add_(2, target, gated_cnt)

        del idx, dkey, dvalue, dscore, dcnt, cosine, max_sim, accepted, gated_cnt, dweight

    total_weight = total_weight.clamp_min(float(epsilon_scaled))
    new_key = weighted_key / total_weight.unsqueeze(-1)
    new_value = weighted_value / total_weight.unsqueeze(-1)
    return new_key.to(key.dtype), new_value.to(value.dtype), total_cnt, rejected_votes


def compress_keepkv_oneshot_prefill(
    model,
    past_key_values: Sequence,
    sim_thr: Optional[float] = None,
):
    """Compress a completed prompt from streaming sufficient statistics.

    The heavy selection and merge weights are mathematically the same as using
    the final R recent-query score rows.  Score history is intentionally not
    reconstructed after compression because LongBench performs no subsequent
    KeepKV repair during decode; only electoral-vote counts are required.
    """
    layers = _get_layers(model)
    cfg = getattr(model, "_keepkv_oneshot_config", None)
    budgets = getattr(model, "_keepkv_oneshot_budgets", None)
    if cfg is None or budgets is None:
        raise RuntimeError("prepare_keepkv_oneshot_prefill() was not called")
    if len(past_key_values) != len(layers):
        raise ValueError("Cache/model layer count mismatch")

    threshold = float(cfg.prefill_similarity_threshold if sim_thr is None else sim_thr)
    similarity_chunk_size = int(getattr(model, "_keepkv_similarity_chunk_size", 256))
    states = _ensure_state_list(model, len(layers))
    compressed = []
    count_list = []

    for layer_idx, ((key, value), budget, state, layer) in enumerate(
        zip(past_key_values, budgets, states, layers)
    ):
        attn = layer.self_attn
        score_sum = getattr(attn, "_keepkv_summary_sum", None)
        weighted_sum = getattr(attn, "_keepkv_summary_weighted_sum", None)
        log_scale = getattr(attn, "_keepkv_summary_log_scale", None)
        if score_sum is None or weighted_sum is None or log_scale is None:
            raise RuntimeError(f"Layer {layer_idx}: KeepKV prompt summary is missing")
        seq_len = int(key.shape[-2])
        if score_sum.shape[-1] != seq_len or weighted_sum.shape[-1] != seq_len:
            raise RuntimeError(f"Layer {layer_idx}: summary/cache length mismatch")

        bsz, heads = key.shape[:2]
        counts = torch.ones((bsz, heads, seq_len), device=key.device, dtype=torch.float32)
        if seq_len <= budget.total:
            new_key, new_value, new_cnt = key, value, counts
            keep_idx = torch.arange(seq_len, device=key.device).view(1,1,-1).expand(bsz,heads,-1)
            rejected = torch.zeros((bsz, heads), device=key.device)
        else:
            importance = weighted_sum.to(key.device)
            mean_scores = (score_sum / max(1, int(budget.recent))).to(key.device)
            candidate_start = int(budget.sink)
            candidate_end = seq_len - int(budget.recent)
            pieces = []
            if budget.sink:
                pieces.append(torch.arange(budget.sink, device=key.device).view(1,1,-1).expand(bsz,heads,-1))
            if budget.heavy:
                candidate = importance[..., candidate_start:candidate_end]
                topk = torch.topk(candidate, budget.heavy, dim=-1).indices.sort(dim=-1).values + candidate_start
                pieces.append(topk)
            recent_idx = torch.arange(seq_len-budget.recent, seq_len, device=key.device).view(1,1,-1).expand(bsz,heads,-1)
            pieces.append(recent_idx)
            keep_idx = torch.cat(pieces, dim=-1)

            keep_mask = torch.zeros((bsz, heads, seq_len), device=key.device, dtype=torch.bool)
            keep_mask.scatter_(-1, keep_idx, True)
            all_idx = torch.arange(seq_len, device=key.device).view(1,1,-1).expand(bsz,heads,-1)
            discard_idx = all_idx.masked_select(~keep_mask).view(bsz, heads, seq_len-budget.total)

            finfo = torch.finfo(mean_scores.dtype)
            scale_value = float(log_scale)
            eps_value = cfg.epsilon * math.exp(-scale_value) if scale_value < 80.0 else 0.0
            epsilon_scaled = max(float(eps_value), float(finfo.tiny))
            new_key, new_value, new_cnt, rejected = _oneshot_similarity_merge(
                key, value, keep_idx, discard_idx, mean_scores, counts,
                budget.recent, threshold, epsilon_scaled, similarity_chunk_size,
            )

        state.previous_recent_scores = None
        state.cluster_cnt = new_cnt
        state.last_keep_idx = keep_idx.detach()
        state.last_rejected_vote_count = rejected
        state.initialized = True
        state.score_log_scale = float(log_scale)
        compressed.append((new_key, new_value))
        count_list.append(new_cnt)

        # Decode only needs electoral-vote counts. Disable all score capture and
        # release prompt summaries immediately.
        attn._keepkv_counts = new_cnt
        attn._keepkv_enabled = True
        attn._keepkv_capture_disabled = True
        attn._keepkv_summary_mode = False
        attn._keepkv_summary_sum = None
        attn._keepkv_summary_weighted_sum = None
        attn._keepkv_summary_log_scale = None
        attn._keepkv_masked_exp_scores = None
        attn._keepkv_score_log_scale = None

    model._keepkv_counts = count_list
    model._keepkv_oneshot_budgets = None
    return tuple(compressed)
