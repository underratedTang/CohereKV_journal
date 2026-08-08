"""Graph-MIS certified ``mymethod`` KV-cache merging for LLaMA models."""

from __future__ import annotations

from typing import List, Optional

import torch

from models.common.mymethod_common import (
    DEFAULT_DISTANCE_CHUNK_SIZE,
    DEFAULT_KEEPKV_RECENCY_BETA,
    DEFAULT_KEEPKV_RECENT_PROTECT_SIZE,
    DEFAULT_KEEPKV_SCORE_WINDOW,
    DEFAULT_KEEPKV_SINK_SIZE,
    DEFAULT_D2O_RECENT_SIZE,
    DEFAULT_D2O_SINK_SIZE,
    DEFAULT_SINK_CAN_ABSORB,
    DEFAULT_LAMBDA,
    DEFAULT_SELECTOR,
    DEFAULT_GROUP_MODE,
    DEFAULT_GRAPH_BACKEND,
    DEFAULT_ROUTE_BACKEND,
    append_score_rows,
    as_legacy_tuple,
    base_score_rows_for_kv_heads,
    attention_importance_for_kv_heads,
    initial_or_existing_state,
    merge_layer,
    normalize_selector,
    reference_scale_or_initialize,
)


def reset_mymethod_state(model) -> None:
    """Clear all sequence-local certificate and selector metadata."""
    for name in (
        "_mymethod_counts",
        "_mymethod_radii",
        "_mymethod_diameters",
        "_mymethod_score_history",
        "_mymethod_score_history_ptrs",
        "_mymethod_score_log_scales",
        "_mymethod_positions",
        "_mymethod_next_positions",
        "_mymethod_reference_scales",
    ):
        setattr(model, name, None)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = layer.self_attn
            attn._mymethod_eager_enabled = False
            attn._mymethod_sdpa_enabled = False
            attn._mymethod_counts_kv = None
            attn._kvmerge_capture_attention_scores = False
            for name in (
                "_kvmerge_captured_attention_score",
                "_kvmerge_captured_score_log_scale",
                "_mymethod_d2o_cumulative_importance",
            ):
                if hasattr(attn, name):
                    delattr(attn, name)


def disable_mymethod_sdpa(model) -> None:
    reset_mymethod_state(model)


def disable_mymethod_eager(model) -> None:
    reset_mymethod_state(model)


def _enable_mymethod_eager(model, counts) -> None:
    model._mymethod_counts = counts
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._mymethod_eager_enabled = True
        attn._mymethod_sdpa_enabled = False
        attn._mymethod_counts_kv = counts[layer_idx]
        attn._keepkv_eager_enabled = False
        attn._keepkv_log_bias_q = None


def Repair(
    model,
    key_value_cache,
    attention,
    hidden_states: Optional[List[torch.Tensor]],
    ratio: float,
    lambda_mymethod: float = DEFAULT_LAMBDA,
    target_length: Optional[int] = None,
    keepkv_sink_size: int = DEFAULT_KEEPKV_SINK_SIZE,
    keepkv_recent_protect_size: int = DEFAULT_KEEPKV_RECENT_PROTECT_SIZE,
    keepkv_score_window: int = DEFAULT_KEEPKV_SCORE_WINDOW,
    keepkv_recency_beta: float = DEFAULT_KEEPKV_RECENCY_BETA,
    keepkv_sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    d2o_sink_size: int = DEFAULT_D2O_SINK_SIZE,
    d2o_recent_size: int = DEFAULT_D2O_RECENT_SIZE,
    d2o_sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    d2o_use_log_count: bool = True,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    selector: str = DEFAULT_SELECTOR,
    group_mode: str = DEFAULT_GROUP_MODE,
    graph_backend: str = DEFAULT_GRAPH_BACKEND,
    route_backend: str = DEFAULT_ROUTE_BACKEND,
):
    """Compress a LLaMA KV cache with graph-MIS certified grouping."""
    del hidden_states
    selector = normalize_selector(selector)
    cache = as_legacy_tuple(key_value_cache)
    if cache is None:
        raise ValueError("key_value_cache is None")
    if attention is None:
        raise ValueError(
            "attention is None. mymethod requires selector scores for retained/evicted splitting."
        )

    new_past = []
    new_counts = []
    new_radii = []
    new_diameters = []
    new_history = []
    new_history_ptrs = []
    new_log_scales = []
    new_positions = []
    new_next_positions = []
    new_reference_scales = []
    new_d2o_importance = []

    if selector == "keepkv":
        selector_sink_size = max(int(keepkv_sink_size), 0)
        selector_recent_size = max(int(keepkv_recent_protect_size), 0)
        selector_score_window = max(int(keepkv_score_window), 0)
        selector_recency_beta = float(keepkv_recency_beta)
        selector_sink_can_absorb = bool(keepkv_sink_can_absorb)
    elif selector == "d2o":
        selector_sink_size = max(int(d2o_sink_size), 0)
        selector_recent_size = max(int(d2o_recent_size), 0)
        selector_score_window = 0
        selector_recency_beta = 1.0
        selector_sink_can_absorb = bool(d2o_sink_can_absorb)
    else:
        selector_sink_size = 0
        selector_recent_size = 0
        selector_score_window = 0
        selector_recency_beta = 1.0
        selector_sink_can_absorb = True

    for layer_idx, (keys, values) in enumerate(cache):
        bsz, h_kv, length, _ = keys.shape
        if layer_idx >= len(attention) or attention[layer_idx] is None:
            raise ValueError(f"Missing mymethod base-score rows for layer {layer_idx}")
        counts, radii, diameters, history, ptr, positions, next_position, old_scale = (
            initial_or_existing_state(model, layer_idx, keys, selector_score_window)
        )
        reference_scale = reference_scale_or_initialize(model, layer_idx, keys)
        current_importance = None
        if selector == "keepkv":
            score_rows = base_score_rows_for_kv_heads(attention[layer_idx], h_kv).to(
                device=keys.device, dtype=torch.float32
            )
            if score_rows.shape[:2] != (bsz, h_kv) or score_rows.shape[-1] != length:
                raise ValueError(
                    f"Layer score shape mismatch at layer {layer_idx}: "
                    f"expected [B={bsz},H={h_kv},Q,K={length}], got {tuple(score_rows.shape)}"
                )
            attn_module = model.model.layers[layer_idx].self_attn
            current_scale = float(
                getattr(attn_module, "_kvmerge_captured_score_log_scale", 0.0)
            )
            history, ptr, common_scale = append_score_rows(
                history, ptr, old_scale, score_rows, current_scale
            )
        else:
            current_importance = attention_importance_for_kv_heads(
                attention[layer_idx], h_kv
            ).to(device=keys.device, dtype=torch.float32)
            if tuple(current_importance.shape) != (bsz, h_kv, length):
                raise ValueError(
                    f"Layer attention selector score mismatch at layer {layer_idx}: "
                    f"expected {(bsz, h_kv, length)}, got {tuple(current_importance.shape)}"
                )
            common_scale = None

        merge_result = merge_layer(
            keys=keys,
            values=values,
            counts=counts,
            radii=radii,
            diameters=diameters,
            score_history=history,
            history_ptr=ptr,
            positions=positions,
            ratio=ratio,
            reference_scale=reference_scale,
            lambda_mymethod=lambda_mymethod,
            target_length=target_length,
            sink_size=selector_sink_size,
            recent_window=selector_recent_size,
            recency_beta=selector_recency_beta,
            distance_chunk_size=distance_chunk_size,
            sink_can_absorb=selector_sink_can_absorb,
            selector=selector,
            current_importance=current_importance,
            group_mode=group_mode,
            graph_backend=graph_backend,
            route_backend=route_backend,
            diagnostics_sink=getattr(model, "_mymethod_diagnostics_rows", None),
            diagnostics_context={"layer_idx": layer_idx},
            return_selector_state=(selector == "d2o"),
        )
        if selector == "d2o":
            knew, vnew, cnew, rnew, dnew, hnew, pnew, d2o_state_new = merge_result
            if d2o_state_new is None:
                raise RuntimeError("selector=d2o did not return cumulative importance state")
            new_d2o_importance.append(d2o_state_new)
        else:
            knew, vnew, cnew, rnew, dnew, hnew, pnew = merge_result
        new_past.append((knew, vnew))
        new_counts.append(cnew)
        new_radii.append(rnew)
        new_diameters.append(dnew)
        new_history.append(hnew)
        new_history_ptrs.append(ptr)
        new_log_scales.append(common_scale)
        new_positions.append(pnew)
        new_next_positions.append(next_position)
        new_reference_scales.append(reference_scale)

    model._mymethod_radii = new_radii
    model._mymethod_diameters = new_diameters
    model._mymethod_score_history = new_history
    model._mymethod_score_history_ptrs = new_history_ptrs
    model._mymethod_score_log_scales = new_log_scales
    model._mymethod_positions = new_positions
    model._mymethod_next_positions = new_next_positions
    model._mymethod_reference_scales = new_reference_scales
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._mymethod_selector = selector
        attn._mymethod_score_window = int(selector_score_window) if selector == "keepkv" else 0
        attn._mymethod_d2o_use_log_count = bool(d2o_use_log_count)
        if selector == "d2o":
            attn._mymethod_d2o_cumulative_importance = new_d2o_importance[layer_idx].detach()
        elif hasattr(attn, "_mymethod_d2o_cumulative_importance"):
            delattr(attn, "_mymethod_d2o_cumulative_importance")
    _enable_mymethod_eager(model, new_counts)
    return tuple(new_past)
