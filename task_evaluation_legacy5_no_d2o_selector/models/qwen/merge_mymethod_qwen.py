"""Graph-MIS certified ``mymethod`` KV-cache merging for Qwen models."""

from __future__ import annotations

from typing import List, Optional

import torch

from models.common.mymethod_common import (
    DEFAULT_DISTANCE_CHUNK_SIZE,
    DEFAULT_RECENCY_BETA,
    DEFAULT_RECENT_WINDOW,
    DEFAULT_SINK_CAN_ABSORB,
    DEFAULT_SINK_SIZE,
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
            ):
                if hasattr(attn, name):
                    delattr(attn, name)


def disable_mymethod_sdpa(model) -> None:
    reset_mymethod_state(model)


def disable_mymethod_eager(model) -> None:
    reset_mymethod_state(model)


def _enable_mymethod_sdpa(model, counts) -> None:
    model._mymethod_counts = counts
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._mymethod_sdpa_enabled = True
        attn._mymethod_eager_enabled = True
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
    sink_size: int = DEFAULT_SINK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    recency_beta: float = DEFAULT_RECENCY_BETA,
    distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
    sink_can_absorb: bool = DEFAULT_SINK_CAN_ABSORB,
    selector: str = DEFAULT_SELECTOR,
    group_mode: str = DEFAULT_GROUP_MODE,
    graph_backend: str = DEFAULT_GRAPH_BACKEND,
    route_backend: str = DEFAULT_ROUTE_BACKEND,
):
    """Compress a Qwen KV cache with graph-MIS certified grouping."""
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

    for layer_idx, (keys, values) in enumerate(cache):
        bsz, h_kv, length, _ = keys.shape
        if layer_idx >= len(attention) or attention[layer_idx] is None:
            raise ValueError(f"Missing mymethod base-score rows for layer {layer_idx}")
        score_window = recent_window if selector == "keepkv" else 0
        counts, radii, diameters, history, ptr, positions, next_position, old_scale = (
            initial_or_existing_state(model, layer_idx, keys, score_window)
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

        knew, vnew, cnew, rnew, dnew, hnew, pnew = merge_layer(
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
            sink_size=sink_size,
            recent_window=recent_window,
            recency_beta=recency_beta,
            distance_chunk_size=distance_chunk_size,
            sink_can_absorb=sink_can_absorb,
            selector=selector,
            current_importance=current_importance,
            group_mode=group_mode,
            graph_backend=graph_backend,
            route_backend=route_backend,
            diagnostics_sink=getattr(model, "_mymethod_diagnostics_rows", None),
            diagnostics_context={"layer_idx": layer_idx},
        )
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
    for layer in model.model.layers:
        layer.self_attn._mymethod_selector = selector
        layer.self_attn._mymethod_score_window = (
            int(recent_window) if selector == "keepkv" else 0
        )
    _enable_mymethod_sdpa(model, new_counts)
    return tuple(new_past)


def Repair_qwen(*args, **kwargs):
    return Repair(*args, **kwargs)
