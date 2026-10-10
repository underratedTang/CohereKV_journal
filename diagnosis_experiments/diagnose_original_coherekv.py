#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EVAL_ROOT = PROJECT_ROOT / "task_evaluation"
if str(TASK_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_EVAL_ROOT))

from models.common.mymethod_common import (  # noqa: E402
    build_merge_plan,
    effective_radius,
    recent_window_importance,
    reference_scale_from_original_keys,
)
from experiment_utils import parse_float_csv, resolve_selection  # noqa: E402


@dataclass
class HeadState:
    keys: torch.Tensor          # [L,d], physical cached representatives
    counts: torch.Tensor        # [L] int32
    radii: torch.Tensor         # [L] fp32
    diameters: torch.Tensor     # [L] fp32
    history: torch.Tensor       # [W,L] fp32 KeepKV score history
    positions: torch.Tensor     # [L] int32
    original_to_state: np.ndarray  # [L0], -1 if already discarded, otherwise current state id


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Diagnose why the original CohereKV graph-MIS method can approach Eviction-Only. "
            "The script preserves the production C/R/D/Gamma planner and tests fixed-Delta "
            "certificate saturation, routing rejection, conflict-graph filtering, and matched "
            "Eviction-Only / route-compatible Merge-All counterfactuals."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--model_seqlen", type=int, default=8192)
    p.add_argument("--probe_tokens", type=int, default=8)
    p.add_argument("--max_eval_sequences", type=int, default=1)
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--prefill_chunk_size", type=int, default=128)

    p.add_argument("--single_round_budgets", default="0.2,0.4,0.6,0.8")
    p.add_argument("--lambda_values", default="0.5,0.7,0.9")
    p.add_argument("--repeated_rounds", type=int, default=5)
    p.add_argument("--round_keep_ratio", type=float, default=0.8)
    p.add_argument("--min_physical_entries", type=int, default=256)

    p.add_argument("--keepkv_sink_size", type=int, default=4)
    p.add_argument("--keepkv_recent_protect_size", type=int, default=128)
    p.add_argument("--keepkv_score_window", type=int, default=128)
    p.add_argument("--keepkv_recency_beta", type=float, default=1.0)
    p.add_argument("--distance_chunk_size", type=int, default=1024)
    p.add_argument("--geometry_chunk_size", type=int, default=512)
    p.add_argument("--layer_selection", default="0,8,16,24")
    p.add_argument("--kv_head_selection", default="0,2,4,6")
    p.add_argument("--numeric_eps", type=float, default=1e-6)
    p.add_argument("--attention_prob_eps", type=float, default=1e-30)
    p.add_argument("--output_norm_eps", type=float, default=1e-20)
    p.add_argument("--progress_every_head", action="store_true")
    p.add_argument("--output_dir", required=True)
    return p


def parse_dtype(name: str):
    x = str(name).lower()
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if x not in table:
        raise ValueError(f"Unsupported torch dtype {name!r}")
    return table[x]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_stream(tokenizer, dataset_name: str) -> torch.Tensor:
    if dataset_name != "wikitext2":
        raise ValueError("This diagnosis currently supports only wikitext2.")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    return enc.input_ids


def cache_key_value_layers(past_key_values) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if past_key_values is None:
        raise RuntimeError("Model returned no past_key_values although use_cache=True.")
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    if isinstance(past_key_values, (tuple, list)):
        return [layer[0] for layer in past_key_values], [layer[1] for layer in past_key_values]
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [layer[0] for layer in legacy], [layer[1] for layer in legacy]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def first_execution_device(model, requested: str) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(requested)


def transformer_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Cannot locate transformer layers on the loaded model.")


def selector_patch_functions(model_type: str):
    mt = str(model_type).lower()
    if mt == "llama":
        from models.llama.mymethod_runtime_patch_llama import (
            apply_mymethod_runtime_patch,
            remove_mymethod_runtime_patch,
        )
        return apply_mymethod_runtime_patch, remove_mymethod_runtime_patch
    if mt in {"qwen2", "qwen"}:
        from models.qwen.mymethod_runtime_patch_qwen import (
            apply_mymethod_qwen_runtime_patch,
            remove_mymethod_qwen_runtime_patch,
        )
        return apply_mymethod_qwen_runtime_patch, remove_mymethod_qwen_runtime_patch
    raise ValueError(
        f"Exact KeepKV-score capture currently supports Llama/Qwen2, got model_type={model_type!r}."
    )


def forward_cache_chunk(model, chunk: torch.Tensor, past, device: torch.device):
    with torch.inference_mode():
        return model(
            input_ids=chunk.to(device),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )


def build_full_cache_with_keepkv_scores(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    score_window: int,
    model_type: str,
) -> Tuple[object, List[torch.Tensor], List[float]]:
    if chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be > 0")
    length = int(input_ids.shape[1])
    window = min(max(int(score_window), 1), length)
    warmup_end = length - window
    past = None
    for start in range(0, warmup_end, chunk_size):
        end = min(start + chunk_size, warmup_end)
        out = forward_cache_chunk(model, input_ids[:, start:end], past, device)
        past = out.past_key_values
        del out

    layers = transformer_layers(model)
    apply_patch, remove_patch = selector_patch_functions(model_type)
    apply_patch()
    try:
        for layer in layers:
            attn = layer.self_attn
            attn._mymethod_selector = "keepkv"
            attn._mymethod_score_window = window
            attn._kvmerge_capture_attention_scores = True
            attn._kvmerge_attention_chunk_size = min(chunk_size, window)
            attn._mymethod_eager_enabled = False
            attn._mymethod_counts_kv = None
            attn._keepkv_eager_enabled = False
            attn._keepkv_log_bias_q = None
            for name in ("_kvmerge_captured_attention_score", "_kvmerge_captured_score_log_scale"):
                if hasattr(attn, name):
                    delattr(attn, name)

        out = forward_cache_chunk(model, input_ids[:, warmup_end:length], past, device)
        past = out.past_key_values
        del out
        score_rows: List[torch.Tensor] = []
        score_scales: List[float] = []
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            score = getattr(attn, "_kvmerge_captured_attention_score", None)
            if score is None:
                raise RuntimeError(f"No KeepKV selector score captured for layer {layer_idx}.")
            if score.ndim != 4 or int(score.shape[-2]) != window or int(score.shape[-1]) != length:
                raise RuntimeError(
                    f"Bad KeepKV selector score shape at layer {layer_idx}: {tuple(score.shape)}; "
                    f"expected [B,Hkv,{window},{length}]."
                )
            score_rows.append(score.detach())
            score_scales.append(float(getattr(attn, "_kvmerge_captured_score_log_scale", 0.0)))
    finally:
        for layer in layers:
            layer.self_attn._kvmerge_capture_attention_scores = False
        remove_patch()
    return past, score_rows, score_scales


def run_probe_queries(model, past, probe_ids: torch.Tensor, device: torch.device):
    with torch.inference_mode():
        out = model(
            input_ids=probe_ids.to(device),
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    if out.attentions is None:
        raise RuntimeError("Probe forward returned no attentions. Use --attn_implementation eager.")
    return out


def initial_head_state(raw_keys: torch.Tensor, score_history: torch.Tensor) -> HeadState:
    length = int(raw_keys.shape[0])
    device = raw_keys.device
    return HeadState(
        keys=raw_keys.contiguous(),
        counts=torch.ones(length, dtype=torch.int32, device=device),
        radii=torch.zeros(length, dtype=torch.float32, device=device),
        diameters=torch.zeros(length, dtype=torch.float32, device=device),
        history=score_history.float().contiguous(),
        positions=torch.arange(length, dtype=torch.int32, device=device),
        original_to_state=np.arange(length, dtype=np.int64),
    )


def state_importance(state: HeadState, recency_beta: float) -> torch.Tensor:
    h = state.history.unsqueeze(0)
    return recent_window_importance(h, ptr=0, beta=recency_beta)[0]


def frozen_reference_scale(raw_keys: torch.Tensor) -> torch.Tensor:
    return reference_scale_from_original_keys(raw_keys.unsqueeze(0).unsqueeze(0))[0, 0, 0].detach()


def current_reference_scale(keys: torch.Tensor) -> torch.Tensor:
    return reference_scale_from_original_keys(keys.unsqueeze(0).unsqueeze(0))[0, 0, 0].detach()


def build_plan(
    state: HeadState,
    keep_len: int,
    reference_scale: torch.Tensor,
    lam: float,
    sink_size: int,
    recent_protect_size: int,
    recency_beta: float,
    distance_chunk_size: int,
    reset_certificates: bool = False,
):
    length = int(state.keys.shape[0])
    if not 1 <= keep_len < length:
        raise ValueError(f"keep_len={keep_len} must be in [1,{length-1}]")
    imp = state_importance(state, recency_beta)
    radii = torch.zeros_like(state.radii) if reset_certificates else state.radii
    diameters = torch.zeros_like(state.diameters) if reset_certificates else state.diameters
    return build_merge_plan(
        keys=state.keys.unsqueeze(0),
        counts=state.counts.float().unsqueeze(0),
        radii=radii.unsqueeze(0),
        diameters=diameters.unsqueeze(0),
        importance=imp.unsqueeze(0),
        positions=state.positions.unsqueeze(0),
        keep_len=int(keep_len),
        reference_scale=reference_scale.reshape(1, 1),
        lambda_mymethod=float(lam),
        sink_size=int(sink_size),
        recent_window=int(recent_protect_size),
        distance_chunk_size=int(distance_chunk_size),
        selector="keepkv",
        group_mode="full",
        graph_backend="torch",
        route_backend="torch",
    )


def mapping_after_plan(current_mapping: np.ndarray, plan, accepted_mask: torch.Tensor) -> np.ndarray:
    keep_idx = plan.keep_idx[0].detach().cpu().numpy().astype(np.int64, copy=False)
    evict_idx = plan.evict_idx[0].detach().cpu().numpy().astype(np.int64, copy=False)
    best_idx = plan.best_idx[0].detach().cpu().numpy().astype(np.int64, copy=False)
    accepted = accepted_mask.detach().cpu().numpy().astype(bool, copy=False)
    old_len = int(plan.keep_idx.shape[1] + plan.evict_idx.shape[1])
    old_to_new = np.full(old_len, -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    if evict_idx.size:
        chosen = np.nonzero(accepted)[0]
        if chosen.size:
            old_to_new[evict_idx[chosen]] = best_idx[chosen]
    out = np.full_like(current_mapping, -1)
    live = current_mapping >= 0
    out[live] = old_to_new[current_mapping[live]]
    return out


def apply_plan(state: HeadState, plan, distance_chunk_size: int) -> HeadState:
    keep_idx = plan.keep_idx[0]
    evict_idx = plan.evict_idx[0]
    accepted = plan.accepted[0]
    best_idx = plan.best_idx[0]
    m = int(keep_idx.numel())
    d = int(state.keys.shape[-1])

    km = state.keys.index_select(0, keep_idx)
    cm_i = state.counts.index_select(0, keep_idx)
    cm = cm_i.float()
    dm = state.diameters.index_select(0, keep_idx)
    hm = state.history.index_select(1, keep_idx)
    pm = state.positions.index_select(0, keep_idx)

    kn = state.keys.index_select(0, evict_idx)
    cn_i = state.counts.index_select(0, evict_idx)
    cn = cn_i.float()
    hn = state.history.index_select(1, evict_idx)

    accepted_i = accepted.to(torch.int32)
    accepted_f = accepted.float()
    add_counts = torch.zeros(m, dtype=torch.int32, device=state.keys.device)
    if evict_idx.numel():
        add_counts.scatter_add_(0, best_idx, cn_i * accepted_i)
    cnew_i = cm_i + add_counts
    cnew = cnew_i.float()

    add_k = torch.zeros((m, d), dtype=torch.float32, device=state.keys.device)
    if evict_idx.numel():
        src = kn.float() * (cn * accepted_f).unsqueeze(-1)
        add_k.scatter_add_(0, best_idx.unsqueeze(-1).expand(-1, d), src)
    knew_f = (cm.unsqueeze(-1) * km.float() + add_k) / cnew.clamp_min(1e-8).unsqueeze(-1)
    knew = knew_f.to(state.keys.dtype)
    knew_cert = knew.float()

    w = int(state.history.shape[0])
    add_h = torch.zeros((w, m), dtype=torch.float32, device=state.keys.device)
    if evict_idx.numel() and w:
        src_h = hn.float() * (cn * accepted_f).unsqueeze(0)
        add_h.scatter_add_(1, best_idx.unsqueeze(0).expand(w, -1), src_h)
    hnew = (cm.unsqueeze(0) * hm.float() + add_h) / cnew.clamp_min(1e-8).unsqueeze(0)

    dnew = torch.maximum(dm.float(), plan.group_gamma_max[0].float())
    rhat_all = effective_radius(state.counts.float(), state.radii, state.diameters)
    rhat_m = rhat_all.index_select(0, keep_idx)
    rhat_n = rhat_all.index_select(0, evict_idx)
    rnew_tilde = rhat_m + torch.linalg.vector_norm(km.float() - knew_cert, dim=-1)
    if evict_idx.numel():
        if not hasattr(rnew_tilde, "scatter_reduce_"):
            raise RuntimeError("This diagnosis requires torch.Tensor.scatter_reduce_.")
        chunk = int(evict_idx.numel()) if distance_chunk_size <= 0 else min(max(int(distance_chunk_size), 1), int(evict_idx.numel()))
        for start in range(0, int(evict_idx.numel()), chunk):
            end = min(start + chunk, int(evict_idx.numel()))
            idx = best_idx[start:end]
            centers = knew_cert.index_select(0, idx)
            cr = rhat_n[start:end] + torch.linalg.vector_norm(kn[start:end].float() - centers, dim=-1)
            cr = cr.masked_fill(~accepted[start:end], float("-inf"))
            rnew_tilde.scatter_reduce_(0, idx, cr, reduce="amax", include_self=True)
    rnew = torch.minimum(rnew_tilde, dnew)

    new_mapping = mapping_after_plan(state.original_to_state, plan, accepted)
    return HeadState(
        keys=knew.contiguous(),
        counts=cnew_i.contiguous(),
        radii=rnew.float().contiguous(),
        diameters=dnew.float().contiguous(),
        history=hnew.float().contiguous(),
        positions=pm.contiguous(),
        original_to_state=new_mapping,
    )


def compressed_output_from_mapping(
    attention_probs: torch.Tensor,
    values: torch.Tensor,
    prefix_len: int,
    mapping: np.ndarray,
    prob_eps: float,
) -> torch.Tensor:
    """Count-mean K/V + log(C) output implied by a partition of original prefix tokens.

    This is the same post-hoc attention-mass construction used by motivation_experiments,
    generalized to multi-round partitions. It evaluates the ideal arithmetic/count mean
    represented by the partition; tiny runtime dtype-rounding effects are intentionally
    outside this diagnosis.
    """
    a = attention_probs.float().cpu()
    v = values.float().cpu()
    owner = torch.as_tensor(mapping, dtype=torch.long)
    if owner.numel() != prefix_len:
        raise ValueError(f"mapping length {owner.numel()} != prefix_len {prefix_len}")
    live = owner >= 0
    if not bool(live.any()):
        raise RuntimeError("Counterfactual discarded the entire prefix.")
    live_pos = live.nonzero(as_tuple=False).flatten()
    live_owner = owner.index_select(0, live_pos)
    groups = int(live_owner.max().item()) + 1

    counts = torch.zeros(groups, dtype=torch.float32)
    counts.index_add_(0, live_owner, torch.ones_like(live_owner, dtype=torch.float32))
    value_sum = torch.zeros((groups, int(v.shape[-1])), dtype=torch.float32)
    value_sum.index_add_(0, live_owner, v[:prefix_len].index_select(0, live_pos))
    mean_value = value_sum / counts.clamp_min(1.0).unsqueeze(-1)

    logp = torch.log(torch.clamp(a[..., :prefix_len], min=float(prob_eps)))
    logp_live = logp.index_select(-1, live_pos).permute(2, 0, 1).contiguous()
    logp_sum = torch.zeros((groups, int(a.shape[0]), int(a.shape[1])), dtype=torch.float32)
    logp_sum.index_add_(0, live_owner, logp_live)
    merged_mass_g = counts.view(-1, 1, 1) * torch.exp(logp_sum / counts.clamp_min(1.0).view(-1, 1, 1))
    merged_mass = merged_mass_g.permute(1, 2, 0).contiguous()
    num = torch.einsum("rqg,gd->rqd", merged_mass, mean_value)
    denom = merged_mass.sum(dim=-1)

    if prefix_len < int(a.shape[-1]):
        tail_a = a[..., prefix_len:]
        tail_v = v[prefix_len:]
        num = num + torch.matmul(tail_a, tail_v)
        denom = denom + tail_a.sum(dim=-1)
    return num / denom.clamp_min(float(prob_eps)).unsqueeze(-1)


def counterfactual_error(
    attention_probs: torch.Tensor,
    values: torch.Tensor,
    prefix_len: int,
    mapping: np.ndarray,
    prob_eps: float,
    output_norm_eps: float,
) -> Tuple[float, float, int]:
    a = attention_probs.float().cpu()
    v = values.float().cpu()
    original = torch.matmul(a, v)
    approx = compressed_output_from_mapping(a, v, prefix_len, mapping, prob_eps)
    denom = torch.linalg.vector_norm(original, dim=-1).clamp_min(float(output_norm_eps))
    err = torch.linalg.vector_norm(approx - original, dim=-1) / denom
    return float(err.mean().item()), float(torch.quantile(err.flatten(), 0.90).item()), int(err.numel())


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    y = x.detach().float().flatten().cpu()
    return {
        "mean": float(y.mean().item()),
        "p10": float(torch.quantile(y, 0.10).item()),
        "p50": float(torch.quantile(y, 0.50).item()),
        "p90": float(torch.quantile(y, 0.90).item()),
        "max": float(y.max().item()),
    }


def min_gamma_to_any_anchor(
    state: HeadState,
    plan,
    delta: float,
    chunk_size: int,
) -> Dict[str, float]:
    keep = plan.keep_idx[0]
    evict = plan.evict_idx[0]
    if evict.numel() == 0:
        return {
            "min_gamma_over_delta_mean": 0.0,
            "min_gamma_over_delta_p50": 0.0,
            "min_gamma_over_delta_p90": 0.0,
            "min_gamma_over_delta_max": 0.0,
            "min_gamma_excess_mean": 0.0,
            "fraction_no_compatible_anchor_by_min_gamma": 0.0,
        }
    rhat = effective_radius(state.counts.float(), state.radii, state.diameters)
    kk = state.keys.index_select(0, keep).float()
    ek = state.keys.index_select(0, evict).float()
    rr = rhat.index_select(0, keep).float()
    er = rhat.index_select(0, evict).float()
    rd = state.diameters.index_select(0, keep).float()
    ed = state.diameters.index_select(0, evict).float()
    mins: List[torch.Tensor] = []
    n = int(evict.numel())
    chunk = n if chunk_size <= 0 else min(max(int(chunk_size), 1), n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist = torch.cdist(ek[start:end], kk, p=2)
        gamma = torch.maximum(
            torch.maximum(ed[start:end].unsqueeze(1), rd.unsqueeze(0)),
            er[start:end].unsqueeze(1) + dist + rr.unsqueeze(0),
        )
        mins.append(gamma.min(dim=1).values)
    ratio = torch.cat(mins) / max(float(delta), 1e-12)
    s = tensor_stats(ratio)
    return {
        "min_gamma_over_delta_mean": s["mean"],
        "min_gamma_over_delta_p50": s["p50"],
        "min_gamma_over_delta_p90": s["p90"],
        "min_gamma_over_delta_max": s["max"],
        "min_gamma_excess_mean": float(torch.clamp(ratio - 1.0, min=0.0).mean().item()),
        "fraction_no_compatible_anchor_by_min_gamma": float((ratio > 1.0).float().mean().item()),
    }


def plan_diagnostics(
    state: HeadState,
    plan,
    geometry_chunk_size: int,
    prefix: str = "",
    compute_min_gamma: bool = True,
    stats_reset_certificates: bool = False,
) -> Dict[str, float]:
    delta = float(plan.delta[0, 0].item())
    evict = plan.evict_idx[0]
    keep = plan.keep_idx[0]
    route = plan.route_compatible[0]
    accepted = plan.accepted[0]
    n_evict = int(evict.numel())
    routed = int(route.sum().item())
    selected = int(accepted.sum().item())
    pool = plan.candidate_pool_sizes[0].float()
    possible_edges = float((pool * (pool - 1.0) / 2.0).sum().item())
    edges = float(plan.conflict_edges_per_anchor[0].sum().item())
    candidate_vertices = float(pool.sum().item())

    evict_counts = state.counts.index_select(0, evict).float()
    routed_mass = float((evict_counts * route.float()).sum().item())
    accepted_mass = float((evict_counts * accepted.float()).sum().item())
    evicted_mass = float(evict_counts.sum().item())

    stats_radii = torch.zeros_like(state.radii) if stats_reset_certificates else state.radii
    stats_diameters = torch.zeros_like(state.diameters) if stats_reset_certificates else state.diameters
    rhat = effective_radius(state.counts.float(), stats_radii, stats_diameters)
    rratio = rhat / max(delta, 1e-12)
    dratio = stats_diameters.float() / max(delta, 1e-12)
    rs = tensor_stats(rratio)
    ds = tensor_stats(dratio)

    rr = rhat.index_select(0, keep) if keep.numel() else torch.empty(0, device=state.keys.device)
    er = rhat.index_select(0, evict) if evict.numel() else torch.empty(0, device=state.keys.device)
    if er.numel() and rr.numel():
        best_radius_only_slack = (delta - er - rr.min()) / max(delta, 1e-12)
        bss = tensor_stats(best_radius_only_slack)
        radius_impossible = float((best_radius_only_slack <= 0.0).float().mean().item())
    else:
        bss = tensor_stats(torch.empty(0))
        radius_impossible = 0.0

    if routed:
        best = plan.best_idx[0][route]
        routed_er = er[route]
        routed_rr = rr.index_select(0, best)
        actual_slack = (delta - routed_er - routed_rr) / max(delta, 1e-12)
        ass = tensor_stats(actual_slack)
    else:
        ass = tensor_stats(torch.empty(0))

    out = {
        f"{prefix}delta": delta,
        f"{prefix}reference_scale": float(plan.scale[0, 0].item()),
        f"{prefix}evicted_entries": n_evict,
        f"{prefix}routed_entries": routed,
        f"{prefix}accepted_entries": selected,
        f"{prefix}candidate_admission_rate": float(routed / max(n_evict, 1)),
        f"{prefix}mis_survival_given_routed": float(selected / max(routed, 1)),
        f"{prefix}overall_recovery_rate": float(selected / max(n_evict, 1)),
        f"{prefix}route_rejection_fraction": float((n_evict - routed) / max(n_evict, 1)),
        f"{prefix}mis_rejection_fraction": float((routed - selected) / max(n_evict, 1)),
        f"{prefix}evicted_count_mass": evicted_mass,
        f"{prefix}routed_count_mass": routed_mass,
        f"{prefix}accepted_count_mass": accepted_mass,
        f"{prefix}candidate_admission_count_mass_rate": routed_mass / max(evicted_mass, 1e-12),
        f"{prefix}recovery_count_mass_rate": accepted_mass / max(evicted_mass, 1e-12),
        f"{prefix}conflict_edges": edges,
        f"{prefix}possible_conflict_edges": possible_edges,
        f"{prefix}conflict_graph_density": edges / max(possible_edges, 1.0),
        f"{prefix}mean_conflict_degree": (2.0 * edges / max(candidate_vertices, 1.0)),
        f"{prefix}nonempty_candidate_pools": int((pool > 0).sum().item()),
        f"{prefix}max_candidate_pool_size": int(pool.max().item()) if pool.numel() else 0,
        f"{prefix}max_mis_rounds": int(plan.mis_rounds_per_anchor[0].max().item()) if plan.mis_rounds_per_anchor.numel() else 0,
        f"{prefix}rhat_over_delta_mean": rs["mean"],
        f"{prefix}rhat_over_delta_p50": rs["p50"],
        f"{prefix}rhat_over_delta_p90": rs["p90"],
        f"{prefix}rhat_over_delta_max": rs["max"],
        f"{prefix}diameter_over_delta_mean": ds["mean"],
        f"{prefix}diameter_over_delta_p50": ds["p50"],
        f"{prefix}diameter_over_delta_p90": ds["p90"],
        f"{prefix}diameter_over_delta_max": ds["max"],
        f"{prefix}diameter_at_or_above_delta_fraction": float((dratio >= 1.0).float().mean().item()),
        f"{prefix}best_radius_only_slack_mean": bss["mean"],
        f"{prefix}best_radius_only_slack_p10": bss["p10"],
        f"{prefix}best_radius_only_slack_p50": bss["p50"],
        f"{prefix}radius_only_impossible_fraction": radius_impossible,
        f"{prefix}routed_pair_slack_mean": ass["mean"],
        f"{prefix}routed_pair_slack_p10": ass["p10"],
        f"{prefix}routed_pair_slack_p50": ass["p50"],
    }
    if compute_min_gamma:
        out.update({f"{prefix}{k}": v for k, v in min_gamma_to_any_anchor(state, plan, delta, geometry_chunk_size).items()})
    return out


def evaluate_plan_counterfactuals(
    state: HeadState,
    plan,
    fresh_plan,
    refreshed_scale_plan,
    attention_probs: torch.Tensor,
    values: torch.Tensor,
    prefix_len: int,
    prob_eps: float,
    output_norm_eps: float,
) -> Dict[str, float]:
    masks = {
        "coherekv": plan.accepted[0],
        "compatible_merge_all": plan.route_compatible[0],
        "eviction_only": torch.zeros_like(plan.accepted[0]),
        "fresh_certificate": fresh_plan.accepted[0],
        "refreshed_scale": refreshed_scale_plan.accepted[0],
    }
    plans = {
        "coherekv": plan,
        "compatible_merge_all": plan,
        "eviction_only": plan,
        "fresh_certificate": fresh_plan,
        "refreshed_scale": refreshed_scale_plan,
    }
    out: Dict[str, float] = {}
    qobs = 0
    for name in masks:
        mapping = mapping_after_plan(state.original_to_state, plans[name], masks[name])
        mean, p90, n = counterfactual_error(
            attention_probs, values, prefix_len, mapping, prob_eps, output_norm_eps
        )
        out[f"mean_{name}_output_rel_error"] = mean
        out[f"p90_{name}_output_rel_error"] = p90
        qobs = n
    out["query_observations"] = qobs
    ev = out["mean_eviction_only_output_rel_error"]
    full = out["mean_coherekv_output_rel_error"]
    ma = out["mean_compatible_merge_all_output_rel_error"]
    out["coherekv_error_reduction_vs_eviction"] = (ev - full) / max(ev, output_norm_eps)
    out["merge_all_error_reduction_vs_eviction"] = (ev - ma) / max(ev, output_norm_eps)
    out["coherekv_absolute_error_gap_to_eviction"] = abs(ev - full)
    return out


def single_round_row(
    state: HeadState,
    keep_len: int,
    frozen_scale: torch.Tensor,
    lam: float,
    attention_probs: torch.Tensor,
    values: torch.Tensor,
    prefix_len: int,
    args,
) -> Dict[str, float]:
    plan = build_plan(
        state, keep_len, frozen_scale, lam,
        args.keepkv_sink_size, args.keepkv_recent_protect_size,
        args.keepkv_recency_beta, args.distance_chunk_size,
        reset_certificates=False,
    )
    # In the first round R=D=0, so the two diagnostic planners are identical to the original.
    mapping_full = mapping_after_plan(state.original_to_state, plan, plan.accepted[0])
    mapping_merge_all = mapping_after_plan(state.original_to_state, plan, plan.route_compatible[0])
    mapping_evict = mapping_after_plan(state.original_to_state, plan, torch.zeros_like(plan.accepted[0]))
    out = plan_diagnostics(state, plan, args.geometry_chunk_size)
    qobs = 0
    for name, mapping in (
        ("coherekv", mapping_full),
        ("compatible_merge_all", mapping_merge_all),
        ("eviction_only", mapping_evict),
    ):
        mean, p90, n = counterfactual_error(
            attention_probs, values, prefix_len, mapping,
            args.attention_prob_eps, args.output_norm_eps,
        )
        out[f"mean_{name}_output_rel_error"] = mean
        out[f"p90_{name}_output_rel_error"] = p90
        qobs = n
    out["query_observations"] = qobs
    ev = out["mean_eviction_only_output_rel_error"]
    out["coherekv_error_reduction_vs_eviction"] = (
        ev - out["mean_coherekv_output_rel_error"]
    ) / max(ev, args.output_norm_eps)
    return out


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def weighted_mean(rows: Sequence[Dict], key: str, weight_key: Optional[str] = None) -> float:
    vals = []
    weights = []
    for r in rows:
        if key not in r:
            continue
        vals.append(float(r[key]))
        weights.append(float(r.get(weight_key, 1.0)) if weight_key else 1.0)
    if not vals:
        return 0.0
    v = np.asarray(vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(v * w) / max(float(np.sum(w)), 1e-20))


def aggregate_rows(rows: Sequence[Dict], group_keys: Sequence[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    out: List[Dict] = []
    additive_suffixes = {
        "evicted_entries", "routed_entries", "accepted_entries",
        "evicted_count_mass", "routed_count_mass", "accepted_count_mass",
        "conflict_edges", "possible_conflict_edges", "query_observations",
    }
    for key, sub in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        # Preserve useful metadata shared by the group.
        for k in ("input_length", "retained_length", "cumulative_retained_fraction"):
            if k in sub[0]:
                row[k] = weighted_mean(sub, k)
        all_fields = sorted(set().union(*(r.keys() for r in sub)))
        for field in all_fields:
            if field in group_keys or field in {"sequence_idx", "layer_idx", "kv_head_idx"}:
                continue
            if any(field == suffix or field.endswith("_" + suffix) for suffix in additive_suffixes):
                row[field] = float(sum(float(r.get(field, 0.0)) for r in sub))
            elif field.startswith("mean_") or field.startswith("p90_") or field.endswith("output_rel_error"):
                row[field] = weighted_mean(sub, field, "query_observations")
            elif isinstance(sub[0].get(field), (int, float, np.integer, np.floating)):
                row[field] = weighted_mean(sub, field)
        # Recompute exact aggregate rates where possible, including diagnostic-shadow prefixes.
        for prefix in ("", "original_", "fresh_cert_", "refreshed_scale_"):
            ev_key = prefix + "evicted_entries"
            if ev_key in row:
                ev = max(row[ev_key], 1.0)
                routed = row.get(prefix + "routed_entries", 0.0)
                acc = row.get(prefix + "accepted_entries", 0.0)
                row[prefix + "candidate_admission_rate"] = routed / ev
                row[prefix + "overall_recovery_rate"] = acc / ev
                row[prefix + "route_rejection_fraction"] = (ev - routed) / ev
                row[prefix + "mis_rejection_fraction"] = (routed - acc) / ev
                row[prefix + "mis_survival_given_routed"] = acc / max(routed, 1.0)
            pe_key = prefix + "possible_conflict_edges"
            if pe_key in row:
                row[prefix + "conflict_graph_density"] = row.get(prefix + "conflict_edges", 0.0) / max(row[pe_key], 1.0)
            em_key = prefix + "evicted_count_mass"
            if em_key in row:
                em = max(row[em_key], 1e-12)
                row[prefix + "candidate_admission_count_mass_rate"] = row.get(prefix + "routed_count_mass", 0.0) / em
                row[prefix + "recovery_count_mass_rate"] = row.get(prefix + "accepted_count_mass", 0.0) / em
        if "mean_eviction_only_output_rel_error" in row and "mean_coherekv_output_rel_error" in row:
            ev_err = row["mean_eviction_only_output_rel_error"]
            row["coherekv_error_reduction_vs_eviction"] = (
                ev_err - row["mean_coherekv_output_rel_error"]
            ) / max(ev_err, 1e-20)
            if "mean_compatible_merge_all_output_rel_error" in row:
                row["merge_all_error_reduction_vs_eviction"] = (
                    ev_err - row["mean_compatible_merge_all_output_rel_error"]
                ) / max(ev_err, 1e-20)
        out.append(row)
    return out


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(parse_float_csv(args.single_round_budgets)))
    lambdas = sorted(set(parse_float_csv(args.lambda_values)))
    for rho in budgets:
        if not 0.0 < rho < 1.0:
            raise ValueError(f"single-round budget must be in (0,1), got {rho}")
    for lam in lambdas:
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"lambda must be finite and >0, got {lam}")
    if not 0.0 < args.round_keep_ratio < 1.0:
        raise ValueError("round_keep_ratio must be in (0,1)")
    if args.repeated_rounds < 1:
        raise ValueError("repeated_rounds must be >=1")
    if str(args.attn_implementation).lower() != "eager":
        raise ValueError("This experiment requires eager attention for future-query probes.")

    set_seed(0)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_kwargs = dict(
        config=config,
        torch_dtype=parse_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if str(args.device_map).lower() != "none":
        model_kwargs["device_map"] = args.device_map
    print(f"Loading model: {args.model}", flush=True)
    print(f"Transformers version: {transformers.__version__}", flush=True)
    print(f"Original CohereKV lambda sweep: {lambdas}", flush=True)
    print(
        "Diagnosis: production C/R/D/Gamma routing + conflict graph + Parallel MIS; "
        "compare matched Eviction-Only, route-compatible Merge-All, fresh-certificate shadow, "
        "and refreshed-S_ref shadow. Repeated rounds advance only the original CohereKV trajectory.",
        flush=True,
    )
    model = AutoModel.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if str(args.device_map).lower() == "none":
        requested = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        model.to(requested)
    input_device = first_execution_device(model, args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    data = load_token_stream(tokenizer, args.dataset)
    block_len = int(args.model_seqlen) + int(args.probe_tokens)
    n_seq = min(int(args.max_eval_sequences), int(data.shape[1]) // block_len)
    if n_seq < 1:
        raise RuntimeError("Insufficient dataset tokens for prefix + probe.")

    single_rows: List[Dict] = []
    round_rows: List[Dict] = []
    selected_layers: Optional[List[int]] = None
    selected_heads: Optional[List[int]] = None
    t0 = time.time()

    for seq_idx in range(n_seq):
        s0 = seq_idx * block_len
        prefix_end = s0 + args.model_seqlen
        probe_end = prefix_end + args.probe_tokens
        prefix_ids = data[:, s0:prefix_end]
        probe_ids = data[:, prefix_end:probe_end]
        print(f"\n[SEQUENCE {seq_idx}] prefix [{s0},{prefix_end}), probe [{prefix_end},{probe_end})", flush=True)

        past, selector_scores_all, selector_log_scales = build_full_cache_with_keepkv_scores(
            model, prefix_ids, input_device, args.prefill_chunk_size,
            args.keepkv_score_window, config.model_type,
        )
        prefix_keys_all, _ = cache_key_value_layers(past)
        probe_out = run_probe_queries(model, past, probe_ids, input_device)
        _, probe_values_all = cache_key_value_layers(probe_out.past_key_values)

        n_layers = len(prefix_keys_all)
        if selected_layers is None:
            selected_layers = resolve_selection(args.layer_selection, n_layers)
        for layer_idx in selected_layers:
            raw_layer_keys = prefix_keys_all[layer_idx][0]  # [Hkv,L,d]
            hkv = int(raw_layer_keys.shape[0])
            if selected_heads is None:
                selected_heads = resolve_selection(args.kv_head_selection, hkv)
            score_layer = selector_scores_all[layer_idx][0]  # [Hkv,W,L]
            attn_layer = probe_out.attentions[layer_idx][0].detach().float().cpu()  # [Hq,Q,L+probe]
            value_layer = probe_values_all[layer_idx][0].detach().float().cpu()      # [Hkv,L+probe,d]
            hq = int(attn_layer.shape[0])
            if hq % hkv != 0:
                raise RuntimeError(f"Layer {layer_idx}: Hq={hq} not divisible by Hkv={hkv}")
            nrep = hq // hkv

            for head_idx in selected_heads:
                if args.progress_every_head:
                    print(f"  [HEAD] layer={layer_idx} kv_head={head_idx}", flush=True)
                raw_keys = raw_layer_keys[head_idx].contiguous()
                score_history = score_layer[head_idx].to(device=raw_keys.device, dtype=torch.float32).contiguous()
                base_state = initial_head_state(raw_keys, score_history)
                scale0 = frozen_reference_scale(raw_keys)
                q0, q1 = head_idx * nrep, (head_idx + 1) * nrep
                attn_h = attn_layer[q0:q1]
                values_h = value_layer[head_idx]
                prefix_len = int(raw_keys.shape[0])

                # A. First-round budget sweep: isolates same-Delta routing/filtering before any certificate accumulation.
                for lam in lambdas:
                    for rho in budgets:
                        keep_len = max(1, min(prefix_len - 1, int(round(rho * prefix_len))))
                        d = single_round_row(
                            base_state, keep_len, scale0, lam,
                            attn_h, values_h, prefix_len, args,
                        )
                        d.update({
                            "sequence_idx": seq_idx,
                            "layer_idx": layer_idx,
                            "kv_head_idx": head_idx,
                            "lambda_mymethod": lam,
                            "anchor_budget": rho,
                            "input_length": prefix_len,
                            "retained_length": keep_len,
                        })
                        single_rows.append(d)

                # B. Controlled repeated-compression trajectory: advance production original CohereKV only.
                for lam in lambdas:
                    state = initial_head_state(raw_keys, score_history)
                    for round_idx in range(1, int(args.repeated_rounds) + 1):
                        cur_len = int(state.keys.shape[0])
                        if cur_len <= max(int(args.min_physical_entries), 2):
                            break
                        keep_len = int(round(cur_len * float(args.round_keep_ratio)))
                        keep_len = max(int(args.min_physical_entries), keep_len)
                        keep_len = min(cur_len - 1, keep_len)
                        if keep_len >= cur_len:
                            break

                        plan = build_plan(
                            state, keep_len, scale0, lam,
                            args.keepkv_sink_size, args.keepkv_recent_protect_size,
                            args.keepkv_recency_beta, args.distance_chunk_size,
                            reset_certificates=False,
                        )
                        fresh_plan = build_plan(
                            state, keep_len, scale0, lam,
                            args.keepkv_sink_size, args.keepkv_recent_protect_size,
                            args.keepkv_recency_beta, args.distance_chunk_size,
                            reset_certificates=True,
                        )
                        scale_current = current_reference_scale(state.keys)
                        refreshed_plan = build_plan(
                            state, keep_len, scale_current, lam,
                            args.keepkv_sink_size, args.keepkv_recent_protect_size,
                            args.keepkv_recency_beta, args.distance_chunk_size,
                            reset_certificates=False,
                        )

                        row: Dict[str, float] = {
                            "sequence_idx": seq_idx,
                            "layer_idx": layer_idx,
                            "kv_head_idx": head_idx,
                            "lambda_mymethod": lam,
                            "round_idx": round_idx,
                            "input_length": cur_len,
                            "retained_length": keep_len,
                            "cumulative_retained_fraction": keep_len / float(prefix_len),
                            "frozen_reference_scale": float(scale0.item()),
                            "current_reference_scale": float(scale_current.item()),
                            "current_scale_over_frozen_scale": float(scale_current.item() / max(float(scale0.item()), 1e-12)),
                        }
                        row.update(plan_diagnostics(state, plan, args.geometry_chunk_size, prefix="original_"))
                        row.update(plan_diagnostics(state, fresh_plan, args.geometry_chunk_size, prefix="fresh_cert_", compute_min_gamma=False, stats_reset_certificates=True))
                        row.update(plan_diagnostics(state, refreshed_plan, args.geometry_chunk_size, prefix="refreshed_scale_", compute_min_gamma=False))
                        row.update(evaluate_plan_counterfactuals(
                            state, plan, fresh_plan, refreshed_plan,
                            attn_h, values_h, prefix_len,
                            args.attention_prob_eps, args.output_norm_eps,
                        ))

                        state = apply_plan(state, plan, args.distance_chunk_size)
                        post_delta = max(float(plan.delta[0, 0].item()), 1e-12)
                        post_rhat = effective_radius(state.counts.float(), state.radii, state.diameters)
                        prs = tensor_stats(post_rhat / post_delta)
                        pds = tensor_stats(state.diameters / post_delta)
                        row.update({
                            "post_rhat_over_delta_mean": prs["mean"],
                            "post_rhat_over_delta_p50": prs["p50"],
                            "post_rhat_over_delta_p90": prs["p90"],
                            "post_rhat_over_delta_max": prs["max"],
                            "post_diameter_over_delta_mean": pds["mean"],
                            "post_diameter_over_delta_p50": pds["p50"],
                            "post_diameter_over_delta_p90": pds["p90"],
                            "post_diameter_over_delta_max": pds["max"],
                            "post_live_original_token_fraction": float(np.mean(state.original_to_state >= 0)),
                        })
                        round_rows.append(row)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del probe_out, past, prefix_keys_all, probe_values_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    single_overall = aggregate_rows(single_rows, ["lambda_mymethod", "anchor_budget"])
    round_overall = aggregate_rows(round_rows, ["lambda_mymethod", "round_idx"])
    write_csv(out_dir / "single_round_budget_diagnosis_by_head.csv", single_rows)
    write_csv(out_dir / "single_round_budget_diagnosis_overall.csv", single_overall)
    write_csv(out_dir / "repeated_round_diagnosis_by_head.csv", round_rows)
    write_csv(out_dir / "repeated_round_diagnosis_overall.csv", round_overall)

    run_config = {
        "experiment": "original_coherekv_diagnosis_v1",
        "model": args.model,
        "dataset": args.dataset,
        "model_seqlen": args.model_seqlen,
        "probe_tokens": args.probe_tokens,
        "max_eval_sequences": args.max_eval_sequences,
        "single_round_budgets": budgets,
        "lambda_values": lambdas,
        "repeated_rounds": args.repeated_rounds,
        "round_keep_ratio": args.round_keep_ratio,
        "min_physical_entries": args.min_physical_entries,
        "layer_selection": args.layer_selection,
        "kv_head_selection": args.kv_head_selection,
        "keepkv_sink_size": args.keepkv_sink_size,
        "keepkv_recent_protect_size": args.keepkv_recent_protect_size,
        "keepkv_score_window": args.keepkv_score_window,
        "keepkv_recency_beta": args.keepkv_recency_beta,
        "distance_chunk_size": args.distance_chunk_size,
        "geometry_chunk_size": args.geometry_chunk_size,
        "attention_metric_note": (
            "Post-hoc count-mean K/V + log(C) attention-output error generalized to multi-round partitions; "
            "used only for diagnosis, never for selection."
        ),
        "repeated_round_note": (
            "Controlled repeated compression of one fixed prefix. Original CohereKV state is advanced each round; "
            "fresh-certificate and refreshed-S_ref variants are same-state shadows and do not alter the trajectory."
        ),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"\nWrote diagnosis results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
