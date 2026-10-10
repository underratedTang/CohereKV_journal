#!/usr/bin/env python3
from __future__ import annotations

"""Figure 3-left: focused single-round Original vs Modified CohereKV.

Each observation starts from the same uncompressed prefix and exactly one natural
next-token query. Only one compression decision is made. For every selected layer,
all KV/query-head outputs are reconstructed, concatenated in model order, passed
through that layer's real attention ``o_proj``, and evaluated with the same
whole-layer relative-error metric used by Figure 2. The plotted variants are:

* ``eviction_only``: use the same retained partition but drop all evicted sources;
* ``merge_all``: Euclidean K routing, accept every routed source;
* ``original_coherekv``: repository historical R/D planner, first round;
* ``modified_coherekv``: Euclidean K routing + current-round K-only conflict +
  deterministic parallel MIS.

Merge-All and Modified share one partition and one routing computation per head;
Modified conflict/MIS is the tensorized batched implementation from
``star_group_planner.py``. ``kv_head_selection`` is retained only for launch-script
compatibility: whole-layer Figure 3-left necessarily evaluates every head.
"""

import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "diagnosis_experiments"
TASK_EVAL_ROOT = PROJECT_ROOT / "task_evaluation"
HERE = Path(__file__).resolve().parent
for p in (DIAG_ROOT, TASK_EVAL_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_original_coherekv import (  # noqa: E402
    HeadState,
    apply_plan,
    build_full_cache_with_keepkv_scores,
    build_plan,
    cache_key_value_layers,
    first_execution_device,
    frozen_reference_scale,
    initial_head_state,
    load_token_stream,
    set_seed,
    state_importance,
)
from experiment_utils import resolve_selection  # noqa: E402
from random_query_probes import collect_independent_query_states, _resolve_attention_module  # noqa: E402
from star_group_planner import (  # noqa: E402
    fixed_partition,
    precompute_routing_scores,
    route_threshold,
    route_view,
    select_conflict_free_routed_batched,
)
from whole_cache_attention import (  # noqa: E402
    WholePrefixAttentionEvaluator, whole_layer_attention_relative_error,
)
from calibration_scale import load_calibration_scales, scale_for, validate_calibration_scales  # noqa: E402

VARIANTS = ("eviction_only", "merge_all", "original_coherekv", "modified_coherekv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused single-round CohereKV comparison")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--model_seqlen", type=int, default=4096)
    p.add_argument("--max_eval_prefixes", type=int, default=4)
    p.add_argument("--prefix_stride", type=int, default=0)
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_map", default="none")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--prefill_chunk_size", type=int, default=256)

    p.add_argument("--keep_ratio", type=float, default=0.20)
    p.add_argument("--calibration_scale_file", required=True,
                   help="JSON produced by calibrate_layer_head_scales.py; frozen S_{layer,head} for Merge-All/Modified")
    p.add_argument("--original_lambda", type=float, default=0.40)
    p.add_argument("--route_lambda", type=float, default=0.40)
    p.add_argument("--conflict_lambda", type=float, default=0.10)
    p.add_argument("--enable_hparam_sweep", action="store_true",
                   help="run a compact one-factor-at-a-time sweep around the primary configuration")
    p.add_argument("--original_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--route_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--conflict_lambda_grid", default="0.05,0.10,0.20")
    p.add_argument("--keepkv_sink_size", type=int, default=4)
    p.add_argument("--keepkv_recent_protect_size", type=int, default=128)
    p.add_argument("--keepkv_score_window", type=int, default=128)
    p.add_argument("--keepkv_recency_beta", type=float, default=1.0)
    p.add_argument("--distance_chunk_size", type=int, default=1024)
    p.add_argument("--layer_selection", default="0,8,16,24,31")
    p.add_argument("--kv_head_selection", default="0,1,2,3,4,5,6,7",
                   help="retained for CLI compatibility; Figure 3-left whole-layer evaluation always uses all KV/query heads")
    p.add_argument("--output_norm_eps", type=float, default=1e-20)
    p.add_argument("--progress_every_head", action="store_true")
    p.add_argument("--output_dir", required=True)
    return p


def parse_dtype(name: str):
    table = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(name)
    return table[key]






def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def build_ofat_configs(args) -> List[Dict]:
    """Primary point plus one-factor-at-a-time sweep; never a Cartesian product."""
    base = {
        "config_id": "primary", "is_primary": 1,
        "original_lambda": float(args.original_lambda),
        "route_lambda": float(args.route_lambda),
        "conflict_lambda": float(args.conflict_lambda),
    }
    out = [base]
    if not args.enable_hparam_sweep:
        return out
    seen = {(base["original_lambda"], base["route_lambda"], base["conflict_lambda"])}
    def add(kind: str, value: float, **kw):
        c = dict(base); c.update(kw); c["config_id"] = f"{kind}_{value:g}"; c["is_primary"] = 0
        key = (c["original_lambda"], c["route_lambda"], c["conflict_lambda"])
        if key not in seen:
            seen.add(key); out.append(c)
    for x in parse_float_list(args.original_lambda_grid):
        add("original", x, original_lambda=float(x))
    for x in parse_float_list(args.route_lambda_grid):
        add("route", x, route_lambda=float(x))
    for x in parse_float_list(args.conflict_lambda_grid):
        add("conflict", x, conflict_lambda=float(x))
    return out


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def clone_state(s: HeadState) -> HeadState:
    return HeadState(
        keys=s.keys.clone(), counts=s.counts.clone(), radii=s.radii.clone(), diameters=s.diameters.clone(),
        history=s.history.clone(), positions=s.positions.clone(), original_to_state=s.original_to_state.copy(),
    )


def strip_certificates(s: HeadState) -> HeadState:
    s.radii = torch.zeros_like(s.radii)
    s.diameters = torch.zeros_like(s.diameters)
    return s


def _local_plan(*, keep_idx, evict_idx, best, routed, accepted, anchor_gamma,
                conflict_edges, route_lambda: float, k_scale: float):
    m = int(keep_idx.shape[1])
    return SimpleNamespace(
        keep_idx=keep_idx,
        evict_idx=evict_idx,
        best_idx=best.unsqueeze(0),
        route_compatible=routed.unsqueeze(0),
        accepted=accepted.unsqueeze(0),
        conflict_edges_per_anchor=conflict_edges,
        anchor_gamma=anchor_gamma.unsqueeze(0),
        anchor_k_gamma=anchor_gamma.unsqueeze(0),
        group_gamma_max=torch.zeros((1, m), dtype=torch.float32, device=keep_idx.device),
        delta=torch.tensor([[route_lambda * k_scale]], dtype=torch.float32, device=keep_idx.device),
    )


@torch.inference_mode()
def build_local_plans(state: HeadState, *, keep_len: int, k_scale: float,
                      route_lambda: float, conflict_lambda: float,
                      sink_size: int, recent_window: int, recency_beta: float,
                      distance_chunk_size: int):
    """Build Merge-All + Modified from one shared partition/routing pass."""
    importance_1d = state_importance(state, recency_beta)
    keep_idx, evict_idx = fixed_partition(
        importance=importance_1d.unsqueeze(0), positions=state.positions.unsqueeze(0),
        keep_len=keep_len, sink_size=sink_size, recent_window=recent_window,
    )
    scores = precompute_routing_scores(
        keys=state.keys.unsqueeze(0), keep_idx=keep_idx, evict_idx=evict_idx,
        chunk_size=distance_chunk_size,
    )
    view = route_view(
        scores=scores, metric="euclidean",
        threshold=route_threshold("euclidean", route_lambda, k_scale),
    )
    routed = view.compatible.clone()
    best = view.best_idx.clone().long()
    anchor_gamma = (view.assigned_distance / max(k_scale, 1e-12)).float()
    ev = evict_idx[0].long()
    m = int(keep_idx.shape[1])
    zeros = torch.zeros((1, m), dtype=torch.int32, device=state.keys.device)

    eviction_only = _local_plan(
        keep_idx=keep_idx, evict_idx=evict_idx, best=best, routed=routed,
        accepted=torch.zeros_like(routed), anchor_gamma=anchor_gamma,
        conflict_edges=zeros.clone(), route_lambda=route_lambda, k_scale=k_scale,
    )
    merge_all = _local_plan(
        keep_idx=keep_idx, evict_idx=evict_idx, best=best, routed=routed,
        accepted=routed.clone(), anchor_gamma=anchor_gamma,
        conflict_edges=zeros.clone(), route_lambda=route_lambda, k_scale=k_scale,
    )

    evicted_keys = state.keys.index_select(0, ev).unsqueeze(0)
    mis = select_conflict_free_routed_batched(
        evicted_keys=evicted_keys,
        best_idx=best.unsqueeze(0),
        route_compatible=routed.unsqueeze(0),
        anchor_gamma=anchor_gamma.unsqueeze(0),
        physical_positions=state.positions.index_select(0, ev).unsqueeze(0),
        keep_len=m,
        k_scale=torch.tensor([k_scale], dtype=torch.float32, device=state.keys.device),
        conflict_lambda=conflict_lambda,
    )
    modified = _local_plan(
        keep_idx=keep_idx, evict_idx=evict_idx, best=best, routed=routed,
        accepted=mis.accepted[0], anchor_gamma=anchor_gamma,
        conflict_edges=mis.conflict_edges_per_anchor,
        route_lambda=route_lambda, k_scale=k_scale,
    )
    return eviction_only, merge_all, modified


def plan_stats(plan) -> Dict[str, float]:
    ev = plan.evict_idx[0].long()
    routed = plan.route_compatible[0].bool()
    accepted = plan.accepted[0].bool()
    routed_n = int(routed.sum().item())
    accepted_n = int(accepted.sum().item())
    return {
        "evicted_entries": int(ev.numel()),
        "routed_entries": routed_n,
        "accepted_entries": accepted_n,
        "accepted_entry_given_routed": float(accepted_n / max(routed_n, 1)),
        "conflict_edges": int(plan.conflict_edges_per_anchor.sum().item()) if hasattr(plan, "conflict_edges_per_anchor") else 0,
    }


def aggregate(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    config_ids = []
    for r in rows:
        cid = str(r.get("config_id", "primary"))
        if cid not in config_ids:
            config_ids.append(cid)
    for cid in config_ids:
        cr = [r for r in rows if str(r.get("config_id", "primary")) == cid]
        for variant in VARIANTS:
            rr = [r for r in cr if r["variant"] == variant]
            if not rr:
                continue
            out.append({
                "config_id": cid,
                "is_primary": int(rr[0].get("is_primary", 0)),
                "original_lambda": float(rr[0]["original_lambda"]),
                "route_lambda": float(rr[0]["route_lambda"]),
                "conflict_lambda": float(rr[0]["conflict_lambda"]),
                "variant": variant,
                "observations": len(rr),
                "output_rel_error": float(np.mean([r["output_rel_error"] for r in rr])),
                "output_rel_error_std": float(np.std([r["output_rel_error"] for r in rr], ddof=1)) if len(rr) > 1 else 0.0,
                "output_p90_error": float(np.mean([r["output_p90_error"] for r in rr])),
                "accepted_entry_given_routed": float(np.mean([r["accepted_entry_given_routed"] for r in rr])),
                "conflict_edges": float(np.mean([r["conflict_edges"] for r in rr])),
            })
    return out


def main() -> None:
    args = build_parser().parse_args()
    if str(args.attn_implementation).lower() != "eager":
        raise ValueError("requires eager attention")
    if not 0 < float(args.keep_ratio) < 1:
        raise ValueError("keep_ratio must be in (0,1)")
    configs = build_ofat_configs(args)

    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    calibration_scales = load_calibration_scales(args.calibration_scale_file, args.model)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    validate_calibration_scales(calibration_scales, config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code, padding_side="left",
    )
    model_kwargs = dict(
        config=config, torch_dtype=parse_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code, attn_implementation=args.attn_implementation,
    )
    if str(args.device_map).lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModel.from_pretrained(args.model, **model_kwargs); model.eval()
    if str(args.device_map).lower() == "none":
        model.to(torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"))
    input_device = first_execution_device(model, args.device)
    tokens = load_token_stream(tokenizer, args.dataset)

    stride = int(args.prefix_stride) if int(args.prefix_stride) > 0 else int(args.model_seqlen) + 1
    max_start = int(tokens.shape[1]) - int(args.model_seqlen) - 1
    starts = list(range(0, max_start + 1, stride))[: int(args.max_eval_prefixes)]
    if not starts:
        raise RuntimeError("insufficient dataset tokens")

    rows: List[Dict] = []
    selected_layers = None
    t0 = time.time()
    print("=== Figure 3-left: focused single-round CohereKV comparison ===", flush=True)
    print("one natural next-token Q per frozen prefix; one compression only", flush=True)
    print("metric=whole attention-layer output after real o_proj; all KV/query heads", flush=True)
    print(f"Modified/Merge-All frozen calibration scales S_lh shape={tuple(calibration_scales.shape)}", flush=True)
    print(f"configs={len(configs)}; sweep={'on' if args.enable_hparam_sweep else 'off'}", flush=True)

    for prefix_idx, s0 in enumerate(starts):
        prefix_ids = tokens[:, s0:s0 + int(args.model_seqlen)]
        next_id = int(tokens[0, s0 + int(args.model_seqlen)].item())
        past, selector_scores_all, _ = build_full_cache_with_keepkv_scores(
            model=model, input_ids=prefix_ids, device=input_device,
            chunk_size=args.prefill_chunk_size, score_window=args.keepkv_score_window,
            model_type=config.model_type,
        )
        prefix_keys_all, prefix_values_all = cache_key_value_layers(past)
        if selected_layers is None:
            selected_layers = resolve_selection(args.layer_selection, len(prefix_keys_all))
        probe = collect_independent_query_states(
            model=model, prefix_past=past, input_device=input_device,
            token_ids=[next_id], selected_layers=selected_layers,
            prefix_len=int(args.model_seqlen), store_dtype=torch.float16,
        )

        for layer_idx in selected_layers:
            raw_layer_keys = prefix_keys_all[layer_idx][0]
            raw_layer_values = prefix_values_all[layer_idx][0]
            score_layer = selector_scores_all[layer_idx][0]
            q_layer = probe.queries_by_layer[layer_idx]
            hkv = int(raw_layer_keys.shape[0]); hq = int(q_layer.shape[0])
            if hq % hkv != 0:
                raise RuntimeError("query heads not divisible by KV heads")
            nrep = hq // hkv
            num_probes = int(q_layer.shape[1])

            # Whole-layer evaluation requires every KV/query head.  Keep the CLI
            # head-selection argument only for backward compatibility with older
            # launch scripts; it is intentionally ignored here.
            full_head_outputs = [None] * hkv
            compressed_head_outputs = {
                (str(cfg_item["config_id"]), variant): [None] * hkv
                for cfg_item in configs for variant in VARIANTS
            }
            stat_acc = {
                (str(cfg_item["config_id"]), variant): {
                    "evicted_entries": 0,
                    "routed_entries": 0,
                    "accepted_entries": 0,
                    "conflict_edges": 0,
                }
                for cfg_item in configs for variant in VARIANTS
            }
            calibration_scale_values: List[float] = []
            original_scale_values: List[float] = []
            retained_lengths: List[int] = []
            input_lengths: List[int] = []

            for head_idx in range(hkv):
                if args.progress_every_head:
                    print(f"  prefix={prefix_idx} layer={layer_idx} head={head_idx}/{hkv}", flush=True)
                raw_keys = raw_layer_keys[head_idx].contiguous()
                raw_values = raw_layer_values[head_idx].contiguous()
                base = initial_head_state(
                    raw_keys, score_layer[head_idx].to(raw_keys.device, dtype=torch.float32).contiguous()
                )
                local = strip_certificates(clone_state(base))
                original_scale = float(frozen_reference_scale(raw_keys).item())
                k_scale = scale_for(calibration_scales, layer_idx, head_idx)
                keep_len = max(2, min(
                    int(raw_keys.shape[0]) - 1,
                    int(round(raw_keys.shape[0] * float(args.keep_ratio))),
                ))
                calibration_scale_values.append(float(k_scale))
                original_scale_values.append(float(original_scale))
                retained_lengths.append(int(keep_len))
                input_lengths.append(int(raw_keys.shape[0]))

                original_cache = {}
                local_cache = {}
                q0, q1 = head_idx * nrep, (head_idx + 1) * nrep
                evaluator = WholePrefixAttentionEvaluator(
                    q_layer[q0:q1], raw_keys, raw_values,
                    output_norm_eps=args.output_norm_eps,
                )
                full_head_outputs[head_idx] = evaluator.full_output.detach()

                for cfg_item in configs:
                    cid = str(cfg_item["config_id"])
                    ol = float(cfg_item["original_lambda"])
                    rl = float(cfg_item["route_lambda"])
                    cl = float(cfg_item["conflict_lambda"])
                    if ol not in original_cache:
                        original_cache[ol] = build_plan(
                            clone_state(base), keep_len,
                            torch.tensor(original_scale, device=raw_keys.device), ol,
                            args.keepkv_sink_size, args.keepkv_recent_protect_size,
                            args.keepkv_recency_beta, args.distance_chunk_size,
                            reset_certificates=False,
                        )
                    key = (rl, cl)
                    if key not in local_cache:
                        local_cache[key] = build_local_plans(
                            clone_state(local), keep_len=keep_len, k_scale=k_scale,
                            route_lambda=rl, conflict_lambda=cl,
                            sink_size=args.keepkv_sink_size,
                            recent_window=args.keepkv_recent_protect_size,
                            recency_beta=args.keepkv_recency_beta,
                            distance_chunk_size=args.distance_chunk_size,
                        )
                    eviction_only, merge_all, modified = local_cache[key]
                    plans = {
                        "eviction_only": eviction_only,
                        "merge_all": merge_all,
                        "original_coherekv": original_cache[ol],
                        "modified_coherekv": modified,
                    }
                    for variant in VARIANTS:
                        st = clone_state(base) if variant == "original_coherekv" else clone_state(local)
                        plan = plans[variant]
                        new_state = apply_plan(st, plan, args.distance_chunk_size)
                        compressed_head_outputs[(cid, variant)][head_idx] = (
                            evaluator.compressed_output_from_mapping(new_state.original_to_state).detach()
                        )
                        ps = plan_stats(plan)
                        acc = stat_acc[(cid, variant)]
                        acc["evicted_entries"] += int(ps["evicted_entries"])
                        acc["routed_entries"] += int(ps["routed_entries"])
                        acc["accepted_entries"] += int(ps["accepted_entries"])
                        acc["conflict_edges"] += int(ps["conflict_edges"])
                del evaluator

            if any(x is None for x in full_head_outputs):
                raise RuntimeError("Figure 3-left did not collect all KV/query heads for the layer")
            attn_module = _resolve_attention_module(model, layer_idx)
            o_proj = getattr(attn_module, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(
                    f"Figure 3-left requires the real attention o_proj for layer {layer_idx}, but none was found"
                )

            for cfg_item in configs:
                cid = str(cfg_item["config_id"])
                ol = float(cfg_item["original_lambda"])
                rl = float(cfg_item["route_lambda"])
                cl = float(cfg_item["conflict_lambda"])
                for variant in VARIANTS:
                    comp_heads = compressed_head_outputs[(cid, variant)]
                    if any(x is None for x in comp_heads):
                        raise RuntimeError(f"missing whole-layer head output for config={cid}, variant={variant}")
                    rel = whole_layer_attention_relative_error(
                        full_head_outputs=full_head_outputs,
                        compressed_head_outputs=comp_heads,
                        num_probes=num_probes,
                        o_proj=o_proj,
                        output_norm_eps=args.output_norm_eps,
                    )
                    acc = stat_acc[(cid, variant)]
                    routed_total = int(acc["routed_entries"])
                    accepted_total = int(acc["accepted_entries"])
                    rows.append({
                        "config_id": cid,
                        "is_primary": int(cfg_item["is_primary"]),
                        "original_lambda": ol,
                        "route_lambda": rl,
                        "conflict_lambda": cl,
                        "prefix_idx": prefix_idx,
                        "layer_idx": layer_idx,
                        "variant": variant,
                        "evaluation_scope": "whole_layer_post_o_proj",
                        "num_kv_heads": hkv,
                        "num_query_heads": hq,
                        "num_probes": num_probes,
                        "input_length": int(input_lengths[0]) if input_lengths else 0,
                        "retained_length": int(retained_lengths[0]) if retained_lengths else 0,
                        "calibration_scale_mean": float(np.mean(calibration_scale_values)),
                        "calibration_scale_min": float(np.min(calibration_scale_values)),
                        "calibration_scale_max": float(np.max(calibration_scale_values)),
                        "original_head_scale_mean": float(np.mean(original_scale_values)),
                        "output_rel_error": float(rel),
                        # One natural next-token probe is used, so this equals the
                        # whole-layer scalar error.  Retained for summary-schema compatibility.
                        "output_p90_error": float(rel),
                        "evicted_entries": int(acc["evicted_entries"]),
                        "routed_entries": routed_total,
                        "accepted_entries": accepted_total,
                        "accepted_entry_given_routed": float(accepted_total / max(routed_total, 1)),
                        "conflict_edges": int(acc["conflict_edges"]),
                    })

        del probe, past, prefix_keys_all, prefix_values_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg = aggregate(rows)
    write_csv(out_dir / "single_round_by_layer.csv", rows)
    write_csv(out_dir / "single_round_summary.csv", avg)
    cfg = vars(args).copy(); cfg.update({
        "experiment": "coherekv_focused_single_round_whole_layer",
        "variants": list(VARIANTS),
        "query_protocol": "one natural next-token Q per frozen prefix; query token KV excluded",
        "evaluation_scope": "whole attention-layer output after real o_proj; all KV/query heads; one observation per prefix x layer x method",
        "kv_head_selection_note": "ignored by Figure 3-left because whole-layer evaluation requires all KV/query heads",
        "calibration_scale_file": str(Path(args.calibration_scale_file).resolve()),
        "calibration_scale_shape": list(calibration_scales.shape),
        "modified_scale_protocol": "multiple calibration sequences; RMS-aggregated frozen per-(layer,KV-head) S_lh reused for all later rounds",
        "modified_conflict": "current-round ||K_u-K_v||_2/S_lh^calib > conflict_lambda",
        "modified_priority": "(-degree, -normalized_anchor_K_distance, -stable_hash, -position)",
        "engineering": "Eviction-Only/Merge-All/Modified share partition+routing; conflict pools packed; MIS fully tensorized",
        "sweep_design": "one-factor-at-a-time around primary; no Cartesian product",
        "sweep_configs": configs,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "elapsed_seconds": time.time() - t0,
    })
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Focused single-round CohereKV — whole-layer post-o_proj error",
        "=============================================================",
    ]
    for r in avg:
        tag = "PRIMARY" if int(r.get("is_primary", 0)) else r["config_id"]
        lines.append(f"[{tag}] {r['variant']}: {100*r['output_rel_error']:.4f}%")
    (out_dir / "RESULT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
