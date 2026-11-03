#!/usr/bin/env python3
from __future__ import annotations

"""Focused Motivation 1/2 preliminary experiments.

Figure 1 / Motivation 1
----------------------
At operating points with unified Euclidean Delta and cosine tau controls, test both routings across
multiple layers and KV heads.  Figure 1 measures directly:
  (i) Star-conflict prevalence in every layer/head;
  (ii) within a conflict group, the counterfactual gain after resolving up to K
       severe conflicts.  For each severe conflict pair (u, v), the two legal
       filtering outcomes (evict u vs evict v) are evaluated and the better
       branch is followed.  This is an existence/potential diagnostic, not the
       deployed Modified CohereKV selector.

Figure 2 / Motivation 2
----------------------
For each routing metric, sweep a small set of routing thresholds.  At *every*
routing threshold, run the same per-Star-group counterfactual used by Figure 1:
each eligible Star group independently resolves up to K severe conflicts, with
the lower-error endpoint-filtering branch selected at each local step.  After
all KV/query heads in the layer are processed, their attention-head outputs are
concatenated in model order, passed through the layer's real ``o_proj``, and one
whole-layer attention-output relative error is evaluated.  With the optional
CR-grid switch, multiple K values are evaluated and plotted as separate curves.
This makes the best routing-only Merge-All point directly visible on the same
x-axis and tests whether explicit conflict reduction improves upon
routing-threshold tuning.

Every frozen prefix contributes exactly one natural next-token query; the query
KV is excluded from the evaluated prefix cache.
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
from typing import Dict, Iterable, List, Sequence, Tuple

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
    build_full_cache_with_keepkv_scores,
    cache_key_value_layers,
    first_execution_device,
    initial_head_state,
    load_token_stream,
    set_seed,
    state_importance,
)
from experiment_utils import resolve_selection  # noqa: E402
from diagnostic_conflict_pruning import (  # noqa: E402
    counterfactual_conflict_reduction_mask,
    conflict_mask_from_score,
    conflict_stats,
    most_severe_conflict_edge,
)
from group_local_attention import GroupLocalAttentionEvaluator  # noqa: E402
from random_query_probes import collect_independent_query_states, _resolve_attention_module  # noqa: E402
from star_group_planner import (  # noqa: E402
    enumerate_candidate_groups,
    fixed_partition,
    normalized_k_conflict_score,
    pairwise_candidate_distances,
    precompute_routing_scores,
    route_stats,
    route_threshold,
    route_view,
)
from whole_cache_attention import (  # noqa: E402
    WholePrefixAttentionEvaluator, whole_layer_attention_relative_error,
)
from calibration_scale import load_calibration_scales, scale_for, validate_calibration_scales  # noqa: E402
from threshold_protocol import shared_lambda_at_point  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused CohereKV Motivation 1/2")
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
    p.add_argument("--routing_score_chunk_size", type=int, default=1024)
    p.add_argument("--calibration_scale_file", required=True,
                   help="JSON produced by calibrate_layer_head_scales.py; frozen S_{layer,head} for Euclidean routing/conflict")

    # One Euclidean compactness coefficient: Delta_(layer,head) = lambda*S_calib.
    # Euclidean routing AND Euclidean conflict detection use this identical Delta.
    # Cosine routing is a separate motivation-only control with cosine tau,
    # while its conflict detection still uses the same Euclidean Delta.
    p.add_argument("--lambda_value", type=float, default=0.40)

    p.add_argument("--fig1_cosine_tau", type=float, default=0.90)
    p.add_argument("--fig1_enable_hparam_sweep", action="store_true")
    p.add_argument("--fig1_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--fig1_cosine_tau_grid", default="0.85,0.90,0.95")
    p.add_argument("--fig1_cr_num_removals", type=int, default=1,
                   help="max severe conflicts resolved per sampled Star group")
    p.add_argument("--fig1_cr_num_removals_grid", default="1,2,4")

    # Figure 2: Euclidean x-axis sweeps shared lambda (routing AND conflicts).
    # Cosine x-axis sweeps tau_cos with Euclidean conflict Delta fixed by lambda.
    p.add_argument("--fig2_cosine_tau", type=float, default=0.90)
    p.add_argument("--fig2_euclidean_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--fig2_cosine_tau_grid", default="0.85,0.90,0.95")
    p.add_argument("--fig2_cr_num_removals", type=int, default=1,
                   help="max severe conflicts resolved PER STAR GROUP for Figure-2 CR curve")
    p.add_argument("--fig2_enable_cr_grid", action="store_true")
    p.add_argument("--fig2_cr_num_removals_grid", default="1,2,4")

    p.add_argument("--max_groups_per_head", type=int, default=16,
                   help="max Figure-1 conflict groups sampled per head for the local intervention; prevalence uses all groups")

    p.add_argument("--keepkv_sink_size", type=int, default=4)
    p.add_argument("--keepkv_recent_protect_size", type=int, default=128)
    p.add_argument("--keepkv_score_window", type=int, default=128)
    p.add_argument("--keepkv_recency_beta", type=float, default=1.0)
    p.add_argument("--layer_selection", default="0,8,16,24,31")
    p.add_argument("--kv_head_selection", default="0,1,2,3,4,5,6,7")
    p.add_argument("--prob_eps", type=float, default=1e-30)
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


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals or any(x < 0 for x in vals):
        raise ValueError("empty/negative integer list")
    return vals


def _unique_configs(configs: Sequence[Dict]) -> List[Dict]:
    out = []
    seen = set()
    for c in configs:
        key = (str(c["routing_metric"]), float(c["route_param"]), float(c["lambda_value"]), int(c.get("cr_num_removals", 0)))
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def build_fig1_configs(args, metric: str) -> List[Dict]:
    """Euclidean: route lambda == conflict lambda; cosine: tau_cos and Delta."""
    base_lambda = float(args.lambda_value)
    tau_cos = float(args.fig1_cosine_tau)
    route_main = base_lambda if metric == "euclidean" else tau_cos
    k_main = int(args.fig1_cr_num_removals)
    configs = [{
        "config_id": "primary", "is_primary": 1, "routing_metric": metric,
        "route_param": route_main, "lambda_value": base_lambda,
        "cr_num_removals": k_main,
    }]
    if args.fig1_enable_hparam_sweep:
        # A Euclidean lambda sweep changes BOTH routing admission and conflicts.
        # For cosine, only the Euclidean conflict bound changes in this sweep.
        for val in parse_float_list(args.fig1_lambda_grid):
            configs.append({
                "config_id": f"lambda_{val:g}", "is_primary": 0, "routing_metric": metric,
                "route_param": val if metric == "euclidean" else tau_cos,
                "lambda_value": val, "cr_num_removals": k_main,
            })
        if metric == "cosine":
            for tau in parse_float_list(args.fig1_cosine_tau_grid):
                configs.append({
                    "config_id": f"tau_cos_{tau:g}", "is_primary": 0,
                    "routing_metric": metric, "route_param": tau,
                    "lambda_value": base_lambda, "cr_num_removals": k_main,
                })
        for k in parse_int_list(args.fig1_cr_num_removals_grid):
            configs.append({
                "config_id": f"cr_k_{k}", "is_primary": 0, "routing_metric": metric,
                "route_param": route_main, "lambda_value": base_lambda,
                "cr_num_removals": int(k),
            })
    return _unique_configs(configs)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def mean_finite(vals: Iterable[float]) -> float:
    a = np.asarray([float(x) for x in vals if math.isfinite(float(x))], dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


def deterministic_subsample(items: Sequence, limit: int) -> List:
    items = list(items)
    if limit <= 0 or len(items) <= limit:
        return items
    ids = np.linspace(0, len(items) - 1, int(limit), dtype=np.int64)
    return [items[int(i)] for i in ids]


@torch.inference_mode()
def prepare_group_infos(*, view, routing_metric: str, evict_global: torch.Tensor,
                        keep_global: torch.Tensor, state, k_scale: float,
                        lambda_value: float) -> List[Dict]:
    infos: List[Dict] = []
    metric = str(routing_metric).lower()
    for anchor_local, cand_local in enumerate_candidate_groups(view=view, min_candidates=2):
        cand_pos = evict_global.index_select(0, cand_local)
        anchor_pos = int(keep_global[int(anchor_local)].item())
        k_pair = pairwise_candidate_distances(state.keys.index_select(0, cand_pos))
        k_score = normalized_k_conflict_score(k_pairwise_dist=k_pair, k_scale=k_scale)
        conf = conflict_mask_from_score(k_score, lambda_value)
        edges, vertices = conflict_stats(conf)
        anchor_dist_norm = view.assigned_distance.index_select(0, cand_local) / max(k_scale, 1e-12)
        anchor_cosine = view.assigned_cosine.index_select(0, cand_local)
        if metric == "euclidean":
            # Larger normalized anchor distance = worse routing compatibility.
            route_badness = anchor_dist_norm
        elif metric == "cosine":
            # Larger badness = lower cosine compatibility.
            route_badness = -anchor_cosine
        else:
            raise ValueError(metric)
        infos.append({
            "anchor_local": int(anchor_local),
            "anchor_pos": anchor_pos,
            "cand_local": cand_local,
            "cand_pos": cand_pos,
            "k_conflict_score": k_score,
            "anchor_gamma": anchor_dist_norm,
            "anchor_cosine": anchor_cosine,
            "route_badness": route_badness,
            "star_edges": int(edges),
            "star_vertices": int(vertices),
            "has_star_conflict": int(edges > 0),
            "severity": float(k_score.max().item()),
        })
    return infos


def _mean_num_error(errors) -> float:
    return float(errors.numerator_rel.float().mean().item())


@torch.inference_mode()
def group_counterfactual_conflict_reduction(*, evaluator: GroupLocalAttentionEvaluator,
                                            group_positions: torch.Tensor,
                                            k_conflict_score: torch.Tensor,
                                            lambda_value: float,
                                            max_removals: int):
    """Motivation-only K-step filtering counterfactual for one Star group.

    Conflict identification is K-space only.  For the current most-severe
    active conflict edge, evaluate evicting either endpoint and follow the lower
    attention-error branch.  The branch choice measures *existence/potential*;
    it is not the Figure-3 Modified CohereKV selection rule.
    """
    n = int(group_positions.numel()) - 1

    def evaluate_keep(candidate_keep: torch.Tensor) -> float:
        return _mean_num_error(evaluator.evaluate_merge(
            group_positions=group_positions,
            accepted_mask=candidate_keep,
        ))

    keep, removed, trace = counterfactual_conflict_reduction_mask(
        score=k_conflict_score,
        threshold=float(lambda_value),
        max_removals=int(max_removals),
        evaluate_keep=evaluate_keep,
        initial_keep=torch.ones(n, dtype=torch.bool, device=group_positions.device),
    )
    final_error = evaluate_keep(keep)
    return keep, removed, trace, final_error


@torch.inference_mode()
def group_counterfactual_keep_curve(*, evaluator: GroupLocalAttentionEvaluator,
                                    group_positions: torch.Tensor,
                                    k_conflict_score: torch.Tensor,
                                    lambda_value: float,
                                    k_values: Sequence[int]):
    """Return local accepted masks for several per-Star-group removal budgets.

    The branch decisions are exactly the same as Figure 1: identify the current
    most-severe conflict in K-space, evaluate evicting either endpoint using the
    group-local counterfactual, follow the lower-error branch, and recompute the
    remaining conflicts.  Running once up to max(K) lets Figure 2 reuse the same
    deterministic prefix of decisions for every requested K.
    """
    ks = sorted(set(max(0, int(k)) for k in k_values))
    max_k = max(ks) if ks else 0
    n = int(group_positions.numel()) - 1

    def evaluate_keep(candidate_keep: torch.Tensor) -> float:
        return _mean_num_error(evaluator.evaluate_merge(
            group_positions=group_positions,
            accepted_mask=candidate_keep,
        ))

    keep, removed, trace = counterfactual_conflict_reduction_mask(
        score=k_conflict_score,
        threshold=float(lambda_value),
        max_removals=int(max_k),
        evaluate_keep=evaluate_keep,
        initial_keep=torch.ones(n, dtype=torch.bool, device=group_positions.device),
    )
    del keep  # reconstructed below from the ordered removal trace
    by_k: Dict[int, torch.Tensor] = {}
    for k in ks:
        local_keep = torch.ones(n, dtype=torch.bool, device=group_positions.device)
        for local_idx in removed[: int(k)]:
            local_keep[int(local_idx)] = False
        by_k[int(k)] = local_keep
    return by_k, removed, trace


@torch.inference_mode()
def per_star_group_counterfactual_curve(*, group_evaluator: GroupLocalAttentionEvaluator,
                                        view, infos: Sequence[Dict],
                                        lambda_value: float, k_values: Sequence[int]):
    """Return Figure-2 accepted masks with K defined per Star group.

    Every eligible routed Star group independently resolves up to K severe
    conflicts using the exact Figure-1 group-local counterfactual.  No per-head
    relative error is computed here.  Figure 2 later composes *all* query-head
    attention outputs in the layer, applies the real attention ``o_proj``, and
    evaluates one whole-layer relative error.
    """
    ks = sorted(set(max(0, int(k)) for k in k_values))
    accepted_by_k = {int(k): view.compatible.clone() for k in ks}
    removed_by_k = {int(k): 0 for k in ks}
    affected_groups_by_k = {int(k): 0 for k in ks}

    for info in infos:
        if int(info.get("has_star_conflict", 0)) != 1:
            continue
        cand_pos = info["cand_pos"]
        cand_local = info["cand_local"]
        if int(cand_pos.numel()) == 0:
            continue
        group_pos = torch.cat([
            torch.tensor([info["anchor_pos"]], dtype=torch.long, device=cand_pos.device),
            cand_pos,
        ])
        local_by_k, removed, _ = group_counterfactual_keep_curve(
            evaluator=group_evaluator,
            group_positions=group_pos,
            k_conflict_score=info["k_conflict_score"],
            lambda_value=float(lambda_value),
            k_values=ks,
        )
        for k in ks:
            local_keep = local_by_k[int(k)]
            accepted_by_k[int(k)][cand_local] = local_keep
            n_removed = min(int(k), len(removed))
            removed_by_k[int(k)] += n_removed
            affected_groups_by_k[int(k)] += int(n_removed > 0)

    return accepted_by_k, removed_by_k, affected_groups_by_k


def summarize_figure1(group_rows: Sequence[Dict], route_rows: Sequence[Dict], args):
    """Summarize prevalence + within-group intervention, without clean matching."""
    overall: List[Dict] = []
    by_obs: List[Dict] = []
    layer_head: List[Dict] = []

    config_keys = []
    for r in route_rows:
        if int(r.get("figure", 0)) != 1:
            continue
        key = (str(r["routing_metric"]), str(r.get("config_id", "primary")))
        if key not in config_keys:
            config_keys.append(key)

    for metric, config_id in config_keys:
        route_cfg = [r for r in route_rows
                     if int(r.get("figure", 0)) == 1
                     and r["routing_metric"] == metric
                     and str(r.get("config_id", "primary")) == config_id]
        group_cfg = [r for r in group_rows
                     if r["routing_metric"] == metric
                     and str(r.get("config_id", "primary")) == config_id]
        if not route_cfg:
            continue
        total_groups = sum(int(r.get("all_candidate_groups", 0)) for r in route_cfg)
        conflict_groups = sum(int(r.get("all_conflict_groups", 0)) for r in route_cfg)
        gains = [float(r["conflict_reduced_gain"]) for r in group_cfg
                 if math.isfinite(float(r["conflict_reduced_gain"]))]
        overall.append({
            "routing_metric": metric,
            "config_id": config_id,
            "is_primary": int(route_cfg[0].get("is_primary", 0)),
            "route_param": float(route_cfg[0]["route_param"]),
            "lambda_value": float(route_cfg[0]["lambda_value"]),
            "cr_num_removals": int(route_cfg[0].get("cr_num_removals", 0)),
            "candidate_groups": total_groups,
            "conflict_groups": conflict_groups,
            "conflict_prevalence": (conflict_groups / total_groups) if total_groups else float("nan"),
            "intervention_groups": len(gains),
            "conflict_reduced_gain": mean_finite(gains),
            "conflict_reduced_positive_fraction": mean_finite(float(x > 0) for x in gains),
        })

        obs_keys = sorted({
            (int(r["prefix_idx"]), int(r["layer_idx"]), int(r["kv_head_idx"]))
            for r in route_cfg
        })
        for pref, layer, head in obs_keys:
            rr = [r for r in route_cfg
                  if int(r["prefix_idx"]) == pref and int(r["layer_idx"]) == layer
                  and int(r["kv_head_idx"]) == head]
            gg = [r for r in group_cfg
                  if int(r["prefix_idx"]) == pref and int(r["layer_idx"]) == layer
                  and int(r["kv_head_idx"]) == head]
            total = sum(int(r.get("all_candidate_groups", 0)) for r in rr)
            conf = sum(int(r.get("all_conflict_groups", 0)) for r in rr)
            obs_gains = [float(r["conflict_reduced_gain"]) for r in gg
                         if math.isfinite(float(r["conflict_reduced_gain"]))]
            by_obs.append({
                "routing_metric": metric, "config_id": config_id,
                "is_primary": int(route_cfg[0].get("is_primary", 0)),
                "route_param": float(route_cfg[0]["route_param"]),
                "lambda_value": float(route_cfg[0]["lambda_value"]),
                "cr_num_removals": int(route_cfg[0].get("cr_num_removals", 0)),
                "prefix_idx": pref, "layer_idx": layer, "kv_head_idx": head,
                "candidate_groups": total, "conflict_groups": conf,
                "conflict_prevalence": (conf / total) if total else float("nan"),
                "intervention_groups": len(obs_gains),
                "intervention_gain_mean": mean_finite(obs_gains),
                "intervention_positive_fraction": mean_finite(float(x > 0) for x in obs_gains),
            })

        lh_keys = sorted({(int(r["layer_idx"]), int(r["kv_head_idx"])) for r in route_cfg})
        for layer, head in lh_keys:
            rr = [r for r in route_cfg
                  if int(r["layer_idx"]) == layer and int(r["kv_head_idx"]) == head]
            gg = [r for r in group_cfg
                  if int(r["layer_idx"]) == layer and int(r["kv_head_idx"]) == head]
            total = sum(int(r.get("all_candidate_groups", 0)) for r in rr)
            conf = sum(int(r.get("all_conflict_groups", 0)) for r in rr)
            lh_gains = [float(r["conflict_reduced_gain"]) for r in gg
                        if math.isfinite(float(r["conflict_reduced_gain"]))]
            layer_head.append({
                "routing_metric": metric, "config_id": config_id,
                "is_primary": int(route_cfg[0].get("is_primary", 0)),
                "route_param": float(route_cfg[0]["route_param"]),
                "lambda_value": float(route_cfg[0]["lambda_value"]),
                "cr_num_removals": int(route_cfg[0].get("cr_num_removals", 0)),
                "layer_idx": layer, "kv_head_idx": head,
                "prefix_observations": len(rr),
                "candidate_groups": total, "conflict_groups": conf,
                "conflict_prevalence": (conf / total) if total else float("nan"),
                "intervention_groups": len(lh_gains),
                "intervention_gain_mean": mean_finite(lh_gains),
                "intervention_positive_fraction": mean_finite(float(x > 0) for x in lh_gains),
            })
    return overall, by_obs, layer_head


@torch.inference_mode()
def whole_prefix_counterfactual_curve(*, evaluator: WholePrefixAttentionEvaluator,
                                      keep_idx: torch.Tensor, evict_idx: torch.Tensor,
                                      view, infos: Sequence[Dict], counts_1d: torch.Tensor,
                                      lambda_value: float, k_values: Sequence[int]):
    """Return a motivation-only K-step conflict-resolution curve for one head.

    Starting from Merge-All at the pre-declared routing point, repeatedly find
    the *globally* most-severe active Star-conflict edge.  For its two endpoints
    evaluate the two legal filtering counterfactuals (evict u vs evict v), keep
    the lower-error branch, update the active conflict set, and continue.

    This intentionally measures the *potential* benefit of resolving already
    identified Star conflicts.  It is not the Figure-3 MIS or a deployable
    endpoint selector.
    """
    ks = sorted(set(max(0, int(k)) for k in k_values))
    max_k = max(ks) if ks else 0
    accepted = view.compatible.clone()

    def eval_mask(mask: torch.Tensor) -> float:
        return float(evaluator.compressed_error(
            keep_idx=keep_idx, evict_idx=evict_idx, best_idx=view.best_idx,
            accepted_mask=mask, counts_1d=counts_1d,
        ).mean().item())

    current_err = eval_mask(accepted)
    by_k = {0: current_err}
    trace: List[Dict] = []

    for step in range(1, max_k + 1):
        best = None
        for info_idx, info in enumerate(infos):
            if int(info.get("has_star_conflict", 0)) != 1:
                continue
            cand_local = info["cand_local"]
            local_keep = accepted.index_select(0, cand_local).bool()
            edge = most_severe_conflict_edge(
                info["k_conflict_score"], float(lambda_value), keep=local_keep,
            )
            if edge is None:
                continue
            u, v = int(edge[0]), int(edge[1])
            severity = float(info["k_conflict_score"][u, v].item())
            # Severity decides which conflict is diagnosed.  Remaining terms are
            # deterministic tie-breakers only and do not inspect attention error.
            key = (severity, -int(info["anchor_local"]), -min(u, v), -max(u, v))
            if best is None or key > best[0]:
                best = (key, int(info_idx), u, v)
        if best is None:
            break

        _, info_idx, u, v = best
        info = infos[info_idx]
        ev_local_u = int(info["cand_local"][u].item())
        ev_local_v = int(info["cand_local"][v].item())
        mask_u = accepted.clone(); mask_u[ev_local_u] = False
        mask_v = accepted.clone(); mask_v[ev_local_v] = False
        err_u = eval_mask(mask_u)
        err_v = eval_mask(mask_v)
        if err_u <= err_v:
            accepted = mask_u
            chosen_local = ev_local_u
            chosen_endpoint = u
            current_err = err_u
        else:
            accepted = mask_v
            chosen_local = ev_local_v
            chosen_endpoint = v
            current_err = err_v
        by_k[step] = current_err
        trace.append({
            "step": step,
            "anchor_local": int(info["anchor_local"]),
            "u": u, "v": v,
            "severity": float(info["k_conflict_score"][u, v].item()),
            "error_drop_u": err_u,
            "error_drop_v": err_v,
            "chosen_endpoint": int(chosen_endpoint),
            "chosen_evict_local": int(chosen_local),
            "chosen_error": float(current_err),
        })

    # If conflicts are exhausted early, larger requested K values simply retain
    # the last valid state rather than inventing non-conflict deletions.
    last = current_err
    for k in ks:
        if k not in by_k:
            by_k[k] = last
    return accepted, by_k, trace


def aggregate_figure2(ma_rows: Sequence[Dict], cr_rows: Sequence[Dict]):
    """Aggregate routing-threshold curves for Merge-All and per-Star-group CR(K)."""
    route_summary: List[Dict] = []
    cr_curve: List[Dict] = []

    for metric in ("euclidean", "cosine"):
        mr = [r for r in ma_rows if r["routing_metric"] == metric]
        if not mr:
            continue
        params = sorted({float(r["route_param"]) for r in mr})
        param_means = {}
        for param in params:
            rr = [r for r in mr if abs(float(r["route_param"]) - param) < 1e-12]
            param_means[param] = mean_finite(r["merge_all_output_rel_error"] for r in rr)
        best_param = min(params, key=lambda p: param_means[p])
        best_mean = float(param_means[best_param])

        for param in params:
            rr = [r for r in mr if abs(float(r["route_param"]) - param) < 1e-12]
            route_summary.append({
                "routing_metric": metric,
                "route_param": param,
                "observations": len(rr),
                "merge_all_output_rel_error": float(param_means[param]),
                "is_best_routing_only": int(abs(param - best_param) < 1e-12),
                "best_route_param": best_param,
                "best_merge_all_output_rel_error": best_mean,
            })

        cr_metric = [r for r in cr_rows
                     if r["routing_metric"] == metric and int(r.get("is_primary", 1)) == 1]
        if not cr_metric:
            continue
        k_values = sorted({int(r["cr_num_removals"]) for r in cr_metric})
        for k in k_values:
            for param in params:
                rr = [r for r in cr_metric
                      if int(r["cr_num_removals"]) == k
                      and abs(float(r["route_param"]) - param) < 1e-12]
                if not rr:
                    continue
                ma_same = float(param_means[param])
                cr_mean = mean_finite(r["conflict_reduced_output_rel_error"] for r in rr)
                cr_curve.append({
                    "routing_metric": metric,
                    "route_param": param,
                    "lambda_value": float(rr[0]["lambda_value"]),
                    "cr_num_removals": int(k),
                    "observations": len(rr),
                    "conflict_reduced_output_rel_error": cr_mean,
                    "merge_all_output_rel_error": ma_same,
                    "cr_gain_vs_same_route": ma_same - cr_mean,
                    "mean_actual_removed_sources": mean_finite(r["actual_removed_sources"] for r in rr),
                    "mean_affected_star_groups": mean_finite(r["affected_star_groups"] for r in rr),
                    "best_route_param": best_param,
                    "best_merge_all_output_rel_error": best_mean,
                    "is_best_routing_only": int(abs(param - best_param) < 1e-12),
                })
    return route_summary, cr_curve

def main() -> None:
    args = build_parser().parse_args()
    if str(args.attn_implementation).lower() != "eager":
        raise ValueError("motivation requires eager attention")
    if not 0 < float(args.keep_ratio) < 1:
        raise ValueError("keep_ratio must be in (0,1)")
    if not math.isfinite(args.lambda_value) or args.lambda_value <= 0:
        raise ValueError("shared lambda must be finite and positive")
    if not 0 < args.fig1_cosine_tau <= 1 or not 0 < args.fig2_cosine_tau <= 1:
        raise ValueError("cosine routing tau must lie in (0,1]")
    if int(args.fig1_cr_num_removals) < 0 or int(args.fig2_cr_num_removals) < 0:
        raise ValueError("Conflict-Reduced removal budgets must be non-negative")

    for val in parse_float_list(args.fig1_lambda_grid):
        shared_lambda_at_point("euclidean", val, args.lambda_value)
    for val in parse_float_list(args.fig1_cosine_tau_grid):
        shared_lambda_at_point("cosine", val, args.lambda_value)
    fig1_configs = {m: build_fig1_configs(args, m) for m in ("euclidean", "cosine")}
    fig2_route_grids = {
        "euclidean": sorted(set(parse_float_list(args.fig2_euclidean_lambda_grid) + [float(args.lambda_value)])),
        "cosine": sorted(set(parse_float_list(args.fig2_cosine_tau_grid) + [float(args.fig2_cosine_tau)])),
    }
    for metric, grid in fig2_route_grids.items():
        for val in grid:
            shared_lambda_at_point(metric, val, args.lambda_value)
    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
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
    calibration_scales = load_calibration_scales(args.calibration_scale_file, args.model)
    validate_calibration_scales(calibration_scales, config)

    stride = int(args.prefix_stride) if int(args.prefix_stride) > 0 else int(args.model_seqlen) + 1
    max_start = int(tokens.shape[1]) - int(args.model_seqlen) - 1
    starts = list(range(0, max_start + 1, stride))[: int(args.max_eval_prefixes)]
    if not starts:
        raise RuntimeError("insufficient dataset tokens")

    group_rows: List[Dict] = []
    route_rows: List[Dict] = []
    q2_ma_rows: List[Dict] = []
    q2_cr_rows: List[Dict] = []
    selected_layers = None
    t0 = time.time()
    print("=== Figure 1/2 focused motivation ===", flush=True)
    print(f"Frozen calibration S_lh shape={tuple(calibration_scales.shape)}; "
          f"min={float(calibration_scales.min()):.6g}, max={float(calibration_scales.max()):.6g}", flush=True)
    print("Figure 1: Euclidean + cosine routing; K-step severe-conflict filtering counterfactual", flush=True)
    fig1_counts = {m: len(v) for m, v in fig1_configs.items()}
    print(f"Shared Delta=lambda*S_calib; lambda={args.lambda_value}; cosine tau (fig1/fig2)="
          f"{args.fig1_cosine_tau}/{args.fig2_cosine_tau}", flush=True)
    print(f"Figure 1 sweep={'on' if args.fig1_enable_hparam_sweep else 'off'}; configs={fig1_counts}", flush=True)
    fig2_k_values = (sorted(set(parse_int_list(args.fig2_cr_num_removals_grid)
                                + [int(args.fig2_cr_num_removals)]))
                     if args.fig2_enable_cr_grid else [int(args.fig2_cr_num_removals)])
    print(f"Figure 2: routing-threshold x-axis; Merge-All + per-Star-group CR(K); "
          f"route grids={fig2_route_grids}; K curves={fig2_k_values}; "
          f"CR-grid={'on' if args.fig2_enable_cr_grid else 'off'}", flush=True)

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
            fig1_heads = set(resolve_selection(args.kv_head_selection, hkv))

            # Figure 2 is a whole-layer metric.  It must therefore include every
            # KV/query head in this attention layer, even when Figure 1 is
            # intentionally restricted to a subset of heads for its heatmaps.
            full_head_outputs: List[torch.Tensor | None] = [None] * hkv
            q2_ma_head_outputs: Dict[Tuple[str, float], List[torch.Tensor | None]] = {}
            q2_cr_head_outputs: Dict[Tuple[str, str, float, float, int], List[torch.Tensor | None]] = {}
            q2_cr_meta: Dict[Tuple[str, str, float, float, int], Dict[str, int]] = {}
            layer_scales: List[float] = []

            for head_idx in range(hkv):
                if args.progress_every_head:
                    print(f"  prefix={prefix_idx} layer={layer_idx} head={head_idx}", flush=True)
                raw_keys = raw_layer_keys[head_idx].contiguous()
                raw_values = raw_layer_values[head_idx].contiguous()
                state = initial_head_state(
                    raw_keys, score_layer[head_idx].to(raw_keys.device, dtype=torch.float32).contiguous()
                )
                k_scale = scale_for(calibration_scales, layer_idx, head_idx)
                layer_scales.append(float(k_scale))
                keep_len = max(2, min(
                    int(raw_keys.shape[0]) - 1,
                    int(round(raw_keys.shape[0] * float(args.keep_ratio))),
                ))
                importance = state_importance(state, args.keepkv_recency_beta).unsqueeze(0)
                keep_idx, evict_idx = fixed_partition(
                    importance=importance,
                    positions=state.positions.unsqueeze(0),
                    keep_len=keep_len,
                    sink_size=args.keepkv_sink_size,
                    recent_window=args.keepkv_recent_protect_size,
                )
                scores = precompute_routing_scores(
                    keys=state.keys.unsqueeze(0), keep_idx=keep_idx, evict_idx=evict_idx,
                    chunk_size=args.routing_score_chunk_size,
                )

                q0, q1 = head_idx * nrep, (head_idx + 1) * nrep
                qh = q_layer[q0:q1]
                group_eval = GroupLocalAttentionEvaluator.from_queries(
                    queries=qh, keys=raw_keys, values=raw_values,
                    prob_eps=args.prob_eps, norm_eps=args.output_norm_eps,
                )
                whole_eval = WholePrefixAttentionEvaluator(
                    qh, raw_keys, raw_values, output_norm_eps=args.output_norm_eps,
                )
                full_head_outputs[head_idx] = whole_eval.full_output.detach()
                evict_global = evict_idx[0].long(); keep_global = keep_idx[0].long()
                ones = torch.ones(int(raw_keys.shape[0]), dtype=torch.float32, device=raw_keys.device)

                # ---------------- Figure 1 ----------------
                # Figure 1: only the user-selected
                # heads contribute to the layer/head generality heatmaps.
                if head_idx in fig1_heads:
                    for metric in ("euclidean", "cosine"):
                        for cfg_item in fig1_configs[metric]:
                            param = float(cfg_item["route_param"])
                            lambda_value = float(cfg_item["lambda_value"])
                            view = route_view(
                                scores=scores, metric=metric,
                                threshold=route_threshold(metric, param, k_scale),
                            )
                            rs = route_stats(view, int(keep_idx.numel()), int(evict_idx.numel()))
                            infos_all = prepare_group_infos(
                                view=view, routing_metric=metric, evict_global=evict_global, keep_global=keep_global,
                                state=state, k_scale=k_scale, lambda_value=lambda_value,
                            )
                            conflict_infos = [x for x in infos_all if int(x["has_star_conflict"]) == 1]
                            route_rows.append({
                                "figure": 1, "config_id": cfg_item["config_id"],
                                "is_primary": int(cfg_item["is_primary"]),
                                "prefix_idx": prefix_idx, "layer_idx": layer_idx, "kv_head_idx": head_idx,
                                "routing_metric": metric, "route_param": param,
                                "lambda_value": lambda_value, "calibration_scale": k_scale,
                                "cr_num_removals": int(cfg_item["cr_num_removals"]),
                                "all_candidate_groups": len(infos_all),
                                "all_conflict_groups": len(conflict_infos),
                                **rs,
                            })

                            # Intervention cost is bounded: sample only conflict groups,
                            # while prevalence above is computed over *all* eligible groups.
                            for info in deterministic_subsample(conflict_infos, args.max_groups_per_head):
                                cand_pos = info["cand_pos"]; n = int(cand_pos.numel())
                                group_pos = torch.cat([
                                    torch.tensor([info["anchor_pos"]], dtype=torch.long, device=cand_pos.device),
                                    cand_pos,
                                ])
                                all_keep = torch.ones(n, dtype=torch.bool, device=cand_pos.device)
                                ma = _mean_num_error(group_eval.evaluate_merge(
                                    group_positions=group_pos, accepted_mask=all_keep
                                ))
                                cr_keep, removed_local, cr_trace, cr = group_counterfactual_conflict_reduction(
                                    evaluator=group_eval,
                                    group_positions=group_pos,
                                    k_conflict_score=info["k_conflict_score"],
                                    lambda_value=lambda_value,
                                    max_removals=int(cfg_item["cr_num_removals"]),
                                )
                                group_rows.append({
                                    "config_id": cfg_item["config_id"], "is_primary": int(cfg_item["is_primary"]),
                                    "prefix_idx": prefix_idx, "layer_idx": layer_idx, "kv_head_idx": head_idx,
                                    "routing_metric": metric, "route_param": param,
                                    "lambda_value": lambda_value, "calibration_scale": k_scale,
                                    "cr_num_removals": int(cfg_item["cr_num_removals"]),
                                    "candidate_count": n,
                                    "anchor_distance_mean_norm": float(info["anchor_gamma"].mean().item()),
                                    "anchor_distance_max_norm": float(info["anchor_gamma"].max().item()),
                                    "max_pair_k_distance_norm": float(info["severity"]),
                                    "has_star_conflict": 1,
                                    "star_edges": int(info["star_edges"]), "star_vertices": int(info["star_vertices"]),
                                    "merge_all_numerator_rel_error": ma,
                                    "conflict_reduced_numerator_rel_error": cr,
                                    "conflict_reduced_gain": ma - cr,
                                    "removed_sources": len(removed_local),
                                    "resolved_conflicts": len(cr_trace),
                                })

                # ---------------- Figure 2 ----------------
                # For each routing configuration, build this head's compressed
                # pre-o_proj output.  After *all* KV/query heads are processed,
                # the outputs are concatenated and passed through the real o_proj;
                # only then is one whole-layer relative error computed.
                for metric in ("euclidean", "cosine"):
                    k_values = fig2_k_values

                    for route_param in fig2_route_grids[metric]:
                        route_param = float(route_param)
                        lambda_value = shared_lambda_at_point(metric, route_param, args.lambda_value)
                        view = route_view(
                            scores=scores, metric=metric,
                            threshold=route_threshold(metric, route_param, k_scale),
                        )
                        ma_key = (metric, route_param)
                        q2_ma_head_outputs.setdefault(ma_key, [None] * hkv)[head_idx] = whole_eval.compressed_output(
                            keep_idx=keep_idx, evict_idx=evict_idx, best_idx=view.best_idx,
                            accepted_mask=view.compatible, counts_1d=ones,
                        ).detach()

                        infos = prepare_group_infos(
                            view=view, routing_metric=metric,
                            evict_global=evict_global, keep_global=keep_global,
                            state=state, k_scale=k_scale,
                            lambda_value=lambda_value,
                        )
                        accepted_by_k, removed_by_k, affected_groups_by_k = per_star_group_counterfactual_curve(
                            group_evaluator=group_eval,
                            view=view,
                            infos=infos,
                            lambda_value=lambda_value,
                            k_values=k_values,
                        )
                        available_groups = sum(int(x["has_star_conflict"]) for x in infos)
                        for k in k_values:
                            cr_key = ("primary_route_curve", metric, route_param, lambda_value, int(k))
                            q2_cr_head_outputs.setdefault(cr_key, [None] * hkv)[head_idx] = whole_eval.compressed_output(
                                keep_idx=keep_idx, evict_idx=evict_idx, best_idx=view.best_idx,
                                accepted_mask=accepted_by_k[int(k)], counts_1d=ones,
                            ).detach()
                            meta = q2_cr_meta.setdefault(cr_key, {
                                "actual_removed_sources": 0,
                                "affected_star_groups": 0,
                                "available_conflict_groups": 0,
                            })
                            meta["actual_removed_sources"] += int(removed_by_k[int(k)])
                            meta["affected_star_groups"] += int(affected_groups_by_k[int(k)])
                            meta["available_conflict_groups"] += int(available_groups)

                del group_eval, whole_eval, scores

            # Figure 2: one observation per prefix × layer × routing configuration.
            attn_module = _resolve_attention_module(model, layer_idx)
            o_proj = getattr(attn_module, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(
                    f"Figure 2 requires the real attention o_proj for layer {layer_idx}, but none was found"
                )
            if any(x is None for x in full_head_outputs):
                raise RuntimeError("Figure 2 did not collect all KV/query heads for the layer")
            scale_mean = float(np.mean(layer_scales)) if layer_scales else float("nan")
            scale_min = float(np.min(layer_scales)) if layer_scales else float("nan")
            scale_max = float(np.max(layer_scales)) if layer_scales else float("nan")

            for (metric, route_param), comp_heads in q2_ma_head_outputs.items():
                fma = whole_layer_attention_relative_error(
                    full_head_outputs=full_head_outputs,
                    compressed_head_outputs=comp_heads,
                    num_probes=num_probes,
                    o_proj=o_proj,
                    output_norm_eps=args.output_norm_eps,
                )
                q2_ma_rows.append({
                    "prefix_idx": prefix_idx, "layer_idx": layer_idx,
                    "routing_metric": metric, "route_param": float(route_param),
                    "evaluation_scope": "whole_layer_post_o_proj",
                    "num_kv_heads": hkv, "num_query_heads": hq,
                    "calibration_scale_mean": scale_mean,
                    "calibration_scale_min": scale_min,
                    "calibration_scale_max": scale_max,
                    "merge_all_output_rel_error": float(fma),
                })

            for cr_key, comp_heads in q2_cr_head_outputs.items():
                config_id, metric, route_param, lambda_value, k = cr_key
                fcr = whole_layer_attention_relative_error(
                    full_head_outputs=full_head_outputs,
                    compressed_head_outputs=comp_heads,
                    num_probes=num_probes,
                    o_proj=o_proj,
                    output_norm_eps=args.output_norm_eps,
                )
                meta = q2_cr_meta[cr_key]
                q2_cr_rows.append({
                    "config_id": config_id,
                    "is_primary": int(config_id == "primary_route_curve"),
                    "prefix_idx": prefix_idx, "layer_idx": layer_idx,
                    "routing_metric": metric, "route_param": float(route_param),
                    "lambda_value": float(lambda_value),
                    "cr_num_removals": int(k),
                    "evaluation_scope": "whole_layer_post_o_proj",
                    "num_kv_heads": hkv, "num_query_heads": hq,
                    "calibration_scale_mean": scale_mean,
                    "calibration_scale_min": scale_min,
                    "calibration_scale_max": scale_max,
                    "conflict_reduced_output_rel_error": float(fcr),
                    "actual_removed_sources": int(meta["actual_removed_sources"]),
                    "affected_star_groups": int(meta["affected_star_groups"]),
                    "available_conflict_groups": int(meta["available_conflict_groups"]),
                })

        del probe, past, prefix_keys_all, prefix_values_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    q1_overall, q1_by_obs, q1_lh = summarize_figure1(group_rows, route_rows, args)
    q2_route_summary, q2_cr_curve = aggregate_figure2(q2_ma_rows, q2_cr_rows)
    write_csv(out_dir / "route_head_stats.csv", route_rows)
    write_csv(out_dir / "motivation_by_group.csv", group_rows)
    write_csv(out_dir / "q1_figure_overall.csv", q1_overall)
    write_csv(out_dir / "q1_figure_by_observation.csv", q1_by_obs)
    write_csv(out_dir / "q1_layer_head_summary.csv", q1_lh)
    write_csv(out_dir / "q2_merge_all_by_observation.csv", q2_ma_rows)
    write_csv(out_dir / "q2_conflict_reduced_by_observation.csv", q2_cr_rows)
    write_csv(out_dir / "q2_routing_sweep.csv", q2_route_summary)
    write_csv(out_dir / "q2_conflict_reduced_curve.csv", q2_cr_curve)
    write_csv(out_dir / "q2_hparam_sweep.csv", [r for r in q2_cr_rows if int(r.get("is_primary", 1)) == 0])

    cfg = vars(args).copy(); cfg.update({
        "experiment": "coherekv_focused_motivation",
        "evaluated_prefixes": len(starts),
        "figure1_routing_metrics": ["euclidean", "cosine"],
        "figure1_configs": fig1_configs,
        "figure1_sweep_design": "Euclidean: lambda controls both admission and conflicts; cosine: tau_cos controls routing, lambda controls Euclidean conflicts; optionally vary K",
        "figure1_design": "no clean matching; report Star-conflict prevalence and within-group counterfactual severe-conflict filtering gain across layer/head",
        "figure1_intervention": "identify the current most-severe K-space conflict pair; evaluate evict-u and evict-v; follow the lower-error filtering branch; repeat up to K times. This is an existence/potential diagnostic, not the Figure-3 selector",
        "figure2_routing_metrics": ["euclidean", "cosine"],
        "figure2_route_grids": fig2_route_grids,
        "figure2_lambda_grid": fig2_route_grids["euclidean"],
        "figure2_cosine_tau_grid": fig2_route_grids["cosine"],
        "figure2_cr_num_removals_grid": parse_int_list(args.fig2_cr_num_removals_grid),
        "figure2_cr_grid_enabled": bool(args.fig2_enable_cr_grid),
        "figure2_design": "Euclidean sweep: one lambda simultaneously controls route and conflict Delta at each point; cosine sweep: tau_cos varies while Euclidean conflict Delta uses fixed lambda. Each Star group independently resolves up to K severe conflicts with a real-next-Q oracle; error is measured after the real o_proj",
        "figure2_error_scope": "whole attention-layer output after o_proj; one observation per prefix x layer x routing configuration; no macro-average of per-head relative errors",
        "calibration_scale_file": str(Path(args.calibration_scale_file).resolve()),
        "calibration_scale_shape": list(calibration_scales.shape),
        "scale_protocol": "multiple independent calibration sequences; median-aggregated frozen per-(layer,KV-head) S_lh for all Euclidean routing/conflict decisions",
        "conflict_definition": "||K_u-K_v||_2 > Delta_(l,h), Delta=lambda*S_lh^calib; current-round K only",
        "unified_delta_protocol": "euclidean routing and conflicts share lambda per experiment point; cosine routing uses tau_cos but Euclidean conflicts use lambda",
        "query_protocol": "one natural next-token Q per frozen prefix; query token KV excluded",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "elapsed_seconds": time.time() - t0,
    })
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["Focused CohereKV Motivation 1/2", "================================"]
    for r in q1_overall:
        if int(r.get("is_primary", 0)) != 1:
            continue
        lines.append(
            f"M1 {r['routing_metric']}: conflict prevalence={100*float(r['conflict_prevalence']):.1f}%; "
            f"CR gain={100*float(r['conflict_reduced_gain']):+.4f}pp; "
            f"positive={100*float(r['conflict_reduced_positive_fraction']):.1f}%"
        )
    for metric in ("euclidean", "cosine"):
        rr = [r for r in q2_cr_curve if r["routing_metric"] == metric]
        if not rr:
            continue
        target_k = int(args.fig2_cr_num_removals)
        rr_k = [r for r in rr if int(r["cr_num_removals"]) == target_k]
        best_cr = min(rr_k, key=lambda r: float(r["conflict_reduced_output_rel_error"])) if rr_k else None
        r0 = rr[0]
        if best_cr is not None:
            lines.append(
                f"M2 {metric}: best routing-only MA={100*float(r0['best_merge_all_output_rel_error']):.4f}% "
                f"@route={float(r0['best_route_param']):g}; "
                f"best CR(K={target_k})={100*float(best_cr['conflict_reduced_output_rel_error']):.4f}% "
                f"@route={float(best_cr['route_param']):g}; "
                f"same-route CR gain={100*float(best_cr['cr_gain_vs_same_route']):+.4f}pp"
            )
    (out_dir / "RESULT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[done] {out_dir} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
