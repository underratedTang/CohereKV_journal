#!/usr/bin/env python3
from __future__ import annotations

"""Figure 3-right: repository-standard PPL for
Full / Eviction-only / Merge-All / Original / Modified.

The metric itself is not reimplemented here.  This script imports and calls
``task_evaluation/evaluate_ppl_kvmerge.py::evaluate_perplexity`` directly.
Modified CohereKV only replaces the local compression planner used by the cache.

Engineering change versus the previous preliminary script:
  * all KV heads of a layer are compressed together;
  * partition/routing are batched across B*Hkv rows;
  * all routed anchor pools are packed and conflict pair distances use batched GEMM;
  * deterministic parallel MIS is tensorized, with no per-node/per-neighbor Python;
  * only one small host sync is needed to allocate the largest packed pool, rather
    than thousands of ``.item()``/``float(cuda_tensor)`` synchronizations.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EVAL_ROOT = PROJECT_ROOT / "task_evaluation"
DIAG_ROOT = PROJECT_ROOT / "diagnosis_experiments"
HERE = Path(__file__).resolve().parent
for p in (TASK_EVAL_ROOT, DIAG_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate_ppl_kvmerge as task_ppl  # noqa: E402
from diagnose_original_coherekv import first_execution_device, set_seed  # noqa: E402
from calibration_scale import load_calibration_scales, validate_calibration_scales  # noqa: E402
from star_group_planner import (  # noqa: E402
    fixed_partition,
    precompute_routing_scores_batched,
    route_view_batched,
    select_conflict_free_routed_batched,
)
from models.common.mymethod_common import (  # noqa: E402
    base_score_rows_for_kv_heads,
    recent_window_importance,
)
from evaluate_ppl_kvmerge import (  # noqa: E402
    KVCompressingDynamicCache,
    _as_legacy_tuple,
    _legacy_kv_len,
    _replace_dynamic_cache_inplace,
    apply_kv_method,
    force_attn_hidden_and_compress,
)
from models.llama.merge_mymethod_llama import reset_mymethod_state  # noqa: E402
from models.llama.mymethod_runtime_patch_llama import (  # noqa: E402
    apply_mymethod_runtime_patch,
    remove_mymethod_runtime_patch,
)

ALL_VARIANTS = (
    "full",
    "eviction_only",
    "merge_all",
    "original_coherekv",
    "modified_coherekv",
)
_EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused CohereKV repository-style PPL")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--model_seqlen", type=int, default=8192)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--max_eval_sequences", type=int, default=2,
                   help="0 evaluates all complete fixed-length windows")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.80,
                   help="removed fraction; retained physical target ~= 1-ratio")
    p.add_argument("--min_kv_len", type=int, default=2)
    p.add_argument("--compress_every", type=int, default=1)
    p.add_argument("--calibration_scale_file", required=True,
                   help="JSON produced by calibrate_layer_head_scales.py; frozen S_{layer,head} for Merge-All/Modified")

    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--amp_dtype", default="bfloat16",
                   choices=["none", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_map", default="none")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS),
                   help=("comma-separated subset of full,eviction_only,merge_all,"
                         "original_coherekv,modified_coherekv"))

    p.add_argument("--original_lambda", type=float, default=0.40)
    p.add_argument("--route_lambda", type=float, default=0.40)
    p.add_argument("--conflict_lambda", type=float, default=0.10)
    p.add_argument("--enable_hparam_sweep", action="store_true",
                   help="compact one-factor-at-a-time sweep around the primary PPL configuration")
    p.add_argument("--original_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--route_lambda_grid", default="0.30,0.40,0.50")
    p.add_argument("--conflict_lambda_grid", default="0.05,0.10,0.20")
    p.add_argument("--keepkv_sink_size", type=int, default=4)
    p.add_argument("--keepkv_recent_protect_size", type=int, default=128)
    p.add_argument("--keepkv_score_window", type=int, default=128)
    p.add_argument("--keepkv_recency_beta", type=float, default=1.0)
    p.add_argument("--distance_chunk_size", type=int, default=1024)
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


def parse_amp_dtype(name: str):
    if str(name).lower() in {"none", "auto"}:
        return None
    return parse_dtype(name)


def parse_variants(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in ALL_VARIANTS]
    if bad:
        raise ValueError(f"unknown variants: {bad}")
    if not vals:
        raise ValueError("variants cannot be empty")
    return vals






def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def build_ofat_configs(args) -> List[Dict]:
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
        path.write_text("", encoding="utf-8"); return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def _gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if x.ndim == 2:
        return torch.gather(x, 1, idx)
    raise ValueError(tuple(x.shape))


@torch.inference_mode()
def evict_layer_local(*, keys: torch.Tensor, values: torch.Tensor,
                      counts: torch.Tensor, positions: torch.Tensor,
                      importance: torch.Tensor, keep_len: int,
                      sink_size: int, recent_window: int):
    """Eviction-only baseline using exactly the same KeepKV partition policy.

    No routing or merge is performed.  The retained entries, multiplicities and
    physical positions are gathered directly from ``fixed_partition``.  This is
    intentionally a separate fast path so Figure 3-right does not pay routing or
    conflict costs for the eviction-only control.
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys/values must be [B,H,L,D]")
    bsz, hkv, length, dim = keys.shape
    x = int(bsz * hkv)
    k = keys.reshape(x, length, dim)
    v = values.reshape(x, length, dim)
    c = counts.reshape(x, length)
    p = positions.reshape(x, length).long()
    imp = importance.reshape(x, length).float()

    keep_idx, _ = fixed_partition(
        importance=imp,
        positions=p,
        keep_len=int(keep_len),
        sink_size=int(sink_size),
        recent_window=int(recent_window),
    )
    knew = _gather_rows(k, keep_idx).contiguous()
    vnew = _gather_rows(v, keep_idx).contiguous()
    cnew = _gather_rows(c, keep_idx).contiguous()
    pnew = _gather_rows(p, keep_idx).contiguous()
    new_len = int(keep_idx.shape[1])
    return (
        knew.reshape(bsz, hkv, new_len, dim),
        vnew.reshape(bsz, hkv, new_len, dim),
        cnew.reshape(bsz, hkv, new_len),
        pnew.reshape(bsz, hkv, new_len),
    )


@torch.inference_mode()
def compress_layer_local(*, keys: torch.Tensor, values: torch.Tensor,
                            counts: torch.Tensor, positions: torch.Tensor,
                            importance: torch.Tensor, keep_len: int,
                            route_lambda: float, conflict_lambda: float, calibrated_head_scales: torch.Tensor,
                            sink_size: int, recent_window: int,
                            distance_chunk_size: int, use_conflict_reduction: bool):
    """Compress all B*Hkv rows of one layer in one tensorized call.

    Merge-All and Modified share partition/routing/aggregation.  Only Modified
    invokes conflict construction + MIS.
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys/values must be [B,H,L,D]")
    bsz, hkv, length, dim = keys.shape
    x = int(bsz * hkv)
    k = keys.reshape(x, length, dim)
    v = values.reshape(x, length, dim)
    c = counts.reshape(x, length).float()
    p = positions.reshape(x, length).long()
    imp = importance.reshape(x, length).float()

    keep_idx, evict_idx = fixed_partition(
        importance=imp, positions=p, keep_len=int(keep_len),
        sink_size=int(sink_size), recent_window=int(recent_window),
    )
    hs = calibrated_head_scales.detach().float().flatten().to(k.device)
    if hs.numel() != hkv:
        raise ValueError(f"calibrated_head_scales must have {hkv} values, got {hs.numel()}")
    scale = hs.view(1, hkv).expand(bsz, hkv).reshape(x).contiguous()
    scores = precompute_routing_scores_batched(
        keys=k, keep_idx=keep_idx, evict_idx=evict_idx,
        chunk_size=int(distance_chunk_size),
    )
    view = route_view_batched(
        scores=scores, metric="euclidean",
        threshold=scale * float(route_lambda),
    )

    evicted_keys = _gather_rows(k, evict_idx)
    if use_conflict_reduction:
        evicted_pos = _gather_rows(p, evict_idx)
        anchor_gamma = view.assigned_distance / scale.clamp_min(_EPS).unsqueeze(1)
        mis = select_conflict_free_routed_batched(
            evicted_keys=evicted_keys,
            best_idx=view.best_idx,
            route_compatible=view.compatible,
            anchor_gamma=anchor_gamma,
            physical_positions=evicted_pos,
            keep_len=int(keep_idx.shape[1]),
            k_scale=scale,
            conflict_lambda=float(conflict_lambda),
        )
        accepted_mask = mis.accepted
        conflict_edges = mis.conflict_edges_per_anchor.sum()
    else:
        accepted_mask = view.compatible
        conflict_edges = torch.zeros((), dtype=torch.int64, device=k.device)

    kk = _gather_rows(k, keep_idx).float()
    vv = _gather_rows(v, keep_idx).float()
    ck = _gather_rows(c, keep_idx).float()
    pp = _gather_rows(p, keep_idx)
    ke = evicted_keys.float()
    ve = _gather_rows(v, evict_idx).float()
    ce = _gather_rows(c, evict_idx).float()

    dest = view.best_idx.long()
    weight = ce * accepted_mask.to(ce.dtype)
    total = ck.clone()
    total.scatter_add_(1, dest, weight)
    dest_d = dest.unsqueeze(-1).expand(-1, -1, dim)
    ksum = kk * ck.unsqueeze(-1)
    vsum = vv * ck.unsqueeze(-1)
    ksum.scatter_add_(1, dest_d, ke * weight.unsqueeze(-1))
    vsum.scatter_add_(1, dest_d, ve * weight.unsqueeze(-1))

    knew = (ksum / total.clamp_min(1.0).unsqueeze(-1)).to(keys.dtype)
    vnew = (vsum / total.clamp_min(1.0).unsqueeze(-1)).to(values.dtype)
    cnew = total.to(counts.dtype)
    new_len = int(keep_idx.shape[1])
    return (
        knew.reshape(bsz, hkv, new_len, dim),
        vnew.reshape(bsz, hkv, new_len, dim),
        cnew.reshape(bsz, hkv, new_len),
        pp.reshape(bsz, hkv, new_len).contiguous(),
        view.compatible.sum(),
        accepted_mask.sum(),
        conflict_edges,
    )


class ProportionalCohereCache(KVCompressingDynamicCache):
    """Task-evaluator proportional cache with focused Original/Modified variants."""

    def __init__(self, *, model, variant: str, compress_ratio: float,
                 min_kv_len: int, compress_every: int,
                 route_lambda: float, conflict_lambda: float, original_lambda: float, calibration_scales: torch.Tensor,
                 sink_size: int, recent_window: int, score_window: int,
                 recency_beta: float, distance_chunk_size: int, stats: dict):
        super().__init__(
            model=model, method="mymethod", family="llama",
            compress_ratio=float(compress_ratio), sim_thr=0.0,
            method_sim_thrs={}, keepkv_config={}, mymethod_config={}, centroidkv_config={},
            min_kv_len=int(min_kv_len), compress_every=int(compress_every),
            stats=stats, task_eval_dir=TASK_EVAL_ROOT, target_kv_len=None,
        )
        self._cohere_variant = str(variant)
        self._route_lambda_local = float(route_lambda)
        self._conflict_lambda_local = float(conflict_lambda)
        self._original_lambda_local = float(original_lambda)
        self._calibration_scales_local = calibration_scales.detach().float().cpu().contiguous()
        self._sink_local = int(sink_size)
        self._recent_local = int(recent_window)
        self._score_window_local = int(score_window)
        self._recency_beta_local = float(recency_beta)
        self._distance_chunk_local = int(distance_chunk_size)
        self._counts_local = None
        self._positions_local = None
        self._scale_device_cache = {}

    def _sync_metadata(self, legacy) -> None:
        logical_seen = int(getattr(self, "_ppl_logical_seen_tokens", _legacy_kv_len(self)))
        if self._counts_local is None:
            self._counts_local = []
            self._positions_local = []
            for k, _ in legacy:
                b, h, l, _ = k.shape
                self._counts_local.append(torch.ones((b, h, l), dtype=torch.float32, device=k.device))
                pos = torch.arange(l, device=k.device, dtype=torch.long).view(1, 1, l).expand(b, h, l).clone()
                self._positions_local.append(pos)
            return
        for li, (k, _) in enumerate(legacy):
            old = int(self._counts_local[li].shape[-1])
            now = int(k.shape[-2])
            add = now - old
            if add < 0:
                raise RuntimeError("physical cache shrank outside CohereKV compression")
            if add == 0:
                continue
            b, h = int(k.shape[0]), int(k.shape[1])
            cadd = torch.ones((b, h, add), dtype=torch.float32, device=k.device)
            start = logical_seen - add
            padd = torch.arange(start, logical_seen, device=k.device, dtype=torch.long).view(1, 1, add).expand(b, h, add)
            self._counts_local[li] = torch.cat([self._counts_local[li].to(k.device), cadd], dim=-1)
            self._positions_local[li] = torch.cat([self._positions_local[li].to(k.device), padd], dim=-1)

    def _layer_calibration_scales(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        key = (int(layer_idx), device.type, device.index)
        cached = self._scale_device_cache.get(key)
        if cached is None:
            cached = self._calibration_scales_local[int(layer_idx)].to(device=device, dtype=torch.float32).contiguous()
            self._scale_device_cache[key] = cached
        return cached

    def _enable_count_bias(self, counts_layers) -> None:
        self._ppl_model._mymethod_counts = counts_layers
        for li, layer in enumerate(self._ppl_model.model.layers):
            attn = layer.self_attn
            attn._mymethod_eager_enabled = True
            attn._mymethod_sdpa_enabled = False
            attn._mymethod_counts_kv = counts_layers[li]
            attn._mymethod_selector = "keepkv"
            attn._mymethod_score_window = int(self._score_window_local)

    @torch.inference_mode()
    def maybe_compress(self, attentions, hidden_states) -> None:
        del hidden_states
        self._forward_calls += 1
        if self._compress_every > 1 and (self._forward_calls % self._compress_every) != 0:
            return
        before_len = _legacy_kv_len(self)
        if before_len < self._min_kv_len:
            return
        logical_seen = int(getattr(self, "_ppl_logical_seen_tokens", before_len))
        keep_len = max(1, int(round(logical_seen * (1.0 - float(self._compress_ratio)))))
        protected_floor = int(self._sink_local) + int(self._recent_local)
        if keep_len < protected_floor:
            self._stats["deferred_protected_budget"] = self._stats.get("deferred_protected_budget", 0) + 1
            return
        keep_len = min(keep_len, before_len)
        if keep_len >= before_len:
            return
        legacy = _as_legacy_tuple(self)
        if legacy is None:
            return
        if attentions is None:
            raise RuntimeError("compressed PPL requires current attention-score capture")

        self._stats["num_compressions"] = self._stats.get("num_compressions", 0) + 1
        self._stats.setdefault("target_kv_lens", []).append(int(keep_len))

        if self._cohere_variant == "original_coherekv":
            merged = apply_kv_method(
                "mymethod", "llama", self._ppl_model, legacy, attentions, None,
                float(self._compress_ratio), task_eval_dir=TASK_EVAL_ROOT,
                target_kv_len=int(keep_len),
                mymethod_config={
                    "selector": "keepkv",
                    "lambda_mymethod": self._original_lambda_local,
                    "group_mode": "full",
                    "graph_backend": "torch",
                    "route_backend": "torch",
                    "keepkv_sink_size": self._sink_local,
                    "keepkv_recent_protect_size": self._recent_local,
                    "keepkv_score_window": self._score_window_local,
                    "keepkv_recency_beta": self._recency_beta_local,
                    "keepkv_use_log_count": False,
                    "distance_chunk_size": self._distance_chunk_local,
                },
            )
            _replace_dynamic_cache_inplace(self, merged)
            self._stats.setdefault("compressed_kv_lens", []).append(_legacy_kv_len(self))
            return

        if self._cohere_variant not in {"eviction_only", "merge_all", "modified_coherekv"}:
            raise ValueError(self._cohere_variant)

        self._sync_metadata(legacy)
        new_past = []
        new_counts = []
        new_positions = []
        routed_stats = []
        accepted_stats = []
        edge_stats = []

        for li, (keys, values) in enumerate(legacy):
            bsz, hkv, length, _ = keys.shape
            score_rows = base_score_rows_for_kv_heads(attentions[li], hkv).to(keys.device, dtype=torch.float32)
            if int(score_rows.shape[-1]) != length:
                raise RuntimeError(
                    f"score/cache length mismatch at layer {li}: {score_rows.shape[-1]} vs {length}"
                )
            flat = score_rows.reshape(bsz * hkv, int(score_rows.shape[-2]), length)
            imp = recent_window_importance(
                flat, ptr=0, beta=self._recency_beta_local
            ).reshape(bsz, hkv, length)
            if self._cohere_variant == "eviction_only":
                knew, vnew, cnew, pnew = evict_layer_local(
                    keys=keys,
                    values=values,
                    counts=self._counts_local[li].to(keys.device),
                    positions=self._positions_local[li].to(keys.device),
                    importance=imp,
                    keep_len=keep_len,
                    sink_size=self._sink_local,
                    recent_window=self._recent_local,
                )
                routed = torch.zeros((), dtype=torch.int64, device=keys.device)
                accepted = torch.zeros((), dtype=torch.int64, device=keys.device)
                edges = torch.zeros((), dtype=torch.int64, device=keys.device)
            else:
                knew, vnew, cnew, pnew, routed, accepted, edges = compress_layer_local(
                    keys=keys,
                    values=values,
                    counts=self._counts_local[li].to(keys.device),
                    positions=self._positions_local[li].to(keys.device),
                    importance=imp,
                    keep_len=keep_len,
                    route_lambda=self._route_lambda_local,
                    conflict_lambda=self._conflict_lambda_local,
                    calibrated_head_scales=self._layer_calibration_scales(li, keys.device),
                    sink_size=self._sink_local,
                    recent_window=self._recent_local,
                    distance_chunk_size=self._distance_chunk_local,
                    use_conflict_reduction=(self._cohere_variant == "modified_coherekv"),
                )
            new_past.append((knew, vnew))
            new_counts.append(cnew)
            new_positions.append(pnew)
            routed_stats.append(routed)
            accepted_stats.append(accepted)
            edge_stats.append(edges)

        self._counts_local = new_counts
        self._positions_local = new_positions
        _replace_dynamic_cache_inplace(self, tuple(new_past))
        self._enable_count_bias(new_counts)
        self._stats.setdefault("compressed_kv_lens", []).append(_legacy_kv_len(self))
        # Three host reads per *whole-model compression*, not per node/group/head.
        self._stats["routed_total"] = self._stats.get("routed_total", 0) + int(torch.stack(routed_stats).sum().item())
        self._stats["accepted_total"] = self._stats.get("accepted_total", 0) + int(torch.stack(accepted_stats).sum().item())
        self._stats["conflict_edges_total"] = self._stats.get("conflict_edges_total", 0) + int(torch.stack(edge_stats).sum().item())


def configure_keepkv_score_capture(model, *, score_window: int, chunk_size: int) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("expected Llama model.model.layers")
    for layer in layers:
        attn = layer.self_attn
        attn._mymethod_selector = "keepkv"
        attn._mymethod_score_window = int(score_window)
        attn._kvmerge_attention_chunk_size = int(chunk_size)


def clear_sequence_state(model) -> None:
    reset_mymethod_state(model)
    if hasattr(model, "_mymethod_counts"):
        delattr(model, "_mymethod_counts")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer in layers:
            attn = layer.self_attn
            for name in (
                "_mymethod_counts_kv", "_kvmerge_captured_attention_score",
                "_kvmerge_captured_score_log_scale",
            ):
                if hasattr(attn, name):
                    delattr(attn, name)


def make_cache_factory(model, variant: str, args, cfg_item: Dict, stats: dict):
    def _factory():
        clear_sequence_state(model)
        configure_keepkv_score_capture(
            model, score_window=args.keepkv_score_window, chunk_size=args.chunk_size
        )
        return ProportionalCohereCache(
            model=model,
            variant=variant,
            compress_ratio=args.compress_ratio,
            min_kv_len=args.min_kv_len,
            compress_every=args.compress_every,
            route_lambda=cfg_item["route_lambda"],
            conflict_lambda=cfg_item["conflict_lambda"],
            original_lambda=cfg_item["original_lambda"],
            calibration_scales=args._calibration_scales_value,
            sink_size=args.keepkv_sink_size,
            recent_window=args.keepkv_recent_protect_size,
            score_window=args.keepkv_score_window,
            recency_beta=args.keepkv_recency_beta,
            distance_chunk_size=args.distance_chunk_size,
            stats=stats,
        )
    return _factory


def source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = build_parser().parse_args()
    if str(args.attn_implementation).lower() != "eager":
        raise ValueError("compressed CohereKV PPL requires eager attention")
    if not 0.0 < float(args.compress_ratio) < 1.0:
        raise ValueError("compress_ratio must be in (0,1)")
    if int(args.model_seqlen) <= 1 or int(args.chunk_size) <= 0:
        raise ValueError("model_seqlen/chunk_size must be positive")
    if int(args.keepkv_score_window) > int(args.chunk_size):
        raise ValueError("use keepkv_score_window <= chunk_size so current importance is observed")
    variants = parse_variants(args.variants)
    configs = build_ofat_configs(args)
    calibration_scales = load_calibration_scales(args.calibration_scale_file, args.model)
    args._calibration_scales_value = calibration_scales

    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    validate_calibration_scales(calibration_scales, config)
    if str(config.model_type).lower() != "llama":
        raise ValueError("this focused PPL preexperiment currently supports Llama-family models only")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code, padding_side="left",
    )
    model_kwargs = dict(
        config=config,
        torch_dtype=parse_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if str(args.device_map).lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs); model.eval()
    if str(args.device_map).lower() == "none":
        model.to(torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"))
    device = first_execution_device(model, args.device)

    # User-requested repository function: imported and called directly.
    data = task_ppl.get_eval_data(args.dataset, int(args.model_seqlen), tokenizer, nsamples=int(args.nsamples))
    task_ppl._EVAL_MAX_SEQUENCES = int(args.max_eval_sequences)

    rows: List[Dict] = []
    t0 = time.time()
    result_cache: Dict[tuple, Dict] = {}
    print(f"PPL configs={len(configs)}; sweep={'on' if args.enable_hparam_sweep else 'off'}", flush=True)
    print(f"Modified/Merge-All frozen S_lh calibration shape={tuple(calibration_scales.shape)}; "
          f"min={float(calibration_scales.min()):.6g}, max={float(calibration_scales.max()):.6g}", flush=True)

    def eval_key(variant: str, cfg_item: Dict):
        if variant == "full":
            return ("full",)
        if variant == "eviction_only":
            # Figure-3-right OFAT only varies routing/conflict/original lambdas;
            # none of them affects the eviction-only control.
            return ("eviction_only",)
        if variant == "original_coherekv":
            return (variant, float(cfg_item["original_lambda"]))
        if variant == "merge_all":
            return (variant, float(cfg_item["route_lambda"]))
        return (variant, float(cfg_item["route_lambda"]), float(cfg_item["conflict_lambda"]))

    for cfg_item in configs:
        for variant in variants:
            key = eval_key(variant, cfg_item)
            reused = key in result_cache
            if not reused:
                print(f"=== repository-style PPL: {variant} / {cfg_item['config_id']} ===", flush=True)
                clear_sequence_state(model)
                stats: Dict = {}
                _sync_cuda(); tv = time.perf_counter()
                if variant == "full":
                    ppl = task_ppl.evaluate_perplexity(
                        model=model, data=data, seqlen=int(args.model_seqlen), device=device,
                        amp_dtype=parse_amp_dtype(args.amp_dtype), step_size=int(args.chunk_size),
                        cache_factory=None,
                    )
                else:
                    apply_mymethod_runtime_patch()
                    try:
                        cache_factory = make_cache_factory(model, variant, args, cfg_item, stats)
                        with force_attn_hidden_and_compress(model):
                            ppl = task_ppl.evaluate_perplexity(
                                model=model, data=data, seqlen=int(args.model_seqlen), device=device,
                                amp_dtype=parse_amp_dtype(args.amp_dtype), step_size=int(args.chunk_size),
                                cache_factory=cache_factory,
                            )
                    finally:
                        clear_sequence_state(model)
                        remove_mymethod_runtime_patch()
                _sync_cuda(); wall = time.perf_counter() - tv
                routed = int(stats.get("routed_total", 0))
                accepted = int(stats.get("accepted_total", 0))
                lens = list(stats.get("compressed_kv_lens", []))
                result_cache[key] = {
                    "ppl": float(ppl),
                    "wall_seconds": float(wall),
                    "num_compressions": int(stats.get("num_compressions", 0)),
                    "mean_compressed_kv_len": float(np.mean(lens)) if lens else 0.0,
                    "routed_total": routed,
                    "accepted_total": accepted,
                    "accepted_given_routed": float(accepted / max(routed, 1)) if variant in {"merge_all", "modified_coherekv"} else float("nan"),
                    "conflict_edges_total": int(stats.get("conflict_edges_total", 0)),
                }
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            base_result = dict(result_cache[key])
            rows.append({
                "config_id": cfg_item["config_id"],
                "is_primary": int(cfg_item["is_primary"]),
                "variant": variant,
                **base_result,
                "reused_result": int(reused),
                "model_seqlen": int(args.model_seqlen),
                "chunk_size": int(args.chunk_size),
                "compress_ratio": float(args.compress_ratio),
                "retained_ratio": 1.0 - float(args.compress_ratio),
                "route_lambda": float(cfg_item["route_lambda"]),
                "conflict_lambda": float(cfg_item["conflict_lambda"]),
                "original_lambda": float(cfg_item["original_lambda"]),
                "calibration_scale_min": float(calibration_scales.min()),
                "calibration_scale_max": float(calibration_scales.max()),
            })
    clear_sequence_state(model)

    write_csv(out_dir / "ppl_summary.csv", rows)
    evaluator_path = TASK_EVAL_ROOT / "evaluate_ppl_kvmerge.py"
    cfg = vars(args).copy(); cfg.pop("_calibration_scales_value", None); cfg.update({
        "experiment": "coherekv_focused_repository_ppl",
        "protocol": "task_evaluation/evaluate_ppl_kvmerge.py fixed-length PPL",
        "evaluate_perplexity": "imported and called directly; source file not modified",
        "evaluate_ppl_kvmerge_sha256": source_sha256(evaluator_path),
        "budget_semantics": "dynamic target ~= (1-compress_ratio)*logical_seen",
        "calibration_scale_file": str(Path(args.calibration_scale_file).resolve()),
        "calibration_scale_shape": list(calibration_scales.shape),
        "modified_scale_protocol": "multiple calibration sequences; RMS-aggregated frozen per-(layer,KV-head) S_lh reused for every PPL compression round",
        "modified_conflict": "current-round ||K_u-K_v||_2/S_lh^calib > conflict_lambda",
        "modified_priority": "(-degree, -normalized_anchor_K_distance, -stable_hash, -position)",
        "engineering": "B*H layer batching; Merge-All/Modified share routing+aggregation; Modified adds packed conflict pools + tensorized MIS",
        "sweep_design": "one-factor-at-a-time with result de-duplication by effective variant parameters",
        "sweep_configs": configs,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "elapsed_seconds": time.time() - t0,
    })
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Focused CohereKV repository-style PPL",
        "=======================================",
        f"seqlen={args.model_seqlen}; step={args.chunk_size}; compress_ratio={args.compress_ratio}",
    ]
    for r in rows:
        suffix = ""
        if r["variant"] in {"merge_all", "modified_coherekv"}:
            suffix = f"; accepted/routed={100*r['accepted_given_routed']:.2f}%"
        tag = "PRIMARY" if int(r.get("is_primary", 0)) else r["config_id"]
        lines.append(
            f"[{tag}] {r['variant']}: PPL={r['ppl']:.6f}; wall={r['wall_seconds']:.2f}s; "
            f"compressions={r['num_compressions']}{suffix}"
        )
    (out_dir / "RESULT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
