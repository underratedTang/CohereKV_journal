#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Component and oracle ablations for the updated ``mymethod``.

The original comparison-method implementations are not modified.  These
ablations only change mymethod components while preserving its updated
KeepKV-style recent-window selector unless a component is explicitly removed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch


VARIANTS = (
    "eviction_only",
    "certified_merge_no_count",
    "nearest_merge_no_certificate",
    "nearest_anchor_then_certify",
    "full_no_sink_absorb",
    "full",
)


def _legacy(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def _ensure_import_paths(task_eval_dir: Path, family: str) -> None:
    root = task_eval_dir.resolve()
    method_dir = root / "models" / family
    for path in (root, method_dir):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _mymethod_module(task_eval_dir: Path, family: str):
    _ensure_import_paths(task_eval_dir, family)
    return importlib.import_module(f"merge_mymethod_{family}")


def _common_module(task_eval_dir: Path):
    root = task_eval_dir.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("models.common.mymethod_common")


def _attention_module(model, layer_idx: int):
    layer = model.model.layers[layer_idx]
    return getattr(layer, "self_attn", getattr(layer, "attention", None))


def _disable_count_calibration(model) -> None:
    """Keep true counts for future reductions but expose unit counts to attention."""
    counts = getattr(model, "_mymethod_counts", None)
    if counts is None:
        return
    for layer_idx, count in enumerate(counts):
        attn = _attention_module(model, layer_idx)
        if attn is not None:
            attn._mymethod_counts_kv = torch.ones_like(count)


def _disable_mymethod_runtime(model) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer_idx, _ in enumerate(model.model.layers):
        attn = _attention_module(model, layer_idx)
        if attn is None:
            continue
        attn._mymethod_eager_enabled = False
        attn._mymethod_sdpa_enabled = False
        attn._mymethod_counts_kv = None


def _eviction_only(
    model,
    past_key_values,
    attention,
    ratio: float,
    task_eval_dir: Path,
    sink_size: int,
    recent_window: int,
    recency_beta: float,
    selector: str,
    target_length=None,
):
    """Use the updated selector, but discard all evicted entries without merging."""
    common = _common_module(task_eval_dir)
    cache = _legacy(past_key_values)
    if attention is None:
        raise ValueError("eviction_only requires mymethod recent base-score rows")

    new_past = []
    new_counts = []
    new_radii = []
    new_diameters = []
    new_history = []
    new_ptrs = []
    new_scales = []
    new_positions = []
    new_next_positions = []

    for layer_idx, (keys, values) in enumerate(cache):
        bsz, h_kv, length, dim = keys.shape
        if target_length is not None:
            keep_len = min(max(int(target_length), 1), length)
        else:
            num_evict = min(max(int(length * float(ratio)), 0), max(length - 1, 0))
            keep_len = length - num_evict

        selector = common.normalize_selector(selector)
        score_window = recent_window if selector == "keepkv" else 0
        counts, radii, diameters, history, ptr, positions, next_position, old_scale = (
            common.initial_or_existing_state(model, layer_idx, keys, score_window)
        )
        flat_positions = positions.reshape(bsz * h_kv, length)
        if selector == "keepkv":
            rows = common.base_score_rows_for_kv_heads(attention[layer_idx], h_kv).to(
                device=keys.device, dtype=torch.float32
            )
            attn = _attention_module(model, layer_idx)
            current_scale = float(getattr(attn, "_kvmerge_captured_score_log_scale", 0.0))
            history, ptr, common_scale = common.append_score_rows(
                history, ptr, old_scale, rows, current_scale
            )
            importance = common.recent_window_importance(
                history.reshape(bsz * h_kv, history.shape[-2], length),
                ptr, recency_beta
            )
        else:
            importance = common.grouplogitkv_importance_for_kv_heads(
                attention[layer_idx], h_kv
            ).reshape(bsz * h_kv, length).float()
            common_scale = None
        if keep_len >= length:
            flat_keep = torch.arange(length, device=keys.device).view(1, length)
            flat_keep = flat_keep.expand(bsz * h_kv, -1)
        elif selector == "grouplogitkv":
            flat_keep = common.select_keep_indices_grouplogitkv(importance, keep_len)
        else:
            flat_keep = common.select_keep_indices(
                importance,
                flat_positions,
                keep_len,
                sink_size=sink_size,
                recent_window=recent_window,
            )
        keep_idx = flat_keep.reshape(bsz, h_kv, keep_len)
        gather_vec = keep_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        new_past.append((
            torch.gather(keys, 2, gather_vec).contiguous(),
            torch.gather(values, 2, gather_vec).contiguous(),
        ))
        new_counts.append(torch.gather(counts, 2, keep_idx).to(torch.int32).contiguous())
        new_radii.append(torch.gather(radii, 2, keep_idx).float().contiguous())
        new_diameters.append(torch.gather(diameters, 2, keep_idx).float().contiguous())
        hist_idx = keep_idx.unsqueeze(-2).expand(-1, -1, history.shape[-2], -1)
        new_history.append(torch.gather(history, 3, hist_idx).float().contiguous())
        new_ptrs.append(ptr)
        new_scales.append(common_scale)
        new_positions.append(torch.gather(positions, 2, keep_idx).contiguous())
        new_next_positions.append(next_position)

    model._mymethod_counts = new_counts
    model._mymethod_radii = new_radii
    model._mymethod_diameters = new_diameters
    model._mymethod_score_history = new_history
    model._mymethod_score_history_ptrs = new_ptrs
    model._mymethod_score_log_scales = new_scales
    model._mymethod_positions = new_positions
    model._mymethod_next_positions = new_next_positions
    for layer in model.model.layers:
        layer.self_attn._mymethod_selector = selector
        layer.self_attn._mymethod_score_window = recent_window if selector == "keepkv" else 0
    _disable_mymethod_runtime(model)
    return tuple(new_past)


def _apply_direct(
    variant: str,
    family: str,
    model,
    past_key_values,
    attention,
    hidden_states,
    compress_ratio: float,
    task_eval_dir: Path,
    config: dict,
    target_kv_len=None,
):
    if variant == "eviction_only":
        return _eviction_only(
            model,
            past_key_values,
            attention,
            compress_ratio,
            task_eval_dir,
            sink_size=int(config["sink_size"]),
            recent_window=int(config["recent_window"]),
            recency_beta=float(config["recency_beta"]),
            selector=str(config.get("selector", "keepkv")),
            target_length=target_kv_len,
        )

    module = _mymethod_module(task_eval_dir, family)
    result = module.Repair(
        model,
        past_key_values,
        attention,
        hidden_states,
        compress_ratio,
        float(config["tau"]),
        target_length=target_kv_len,
        sink_size=int(config["sink_size"]),
        recent_window=int(config["recent_window"]),
        recency_beta=float(config["recency_beta"]),
        distance_chunk_size=int(config["distance_chunk_size"]),
        certify_groups=(variant != "nearest_merge_no_certificate"),
        sink_can_absorb=(
            False if variant == "full_no_sink_absorb"
            else bool(config["sink_can_absorb"])
        ),
        selector=str(config.get("selector", "keepkv")),
        route_mode=(
            "nearest_then_certify"
            if variant == "nearest_anchor_then_certify"
            else "nearest_feasible"
        ),
    )
    if variant == "certified_merge_no_count":
        _disable_count_calibration(model)
    return result


def run_ppl(variant: str, forwarded: List[str]) -> None:
    import evaluate_ppl_kvmerge as ppl

    def patched(
        method,
        family,
        model,
        past_key_values,
        attention,
        hidden_states,
        compress_ratio,
        method_sim_thrs=None,
        keepkv_config=None,
        task_eval_dir=None,
        target_kv_len=None,
        mymethod_config=None,
        centroidkv_config=None,
        **unused_kwargs,
    ):
        del method, method_sim_thrs, keepkv_config, centroidkv_config, unused_kwargs
        config = {
            "selector": "keepkv",
            "tau": 0.7,
            "sink_size": 4,
            "recent_window": 128,
            "recency_beta": 1.0,
            "distance_chunk_size": 256,
            "sink_can_absorb": True,
        }
        config.update(mymethod_config or {})
        root = Path(task_eval_dir or Path(__file__).resolve().parent)
        return _apply_direct(
            variant, family, model, past_key_values, attention, hidden_states,
            compress_ratio, root, config, target_kv_len=target_kv_len,
        )

    ppl.apply_kv_method = patched
    sys.argv = ["evaluate_ppl_kvmerge.py", *forwarded]
    if "--kv_method" not in forwarded:
        sys.argv += ["--kv_method", "mymethod"]
    ppl.main()


def run_longbench(variant: str, forwarded: List[str]) -> None:
    forwarded = [x for x in forwarded if x != "--overwrite"]
    import evaluate_longbench_kvmerge as lb

    original_setup = lb.setup_kv_imports

    def patched_setup(
        task_eval_dir,
        arch,
        kv_method,
        simcalkv_similarity_chunk_size=256,
        mymethod_tau=0.7,
        mymethod_selector="keepkv",
        mymethod_sink_size=4,
        mymethod_recent_window=128,
        mymethod_recency_beta=1.0,
        mymethod_distance_chunk_size=256,
        mymethod_sink_can_absorb=True,
    ):
        del kv_method
        _, reset = original_setup(
            task_eval_dir,
            arch,
            "mymethod",
            simcalkv_similarity_chunk_size=simcalkv_similarity_chunk_size,
            mymethod_tau=mymethod_tau,
            mymethod_selector=mymethod_selector,
            mymethod_sink_size=mymethod_sink_size,
            mymethod_recent_window=mymethod_recent_window,
            mymethod_recency_beta=mymethod_recency_beta,
            mymethod_distance_chunk_size=mymethod_distance_chunk_size,
            mymethod_sink_can_absorb=mymethod_sink_can_absorb,
        )
        config = {
            "selector": str(mymethod_selector),
            "tau": float(mymethod_tau),
            "sink_size": int(mymethod_sink_size),
            "recent_window": int(mymethod_recent_window),
            "recency_beta": float(mymethod_recency_beta),
            "distance_chunk_size": int(mymethod_distance_chunk_size),
            "sink_can_absorb": bool(mymethod_sink_can_absorb),
        }
        root = Path(task_eval_dir)

        def wrapped(method, model, pkv, attn, hidden, ratio, sim_thr=None):
            del method, sim_thr
            return _apply_direct(
                variant, arch, model, pkv, attn, hidden, ratio, root, config
            )

        return wrapped, reset

    lb.setup_kv_imports = patched_setup
    sys.argv = ["evaluate_longbench_kvmerge.py", *forwarded]
    if "--kv_method" not in forwarded:
        sys.argv += ["--kv_method", "mymethod"]
    lb.main()


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    if spec.strip().lower() == "auto":
        values = [0, n_layers // 3, (2 * n_layers) // 3, n_layers - 1]
    else:
        values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    values = sorted(set(x if x >= 0 else n_layers + x for x in values))
    if not values or values[0] < 0 or values[-1] >= n_layers:
        raise ValueError(f"Invalid --oracle_layers={spec!r} for {n_layers} layers")
    return values


def _first_round_history(
    common,
    score_rows: torch.Tensor,
    window: int,
    recency_beta: float,
):
    bsz, h_kv, q_len, length = score_rows.shape
    history = torch.zeros(
        (bsz, h_kv, max(int(window), 0), length),
        device=score_rows.device,
        dtype=torch.float32,
    )
    if window > 0:
        tail = score_rows[..., -window:, :].float()
        history[..., -tail.shape[-2]:, :] = tail
    importance = common.recent_window_importance(
        history.reshape(bsz * h_kv, history.shape[-2], length),
        0,
        recency_beta,
    )
    return history, importance


def _build_certified_groups(
    common,
    keys: torch.Tensor,
    score_rows: torch.Tensor,
    ratio: float,
    tau: float,
    sink_size: int,
    recent_window: int,
    recency_beta: float,
    distance_chunk_size: int,
    sink_can_absorb: bool,
    selector: str,
) -> List[List[List[int]]]:
    """Return first-round groups[row][retained_slot] as original token indices."""
    bsz, h_kv, length, dim = keys.shape
    num_evict = min(max(int(length * ratio), 0), length - 1)
    keep_len = length - num_evict
    x = bsz * h_kv
    selector = common.normalize_selector(selector)
    if selector == "grouplogitkv":
        importance = common.grouplogitkv_importance_for_kv_heads(
            score_rows, h_kv
        ).reshape(x, length)
    else:
        _, importance = _first_round_history(
            common, score_rows, recent_window, recency_beta
        )
    positions = torch.arange(length, device=keys.device, dtype=torch.long)
    positions = positions.view(1, length).expand(x, -1)
    counts = torch.ones((x, length), device=keys.device, dtype=torch.float32)
    zeros = torch.zeros_like(counts)
    plan = common.build_merge_plan(
        keys=keys.reshape(x, length, dim),
        counts=counts,
        radii=zeros,
        diameters=zeros,
        importance=importance,
        positions=positions,
        keep_len=keep_len,
        tau=tau,
        sink_size=sink_size,
        recent_window=recent_window,
        distance_chunk_size=distance_chunk_size,
        certify_groups=True,
        sink_can_absorb=sink_can_absorb,
        selector=selector,
    )

    groups: List[List[List[int]]] = []
    for row in range(x):
        row_groups = [[int(v)] for v in plan.keep_idx[row].tolist()]
        for evict_slot, original_idx in enumerate(plan.evict_idx[row].tolist()):
            if bool(plan.accepted[row, evict_slot]):
                row_groups[int(plan.best_idx[row, evict_slot])].append(int(original_idx))
        groups.append(row_groups)
    return groups


def _oracle_prompt(sample: Dict, prompt_format: str, tokenizer, model_name: str, input_length: int):
    import evaluate_longbench_kvmerge as lb
    text = lb.build_chat(tokenizer, prompt_format.format(**sample), model_name)
    ids = tokenizer(text, add_special_tokens=True, truncation=False, return_tensors="pt").input_ids
    if ids.shape[1] < input_length:
        return None
    return ids[:, :input_length].contiguous()


def run_oracle(args) -> None:
    import evaluate_longbench_kvmerge as lb

    lb.seed_everything(args.seed)
    task_eval_dir = Path(args.task_eval_dir or Path(__file__).resolve().parent).resolve()
    arch = args.arch if args.arch != "auto" else lb.infer_arch(args.model, args.model_path)
    _ensure_import_paths(task_eval_dir, arch)
    common = _common_module(task_eval_dir)
    model2path = json.load(open(task_eval_dir / "LongBench/config/model2path.json", "r"))
    prompt_map = json.load(open(task_eval_dir / "LongBench/config/dataset2prompt.json", "r"))
    model_path = args.model_path or model2path[args.model]
    data = lb.load_longbench_dataset(args.oracle_dataset, task_eval_dir)
    if args.oracle_dataset not in prompt_map:
        raise KeyError(f"No prompt format for {args.oracle_dataset}")

    _, reset_kv = lb.setup_kv_imports(
        task_eval_dir,
        arch,
        "mymethod",
        mymethod_tau=args.tau_mymethod,
        mymethod_selector=args.mymethod_selector,
        mymethod_sink_size=args.mymethod_sink_size,
        mymethod_recent_window=args.mymethod_recent_window,
        mymethod_recency_beta=args.mymethod_recency_beta,
        mymethod_distance_chunk_size=args.mymethod_distance_chunk_size,
        mymethod_sink_can_absorb=bool(args.mymethod_sink_can_absorb),
    )
    model, tokenizer = lb.load_model_and_tokenizer(
        model_path, args.model, args.device, lb.get_torch_dtype(args.dtype),
        args.attn_implementation,
    )
    model.eval()
    for attn_mod in lb._self_attention_modules(model):
        attn_mod._kvmerge_attention_chunk_size = int(args.mymethod_attention_chunk_size)
        attn_mod._mymethod_selector = str(args.mymethod_selector)
        attn_mod._mymethod_score_window = (
            int(args.mymethod_recent_window) if args.mymethod_selector == "keepkv" else 0
        )

    layers = _parse_layers(args.oracle_layers, len(model.model.layers))
    qualified = []
    for idx, sample in enumerate(data):
        ids = _oracle_prompt(
            sample, prompt_map[args.oracle_dataset], tokenizer, args.model,
            args.oracle_input_length,
        )
        if ids is not None:
            qualified.append((idx, ids))
            if len(qualified) >= args.oracle_num_prompts:
                break
    if len(qualified) < args.oracle_num_prompts:
        raise RuntimeError(f"Found only {len(qualified)} qualified prompts; need {args.oracle_num_prompts}")

    rng = random.Random(args.seed)
    records = []
    eps = 1e-30
    for sample_no, (dataset_idx, ids_cpu) in enumerate(qualified):
        ids = ids_cpu.to(args.device)
        mask = torch.ones_like(ids)
        lb._set_reduced_attention_capture(model, True, reduce="sum")
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=ids, attention_mask=mask, use_cache=True,
                    output_attentions=False, return_dict=True,
                )
            scores_all = lb._get_reduced_attention_scores(model)
        finally:
            lb._set_reduced_attention_capture(model, False, reduce="sum")

        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prompt_cache = _legacy(out.past_key_values)
        with torch.inference_mode():
            qout = model(
                input_ids=first,
                attention_mask=torch.ones((1, ids.shape[1] + 1), dtype=mask.dtype, device=args.device),
                position_ids=torch.tensor([[ids.shape[1]]], dtype=torch.long, device=args.device),
                past_key_values=out.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        for layer_idx in layers:
            keys = prompt_cache[layer_idx][0]
            rows = common.base_score_rows_for_kv_heads(scores_all[layer_idx], keys.shape[1])
            groups = _build_certified_groups(
                common, keys, rows, args.compress_ratio, args.tau_mymethod,
                args.mymethod_sink_size, args.mymethod_recent_window,
                args.mymethod_recency_beta, args.mymethod_distance_chunk_size,
                bool(args.mymethod_sink_can_absorb), args.mymethod_selector,
            )
            attn = qout.attentions[layer_idx][..., : ids.shape[1]].float().clamp_min(eps)
            logp = attn.log()[0, :, 0, :]
            h_q = logp.shape[0]
            h_kv = keys.shape[1]
            q_per_kv = h_q // h_kv
            candidates = []
            for qh in range(h_q):
                kvh = qh // q_per_kv
                for group in groups[kvh]:
                    if len(group) <= 1:
                        continue
                    index = torch.tensor(group, device=logp.device)
                    vals = logp[qh, index]
                    centroid = vals.mean()
                    oracle = torch.logsumexp(vals, dim=0)
                    calibrated = centroid + math.log(len(group))
                    candidates.append((
                        qh, kvh, len(group),
                        float(abs(centroid - oracle)),
                        float(abs(calibrated - oracle)),
                    ))
            if len(candidates) > args.oracle_pairs_per_layer:
                candidates = rng.sample(candidates, args.oracle_pairs_per_layer)
            for qh, kvh, count, err0, err1 in candidates:
                records.append({
                    "sample": sample_no,
                    "dataset_index": dataset_idx,
                    "layer": layer_idx,
                    "query_head": qh,
                    "kv_head": kvh,
                    "group_count": count,
                    "centroid_abs_error": err0,
                    "calibrated_abs_error": err1,
                })

        reset_kv(model)
        lb._clear_reduced_attention_scores(model)
        del out, qout, prompt_cache, scores_all

    if not records:
        raise RuntimeError("No certified merged groups were sampled. Lower tau or increase compression.")

    def mean(key):
        return sum(r[key] for r in records) / len(records)

    centroid_mae = mean("centroid_abs_error")
    calibrated_mae = mean("calibrated_abs_error")
    per_layer = []
    for layer in layers:
        rows = [r for r in records if r["layer"] == layer]
        if rows:
            per_layer.append({
                "layer": layer,
                "pairs": len(rows),
                "centroid_only_mae": sum(r["centroid_abs_error"] for r in rows) / len(rows),
                "count_calibrated_mae": sum(r["calibrated_abs_error"] for r in rows) / len(rows),
            })

    result = {
        "task": "oracle_mymethod_group_logit",
        "model_name": args.model,
        "model_id": model_path,
        "arch": arch,
        "dataset": args.oracle_dataset,
        "input_length": args.oracle_input_length,
        "compress_ratio": args.compress_ratio,
        "retained_ratio": 1.0 - args.compress_ratio,
        "tau_mymethod": args.tau_mymethod,
        "mymethod_selector": args.mymethod_selector,
        "mymethod_sink_size": args.mymethod_sink_size,
        "mymethod_recent_window": args.mymethod_recent_window,
        "mymethod_recency_beta": args.mymethod_recency_beta,
        "mymethod_sink_can_absorb": bool(args.mymethod_sink_can_absorb),
        "layers": layers,
        "num_prompts": args.oracle_num_prompts,
        "num_query_group_pairs": len(records),
        "centroid_only_mae": centroid_mae,
        "count_calibrated_mae": calibrated_mae,
        "relative_mae_reduction": (
            (centroid_mae - calibrated_mae) / centroid_mae if centroid_mae else 0.0
        ),
        "per_layer": per_layer,
        "definition": {
            "groups": "first-round groups passing complete-diameter feasibility",
            "selector": "sink/recent/recent-score heavy hitters",
            "oracle": "logsumexp of original member logits",
            "centroid_only": "mean of original member logits",
            "count_calibrated": "mean of original member logits + log(group count)",
            "query": "first autoregressive decode query over the uncompressed prompt cache",
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _oracle_parser():
    parser = argparse.ArgumentParser(description="mymethod oracle group-logit ablation")
    parser.add_argument("--task", choices=["oracle"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--task_eval_dir")
    parser.add_argument("--arch", choices=["llama", "qwen", "auto"], default="auto")
    parser.add_argument("--compress_ratio", type=float, default=0.6)
    parser.add_argument("--tau_mymethod", type=float, default=0.7)
    parser.add_argument("--mymethod_selector", choices=["keepkv", "grouplogitkv"], default="keepkv")
    parser.add_argument("--mymethod_sink_size", type=int, default=4)
    parser.add_argument("--mymethod_recent_window", type=int, default=128)
    parser.add_argument("--mymethod_recency_beta", type=float, default=1.0)
    parser.add_argument("--mymethod_distance_chunk_size", type=int, default=256)
    parser.add_argument("--mymethod_attention_chunk_size", type=int, default=128)
    parser.add_argument("--mymethod_sink_can_absorb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--oracle_dataset", default="gov_report")
    parser.add_argument("--oracle_input_length", type=int, default=8192)
    parser.add_argument("--oracle_num_prompts", type=int, default=8)
    parser.add_argument("--oracle_layers", default="auto")
    parser.add_argument("--oracle_pairs_per_layer", type=int, default=512)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    return parser


def main() -> None:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=["ppl", "longbench", "oracle"], required=True)
    selector.add_argument("--variant", choices=VARIANTS)
    selected, forwarded = selector.parse_known_args()
    if selected.task == "oracle":
        args = _oracle_parser().parse_args()
        if args.mymethod_recency_beta < 1.0:
            raise ValueError("--mymethod_recency_beta must be >= 1")
        run_oracle(args)
        return
    if selected.variant is None:
        selector.error("--task ppl/longbench requires --variant")
    if selected.task == "ppl":
        run_ppl(selected.variant, forwarded)
    else:
        run_longbench(selected.variant, forwarded)


if __name__ == "__main__":
    main()
