#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

from star_metrics import (
    aggregate_group_metrics,
    compute_group_metrics,
    deterministic_anchor_partition,
    parse_float_csv,
    parse_int_csv,
    resolve_selection,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neutral real-key star-structure pre-experiment.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--model_seqlen", type=int, default=8192)
    p.add_argument("--max_eval_sequences", type=int, default=1)
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--prefill_chunk_size", type=int, default=128)
    p.add_argument("--anchor_budgets", default="0.2,0.4,0.6,0.8")
    p.add_argument("--cosine_thresholds", default="0.70,0.80,0.90,0.95")
    p.add_argument("--record_tau", type=float, default=0.80)
    p.add_argument("--anchor_seeds", default="0,1,2")
    p.add_argument("--similarity_chunk_size", type=int, default=128)
    p.add_argument("--layer_selection", default="all")
    p.add_argument("--kv_head_selection", default="all")
    p.add_argument("--numeric_eps", type=float, default=1e-6)
    p.add_argument("--min_eligible_groups_warn", type=int, default=100)
    p.add_argument("--output_dir", required=True)
    return p


def parse_dtype(name: str):
    x = str(name).lower()
    if x == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if x not in mapping:
        raise ValueError(f"Unsupported torch dtype {name!r}.")
    return mapping[x]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_stream(tokenizer, dataset_name: str) -> torch.Tensor:
    if dataset_name != "wikitext2":
        raise ValueError("This pre-experiment currently supports only wikitext2.")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    return enc.input_ids


def cache_layers(past_key_values) -> List[torch.Tensor]:
    if past_key_values is None:
        raise RuntimeError("Model returned no past_key_values although use_cache=True.")
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)
    if isinstance(past_key_values, (tuple, list)):
        return [layer[0] for layer in past_key_values]
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [layer[0] for layer in legacy]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def first_execution_device(model, requested: str) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(requested)


def build_full_cache(model, input_ids: torch.Tensor, device: torch.device, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be > 0")
    past = None
    length = int(input_ids.shape[1])
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk = input_ids[:, start:end].to(device)
        with torch.inference_mode():
            out = model(
                input_ids=chunk,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        past = out.past_key_values
        del out, chunk
    return past


def route_candidates_all_heads(
    normalized_keys: torch.Tensor,
    anchor_positions: np.ndarray,
    candidate_positions: np.ndarray,
    similarity_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-anchor cosine routing for all KV heads of one layer.

    normalized_keys: [Hkv, L, d] float32 tensor.
    Returns CPU arrays [Hkv, Ncand] for local anchor index and max cosine.
    """
    if similarity_chunk_size <= 0:
        raise ValueError("similarity_chunk_size must be > 0")
    device = normalized_keys.device
    a_idx = torch.as_tensor(anchor_positions, dtype=torch.long, device=device)
    c_idx = torch.as_tensor(candidate_positions, dtype=torch.long, device=device)
    anchors = normalized_keys.index_select(1, a_idx)
    anchors_t = anchors.transpose(-1, -2).contiguous()
    h = int(normalized_keys.shape[0])
    n_c = int(c_idx.numel())
    assignment = torch.empty((h, n_c), dtype=torch.int64, device="cpu")
    max_cos = torch.empty((h, n_c), dtype=torch.float32, device="cpu")
    for start in range(0, n_c, similarity_chunk_size):
        end = min(start + similarity_chunk_size, n_c)
        cand = normalized_keys.index_select(1, c_idx[start:end])
        sim = torch.matmul(cand, anchors_t)
        vals, idx = sim.max(dim=-1)
        assignment[:, start:end] = idx.detach().cpu()
        max_cos[:, start:end] = vals.detach().float().cpu()
        del cand, sim, vals, idx
    del anchors, anchors_t, a_idx, c_idx
    return assignment.numpy(), max_cos.numpy()


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def is_close_tau(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_float_csv(args.anchor_budgets)
    thresholds = parse_float_csv(args.cosine_thresholds)
    seeds = parse_int_csv(args.anchor_seeds)
    for rho in budgets:
        if not 0.0 < rho < 1.0:
            raise ValueError(f"Every anchor budget must be in (0,1); got {rho}.")
    for tau in thresholds:
        if not -1.0 < tau < 1.0:
            raise ValueError(f"Every cosine threshold must be in (-1,1); got {tau}.")
    if not any(is_close_tau(args.record_tau, x) for x in thresholds):
        raise ValueError("--record_tau must be one of --cosine_thresholds so the plotted raw distribution is reproducible.")
    if args.model_seqlen < 3 or args.max_eval_sequences < 1:
        raise ValueError("model_seqlen must be >=3 and max_eval_sequences must be >=1.")

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
    print(f"Attention backend: {args.attn_implementation}", flush=True)
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
    total_full = int(data.shape[1]) // int(args.model_seqlen)
    n_seq = min(int(args.max_eval_sequences), total_full)
    if n_seq < 1:
        raise RuntimeError(
            f"Dataset has only {int(data.shape[1])} tokens, insufficient for model_seqlen={args.model_seqlen}."
        )
    print(
        f"Dataset={args.dataset}, tokens={int(data.shape[1])}, seqlen={args.model_seqlen}, sequences={n_seq}",
        flush=True,
    )

    summary_rows: List[Dict] = []
    record_lists: Dict[str, List] = defaultdict(list)

    t0 = time.time()
    selected_layers: Optional[List[int]] = None
    selected_heads: Optional[List[int]] = None

    for seq_idx in range(n_seq):
        s0 = seq_idx * args.model_seqlen
        s1 = s0 + args.model_seqlen
        seq_ids = data[:, s0:s1]
        print(f"\n[SEQUENCE {seq_idx}] building full uncompressed cache for tokens [{s0}, {s1})", flush=True)
        past = build_full_cache(model, seq_ids, input_device, args.prefill_chunk_size)
        layers = cache_layers(past)
        if selected_layers is None:
            selected_layers = resolve_selection(args.layer_selection, len(layers))
            first_key = layers[selected_layers[0]]
            n_heads = int(first_key.shape[1])
            selected_heads = resolve_selection(args.kv_head_selection, n_heads)
            print(f"Selected layers: {selected_layers}", flush=True)
            print(f"Selected KV heads: {selected_heads}", flush=True)

        assert selected_layers is not None and selected_heads is not None
        for layer_idx in selected_layers:
            key = layers[layer_idx]
            if key.ndim != 4 or key.shape[0] != 1:
                raise RuntimeError(f"Expected cache key [1,H,L,d], got {tuple(key.shape)} at layer {layer_idx}.")
            if int(key.shape[2]) != args.model_seqlen:
                raise RuntimeError(
                    f"Cache length mismatch at layer {layer_idx}: got {int(key.shape[2])}, expected {args.model_seqlen}."
                )
            key = key[0].float()
            key = F.normalize(key, p=2, dim=-1, eps=1e-12)
            head_index_tensor = torch.as_tensor(selected_heads, dtype=torch.long, device=key.device)
            key_selected = key.index_select(0, head_index_tensor)
            key_cpu = key_selected.detach().cpu().numpy().astype(np.float32, copy=False)

            for rho in budgets:
                for seed in seeds:
                    # Include sequence index in the effective seed so repeated sequences do not reuse the same partition.
                    effective_seed = int(seed) + 1_000_003 * int(seq_idx)
                    anchors, candidates = deterministic_anchor_partition(args.model_seqlen, rho, effective_seed)
                    assignment_all, maxcos_all = route_candidates_all_heads(
                        key_selected,
                        anchors,
                        candidates,
                        args.similarity_chunk_size,
                    )

                    for local_head_idx, head_idx in enumerate(selected_heads):
                        normalized_head = key_cpu[local_head_idx]
                        assignment = assignment_all[local_head_idx]
                        maxcos = maxcos_all[local_head_idx]

                        for tau in thresholds:
                            groups = compute_group_metrics(
                                normalized_head,
                                candidates,
                                assignment,
                                maxcos,
                                anchors,
                                tau=tau,
                                numeric_eps=args.numeric_eps,
                            )
                            agg = aggregate_group_metrics(groups)
                            assigned_candidates = int(np.count_nonzero(maxcos >= tau))
                            row = {
                                "model": args.model,
                                "dataset": args.dataset,
                                "sequence_index": seq_idx,
                                "layer": layer_idx,
                                "kv_head": head_idx,
                                "anchor_budget": rho,
                                "anchor_seed": seed,
                                "cosine_threshold": tau,
                                "num_tokens": args.model_seqlen,
                                "num_anchors": int(anchors.size),
                                "num_candidates": int(candidates.size),
                                "assigned_candidates": assigned_candidates,
                                **agg,
                            }
                            summary_rows.append(row)

                            if is_close_tau(tau, args.record_tau):
                                for g in groups:
                                    record_lists["sequence_index"].append(seq_idx)
                                    record_lists["layer"].append(layer_idx)
                                    record_lists["kv_head"].append(head_idx)
                                    record_lists["anchor_budget"].append(rho)
                                    record_lists["anchor_seed"].append(seed)
                                    record_lists["anchor_position"].append(g.anchor_position)
                                    record_lists["candidate_count"].append(g.candidate_count)
                                    record_lists["min_leaf_cosine"].append(g.min_leaf_cosine)
                                    record_lists["max_leaf_angle"].append(g.max_leaf_angle)
                                    record_lists["normalized_max_angle"].append(g.normalized_max_angle)
                                    record_lists["unsafe"].append(int(g.unsafe))
                                    record_lists["pair_count"].append(g.pair_count)
                                    record_lists["conflict_pair_count"].append(g.conflict_pair_count)
                                    record_lists["conflict_pair_fraction"].append(g.conflict_pair_fraction)
                                    record_lists["min_anchor_cosine"].append(g.min_anchor_cosine)
                                    record_lists["mean_anchor_cosine"].append(g.mean_anchor_cosine)

            del key, key_selected, head_index_tensor, key_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[sequence={seq_idx}] finished layer {layer_idx}", flush=True)

        del past, layers, seq_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-head/layer summary.
    summary_path = out_dir / "summary_by_head.csv"
    write_csv(summary_path, summary_rows)

    # Pooled summary using group counts as the denominator, not an unweighted mean of per-head rates.
    pooled: Dict[Tuple[float, float], Dict[str, float]] = {}
    for rho in budgets:
        for tau in thresholds:
            rows = [r for r in summary_rows if is_close_tau(r["anchor_budget"], rho) and is_close_tau(r["cosine_threshold"], tau)]
            eligible = int(sum(int(r["eligible_groups"]) for r in rows))
            unsafe = int(sum(int(r["unsafe_groups"]) for r in rows))
            pairs = int(sum(int(r["candidate_pairs"]) for r in rows))
            conflicts = int(sum(int(r["conflict_pairs"]) for r in rows))
            assigned = int(sum(int(r["assigned_candidates"]) for r in rows))
            num_candidates = int(sum(int(r["num_candidates"]) for r in rows))
            pooled[(rho, tau)] = {
                "model": args.model,
                "dataset": args.dataset,
                "anchor_budget": rho,
                "cosine_threshold": tau,
                "num_observation_units": len(rows),
                "eligible_groups": eligible,
                "unsafe_groups": unsafe,
                "unsafe_group_rate": float(unsafe / eligible) if eligible else float("nan"),
                "candidate_pairs": pairs,
                "conflict_pairs": conflicts,
                "conflict_pair_rate": float(conflicts / pairs) if pairs else float("nan"),
                "assigned_candidates": assigned,
                "candidate_acceptance_rate": float(assigned / num_candidates) if num_candidates else float("nan"),
            }
    overall_rows = [pooled[(rho, tau)] for rho in budgets for tau in thresholds]
    overall_path = out_dir / "summary_overall.csv"
    write_csv(overall_path, overall_rows)

    npz_path = raw_dir / "group_records_plot_tau.npz"
    npz_payload = {k: np.asarray(v) for k, v in record_lists.items()}
    npz_payload["record_tau"] = np.asarray([args.record_tau], dtype=np.float64)
    npz_payload["anchor_budgets"] = np.asarray(budgets, dtype=np.float64)
    np.savez_compressed(npz_path, **npz_payload)

    config_path = out_dir / "run_config.json"
    config_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "model_seqlen": args.model_seqlen,
        "max_eval_sequences": args.max_eval_sequences,
        "actual_eval_sequences": n_seq,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "prefill_chunk_size": args.prefill_chunk_size,
        "anchor_budgets": budgets,
        "cosine_thresholds": thresholds,
        "record_tau": args.record_tau,
        "anchor_seeds": seeds,
        "similarity_chunk_size": args.similarity_chunk_size,
        "layer_selection": selected_layers,
        "kv_head_selection": selected_heads,
        "numeric_eps": args.numeric_eps,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": time.time() - t0,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print("\n=== Pooled star statistics ===", flush=True)
    for rho in budgets:
        row = pooled[(rho, args.record_tau)]
        print(
            f"budget={rho:.2f} tau={args.record_tau:.3f} "
            f"eligible={row['eligible_groups']} unsafe={row['unsafe_groups']} "
            f"unsafe_rate={row['unsafe_group_rate']:.6f} "
            f"conflict_pair_rate={row['conflict_pair_rate']:.6f}",
            flush=True,
        )
        if int(row["eligible_groups"]) < args.min_eligible_groups_warn:
            print(
                f"[WARN] budget={rho:.2f} has only {row['eligible_groups']} eligible groups at tau={args.record_tau:.3f}; "
                "the CCDF may be noisy.",
                flush=True,
            )

    print(f"\nSaved: {summary_path}", flush=True)
    print(f"Saved: {overall_path}", flush=True)
    print(f"Saved: {npz_path}", flush=True)
    print(f"Saved: {config_path}", flush=True)


if __name__ == "__main__":
    main()
