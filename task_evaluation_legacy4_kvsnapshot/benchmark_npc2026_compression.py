#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU microbenchmark for compression cores with explicit timing boundaries.

This script deliberately avoids end-to-end generation latency. It measures only
post-prefill cache transformation and the mymethod routing/certificate core,
which is the defensible systems metric for the NPC paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn: Callable[[], object], device: torch.device, warmup: int, repeats: int):
    for _ in range(warmup):
        out = fn()
        del out
    sync(device)
    values = []
    if device.type == "cuda":
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
            del out
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            values.append((time.perf_counter() - t0) * 1000.0)
            del out
    sync(device)
    return {
        "latency_ms_mean": statistics.mean(values),
        "latency_ms_median": statistics.median(values),
        "latency_ms_stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "latency_ms_min": min(values),
        "latency_ms_max": max(values),
    }


def peak_bytes(fn: Callable[[], object], device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    out = fn()
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    del out
    sync(device)
    return max(0, int(peak - baseline))


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task_eval_dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_kv_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--lengths", default="512,1024,2048,4096")
    p.add_argument("--compress_ratios", default="0.2,0.4,0.6")
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--distance_chunk_sizes", default="0,256")
    p.add_argument("--grouplogitkv_sim_thr", type=float, default=0.8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", required=True)
    args = p.parse_args()

    task_dir = Path(args.task_eval_dir).resolve()
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))
    from models.common import mymethod_common as common
    from models.common.centroidkv_common import CentroidKVConfig, compress_layer
    from models.llama.merge_grouplogitkv_llama import _batched_group_merge_layer

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >=0 and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows: List[dict] = []
    for length in parse_ints(args.lengths):
        if length <= 1:
            continue
        b, h, d = args.batch_size, args.num_kv_heads, args.head_dim
        keys = torch.randn((b, h, length, d), device=device, dtype=dtype)
        values = torch.randn_like(keys)
        counts_i = torch.ones((b, h, length), device=device, dtype=torch.int32)
        radii = torch.zeros((b, h, length), device=device, dtype=torch.float32)
        diameters = torch.zeros_like(radii)
        positions = torch.arange(length, device=device, dtype=torch.int32).view(1, 1, -1).expand(b, h, -1)
        history = torch.zeros((b, h, 0, length), device=device, dtype=torch.float32)
        importance = torch.rand((b, h, length), device=device, dtype=torch.float32)

        for ratio in parse_floats(args.compress_ratios):
            keep_len = max(1, min(length, int(round(length * (1.0 - ratio)))))
            ne, nr = length - keep_len, keep_len
            if ne <= 0:
                continue
            for chunk in parse_ints(args.distance_chunk_sizes):
                def route_full():
                    x = b * h
                    return common.build_merge_plan(
                        keys.reshape(x, length, d), counts_i.reshape(x, length).float(),
                        radii.reshape(x, length), diameters.reshape(x, length),
                        importance.reshape(x, length), positions.reshape(x, length),
                        keep_len=keep_len, tau=args.tau, selector="grouplogitkv",
                        sink_size=0, recent_window=0, distance_chunk_size=chunk,
                        certify_groups=True, route_mode="nearest_feasible",
                    )

                def merge_full():
                    return common.merge_layer(
                        keys, values, counts_i, radii, diameters, history, 0, positions,
                        ratio=ratio, target_length=keep_len, tau=args.tau,
                        sink_size=0, recent_window=0, recency_beta=1.0,
                        distance_chunk_size=chunk, certify_groups=True,
                        sink_can_absorb=True, selector="grouplogitkv",
                        current_importance=importance, route_mode="nearest_feasible",
                    )

                def merge_no_cert():
                    return common.merge_layer(
                        keys, values, counts_i, radii, diameters, history, 0, positions,
                        ratio=ratio, target_length=keep_len, tau=args.tau,
                        sink_size=0, recent_window=0, recency_beta=1.0,
                        distance_chunk_size=chunk, certify_groups=False,
                        sink_can_absorb=True, selector="grouplogitkv",
                        current_importance=importance, route_mode="nearest_feasible",
                    )

                for variant, fn in (
                    ("mymethod_route_certified", route_full),
                    ("mymethod_full", merge_full),
                    ("mymethod_no_certificate", merge_no_cert),
                ):
                    stats = timed(fn, device, args.warmup, args.repeats)
                    peak = peak_bytes(fn, device)
                    row = {
                        "variant": variant,
                        "length": length,
                        "keep_len": keep_len,
                        "evicted_len": ne,
                        "compress_ratio": ratio,
                        "batch_size": b,
                        "num_kv_heads": h,
                        "head_dim": d,
                        "distance_chunk_size": chunk,
                        "pair_count": b * h * ne * nr,
                        "peak_temporary_mib": peak / (1024 ** 2),
                    }
                    row.update(stats)
                    row["million_pairs_per_second"] = (
                        row["pair_count"] / max(row["latency_ms_median"], 1e-12) / 1e3
                    )
                    rows.append(row)

            def grouplogit():
                return _batched_group_merge_layer(
                    keys, values, counts_i.float(), importance,
                    ratio=ratio, sim_thr=args.grouplogitkv_sim_thr,
                    target_length=keep_len,
                )

            centroid_config = CentroidKVConfig()
            def centroid():
                return compress_layer(keys, values, counts_i, keep_len, centroid_config)

            for variant, fn in (("GroupLogitKV", grouplogit), ("CentroidKV", centroid)):
                stats = timed(fn, device, args.warmup, args.repeats)
                peak = peak_bytes(fn, device)
                row = {
                    "variant": variant,
                    "length": length,
                    "keep_len": keep_len,
                    "evicted_len": ne,
                    "compress_ratio": ratio,
                    "batch_size": b,
                    "num_kv_heads": h,
                    "head_dim": d,
                    "distance_chunk_size": None,
                    "pair_count": b * h * ne * nr,
                    "peak_temporary_mib": peak / (1024 ** 2),
                }
                row.update(stats)
                row["million_pairs_per_second"] = (
                    row["pair_count"] / max(row["latency_ms_median"], 1e-12) / 1e3
                )
                rows.append(row)

        del keys, values, counts_i, radii, diameters, positions, history, importance
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {
        "config": vars(args),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "notes": (
            "Explicit post-prefill compression only. No tokenization, model prefill, "
            "decode, or end-to-end request latency is included."
        ),
    }
    output.write_text(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2))
    write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} benchmark rows to {output}")


if __name__ == "__main__":
    main()
