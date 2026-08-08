#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce a torch.profiler trace for the mymethod compression core."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task_eval_dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_kv_heads", type=int, default=8)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--compress_ratio", type=float, default=0.4)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--distance_chunk_size", type=int, default=0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--trace_json", required=True)
    p.add_argument("--table_txt", required=True)
    args = p.parse_args()

    root = Path(args.task_eval_dir).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.common import mymethod_common as common

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    b, h, l, d = args.batch_size, args.num_kv_heads, args.length, args.head_dim
    keep = max(1, int(round(l * (1.0 - args.compress_ratio))))
    keys = torch.randn((b, h, l, d), device=device, dtype=dtype)
    values = torch.randn_like(keys)
    counts = torch.ones((b, h, l), device=device, dtype=torch.int32)
    radii = torch.zeros((b, h, l), device=device, dtype=torch.float32)
    diameters = torch.zeros_like(radii)
    history = torch.zeros((b, h, 0, l), device=device, dtype=torch.float32)
    positions = torch.arange(l, device=device, dtype=torch.int32).view(1, 1, -1).expand(b, h, -1)
    importance = torch.rand((b, h, l), device=device, dtype=torch.float32)

    def run_once():
        return common.merge_layer(
            keys, values, counts, radii, diameters, history, 0, positions,
            ratio=args.compress_ratio, target_length=keep, tau=args.tau,
            sink_size=0, recent_window=0, recency_beta=1.0,
            distance_chunk_size=args.distance_chunk_size, certify_groups=True,
            sink_can_absorb=True, selector="grouplogitkv",
            current_importance=importance, route_mode="nearest_feasible",
        )

    for _ in range(3):
        out = run_once()
        del out
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            with record_function("mymethod_full_compression"):
                out = run_once()
                del out
    if device.type == "cuda":
        torch.cuda.synchronize()

    trace = Path(args.trace_json)
    table = Path(args.table_txt)
    trace.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    text = prof.key_averages(group_by_input_shape=True).table(
        sort_by=sort_by, row_limit=80
    )
    table.write_text(text, encoding="utf-8")
    print(text)
    print(f"Trace: {trace}")
    print(f"Table: {table}")


if __name__ == "__main__":
    main()
