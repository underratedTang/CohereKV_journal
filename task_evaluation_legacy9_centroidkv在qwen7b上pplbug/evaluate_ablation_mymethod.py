#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused ablations for graph-MIS mymethod.

Variants intentionally isolate only the new method's core design:
  * eviction_only: selector only; all evicted entries are dropped;
  * anchor_only: pair-compatible nearest-anchor routing, but no candidate-candidate MIS;
  * full_no_count: full graph-MIS grouping, but decode-time log(C) is disabled;
  * full: complete method.

All public comparison methods stay in the ordinary PPL/LongBench/system scripts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

VARIANTS = ("eviction_only", "anchor_only", "full_no_count", "full")


def _attention_module(model, layer_idx: int):
    layer = model.model.layers[layer_idx]
    return getattr(layer, "self_attn", getattr(layer, "attention", None))


def _disable_count_calibration(model) -> None:
    counts = getattr(model, "_mymethod_counts", None)
    if counts is None:
        return
    for layer_idx, count in enumerate(counts):
        attn = _attention_module(model, layer_idx)
        if attn is not None:
            attn._mymethod_counts_kv = torch.ones_like(count)


def _ensure_arg(argv, name: str, value: str):
    if name in argv:
        idx = argv.index(name)
        if idx + 1 >= len(argv):
            raise ValueError(f"Missing value after {name}")
        argv[idx + 1] = value
    else:
        argv.extend([name, value])


def _variant_group_mode(variant: str) -> str:
    if variant == "eviction_only":
        return "eviction_only"
    if variant == "anchor_only":
        return "anchor_only"
    return "full"


def run_ppl(variant: str, forwarded: list[str]) -> None:
    import evaluate_ppl_kvmerge as ppl
    _ensure_arg(forwarded, "--kv_method", "mymethod")
    _ensure_arg(forwarded, "--mymethod_group_mode", _variant_group_mode(variant))
    if variant == "full_no_count":
        original = ppl.apply_kv_method
        def patched(*args, **kwargs):
            result = original(*args, **kwargs)
            model = args[2] if len(args) > 2 else kwargs.get("model")
            if model is not None:
                _disable_count_calibration(model)
            return result
        ppl.apply_kv_method = patched
    sys.argv = ["evaluate_ppl_kvmerge.py", *forwarded]
    ppl.main()


def _patch_longbench_setup(lb, variant: str):
    original = lb.setup_kv_imports
    def patched(*args, **kwargs):
        apply, reset = original(*args, **kwargs)
        if variant != "full_no_count":
            return apply, reset
        def wrapped(method, model, pkv, attn, hidden, ratio, sim_thr):
            out = apply(method, model, pkv, attn, hidden, ratio, sim_thr)
            if str(method).lower() == "mymethod":
                _disable_count_calibration(model)
            return out
        return wrapped, reset
    lb.setup_kv_imports = patched


def run_longbench(variant: str, forwarded: list[str]) -> None:
    import evaluate_longbench_kvmerge as lb
    _ensure_arg(forwarded, "--kv_method", "mymethod")
    _ensure_arg(forwarded, "--mymethod_group_mode", _variant_group_mode(variant))
    _patch_longbench_setup(lb, variant)
    sys.argv = ["evaluate_longbench_kvmerge.py", *forwarded]
    lb.main()


def run_system(variant: str, forwarded: list[str]) -> None:
    import evaluate_longbench_kvmerge as lb
    _patch_longbench_setup(lb, variant)
    import evaluate_system_kvmerge as system
    _ensure_arg(forwarded, "--kv_method", "mymethod")
    _ensure_arg(forwarded, "--mymethod_group_mode", _variant_group_mode(variant))
    sys.argv = ["evaluate_system_kvmerge.py", *forwarded]
    system.main()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=["ppl", "longbench", "system"], default="ppl")
    args, forwarded = parser.parse_known_args()
    if args.mode == "ppl":
        run_ppl(args.variant, forwarded)
    elif args.mode == "longbench":
        run_longbench(args.variant, forwarded)
    else:
        run_system(args.variant, forwarded)


if __name__ == "__main__":
    main()
