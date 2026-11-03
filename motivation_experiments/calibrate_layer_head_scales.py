#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate frozen per-(layer, KV-head) Euclidean K scales from multiple sequences.

For each independent calibration sequence ``m`` and every layer x KV head:

    S_lh,m^2 = 2/(L-1) * sum_i ||K_i - mean(K)||_2^2.

The operational frozen scale is the median across calibration sequences:

    S_lh^calib = median_m S_lh,m.

The full matrix ``S_lh^calib`` is written once and reused unchanged for every
later prefix/compression round.  We additionally save the per-(layer, head)
mean/std/CV across calibration sequences as diagnostics.  A model-global RMS is
emitted only as a descriptive summary and is never used for routing/conflict.
"""

import argparse
import json
import math
import platform
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

_EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate frozen per-layer/head CohereKV K scales")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--calibration_split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--calibration_num_sequences", type=int, default=16)
    p.add_argument("--calibration_seqlen", type=int, default=8192)
    p.add_argument("--calibration_seed", type=int, default=42)
    p.add_argument("--prefill_chunk_size", type=int, default=256)
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_map", default="none")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast_tokenizer", action="store_true")
    p.add_argument("--output", required=True)
    return p


def parse_dtype(name: str):
    table = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float32": torch.float32, "fp32": torch.float32}
    key = str(name).lower()
    if key not in table:
        raise ValueError(name)
    return table[key]


def first_execution_device(model, requested: str) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(requested)


def cache_keys(past_key_values) -> List[torch.Tensor]:
    if past_key_values is None:
        raise RuntimeError("calibration forward returned no past_key_values")
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)
    if isinstance(past_key_values, (tuple, list)):
        return [layer[0] for layer in past_key_values]
    if hasattr(past_key_values, "to_legacy_cache"):
        return [layer[0] for layer in past_key_values.to_legacy_cache()]
    raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")


def load_calibration_token_stream(tokenizer, dataset: str, split: str) -> torch.Tensor:
    from datasets import load_dataset
    if dataset != "wikitext2":
        raise ValueError(dataset)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids.contiguous()


def select_disjoint_windows(
    ids: torch.Tensor,
    *,
    num_sequences: int,
    seqlen: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Pick reproducible non-overlapping windows from disjoint corpus segments."""
    if ids.ndim != 2 or int(ids.shape[0]) != 1:
        raise ValueError("ids must be [1,N]")
    m = int(num_sequences)
    L = int(seqlen)
    n = int(ids.shape[1])
    if m <= 0 or L < 2:
        raise ValueError("num_sequences > 0 and seqlen >= 2 required")
    if n < m * L:
        raise RuntimeError(
            f"calibration split has {n} tokens, need at least {m * L} for "
            f"{m} disjoint windows of length {L}"
        )

    rng = random.Random(int(seed))
    starts: List[int] = []
    windows: List[torch.Tensor] = []
    # One window per disjoint corpus segment.  Because segment length >= L,
    # windows from adjacent segments cannot overlap.
    for i in range(m):
        seg_lo = (i * n) // m
        seg_hi = ((i + 1) * n) // m
        latest = seg_hi - L
        if latest < seg_lo:
            raise RuntimeError("internal calibration segment shorter than requested window")
        start = rng.randint(seg_lo, latest) if latest > seg_lo else seg_lo
        starts.append(int(start))
        windows.append(ids[:, start:start + L].contiguous())
    return windows, starts


@torch.inference_mode()
def build_calibration_cache(model, ids: torch.Tensor, device: torch.device, chunk_size: int):
    past = None
    n = int(ids.shape[1])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        out = model(input_ids=ids[:, start:end].to(device), past_key_values=past,
                    use_cache=True, output_attentions=False, output_hidden_states=False,
                    return_dict=True)
        past = out.past_key_values
        del out
    return past


@torch.inference_mode()
def layer_head_scales_from_key_layers(key_layers: List[torch.Tensor]):
    rows = []
    layer_stats = []
    expected_len = None
    expected_heads = None
    for li, keys in enumerate(key_layers):
        if keys.ndim != 4 or int(keys.shape[0]) != 1:
            raise ValueError(f"layer {li}: expected [1,H,L,D], got {tuple(keys.shape)}")
        length = int(keys.shape[-2])
        if length < 2:
            raise ValueError("calibration sequence must contain at least two tokens")
        if expected_len is None:
            expected_len = length
        elif length != expected_len:
            raise RuntimeError("KV cache lengths differ across layers during calibration")
        h = int(keys.shape[1])
        if expected_heads is None:
            expected_heads = h
        elif h != expected_heads:
            raise RuntimeError("KV-head count differs across layers; unsupported calibration layout")
        x = keys[0].float()
        mean = x.mean(dim=1, keepdim=True)
        m2 = (x - mean).square().sum(dim=(1, 2))
        s2 = (2.0 * m2 / float(length - 1)).clamp_min(0.0) + _EPS ** 2
        s = torch.sqrt(s2).detach().double().cpu()
        rows.append(s)
        layer_stats.append({"layer_idx": li, "kv_heads": h,
                            "mean_scale": float(s.mean()), "min_scale": float(s.min()),
                            "max_scale": float(s.max())})
    if not rows:
        raise RuntimeError("model produced no KV layers")
    mat = torch.stack(rows, dim=0)
    summary_global_rms = math.sqrt(float((mat * mat).mean().item()))
    return mat, int(expected_len), layer_stats, summary_global_rms


def aggregate_sequence_scales(sequence_scales: Sequence[torch.Tensor]):
    """Aggregate [M,L,H] per-sequence scales by statistical median."""
    if not sequence_scales:
        raise ValueError("need at least one calibration scale matrix")
    mats = [x.detach().double().cpu() for x in sequence_scales]
    shape = tuple(mats[0].shape)
    if len(shape) != 2 or any(tuple(x.shape) != shape for x in mats):
        raise ValueError("calibration scale matrices must have the same [layers,heads] shape")
    stack = torch.stack(mats, dim=0)
    frozen = torch.quantile(stack, 0.5, dim=0)
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False)
    cv = std / mean.clamp_min(_EPS)
    return frozen, mean, std, cv


# Backward-compatible pure helper retained for old unit tests / notebooks.
@torch.inference_mode()
def global_scale_from_key_layers(key_layers: List[torch.Tensor]):
    mat, length, stats, global_rms = layer_head_scales_from_key_layers(key_layers)
    return global_rms, int(mat.numel()), length, stats


def main() -> None:
    args = build_parser().parse_args()
    if int(args.calibration_num_sequences) <= 0:
        raise ValueError("calibration_num_sequences must be > 0")
    if int(args.calibration_seqlen) < 2 or int(args.prefill_chunk_size) <= 0:
        raise ValueError("calibration_seqlen >=2 and prefill_chunk_size >0 required")
    random.seed(int(args.calibration_seed))
    np.random.seed(int(args.calibration_seed))
    torch.manual_seed(int(args.calibration_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.calibration_seed))

    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast_tokenizer,
                                               trust_remote_code=args.trust_remote_code,
                                               padding_side="left")
    kwargs = dict(config=config, torch_dtype=parse_dtype(args.torch_dtype),
                  trust_remote_code=args.trust_remote_code,
                  attn_implementation=args.attn_implementation)
    if str(args.device_map).lower() != "none":
        kwargs["device_map"] = args.device_map
    model = AutoModel.from_pretrained(args.model, **kwargs); model.eval()
    if str(args.device_map).lower() == "none":
        requested = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
        model.to(torch.device(requested))
    device = first_execution_device(model, args.device)

    token_stream = load_calibration_token_stream(tokenizer, args.dataset, args.calibration_split)
    windows, starts = select_disjoint_windows(
        token_stream,
        num_sequences=int(args.calibration_num_sequences),
        seqlen=int(args.calibration_seqlen),
        seed=int(args.calibration_seed),
    )
    print(
        f"Calibrating frozen S_lh on {args.dataset}/{args.calibration_split}: "
        f"sequences={len(windows)}, tokens/sequence={args.calibration_seqlen}, "
        f"chunk={args.prefill_chunk_size}, seed={args.calibration_seed}",
        flush=True,
    )

    sequence_mats: List[torch.Tensor] = []
    sequence_stats = []
    cache_len = None
    for seq_idx, (ids, start) in enumerate(zip(windows, starts)):
        print(f"  calibration sequence {seq_idx+1}/{len(windows)}: token_start={start}", flush=True)
        past = build_calibration_cache(model, ids, device, int(args.prefill_chunk_size))
        mat, this_len, _, global_rms = layer_head_scales_from_key_layers(cache_keys(past))
        if cache_len is None:
            cache_len = int(this_len)
        elif int(this_len) != int(cache_len):
            raise RuntimeError("calibration cache length changed across sequences")
        sequence_mats.append(mat)
        sequence_stats.append({
            "sequence_idx": int(seq_idx),
            "token_start": int(start),
            "token_end_exclusive": int(start + args.calibration_seqlen),
            "summary_global_rms_scale": float(global_rms),
            "min_scale": float(mat.min()),
            "max_scale": float(mat.max()),
        })
        del past, mat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frozen, seq_mean, seq_std, seq_cv = aggregate_sequence_scales(sequence_mats)
    global_rms = math.sqrt(float((frozen * frozen).mean().item()))
    layer_stats = []
    for li in range(int(frozen.shape[0])):
        row = frozen[li]
        layer_stats.append({
            "layer_idx": int(li),
            "kv_heads": int(frozen.shape[1]),
            "mean_scale": float(row.mean()),
            "min_scale": float(row.min()),
            "max_scale": float(row.max()),
            "mean_sequence_cv": float(seq_cv[li].mean()),
            "max_sequence_cv": float(seq_cv[li].max()),
        })

    payload = {
        "layer_head_scales": frozen.tolist(),
        "sequence_scale_mean": seq_mean.tolist(),
        "sequence_scale_std": seq_std.tolist(),
        "sequence_scale_cv": seq_cv.tolist(),
        "definition_per_sequence": "S_lh,m=sqrt(2*sum_i||K_i-mean(K)||^2/(L-1)+eps^2)",
        "aggregation": "S_lh_calib=median_m(S_lh,m)",
        "protocol": "multiple disjoint calibration sequences; freeze each layer/head median scale for all later prefixes and compression rounds",
        "summary_global_rms_scale": float(global_rms),
        "frozen": True,
        "model": args.model,
        "dataset": args.dataset,
        "calibration_split": args.calibration_split,
        "calibration_num_sequences": int(args.calibration_num_sequences),
        "calibration_seqlen": int(args.calibration_seqlen),
        "calibration_seed": int(args.calibration_seed),
        "calibration_sequence_starts": [int(x) for x in starts],
        "cache_seqlen": int(cache_len),
        "num_layers": int(frozen.shape[0]),
        "num_kv_heads": int(frozen.shape[1]),
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "sequence_stats": sequence_stats,
        "layer_stats": layer_stats,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    out = Path(args.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Frozen S_lh matrix: shape={tuple(frozen.shape)}, min={float(frozen.min()):.6g}, "
        f"max={float(frozen.max()):.6g}, summary RMS={global_rms:.6g}, "
        f"mean CV={float(seq_cv.mean()):.4g}, max CV={float(seq_cv.max()):.4g}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
