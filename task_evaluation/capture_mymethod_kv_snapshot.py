#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture a small real-model KV snapshot for graph-MIS mymethod diagnostics.

The snapshot is intentionally small: only user-selected layers are copied to CPU.
It is consumed by ``analyze_mymethod_graph_mis.py`` so expensive model prefill is
performed once while grouping/certificate analyses can be repeated cheaply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import evaluate_longbench_kvmerge as lb
import evaluate_ppl_kvmerge as ppl


def parse_dtype(name: str):
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        return "auto"
    raise ValueError(name)


def parse_layers(spec: str, total: int) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(total))
    out = []
    aliases = {"first": 0, "mid": total // 2, "last": total - 1}
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        idx = aliases.get(token, int(token) if token.lstrip("-").isdigit() else None)
        if idx is None:
            raise ValueError(f"Unknown layer token: {raw!r}")
        if idx < 0:
            idx += total
        if not 0 <= idx < total:
            raise ValueError(f"Layer {idx} outside [0,{total})")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("No layers selected")
    return out


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name", required=True, help="Hugging Face ID or local model path")
    p.add_argument("--family", choices=["llama", "qwen"], required=True)
    p.add_argument("--task_eval_dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    p.add_argument("--model_seqlen", type=int, default=2048)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--layers", default="first,mid,last")
    p.add_argument("--torch_dtype", default="bf16",
                   choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument(
        "--attention_chunk_size", type=int, default=64,
        help="Exact query-block size used while capturing reduced attention scores. "
             "Use a small positive value on memory-constrained GPUs; 0 means full-query attention.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.model_seqlen <= 1 or args.sample_index < 0:
        raise ValueError("model_seqlen must be >1 and sample_index must be non-negative")
    if args.attention_chunk_size < 0:
        raise ValueError("attention_chunk_size must be >=0; use 0 only for full-query attention")
    task_dir = Path(args.task_eval_dir).resolve()
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))

    ppl.set_seed(args.seed)
    # Patch the model class before loading. The attention selector mode captures
    # current post-softmax attention sums [B,H_kv,K], matching the comparison method.
    ppl.apply_runtime_patch_if_needed(args.family, task_dir, "mymethod")

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    kwargs = dict(
        config=config,
        torch_dtype=parse_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.device_map.lower() != "none":
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False, trust_remote_code=args.trust_remote_code
    )

    modules = lb._self_attention_modules(model)
    layers = parse_layers(args.layers, len(modules))
    for attn in modules:
        attn._mymethod_selector = "attention"
        attn._mymethod_score_window = 0
        # IMPORTANT: score capture is exact but O(L^2).  Chunk the query dimension so
        # 8K snapshots do not materialize a full [H_q,L,L] logits/FP32-softmax tensor.
        # This changes only peak workspace, not the mathematical attention result.
        attn._kvmerge_attention_chunk_size = int(args.attention_chunk_size)

    # The WikiText-2 helper tokenizes the full evaluation corpus before we slice a
    # short snapshot.  Temporarily relax tokenizer.model_max_length so Transformers
    # does not emit a misleading "289k > model max" warning; only the sliced
    # model_seqlen tokens below are ever sent to the model.
    _tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = max(int(_tokenizer_model_max_length or 0), 1_000_000_000)
        data = ppl.get_eval_data(args.dataset, args.model_seqlen, tokenizer, nsamples=1)
    finally:
        if _tokenizer_model_max_length is not None:
            tokenizer.model_max_length = _tokenizer_model_max_length
    start = args.sample_index * args.model_seqlen
    end = start + args.model_seqlen
    if end >= data.shape[1]:
        raise RuntimeError(
            f"sample_index={args.sample_index} requires tokens [{start},{end}), "
            f"but dataset has {data.shape[1]} tokens"
        )
    input_ids = data[:, start:end]
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids)

    lb._set_reduced_attention_capture(model, True, reduce="sum")
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        scores = lb._get_reduced_attention_scores(model)
        legacy = ppl._as_legacy_tuple(out.past_key_values)
        payload_layers = []
        for layer_idx in layers:
            keys = legacy[layer_idx][0].detach().to("cpu")
            importance = scores[layer_idx].detach().float().to("cpu")
            if importance.dim() != 3:
                raise RuntimeError(
                    f"Expected reduced score [B,H_kv,L] at layer {layer_idx}, "
                    f"got {tuple(importance.shape)}"
                )
            payload_layers.append({
                "layer_idx": layer_idx,
                "keys": keys,
                "importance": importance,
            })
    finally:
        lb._set_reduced_attention_capture(model, False, reduce="sum")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_name": args.model_name,
        "family": args.family,
        "dataset": args.dataset,
        "model_seqlen": args.model_seqlen,
        "sample_index": args.sample_index,
        "selected_layers": layers,
        "attention_chunk_size": int(args.attention_chunk_size),
        "score_semantics": "GroupLogitKV current-forward post-softmax attention; GQA mean then query sum",
        "layers": payload_layers,
    }
    torch.save(payload, output)
    meta = {k: v for k, v in payload.items() if k != "layers"}
    meta["layer_shapes"] = [
        {
            "layer_idx": item["layer_idx"],
            "keys": list(item["keys"].shape),
            "importance": list(item["importance"].shape),
        }
        for item in payload_layers
    ]
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved snapshot: {output}")


if __name__ == "__main__":
    main()
