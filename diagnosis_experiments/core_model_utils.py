#!/usr/bin/env python3
"""Pretrained-model prefill and KeepKV routing-score capture for query-independent cosine-routing MIS fusion.

No future or sampled query is used to make routing or core-fusion decisions.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Dict
from datasets import load_dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EVAL_ROOT = PROJECT_ROOT / 'task_evaluation'
if str(TASK_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_EVAL_ROOT))
from models.common.mymethod_common import reference_scale_from_original_keys

def parse_dtype(name: str):
    x = str(name).lower()
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if x not in table:
        raise ValueError(f"Unsupported torch dtype {name!r}")
    return table[x]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_stream(tokenizer, dataset_name: str) -> torch.Tensor:
    if dataset_name != "wikitext2":
        raise ValueError("This diagnosis currently supports only wikitext2.")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    return enc.input_ids


def cache_key_value_layers(past_key_values) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if past_key_values is None:
        raise RuntimeError("Model returned no past_key_values although use_cache=True.")
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    if isinstance(past_key_values, (tuple, list)):
        return [layer[0] for layer in past_key_values], [layer[1] for layer in past_key_values]
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [layer[0] for layer in legacy], [layer[1] for layer in legacy]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def first_execution_device(model, requested: str) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(requested)


def transformer_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Cannot locate transformer layers on the loaded model.")


def selector_patch_functions(model_type: str):
    mt = str(model_type).lower()
    if mt == "llama":
        from models.llama.mymethod_runtime_patch_llama import (
            apply_mymethod_runtime_patch,
            remove_mymethod_runtime_patch,
        )
        return apply_mymethod_runtime_patch, remove_mymethod_runtime_patch
    if mt in {"qwen2", "qwen"}:
        from models.qwen.mymethod_runtime_patch_qwen import (
            apply_mymethod_qwen_runtime_patch,
            remove_mymethod_qwen_runtime_patch,
        )
        return apply_mymethod_qwen_runtime_patch, remove_mymethod_qwen_runtime_patch
    raise ValueError(
        f"Exact KeepKV-score capture currently supports Llama/Qwen2, got model_type={model_type!r}."
    )


def forward_cache_chunk(model, chunk: torch.Tensor, past, device: torch.device):
    with torch.inference_mode():
        return model(
            input_ids=chunk.to(device),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )


def build_full_cache_with_keepkv_scores(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    score_window: int,
    model_type: str,
) -> Tuple[object, List[torch.Tensor], List[float]]:
    if chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be > 0")
    length = int(input_ids.shape[1])
    window = min(max(int(score_window), 1), length)
    warmup_end = length - window
    past = None
    for start in range(0, warmup_end, chunk_size):
        end = min(start + chunk_size, warmup_end)
        out = forward_cache_chunk(model, input_ids[:, start:end], past, device)
        past = out.past_key_values
        del out

    layers = transformer_layers(model)
    apply_patch, remove_patch = selector_patch_functions(model_type)
    apply_patch()
    try:
        for layer in layers:
            attn = layer.self_attn
            attn._mymethod_selector = "keepkv"
            attn._mymethod_score_window = window
            attn._kvmerge_capture_attention_scores = True
            attn._kvmerge_attention_chunk_size = min(chunk_size, window)
            attn._mymethod_eager_enabled = False
            attn._mymethod_counts_kv = None
            attn._keepkv_eager_enabled = False
            attn._keepkv_log_bias_q = None
            for name in ("_kvmerge_captured_attention_score", "_kvmerge_captured_score_log_scale"):
                if hasattr(attn, name):
                    delattr(attn, name)

        out = forward_cache_chunk(model, input_ids[:, warmup_end:length], past, device)
        past = out.past_key_values
        del out
        score_rows: List[torch.Tensor] = []
        score_scales: List[float] = []
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            score = getattr(attn, "_kvmerge_captured_attention_score", None)
            if score is None:
                raise RuntimeError(f"No KeepKV selector score captured for layer {layer_idx}.")
            if score.ndim != 4 or int(score.shape[-2]) != window or int(score.shape[-1]) != length:
                raise RuntimeError(
                    f"Bad KeepKV selector score shape at layer {layer_idx}: {tuple(score.shape)}; "
                    f"expected [B,Hkv,{window},{length}]."
                )
            score_rows.append(score.detach())
            score_scales.append(float(getattr(attn, "_kvmerge_captured_score_log_scale", 0.0)))
    finally:
        for layer in layers:
            layer.self_attn._kvmerge_capture_attention_scores = False
        remove_patch()
    return past, score_rows, score_scales

def frozen_reference_scale(raw_keys: torch.Tensor) -> torch.Tensor:
    return reference_scale_from_original_keys(raw_keys.unsqueeze(0).unsqueeze(0))[0, 0, 0].detach()
