#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPL evaluation for KV-cache merging methods in task_evaluation.

Constraints followed:
  - evaluate_perplexity(...) keeps the same signature and implementation as the supplied AquaKV script.
  - Dataset helpers for WikiText-2 and C4 are copied directly from datautils.py.
  - KV merge methods are exposed through cache_factory by returning a DynamicCache-compatible object.

Example:
  python evaluate_ppl_kvmerge.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --task_eval_dir ./task_evaluation \
    --family qwen \
    --datasets wikitext2 c4 \
    --kv_method GroupLogitKV \
    --compress_ratio 0.4 \
    --sim_thr -1 \
    --model_seqlen 2048 \
    --chunk_size 128 \
    --output_jsonl results/ppl_runs.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm

import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


# --------------------------------------------------------------------------------------
# Strictly keep this function interface and implementation unchanged.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_perplexity(
        model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device,
        amp_dtype: Optional[torch.dtype] = None, step_size: Optional[int] = None,
        cache_factory: Optional[Callable[[], DynamicCache]] = None
        ) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    if step_size is None:
        step_size = seqlen
    inps = [data[:, start : start + seqlen]
            for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
            ]  # ignore last incomplete sequence as in the GPTQ paper
    m = 3
    inps = inps[:m]

    num_sequences_without_padding = len(inps)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity")):
        input_ids = input_ids.to(device)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            if cache_factory is None:
                cache = DynamicCache()
            else:
                cache = cache_factory()
            dtype = amp_dtype or next(model.parameters()).dtype
            lm_logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], model.get_output_embeddings().out_features), device=device, dtype=dtype)
            for i in range(0, input_ids.shape[1], step_size):
                out = model(input_ids[:, i: i + step_size], use_cache=True, past_key_values=cache)
                assert out.past_key_values is cache
                lm_logits[:, i: i + step_size, ...] = out.logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()
        else:
            raise RuntimeError

    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()

# --------------------------------------------------------------------------------------
# Copied from datautils.py: seeds + WikiText-2 and C4 dataset functions.
# --------------------------------------------------------------------------------------
def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_eval_data(name: str, seqlen: int, tokenizer, nsamples: int = 128) -> torch.Tensor:
    lower = name.lower()
    if lower in {"wikitext2", "wikitext-2", "wikitext"}:
        data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=True)
    elif lower == "c4":
        data = get_c4(nsamples, seqlen, tokenizer, eval_mode=True)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Only wikitext2 and c4 are supported.")
    return data.input_ids if hasattr(data, "input_ids") else data


# --------------------------------------------------------------------------------------
# KV merge adapter.
# --------------------------------------------------------------------------------------
def _as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return past_key_values
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _legacy_kv_len(past_key_values) -> int:
    legacy = _as_legacy_tuple(past_key_values)
    if legacy is None or len(legacy) == 0:
        return 0
    return int(legacy[0][0].shape[-2])


def _legacy_kv_mib(past_key_values) -> float:
    legacy = _as_legacy_tuple(past_key_values)
    if legacy is None:
        return 0.0
    total = 0
    for k, v in legacy:
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total / (1024 ** 2)


def _dynamic_cache_from_legacy(legacy):
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(legacy))
    cache = DynamicCache()
    cache.key_cache = [kv[0] for kv in legacy]
    cache.value_cache = [kv[1] for kv in legacy]
    return cache


def _replace_dynamic_cache_inplace(cache: DynamicCache, legacy) -> None:
    """Replace cached K/V tensors without resetting logical decoding progress.

    Important: compressed KV length is a physical storage length. It must not be
    written into DynamicCache._seen_tokens, because _seen_tokens/cache_position are
    used by RoPE/position logic as the logical number of processed tokens.
    """
    legacy = tuple(legacy)

    old_seen_tokens = getattr(cache, "_seen_tokens", None)
    old_logical_seen = getattr(cache, "_ppl_logical_seen_tokens", None)

    # Transformers <= 4.44-style DynamicCache.
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            cache.key_cache = [kv[0] for kv in legacy]
            cache.value_cache = [kv[1] for kv in legacy]
            if old_seen_tokens is not None:
                cache._seen_tokens = old_seen_tokens
            if old_logical_seen is not None:
                cache._ppl_logical_seen_tokens = old_logical_seen
            return
        except Exception:
            pass

    # Generic fallback: build a fresh DynamicCache and copy its state into the existing object.
    fresh = _dynamic_cache_from_legacy(legacy)
    if hasattr(cache, "__dict__") and hasattr(fresh, "__dict__"):
        cache.__dict__.clear()
        cache.__dict__.update(fresh.__dict__)
        if old_seen_tokens is not None:
            cache._seen_tokens = old_seen_tokens
        if old_logical_seen is not None:
            cache._ppl_logical_seen_tokens = old_logical_seen
        return

    raise RuntimeError(
        "Could not replace DynamicCache content in-place for this Transformers version. "
        "Use a Transformers version whose DynamicCache supports key_cache/value_cache or __dict__ state."
    )


def _cuda_sync_for_model(model) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class KVCompressingDynamicCache(DynamicCache):
    def __init__(
        self,
        *,
        model: nn.Module,
        method: str,
        family: str,
        compress_ratio: float,
        sim_thr: float,
        min_kv_len: int,
        compress_every: int,
        stats: dict,
    ):
        super().__init__()
        self._ppl_model = model
        self._kv_method = method
        self._family = family
        self._compress_ratio = float(compress_ratio)
        self._sim_thr = float(sim_thr)
        self._min_kv_len = int(min_kv_len)
        self._compress_every = int(compress_every)
        self._stats = stats
        self._forward_calls = 0
        self._ppl_logical_seen_tokens = 0

    def prepare_position_kwargs(self, args, kwargs) -> None:
        """Inject logical positions and a physical-layout causal mask.

        After KV compression, the cached prefix is a shorter physical prefix that
        represents a longer logical prefix. RoPE must use logical positions, while
        the causal mask must use the current physical layout:

            [compressed_prefix_slots][current_chunk_tokens]

        Therefore every query in the current chunk may attend to all compressed
        prefix slots, but within the current chunk it may only attend to itself
        and earlier chunk tokens. This keeps AquaKV-style chunked PPL evaluation
        while avoiding future-token leakage.
        """
        if not self.enabled:
            return

        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        if input_ids is not None:
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1])
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = int(inputs_embeds.shape[0])
            seq_len = int(inputs_embeds.shape[1])
            device = inputs_embeds.device
        else:
            return

        start = int(getattr(self, "_ppl_logical_seen_tokens", 0))
        cache_position = torch.arange(start, start + seq_len, device=device, dtype=torch.long)

        # RoPE / rotary embedding positions must follow logical decoding length.
        if "position_ids" not in kwargs or kwargs["position_ids"] is None:
            kwargs["position_ids"] = cache_position.unsqueeze(0).expand(batch_size, -1)
        if "cache_position" not in kwargs or kwargs["cache_position"] is None:
            kwargs["cache_position"] = cache_position

        # Build a 4D additive mask over the physical K layout. This is necessary
        # for chunk_size > 1: HF's default mask derives column positions from the
        # physical key length and can otherwise let a query see future tokens in
        # the same chunk when cache_position is logical.
        if "attention_mask" not in kwargs or kwargs["attention_mask"] is None:
            past_physical_len = _legacy_kv_len(self)
            key_len = past_physical_len + seq_len
            try:
                dtype = next(self._ppl_model.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
            if dtype == torch.float32:
                mask_dtype = torch.float32
            elif dtype == torch.float16:
                mask_dtype = torch.float16
            elif dtype == torch.bfloat16:
                mask_dtype = torch.bfloat16
            else:
                mask_dtype = torch.float32
            min_val = torch.finfo(mask_dtype).min
            mask = torch.zeros((batch_size, 1, seq_len, key_len), device=device, dtype=mask_dtype)
            if seq_len > 1:
                chunk_future = torch.triu(
                    torch.full((seq_len, seq_len), min_val, device=device, dtype=mask_dtype),
                    diagonal=1,
                )
                mask[:, :, :, past_physical_len:key_len] = chunk_future.view(1, 1, seq_len, seq_len)
            kwargs["attention_mask"] = mask

    def mark_forward_consumed(self, seq_len: int) -> None:
        if not self.enabled:
            return
        self._ppl_logical_seen_tokens = int(getattr(self, "_ppl_logical_seen_tokens", 0)) + int(seq_len)
        # Keep this logical. Never set it to compressed physical KV length.
        if hasattr(self, "_seen_tokens"):
            self._seen_tokens = self._ppl_logical_seen_tokens

    @property
    def enabled(self) -> bool:
        return self._kv_method.lower() not in {"none", "baseline", "no", "no_compress"} and self._compress_ratio > 0

    def maybe_compress(self, attentions, hidden_states) -> None:
        self._forward_calls += 1
        if not self.enabled:
            return
        if self._compress_every > 1 and (self._forward_calls % self._compress_every) != 0:
            return

        before_len = _legacy_kv_len(self)
        if before_len < self._min_kv_len:
            return

        legacy_before = _as_legacy_tuple(self)
        if legacy_before is None:
            return

        _cuda_sync_for_model(self._ppl_model)
        t0 = time.time()
        merged = apply_kv_method(
            self._kv_method,
            self._family,
            self._ppl_model,
            legacy_before,
            attentions,
            hidden_states,
            self._compress_ratio,
            self._sim_thr,
        )
        _cuda_sync_for_model(self._ppl_model)
        dt = time.time() - t0

        _replace_dynamic_cache_inplace(self, merged)
        after_len = _legacy_kv_len(self)
        after_mib = _legacy_kv_mib(self)

        self._stats["num_compressions"] = self._stats.get("num_compressions", 0) + 1
        self._stats["total_compress_time_s"] = self._stats.get("total_compress_time_s", 0.0) + dt
        self._stats.setdefault("compressed_kv_lens", []).append(after_len)
        self._stats.setdefault("compressed_kv_mib", []).append(after_mib)


def _import_from_family(family: str, task_eval_dir: Path, module_name: str):
    family_dir = task_eval_dir / family
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    return importlib.import_module(module_name)


def apply_runtime_patch_if_needed(family: str, task_eval_dir: Path, method: str) -> None:
    if method != "GroupLogitKV":
        return
    if family == "qwen":
        mod = _import_from_family(family, task_eval_dir, "grouplogitkv_runtime_patch_qwen")
        if hasattr(mod, "apply_grouplogitkv_qwen_runtime_patch"):
            mod.apply_grouplogitkv_qwen_runtime_patch()
    elif family == "llama":
        mod = _import_from_family(family, task_eval_dir, "grouplogitkv_runtime_patch_llama")
        if hasattr(mod, "apply_grouplogitkv_runtime_patch"):
            mod.apply_grouplogitkv_runtime_patch()


def reset_grouplogit_state_if_needed(model: nn.Module, family: str, task_eval_dir: Path) -> None:
    # Keep the same model clean between dataset/method runs.
    try:
        if family == "qwen":
            mod = _import_from_family(family, task_eval_dir, "merge_grouplogitkv_qwen")
            if hasattr(mod, "disable_grouplogitkv_sdpa"):
                mod.disable_grouplogitkv_sdpa(model)
            if hasattr(mod, "disable_grouplogitkv_eager"):
                mod.disable_grouplogitkv_eager(model)
        elif family == "llama":
            mod = _import_from_family(family, task_eval_dir, "merge_grouplogitkv_llama")
            if hasattr(mod, "disable_grouplogitkv_eager"):
                mod.disable_grouplogitkv_eager(model)
            if hasattr(mod, "disable_grouplogitkv_sdpa"):
                mod.disable_grouplogitkv_sdpa(model)
    except Exception:
        # Non-GroupLogitKV methods do not need this; do not fail a PPL run for cleanup.
        pass


def apply_kv_method(method, family, model, past_key_values, attention, hidden_states, compress_ratio, sim_thr):
    if family not in {"qwen", "llama"}:
        raise ValueError("--family must be qwen or llama")

    if method == "SimCalKV":
        if family == "qwen":
            return importlib.import_module("merge_simcalkv_qwen").Repair(
                model, past_key_values, attention, hidden_states, compress_ratio
            )
        return importlib.import_module("merge_simcalkv_llama").Repair(
            model, past_key_values, attention, hidden_states, compress_ratio, sim_thr
        )

    if method == "GroupLogitKV":
        if family == "qwen":
            return importlib.import_module("merge_grouplogitkv_qwen").Repair(
                model, past_key_values, attention, hidden_states, compress_ratio, sim_thr
            )
        return importlib.import_module("merge_grouplogitkv_llama").Repair(
            model, past_key_values, attention, hidden_states, compress_ratio, sim_thr
        )

    if method == "KeepKV":
        if family == "qwen":
            return importlib.import_module("merge_keepkv_qwen").Repair_keepkv(
                model, past_key_values, attention, hidden_states, compress_ratio
            )
        return importlib.import_module("merge_keepkv").KeepKV(
            model, past_key_values, attention, hidden_states, compress_ratio, sim_thr
        )

    if method == "PyramidInfer":
        if family == "qwen":
            return importlib.import_module("pyramid_qwen").Decode(
                model, past_key_values, attention, hidden_states, compress_ratio
            )
        return importlib.import_module("eviction_pyramid").Decode(
            model, past_key_values, attention, hidden_states, compress_ratio, sim_thr
        )

    if method == "CaM":
        if family != "llama":
            raise ValueError("CaM is only present in task_evaluation/llama")
        mod = importlib.import_module("merge_cam")
        if hasattr(mod, "Repair"):
            return mod.Repair(model, past_key_values, attention, hidden_states, compress_ratio, sim_thr)
        if hasattr(mod, "CaMPress"):
            compressor = mod.CaMPress(compression_ratio=compress_ratio)
            new_cache = []
            layers = getattr(model.model, "layers")
            for layer_idx, (k, v) in enumerate(_as_legacy_tuple(past_key_values)):
                module = layers[layer_idx].self_attn
                nk, nv = compressor.compress(module, hidden_states[layer_idx], k, v, attention[layer_idx], {})
                new_cache.append((nk, nv))
            return new_cache
        raise ValueError("merge_cam.py has neither Repair nor CaMPress")

    raise ValueError(f"Unknown KV method: {method}")


@contextlib.contextmanager
def force_attn_hidden_and_compress(model: nn.Module):
    """Force attentions/hidden states and compress cache after each forward step."""
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        cache = kwargs.get("past_key_values", None)
        should_manage = isinstance(cache, KVCompressingDynamicCache)
        seq_len = None
        if should_manage:
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            cache.prepare_position_kwargs(args, kwargs)
            input_ids = kwargs.get("input_ids", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
                input_ids = args[0]
            if input_ids is not None:
                seq_len = int(input_ids.shape[1])
            elif inputs_embeds is not None:
                seq_len = int(inputs_embeds.shape[1])
        out = original_forward(*args, **kwargs)
        if should_manage:
            if seq_len is not None:
                cache.mark_forward_consumed(seq_len)
            cache.maybe_compress(getattr(out, "attentions", None), getattr(out, "hidden_states", None))
            # Most HF models already return the same object. Assign for ModelOutput variants that permit it.
            try:
                out.past_key_values = cache
            except Exception:
                pass
        return out

    model.forward = wrapped_forward
    try:
        yield
    finally:
        model.forward = original_forward


def make_cache_factory(model, method, family, task_eval_dir, ratio, sim_thr, min_kv_len, compress_every, stats):
    def _factory():
        # Each GPTQ/AquaKV-style fixed-length sequence creates a fresh cache.
        # GroupLogitKV stores its log-bias metadata on the model/attention modules,
        # so clear that model-level runtime state here to avoid cross-sequence leakage.
        if method == "GroupLogitKV":
            reset_grouplogit_state_if_needed(model, family, task_eval_dir)
        return KVCompressingDynamicCache(
            model=model,
            method=method,
            family=family,
            compress_ratio=ratio,
            sim_thr=sim_thr,
            min_kv_len=min_kv_len,
            compress_every=compress_every,
            stats=stats,
        )
    return _factory


def parse_dtype(name: str):
    if name in {None, "auto"}:
        return "auto"
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def parse_amp_dtype(name: str):
    if name in {None, "none", "None", "auto"}:
        return None
    parsed = parse_dtype(name)
    if parsed == "auto":
        return None
    return parsed


def stats_summary(stats: dict) -> dict:
    n = int(stats.get("num_compressions", 0))
    lens = stats.get("compressed_kv_lens", [])
    mib = stats.get("compressed_kv_mib", [])
    return {
        "num_compressions": n,
        "total_compress_time_s": float(stats.get("total_compress_time_s", 0.0)),
        "avg_compress_time_s": float(stats.get("total_compress_time_s", 0.0) / n) if n else 0.0,
        "avg_compressed_kv_len": float(sum(lens) / len(lens)) if lens else 0.0,
        "avg_compressed_kv_mib": float(sum(mib) / len(mib)) if mib else 0.0,
    }


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task_eval_dir", default="task_evaluation")
    parser.add_argument("--family", choices=["qwen", "llama"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"], choices=["wikitext2", "c4"])
    parser.add_argument("--kv_method", default="none", choices=["none", "SimCalKV", "GroupLogitKV", "KeepKV", "PyramidInfer", "CaM"])
    parser.add_argument("--compress_ratio", type=float, default=0.0)
    parser.add_argument("--sim_thr", type=float, default=0.0)
    parser.add_argument("--model_seqlen", type=int, default=2048)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--min_kv_len", type=int, default=2)
    parser.add_argument("--compress_every", type=int, default=1, help="Compress every N forward chunks; one forward chunk has --chunk_size tokens.")
    parser.add_argument("--nsamples", type=int, default=128, help="Kept for datautils compatibility; eval C4 uses 256 fixed samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16", "fp16", "fp32", "bf16"])
    parser.add_argument("--amp_dtype", default="none", choices=["none", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--attn_implementation", default="eager", help="Use eager to ensure output_attentions works.")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_json", default=None, help="Optional per-run JSON path. Useful when one command runs one dataset.")
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    task_eval_dir = Path(args.task_eval_dir).resolve()
    if str(task_eval_dir / args.family) not in sys.path:
        sys.path.insert(0, str(task_eval_dir / args.family))

    apply_runtime_patch_if_needed(args.family, task_eval_dir, args.kv_method)

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model_kwargs = dict(
        config=config,
        torch_dtype=parse_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model: {args.model_name}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"KV method: {args.kv_method}, ratio={args.compress_ratio}, sim_thr={args.sim_thr}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not hasattr(model, "hf_device_map") and device.type != "cpu":
        model.to(device)
    first_param_device = next(model.parameters()).device
    eval_device = first_param_device if first_param_device.type != "meta" else device

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset_name in args.datasets:
        print(f"Loading eval dataset: {dataset_name}")
        data = get_eval_data(dataset_name, args.model_seqlen, tokenizer, nsamples=args.nsamples)
        stats = {}
        cache_factory = None
        if args.kv_method.lower() not in {"none", "baseline", "no", "no_compress"}:
            cache_factory = make_cache_factory(
                model=model,
                method=args.kv_method,
                family=args.family,
                task_eval_dir=task_eval_dir,
                ratio=args.compress_ratio,
                sim_thr=args.sim_thr,
                min_kv_len=args.min_kv_len,
                compress_every=args.compress_every,
                stats=stats,
            )
        else:
            stats = {"num_compressions": 0, "total_compress_time_s": 0.0}

        reset_grouplogit_state_if_needed(model, args.family, task_eval_dir)
        # Follow the AquaKV/GPTQ-style PPL loop: process each fixed-length sequence
        # in chunks of size args.chunk_size. For compressed caches,
        # KVCompressingDynamicCache injects logical RoPE positions plus a 4D
        # physical-layout causal mask, so chunk_size > 1 remains valid.
        actual_step_size = args.chunk_size

        t0 = time.time()
        with force_attn_hidden_and_compress(model):
            ppl = evaluate_perplexity(
                model=model,
                data=data,
                seqlen=args.model_seqlen,
                device=eval_device,
                amp_dtype=parse_amp_dtype(args.amp_dtype),
                step_size=actual_step_size,
                cache_factory=cache_factory,
            )
        elapsed = time.time() - t0
        reset_grouplogit_state_if_needed(model, args.family, task_eval_dir)

        result = {
            "model_name": args.model_name,
            "family": args.family,
            "dataset": dataset_name,
            "kv_method": args.kv_method,
            "compress_ratio": args.compress_ratio,
            "sim_thr": args.sim_thr,
            "model_seqlen": args.model_seqlen,
            "chunk_size": args.chunk_size,
            "actual_step_size": actual_step_size,
            "min_kv_len": args.min_kv_len,
            "compress_every": args.compress_every,
            "torch_dtype": args.torch_dtype,
            "amp_dtype": args.amp_dtype,
            "attn_implementation": args.attn_implementation,
            "seed": args.seed,
            "ppl": ppl,
            "elapsed_s": elapsed,
            "num_eval_tokens": int(data.shape[1]),
        }
        result.update(stats_summary(stats))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        all_results.append(result)

        if args.output_jsonl:
            with open(args.output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if args.output_json:
        payload = all_results[0] if len(all_results) == 1 else all_results
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
