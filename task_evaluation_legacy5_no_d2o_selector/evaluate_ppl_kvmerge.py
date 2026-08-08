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
    --sim_thr_grouplogitkv 0.8 \
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

_EVAL_MAX_SEQUENCES = 0


# --------------------------------------------------------------------------------------
# Strictly keep this function interface and implementation unchanged.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_perplexity(
        model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device,
        amp_dtype: Optional[torch.dtype] = None, step_size: Optional[int] = None,
        cache_factory: Optional[Callable[[], DynamicCache]] = None
        ) -> float:
    """Evaluate causal-LM perplexity without materializing full-sequence logits.

    Each forward chunk is scored immediately against the globally shifted labels.
    Cross entropy is evaluated in float32, but only for one chunk at a time. This
    avoids the multi-gigabyte ``[seqlen, vocab]`` float32 allocation that occurs
    when a full sequence of BF16 logits is converted at once.
    """
    if step_size is None:
        step_size = seqlen
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    inps = [
        data[:, start:start + seqlen]
        for start in range(0, data.shape[1], seqlen)
        if start + seqlen < data.shape[1]
    ]
    max_eval_sequences = int(_EVAL_MAX_SEQUENCES)
    if max_eval_sequences > 0:
        inps = inps[:max_eval_sequences]
    print(
        f"PPL evaluation windows: {len(inps)} "
        f"(seqlen={seqlen}, dataset_tokens={int(data.shape[1])}, limit={max_eval_sequences})"
    )

    num_sequences_without_padding = len(inps)
    total_nll = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    for sequence_idx, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity")):
        input_ids = input_ids.to(device)
        cache = DynamicCache() if cache_factory is None else cache_factory()
        sequence_length = input_ids.shape[1]

        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None,
            dtype=amp_dtype or torch.float32,
        ):
            for i in range(0, sequence_length, step_size):
                chunk_end = min(i + step_size, sequence_length)
                out = model(
                    input_ids[:, i:chunk_end],
                    use_cache=True,
                    past_key_values=cache,
                )
                assert out.past_key_values is cache

                # Logit at global position p predicts token p+1. The final logit
                # of the complete sequence has no in-window target and is skipped.
                scored_end = min(chunk_end, sequence_length - 1)
                scored_tokens = scored_end - i
                if scored_tokens > 0:
                    chunk_logits = out.logits[:, :scored_tokens, :]
                    chunk_labels = input_ids[:, i + 1:scored_end + 1]
                    logits_fp32 = chunk_logits.float()
                    if not torch.isfinite(logits_fp32).all():
                        bad = int((~torch.isfinite(logits_fp32)).sum().item())
                        raise RuntimeError(
                            "Non-finite model logits during PPL evaluation: "
                            f"sequence_idx={sequence_idx}, chunk_start={i}, bad_values={bad}."
                        )
                    loss_sum = loss_fct(
                        logits_fp32.reshape(-1, logits_fp32.size(-1)),
                        chunk_labels.reshape(-1),
                    )
                    if not torch.isfinite(loss_sum):
                        raise RuntimeError(
                            "Non-finite cross-entropy loss: "
                            f"sequence_idx={sequence_idx}, chunk_start={i}."
                        )
                    total_nll += float(loss_sum.item())
                    total_tokens += int(chunk_labels.numel())

                # Drop the very large vocabulary logits before the next forward.
                del out

        del cache, input_ids

    if total_tokens == 0:
        raise RuntimeError("No evaluation tokens were produced. Check seqlen and dataset length.")

    ppl = float(np.exp(total_nll / total_tokens))
    os.environ["KV_EVAL_ACTUAL_TOKENS"] = str(total_tokens)
    os.environ["KV_EVAL_ACTUAL_SEQUENCES"] = str(num_sequences_without_padding)
    return ppl

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
        method_sim_thrs: Optional[dict],
        keepkv_config: Optional[dict],
        mymethod_config: Optional[dict],
        centroidkv_config: Optional[dict],
        min_kv_len: int,
        compress_every: int,
        stats: dict,
        task_eval_dir: Path,
        target_kv_len: Optional[int],
    ):
        super().__init__()
        self._ppl_model = model
        self._kv_method = method
        self._family = family
        self._compress_ratio = float(compress_ratio)
        self._sim_thr = float(sim_thr)
        self._method_sim_thrs = method_sim_thrs or {}
        raw_keepkv_config = dict(keepkv_config or {})
        self._keepkv_runtime_only = bool(raw_keepkv_config.pop("diagnostic_runtime_only", False))
        self._keepkv_config = raw_keepkv_config
        self._mymethod_config = dict(mymethod_config or {})
        self._centroidkv_config = dict(centroidkv_config or {})
        self._min_kv_len = int(min_kv_len)
        self._compress_every = int(compress_every)
        self._stats = stats
        self._task_eval_dir = Path(task_eval_dir)
        self._target_kv_len = None if target_kv_len is None else int(target_kv_len)
        self._forward_calls = 0
        self._ppl_logical_seen_tokens = 0
        # Runtime attention patches use this marker to build the causal mask
        # from each layer's own physical KV length. This avoids future-token
        # leakage for compressed caches.
        self._ppl_use_physical_causal_mask = True

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
        if self._kv_method == "KeepKV" and self._keepkv_runtime_only:
            return
        if self._compress_every > 1 and (self._forward_calls % self._compress_every) != 0:
            return

        before_len = _legacy_kv_len(self)
        if before_len < self._min_kv_len:
            return

        legacy_before = _as_legacy_tuple(self)
        if legacy_before is None:
            return

        # Dynamic proportional budget: after processing t logical tokens, keep
        # approximately (1 - compress_ratio) * t physical KV slots per layer.
        logical_seen = int(getattr(self, "_ppl_logical_seen_tokens", 0))
        retained_ratio = 1.0 - float(self._compress_ratio)
        dynamic_target_kv_len = max(1, int(round(logical_seen * retained_ratio)))

        if self._kv_method == "mymethod":
            sink_size = int(self._mymethod_config.get("sink_size", 4))
            if dynamic_target_kv_len < sink_size:
                self._stats.setdefault("mymethod_deferred_sink_budget", 0)
                self._stats["mymethod_deferred_sink_budget"] += 1
                return

        if self._kv_method == "CentroidKV":
            sink_budget = int(self._centroidkv_config.get("sink_budget", 16))
            local_budget = int(self._centroidkv_config.get("local_budget", 64))
            # The official method never merges the protected regions. Clamp the
            # dynamic target to that structural floor instead of spending merge
            # iterations on an impossible smaller target.
            protected_floor = min(before_len, sink_budget + local_budget)
            if dynamic_target_kv_len < protected_floor:
                self._stats.setdefault("centroidkv_budget_floor_applied", 0)
                self._stats["centroidkv_budget_floor_applied"] += 1
                dynamic_target_kv_len = protected_floor

        # CaM's unmodified official merge rule requires recent_size >= merge_size.
        # For very short prefixes the proportional budget can be smaller than this
        # structural minimum. In that case, defer CaM compression rather than
        # modifying the algorithm or passing an invalid configuration.
        if self._kv_method == "CaM":
            cam_cfg = getattr(self._ppl_model, "_cam_eval_config", {}) or {}
            merge_size = int(cam_cfg.get("merge_size", 16))
            start_fraction = float(cam_cfg.get("start_fraction", 0.1))
            start_size = int(round(dynamic_target_kv_len * start_fraction))
            start_size = min(max(start_size, 0), max(dynamic_target_kv_len - 1, 0))
            recent_size = dynamic_target_kv_len - start_size
            if recent_size < merge_size:
                self._stats.setdefault("cam_deferred_short_prefix", 0)
                self._stats["cam_deferred_short_prefix"] += 1
                return
            # CaM derives its cache size from logical_length and compress_ratio.
            # Updating this evaluator-owned config is sufficient to realize the
            # same dynamic target without changing any CaM source file.
            cam_cfg = dict(cam_cfg)
            cam_cfg["logical_length"] = logical_seen
            self._ppl_model._cam_eval_config = cam_cfg

        dynamic_keepkv_config = self._keepkv_config
        if self._kv_method == "KeepKV":
            dynamic_keepkv_config = dict(self._keepkv_config)
            dynamic_keepkv_config["logical_length"] = logical_seen

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
            self._method_sim_thrs,
            dynamic_keepkv_config,
            self._task_eval_dir,
            target_kv_len=dynamic_target_kv_len,
            mymethod_config=self._mymethod_config,
            centroidkv_config=self._centroidkv_config,
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
    if str(task_eval_dir) not in sys.path:
        sys.path.insert(0, str(task_eval_dir))
    return importlib.import_module(f"models.{family}.{module_name}")


def apply_runtime_patch_if_needed(family: str, task_eval_dir: Path, method: str) -> None:
    """Install the method-specific eager-attention runtime patch."""
    if method.lower() in {"none", "baseline", "no", "no_compress"}:
        return
    if method == "mymethod":
        if family == "qwen":
            mod = _import_from_family(family, task_eval_dir, "mymethod_runtime_patch_qwen")
            mod.apply_mymethod_qwen_runtime_patch()
        elif family == "llama":
            mod = _import_from_family(family, task_eval_dir, "mymethod_runtime_patch_llama")
            mod.apply_mymethod_runtime_patch()
        return
    if method == "KeepKV":
        if family == "qwen":
            mod = _import_from_family(family, task_eval_dir, "keepkv_runtime_patch_qwen")
            mod.apply_keepkv_qwen_runtime_patch()
        elif family == "llama":
            mod = _import_from_family(family, task_eval_dir, "keepkv_runtime_patch_llama")
            mod.apply_keepkv_runtime_patch()
        return

    # Existing GroupLogitKV patches remain the generic compressed-cache eager path
    # for the other methods in this evaluation project.
    if family == "qwen":
        mod = _import_from_family(family, task_eval_dir, "grouplogitkv_runtime_patch_qwen")
        if hasattr(mod, "apply_grouplogitkv_qwen_runtime_patch"):
            mod.apply_grouplogitkv_qwen_runtime_patch()
    elif family == "llama":
        mod = _import_from_family(family, task_eval_dir, "grouplogitkv_runtime_patch_llama")
        if hasattr(mod, "apply_grouplogitkv_runtime_patch"):
            mod.apply_grouplogitkv_runtime_patch()


def reset_method_state(model: nn.Module, family: str, task_eval_dir: Path, method: str) -> None:
    """Clear sequence-local runtime metadata between independent PPL samples."""
    try:
        if method == "mymethod":
            mod = _import_from_family(family, task_eval_dir, f"merge_mymethod_{family}")
            if hasattr(mod, "reset_mymethod_state"):
                mod.reset_mymethod_state(model)
            return
        if method == "KeepKV":
            mod = _import_from_family(family, task_eval_dir, f"merge_keepkv_{family}")
            if hasattr(mod, "reset_keepkv_state"):
                mod.reset_keepkv_state(model)
            return
        if method == "CentroidKV":
            mod = _import_from_family(family, task_eval_dir, f"merge_centroidkv_{family}")
            if hasattr(mod, "reset_centroidkv_state"):
                mod.reset_centroidkv_state(model)
            return

        if method == "CaM":
            mod = _import_from_family(family, task_eval_dir, f"merge_cam_{family}")
            if hasattr(mod, "reset_cam_state"):
                mod.reset_cam_state(model)
            return

        if family == "qwen":
            mod = _import_from_family(family, task_eval_dir, "merge_grouplogitkv_qwen")
        elif family == "llama":
            mod = _import_from_family(family, task_eval_dir, "merge_grouplogitkv_llama")
        else:
            mod = None
        if mod is not None:
            if hasattr(mod, "disable_grouplogitkv_sdpa"):
                mod.disable_grouplogitkv_sdpa(model)
            if hasattr(mod, "disable_grouplogitkv_eager"):
                mod.disable_grouplogitkv_eager(model)
    except Exception:
        pass

def _effective_sim_thr(method: str, method_sim_thrs: Optional[dict] = None) -> Optional[float]:
    """Return the per-method similarity threshold.

    Only GroupLogitKV, SimCalKV and KeepKV use similarity thresholds.
    CaM does not accept/use sim_thr in the evaluator.
    SimCalKV defaults to 0.0. GroupLogitKV defaults to the paper setting 0.8. KeepKV uses its own configuration.
    """
    if method not in {"GroupLogitKV", "SimCalKV", "KeepKV"}:
        return None
    if method_sim_thrs and method_sim_thrs.get(method) is not None:
        return float(method_sim_thrs[method])
    # KeepKV has separate official prefill/decode defaults in KeepKVConfig.
    # A missing unified override must not disable its similarity gate.
    if method == "KeepKV":
        return None
    if method == "SimCalKV":
        return 0.0
    return 0.8


def apply_kv_method(
    method, family, model, past_key_values, attention, hidden_states,
    compress_ratio, method_sim_thrs=None, keepkv_config=None, task_eval_dir=None,
    target_kv_len=None, mymethod_config=None, centroidkv_config=None,
):
    if family not in {"qwen", "llama"}:
        raise ValueError("--family must be qwen or llama")

    eff_sim_thr = _effective_sim_thr(method, method_sim_thrs)
    import_root = Path(task_eval_dir) if task_eval_dir is not None else Path(__file__).resolve().parent

    if method == "SimCalKV":
        return _import_from_family(family, import_root, f"merge_simcalkv_{family}").Repair(
            model, past_key_values, attention, hidden_states, compress_ratio, eff_sim_thr,
            target_length=target_kv_len,
        )

    if method == "GroupLogitKV":
        return _import_from_family(family, import_root, f"merge_grouplogitkv_{family}").Repair(
            model, past_key_values, attention, hidden_states, compress_ratio, eff_sim_thr,
            target_length=target_kv_len,
        )

    if method == "mymethod":
        cfg = dict(mymethod_config or {})
        return _import_from_family(family, import_root, f"merge_mymethod_{family}").Repair(
            model,
            past_key_values,
            attention,
            hidden_states,
            compress_ratio,
            float(cfg.get("lambda_mymethod", 0.5)),
            target_length=target_kv_len,
            sink_size=int(cfg.get("sink_size", 4)),
            recent_window=int(cfg.get("recent_window", 128)),
            recency_beta=float(cfg.get("recency_beta", 1.0)),
            distance_chunk_size=int(cfg.get("distance_chunk_size", 256)),
            sink_can_absorb=bool(cfg.get("sink_can_absorb", True)),
            selector=str(cfg.get("selector", "keepkv")),
            group_mode=str(cfg.get("group_mode", "full")),
            graph_backend=str(cfg.get("graph_backend", "auto")),
            route_backend=str(cfg.get("route_backend", "auto")),
        )

    if method == "CentroidKV":
        cfg = dict(centroidkv_config or {})
        return _import_from_family(family, import_root, f"merge_centroidkv_{family}").Repair(
            model, past_key_values, attention, hidden_states, compress_ratio,
            target_length=target_kv_len,
            sink_budget=int(cfg.get("sink_budget", 16)),
            local_budget=int(cfg.get("local_budget", 64)),
            chunk_size=int(cfg.get("chunk_size", 256)),
            merge_ratio=float(cfg.get("merge_ratio", 0.8)),
            decrease_rate=float(cfg.get("decrease_rate", 0.2)),
        )

    if method == "KeepKV":
        return _import_from_family(family, import_root, f"merge_keepkv_{family}").Repair_keepkv(
            model, past_key_values, attention, hidden_states, compress_ratio, eff_sim_thr,
            config=keepkv_config,
        )


    if method == "CaM":
        # CaM uses its own attention-based Bernoulli merge mask; no sim_thr.
        mod = _import_from_family(family, import_root, f"merge_cam_{family}")
        return mod.Repair(model, past_key_values, attention, hidden_states, compress_ratio)

    raise ValueError(f"Unknown kv_method: {method}")

def make_cache_factory(
    model, method, family, task_eval_dir, ratio, method_sim_thrs,
    keepkv_config, mymethod_config, centroidkv_config, min_kv_len, compress_every, stats, model_seqlen
):
    def _factory():
        # Each GPTQ/AquaKV-style fixed-length sequence creates a fresh cache.
        # GroupLogitKV stores its log-bias metadata on the model/attention modules,
        # so clear that model-level runtime state here to avoid cross-sequence leakage.
        reset_method_state(model, family, task_eval_dir, method)
        return KVCompressingDynamicCache(
            model=model,
            method=method,
            family=family,
            compress_ratio=ratio,
            sim_thr=-1.0,
            method_sim_thrs=method_sim_thrs,
            keepkv_config=keepkv_config,
            mymethod_config=mymethod_config,
            centroidkv_config=centroidkv_config,
            min_kv_len=min_kv_len,
            compress_every=compress_every,
            stats=stats,
            task_eval_dir=task_eval_dir,
            # The per-round target is computed from logical_seen_tokens inside
            # maybe_compress(); no fixed final-sequence target is stored here.
            target_kv_len=None,
        )
    return _factory




@contextlib.contextmanager
def force_attn_hidden_and_compress(model: nn.Module):
    """Temporarily wrap model.forward for KV-compression PPL evaluation.

    The AquaKV-style evaluate_perplexity() function is intentionally left
    untouched. This wrapper supplies the information needed by KV merge methods:
      - output_attentions=True for importance / merge scores;
      - output_hidden_states=True for q_t reconstruction;
      - logical position_ids/cache_position plus a physical-layout causal mask
        through KVCompressingDynamicCache.prepare_position_kwargs();
      - after each forward, update logical cache progress and run compression.
    """
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        cache = kwargs.get("past_key_values", None)

        # Determine current chunk length before kwargs are mutated.
        seq_len = None
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]
        if input_ids is not None:
            seq_len = int(input_ids.shape[1])
        elif inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])

        # Compressed-cache PPL needs logical RoPE positions and a physical-layout
        # causal mask. Baseline DynamicCache does not have this method.
        if hasattr(cache, "prepare_position_kwargs"):
            cache.prepare_position_kwargs(args, kwargs)

        # KeepKV captures the required masked pre-vote scores inside the eager
        # attention patch. Returning all post-softmax attentions and all hidden
        # states duplicates large tensors and is unnecessary. Other methods keep
        # their existing requirements.
        is_keepkv = getattr(cache, "_kv_method", None) == "KeepKV"
        is_cam = getattr(cache, "_kv_method", None) == "CaM"
        is_mymethod = getattr(cache, "_kv_method", None) == "mymethod"
        is_centroidkv = getattr(cache, "_kv_method", None) == "CentroidKV"
        if is_mymethod and hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:
                attn_mod = layer.self_attn
                attn_mod._kvmerge_capture_attention_scores = True
                attn_mod._kvmerge_attention_score_reduce = "sum"
                if hasattr(attn_mod, "_kvmerge_captured_attention_score"):
                    delattr(attn_mod, "_kvmerge_captured_attention_score")
                if hasattr(attn_mod, "_kvmerge_captured_score_log_scale"):
                    delattr(attn_mod, "_kvmerge_captured_score_log_scale")
        kwargs["output_attentions"] = not (is_keepkv or is_mymethod or is_centroidkv)
        kwargs["output_hidden_states"] = not (is_keepkv or is_cam or is_mymethod or is_centroidkv)
        kwargs["return_dict"] = True
        kwargs["use_cache"] = True

        out = original_forward(*args, **kwargs)

        captured_scores = None
        if is_mymethod and hasattr(model, "model") and hasattr(model.model, "layers"):
            captured_scores = tuple(
                getattr(layer.self_attn, "_kvmerge_captured_attention_score", None)
                for layer in model.model.layers
            )
            for layer in model.model.layers:
                layer.self_attn._kvmerge_capture_attention_scores = False
        if seq_len is not None and hasattr(cache, "mark_forward_consumed"):
            cache.mark_forward_consumed(seq_len)
        if hasattr(cache, "maybe_compress"):
            cache.maybe_compress(
                captured_scores if is_mymethod else getattr(out, "attentions", None),
                getattr(out, "hidden_states", None),
            )
        if is_mymethod and hasattr(model, "model") and hasattr(model.model, "layers"):
            # The tuple above owns the tensors for the duration of compression;
            # remove module references immediately afterwards to lower peak-live state.
            for layer in model.model.layers:
                if hasattr(layer.self_attn, "_kvmerge_captured_attention_score"):
                    delattr(layer.self_attn, "_kvmerge_captured_attention_score")
                if hasattr(layer.self_attn, "_kvmerge_captured_score_log_scale"):
                    delattr(layer.self_attn, "_kvmerge_captured_score_log_scale")

        return out

    model.forward = wrapped_forward
    try:
        yield
    finally:
        model.forward = original_forward


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


def mymethod_diagnostics_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mymethod_diagnostic_events": 0,
            "mymethod_mean_accepted_entry_ratio": None,
            "mymethod_mean_accepted_count_mass_ratio": None,
            "mymethod_total_evicted_count_mass": 0.0,
            "mymethod_total_accepted_count_mass": 0.0,
            "mymethod_total_rejected_count_mass": 0.0,
            "mymethod_mean_route_compatible_ratio": None,
            "mymethod_mean_mis_accept_given_routed_ratio": None,
            "mymethod_max_candidate_pool_size": 0,
            "mymethod_max_selected_candidates_per_group": 0,
            "mymethod_total_conflict_edges": 0,
            "mymethod_max_mis_rounds": 0,
            "mymethod_max_post_diameter_over_delta": None,
            "mymethod_max_post_radius_over_delta": None,
        }
    total_evicted = sum(float(row.get("evicted_count_mass", 0.0)) for row in rows)
    total_accepted = sum(float(row.get("accepted_count_mass", 0.0)) for row in rows)
    total_rejected = sum(float(row.get("rejected_count_mass", 0.0)) for row in rows)
    return {
        "mymethod_diagnostic_events": len(rows),
        "mymethod_mean_accepted_entry_ratio": sum(
            float(row.get("accepted_entry_ratio", 0.0)) for row in rows
        ) / len(rows),
        "mymethod_mean_accepted_count_mass_ratio": (
            total_accepted / total_evicted if total_evicted > 0.0 else 0.0
        ),
        "mymethod_total_evicted_count_mass": total_evicted,
        "mymethod_total_accepted_count_mass": total_accepted,
        "mymethod_total_rejected_count_mass": total_rejected,
        "mymethod_mean_route_compatible_ratio": sum(
            float(row.get("route_compatible_ratio", 0.0)) for row in rows
        ) / len(rows),
        "mymethod_mean_mis_accept_given_routed_ratio": sum(
            float(row.get("mis_accept_given_routed_ratio", 0.0)) for row in rows
        ) / len(rows),
        "mymethod_max_candidate_pool_size": max(
            int(row.get("max_candidate_pool_size", 0)) for row in rows
        ),
        "mymethod_max_selected_candidates_per_group": max(
            int(row.get("max_selected_candidates_per_group", 0)) for row in rows
        ),
        "mymethod_total_conflict_edges": sum(
            int(row.get("total_conflict_edges", 0)) for row in rows
        ),
        "mymethod_max_mis_rounds": max(
            int(row.get("max_mis_rounds", 0)) for row in rows
        ),
        "mymethod_max_post_diameter_over_delta": max(
            float(row.get("post_diameter_over_delta_max", 0.0)) for row in rows
        ),
        "mymethod_max_post_radius_over_delta": max(
            float(row.get("post_radius_over_delta_max", 0.0)) for row in rows
        ),
    }


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task_eval_dir", default="task_evaluation")
    parser.add_argument("--family", choices=["qwen", "llama"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"], choices=["wikitext2", "c4"])
    parser.add_argument("--kv_method", default="none", choices=["none", "SimCalKV", "GroupLogitKV", "mymethod", "KeepKV", "CaM", "CentroidKV"])
    parser.add_argument(
        "--compress_ratio", type=float, default=0.0,
        help="Fraction of KV tokens to remove; e.g. 0.4 compresses 40% and retains 60%.",
    )
    parser.add_argument("--cam_merge_size", type=int, default=16, help="Official CaM value-receiver window; default 16 for the PPL code path.")
    parser.add_argument("--cam_start_fraction", type=float, default=0.1, help="Fraction of the retained CaM budget assigned to protected start tokens.")
    parser.add_argument("--sim_thr_grouplogitkv", type=float, default=None, help="Similarity threshold for GroupLogitKV (default: 0.8); -1 disables threshold rejection.")
    parser.add_argument("--lambda_mymethod", type=float, default=0.5, help="mymethod normalized Euclidean complete-diameter threshold lambda (>0), with Delta=lambda*S_ref.")
    parser.add_argument("--mymethod_group_mode", choices=["full", "anchor_only", "eviction_only"], default="full", help="full=graph-MIS method; anchor_only=internal no-conflict-selection ablation; eviction_only=selector-only ablation.")
    parser.add_argument("--mymethod_selector", choices=["keepkv", "attention"], default="keepkv", help="Retained/evicted selector. attention uses current post-softmax attention mass; keepkv uses recent score history.")
    parser.add_argument("--mymethod_graph_backend", choices=["auto", "cuda", "torch"], default="auto", help="Graph-MIS backend. auto uses independent mymethod_cuda on CUDA and PyTorch otherwise.")
    parser.add_argument("--mymethod_route_backend", choices=["auto", "cuda", "torch"], default="torch", help="Certified routing backend. torch is the validated default; cuda remains available for experiments/memory-sensitive cases.")
    parser.add_argument("--mymethod_sink_size", type=int, default=4, help="Mandatory original-position sink entries for mymethod.")
    parser.add_argument("--mymethod_recent_window", type=int, default=128, help="Shared recent-token protection and score-observation window W.")
    parser.add_argument("--mymethod_recency_beta", type=float, default=1.0, help="Newest/oldest linear score-history weight ratio beta (>=1).")
    parser.add_argument("--mymethod_distance_chunk_size", type=int, default=256, help="Evicted-entry chunk size for full-retained distance GEMM, feasibility, and reductions; <=0 processes all evicted entries at once.")
    parser.add_argument("--mymethod_attention_chunk_size", type=int, default=128, help="Exact query-block size while capturing last-W base-score rows.")
    parser.add_argument("--mymethod_sink_can_absorb", type=int, choices=[0, 1], default=1, help="1 lets sink entries act as certified anchors; 0 forbids sink absorption.")
    parser.add_argument("--centroidkv_sink_budget", type=int, default=16, help="Official CentroidKV protected initial-token budget.")
    parser.add_argument("--centroidkv_local_budget", type=int, default=64, help="Official CentroidKV protected recent-token budget.")
    parser.add_argument("--centroidkv_chunk_size", type=int, default=256, help="Even chunk size for CentroidKV bipartite soft matching.")
    parser.add_argument("--centroidkv_merge_ratio", type=float, default=0.8, help="Initial official CentroidKV per-step merge ratio.")
    parser.add_argument("--centroidkv_decrease_rate", type=float, default=0.2, help="Official CentroidKV per-iteration merge-ratio decrease.")
    parser.add_argument(
        "--sim_thr_simcalkv", type=float, default=None,
        help="Similarity threshold for SimCalKV. Default: official value 0.0; -1 disables threshold rejection.",
    )
    parser.add_argument("--sim_thr_keepkv", type=float, default=None, help="Optional override for KeepKV decoding similarity threshold. Default uses the official value 0.8.")
    parser.add_argument("--keepkv_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--keepkv_prefill_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--keepkv_epsilon_times", type=float, default=1.0)
    parser.add_argument("--keepkv_pyramid_recent_ratio", type=float, default=0.2)
    parser.add_argument(
        "--keepkv_pyramid_budget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable official KeepKV pyramid layer-wise budget weights. "
            "Disabled by default; pass --keepkv-pyramid-budget to enable it. "
            "The exact global physical budget is preserved in both modes."
        ),
    )
    parser.add_argument("--keepkv_pyramid_decay_strategy", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--keepkv_pyramid_decay_ratio", type=float, default=0.8)
    parser.add_argument("--keepkv_distance_weight", type=float, default=1.2)
    parser.add_argument("--keepkv_sink_len", type=int, default=4)
    parser.add_argument(
        "--keepkv_protect_sink", action="store_true",
        help=(
            "Protect the first --keepkv_sink_len tokens from KeepKV eviction/merge. "
            "Disabled by default for fair cross-method comparison."
        ),
    )
    parser.add_argument(
        "--keepkv_assert_causal", action="store_true",
        help="Abort immediately if a current-chunk future token has a finite pre-electoral attention logit or nonzero captured importance.",
    )
    parser.add_argument(
        "--keepkv_disable_vote_bias", action="store_true",
        help="Diagnostic only: disable log(cluster_count) electoral-vote bias while retaining KeepKV compression.",
    )
    parser.add_argument(
        "--keepkv_runtime_only", action="store_true",
        help="Diagnostic only: install the KeepKV attention path and physical-layout mask but do not compress or update KeepKV state.",
    )
    parser.add_argument(
        "--max_eval_sequences", type=int, default=0,
        help="Limit PPL evaluation to the first N fixed-length sequences; 0 evaluates all sequences.",
    )
    # Deprecated aliases retained only for command-line compatibility.
    parser.add_argument("--keepkv_recent_ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--keepkv_sink_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--keepkv_score_epsilon", type=float, default=None, help=argparse.SUPPRESS)
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
    parser.add_argument(
        "--mymethod_diagnostics_jsonl",
        default=None,
        help=(
            "Optional JSONL path for per-layer/per-compression mymethod coverage and "
            "certificate diagnostics. Disabled by default to avoid GPU synchronization."
        ),
    )
    return parser


def main():
    args = build_argparser().parse_args()
    if args.kv_method == "mymethod" and not (args.lambda_mymethod > 0.0):
        raise ValueError("--lambda_mymethod must be > 0")
    if args.mymethod_sink_size < 0 or args.mymethod_recent_window < 0:
        raise ValueError("mymethod sink/recent sizes must be non-negative")
    if args.mymethod_recency_beta < 1.0:
        raise ValueError("--mymethod_recency_beta must be >= 1")
    if args.centroidkv_sink_budget < 0 or args.centroidkv_local_budget < 0:
        raise ValueError("CentroidKV sink/local budgets must be non-negative")
    if args.centroidkv_chunk_size < 2 or args.centroidkv_chunk_size % 2 != 0:
        raise ValueError("--centroidkv_chunk_size must be an even integer >= 2")
    if not 0.0 <= args.centroidkv_merge_ratio <= 1.0:
        raise ValueError("--centroidkv_merge_ratio must be in [0,1]")
    if args.centroidkv_decrease_rate < 0.0:
        raise ValueError("--centroidkv_decrease_rate must be non-negative")
    set_seed(args.seed)
    global _EVAL_MAX_SEQUENCES
    _EVAL_MAX_SEQUENCES = max(0, int(args.max_eval_sequences))

    task_eval_dir = Path(args.task_eval_dir).resolve()
    if str(task_eval_dir) not in sys.path:
        sys.path.insert(0, str(task_eval_dir))

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
    print(f"KV method: {args.kv_method}, ratio={args.compress_ratio}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    if args.kv_method == "CaM":
        if args.compress_every != 1:
            raise ValueError("Query-ordered chunk CaM requires --compress_every 1.")
        model._cam_eval_config = {
            "logical_length": int(args.model_seqlen),
            "merge_size": int(args.cam_merge_size),
            "start_fraction": float(args.cam_start_fraction),
        }
    if args.kv_method == "mymethod":
        for module in model.modules():
            if "attention" in module.__class__.__name__.lower() and hasattr(module, "forward"):
                module._kvmerge_attention_chunk_size = int(args.mymethod_attention_chunk_size)
                module._mymethod_selector = str(args.mymethod_selector)
                module._mymethod_score_window = (
                    int(args.mymethod_recent_window)
                    if args.mymethod_selector == "keepkv" else 0
                )
    if args.kv_method == "KeepKV":
        configured_attention_modules = 0
        for module in model.modules():
            name = module.__class__.__name__.lower()
            if "attention" in name and hasattr(module, "forward"):
                module._keepkv_assert_causal = bool(args.keepkv_assert_causal)
                module._keepkv_disable_vote_bias = bool(args.keepkv_disable_vote_bias)
                configured_attention_modules += 1
        if configured_attention_modules == 0:
            raise RuntimeError("No attention modules were found for KeepKV runtime diagnostics.")

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
    if args.mymethod_diagnostics_jsonl:
        Path(args.mymethod_diagnostics_jsonl).parent.mkdir(parents=True, exist_ok=True)

    method_sim_thrs = {
        "GroupLogitKV": args.sim_thr_grouplogitkv,
        "SimCalKV": args.sim_thr_simcalkv,
        "KeepKV": args.sim_thr_keepkv,
    }
    # Resolve deprecated aliases without changing official defaults.
    pyramid_recent_ratio = (
        args.keepkv_recent_ratio
        if args.keepkv_recent_ratio is not None
        else args.keepkv_pyramid_recent_ratio
    )
    sink_len = args.keepkv_sink_size if args.keepkv_sink_size is not None else args.keepkv_sink_len
    if args.keepkv_score_epsilon is not None:
        raise ValueError(
            "--keepkv_score_epsilon is obsolete and is not part of official KeepKV. "
            "Use --keepkv_epsilon_times instead."
        )
    mymethod_config = {
        "selector": str(args.mymethod_selector),
        "lambda_mymethod": float(args.lambda_mymethod),
        "group_mode": str(args.mymethod_group_mode),
        "graph_backend": str(args.mymethod_graph_backend),
        "route_backend": str(args.mymethod_route_backend),
        "sink_size": int(args.mymethod_sink_size),
        "recent_window": int(args.mymethod_recent_window),
        "recency_beta": float(args.mymethod_recency_beta),
        "distance_chunk_size": int(args.mymethod_distance_chunk_size),
        "sink_can_absorb": bool(args.mymethod_sink_can_absorb),
    }
    centroidkv_config = {
        "sink_budget": int(args.centroidkv_sink_budget),
        "local_budget": int(args.centroidkv_local_budget),
        "chunk_size": int(args.centroidkv_chunk_size),
        "merge_ratio": float(args.centroidkv_merge_ratio),
        "decrease_rate": float(args.centroidkv_decrease_rate),
    }

    keepkv_config = {
        "similarity_threshold": (
            args.sim_thr_keepkv
            if args.sim_thr_keepkv is not None
            else args.keepkv_similarity_threshold
        ),
        "prefill_similarity_threshold": args.keepkv_prefill_similarity_threshold,
        "epsilon_times": args.keepkv_epsilon_times,
        "pyramid_recent_ratio": pyramid_recent_ratio,
        "use_pyramid_budget": bool(args.keepkv_pyramid_budget),
        "pyramid_decay_strategy": args.keepkv_pyramid_decay_strategy,
        "pyramid_decay_ratio": args.keepkv_pyramid_decay_ratio,
        "distance_weight": args.keepkv_distance_weight,
        "sink_len": sink_len,
        "protect_sink": bool(args.keepkv_protect_sink),
        "logical_length": args.model_seqlen,
        "diagnostic_runtime_only": bool(args.keepkv_runtime_only),
    }

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
                method_sim_thrs=method_sim_thrs,
                keepkv_config=keepkv_config,
                mymethod_config=mymethod_config,
                centroidkv_config=centroidkv_config,
                min_kv_len=args.min_kv_len,
                compress_every=args.compress_every,
                stats=stats,
                model_seqlen=args.model_seqlen,
            )
        else:
            stats = {"num_compressions": 0, "total_compress_time_s": 0.0}

        reset_method_state(model, args.family, task_eval_dir, args.kv_method)
        if args.kv_method == "mymethod" and args.mymethod_diagnostics_jsonl:
            model._mymethod_diagnostics_rows = []
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
        diagnostic_rows = list(getattr(model, "_mymethod_diagnostics_rows", []))
        if args.kv_method == "mymethod" and args.mymethod_diagnostics_jsonl:
            with open(args.mymethod_diagnostics_jsonl, "a", encoding="utf-8") as f:
                for row in diagnostic_rows:
                    enriched = {
                        "model_name": args.model_name,
                        "family": args.family,
                        "dataset": dataset_name,
                        "compress_ratio": float(args.compress_ratio),
                        "lambda_mymethod": float(args.lambda_mymethod),
                        "group_mode": str(args.mymethod_group_mode),
                        "graph_backend": str(args.mymethod_graph_backend),
                        "route_backend": str(args.mymethod_route_backend),
                        **row,
                    }
                    f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
        reset_method_state(model, args.family, task_eval_dir, args.kv_method)

        result = {
            "model_name": args.model_name,
            "family": args.family,
            "dataset": dataset_name,
            "kv_method": args.kv_method,
            "compress_ratio": args.compress_ratio,
            "cam_protocol": "query_ordered_chunk_approximation" if args.kv_method == "CaM" else None,
            "cam_merge_size": args.cam_merge_size if args.kv_method == "CaM" else None,
            "cam_start_fraction": args.cam_start_fraction if args.kv_method == "CaM" else None,
            "sim_thr_effective": _effective_sim_thr(args.kv_method, method_sim_thrs),
            "sim_thr_grouplogitkv": args.sim_thr_grouplogitkv,
            "lambda_mymethod": args.lambda_mymethod,
            "mymethod_group_mode": args.mymethod_group_mode,
            "mymethod_selector": args.mymethod_selector,
            "mymethod_graph_backend": args.mymethod_graph_backend,
            "mymethod_route_backend": args.mymethod_route_backend,
            "mymethod_sink_size": args.mymethod_sink_size,
            "mymethod_recent_window": args.mymethod_recent_window,
            "mymethod_recency_beta": args.mymethod_recency_beta,
            "mymethod_distance_chunk_size": args.mymethod_distance_chunk_size,
            "mymethod_attention_chunk_size": args.mymethod_attention_chunk_size,
            "mymethod_sink_can_absorb": bool(args.mymethod_sink_can_absorb),
            "centroidkv_sink_budget": args.centroidkv_sink_budget if args.kv_method == "CentroidKV" else None,
            "centroidkv_local_budget": args.centroidkv_local_budget if args.kv_method == "CentroidKV" else None,
            "centroidkv_chunk_size": args.centroidkv_chunk_size if args.kv_method == "CentroidKV" else None,
            "centroidkv_merge_ratio": args.centroidkv_merge_ratio if args.kv_method == "CentroidKV" else None,
            "centroidkv_decrease_rate": args.centroidkv_decrease_rate if args.kv_method == "CentroidKV" else None,
            "sim_thr_simcalkv": args.sim_thr_simcalkv,
            "sim_thr_keepkv": args.sim_thr_keepkv,
            "keepkv_similarity_threshold": keepkv_config["similarity_threshold"],
            "keepkv_prefill_similarity_threshold": keepkv_config["prefill_similarity_threshold"],
            "keepkv_epsilon_times": keepkv_config["epsilon_times"],
            "keepkv_pyramid_recent_ratio": keepkv_config["pyramid_recent_ratio"],
            "keepkv_pyramid_budget": keepkv_config["use_pyramid_budget"],
            "keepkv_pyramid_decay_strategy": keepkv_config["pyramid_decay_strategy"],
            "keepkv_pyramid_decay_ratio": keepkv_config["pyramid_decay_ratio"],
            "keepkv_distance_weight": keepkv_config["distance_weight"],
            "keepkv_sink_len": keepkv_config["sink_len"],
            "keepkv_protect_sink": keepkv_config["protect_sink"],
            "keepkv_assert_causal": bool(args.keepkv_assert_causal),
            "keepkv_disable_vote_bias": bool(args.keepkv_disable_vote_bias),
            "keepkv_runtime_only": bool(args.keepkv_runtime_only),
            "max_eval_sequences": int(args.max_eval_sequences),
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
            "num_eval_tokens": int(os.environ.get("KV_EVAL_ACTUAL_TOKENS", data.shape[1])),
            "num_eval_sequences": int(os.environ.get("KV_EVAL_ACTUAL_SEQUENCES", 0)),
        }
        result.update(stats_summary(stats))
        if args.kv_method == "mymethod":
            result.update(mymethod_diagnostics_summary(diagnostic_rows))
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
