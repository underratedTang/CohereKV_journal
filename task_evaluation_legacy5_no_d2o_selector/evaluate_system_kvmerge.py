#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-ready system efficiency benchmark for KV-cache compression.

The benchmark reports three paper-facing metrics with identical boundaries:
  1. compression latency: explicit post-prefill compression entry -> cache ready;
  2. end-to-end latency: request input transfer/prefill -> fixed-length decode complete;
  3. throughput: generated output tokens / end-to-end latency.

Method-specific statistic collection performed during prefill is included in end-to-end
latency, but not double-counted in explicit compression latency. Existing comparison-method implementations under models/ are not modified.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import evaluate_longbench_kvmerge as lb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name", "--model", dest="model_name", required=True,
                   help="LongBench logical model key, e.g. llama-3.1-8B-Instruct")
    p.add_argument("--model_id", "--model_path", dest="model_id", default=None,
                   help="Hugging Face model ID or local model path")
    p.add_argument("--task_eval_dir", default=None)
    p.add_argument("--arch", choices=["llama", "qwen", "auto"], default="auto")
    p.add_argument("--kv_method", choices=lb.METHODS, required=True)
    p.add_argument("--compress_ratio", type=float, default=0.2,
                   help="Fraction removed; retained ratio is 1-compress_ratio")
    p.add_argument("--sim_thr_grouplogitkv", type=float, default=0.8)
    p.add_argument("--lambda_mymethod", type=float, default=0.5)
    p.add_argument("--mymethod_group_mode", choices=["full", "anchor_only", "eviction_only"], default="full")
    p.add_argument("--mymethod_selector", choices=["keepkv", "attention"], default="keepkv")
    p.add_argument("--mymethod_graph_backend", choices=["auto", "cuda", "torch"], default="auto")
    p.add_argument("--mymethod_route_backend", choices=["auto", "cuda", "torch"], default="torch")
    p.add_argument("--mymethod_sink_size", type=int, default=4)
    p.add_argument("--mymethod_recent_window", type=int, default=128)
    p.add_argument("--mymethod_recency_beta", type=float, default=1.0)
    p.add_argument("--mymethod_distance_chunk_size", type=int, default=256)
    p.add_argument("--mymethod_attention_chunk_size", type=int, default=128)
    p.add_argument("--mymethod_sink_can_absorb", type=int, choices=[0, 1], default=1)
    p.add_argument("--centroidkv_sink_budget", type=int, default=16)
    p.add_argument("--centroidkv_local_budget", type=int, default=64)
    p.add_argument("--centroidkv_chunk_size", type=int, default=256)
    p.add_argument("--centroidkv_merge_ratio", type=float, default=0.8)
    p.add_argument("--centroidkv_decrease_rate", type=float, default=0.2)
    p.add_argument("--sim_thr_simcalkv", type=float, default=0.0)
    p.add_argument("--sim_thr_keepkv", type=float, default=0.8)

    p.add_argument("--dataset", default="gov_report", choices=["gov_report"])
    p.add_argument("--input_length", type=int, default=8192)
    p.add_argument("--generation_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup_runs", type=int, default=3)
    p.add_argument("--measured_runs", type=int, default=10)
    p.add_argument("--sample_start_index", type=int, default=0,
                   help="Start offset in the deterministic length-qualified sample list")

    p.add_argument("--dtype", choices=["fp16", "float16", "bf16", "bfloat16", "auto"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--keepkv_similarity_chunk_size", type=int, default=256,
                   help="Fixed KeepKV similarity chunk used inside timed compression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cleanup_between_runs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.compress_ratio < 1.0):
        raise ValueError("--compress_ratio must satisfy 0 <= ratio < 1")
    for name in ("input_length", "generation_length", "batch_size", "measured_runs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
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
    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be non-negative")
    if args.sample_start_index < 0:
        raise ValueError("--sample_start_index must be non-negative")




def _tensor_tree_bytes(obj, seen=None) -> int:
    """Count unique tensor storage referenced by a nested Python object."""
    if seen is None:
        seen = set()
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        key = (obj.device.type, obj.device.index, obj.untyped_storage().data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_tree_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_tensor_tree_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return _tensor_tree_bytes(vars(obj), seen)
    return 0


def cache_bytes(past) -> int:
    return _tensor_tree_bytes(lb._as_legacy_tuple(past))


def method_side_info_bytes(model, method: str) -> int:
    attrs = {
        "mymethod": (
            "_mymethod_counts", "_mymethod_radii", "_mymethod_diameters",
            "_mymethod_score_history", "_mymethod_positions", "_mymethod_next_positions",
            "_mymethod_reference_scales",
        ),
        "GroupLogitKV": ("_grouplogitkv_counts",),
        "KeepKV": ("_keepkv_layer_states", "_keepkv_counts"),
        "CentroidKV": ("_centroidkv_counts",),
        "CaM": ("_cam_chunkwise_state",),
        "SimCalKV": tuple(),
        "None": tuple(),
    }.get(method, tuple())
    seen = set()
    return sum(_tensor_tree_bytes(getattr(model, name, None), seen) for name in attrs)


def prepare_prompt(sample: Dict[str, Any], prompt_format: str, tokenizer, model_name: str) -> str:
    prompt = prompt_format.format(**sample)
    # gov_report is chat-wrapped in the existing LongBench evaluator.
    return lb.build_chat(tokenizer, prompt, model_name)


def select_fixed_length_batches(
    data: Sequence[Dict[str, Any]],
    prompt_format: str,
    tokenizer,
    model_name: str,
    input_length: int,
    batch_size: int,
    total_runs: int,
    start_index: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[List[int]]]:
    """Select distinct prompts with at least input_length tokens and truncate exactly.

    No text repetition or padding is used. Selection is deterministic and therefore
    identical across independently launched method/budget processes.
    """
    needed = total_runs * batch_size
    qualified: List[Tuple[int, torch.Tensor]] = []
    if not data:
        raise RuntimeError("LongBench dataset is empty")

    # Circular scan makes start_index useful without changing deterministic order.
    for offset in range(len(data)):
        idx = (start_index + offset) % len(data)
        prompt = prepare_prompt(data[idx], prompt_format, tokenizer, model_name)
        ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0]
        if int(ids.numel()) >= input_length:
            qualified.append((idx, ids[:input_length].contiguous()))
            if len(qualified) == needed:
                break

    if len(qualified) < needed:
        raise RuntimeError(
            f"Need {needed} distinct {input_length}-token prompts for "
            f"{total_runs} runs x batch_size {batch_size}, but found {len(qualified)}. "
            "Reduce --measured_runs/--batch_size/--input_length."
        )

    batches: List[Dict[str, torch.Tensor]] = []
    index_batches: List[List[int]] = []
    for run in range(total_runs):
        block = qualified[run * batch_size:(run + 1) * batch_size]
        input_ids = torch.stack([ids for _, ids in block], dim=0)
        batches.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
        index_batches.append([idx for idx, _ in block])
    return batches, index_batches


def move_batch(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=False) for k, v in inputs.items()}


def standard_prefill(model, inputs, method: str):
    """Prompt prefill and method-specific statistic collection."""
    need_scores = lb._need_attention_scores_for_method(method)
    need_hidden = lb._need_hidden_states_for_method(method)
    if need_scores:
        lb._set_reduced_attention_capture(
            model, True, reduce=lb._attention_reduce_mode_for_method(method)
        )
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=need_hidden,
                use_cache=True,
                return_dict=True,
            )
        scores = lb._get_reduced_attention_scores(model) if need_scores else None
    finally:
        if need_scores:
            lb._set_reduced_attention_capture(
                model, False, reduce=lb._attention_reduce_mode_for_method(method)
            )

    # Clone the final-position logits so the full [B, L, V] logits allocation can be released
    # before compression. Preserve only objects required by the selected method.
    last_logits = outputs.logits[:, -1, :].detach().clone()
    full_past = outputs.past_key_values
    hidden_states = outputs.hidden_states if need_hidden else None
    del outputs
    return full_past, last_logits, scores, hidden_states


def keepkv_prefill_only(
    model,
    inputs,
    apply_kv,
    ratio: float,
    device: str,
    similarity_chunk_size: int,
):
    """Run KeepKV prompt prefill/statistic collection without compression.

    This preserves the KeepKV adapter call sequence while using the full
    prompt length as a single prefill chunk. Finalization is deliberately left
    for the separately timed compression region.
    """
    prepare = getattr(apply_kv, "keepkv_prepare_longbench", None)
    set_chunk = getattr(apply_kv, "keepkv_set_chunk", None)
    finalize = getattr(apply_kv, "keepkv_finalize_longbench", None)
    if not callable(prepare) or not callable(set_chunk) or not callable(finalize):
        raise RuntimeError("Installed KeepKV adapter lacks LongBench one-shot APIs")

    input_ids = inputs["input_ids"]
    total_len = int(input_ids.shape[1])
    # Use the complete prompt as a single prefill chunk. This matches the
    # one-shot prompt ingestion used by the other methods and removes the
    # adapter-specific small-chunk overhead without changing KeepKV scoring
    # or compression logic.
    chunk_size = total_len
    prepare(model, total_len, ratio,
            similarity_chunk_size=max(1, int(similarity_chunk_size)))

    past = None
    last_logits = None
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        ids = input_ids[:, start:end]
        q_len = end - start
        legacy = None if past is None else lb._as_legacy_tuple(past)
        physical_prefix = 0 if legacy is None else int(legacy[0][0].shape[-2])
        set_chunk(model, start)
        kwargs = {
            "input_ids": ids,
            "attention_mask": torch.ones(
                (ids.shape[0], physical_prefix + q_len),
                device=device,
                dtype=inputs["attention_mask"].dtype,
            ),
            "position_ids": torch.arange(start, end, device=device,
                                         dtype=torch.long).unsqueeze(0).expand(ids.shape[0], -1),
            "past_key_values": past,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        with torch.inference_mode():
            out = model(**kwargs)
        last_logits = out.logits[:, -1, :].detach().clone()
        past = out.past_key_values
        del out

    if last_logits is None or past is None:
        raise RuntimeError("Empty prompt in KeepKV prefill")
    return past, last_logits, finalize


def cache_layer_lengths(past) -> List[int]:
    legacy = lb._as_legacy_tuple(past)
    return [int(layer[0].shape[-2]) for layer in legacy]


def fixed_length_decode(
    model,
    inputs: Dict[str, torch.Tensor],
    past,
    first_tokens: torch.Tensor,
    generation_length: int,
    device: str,
) -> torch.Tensor:
    """Greedy fixed-length decode; EOS never shortens a system measurement."""
    batch = int(inputs["input_ids"].shape[0])
    logical_prompt_len = int(inputs["input_ids"].shape[1])
    physical_prefix_len = int(lb._as_legacy_tuple(past)[0][0].shape[-2])
    generated = torch.empty((batch, generation_length), dtype=torch.long, device=device)
    generated[:, 0] = first_tokens
    if generation_length == 1:
        return generated

    decode_mask = torch.ones(
        (batch, physical_prefix_len + generation_length - 1),
        dtype=inputs["attention_mask"].dtype,
        device=device,
    )
    positions = torch.arange(
        logical_prompt_len,
        logical_prompt_len + generation_length - 1,
        dtype=torch.long,
        device=device,
    ).view(1, -1).expand(batch, -1)

    current = first_tokens.view(batch, 1)
    with torch.inference_mode():
        for step in range(generation_length - 1):
            out = model(
                input_ids=current,
                attention_mask=decode_mask[:, :physical_prefix_len + step + 1],
                position_ids=positions[:, step:step + 1],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated[:, step + 1] = current[:, 0]
            del out
    return generated


def selected_threshold(args: argparse.Namespace, method: str):
    if method == "GroupLogitKV":
        return float(args.sim_thr_grouplogitkv)
    if method == "mymethod":
        return float(args.lambda_mymethod)
    if method == "SimCalKV":
        return float(args.sim_thr_simcalkv)
    if method == "KeepKV":
        return float(args.sim_thr_keepkv)
    return None


def mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs))


def pstdev(xs: Sequence[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    args = parse_args()
    validate_args(args)
    lb.seed_everything(args.seed)

    root = Path(args.task_eval_dir or Path(__file__).resolve().parent).resolve()
    method = lb.normalize_kv_method(args.kv_method)
    model2path = json.loads((root / "LongBench/config/model2path.json").read_text())
    if args.model_name not in model2path:
        raise KeyError(f"{args.model_name!r} is not in LongBench/config/model2path.json")
    model_id = args.model_id or model2path[args.model_name]
    arch = lb.infer_arch(args.model_name, model_id) if args.arch == "auto" else args.arch

    apply_kv, reset_kv = lb.setup_kv_imports(
        str(root),
        arch,
        method,
        mymethod_lambda=args.lambda_mymethod,
        mymethod_selector=args.mymethod_selector,
        mymethod_sink_size=args.mymethod_sink_size,
        mymethod_recent_window=args.mymethod_recent_window,
        mymethod_recency_beta=args.mymethod_recency_beta,
        mymethod_distance_chunk_size=args.mymethod_distance_chunk_size,
        mymethod_sink_can_absorb=bool(args.mymethod_sink_can_absorb),
        mymethod_group_mode=str(args.mymethod_group_mode),
        mymethod_graph_backend=str(args.mymethod_graph_backend),
        mymethod_route_backend=str(args.mymethod_route_backend),
        centroidkv_sink_budget=args.centroidkv_sink_budget,
        centroidkv_local_budget=args.centroidkv_local_budget,
        centroidkv_chunk_size=args.centroidkv_chunk_size,
        centroidkv_merge_ratio=args.centroidkv_merge_ratio,
        centroidkv_decrease_rate=args.centroidkv_decrease_rate,
    )
    model, tokenizer = lb.load_model_and_tokenizer(
        model_id,
        args.model_name,
        args.device,
        torch_dtype=lb.get_torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    if method == "mymethod":
        for attn_mod in lb._self_attention_modules(model):
            attn_mod._kvmerge_attention_chunk_size = int(args.mymethod_attention_chunk_size)
            attn_mod._mymethod_selector = str(args.mymethod_selector)
            attn_mod._mymethod_score_window = (
                int(args.mymethod_recent_window)
                if args.mymethod_selector == "keepkv" else 0
            )

    data = lb.load_longbench_dataset(args.dataset, str(root))
    prompt_map = json.loads((root / "LongBench/config/dataset2prompt.json").read_text())
    total_runs = args.warmup_runs + args.measured_runs
    cpu_batches, sample_index_batches = select_fixed_length_batches(
        data=data,
        prompt_format=prompt_map[args.dataset],
        tokenizer=tokenizer,
        model_name=args.model_name,
        input_length=args.input_length,
        batch_size=args.batch_size,
        total_runs=total_runs,
        start_index=args.sample_start_index,
    )
    sim_thr = selected_threshold(args, method)

    records: List[Dict[str, Any]] = []
    for run_idx, cpu_inputs in enumerate(cpu_batches):
        reset_kv(model)
        lb._clear_reduced_attention_scores(model)

        # E2E starts before per-request host-to-device transfer and includes prefill,
        # method-specific statistic collection, explicit compression, and decoding.
        sync(args.device)
        e2e_start = time.perf_counter()
        inputs = move_batch(cpu_inputs, args.device)

        scores = None
        hidden_states = None
        if method == "KeepKV":
            full_past, last_logits, keepkv_finalize = keepkv_prefill_only(
                model=model,
                inputs=inputs,
                apply_kv=apply_kv,
                ratio=args.compress_ratio,
                device=args.device,
                similarity_chunk_size=args.keepkv_similarity_chunk_size,
            )
        else:
            full_past, last_logits, scores, hidden_states = standard_prefill(
                model, inputs, method
            )

        pre_compression_lengths = cache_layer_lengths(full_past)

        # Compression time covers only the explicit post-prefill cache transformation.
        # Statistics produced inside prefill remain part of E2E and are not double-counted.
        compression_peak_extra_bytes = 0
        if method == "None":
            compressed_past = full_past
            compression_s = 0.0
        else:
            sync(args.device)
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                compression_baseline_bytes = torch.cuda.memory_allocated()
            else:
                compression_baseline_bytes = 0
            compression_start = time.perf_counter()
            if method == "KeepKV":
                compressed_past = keepkv_finalize(
                    model, lb._as_legacy_tuple(full_past), sim_thr
                )
            else:
                with torch.inference_mode():
                    compressed_past = apply_kv(
                        method,
                        model,
                        full_past,
                        scores,
                        hidden_states,
                        args.compress_ratio,
                        sim_thr,
                    )
            sync(args.device)
            compression_s = time.perf_counter() - compression_start
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                compression_peak_extra_bytes = max(
                    0, int(torch.cuda.max_memory_allocated() - compression_baseline_bytes)
                )

        compressed_cache_bytes = cache_bytes(compressed_past)
        side_info_bytes = method_side_info_bytes(model, method)
        first_tokens = last_logits.argmax(dim=-1)
        post_compression_lengths = cache_layer_lengths(compressed_past)
        generated = fixed_length_decode(
            model=model,
            inputs=inputs,
            past=compressed_past,
            first_tokens=first_tokens,
            generation_length=args.generation_length,
            device=args.device,
        )
        sync(args.device)
        end_to_end_s = time.perf_counter() - e2e_start

        generated_tokens = int(args.batch_size * args.generation_length)
        if run_idx >= args.warmup_runs:
            records.append({
                "measured_run": run_idx - args.warmup_runs,
                "dataset_indices": sample_index_batches[run_idx],
                "compression_time_s": float(compression_s),
                "end_to_end_time_s": float(end_to_end_s),
                "throughput_tok_s": float(generated_tokens / end_to_end_s),
                "generated_tokens": generated_tokens,
                "pre_compression_layer_lengths": pre_compression_lengths,
                "post_compression_layer_lengths": post_compression_lengths,
                "compressed_kv_mib": compressed_cache_bytes / (1024 ** 2),
                "method_side_info_mib": side_info_bytes / (1024 ** 2),
                "persistent_kv_plus_side_info_mib": (compressed_cache_bytes + side_info_bytes) / (1024 ** 2),
                "compression_peak_extra_mib": compression_peak_extra_bytes / (1024 ** 2),
            })

        del inputs, full_past, compressed_past, generated, first_tokens, last_logits
        if scores is not None:
            del scores
        if hidden_states is not None:
            del hidden_states
        reset_kv(model)
        lb._clear_reduced_attention_scores(model)
        if args.cleanup_between_runs:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sync(args.device)

    compression_values = [r["compression_time_s"] for r in records]
    e2e_values = [r["end_to_end_time_s"] for r in records]
    throughput_values = [r["throughput_tok_s"] for r in records]
    compressed_kv_values = [r["compressed_kv_mib"] for r in records]
    side_info_values = [r["method_side_info_mib"] for r in records]
    persistent_values = [r["persistent_kv_plus_side_info_mib"] for r in records]
    compression_peak_values = [r["compression_peak_extra_mib"] for r in records]
    expected_retained = args.input_length if method == "None" else int(round(
        args.input_length * (1.0 - args.compress_ratio)
    ))

    result = {
        "protocol": {
            "dataset": args.dataset,
            "sampling": "distinct length-qualified prompts; deterministic dataset order",
            "input_policy": "chat-wrap, require >= input_length tokens, truncate exactly; no repetition/padding",
            "prefill_timed": True,
            "compression_boundary": "explicit post-prefill compression entry to compressed cache ready, CUDA synchronized",
            "end_to_end_boundary": "per-request host-to-device transfer and prompt prefill to fixed-length generation complete, CUDA synchronized",
            "throughput_definition": "batch_size * generation_length / end_to_end_time",
            "statistic_collection": "included in end-to-end latency when performed during prefill; not double-counted in compression latency",
            "decode": "greedy, fixed length, no EOS early stop",
        },
        "model_name": args.model_name,
        "model_id": model_id,
        "arch": arch,
        "kv_method": method,
        "compress_ratio": 0.0 if method == "None" else args.compress_ratio,
        "retained_ratio": 1.0 if method == "None" else 1.0 - args.compress_ratio,
        "expected_retained_slots_per_layer": expected_retained,
        "sim_thr": "na" if sim_thr is None else sim_thr,
        "lambda_mymethod": args.lambda_mymethod if method == "mymethod" else "na",
        "mymethod_group_mode": args.mymethod_group_mode if method == "mymethod" else "na",
        "mymethod_selector": args.mymethod_selector if method == "mymethod" else "na",
        "mymethod_graph_backend": args.mymethod_graph_backend if method == "mymethod" else "na",
        "mymethod_route_backend": args.mymethod_route_backend if method == "mymethod" else "na",
        "mymethod_sink_size": args.mymethod_sink_size if method == "mymethod" else "na",
        "mymethod_recent_window": args.mymethod_recent_window if method == "mymethod" else "na",
        "mymethod_recency_beta": args.mymethod_recency_beta if method == "mymethod" else "na",
        "mymethod_distance_chunk_size": args.mymethod_distance_chunk_size if method == "mymethod" else "na",
        "mymethod_attention_chunk_size": args.mymethod_attention_chunk_size if method == "mymethod" else "na",
        "mymethod_sink_can_absorb": bool(args.mymethod_sink_can_absorb) if method == "mymethod" else "na",
        "centroidkv_sink_budget": args.centroidkv_sink_budget if method == "CentroidKV" else "na",
        "centroidkv_local_budget": args.centroidkv_local_budget if method == "CentroidKV" else "na",
        "centroidkv_chunk_size": args.centroidkv_chunk_size if method == "CentroidKV" else "na",
        "centroidkv_merge_ratio": args.centroidkv_merge_ratio if method == "CentroidKV" else "na",
        "centroidkv_decrease_rate": args.centroidkv_decrease_rate if method == "CentroidKV" else "na",
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "generation_length": args.generation_length,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "num_distinct_measured_samples": args.measured_runs * args.batch_size,
        "sample_start_index": args.sample_start_index,
        "prefill_chunk_size": args.input_length if method == "KeepKV" else "na",
        "keepkv_similarity_chunk_size": args.keepkv_similarity_chunk_size if method == "KeepKV" else "na",
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "device": args.device,
        "compression_time_ms_mean": 1000.0 * mean(compression_values),
        "compression_time_ms_std": 1000.0 * pstdev(compression_values),
        "end_to_end_time_s_mean": mean(e2e_values),
        "end_to_end_time_s_std": pstdev(e2e_values),
        "throughput_tok_s_mean": mean(throughput_values),
        "throughput_tok_s_std": pstdev(throughput_values),
        "compressed_kv_mib_mean": mean(compressed_kv_values),
        "method_side_info_mib_mean": mean(side_info_values),
        "persistent_kv_plus_side_info_mib_mean": mean(persistent_values),
        "compression_peak_extra_mib_mean": mean(compression_peak_values),
        "compression_peak_extra_mib_std": pstdev(compression_peak_values),
        "paper_recommendation": (
            "Use compression latency, persistent KV+side-info memory, and compression peak extra memory "
            "as primary systems results. Treat end-to-end latency as supplementary because method-specific "
            "attention/statistic collection changes the prefill path."
        ),
        "runs": records,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
