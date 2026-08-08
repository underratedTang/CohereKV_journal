#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Component and oracle ablations for GroupLogitKV.

Component variants (unchanged):
  eviction_only: importance Bottom-K slots are discarded.
  group_merge_no_bias: normal GroupLogitKV merging, but decode-time log(C) is zeroed.
  full: unchanged GroupLogitKV.

Oracle task:
  Reconstructs the same importance selection and nearest-kept-key assignment used
  by GroupLogitKV, then compares
      centroid-only:             mean_i z_i
      count-calibrated:          mean_i z_i + log C
  against the exact group logit  logsumexp_i z_i.
  z_i may be recovered from log attention probabilities because the shared
  softmax normalizer cancels from all three quantities.

Only evaluator behavior is changed; files under models/ are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)


def _kv_scores(attn, h_kv):
    if attn.dim() == 3:
        score = attn
    elif attn.dim() == 4:
        score = attn.sum(dim=-2)
    else:
        raise ValueError(f"Unsupported attention shape: {tuple(attn.shape)}")
    if score.shape[1] != h_kv:
        groups = score.shape[1] // h_kv
        score = score.view(score.shape[0], h_kv, groups, score.shape[-1]).mean(dim=2)
    return score.float()


def eviction_only(model, past_key_values, attention, ratio, target_length=None):
    cache = _legacy(past_key_values)
    out = []
    for layer_idx, (k, v) in enumerate(cache):
        b, h, length, d = k.shape
        keep_len = int(target_length) if target_length is not None else max(1, length - int(length * ratio))
        keep_len = min(length, keep_len)
        if keep_len >= length:
            out.append((k, v)); continue
        scores = _kv_scores(attention[layer_idx], h)
        keep_idx = torch.topk(scores, k=keep_len, dim=-1, largest=True, sorted=False).indices
        keep_idx = keep_idx.sort(dim=-1).values
        gather = keep_idx.unsqueeze(-1).expand(b, h, keep_len, d)
        out.append((torch.gather(k, 2, gather), torch.gather(v, 2, gather)))
    for name in ("_grouplogitkv_counts", "_grouplogitkv_log_bias_q"):
        setattr(model, name, None)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn_mod = layer.self_attn
            attn_mod._grouplogitkv_eager_enabled = False
            attn_mod._grouplogitkv_sdpa_enabled = False
            attn_mod._grouplogitkv_log_bias_q = None
    return tuple(out)


def zero_group_bias(model):
    biases = getattr(model, "_grouplogitkv_log_bias_q", None)
    if biases is not None:
        zeroed = [torch.zeros_like(x) for x in biases]
        model._grouplogitkv_log_bias_q = zeroed
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn._grouplogitkv_log_bias_q = zeroed[idx]


def run_ppl(variant, forwarded):
    import evaluate_ppl_kvmerge as ppl
    original = ppl.apply_kv_method

    def patched(method, family, model, past_key_values, attention, hidden_states,
                compress_ratio, method_sim_thrs=None, keepkv_config=None,
                task_eval_dir=None, target_kv_len=None, mymethod_config=None,
                **unused_kwargs):
        if variant == "eviction_only":
            return eviction_only(model, past_key_values, attention, compress_ratio, target_kv_len)
        result = original(
            "GroupLogitKV", family, model, past_key_values, attention,
            hidden_states, compress_ratio,
            method_sim_thrs=method_sim_thrs,
            keepkv_config=keepkv_config,
            task_eval_dir=task_eval_dir,
            target_kv_len=target_kv_len,
            mymethod_config=mymethod_config,
        )
        if variant == "group_merge_no_bias":
            zero_group_bias(model)
        return result

    ppl.apply_kv_method = patched
    sys.argv = ["evaluate_ppl_kvmerge.py", *forwarded]
    if "--kv_method" not in forwarded:
        sys.argv += ["--kv_method", "GroupLogitKV"]
    ppl.main()


# def run_longbench(variant, forwarded):
#     import evaluate_longbench_kvmerge as lb
#     original_setup = lb.setup_kv_imports

#     def patched_setup(task_eval_dir, arch, kv_method):
#         apply, reset = original_setup(task_eval_dir, arch, "GroupLogitKV")
#         def wrapped(method, model, pkv, attn, hidden, ratio, sim_thr):
#             if variant == "eviction_only":
#                 return eviction_only(model, pkv, attn, ratio)
#             result = apply("GroupLogitKV", model, pkv, attn, hidden, ratio, sim_thr)
#             if variant == "group_merge_no_bias":
#                 zero_group_bias(model)
#             return result
#         return wrapped, reset
def run_longbench(variant, forwarded):
    # ablation 默认复用已有 LongBench 结果，不强制覆盖
    forwarded = [x for x in forwarded if x != "--overwrite"]

    import evaluate_longbench_kvmerge as lb
    original_setup = lb.setup_kv_imports

    def patched_setup(
        task_eval_dir,
        arch,
        kv_method,
        *setup_args,
        **setup_kwargs,
    ):
        # 强制使用 GroupLogitKV，但保留新版 evaluator 传入的其他参数，
        # 包括 simcalkv_similarity_chunk_size。
        apply, reset = original_setup(
            task_eval_dir,
            arch,
            "GroupLogitKV",
            *setup_args,
            **setup_kwargs,
        )

        def wrapped(
            method,
            model,
            pkv,
            attn,
            hidden,
            ratio,
            sim_thr=None,
            *apply_args,
            **apply_kwargs,
        ):
            if variant == "eviction_only":
                target_kv_len = apply_kwargs.get("target_kv_len")
                return eviction_only(
                    model,
                    pkv,
                    attn,
                    ratio,
                    target_length=target_kv_len,
                )

            result = apply(
                "GroupLogitKV",
                model,
                pkv,
                attn,
                hidden,
                ratio,
                sim_thr,
                *apply_args,
                **apply_kwargs,
            )

            if variant == "group_merge_no_bias":
                zero_group_bias(model)

            return result

        return wrapped, reset

    lb.setup_kv_imports = patched_setup
    sys.argv = ["evaluate_longbench_kvmerge.py", *forwarded]

    if "--kv_method" not in forwarded:
        sys.argv += ["--kv_method", "GroupLogitKV"]

    lb.main()


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    if spec.strip().lower() == "auto":
        values = [0, n_layers // 3, (2 * n_layers) // 3, n_layers - 1]
    else:
        values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    values = sorted(set(x if x >= 0 else n_layers + x for x in values))
    if not values or values[0] < 0 or values[-1] >= n_layers:
        raise ValueError(f"Invalid --oracle_layers={spec!r} for {n_layers} layers")
    return values


def _build_groups(k: torch.Tensor, scores: torch.Tensor, ratio: float, sim_thr: float) -> List[List[List[int]]]:
    """Return groups[batch*kv_head][kept_slot] as original prompt indices."""
    bsz, h_kv, length, dim = k.shape
    num_merge = min(max(int(length * ratio), 0), length - 1)
    keep_len = length - num_merge
    x = bsz * h_kv
    kf = k.reshape(x, length, dim)
    sf = scores.reshape(x, length).float()
    merge_idx = torch.topk(-sf, k=num_merge, dim=-1).indices
    keep_mask = torch.ones((x, length), dtype=torch.bool, device=k.device)
    keep_mask.scatter_(1, merge_idx, False)
    keep_idx = keep_mask.nonzero(as_tuple=False)[:, 1].view(x, keep_len)
    kn = torch.gather(kf, 1, merge_idx.unsqueeze(-1).expand(-1, -1, dim))
    km = torch.gather(kf, 1, keep_idx.unsqueeze(-1).expand(-1, -1, dim))
    sim = torch.bmm(F.normalize(kn.float(), dim=-1), F.normalize(km.float(), dim=-1).transpose(1, 2))
    best_sim, best_idx = sim.max(dim=-1)
    accepted = torch.ones_like(best_sim, dtype=torch.bool) if sim_thr <= -1.0 else best_sim >= sim_thr

    groups: List[List[List[int]]] = []
    for row in range(x):
        row_groups = [[int(v)] for v in keep_idx[row].tolist()]
        for n, original_idx in enumerate(merge_idx[row].tolist()):
            if bool(accepted[row, n]):
                row_groups[int(best_idx[row, n])].append(int(original_idx))
        groups.append(row_groups)
    return groups


def _oracle_prompt(sample: Dict, prompt_format: str, tokenizer, model_name: str, input_length: int):
    import evaluate_longbench_kvmerge as lb
    text = lb.build_chat(tokenizer, prompt_format.format(**sample), model_name)
    ids = tokenizer(text, add_special_tokens=True, truncation=False, return_tensors="pt").input_ids
    if ids.shape[1] < input_length:
        return None
    return ids[:, :input_length].contiguous()


def run_oracle(args):
    import evaluate_longbench_kvmerge as lb
    lb.seed_everything(args.seed)
    task_eval_dir = Path(args.task_eval_dir or Path(__file__).resolve().parent).resolve()
    if str(task_eval_dir) not in sys.path:
        sys.path.insert(0, str(task_eval_dir))
    arch = args.arch if args.arch != "auto" else lb.infer_arch(args.model, args.model_path)
    model2path = json.load(open(task_eval_dir / "LongBench/config/model2path.json", "r"))
    prompt_map = json.load(open(task_eval_dir / "LongBench/config/dataset2prompt.json", "r"))
    model_path = args.model_path or model2path[args.model]
    data = lb.load_longbench_dataset(args.oracle_dataset, task_eval_dir)
    if args.oracle_dataset not in prompt_map:
        raise KeyError(f"No prompt format for {args.oracle_dataset}")

    # Installs only the existing attention-score capture/runtime patch; no compressor is called.
    _, reset_kv = lb.setup_kv_imports(task_eval_dir, arch, "GroupLogitKV")
    model, tokenizer = lb.load_model_and_tokenizer(
        model_path, args.model, args.device, lb.get_torch_dtype(args.dtype), args.attn_implementation
    )
    model.eval()
    layers = _parse_layers(args.oracle_layers, len(model.model.layers))
    qualified = []
    for idx, sample in enumerate(data):
        ids = _oracle_prompt(sample, prompt_map[args.oracle_dataset], tokenizer, args.model, args.oracle_input_length)
        if ids is not None:
            qualified.append((idx, ids))
            if len(qualified) >= args.oracle_num_prompts:
                break
    if len(qualified) < args.oracle_num_prompts:
        raise RuntimeError(f"Found only {len(qualified)} qualified prompts; need {args.oracle_num_prompts}")

    rng = random.Random(args.seed)
    records = []
    eps = 1e-30
    for sample_no, (dataset_idx, ids_cpu) in enumerate(qualified):
        ids = ids_cpu.to(args.device)
        mask = torch.ones_like(ids)
        lb._set_reduced_attention_capture(model, True, reduce=lb._attention_reduce_mode_for_method("GroupLogitKV"))
        try:
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=mask, use_cache=True,
                            output_attentions=False, return_dict=True)
            scores_all = lb._get_reduced_attention_scores(model)
        finally:
            lb._set_reduced_attention_capture(model, False, reduce=lb._attention_reduce_mode_for_method("GroupLogitKV"))
        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prompt_cache = _legacy(out.past_key_values)
        with torch.inference_mode():
            qout = model(
                input_ids=first,
                attention_mask=torch.ones((1, ids.shape[1] + 1), dtype=mask.dtype, device=args.device),
                position_ids=torch.tensor([[ids.shape[1]]], dtype=torch.long, device=args.device),
                past_key_values=out.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        for layer_idx in layers:
            k = prompt_cache[layer_idx][0]
            scores = _kv_scores(scores_all[layer_idx], k.shape[1])
            groups = _build_groups(k, scores, args.compress_ratio, args.sim_thr_grouplogitkv)
            attn = qout.attentions[layer_idx][..., :ids.shape[1]].float().clamp_min(eps)
            logp = attn.log()[0, :, 0, :]
            h_q = logp.shape[0]
            h_kv = k.shape[1]
            q_per_kv = h_q // h_kv
            candidates = []
            for qh in range(h_q):
                kvh = qh // q_per_kv
                row_groups = groups[kvh]
                for group in row_groups:
                    if len(group) <= 1:
                        continue
                    vals = logp[qh, torch.tensor(group, device=logp.device)]
                    centroid = vals.mean()
                    oracle = torch.logsumexp(vals, dim=0)
                    calibrated = centroid + math.log(len(group))
                    candidates.append((qh, kvh, len(group), float(abs(centroid-oracle)), float(abs(calibrated-oracle))))
            if len(candidates) > args.oracle_pairs_per_layer:
                candidates = rng.sample(candidates, args.oracle_pairs_per_layer)
            for qh, kvh, count, err0, err1 in candidates:
                records.append({"sample": sample_no, "dataset_index": dataset_idx, "layer": layer_idx,
                                "query_head": qh, "kv_head": kvh, "group_count": count,
                                "centroid_abs_error": err0, "calibrated_abs_error": err1})
        reset_kv(model)
        lb._clear_reduced_attention_scores(model)
        del out, qout, prompt_cache, scores_all

    if not records:
        raise RuntimeError("No merged groups were sampled. Lower the similarity threshold or increase compression ratio.")
    def mean(key): return sum(r[key] for r in records) / len(records)
    centroid_mae = mean("centroid_abs_error")
    calibrated_mae = mean("calibrated_abs_error")
    per_layer = []
    for layer in layers:
        rows = [r for r in records if r["layer"] == layer]
        if rows:
            per_layer.append({"layer": layer, "pairs": len(rows),
                              "centroid_only_mae": sum(r["centroid_abs_error"] for r in rows)/len(rows),
                              "count_calibrated_mae": sum(r["calibrated_abs_error"] for r in rows)/len(rows)})
    result = {
        "task": "oracle_group_logit", "model_name": args.model, "model_id": model_path,
        "arch": arch, "dataset": args.oracle_dataset, "input_length": args.oracle_input_length,
        "compress_ratio": args.compress_ratio, "retained_ratio": 1.0-args.compress_ratio,
        "sim_thr": args.sim_thr_grouplogitkv, "layers": layers,
        "num_prompts": args.oracle_num_prompts, "num_query_group_pairs": len(records),
        "centroid_only_mae": centroid_mae, "count_calibrated_mae": calibrated_mae,
        "relative_mae_reduction": (centroid_mae-calibrated_mae)/centroid_mae if centroid_mae else 0.0,
        "per_layer": per_layer,
        "definition": {
            "oracle": "logsumexp of original member logits",
            "centroid_only": "mean of original member logits",
            "count_calibrated": "mean of original member logits + log(group count)",
            "query": "first autoregressive decode query over the uncompressed prompt cache",
            "sampling": "merged groups only; deterministic prompt order and seeded pair subsampling"
        }
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _oracle_parser():
    p = argparse.ArgumentParser(description="GroupLogitKV oracle group-logit approximation ablation")
    p.add_argument("--task", choices=["oracle"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model_path")
    p.add_argument("--task_eval_dir")
    p.add_argument("--arch", choices=["llama", "qwen", "auto"], default="auto")
    p.add_argument("--compress_ratio", type=float, default=0.6)
    p.add_argument("--sim_thr_grouplogitkv", type=float, default=0.8)
    p.add_argument("--oracle_dataset", default="gov_report")
    p.add_argument("--oracle_input_length", type=int, default=8192)
    p.add_argument("--oracle_num_prompts", type=int, default=8)
    p.add_argument("--oracle_layers", default="auto")
    p.add_argument("--oracle_pairs_per_layer", type=int, default=512)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", required=True)
    return p


def main():
    # Parse only the wrapper selectors first, preserving every delegated argument
    # exactly as in the original component-ablation evaluator.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=["ppl", "longbench", "oracle"], required=True)
    selector.add_argument("--variant", choices=["eviction_only", "group_merge_no_bias", "full"])
    selected, forwarded = selector.parse_known_args()
    if selected.task == "oracle":
        run_oracle(_oracle_parser().parse_args())
        return
    if selected.variant is None:
        selector.error("--task ppl/longbench requires --variant")
    if selected.task == "ppl":
        run_ppl(selected.variant, forwarded)
    else:
        run_longbench(selected.variant, forwarded)


if __name__ == "__main__":
    main()
