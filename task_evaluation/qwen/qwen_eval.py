import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import time
import traceback

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_simcalkv_qwen import Repair
from merge_grouplogitkv_qwen import Repair as Repair_GroupLogitKV, disable_grouplogitkv_sdpa
from utils import generate_token_with_past, generate_text


def cuda_sync(device="cuda"):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def reset_grouplogitkv_sdpa_state(model):
    """
    Clear GroupLogitKV-SDPA state.

    The Qwen runtime attention patch is conditional. It is active only when each attention
    module has `_grouplogitkv_sdpa_enabled=True`. This reset keeps the
    uncompressed "before" run and all non-GroupLogitKV methods clean.
    """
    disable_grouplogitkv_sdpa(model)


def _as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def get_kv_len(past_key_values):
    past_key_values = _as_legacy_tuple(past_key_values)
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    return int(past_key_values[0][0].shape[-2])


def measure_kv_memory(past_key_values):
    past_key_values = _as_legacy_tuple(past_key_values)

    total_bytes = 0
    for layer_cache in past_key_values:
        k = layer_cache[0]
        v = layer_cache[1]
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 ** 2)


def apply_kv_method(method, model, past_key_values, attention, hidden_states, compress_ratio):
    if method == "SimCalKV":
        # Qwen SimCalKV's Repair signature is:
        #   Repair(model, past_key_values, attention, hidden_states, ratio, rope_fn=None)
        # Do not pass the LLaMA trailing q_sign=0 here.
        return Repair(model, past_key_values, attention, hidden_states, compress_ratio)

    if method == "GroupLogitKV":
        return Repair_GroupLogitKV(model, past_key_values, attention, hidden_states, compress_ratio)

    if method == "KeepKV":
        from merge_keepkv_qwen import Repair_keepkv
        return Repair_keepkv(model, past_key_values, attention, hidden_states, compress_ratio)

    if method == "PyramidInfer":
        from pyramid_qwen import Decode
        return Decode(model, past_key_values, attention, hidden_states, compress_ratio)

    raise ValueError(f"Unknown KV method: {method}")


def evaluate_example(model, tokenizer, prompt, reference, method="SimCalKV", compress_ratio=0.1, device="cuda"):
    reset_grouplogitkv_sdpa_state(model)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    memory_before_list, time_before_list, throughput_before_list = [], [], []
    memory_after_list, time_after_list, throughput_after_list = [], [], []
    decode_time_before_list, decode_time_after_list = [], []
    merge_time_list = []
    prefill_kv_len_list, compressed_kv_len_list = [], []
    final_kv_len_before_list, final_kv_len_after_list = [], []

    answer_before, answer_after = "", ""

    for i in range(1):
        print(f"  > Performing inference {i + 1}/1...")

        with torch.no_grad():
            cuda_sync(device)
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            cuda_sync(device)
            past_key_values = outputs.past_key_values
            attention = outputs.attentions
            hidden_states = outputs.hidden_states

        prefill_kv_len_list.append(get_kv_len(past_key_values))

        # Uncompressed baseline generation.
        cuda_sync(device)
        start_time_before = time.time()

        next_token_id, past_key_values = generate_token_with_past(model, inputs)
        total_kv, _, generated_tokens, durations_cached_s, _, _ = generate_text(
            model,
            tokenizer,
            256,
            {
                "input_ids": next_token_id.reshape((1, 1)),
                "attention_mask": torch.cat(
                    [inputs["attention_mask"], torch.tensor([[1]], device=device)],
                    dim=1,
                ),
                "past_key_values": past_key_values,
            },
            device=device,
        )

        cuda_sync(device)
        total_before = time.time() - start_time_before

        memory_before = measure_kv_memory(total_kv)
        n_tokens_before = len(generated_tokens)
        total_generation_time_before = sum(durations_cached_s)
        throughput_before = n_tokens_before / total_generation_time_before if total_generation_time_before > 0 else 0.0

        memory_before_list.append(memory_before)
        time_before_list.append(total_before)
        throughput_before_list.append(throughput_before)
        decode_time_before_list.append(total_generation_time_before)
        final_kv_len_before_list.append(get_kv_len(total_kv))

        if i == 0:
            answer_before = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Compressed generation.
        cuda_sync(device)
        start_time_after = time.time()

        cuda_sync(device)
        merge_start = time.time()
        past_key_values_merged = apply_kv_method(
            method,
            model,
            past_key_values,
            attention,
            hidden_states,
            compress_ratio,
        )
        cuda_sync(device)
        merge_time = time.time() - merge_start

        compressed_kv_len_list.append(get_kv_len(past_key_values_merged))

        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)],
                dim=1,
            ),
            "past_key_values": past_key_values_merged,
        }
        total_kv_merged_1, attention_mask1, generated_tokens_merged_1, durations_cached_s_merged_1, attention_merged, hidden_states_merged = generate_text(
            model,
            tokenizer,
            128,
            next_inputs,
            device=device,
        )

        # Original Qwen schedule: generate 128 tokens, compress again,
        # generate another 128 tokens, then final compress for memory statistics.
        cuda_sync(device)
        remerge_start = time.time()
        kv_decode_merged = apply_kv_method(
            method,
            model,
            total_kv_merged_1,
            attention_merged,
            hidden_states_merged,
            compress_ratio,
        )
        cuda_sync(device)
        remerge_time = time.time() - remerge_start

        next_inputs = {
            "input_ids": generated_tokens_merged_1[-1].reshape((1, 1)),
            "attention_mask": torch.cat(
                [attention_mask1, torch.tensor([[1]], device=device)],
                dim=1,
            ),
            "past_key_values": kv_decode_merged,
        }
        total_kv_merged, _, generated_tokens_merged, durations_cached_s_merged, attention_merged, hidden_states_merged = generate_text(
            model,
            tokenizer,
            128,
            next_inputs,
            device=device,
        )

        cuda_sync(device)
        remerge2_start = time.time()
        kv_decode_merged = apply_kv_method(
            method,
            model,
            total_kv_merged,
            attention_merged,
            hidden_states_merged,
            compress_ratio,
        )
        cuda_sync(device)
        remerge2_time = time.time() - remerge2_start

        cuda_sync(device)
        total_after = time.time() - start_time_after

        memory_after = measure_kv_memory(kv_decode_merged)
        n_tokens_after = len(generated_tokens_merged_1) + len(generated_tokens_merged)
        total_generation_time_after = sum(durations_cached_s_merged_1) + sum(durations_cached_s_merged)
        throughput_after = n_tokens_after / total_generation_time_after if total_generation_time_after > 0 else 0.0

        memory_after_list.append(memory_after)
        time_after_list.append(total_after)
        throughput_after_list.append(throughput_after)
        merge_time_list.append(merge_time + remerge_time + remerge2_time)
        decode_time_after_list.append(total_generation_time_after)
        final_kv_len_after_list.append(get_kv_len(kv_decode_merged))

        if i == 0:
            all_generated_tokens = generated_tokens_merged_1 + generated_tokens_merged
            answer_after = tokenizer.decode(
                all_generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

    avg_decode_time_before = safe_mean(decode_time_before_list)
    avg_decode_time_after = safe_mean(decode_time_after_list)

    reset_grouplogitkv_sdpa_state(model)

    return {
        "memory_before": safe_mean(memory_before_list),
        "memory_after": safe_mean(memory_after_list),
        "time_before": safe_mean(time_before_list),
        "time_after": safe_mean(time_after_list),
        "throughput_before": safe_mean(throughput_before_list),
        "throughput_after": safe_mean(throughput_after_list),
        "decode_time_before": avg_decode_time_before,
        "decode_time_after": avg_decode_time_after,
        "decode_time_saving": avg_decode_time_before - avg_decode_time_after,
        "merge_time_s": safe_mean(merge_time_list),
        "prefill_kv_len": safe_mean(prefill_kv_len_list),
        "compressed_kv_len": safe_mean(compressed_kv_len_list),
        "final_kv_len_before": safe_mean(final_kv_len_before_list),
        "final_kv_len_after": safe_mean(final_kv_len_after_list),
        "answer_before": answer_before,
        "answer_after": answer_after,
        "reference": reference,
        "compression_ratio": compress_ratio,
    }


def load_task_dataset(name, split="validation[:10]"):
    try:
        if name == "cnn_dailymail":
            ds = load_dataset(name, "3.0.0", split=split)
        elif "THUDM/LongBench" in name:
            subtask = name.split("/")[-1]
            ds = load_dataset("THUDM/LongBench", subtask, split=split, trust_remote_code=True)
        elif name == "piqa":
            ds = load_dataset("piqa", split="validation[:1000]", trust_remote_code=True)
        elif name == "winogrande":
            ds = load_dataset("winogrande", "winogrande_debiased", split=split, trust_remote_code=True)
        else:
            # Newer datasets versions reject trust_remote_code for standard datasets such as xsum.
            ds = load_dataset(name, split=split)
    except Exception as e:
        raise ValueError(f"Fail to load dataset {name}: {str(e)}")

    samples = []

    if name == "xsum":
        for ex in ds:
            prompt = (
                "Please summarize the following article in one short sentence.\n\n"
                f"Article:\n{ex['document']}\n\nSummary:"
            )
            samples.append({"dataset": "xsum", "prompt": prompt, "reference": ex["summary"]})

    elif name == "cnn_dailymail":
        for ex in ds:
            prompt = (
                "Please summarize the following news article into 3-5 concise sentences.\n\n"
                f"Article:\n{ex['article']}\n\nSummary:"
            )
            samples.append({"dataset": "cnn_dailymail", "prompt": prompt, "reference": ex["highlights"]})

    elif name == "THUDM/LongBench/gov_report":
        for ex in ds:
            prompt = (
                "Summarize the following government report into a concise abstract.\n\n"
                f"Report:\n{ex['input']}\n\nSummary:"
            )
            ref = ex["answers"][0] if isinstance(ex["answers"], list) else ex["answers"]
            samples.append({"dataset": "govreport", "prompt": prompt, "reference": ref})

    elif name == "piqa":
        for ex in ds:
            prompt = f"Question:\n{ex['goal']}\n\nAnswer with 1 or 2:"
            reference = ex[f"sol{ex['label'] + 1}"]
            samples.append({"dataset": "piqa", "prompt": prompt, "reference": reference})

    elif name == "openbookqa":
        for ex in ds:
            choices_text = "\n".join(
                f"{label}. {text}"
                for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])
            )
            prompt = (
                f"Question:\n{ex['question_stem']}\n\n"
                f"Choices:\n{choices_text}\n\n"
                "Answer with only the letter (A, B, C, D):"
            )
            samples.append({"dataset": "openbookqa", "prompt": prompt, "reference": ex["answerKey"]})

    elif name == "winogrande":
        for ex in ds:
            prompt = (
                f"Sentence with blank:\n{ex['sentence']}\n\n"
                "Fill in the blank with the correct option.\n\n"
                f"Option1:{ex['option1']}\n"
                f"Option2:{ex['option2']}\n"
                "Answer:"
            )
            samples.append({"dataset": "winogrande", "prompt": prompt, "reference": int(ex["answer"].lower()) - int("a")})

    elif name == "THUDM/LongBench/2wikimqa":
        for ex in ds:
            prompt = (
                "Read the following passages and answer the multi-hop question.\n\n"
                f"Passages:\n{ex['context']}\n\n"
                f"Question:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "2wikimqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/narrativeqa":
        for ex in ds:
            prompt = (
                "Read the following story and answer the question.\n\n"
                f"Story:\n{ex['context']}\n\n"
                f"Question:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "narrativeqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/hotpotqa":
        for ex in ds:
            prompt = (
                "Read the following passages and answer the question.\n\n"
                f"Passages:\n{ex['context']}\n\n"
                f"Question:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "hotpotqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/multifieldqa_zh":
        for ex in ds:
            prompt = (
                "Read the document and answer the question。\n\n"
                f"Document:\n{ex['context']}\n\n"
                f"Question:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "multifieldqa_zh", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/lcc":
        for ex in ds:
            prompt = (
                f"Passages:\n{ex['context']}\n\n"
                f"Write a {ex['language']} function to solve the following problem:\n\n"
                f"{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "lcc", "prompt": prompt, "reference": ex["answers"]})

    elif name == "THUDM/LongBench/repobench-p":
        for ex in ds:
            prompt = (
                f"Passages:\n{ex['context']}\n\n"
                f"Write a {ex['language']} function to solve the following problem:\n\n"
                f"{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "repobench-p", "prompt": prompt, "reference": ex["answers"]})

    elif name == "THUDM/LongBench/trec":
        for ex in ds:
            prompt = f"Question:\n\n{ex['input']}\n\nChoices:\n{ex['context']}\n\nAnswer:"
            samples.append({"dataset": "trec", "prompt": prompt, "reference": ex["answers"][0]})

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="xsum", help="Dataset names, e.g. xsum piqa openbookqa winogrande")
    parser.add_argument("--split", type=str, default="test[:5]")
    parser.add_argument("--compress_ratio", type=float, default=0.4)
    parser.add_argument(
        "--kv_method",
        type=str,
        default="SimCalKV",
        choices=["SimCalKV", "GroupLogitKV", "KeepKV", "PyramidInfer"],
    )
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    print(f"Results will be saved to: {args.output_dir}")

    DATASET_METRICS = {
        "xsum": ["rouge"],
        "cnn_dailymail": ["rouge"],
        "piqa": ["accuracy"],
        "openbookqa": ["accuracy"],
        "winogrande_debiased": ["accuracy"],
        "2wikimqa": ["f1"],
        "narrativeqa": ["f1"],
        "gov_report": ["rouge"],
        "hotpotqa": ["f1"],
        "trec": ["accuracy"],
        "multifieldqa_zh": ["f1"],
        "repobench-p": ["bleu"],
        "lcc": ["bleu"],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # >>> GROUPLOGITKV_QWEN_RUNTIME_PATCH_V1
    # Runtime patch only changes Qwen2Attention.forward in memory.
    # It does not modify site-packages/transformers files.
    if args.kv_method == "GroupLogitKV":
        from grouplogitkv_runtime_patch_qwen import apply_grouplogitkv_qwen_runtime_patch
        apply_grouplogitkv_qwen_runtime_patch()
    # <<< GROUPLOGITKV_QWEN_RUNTIME_PATCH_V1

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(">>> Model loading Completed")

    reset_grouplogitkv_sdpa_state(model)

    for dataset in [args.dataset]:
        print(f"\n>>> Processing dataset: {dataset}")
        try:
            samples = load_task_dataset(dataset, args.split)

            dataset_name = dataset.split("/")[-1] if "/" in dataset else dataset
            metrics_to_use = DATASET_METRICS.get(dataset_name, [])
            metric_evaluators = {m: evaluate.load(m) for m in metrics_to_use}
            results = []
            preds_before, preds_after, refs = [], [], []

            for i, example in enumerate(samples):
                metrics = evaluate_example(
                    model,
                    tokenizer,
                    example["prompt"],
                    example["reference"],
                    args.kv_method,
                    args.compress_ratio,
                    device,
                )
                metrics.update({
                    "id": i,
                    "model": args.model_name,
                    "dataset": dataset,
                    "kv_method": args.kv_method,
                })
                results.append(metrics)
                preds_before.append(metrics["answer_before"])
                preds_after.append(metrics["answer_after"])
                refs.append(example["reference"])

            score_before, score_after = {}, {}
            for metric_name, evaluator in metric_evaluators.items():
                if metric_name == "rouge":
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)
                elif metric_name in ["accuracy", "f1", "bleu"]:
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)[metric_name]
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)[metric_name]

            output_csv = f"{args.output_dir}/qwen_results_{dataset_name}_{args.kv_method}.csv"
            df = pd.DataFrame([{
                "id": r["id"],
                "dataset": r["dataset"],
                "kv_method": r["kv_method"],
                "memory_before_MB": r["memory_before"],
                "memory_after_MB": r["memory_after"],
                "time_before_s": r["time_before"],
                "time_after_s": r["time_after"],
                "throughput_before_tokens_s": r["throughput_before"],
                "throughput_after_tokens_s": r["throughput_after"],
                "decode_time_before_s": r["decode_time_before"],
                "decode_time_after_s": r["decode_time_after"],
                "decode_time_saving_s": r["decode_time_saving"],
                "merge_time_s": r["merge_time_s"],
                "prefill_kv_len": r["prefill_kv_len"],
                "compressed_kv_len": r["compressed_kv_len"],
                "final_kv_len_before": r["final_kv_len_before"],
                "final_kv_len_after": r["final_kv_len_after"],
                "answer_before": r["answer_before"],
                "answer_after": r["answer_after"],
            } for r in results])
            df.to_csv(output_csv, index=False)
            print(f"The results are saved in {output_csv}")

            summary_json = f"{args.output_dir}/qwen_{dataset_name}_{args.kv_method}_{args.compress_ratio}.json"
            summary = {
                "dataset": dataset,
                "kv_method": args.kv_method,
                "model_name": args.model_name,
                "compress_ratio": args.compress_ratio,
                "metrics_before": score_before,
                "metrics_after": score_after,
                "avg_memory_saving_MB": df["memory_before_MB"].mean() - df["memory_after_MB"].mean(),
                "time_before": df["time_before_s"].sum(),
                "time_after": df["time_after_s"].sum(),
                "avg_time_saving_s": df["time_before_s"].sum() - df["time_after_s"].sum(),
                "throughput_before": df["throughput_before_tokens_s"].mean(),
                "throughput_after": df["throughput_after_tokens_s"].mean(),
                "avg_throughput_improvement_tokens_s": df["throughput_after_tokens_s"].mean() - df["throughput_before_tokens_s"].mean(),
                "decode_time_before": df["decode_time_before_s"].sum(),
                "decode_time_after": df["decode_time_after_s"].sum(),
                "avg_decode_time_saving_s": df["decode_time_before_s"].sum() - df["decode_time_after_s"].sum(),
                "avg_merge_time_s": df["merge_time_s"].mean(),
                "merge_time_s": df["merge_time_s"].sum(),
                "avg_prefill_kv_len": df["prefill_kv_len"].mean(),
                "avg_compressed_kv_len": df["compressed_kv_len"].mean(),
                "avg_final_kv_len_before": df["final_kv_len_before"].mean(),
                "avg_final_kv_len_after": df["final_kv_len_after"].mean(),
            }
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"The complete results are saved in {summary_json}")

        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
