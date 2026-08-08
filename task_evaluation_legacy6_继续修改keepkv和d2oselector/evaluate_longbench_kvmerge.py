import os
import sys
import json
import time
import random
import argparse
import gc
import zipfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import transformers

from LongBench.eval import collect_all_results

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    CausalLMOutputWithPast as Qwen2CausalLMOutputWithPast,
)

DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "passage_retrieval_en",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec",
    "triviaqa", "samsum", "passage_count", "hotpotqa",
]

MODELS = [
    # Keep this list aligned with the original LongBench/AquaKV entry script.
    "llama2-7b-chat-4k",
    "llama2-13b-chat-4k",
    "longchat-v1.5-7b-32k",
    "xgen-7b-8k",
    "internlm-7b-8k",
    "chatglm2-6b",
    "chatglm2-6b-32k",
    "chatglm3-6b-32k",
    "vicuna-v1.5-7b-16k",
    "llama-3.2-1B",
    "llama-3.2-3B-test",
    "llama-3.2-3B",
    "llama-3.1-8B",
    "llama-3.1-70B",
    "llama-3.2-1B-Instruct",
    "llama-3.2-3B-Instruct",
    "llama-3.1-8B-Instruct",
    "llama-3.1-70B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
]

METHODS = ["GroupLogitKV", "mymethod", "SimCalKV", "KeepKV", "CaM", "CentroidKV", "None", "none"]


def parse_args(args=None):
    p = argparse.ArgumentParser(description="Run LongBench v1 with post-hoc KV merge/compression methods from task_evaluation.")
    p.add_argument("--model", type=str, default=None, choices=MODELS, help="Key in LongBench/config/model2path.json")
    p.add_argument("--model_path", type=str, default=None, help="Optional explicit HF path/local path; overrides model2path.json")
    p.add_argument("--task_eval_dir", type=str, default=None, help="Path to task_evaluation directory. Defaults to this script directory.")
    p.add_argument("--arch", type=str, choices=["llama", "qwen", "auto"], default="auto")
    p.add_argument("--kv_method", type=str, choices=METHODS, required=True)
    p.add_argument("--compress_ratio", type=float, default=0.4, help="Evicted fraction. 0.4 means evict about 40%, keep about 60% physical KV slots.")
    p.add_argument("--sim_thr_grouplogitkv", type=float, default=-1.0, help="Similarity threshold for GroupLogitKV; -1 disables threshold rejection.")
    p.add_argument("--lambda_mymethod", type=float, default=0.5, help="mymethod normalized Euclidean complete-diameter threshold lambda (>0).")
    p.add_argument("--mymethod_group_mode", choices=["full", "anchor_only", "eviction_only"], default="full")
    p.add_argument("--mymethod_selector", choices=["keepkv", "attention", "d2o"], default="keepkv")
    p.add_argument("--mymethod_graph_backend", choices=["auto", "cuda", "torch"], default="auto")
    p.add_argument("--mymethod_route_backend", choices=["auto", "cuda", "torch"], default="torch")
    p.add_argument("--mymethod_keepkv_sink_size", type=int, default=4)
    p.add_argument("--mymethod_keepkv_recent_protect_size", type=int, default=128)
    p.add_argument("--mymethod_keepkv_score_window", type=int, default=128)
    p.add_argument("--mymethod_keepkv_recency_beta", type=float, default=1.0)
    p.add_argument("--mymethod_keepkv_sink_can_absorb", type=int, choices=[0, 1], default=1)
    p.add_argument("--mymethod_d2o_sink_size", type=int, default=4)
    p.add_argument("--mymethod_d2o_recent_size", type=int, default=128)
    p.add_argument("--mymethod_d2o_sink_can_absorb", type=int, choices=[0, 1], default=1)
    p.add_argument("--mymethod_d2o_use_log_count", type=int, choices=[0, 1], default=1)
    p.add_argument("--mymethod_distance_chunk_size", type=int, default=256)
    p.add_argument("--mymethod_attention_chunk_size", type=int, default=128)
    p.add_argument("--centroidkv_sink_budget", type=int, default=16)
    p.add_argument("--centroidkv_local_budget", type=int, default=64)
    p.add_argument("--centroidkv_chunk_size", type=int, default=256)
    p.add_argument("--centroidkv_merge_ratio", type=float, default=0.8)
    p.add_argument("--centroidkv_decrease_rate", type=float, default=0.2)
    p.add_argument("--sim_thr_simcalkv", type=float, default=-1.0, help="Similarity threshold for SimCalKV; -1 disables threshold rejection.")
    p.add_argument("--sim_thr_keepkv", type=float, default=-1.0, help="Similarity threshold for KeepKV; -1 disables threshold rejection.")
    p.add_argument("--out_path", type=str, default="./pred")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated LongBench v1 datasets. If omitted with --all, runs all supported English v1 tasks.")
    p.add_argument("-a", "--all", action="store_true", help="Run all LongBench v1 English tasks used by this package.")
    p.add_argument("-e", dest="e", action="store_true", help="Evaluate on LongBench-E split.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "float16", "bf16", "bfloat16", "auto"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--attn_implementation", type=str, default="eager", help="Use eager for KV-merge methods that need attention logits/weights.")
    p.add_argument("--prefill_chunk_size", type=int, default=128, help="Prompt forward chunk size for memory-efficient KeepKV one-shot prefill.")
    p.add_argument("--keepkv_similarity_chunk_size", type=int, default=256, help="Discard-token block size used by memory-bounded KeepKV cosine matching.")
    p.add_argument("--simcalkv_similarity_chunk_size", type=int, default=256, help="Evicted-token block size used by memory-bounded SimCalKV cosine matching.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset jsonl files.")
    p.add_argument("--cleanup_interval", type=int, default=0, help="Run Python GC/CUDA cache cleanup every N samples; 0 disables forced cleanup (faster and usually lower fragmentation).")
    return p.parse_args(args)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_torch_dtype(name):
    if name in ["fp16", "float16"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name == "auto":
        return "auto"
    raise ValueError(name)


def normalize_kv_method(method):
    if method is None:
        return None
    if str(method).lower() in ["none", "full", "fullkv", "nocompress", "no_compress", "baseline"]:
        return "None"
    return method


def infer_arch(model_name, model_path):
    s = f"{model_name} {model_path or ''}".lower()
    if "qwen" in s:
        return "qwen"
    if "llama" in s or "meta-llama" in s:
        return "llama"
    raise ValueError("Cannot infer --arch from model name/path. Pass --arch llama or --arch qwen.")


def setup_kv_imports(
    task_eval_dir,
    arch,
    kv_method,
    simcalkv_similarity_chunk_size: int = 256,
    mymethod_lambda: float = 0.5,
    mymethod_selector: str = "keepkv",
    mymethod_keepkv_sink_size: int = 4,
    mymethod_keepkv_recent_protect_size: int = 128,
    mymethod_keepkv_score_window: int = 128,
    mymethod_keepkv_recency_beta: float = 1.0,
    mymethod_keepkv_sink_can_absorb: bool = True,
    mymethod_d2o_sink_size: int = 4,
    mymethod_d2o_recent_size: int = 128,
    mymethod_d2o_sink_can_absorb: bool = True,
    mymethod_d2o_use_log_count: bool = True,
    mymethod_distance_chunk_size: int = 256,
    mymethod_group_mode: str = "full",
    mymethod_graph_backend: str = "auto",
    mymethod_route_backend: str = "auto",
    centroidkv_sink_budget: int = 16,
    centroidkv_local_budget: int = 64,
    centroidkv_chunk_size: int = 256,
    centroidkv_merge_ratio: float = 0.8,
    centroidkv_decrease_rate: float = 0.2,
):
    """Load the current KV-compression entry points and runtime patches.

    The method implementations live under ``models/<arch>`` and now expose
    function-based entry points. KeepKV and mymethod have dedicated decode-time
    runtime patches; the remaining legacy compressed-cache methods retain the
    existing GroupLogitKV-compatible eager-attention path.
    """
    kv_method = normalize_kv_method(kv_method)
    if kv_method == "None":
        def reset(model):
            return None

        def apply(method, model, pkv, attn, hidden, ratio, sim_thr):
            return pkv

        return apply, reset

    root = Path(task_eval_dir).resolve()
    method_dir = root / "models" / arch
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing method directory: {method_dir}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(method_dir) not in sys.path:
        sys.path.insert(0, str(method_dir))

    if arch == "llama":
        from merge_simcalkv_llama import Repair as SimCalRepair
        from merge_grouplogitkv_llama import (
            Repair as GroupRepair,
            disable_grouplogitkv_eager,
            disable_grouplogitkv_sdpa,
        )
        from merge_mymethod_llama import (
            Repair as MyMethodRepair,
            disable_mymethod_eager,
            disable_mymethod_sdpa,
        )
        from merge_keepkv_llama import (
            Repair_keepkv as KeepRepair, reset_keepkv_state,
            prepare_longbench_keepkv, set_longbench_keepkv_chunk, finalize_longbench_keepkv,
        )
        from merge_cam_llama import Repair as CaMRepair, reset_cam_state
        from merge_centroidkv_llama import Repair as CentroidRepair, reset_centroidkv_state

        if kv_method == "KeepKV":
            from keepkv_runtime_patch_llama import apply_keepkv_runtime_patch
            apply_keepkv_runtime_patch()
        elif kv_method == "mymethod":
            from mymethod_runtime_patch_llama import apply_mymethod_runtime_patch
            apply_mymethod_runtime_patch()
        else:
            from grouplogitkv_runtime_patch_llama import apply_grouplogitkv_runtime_patch
            apply_grouplogitkv_runtime_patch()

    elif arch == "qwen":
        from merge_simcalkv_qwen import Repair as SimCalRepair
        from merge_grouplogitkv_qwen import (
            Repair as GroupRepair,
            disable_grouplogitkv_eager,
            disable_grouplogitkv_sdpa,
        )
        from merge_mymethod_qwen import (
            Repair as MyMethodRepair,
            disable_mymethod_eager,
            disable_mymethod_sdpa,
        )
        from merge_keepkv_qwen import (
            Repair_keepkv as KeepRepair, reset_keepkv_state,
            prepare_longbench_keepkv, set_longbench_keepkv_chunk, finalize_longbench_keepkv,
        )
        from merge_cam_qwen import Repair as CaMRepair, reset_cam_state
        from merge_centroidkv_qwen import Repair as CentroidRepair, reset_centroidkv_state

        if kv_method == "KeepKV":
            from keepkv_runtime_patch_qwen import apply_keepkv_qwen_runtime_patch
            apply_keepkv_qwen_runtime_patch()
        elif kv_method == "mymethod":
            from mymethod_runtime_patch_qwen import apply_mymethod_qwen_runtime_patch
            apply_mymethod_qwen_runtime_patch()
        else:
            from grouplogitkv_runtime_patch_qwen import apply_grouplogitkv_qwen_runtime_patch
            apply_grouplogitkv_qwen_runtime_patch()
    else:
        raise ValueError(arch)

    def reset(model):
        """Clear sequence-local metadata before each independent prompt."""
        if kv_method == "KeepKV":
            reset_keepkv_state(model)
        elif kv_method == "CaM":
            reset_cam_state(model)
        elif kv_method == "mymethod":
            disable_mymethod_sdpa(model)
            disable_mymethod_eager(model)
        elif kv_method == "CentroidKV":
            reset_centroidkv_state(model)
        else:
            disable_grouplogitkv_sdpa(model)
            disable_grouplogitkv_eager(model)

    def apply(method, model, pkv, attn, hidden, ratio, sim_thr):
        legacy_pkv = _as_legacy_tuple(pkv)
        if method == "SimCalKV":
            return SimCalRepair(
                model,
                legacy_pkv,
                attn,
                hidden,
                ratio,
                sim_thr,
                similarity_chunk_size=simcalkv_similarity_chunk_size,
            )
        if method == "GroupLogitKV":
            return GroupRepair(model, legacy_pkv, attn, hidden, ratio, sim_thr)
        if method == "mymethod":
            return MyMethodRepair(
                model,
                legacy_pkv,
                attn,
                hidden,
                ratio,
                mymethod_lambda,
                keepkv_sink_size=mymethod_keepkv_sink_size,
                keepkv_recent_protect_size=mymethod_keepkv_recent_protect_size,
                keepkv_score_window=mymethod_keepkv_score_window,
                keepkv_recency_beta=mymethod_keepkv_recency_beta,
                keepkv_sink_can_absorb=bool(mymethod_keepkv_sink_can_absorb),
                d2o_sink_size=mymethod_d2o_sink_size,
                d2o_recent_size=mymethod_d2o_recent_size,
                d2o_sink_can_absorb=bool(mymethod_d2o_sink_can_absorb),
                d2o_use_log_count=bool(mymethod_d2o_use_log_count),
                distance_chunk_size=mymethod_distance_chunk_size,
                selector=str(mymethod_selector),
                group_mode=str(mymethod_group_mode),
                graph_backend=str(mymethod_graph_backend),
                route_backend=str(mymethod_route_backend),
            )
        if method == "CentroidKV":
            return CentroidRepair(
                model, legacy_pkv, attn, hidden, ratio,
                sink_budget=centroidkv_sink_budget,
                local_budget=centroidkv_local_budget,
                chunk_size=centroidkv_chunk_size,
                merge_ratio=centroidkv_merge_ratio,
                decrease_rate=centroidkv_decrease_rate,
            )
        if method == "KeepKV":
            # A None threshold preserves KeepKVConfig's official separate
            # prefill/decode defaults; a numeric CLI value overrides both.
            keepkv_thr = None if sim_thr is None else float(sim_thr)
            return KeepRepair(model, legacy_pkv, attn, hidden, ratio, keepkv_thr)
        if method == "CaM":
            # The runtime patch returns query-reduced scores [B,Hkv,K].
            # CaM only sums over Q before reducing query heads, so a singleton
            # query axis is an exact one-shot adapter up to a global scale that
            # cancels in its merge-probability ratio.
            if attn is not None:
                attn = tuple(x.unsqueeze(-2) if x.dim() == 3 else x for x in attn)
            return CaMRepair(model, legacy_pkv, attn, hidden, ratio)
        raise ValueError(method)

    if kv_method == "KeepKV":
        apply.keepkv_prepare_longbench = prepare_longbench_keepkv
        apply.keepkv_set_chunk = set_longbench_keepkv_chunk
        apply.keepkv_finalize_longbench = finalize_longbench_keepkv

    return apply, reset

def _as_legacy_tuple(pkv):
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    if isinstance(pkv, list):
        return tuple(pkv)
    return pkv


def build_chat(tokenizer, prompt, model_name):
    # Same prompt-wrapping policy as evaluate_longbench_aquakv.py.
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif model_name.lower().endswith("instruct"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        bos = tokenizer.bos_token
        if bos is not None and prompt.startswith(bos):
            prompt = prompt[len(bos):]
    return prompt


def post_process(response, model_name):
    # Same post-processing policy as evaluate_longbench_aquakv.py.
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def _sync(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def manual_generate_after_compress(model, tokenizer, inputs, compressed_pkv, first_token_id, max_new_tokens, device, stop_token_ids=None):
    """Greedy decode from a compressed physical cache using logical positions.

    The cache sequence length is smaller than the original prompt length.  HF
    otherwise derives RoPE positions from the *physical* cache length, which is
    wrong after KV compression and causes severe repetition.  Keep the compact
    physical attention mask, but pass the original logical position explicitly.
    """
    generated = [first_token_id.reshape(())]
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    if stop_token_ids:
        stop_ids.update(int(x) for x in stop_token_ids if x is not None)

    if int(first_token_id) in stop_ids:
        return generated

    logical_prompt_len = int(inputs["input_ids"].shape[1])
    legacy = _as_legacy_tuple(compressed_pkv)
    physical_prefix_len = int(legacy[0][0].shape[-2])
    mask_dtype = inputs["attention_mask"].dtype

    past = compressed_pkv
    # Reuse decode metadata buffers instead of allocating new CUDA tensors for
    # every generated token. Slicing is a view and preserves identical values.
    max_physical_len = physical_prefix_len + max(0, max_new_tokens - 1)
    decode_mask = torch.ones((1, max_physical_len), device=device, dtype=mask_dtype)
    decode_positions = torch.arange(
        logical_prompt_len,
        logical_prompt_len + max(0, max_new_tokens - 1),
        device=device,
        dtype=torch.long,
    ).view(1, -1)
    for step in range(max_new_tokens - 1):
        token = generated[-1].view(1, 1)
        physical_len = physical_prefix_len + step + 1
        out = model(
            input_ids=token,
            attention_mask=decode_mask[:, :physical_len],
            position_ids=decode_positions[:, step:step + 1],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        tok = out.logits[:, -1, :].argmax(dim=-1)
        generated.append(tok.reshape(()))
        past = out.past_key_values
        del out
        if tok.item() in stop_ids:
            break
    return generated


def _chunked_prefill_and_compress(
    model,
    inputs,
    method,
    apply_kv,
    ratio,
    sim_thr,
    device,
    chunk_size=64,
    similarity_chunk_size=256,
):
    """Memory-efficient one-shot KeepKV prompt prefill.

    The prompt is forwarded in small chunks, but no physical compression or
    score-history tensor is materialized per chunk.  The KeepKV runtime patch
    accumulates only two [B,Hkv,K] sufficient statistics.  The complete prompt
    cache is compressed exactly once after the final chunk.
    """
    if method != "KeepKV":
        raise ValueError("This optimized path is KeepKV-only")
    prepare = getattr(apply_kv, "keepkv_prepare_longbench", None)
    set_chunk = getattr(apply_kv, "keepkv_set_chunk", None)
    finalize = getattr(apply_kv, "keepkv_finalize_longbench", None)
    if not callable(prepare) or not callable(set_chunk) or not callable(finalize):
        raise RuntimeError("Installed KeepKV adapter lacks LongBench one-shot APIs")

    input_ids = inputs["input_ids"]
    total_len = int(input_ids.shape[1])
    chunk_size = max(1, int(chunk_size))
    prepare(
        model, total_len, ratio,
        similarity_chunk_size=max(1, int(similarity_chunk_size)),
    )

    past = None
    last_logits = None
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        ids = input_ids[:, start:end]
        q_len = end - start
        legacy = None if past is None else _as_legacy_tuple(past)
        physical_prefix = 0 if legacy is None else int(legacy[0][0].shape[-2])
        set_chunk(model, start)
        kwargs = {
            "input_ids": ids,
            "attention_mask": torch.ones(
                (ids.shape[0], physical_prefix + q_len),
                device=device,
                dtype=inputs["attention_mask"].dtype,
            ),
            "position_ids": torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0),
            "past_key_values": past,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        with torch.no_grad():
            out = model(**kwargs)
        # Keep only the final-token logits; release the rest immediately.
        last_logits = out.logits[:, -1, :].detach()
        past = out.past_key_values
        del out

    if last_logits is None or past is None:
        raise RuntimeError("Empty prompt in chunked prefill")
    first_token_id = last_logits[0].argmax()
    compressed = finalize(model, _as_legacy_tuple(past), sim_thr)
    del last_logits, past
    return compressed, first_token_id



class LlamaForCausalLMWithInputPartitioningForGenerationOnly(transformers.LlamaForCausalLM):
    """AquaKV LongBench Llama wrapper for generation-time input partitioning.

    In the KV-merge evaluator the initial full-prompt prefill must still return
    attentions and hidden states for compression. Therefore no-cache calls fall
    back to normal Llama forward; cached generation calls can still be chunked.
    """
    chunk_size_tokens = 8192
    TENSOR_KEYS = {"input_ids", "attention_mask", "cache_position", "position_ids"}

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values", None) is None or not kwargs.get("use_cache", False):
            return super().forward(*args, **kwargs)

        assert not torch.is_grad_enabled()
        assert not args, args
        for key, value in kwargs.items():
            assert (key in self.TENSOR_KEYS) == isinstance(value, torch.Tensor), (key, kwargs)

        input_len = kwargs["input_ids"].shape[1]
        # Manual compressed-cache decoding feeds one token at a time.  Calling
        # the native forward here is required so the returned cache grows; the
        # partitioning wrapper below intentionally reuses an existing cache and
        # is only appropriate for genuinely long cached input blocks.
        if input_len <= self.chunk_size_tokens:
            return super().forward(*args, **kwargs)
        cache_position = kwargs.get("cache_position", None)
        start_position = 0 if cache_position is None else cache_position[0].item()
        last_logits = None
        for chunk_start in range(0, input_len, self.chunk_size_tokens):
            chunk_end = min(chunk_start + self.chunk_size_tokens, input_len)
            chunk_kwargs = dict(kwargs)
            chunk_kwargs["input_ids"] = kwargs["input_ids"][:, chunk_start:chunk_end]
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                chunk_kwargs["attention_mask"] = kwargs["attention_mask"][:, : start_position + chunk_end]
            if cache_position is not None:
                chunk_kwargs["cache_position"] = cache_position[..., chunk_start:chunk_end]
            last_logits = super().forward(**chunk_kwargs).logits

        return transformers.models.llama.modeling_llama.CausalLMOutputWithPast(
            logits=last_logits,
            past_key_values=kwargs["past_key_values"],
        )


class Qwen2ForCausalLMWithInputPartitioningForGenerationOnly(Qwen2ForCausalLM):
    """Qwen2 wrapper from the AquaKV LongBench script; currently kept for parity."""
    chunk_size_tokens = 8192
    TENSOR_KEYS = {"input_ids", "attention_mask", "cache_position", "position_ids"}

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values", None) is None or not kwargs.get("use_cache", False):
            return super().forward(*args, **kwargs)

        assert not torch.is_grad_enabled()
        assert not args, args
        for key, value in kwargs.items():
            assert (key in self.TENSOR_KEYS) == isinstance(value, torch.Tensor), (key, kwargs)

        input_len = kwargs["input_ids"].shape[1]
        # Manual compressed-cache decoding feeds one token at a time.  Calling
        # the native forward here is required so the returned cache grows; the
        # partitioning wrapper below intentionally reuses an existing cache and
        # is only appropriate for genuinely long cached input blocks.
        if input_len <= self.chunk_size_tokens:
            return super().forward(*args, **kwargs)
        cache_position = kwargs.get("cache_position", None)
        start_position = 0 if cache_position is None else cache_position[0].item()
        last_logits = None
        for chunk_start in range(0, input_len, self.chunk_size_tokens):
            chunk_end = min(chunk_start + self.chunk_size_tokens, input_len)
            chunk_kwargs = dict(kwargs)
            chunk_kwargs["input_ids"] = kwargs["input_ids"][:, chunk_start:chunk_end]
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                chunk_kwargs["attention_mask"] = kwargs["attention_mask"][:, : start_position + chunk_end]
            if cache_position is not None:
                chunk_kwargs["cache_position"] = cache_position[..., chunk_start:chunk_end]
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                chunk_kwargs["position_ids"] = kwargs["position_ids"][..., chunk_start:chunk_end]
            last_logits = super().forward(**chunk_kwargs).logits

        return Qwen2CausalLMOutputWithPast(
            logits=last_logits,
            past_key_values=kwargs["past_key_values"],
        )


def load_model_and_tokenizer(path, model_name, device, torch_dtype=torch.bfloat16, attn_implementation="eager"):
    """AquaKV-style LongBench model loader, with only KV-merge-required changes.

    Kept from evaluate_longbench_aquakv.py: model-name dispatch, chatglm/internlm/xgen
    handling, llama-3.x input-partition wrapper, Qwen AutoModel branch,
    pad token setup, and eval mode.

    Intentionally not kept: the original LLaMA flash-attention monkey patch,
    because this KV-merge evaluator should not require flash-attn.

    Not kept: AquaKV quantized-cache arguments and predictor logic.
    """
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).to(device)
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path)
        llama_kwargs = dict(torch_dtype=torch_dtype)
        if attn_implementation:
            llama_kwargs["attn_implementation"] = attn_implementation
        model = LlamaForCausalLM.from_pretrained(path, **llama_kwargs).to(device)
    elif "llama-3.1" in model_name or "llama-3.2" in model_name:
        model = LlamaForCausalLMWithInputPartitioningForGenerationOnly.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    elif "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
    else:
        raise NotImplementedError(f"Could not load {model_name}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Make the requested attention backend explicit. Some transformer versions
    # keep this on config rather than honoring only the from_pretrained kwarg.
    if attn_implementation:
        try:
            model.config._attn_implementation = attn_implementation
        except Exception:
            pass
        try:
            model.generation_config._attn_implementation = attn_implementation
        except Exception:
            pass

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.eval()
    return model, tokenizer


def get_selected_sim_thr(args):
    method = normalize_kv_method(args.kv_method)
    if method == "None":
        return -1.0
    if method == "GroupLogitKV":
        return float(args.sim_thr_grouplogitkv)
    if method == "mymethod":
        return float(args.lambda_mymethod)
    if method == "SimCalKV":
        return float(args.sim_thr_simcalkv)
    if method == "KeepKV":
        return float(args.sim_thr_keepkv)
    return -1.0


def _self_attention_modules(model):
    """Return model self-attention modules in layer order for Llama/Qwen-style models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise AttributeError("Cannot find transformer layers for attention-score capture")
    modules = []
    for layer in layers:
        if hasattr(layer, "self_attn"):
            modules.append(layer.self_attn)
        elif hasattr(layer, "attention"):
            modules.append(layer.attention)
        else:
            raise AttributeError(f"Cannot find self-attention module in layer {layer}")
    return modules


def _set_reduced_attention_capture(model, enabled: bool, reduce: str = "sum"):
    """Enable/disable per-layer reduced attention-score capture.

    This avoids returning full HF attentions. For mymethod the runtime patch
    stores only the last-W masked base-score rows [B,H_kv,Q_tail,K]; legacy
    methods keep their existing reduced-score representation.
    """
    for attn in _self_attention_modules(model):
        setattr(attn, "_kvmerge_capture_attention_scores", bool(enabled))
        setattr(attn, "_kvmerge_attention_score_reduce", reduce)
        if hasattr(attn, "_kvmerge_captured_attention_score"):
            delattr(attn, "_kvmerge_captured_attention_score")
        if hasattr(attn, "_kvmerge_captured_score_log_scale"):
            delattr(attn, "_kvmerge_captured_score_log_scale")


def _get_reduced_attention_scores(model):
    scores = []
    for idx, attn in enumerate(_self_attention_modules(model)):
        score = getattr(attn, "_kvmerge_captured_attention_score", None)
        if score is None:
            raise RuntimeError(f"Missing reduced attention score for layer {idx}")
        scores.append(score)
    return tuple(scores)


def _clear_reduced_attention_scores(model):
    for attn in _self_attention_modules(model):
        if hasattr(attn, "_kvmerge_captured_attention_score"):
            delattr(attn, "_kvmerge_captured_attention_score")
        if hasattr(attn, "_kvmerge_captured_score_log_scale"):
            delattr(attn, "_kvmerge_captured_score_log_scale")


def _attention_reduce_mode_for_method(method: str) -> str:
    method = normalize_kv_method(method)
    # GroupLogitKV and SimCalKV use sum over query positions. mymethod keeps
    # its query axis and ignores this legacy reduction setting.
    # KeepKV and CaM use mean over query positions.  These are computed before
    # returning from the attention layer, instead of storing full attentions.
    return "mean" if method in ["KeepKV", "CaM"] else "sum"


def _need_attention_scores_for_method(method: str) -> bool:
    method = normalize_kv_method(method)
    # KeepKV captures its official masked exponential scores internally through
    # keepkv_runtime_patch_*; Repair_keepkv intentionally ignores this argument.
    return method in {"GroupLogitKV", "mymethod", "SimCalKV", "CaM"}


def _need_hidden_states_for_method(method: str) -> bool:
    return normalize_kv_method(method) == "SimCalKV"




def generate_full_kv_baseline(model, tokenizer, inputs, max_gen, dataset):
    """Full-precision, no-compression baseline.

    This path intentionally uses Hugging Face generate instead of the KV-merge
    prefill/compress/manual-decode path. It is mathematically the full-KV
    baseline and avoids forcing eager attention or any score-capture patch.
    """
    context_length = inputs.input_ids.shape[-1]
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
    )
    if dataset == "samsum":
        stop_ids = [tokenizer.eos_token_id]
        try:
            stop_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
        except Exception:
            pass
        gen_kwargs.update(
            min_length=context_length + 1,
            eos_token_id=[x for x in stop_ids if x is not None],
        )
    output = model.generate(**gen_kwargs)[0]
    return output[context_length:]


def run_dataset(model, tokenizer, apply_kv, reset_kv, data, max_length, max_gen, prompt_format, dataset, model_name, out_file, args):
    for idx, json_obj in enumerate(tqdm(data)):
        cleanup_interval = max(0, int(getattr(args, "cleanup_interval", 0)))
        if cleanup_interval and idx and idx % cleanup_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reset_kv(model)

        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(args.device)

        method = normalize_kv_method(args.kv_method)

        if method == "None":
            with torch.inference_mode():
                generated_tokens = generate_full_kv_baseline(model, tokenizer, inputs, max_gen, dataset)
            pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            pred = post_process(pred, model_name)
            row = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "kv_method": method,
                "compress_ratio": 0.0,
                "sim_thr": "na",
                "merge_time_sec": 0.0,
                "centroidkv_config": None,
            }
            with open(out_file, "a", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
            del inputs, generated_tokens, row
            continue

        reduce_mode = _attention_reduce_mode_for_method(method)
        need_attention_scores = _need_attention_scores_for_method(method)
        need_hidden_states = _need_hidden_states_for_method(method)
        sim_thr = get_selected_sim_thr(args)

        if method == "KeepKV":
            _sync(args.device)
            t0 = time.time()
            compressed_pkv, first_token_id = _chunked_prefill_and_compress(
                model=model,
                inputs=inputs,
                method=method,
                apply_kv=apply_kv,
                ratio=args.compress_ratio,
                sim_thr=sim_thr,
                device=args.device,
                chunk_size=getattr(args, "prefill_chunk_size", 64),
                similarity_chunk_size=getattr(args, "keepkv_similarity_chunk_size", 256),
            )
            _sync(args.device)
            merge_time = time.time() - t0
            outputs = None
            reduced_attentions = None
        else:
            if need_attention_scores:
                _set_reduced_attention_capture(model, True, reduce=reduce_mode)
            try:
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=False,
                        output_hidden_states=need_hidden_states,
                        use_cache=True,
                        return_dict=True,
                    )
                    first_token_id = outputs.logits[0, -1, :].argmax()
                reduced_attentions = _get_reduced_attention_scores(model) if need_attention_scores else None
            finally:
                if need_attention_scores:
                    _set_reduced_attention_capture(model, False, reduce=reduce_mode)

            _sync(args.device)
            t0 = time.time()
            with torch.no_grad():
                compressed_pkv = apply_kv(
                    method,
                    model,
                    outputs.past_key_values,
                    reduced_attentions,
                    outputs.hidden_states if need_hidden_states else None,
                    args.compress_ratio,
                    sim_thr,
                )
            _sync(args.device)
            merge_time = time.time() - t0

            if reduced_attentions is not None:
                del reduced_attentions

        stop_token_ids = None
        if dataset == "samsum":
            # Same special stop condition as the original LongBench script.
            try:
                stop_token_ids = [tokenizer.encode("\n", add_special_tokens=False)[-1]]
            except Exception:
                stop_token_ids = None
        generated = manual_generate_after_compress(
            model, tokenizer, inputs, compressed_pkv, first_token_id, max_gen, args.device, stop_token_ids=stop_token_ids
        )
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        reset_kv(model)
        _clear_reduced_attention_scores(model)

        # Release large tensors before the next sample.  This does not change
        # numerical results; it only removes Python references.
        try:
            del compressed_pkv, generated, first_token_id
            if outputs is not None:
                del outputs
        except Exception:
            pass
        row = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
            "kv_method": args.kv_method,
            "compress_ratio": args.compress_ratio,
            "sim_thr": get_selected_sim_thr(args),
            "merge_time_s": merge_time,
            "centroidkv_config": ({
                "sink_budget": args.centroidkv_sink_budget,
                "local_budget": args.centroidkv_local_budget,
                "chunk_size": args.centroidkv_chunk_size,
                "merge_ratio": args.centroidkv_merge_ratio,
                "decrease_rate": args.centroidkv_decrease_rate,
            } if method == "CentroidKV" else None),
        }
        with open(out_file, "a", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")



def _read_jsonl_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_jsonl_from_zip(zip_path, dataset_name):
    candidates = [
        f"data/{dataset_name}.jsonl",
        f"LongBench/data/{dataset_name}.jsonl",
        f"{dataset_name}.jsonl",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        jsonl_name = next((name for name in candidates if name in names), None)
        if jsonl_name is None:
            matches = [
                name for name in names
                if name.endswith(f"/{dataset_name}.jsonl") or name.endswith(f"{dataset_name}.jsonl")
            ]
            if matches:
                jsonl_name = sorted(matches, key=len)[0]
        if jsonl_name is None:
            raise FileNotFoundError(
                f"Cannot find {dataset_name}.jsonl in {zip_path}. Checked: {candidates}"
            )

        rows = []
        with zf.open(jsonl_name, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line.decode("utf-8")))
    return rows


def load_longbench_dataset(dataset_name, task_eval_dir):
    """Load one LongBench/LongBench-E split.

    Loading order:
      1. local jsonl directories, e.g. LongBench/data/<dataset>.jsonl
      2. local data.zip, e.g. LongBench/data.zip or $LONGBENCH_DATA_ZIP
      3. old datasets loader, for environments with datasets<4
      4. Hugging Face data.zip download, if network is available

    This keeps the script usable on servers without Hugging Face access: place
    data.zip or extracted jsonl files under task_evaluation/LongBench/data/.
    """
    task_eval_dir = Path(task_eval_dir).resolve()
    longbench_dir = task_eval_dir / "LongBench"

    data_dir_env = os.environ.get("LONGBENCH_DATA_DIR", "").strip()
    local_dirs = []
    if data_dir_env:
        local_dirs.append(Path(data_dir_env).expanduser())
    local_dirs.extend([
        longbench_dir / "data",
        task_eval_dir / "data",
        Path.cwd() / "data",
    ])

    for data_dir in local_dirs:
        jsonl_path = data_dir / f"{dataset_name}.jsonl"
        if jsonl_path.exists():
            print(f"[INFO] Load LongBench data from local jsonl: {jsonl_path}", flush=True)
            return _read_jsonl_file(jsonl_path)

    zip_env = os.environ.get("LONGBENCH_DATA_ZIP", "").strip()
    local_zips = []
    if zip_env:
        local_zips.append(Path(zip_env).expanduser())
    local_zips.extend([
        longbench_dir / "data.zip",
        task_eval_dir / "data.zip",
        Path.cwd() / "data.zip",
    ])

    for zip_path in local_zips:
        if zip_path.exists():
            print(f"[INFO] Load LongBench data from local zip: {zip_path}", flush=True)
            return _read_jsonl_from_zip(zip_path, dataset_name)

    try:
        return [x for x in load_dataset("THUDM/LongBench", dataset_name, split="test")]
    except Exception as e:
        msg = str(e)
        if "Dataset scripts are no longer supported" not in msg and "trust_remote_code" not in msg:
            print(f"[WARN] datasets.load_dataset failed: {type(e).__name__}: {e}", flush=True)
        else:
            print(
                "[WARN] datasets cannot load THUDM/LongBench via its old loading script; "
                "falling back to data.zip/jsonl loader.",
                flush=True,
            )
        last_err = e

    for repo_id in ("zai-org/LongBench", "THUDM/LongBench"):
        try:
            zip_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="data.zip",
            )
            print(f"[INFO] Downloaded LongBench data.zip from {repo_id}: {zip_path}", flush=True)
            return _read_jsonl_from_zip(zip_path, dataset_name)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not load LongBench data. Put data.zip at one of:\n"
        f"  {longbench_dir / 'data.zip'}\n"
        f"  {task_eval_dir / 'data.zip'}\n"
        "or set:\n"
        "  export LONGBENCH_DATA_ZIP=/path/to/data.zip\n"
        "Alternatively extract jsonl files to:\n"
        f"  {longbench_dir / 'data'}/<dataset>.jsonl\n"
        "or set:\n"
        "  export LONGBENCH_DATA_DIR=/path/to/data"
    ) from last_err

def main():
    time_started = time.time()
    args = parse_args()
    args.kv_method = normalize_kv_method(args.kv_method)
    if args.kv_method == "mymethod" and not (args.lambda_mymethod > 0.0):
        raise ValueError("--lambda_mymethod must be > 0")
    for name in (
        "mymethod_keepkv_sink_size", "mymethod_keepkv_recent_protect_size",
        "mymethod_keepkv_score_window", "mymethod_d2o_sink_size", "mymethod_d2o_recent_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.mymethod_keepkv_recency_beta < 1.0:
        raise ValueError("--mymethod_keepkv_recency_beta must be >= 1")
    if args.centroidkv_sink_budget < 0 or args.centroidkv_local_budget < 0:
        raise ValueError("CentroidKV sink/local budgets must be non-negative")
    if args.centroidkv_chunk_size < 2 or args.centroidkv_chunk_size % 2 != 0:
        raise ValueError("--centroidkv_chunk_size must be an even integer >= 2")
    if not 0.0 <= args.centroidkv_merge_ratio <= 1.0:
        raise ValueError("--centroidkv_merge_ratio must be in [0,1]")
    if args.centroidkv_decrease_rate < 0.0:
        raise ValueError("--centroidkv_decrease_rate must be non-negative")
    seed_everything(args.seed)
    if args.model is None:
        raise ValueError("--model is required")

    if args.arch == "auto":
        args.arch = infer_arch(args.model, args.model_path)
    if args.arch == "qwen" and args.kv_method == "CaM":
        # CaM implementation exists in task_evaluation/qwen/merge_cam_qwen.py; this script enables it.
        pass

    task_eval_dir = Path(args.task_eval_dir or Path(__file__).resolve().parent).resolve()
    if str(task_eval_dir) not in sys.path:
        sys.path.insert(0, str(task_eval_dir))

    model2path = json.load(open(task_eval_dir / "LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open(task_eval_dir / "LongBench/config/model2maxlen.json", "r"))
    if args.model not in model2path:
        raise KeyError(f"{args.model!r} is not in LongBench/config/model2path.json")
    if args.model not in model2maxlen:
        raise KeyError(f"{args.model!r} is not in LongBench/config/model2maxlen.json")
    model_path = args.model_path or model2path[args.model]
    max_length = model2maxlen[args.model]

    if args.all:
        datasets = DATASETS
    else:
        if args.datasets is None:
            raise ValueError("Pass --datasets or --all")
        datasets = [ds.strip() for ds in args.datasets.split(",")]
        for ds in datasets:
            if ds not in DATASETS:
                raise ValueError(f"dataset {ds} not supported by this LongBench v1 runner")

    method_norm = normalize_kv_method(args.kv_method)
    apply_kv, reset_kv = setup_kv_imports(
        task_eval_dir,
        args.arch,
        method_norm,
        simcalkv_similarity_chunk_size=args.simcalkv_similarity_chunk_size,
        mymethod_lambda=args.lambda_mymethod,
        mymethod_selector=args.mymethod_selector,
        mymethod_keepkv_sink_size=args.mymethod_keepkv_sink_size,
        mymethod_keepkv_recent_protect_size=args.mymethod_keepkv_recent_protect_size,
        mymethod_keepkv_score_window=args.mymethod_keepkv_score_window,
        mymethod_keepkv_recency_beta=args.mymethod_keepkv_recency_beta,
        mymethod_keepkv_sink_can_absorb=bool(args.mymethod_keepkv_sink_can_absorb),
        mymethod_d2o_sink_size=args.mymethod_d2o_sink_size,
        mymethod_d2o_recent_size=args.mymethod_d2o_recent_size,
        mymethod_d2o_sink_can_absorb=bool(args.mymethod_d2o_sink_can_absorb),
        mymethod_d2o_use_log_count=bool(args.mymethod_d2o_use_log_count),
        mymethod_distance_chunk_size=args.mymethod_distance_chunk_size,
        mymethod_group_mode=str(args.mymethod_group_mode),
        mymethod_graph_backend=str(args.mymethod_graph_backend),
        mymethod_route_backend=str(args.mymethod_route_backend),
        centroidkv_sink_budget=args.centroidkv_sink_budget,
        centroidkv_local_budget=args.centroidkv_local_budget,
        centroidkv_chunk_size=args.centroidkv_chunk_size,
        centroidkv_merge_ratio=args.centroidkv_merge_ratio,
        centroidkv_decrease_rate=args.centroidkv_decrease_rate,
    )
    torch_dtype = get_torch_dtype(args.dtype)

    # Full-KV baseline should not use eager attention on long contexts; eager
    # materializes [B,H,S,S] and OOMs for larger models.  The original AquaKV
    # LongBench loader also did not force eager for this path.  Keep eager for
    # KV-merge methods because their runtime patch supplies the exact reduced
    # attention scores needed by the compressors.
    effective_attn_impl = args.attn_implementation
    if method_norm == "None":
        if str(effective_attn_impl).lower() == "eager":
            effective_attn_impl = "sdpa"
        print(f"[INFO] kv_method=None: using full-KV generate path with attn_implementation={effective_attn_impl}.", flush=True)

    model, tokenizer = load_model_and_tokenizer(
        model_path, args.model, args.device, torch_dtype=torch_dtype, attn_implementation=effective_attn_impl
    )
    if method_norm == "mymethod":
        for attn_mod in _self_attention_modules(model):
            attn_mod._kvmerge_attention_chunk_size = int(args.mymethod_attention_chunk_size)
            attn_mod._mymethod_selector = str(args.mymethod_selector)
            attn_mod._mymethod_score_window = (
                int(args.mymethod_keepkv_score_window)
                if args.mymethod_selector == "keepkv" else 0
            )
            attn_mod._mymethod_d2o_use_log_count = bool(args.mymethod_d2o_use_log_count)
    reset_kv(model)

    dataset2prompt = json.load(open(task_eval_dir / "LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(task_eval_dir / "LongBench/config/dataset2maxlen.json", "r"))
    os.makedirs(args.out_path, exist_ok=True)

    for dataset in datasets:
        print(f"\n===== Running dataset: {dataset} | {args.arch} | {args.kv_method} | compress_ratio={args.compress_ratio} | sim_thr={get_selected_sim_thr(args)} =====", flush=True)
        out_file = f"{args.out_path}/{dataset}.jsonl"
        if os.path.exists(out_file):
            if args.overwrite:
                os.remove(out_file)
            else:
                print(f"===== Skip {dataset}, already exists: {out_file} =====", flush=True)
                continue

        data_all = load_longbench_dataset(f"{dataset}_e" if args.e else dataset, task_eval_dir)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        t0 = time.time()
        run_dataset(model, tokenizer, apply_kv, reset_kv, data_all, max_length, max_gen, prompt_format, dataset, args.model, out_file, args)
        print(f"===== Done {dataset}, time: {time.time() - t0:.1f}s =====", flush=True)

    result_name = "result_e.json" if args.e else "result.json"
    result_inp_path = f"{args.out_path}/"
    result_out_path = f"{args.out_path}/../{result_name}"
    collect_all_results(result_inp_path, result_out_path, args_e=args.e)
    print("Total time took:", time.time() - time_started)
    print("Result saved to:", result_out_path)


if __name__ == "__main__":
    main()
