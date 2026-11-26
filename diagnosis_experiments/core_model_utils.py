#!/usr/bin/env python3
"""Shared model, tokenizer, native cache and contiguous data helpers."""
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


def load_token_stream(tokenizer, dataset_name: str, split='test') -> torch.Tensor:
    if dataset_name != "wikitext2":
        raise ValueError("This diagnosis currently supports only wikitext2.")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
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


