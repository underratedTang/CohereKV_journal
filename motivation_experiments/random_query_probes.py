#!/usr/bin/env python3
from __future__ import annotations

"""Independent one-token query probes for CohereKV preliminary experiments.

Each probe token is evaluated as an independent one-token branch from the SAME
frozen prefix cache.  Its newly produced KV is cropped immediately, so probe
tokens never become context for one another and never enter the measured cache.

The v31 experiments pass the *natural tokens immediately following the selected
prefix* as ``token_ids``.  The older random-token sampling helper is retained only
for backward-compatible unit tests.  Post-RoPE query vectors are captured directly
from selected Llama self-attention modules.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class RandomProbeBatch:
    token_ids: List[int]
    queries_by_layer: Dict[int, torch.Tensor]  # [Hq, P, D], post-RoPE


def sample_distinct_random_token_ids(*, vocab_size: int, num_probes: int,
                                     excluded_ids: Iterable[int], seed: int) -> List[int]:
    """Uniformly sample distinct valid embedding IDs, excluding special IDs."""
    vocab = int(vocab_size)
    n = int(num_probes)
    excluded = {int(x) for x in excluded_ids if 0 <= int(x) < vocab}
    available = vocab - len(excluded)
    if n <= 0:
        raise ValueError("num_probes must be > 0")
    if n > available:
        raise ValueError(f"num_probes={n} exceeds available non-special vocabulary={available}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    chosen: List[int] = []
    seen = set(excluded)
    while len(chosen) < n:
        need = n - len(chosen)
        draw = torch.randint(0, vocab, (max(need * 3, 32),), generator=gen).tolist()
        for tid in draw:
            tid = int(tid)
            if tid in seen:
                continue
            seen.add(tid)
            chosen.append(tid)
            if len(chosen) == n:
                break
    return chosen


def _cache_len(cache) -> int | None:
    if cache is None:
        return None
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    if hasattr(cache, "key_cache"):
        try:
            if len(cache.key_cache):
                return int(cache.key_cache[0].shape[-2])
        except Exception:
            pass
    if isinstance(cache, (tuple, list)) and len(cache):
        try:
            return int(cache[0][0].shape[-2])
        except Exception:
            pass
    return None


def _crop_cache_in_place(cache, prefix_len: int) -> None:
    """Restore a mutable HF Cache to the frozen prefix length if possible."""
    if cache is None:
        return
    current = _cache_len(cache)
    if current is None or current <= int(prefix_len):
        return
    if hasattr(cache, "crop"):
        cache.crop(int(prefix_len))
        return
    if isinstance(cache, (tuple, list)):
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i][..., :int(prefix_len), :]
                cache.value_cache[i] = cache.value_cache[i][..., :int(prefix_len), :]
            return
        except Exception as exc:
            raise RuntimeError(
                f"Probe forward mutated cache to length {current}, but cache type {type(cache)!r} "
                "cannot be restored safely."
            ) from exc
    raise RuntimeError(
        f"Probe forward mutated cache to length {current}, but cache type {type(cache)!r} "
        "has no supported crop operation."
    )


def _resolve_attention_module(model, layer_idx: int):
    """Resolve a Llama-like decoder self-attention module without model-specific imports."""
    roots = []
    obj = model
    roots.append(obj)
    if hasattr(obj, "model"):
        roots.append(obj.model)
        if hasattr(obj.model, "model"):
            roots.append(obj.model.model)
    for root in roots:
        layers = getattr(root, "layers", None)
        if layers is None or int(layer_idx) >= len(layers):
            continue
        layer = layers[int(layer_idx)]
        for name in ("self_attn", "attention", "attn"):
            module = getattr(layer, name, None)
            if module is not None and hasattr(module, "q_proj"):
                return module
    raise RuntimeError(
        f"Cannot locate self-attention module for layer {layer_idx}. "
        "v17 query capture currently expects a Llama-like decoder with layer.self_attn.q_proj."
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = int(x.shape[-1]) // 2
    if 2 * half != int(x.shape[-1]):
        raise RuntimeError(f"RoPE head dimension must be even, got {x.shape[-1]}")
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _broadcast_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast common HF RoPE layouts to query shape [B,H,Q,D]."""
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    if cos.ndim == 2:        # [Q,D]
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.ndim == 3:      # [B,Q,D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim != 4:
        raise RuntimeError(f"Unsupported RoPE cos/sin rank: {cos.ndim}")
    return cos, sin


def _apply_rotary_to_query(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = _broadcast_rope(q, cos, sin)
    return (q * cos) + (_rotate_half(q) * sin)


def _num_attention_heads(module, projected_width: int) -> int:
    for obj in (module, getattr(module, "config", None)):
        if obj is None:
            continue
        for name in ("num_heads", "num_attention_heads"):
            val = getattr(obj, name, None)
            if val is not None:
                return int(val)
    head_dim = getattr(module, "head_dim", None)
    if head_dim is not None and int(head_dim) > 0:
        return int(projected_width) // int(head_dim)
    raise RuntimeError("Cannot infer number of query heads from attention module")


def _capture_post_rope_query(module, args, kwargs) -> torch.Tensor:
    """Reconstruct the query tensor exactly at the post-RoPE point used by Llama attention."""
    hidden = kwargs.get("hidden_states", None)
    if hidden is None and len(args):
        hidden = args[0]
    if hidden is None or hidden.ndim != 3:
        raise RuntimeError("Attention pre-hook did not receive [B,Q,H] hidden_states")

    q = module.q_proj(hidden)
    nheads = _num_attention_heads(module, int(q.shape[-1]))
    if int(q.shape[-1]) % nheads != 0:
        raise RuntimeError("q_proj width is not divisible by number of query heads")
    head_dim = int(q.shape[-1]) // nheads
    q = q.view(int(q.shape[0]), int(q.shape[1]), nheads, head_dim).transpose(1, 2)
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)

    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise RuntimeError("position_embeddings must be a (cos, sin) pair")
        cos, sin = position_embeddings
        q = _apply_rotary_to_query(q, cos, sin)
        return q

    # Compatibility path for older HF LlamaAttention versions where RoPE is
    # computed inside the attention module instead of passed from the model.
    rotary = getattr(module, "rotary_emb", None)
    position_ids = kwargs.get("position_ids", None)
    if rotary is not None:
        last_exc = None
        attempts = []
        if position_ids is not None:
            attempts.append(lambda: rotary(q, position_ids))
        attempts.append(lambda: rotary(q, seq_len=int(q.shape[-2])))
        for call in attempts:
            try:
                cos, sin = call()
                return _apply_rotary_to_query(q, cos, sin)
            except Exception as exc:  # pragma: no cover - version dependent
                last_exc = exc
        raise RuntimeError("Could not reconstruct post-RoPE query with legacy rotary_emb") from last_exc

    raise RuntimeError(
        "No position_embeddings/rotary_emb found while capturing queries. "
        "Use a supported Llama/Llama-3 transformers implementation."
    )


@torch.inference_mode()
def collect_independent_query_states(*, model, prefix_past, input_device: torch.device,
                                     token_ids: Sequence[int], selected_layers: Sequence[int],
                                     prefix_len: int,
                                     store_dtype: torch.dtype = torch.float16) -> RandomProbeBatch:
    """Capture post-RoPE query vectors for independent one-token branches.

    The current token is only a source of Q.  Its newly produced KV is cropped
    immediately and never enters any motivation attention calculation.
    """
    layers = [int(x) for x in selected_layers]
    if not layers:
        raise ValueError("selected_layers is empty")
    modules = {l: _resolve_attention_module(model, l) for l in layers}
    captured: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs):
            q = _capture_post_rope_query(module, args, kwargs)
            if int(q.shape[0]) != 1 or int(q.shape[-2]) != 1:
                raise RuntimeError(f"Expected one-token query branch, got {tuple(q.shape)}")
            # [Hq,D], one row per query head for this independent probe.
            captured[layer_idx].append(q[0, :, 0, :].detach().to(dtype=store_dtype).contiguous())
        return hook

    for layer_idx, module in modules.items():
        try:
            handles.append(module.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
        except TypeError as exc:  # pragma: no cover - old torch is unsupported by current Llama stacks
            raise RuntimeError("PyTorch with forward_pre_hook(with_kwargs=True) is required") from exc

    original_len = _cache_len(prefix_past)
    if original_len is not None and int(original_len) != int(prefix_len):
        raise RuntimeError(f"prefix cache length {original_len} != expected {prefix_len}")

    try:
        for token_id in token_ids:
            _crop_cache_in_place(prefix_past, int(prefix_len))
            ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=input_device)
            out = model(
                input_ids=ids,
                past_key_values=prefix_past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            returned_cache = getattr(out, "past_key_values", None)
            del out
            _crop_cache_in_place(returned_cache, int(prefix_len))
            _crop_cache_in_place(prefix_past, int(prefix_len))
    finally:
        for handle in handles:
            handle.remove()

    stacked: Dict[int, torch.Tensor] = {}
    expected = len(token_ids)
    for layer_idx, rows in captured.items():
        if len(rows) != expected:
            raise RuntimeError(
                f"Layer {layer_idx}: captured {len(rows)} queries for {expected} probes; "
                "attention module hook did not fire exactly once per branch"
            )
        # list[P] of [Hq,D] -> [Hq,P,D]
        stacked[layer_idx] = torch.stack(rows, dim=1).contiguous()

    return RandomProbeBatch(
        token_ids=[int(x) for x in token_ids],
        queries_by_layer=stacked,
    )


@torch.inference_mode()
def collect_independent_random_query_states(*, model, prefix_past, input_device: torch.device,
                                            token_ids: Sequence[int], selected_layers: Sequence[int],
                                            prefix_len: int,
                                            store_dtype: torch.dtype = torch.float16) -> RandomProbeBatch:
    """Backward-compatible alias for older tests/scripts."""
    return collect_independent_query_states(
        model=model, prefix_past=prefix_past, input_device=input_device,
        token_ids=token_ids, selected_layers=selected_layers, prefix_len=prefix_len,
        store_dtype=store_dtype,
    )
