"""Method-agnostic memory-bounded *eager* attention for Qwen2.

Used by LongBench Full-KV and CentroidKV, neither of which consumes attention
scores.  For long multi-token prefills it tiles only the query dimension inside
an otherwise eager attention calculation.  Compression timing, KV contents,
RoPE positions, masks, softmax definition, and decode semantics are unchanged.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from models.common.exact_eager_attention import exact_eager_attention_output

_ORIG_QWEN2_ATTENTION_FORWARD = Qwen2Attention.forward
_PATCHED = False


def _chunk_size(attn_module, q_len: int) -> int:
    value = getattr(attn_module, "_kvmerge_attention_chunk_size", None)
    if value is None:
        value = os.environ.get("KVMERGE_ATTENTION_CHUNK_SIZE", "128")
    try:
        value = int(value)
    except Exception:
        value = 128
    if value <= 0:
        return q_len
    return max(1, min(value, q_len))


def _exact_eager_qwen2_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    q_len = int(hidden_states.shape[1])
    tile = _chunk_size(self, q_len)
    # Preserve the stock implementation for short calls, training, and callers
    # that explicitly request the full attention tensor.
    if self.training or output_attentions or q_len <= tile:
        return _ORIG_QWEN2_ATTENTION_FORWARD(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        qwen2_mod.logger.warning_once(
            "Qwen2 attention is transitioning to externally supplied position_embeddings."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = qwen2_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = qwen2_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value_states, self.num_key_value_groups)
    attn_output = exact_eager_attention_output(
        query_states,
        key_states,
        value_states,
        head_dim=self.head_dim,
        attention_mask=attention_mask,
        chunk_size=tile,
        attention_dropout=self.attention_dropout,
        training=self.training,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"Unexpected attention output shape: {tuple(attn_output.shape)}")
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def apply_exact_eager_qwen_runtime_patch():
    global _PATCHED
    if not _PATCHED:
        Qwen2Attention.forward = _exact_eager_qwen2_attention_forward
        _PATCHED = True
    return True


def remove_exact_eager_qwen_runtime_patch():
    global _PATCHED
    if _PATCHED:
        Qwen2Attention.forward = _ORIG_QWEN2_ATTENTION_FORWARD
        _PATCHED = False
    return True


__all__ = ["apply_exact_eager_qwen_runtime_patch", "remove_exact_eager_qwen_runtime_patch"]
