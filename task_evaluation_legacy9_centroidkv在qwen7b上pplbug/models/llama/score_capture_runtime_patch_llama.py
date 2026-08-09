"""Method-agnostic reduced-attention capture runtime for Llama models.

Used by attention-score consumers such as SimCalKV and CaM.  It intentionally
contains no GroupLogitKV count calibration, selector state, or merge behavior.
Multi-token capture uses exact query chunking and immediately reduces scores to
[B, H_kv, K] to bound temporary attention workspace.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama import modeling_llama as llama_mod
from transformers.models.llama.modeling_llama import LlamaAttention


_ORIG_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward
_PATCHED = False


def _reduce_attention_scores_to_kv_heads(attn_weights, num_kv_heads, reduce="sum"):
    if attn_weights.dim() != 4:
        raise ValueError(f"Expected attention [B,H_q,Q,K], got {tuple(attn_weights.shape)}")
    bsz, h_q, q_len, k_len = attn_weights.shape
    if h_q == num_kv_heads:
        attn_kv = attn_weights
    else:
        if h_q % num_kv_heads != 0:
            raise ValueError(f"H_q={h_q} is not divisible by H_kv={num_kv_heads}")
        groups = h_q // num_kv_heads
        attn_kv = attn_weights.view(bsz, num_kv_heads, groups, q_len, k_len).mean(dim=2)
    if reduce == "mean":
        return attn_kv.mean(dim=2).detach()
    if reduce == "sum":
        return attn_kv.sum(dim=2).detach()
    raise ValueError(f"Unknown attention score reduction: {reduce}")


def _get_attention_chunk_size(attn_module, q_len):
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


def _apply_chunk_attention_mask_(attn_weights, attention_mask, q_start, q_end, key_len, full_q_len):
    if attention_mask is not None:
        if attention_mask.shape[-2] == 1:
            mask = attention_mask[:, :, :, :key_len]
        else:
            mask = attention_mask[:, :, q_start:q_end, :key_len]
        return attn_weights + mask
    chunk_len = q_end - q_start
    past_len = key_len - full_q_len
    if full_q_len <= 1 or past_len < 0:
        return attn_weights
    q_abs = torch.arange(q_start, q_end, device=attn_weights.device) + past_len
    k_abs = torch.arange(key_len, device=attn_weights.device)
    future = k_abs.view(1, -1) > q_abs.view(-1, 1)
    if future.any():
        attn_weights = attn_weights.masked_fill(
            future.view(1, 1, chunk_len, key_len), torch.finfo(attn_weights.dtype).min
        )
    return attn_weights


def _accumulate_reduced_attention_scores(score_accum, attn_probs, num_kv_heads):
    bsz, h_q, chunk_len, k_len = attn_probs.shape
    if score_accum is None:
        score_accum = torch.zeros(
            (bsz, num_kv_heads, k_len), device=attn_probs.device, dtype=torch.float32
        )
    if h_q == num_kv_heads:
        return score_accum + attn_probs.float().sum(dim=2)
    if h_q % num_kv_heads != 0:
        raise ValueError(f"H_q={h_q} is not divisible by H_kv={num_kv_heads}")
    groups = h_q // num_kv_heads
    reduced = attn_probs.float().view(
        bsz, num_kv_heads, groups, chunk_len, k_len
    ).mean(dim=2).sum(dim=2)
    return score_accum + reduced


def _exact_chunked_attention_with_reduced_scores(
    attn_module, query_states, key_states, value_states, attention_mask, full_q_len
):
    _, _, q_len, _ = query_states.shape
    key_len = key_states.shape[-2]
    chunk_size = _get_attention_chunk_size(attn_module, q_len)
    outputs = []
    score_accum = None
    for q_start in range(0, q_len, chunk_size):
        q_end = min(q_start + chunk_size, q_len)
        q_chunk = query_states[:, :, q_start:q_end, :]
        attn_weights = torch.matmul(q_chunk, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)
        attn_weights = _apply_chunk_attention_mask_(
            attn_weights, attention_mask, q_start, q_end, key_len, full_q_len
        )
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = nn.functional.dropout(
            attn_probs, p=attn_module.attention_dropout, training=attn_module.training
        )
        score_accum = _accumulate_reduced_attention_scores(
            score_accum, attn_probs, attn_module.num_key_value_heads
        )
        outputs.append(torch.matmul(attn_probs.to(value_states.dtype), value_states))
        del q_chunk, attn_weights, attn_probs
    reduce_mode = getattr(attn_module, "_kvmerge_attention_score_reduce", "sum")
    if reduce_mode == "mean":
        score_accum = score_accum / float(q_len)
    elif reduce_mode != "sum":
        raise ValueError(f"Unknown attention score reduction: {reduce_mode}")
    attn_module._kvmerge_captured_attention_score = score_accum.detach()
    return torch.cat(outputs, dim=2), None


def _add_layer_local_chunk_causal_mask_(attn_weights, q_len, key_len):
    if q_len <= 1:
        return attn_weights
    prefix_len = key_len - q_len
    if prefix_len < 0:
        return attn_weights
    q = torch.arange(q_len, device=attn_weights.device)
    k = torch.arange(key_len, device=attn_weights.device)
    future = k.view(1, -1) > (prefix_len + q).view(-1, 1)
    if future.any():
        attn_weights = attn_weights.masked_fill(
            future.view(1, 1, q_len, key_len), torch.finfo(attn_weights.dtype).min
        )
    return attn_weights


def _score_capture_llama_attention_forward(
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
    del use_cache, kwargs
    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        kv_slice = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        q_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        k_slices = self.k_proj.weight.split(kv_slice, dim=0)
        v_slices = self.v_proj.weight.split(kv_slice, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, q_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        key_states = torch.cat(
            [F.linear(hidden_states, k_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        value_states = torch.cat(
            [F.linear(hidden_states, v_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        llama_mod.logger.warning_once(
            "Llama attention is transitioning to externally supplied position_embeddings."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = llama_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = llama_mod.repeat_kv(value_states, self.num_key_value_groups)
    capture_scores = getattr(self, "_kvmerge_capture_attention_scores", False) and (not self.training)

    if capture_scores and q_len > 1 and not getattr(
        past_key_value, "_ppl_use_physical_causal_mask", False
    ):
        attn_output, attn_weights = _exact_chunked_attention_with_reduced_scores(
            self, query_states, key_states, value_states, attention_mask, full_q_len=q_len
        )
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
            attn_weights = _add_layer_local_chunk_causal_mask_(
                attn_weights, q_len=q_len, key_len=key_states.shape[-2]
            )
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        if capture_scores:
            reduce_mode = getattr(self, "_kvmerge_attention_score_reduce", "sum")
            self._kvmerge_captured_attention_score = _reduce_attention_scores_to_kv_heads(
                attn_weights, self.num_key_value_heads, reduce=reduce_mode
            )
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"Unexpected attention output shape: {tuple(attn_output.shape)}")
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    if self.config.pretraining_tp > 1:
        chunks = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            F.linear(chunks[i], o_slices[i]) for i in range(self.config.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)
    if capture_scores or not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def apply_score_capture_runtime_patch():
    global _PATCHED
    if not _PATCHED:
        LlamaAttention.forward = _score_capture_llama_attention_forward
        _PATCHED = True
    return True


def remove_score_capture_runtime_patch():
    global _PATCHED
    if _PATCHED:
        LlamaAttention.forward = _ORIG_LLAMA_ATTENTION_FORWARD
        _PATCHED = False
    return True


__all__ = ["apply_score_capture_runtime_patch", "remove_score_capture_runtime_patch"]
