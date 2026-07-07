"""
Runtime eager-attention patch for GroupLogitKV on LLaMA-family models.

This patch injects GroupLogitKV's logit-space count calibration into LLaMA eager
attention decoding:

    attention_logits = q @ k.T / sqrt(head_dim) + log(C)

Fast path
---------
If q_len == 1 and tensors are CUDA tensors, the patch uses the custom
calibrated_qk kernel:

    calibrated_qk(query_states, key_states, log_bias_q)

Fallback path
-------------
If the fused CUDA path is unavailable, the patch computes ordinary PyTorch QK
logits and then adds the available compressed-prefix log(C) bias. This keeps the
mathematical behavior consistent with GroupLogitKV even for CPU, non-CUDA, or
q_len > 1 execution, although it is not fused.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama import modeling_llama as llama_mod
from transformers.models.llama.modeling_llama import LlamaAttention

from grouplogitkv_calibrated_qk import calibrated_qk


_ORIG_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward
_PATCHED = False


def _prepare_log_bias(attn_module, log_bias_q, k_len, ref_tensor):
    """
    Keep log_bias_q as compressed-prefix metadata.

    log_bias_q may be shorter than the current key length because generated
    suffix tokens have count C=1 and therefore logC=0. The CUDA kernel treats
    j >= log_bias_len as zero bias, so this function must not pad/cat in the
    decode hot path.
    """
    del attn_module  # reserved for compatibility; currently unused

    if log_bias_q is None:
        return None

    if log_bias_q.shape[-1] > k_len:
        log_bias_q = log_bias_q[:, :, :, :k_len].contiguous()

    if log_bias_q.device != ref_tensor.device or log_bias_q.dtype != ref_tensor.dtype:
        log_bias_q = log_bias_q.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    return log_bias_q


def _add_prefix_log_bias_(attn_weights: torch.Tensor, log_bias_q: torch.Tensor) -> torch.Tensor:
    """
    Add compressed-prefix log(C) bias to attention logits.

    Args:
        attn_weights: [B, H, Q, K]
        log_bias_q:   [B, H, 1, Kc], where Kc <= K is allowed

    Returns:
        attn_weights with log_bias_q added to prefix positions. Suffix positions
        j >= Kc receive zero bias, matching C=1 for generated suffix tokens.
    """
    if log_bias_q is None:
        return attn_weights

    bias_len = min(log_bias_q.shape[-1], attn_weights.shape[-1])
    if bias_len <= 0:
        return attn_weights

    attn_weights[..., :bias_len] = attn_weights[..., :bias_len] + log_bias_q[..., :bias_len]
    return attn_weights



def _add_layer_local_chunk_causal_mask_(attn_weights: torch.Tensor, q_len: int, key_len: int) -> torch.Tensor:
    """Apply causal masking for the current physical chunk using this layer's KV length.

    This is needed when compressed caches have different physical lengths across
    layers. A single model-level 4D attention_mask built from
    layer 0 cannot correctly place the current-chunk triangular mask for every
    layer. The compressed prefix is always in the past, and only the current
    chunk suffix needs a triangular future mask.
    """
    if q_len <= 1:
        return attn_weights
    past_physical_len = key_len - q_len
    if past_physical_len < 0:
        return attn_weights
    min_val = torch.finfo(attn_weights.dtype).min
    chunk_future = torch.triu(
        torch.full((q_len, q_len), min_val, device=attn_weights.device, dtype=attn_weights.dtype),
        diagonal=1,
    )
    attn_weights[..., past_physical_len:key_len] = (
        attn_weights[..., past_physical_len:key_len] + chunk_future.view(1, 1, q_len, q_len)
    )
    return attn_weights

def _grouplogitkv_llama_attention_forward(
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
    del use_cache, kwargs  # preserved in signature for HF compatibility

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        llama_mod.logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = llama_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = llama_mod.repeat_kv(value_states, self.num_key_value_groups)

    log_bias_q = None
    grouplogitkv_enabled = (
        getattr(self, "_grouplogitkv_eager_enabled", False)
        and (not self.training)
        and hasattr(self, "_grouplogitkv_log_bias_q")
        and self._grouplogitkv_log_bias_q is not None
    )
    if grouplogitkv_enabled:
        log_bias_q = _prepare_log_bias(
            self,
            self._grouplogitkv_log_bias_q,
            key_states.shape[-2],
            query_states,
        )

    use_calibrated_qk = (
        log_bias_q is not None
        and q_len == 1
        and query_states.is_cuda
        and key_states.is_cuda
    )

    if use_calibrated_qk:
        attn_weights = calibrated_qk(query_states, key_states, log_bias_q)
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = _add_prefix_log_bias_(attn_weights, log_bias_q)

    if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
        attn_weights = _add_layer_local_chunk_causal_mask_(
            attn_weights, q_len=q_len, key_len=key_states.shape[-2]
        )
    elif attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_grouplogitkv_runtime_patch():
    global _PATCHED
    if not _PATCHED:
        LlamaAttention.forward = _grouplogitkv_llama_attention_forward
        _PATCHED = True
    return True


def remove_grouplogitkv_runtime_patch():
    global _PATCHED
    if _PATCHED:
        LlamaAttention.forward = _ORIG_LLAMA_ATTENTION_FORWARD
        _PATCHED = False
    return True
