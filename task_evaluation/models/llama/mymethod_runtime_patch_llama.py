"""
Runtime eager-attention patch for mymethod on LLaMA-family models.

This patch injects mymethod's logit-space count calibration into LLaMA eager
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
mathematical behavior consistent with mymethod even for CPU, non-CUDA, or
q_len > 1 execution, although it is not fused.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama import modeling_llama as llama_mod
from transformers.models.llama.modeling_llama import LlamaAttention

from models.common.mymethod_score_capture import (
    align_score_chunks,
    fallback_causal_visibility,
    physical_chunk_visibility,
    scaled_base_score_rows,
    score_window,
    slice_attention_mask,
    visibility_from_mask,
)

try:
    from grouplogitkv_calibrated_qk import calibrated_qk
except Exception:
    calibrated_qk = None


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


def _prepare_count_log_bias(attn_module, counts_kv, k_len, ref_tensor):
    """Compute temporary query-head log(C) from persistent KV-head counts."""
    if counts_kv is None:
        return None
    counts_kv = counts_kv[..., :k_len]
    if counts_kv.device != ref_tensor.device:
        counts_kv = counts_kv.to(device=ref_tensor.device)
    log_bias_kv = torch.log(counts_kv.float().clamp_min(1.0)).to(dtype=ref_tensor.dtype)
    groups = int(getattr(attn_module, "num_key_value_groups", 1))
    if groups != 1:
        bsz, h_kv, seqlen = log_bias_kv.shape
        log_bias_q = log_bias_kv[:, :, None, :].expand(
            bsz, h_kv, groups, seqlen
        ).reshape(bsz, h_kv * groups, seqlen)
    else:
        log_bias_q = log_bias_kv
    # The fused extension expects [B,H_q,1,K_prefix] contiguous metadata.
    return log_bias_q.unsqueeze(-2).contiguous()


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





def _get_attention_chunk_size(attn_module, q_len: int) -> int:
    """Query-block size for exact attention and recent base-score capture."""
    value = getattr(attn_module, "_kvmerge_attention_chunk_size", 128)
    try:
        value = int(value)
    except Exception:
        value = 128
    if value <= 0:
        return q_len
    return max(1, min(value, q_len))


def _apply_chunk_attention_mask_(
    attn_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    q_start: int,
    q_end: int,
    key_len: int,
    full_q_len: int,
) -> torch.Tensor:
    if attention_mask is not None:
        mask = slice_attention_mask(attention_mask, q_start, q_end, key_len)
        return attn_weights + mask
    visible = fallback_causal_visibility(
        attn_weights, q_start, q_end, key_len, full_q_len
    )
    return attn_weights.masked_fill(~visible, torch.finfo(attn_weights.dtype).min)


def _mymethod_selector_mode(attn_module) -> str:
    value = str(getattr(attn_module, "_mymethod_selector", "keepkv")).strip().lower()
    if value not in {"keepkv", "grouplogitkv"}:
        raise ValueError(f"Unknown mymethod selector mode: {value}")
    return value


def _reduce_postsoftmax_scores_to_kv_heads(
    attn_probs: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Match GroupLogitKV: average GQA query-head groups, sum query rows."""
    bsz, h_q, q_len, k_len = attn_probs.shape
    if h_q == num_kv_heads:
        return attn_probs.float().sum(dim=2)
    if h_q % num_kv_heads != 0:
        raise ValueError(f"H_q={h_q} is not divisible by H_kv={num_kv_heads}")
    groups = h_q // num_kv_heads
    return attn_probs.float().view(
        bsz, num_kv_heads, groups, q_len, k_len
    ).mean(dim=2).sum(dim=2)


def _accumulate_postsoftmax_scores(
    score_accum: torch.Tensor,
    attn_probs: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    current = _reduce_postsoftmax_scores_to_kv_heads(attn_probs, num_kv_heads)
    return current if score_accum is None else score_accum + current


def _capture_score_slice(
    attn_module,
    base_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    q_start: int,
    q_end: int,
    key_len: int,
    full_q_len: int,
):
    mask = slice_attention_mask(attention_mask, q_start, q_end, key_len)
    if mask is None:
        visible = fallback_causal_visibility(
            base_logits, q_start, q_end, key_len, full_q_len
        )
    else:
        visible = visibility_from_mask(mask, base_logits)
    return scaled_base_score_rows(
        base_logits, visible, attn_module.num_key_value_heads
    )


def _exact_chunked_attention_with_base_scores(
    attn_module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    log_bias_q: torch.Tensor,
    full_q_len: int,
    output_attentions: bool,
):
    """Exact query-blocked attention and last-W masked base-score rows."""
    del output_attentions
    bsz, _, q_len, _ = query_states.shape
    key_len = key_states.shape[-2]
    chunk_size = _get_attention_chunk_size(attn_module, q_len)
    selector_mode = _mymethod_selector_mode(attn_module)
    keep_rows = score_window(attn_module, q_len) if selector_mode == "keepkv" else 0
    capture_start = q_len - keep_rows

    outputs = []
    score_chunks = []
    score_scales = []
    score_accum = None
    for q_start in range(0, q_len, chunk_size):
        q_end = min(q_start + chunk_size, q_len)
        q_chunk = query_states[:, :, q_start:q_end, :]
        base_logits = torch.matmul(
            q_chunk, key_states.transpose(2, 3)
        ) / math.sqrt(attn_module.head_dim)

        if selector_mode == "keepkv" and q_end > capture_start:
            local_start = max(capture_start, q_start) - q_start
            capture_logits = base_logits[:, :, local_start:, :]
            score, scale = _capture_score_slice(
                attn_module,
                capture_logits,
                attention_mask,
                q_start + local_start,
                q_end,
                key_len,
                full_q_len,
            )
            score_chunks.append(score)
            score_scales.append(scale)

        attn_weights = _add_prefix_log_bias_(base_logits, log_bias_q)
        attn_weights = _apply_chunk_attention_mask_(
            attn_weights, attention_mask, q_start, q_end, key_len, full_q_len
        )
        attn_probs = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        )
        attn_probs = nn.functional.dropout(
            attn_probs,
            p=attn_module.attention_dropout,
            training=attn_module.training,
        )
        if selector_mode == "grouplogitkv":
            score_accum = _accumulate_postsoftmax_scores(
                score_accum, attn_probs, attn_module.num_key_value_heads
            )
        outputs.append(torch.matmul(attn_probs.to(value_states.dtype), value_states))
        del attn_weights, attn_probs, q_chunk, base_logits

    if selector_mode == "grouplogitkv":
        if score_accum is None:
            score_accum = torch.zeros(
                (bsz, attn_module.num_key_value_heads, key_len),
                device=query_states.device,
                dtype=torch.float32,
            )
        attn_module._kvmerge_captured_attention_score = score_accum.detach()
        attn_module._kvmerge_captured_score_log_scale = 0.0
    else:
        score_rows, log_scale = align_score_chunks(
            score_chunks,
            score_scales,
            (bsz, attn_module.num_key_value_heads, 0, key_len),
            query_states.device,
        )
        attn_module._kvmerge_captured_attention_score = score_rows.detach()
        attn_module._kvmerge_captured_score_log_scale = float(log_scale)
    return torch.cat(outputs, dim=2), None

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

def _mymethod_llama_attention_forward(
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
    decode_bias_source = None

    mymethod_enabled = (
        getattr(self, "_mymethod_eager_enabled", False)
        and (not self.training)
        and hasattr(self, "_mymethod_counts_kv")
        and self._mymethod_counts_kv is not None
    )
    keepkv_enabled = (
        getattr(self, "_keepkv_eager_enabled", False)
        and (not self.training)
        and hasattr(self, "_keepkv_log_bias_q")
        and self._keepkv_log_bias_q is not None
    )

    if mymethod_enabled:
        log_bias_q = _prepare_count_log_bias(
            self,
            self._mymethod_counts_kv,
            key_states.shape[-2],
            query_states,
        )
        decode_bias_source = "mymethod"
    elif keepkv_enabled:
        log_bias_q = _prepare_log_bias(
            self,
            self._keepkv_log_bias_q,
            key_states.shape[-2],
            query_states,
        )
        decode_bias_source = "KeepKV"

    # Only mymethod uses the calibrated_qk CUDA kernel.  KeepKV may also
    # use a log-count bias, but it must not enter the mymethod kernel path.
    # The patched calibrated_qk extension supports K > 16384; if an old extension
    # is installed, reinstall grouplogitkv_calibrated_qk from the patched source.
    use_calibrated_qk = (
        calibrated_qk is not None
        and decode_bias_source == "mymethod"
        and log_bias_q is not None
        and q_len == 1
        and query_states.is_cuda
        and key_states.is_cuda
    )

    capture_scores = getattr(self, "_kvmerge_capture_attention_scores", False) and (not self.training)
    selector_mode = _mymethod_selector_mode(self)
    # Score capture needs unbiased qK logits, so the fused calibrated kernel is
    # reserved for decode steps that do not update the selector history.
    use_calibrated_qk = use_calibrated_qk and (not capture_scores)

    if capture_scores and q_len > 1 and not getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
        attn_output, attn_weights = _exact_chunked_attention_with_base_scores(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            log_bias_q,
            full_q_len=q_len,
            output_attentions=output_attentions,
        )
    else:
        if use_calibrated_qk:
            attn_weights = calibrated_qk(query_states, key_states, log_bias_q)
        else:
            base_logits = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            if capture_scores and selector_mode == "keepkv":
                keep_rows = score_window(self, q_len)
                capture_start = q_len - keep_rows
                if keep_rows == 0:
                    score_rows = torch.empty(
                        (bsz, self.num_key_value_heads, 0, key_states.shape[-2]),
                        device=query_states.device,
                        dtype=torch.float32,
                    )
                    log_scale = 0.0
                else:
                    capture_logits = base_logits[:, :, capture_start:, :]
                    if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
                        visible = physical_chunk_visibility(base_logits, q_len, key_states.shape[-2])
                        visible = visible[:, :, capture_start:, :]
                    else:
                        mask = slice_attention_mask(
                            attention_mask, capture_start, q_len, key_states.shape[-2]
                        )
                        if mask is None:
                            visible = fallback_causal_visibility(
                                capture_logits,
                                capture_start,
                                q_len,
                                key_states.shape[-2],
                                q_len,
                            )
                        else:
                            visible = visibility_from_mask(mask, capture_logits)
                    score_rows, log_scale = scaled_base_score_rows(
                        capture_logits, visible, self.num_key_value_heads
                    )
                self._kvmerge_captured_attention_score = score_rows.detach()
                self._kvmerge_captured_score_log_scale = float(log_scale)
            attn_weights = _add_prefix_log_bias_(base_logits, log_bias_q)

        if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
            attn_weights = _add_layer_local_chunk_causal_mask_(
                attn_weights, q_len=q_len, key_len=key_states.shape[-2]
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        if capture_scores and selector_mode == "grouplogitkv":
            self._kvmerge_captured_attention_score = (
                _reduce_postsoftmax_scores_to_kv_heads(
                    attn_weights, self.num_key_value_heads
                ).detach()
            )
            self._kvmerge_captured_score_log_scale = 0.0
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

    if getattr(self, "_kvmerge_capture_attention_scores", False) or not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_mymethod_runtime_patch():
    global _PATCHED
    if not _PATCHED:
        LlamaAttention.forward = _mymethod_llama_attention_forward
        _PATCHED = True
    return True


def remove_mymethod_runtime_patch():
    global _PATCHED
    if _PATCHED:
        LlamaAttention.forward = _ORIG_LLAMA_ATTENTION_FORWARD
        _PATCHED = False
    return True
