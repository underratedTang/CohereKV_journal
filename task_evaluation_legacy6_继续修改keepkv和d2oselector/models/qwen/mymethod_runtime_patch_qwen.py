"""
mymethod runtime attention patch for Qwen2-family models.

This patch replaces Qwen2Attention.forward with an eager attention path that
supports mymethod logit-space count calibration:

    attention_logits = q @ k.T / sqrt(head_dim) + log(C)

The independent mymethod CUDA fused QK+log(C) kernel is used for the common single-token decode
case. For non-fused cases, the fallback path still applies the compressed-prefix
log(C) bias in PyTorch so the mathematical behavior remains consistent.
"""

import math
import torch
import torch.nn as nn

from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from models.common.mymethod_common import accumulate_d2o_importance_state

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
    from mymethod_cuda import count_calibrated_qk as _mymethod_count_calibrated_qk
    from mymethod_cuda import is_available as _mymethod_cuda_available
except Exception:
    _mymethod_count_calibrated_qk = None
    _mymethod_cuda_available = lambda: False


_ORIG_QWEN2_ATTENTION_FORWARD = Qwen2Attention.forward
_PATCHED = False


def _prepare_log_bias(log_bias_q, k_len, ref_tensor):
    """
    Keep log_bias_q as compressed-prefix metadata.

    log_bias_q may be shorter than the current key length because generated
    suffix tokens have count C=1 and therefore logC=0. The CUDA kernel treats
    j >= log_bias_len as zero bias, so this function must not pad/cat in the
    decode hot path.
    """
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
    return log_bias_q.unsqueeze(-2).contiguous()


def _add_prefix_log_bias(attn_weights: torch.Tensor, log_bias_q: torch.Tensor) -> torch.Tensor:
    """
    Add compressed-prefix log(C) bias to attention logits.

    attn_weights: [B, H_q, Q, K]
    log_bias_q:   [B, H_q, 1, K_prefix]

    Positions beyond K_prefix are generated suffix tokens with implicit count 1,
    hence zero log-bias.
    """
    if log_bias_q is None:
        return attn_weights

    bias_len = min(log_bias_q.shape[-1], attn_weights.shape[-1])
    if bias_len <= 0:
        return attn_weights

    attn_weights = attn_weights.clone()
    attn_weights[..., :bias_len] = attn_weights[..., :bias_len] + log_bias_q[..., :bias_len]
    return attn_weights





def _get_attention_chunk_size(attn_module, q_len: int) -> int:
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
        return attn_weights + slice_attention_mask(
            attention_mask, q_start, q_end, key_len
        )
    visible = fallback_causal_visibility(
        attn_weights, q_start, q_end, key_len, full_q_len
    )
    return attn_weights.masked_fill(~visible, torch.finfo(attn_weights.dtype).min)


def _mymethod_selector_mode(attn_module) -> str:
    value = str(getattr(attn_module, "_mymethod_selector", "keepkv")).strip().lower()
    if value not in {"keepkv", "attention", "d2o"}:
        raise ValueError(f"Unknown mymethod selector mode: {value}")
    return value


def _d2o_use_log_count(attn_module) -> bool:
    return bool(getattr(attn_module, "_mymethod_d2o_use_log_count", True))


def _accumulate_d2o_persistent(attn_module, current: torch.Tensor) -> torch.Tensor:
    total = accumulate_d2o_importance_state(
        getattr(attn_module, "_mymethod_d2o_cumulative_importance", None),
        current,
    )
    attn_module._mymethod_d2o_cumulative_importance = total.detach()
    return total


def _reduce_postsoftmax_scores_to_kv_heads(
    attn_probs: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Match current-attention: average GQA query-head groups, sum query rows."""
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
            score, scale = _capture_score_slice(
                attn_module,
                base_logits[:, :, local_start:, :],
                attention_mask,
                q_start + local_start,
                q_end,
                key_len,
                full_q_len,
            )
            score_chunks.append(score)
            score_scales.append(scale)

        if selector_mode == "d2o" and not _d2o_use_log_count(attn_module):
            selector_logits = _apply_chunk_attention_mask_(
                base_logits.clone(), attention_mask, q_start, q_end, key_len, full_q_len
            )
            selector_probs = nn.functional.softmax(
                selector_logits, dim=-1, dtype=torch.float32
            )
            score_accum = _accumulate_postsoftmax_scores(
                score_accum, selector_probs, attn_module.num_key_value_heads
            )
            del selector_logits, selector_probs

        attn_weights = _add_prefix_log_bias(base_logits, log_bias_q)
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
        if selector_mode == "attention" or (
            selector_mode == "d2o" and _d2o_use_log_count(attn_module)
        ):
            score_accum = _accumulate_postsoftmax_scores(
                score_accum, attn_probs, attn_module.num_key_value_heads
            )
        outputs.append(torch.matmul(attn_probs.to(value_states.dtype), value_states))
        del attn_weights, attn_probs, q_chunk, base_logits

    if selector_mode in {"attention", "d2o"}:
        if score_accum is None:
            score_accum = torch.zeros(
                (bsz, attn_module.num_key_value_heads, key_len),
                device=query_states.device,
                dtype=torch.float32,
            )
        if selector_mode == "d2o":
            score_accum = _accumulate_d2o_persistent(attn_module, score_accum)
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

def _mymethod_qwen2_attention_forward(
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
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        qwen2_mod.logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = qwen2_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = qwen2_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value_states, self.num_key_value_groups)

    log_bias_q = None
    decode_bias_source = None

    mymethod_enabled = (
        (getattr(self, "_mymethod_eager_enabled", False) or getattr(self, "_mymethod_sdpa_enabled", False))
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
            self._keepkv_log_bias_q,
            key_states.shape[-2],
            query_states,
        )
        decode_bias_source = "KeepKV"

    # Only mymethod uses the mymethod_count_calibrated_qk CUDA kernel.  KeepKV may also
    # use a log-count bias, but it must not enter the mymethod kernel path.
    # The patched mymethod_count_calibrated_qk extension supports K > 16384; if an old extension
    # is installed, install/reinstall the independent mymethod_cuda package.
    use_mymethod_count_calibrated_qk = (
        _mymethod_count_calibrated_qk is not None
        and _mymethod_cuda_available()
        and decode_bias_source == "mymethod"
        and log_bias_q is not None
        and q_len == 1
        and query_states.is_cuda
        and key_states.is_cuda
    )

    capture_scores = getattr(self, "_kvmerge_capture_attention_scores", False) and (not self.training)
    selector_mode = _mymethod_selector_mode(self)
    use_mymethod_count_calibrated_qk = use_mymethod_count_calibrated_qk and (not capture_scores)

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
        if use_mymethod_count_calibrated_qk:
            attn_weights = _mymethod_count_calibrated_qk(query_states, key_states, log_bias_q)
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
                        visible = physical_chunk_visibility(
                            base_logits, q_len, key_states.shape[-2]
                        )[:, :, capture_start:, :]
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
            d2o_no_count_logits = None
            if capture_scores and selector_mode == "d2o" and not _d2o_use_log_count(self):
                d2o_no_count_logits = base_logits.clone()
            attn_weights = _add_prefix_log_bias(base_logits, log_bias_q)

        if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
            attn_weights = _add_layer_local_chunk_causal_mask_(
                attn_weights, q_len=q_len, key_len=key_states.shape[-2]
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        d2o_no_count_probs = None
        if capture_scores and selector_mode == "d2o" and not _d2o_use_log_count(self):
            if d2o_no_count_logits is None:
                raise RuntimeError("missing no-logC D2O selector logits")
            if getattr(past_key_value, "_ppl_use_physical_causal_mask", False):
                d2o_no_count_logits = _add_layer_local_chunk_causal_mask_(
                    d2o_no_count_logits, q_len=q_len, key_len=key_states.shape[-2]
                )
            elif attention_mask is not None:
                d2o_no_count_logits = d2o_no_count_logits + attention_mask[:, :, :, : key_states.shape[-2]]
            d2o_no_count_probs = nn.functional.softmax(
                d2o_no_count_logits, dim=-1, dtype=torch.float32
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        if capture_scores and selector_mode in {"attention", "d2o"}:
            selector_probs = (
                d2o_no_count_probs
                if selector_mode == "d2o" and not _d2o_use_log_count(self)
                else attn_weights
            )
            reduced = _reduce_postsoftmax_scores_to_kv_heads(
                selector_probs, self.num_key_value_heads
            )
            if selector_mode == "d2o":
                reduced = _accumulate_d2o_persistent(self, reduced)
            self._kvmerge_captured_attention_score = reduced.detach()
            self._kvmerge_captured_score_log_scale = 0.0
            if d2o_no_count_probs is not None:
                del d2o_no_count_probs, d2o_no_count_logits
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if getattr(self, "_kvmerge_capture_attention_scores", False) or not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_mymethod_qwen_runtime_patch():
    global _PATCHED
    if not _PATCHED:
        Qwen2Attention.forward = _mymethod_qwen2_attention_forward
        _PATCHED = True
    return True


def remove_mymethod_qwen_runtime_patch():
    global _PATCHED
    if _PATCHED:
        Qwen2Attention.forward = _ORIG_QWEN2_ATTENTION_FORWARD
        _PATCHED = False
    return True
