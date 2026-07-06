import torch
import grouplogitkv_calibrated_qk_ext


def calibrated_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    log_bias_q: torch.Tensor,
) -> torch.Tensor:
    """
    Decode-only calibrated QK score generation.

    Computes:
        scores[j] = QK[j] / sqrt(D) + log_bias_q[j], if j < log_bias_len
        scores[j] = QK[j] / sqrt(D),                 otherwise

    This matches GroupLogitKV decoding because generated suffix tokens have
    count C=1 and therefore logC=0.

    query_states: [B,H,1,D]
    key_states:   [B,H,K,D], after repeat_kv
    log_bias_q:   [B,H,1,Kc], where Kc may be <= K
    returns:      [B,H,1,K]
    """
    if log_bias_q.device != query_states.device or log_bias_q.dtype != query_states.dtype:
        log_bias_q = log_bias_q.to(device=query_states.device, dtype=query_states.dtype)

    return grouplogitkv_calibrated_qk_ext.calibrated_qk_q1(
        query_states,
        key_states,
        log_bias_q.contiguous(),
    )
