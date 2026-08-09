"""Qwen-family entry point for chunk-wise CaM PPL evaluation.

GQA query-head attention is reduced to KV heads in the shared implementation.
"""
from models.common.cam import repair_cam_chunkwise, reset_cam_state


def Repair(model, past_key_values, attention, hidden_states, compress_ratio):
    del hidden_states
    return repair_cam_chunkwise(model, past_key_values, attention, compress_ratio)


__all__ = ["Repair", "reset_cam_state"]
