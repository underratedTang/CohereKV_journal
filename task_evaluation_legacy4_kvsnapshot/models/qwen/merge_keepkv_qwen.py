"""Qwen adapter for the official-aligned KeepKV core."""
from models.common.keepkv_common import (
    KeepKVConfig,
    compress_keepkv,
    prepare_keepkv_oneshot_prefill,
    set_keepkv_oneshot_prefill_chunk,
    compress_keepkv_oneshot_prefill,
    get_keepkv_debug_state,
    reset_keepkv_state,
)


def _normalize_keepkv_config(config):
    if isinstance(config, KeepKVConfig):
        return config
    values = dict(config or {})

    # Backward-compatible aliases from the earlier unified evaluator. These
    # aliases only rename configuration keys; they do not alter KeepKV logic.
    aliases = {
        "recent_ratio": "pyramid_recent_ratio",
        "sink_size": "sink_len",
        "similarity_thr": "similarity_threshold",
        "prefill_similarity_thr": "prefill_similarity_threshold",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values.pop(old)
        else:
            values.pop(old, None)

    # The obsolete score_epsilon parameter represented an absolute epsilon.
    # Official KeepKV exposes epsilon_times instead. Silently dropping it would
    # be dangerous, so reject non-default use with an actionable error.
    if "score_epsilon" in values:
        old_eps = values.pop("score_epsilon")
        if old_eps not in (None, 1e-6):
            raise ValueError(
                "score_epsilon is not an official KeepKV parameter. Use "
                "keepkv_epsilon_times / epsilon_times instead."
            )

    allowed = set(KeepKVConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"Unknown KeepKV configuration fields: {unknown}")
    return KeepKVConfig(**values)


def Repair_keepkv(model, past_key_values, attention, hidden_states, compress_ratio, sim_thr=None, config=None):
    del hidden_states
    cfg = _normalize_keepkv_config(config)
    return compress_keepkv(model, past_key_values, attention, compress_ratio, sim_thr, cfg)



def prepare_longbench_keepkv(model, prompt_length, compress_ratio, config=None, similarity_chunk_size=256):
    cfg = _normalize_keepkv_config(config)
    return prepare_keepkv_oneshot_prefill(
        model, prompt_length, compress_ratio, cfg, similarity_chunk_size
    )


def set_longbench_keepkv_chunk(model, chunk_start):
    return set_keepkv_oneshot_prefill_chunk(model, chunk_start)


def finalize_longbench_keepkv(model, past_key_values, sim_thr=None):
    return compress_keepkv_oneshot_prefill(model, past_key_values, sim_thr)


__all__ = ["Repair_keepkv", "reset_keepkv_state", "get_keepkv_debug_state", "prepare_longbench_keepkv", "set_longbench_keepkv_chunk", "finalize_longbench_keepkv"]
