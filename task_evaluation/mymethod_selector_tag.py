#!/usr/bin/env python3
"""Canonical active-selector tags for mymethod experiment logs/results.

Only hyperparameters that affect the selected retained/evicted partition are
included.  Inactive selector-family parameters are deliberately omitted so
that two tags differ iff their active selector configuration differs.
"""
from __future__ import annotations

from typing import Any, Dict


def _bool01(value: Any) -> int:
    return 1 if bool(value) else 0


def _float_tag(value: Any) -> str:
    text = format(float(value), ".8g")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def active_mymethod_selector_config(
    selector: str,
    *,
    keepkv_sink_size: int = 4,
    keepkv_recent_protect_size: int = 128,
    keepkv_score_window: int = 128,
    keepkv_recency_beta: float = 1.0,
    keepkv_use_log_count: bool = False,
    d2o_sink_size: int = 4,
    d2o_recent_size: int = 128,
    d2o_use_log_count: bool = True,
    d2o_score_reset_interval: int = 0,
) -> Dict[str, Any]:
    selector = str(selector).strip().lower()
    if selector == "attention":
        return {"selector": "attention"}
    if selector == "keepkv":
        return {
            "selector": "keepkv",
            "sink_size": int(keepkv_sink_size),
            "recent_protect_size": int(keepkv_recent_protect_size),
            "score_window": int(keepkv_score_window),
            "recency_beta": float(keepkv_recency_beta),
            "use_log_count": bool(keepkv_use_log_count),
        }
    if selector == "d2o":
        return {
            "selector": "d2o",
            "sink_size": int(d2o_sink_size),
            "recent_size": int(d2o_recent_size),
            "use_log_count": bool(d2o_use_log_count),
            "score_reset_interval": int(d2o_score_reset_interval),
        }
    raise ValueError(f"Unsupported mymethod selector: {selector!r}")


def build_mymethod_selector_tag(selector: str, **kwargs: Any) -> str:
    cfg = active_mymethod_selector_config(selector, **kwargs)
    if cfg["selector"] == "attention":
        return "selector_attention"
    if cfg["selector"] == "keepkv":
        return (
            "selector_keepkv"
            f"__sink{cfg['sink_size']}"
            f"__recent{cfg['recent_protect_size']}"
            f"__window{cfg['score_window']}"
            f"__beta{_float_tag(cfg['recency_beta'])}"
            f"__logc{_bool01(cfg['use_log_count'])}"
        )
    return (
        "selector_d2o"
        f"__sink{cfg['sink_size']}"
        f"__recent{cfg['recent_size']}"
        f"__logc{_bool01(cfg['use_log_count'])}"
        f"__reset{cfg['score_reset_interval']}"
    )
