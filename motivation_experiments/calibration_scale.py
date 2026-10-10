from __future__ import annotations

"""Load frozen per-(layer, KV-head) K scales calibrated once before evaluation."""

import json
import math
from pathlib import Path
from typing import Optional

import torch


def load_calibration_scales(path: str, expected_model: Optional[str] = None) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_model is not None and payload.get("model") not in (None, expected_model):
        raise ValueError(
            f"calibration-scale model mismatch: {payload.get('model')!r} != {expected_model!r}"
        )
    rows = payload.get("layer_head_scales")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) and r for r in rows):
        raise ValueError(f"missing/non-rectangular layer_head_scales in {path}")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"non-rectangular layer_head_scales in {path}")
    scales = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError(f"invalid non-positive/non-finite calibration scale in {path}")
    return scales.contiguous()


def scale_for(scales: torch.Tensor, layer_idx: int, head_idx: int) -> float:
    if scales.ndim != 2:
        raise ValueError("calibration scales must be [layers, kv_heads]")
    li, hi = int(layer_idx), int(head_idx)
    if li < 0 or li >= scales.shape[0] or hi < 0 or hi >= scales.shape[1]:
        raise IndexError(f"calibration scale missing for layer={li}, kv_head={hi}; shape={tuple(scales.shape)}")
    value = float(scales[li, hi].item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid calibration scale at layer={li}, kv_head={hi}: {value}")
    return value


def validate_calibration_scales(scales: torch.Tensor, config) -> None:
    expected_layers = int(getattr(config, "num_hidden_layers"))
    expected_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    got = tuple(int(x) for x in scales.shape)
    expected = (expected_layers, expected_heads)
    if got != expected:
        raise ValueError(f"calibration scale shape mismatch: got {got}, expected {expected}")


def batched_head_scales(scales: torch.Tensor, layer_idx: int, batch_size: int, *, device=None) -> torch.Tensor:
    """Return [B*Hkv] frozen scales aligned with keys.reshape(B*Hkv, L, D)."""
    if scales.ndim != 2:
        raise ValueError("calibration scales must be [layers, kv_heads]")
    li = int(layer_idx)
    if li < 0 or li >= int(scales.shape[0]):
        raise IndexError(li)
    row = scales[li].detach().float()
    if device is not None:
        row = row.to(device)
    return row.view(1, -1).expand(int(batch_size), -1).reshape(-1).contiguous()
