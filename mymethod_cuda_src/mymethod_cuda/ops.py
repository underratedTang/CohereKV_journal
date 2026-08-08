from __future__ import annotations
from typing import Optional
import torch

_EXT_ERROR: Optional[BaseException] = None
try:
    import mymethod_cuda_ext as _ext
except Exception as exc:  # CPU-only machines and source trees remain importable.
    _ext = None
    _EXT_ERROR = exc


def is_available() -> bool:
    return _ext is not None and torch.cuda.is_available()


def extension_error() -> Optional[BaseException]:
    return _EXT_ERROR


def count_calibrated_qk(q: torch.Tensor, k: torch.Tensor, logc: torch.Tensor) -> torch.Tensor:
    if _ext is None:
        raise RuntimeError(f"mymethod_cuda extension is unavailable: {_EXT_ERROR!r}")
    return _ext.count_calibrated_qk_q1(q, k, logc)


def compatible_route(
    evicted_keys: torch.Tensor, retained_keys: torch.Tensor,
    evicted_rhat: torch.Tensor, retained_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor, retained_diameter: torch.Tensor,
    delta: torch.Tensor, anchor_allowed: torch.Tensor,
):
    if _ext is None:
        raise RuntimeError(f"mymethod_cuda extension is unavailable: {_EXT_ERROR!r}")
    return _ext.compatible_route(
        evicted_keys, retained_keys, evicted_rhat, retained_rhat,
        evicted_diameter, retained_diameter, delta, anchor_allowed
    )


def graph_mis_select(
    evicted_keys: torch.Tensor,
    evicted_rhat: torch.Tensor,
    evicted_diameter: torch.Tensor,
    evicted_physical_idx: torch.Tensor,
    best_idx: torch.Tensor,
    route_compatible: torch.Tensor,
    anchor_gamma: torch.Tensor,
    keep_len: int,
    delta: torch.Tensor,
    group_mode: str = "full",
):
    if _ext is None:
        raise RuntimeError(f"mymethod_cuda extension is unavailable: {_EXT_ERROR!r}")
    mode = str(group_mode).strip().lower()
    if mode == "full":
        mode_id = 0
    elif mode == "anchor_only":
        mode_id = 1
    else:
        raise ValueError("CUDA graph_mis_select supports full or anchor_only")
    return _ext.graph_mis_select(
        evicted_keys,
        evicted_rhat,
        evicted_diameter,
        evicted_physical_idx,
        best_idx,
        route_compatible,
        anchor_gamma,
        int(keep_len),
        delta,
        mode_id,
    )
