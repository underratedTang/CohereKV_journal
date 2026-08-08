"""Independent CUDA backend for mymethod.

This package deliberately contains no GroupLogitKV imports or dependencies.
The public API is small so task_evaluation can keep a PyTorch reference path:

* count_calibrated_qk(q, k, logc)
* graph_mis_select(...)
* is_available()
"""
from .ops import count_calibrated_qk, compatible_route, graph_mis_select, is_available, extension_error

__all__ = [
    "count_calibrated_qk",
    "compatible_route",
    "graph_mis_select",
    "is_available",
    "extension_error",
]
