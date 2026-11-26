#!/usr/bin/env python3
"""KeepKV retained/evicted split shared by all PPL methods."""
from __future__ import annotations
from pathlib import Path
import sys
import torch

TASK_ROOT = Path(__file__).resolve().parents[1] / 'task_evaluation'
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
from models.common.mymethod_common import select_keep_indices, _complement_indices


def select_split(keys, importance, positions, keep_len, sink_size, recent_window):
    if (keys.ndim != 3 or importance.shape != keys.shape[:2]
            or positions.shape != importance.shape):
        raise ValueError('Expected keys [X,L,D], importance/positions [X,L]')
    kept = select_keep_indices(importance, positions, keep_len,
                               sink_size=sink_size, recent_window=recent_window)
    return kept, _complement_indices(kept, keys.shape[1])
