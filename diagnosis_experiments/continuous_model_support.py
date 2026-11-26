#!/usr/bin/env python3
"""Model reset and frozen TRAIN calibration validation for continuous PPL."""
import json
from pathlib import Path
import numpy as np
from calibrate_median_reference import validate_payload

def prepare_model_for_new_sequence(model,score_window,chunk_size):
    for layer in model.model.layers:
        attn=layer.self_attn
        attn._mymethod_counts_kv=None
        attn._mymethod_eager_enabled=False
        attn._mymethod_sdpa_enabled=False
        attn._keepkv_eager_enabled=False
        attn._mymethod_selector='keepkv'
        attn._mymethod_score_window=int(score_window)
        attn._kvmerge_attention_chunk_size=int(chunk_size)
        attn._kvmerge_capture_attention_scores=False
        for field in ('_kvmerge_captured_attention_score','_kvmerge_captured_score_log_scale'):
            if hasattr(attn,field):delattr(attn,field)


def load_frozen_calibration(path,model_name,dataset=None):
    info=json.loads(Path(path).read_text())
    scales=validate_payload(info,model=model_name,split='train',dataset=dataset)
    return info,scales
