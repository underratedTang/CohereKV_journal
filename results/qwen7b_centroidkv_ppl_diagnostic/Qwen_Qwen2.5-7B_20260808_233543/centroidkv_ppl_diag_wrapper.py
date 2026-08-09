import json
import os
import sys

TASK_EVAL_DIR = os.environ["CKV_DIAG_TASK_EVAL_DIR"]
if TASK_EVAL_DIR not in sys.path:
    sys.path.insert(0, TASK_EVAL_DIR)

import evaluate_ppl_kvmerge as ev

force_requested = os.environ.get("CKV_DIAG_FORCE_REQUESTED_ATTN", "0") == "1"
trace_every = max(1, int(os.environ.get("CKV_DIAG_TRACE_EVERY", "1")))

if force_requested:
    _orig_effective = ev.effective_attn_implementation

    def _diag_effective_attn_implementation(method: str, requested: str) -> str:
        # Diagnostic only: allow CentroidKV+eager on short sequences.
        # No baseline source/runtime patch is installed by this override.
        if method == "CentroidKV":
            return str(requested)
        return _orig_effective(method, requested)

    ev.effective_attn_implementation = _diag_effective_attn_implementation

_orig_maybe_compress = ev.KVCompressingDynamicCache.maybe_compress


def _layer_lens(cache):
    legacy = ev._as_legacy_tuple(cache)
    if not legacy:
        return []
    out = []
    for layer in legacy:
        if layer is None or len(layer) < 1 or layer[0] is None:
            continue
        out.append(int(layer[0].shape[-2]))
    return out


def _traced_maybe_compress(self, attentions, hidden_states):
    is_ckv = getattr(self, "_kv_method", None) == "CentroidKV"
    if not is_ckv:
        return _orig_maybe_compress(self, attentions, hidden_states)

    before = int(ev._legacy_kv_len(self))
    before_layers = _layer_lens(self)
    logical_seen = int(getattr(self, "_ppl_logical_seen_tokens", 0))
    ratio = float(getattr(self, "_compress_ratio", 0.0))
    retained_ratio = 1.0 - ratio
    target_raw = max(1, int(round(logical_seen * retained_ratio)))

    cfg = dict(getattr(self, "_centroidkv_config", {}) or {})
    sink = int(cfg.get("sink_budget", 16))
    local = int(cfg.get("local_budget", 64))
    floor = min(before, sink + local)
    target_expected = max(target_raw, floor)

    n0 = int(getattr(self, "_stats", {}).get("num_compressions", 0))
    _orig_maybe_compress(self, attentions, hidden_states)
    n1 = int(getattr(self, "_stats", {}).get("num_compressions", 0))

    after = int(ev._legacy_kv_len(self))
    after_layers = _layer_lens(self)
    call_idx = int(getattr(self, "_forward_calls", 0))

    if call_idx % trace_every == 0 or n1 != n0 or before != after:
        row = {
            "event": "CKV_COMPRESS_TRACE",
            "forward_call": call_idx,
            "logical_seen": logical_seen,
            "before": before,
            "target_raw": target_raw,
            "protected_floor": floor,
            "target_expected": target_expected,
            "after": after,
            "removed": before - after,
            "compression_called": int(n1 != n0),
            "num_compressions": n1,
            "before_layer_min": min(before_layers) if before_layers else None,
            "before_layer_max": max(before_layers) if before_layers else None,
            "after_layer_min": min(after_layers) if after_layers else None,
            "after_layer_max": max(after_layers) if after_layers else None,
            "after_minus_target": after - target_expected,
            "actual_physical_over_logical": (after / logical_seen) if logical_seen > 0 else None,
            "expected_retained_ratio": retained_ratio,
        }
        print("[CKV_DIAG] " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


ev.KVCompressingDynamicCache.maybe_compress = _traced_maybe_compress

ev.main()
