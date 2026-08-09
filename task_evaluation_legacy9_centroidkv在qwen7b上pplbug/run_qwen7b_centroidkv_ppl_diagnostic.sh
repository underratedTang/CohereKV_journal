#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen-7B + CentroidKV mini-PPL diagnostic
#
# Purpose:
#   1) Compare Full KV vs CentroidKV on the exact mini-PPL protocol.
#   2) Trace CentroidKV physical KV length before/target/after every compression.
#   3) Check a short-sequence CentroidKV SDPA-vs-eager numerical control.
#   4) Check a short-sequence zero-compression CentroidKV runtime control.
#
# This script DOES NOT modify evaluate_ppl_kvmerge.py.
# It creates a temporary Python wrapper under OUT_DIR only.
# ============================================================

# =========================
# 手动修改区
# =========================

PROJECT_ROOT="/home/tlxr/SimCalKV"
TASK_EVAL_DIR="${PROJECT_ROOT}/task_evaluation"
PPL_SCRIPT="${TASK_EVAL_DIR}/evaluate_ppl_kvmerge.py"

# 必须改成你实际出现异常的 Qwen-7B 模型路径或 HuggingFace ID。
# 不要用别的 7B 模型替代。
# MODEL="__SET_YOUR_ACTUAL_QWEN7B_MODEL__"
MODEL="Qwen/Qwen2.5-7B"
FAMILY="qwen"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 你的当前 mini-PPL 主协议
MODEL_SEQLEN=8192
CHUNK_SIZE=32
MIN_KV_LEN=2
COMPRESS_EVERY=1
MAX_EVAL_SEQUENCES=3
DATASET="wikitext2"

# 改成你观察到异常优势时使用的压缩率。
# 例如 0.4 表示压缩 40%，保留 60%。
COMPRESS_RATIO="0.4"

# CentroidKV 当前正式配置：诊断时不要改这些值。
CENTROIDKV_SINK_BUDGET=16
CENTROIDKV_LOCAL_BUDGET=64
CENTROIDKV_CHUNK_SIZE=256
CENTROIDKV_MERGE_RATIO=0.8
CENTROIDKV_DECREASE_RATE=0.2

TORCH_DTYPE="bfloat16"
AMP_DTYPE="none"
DEVICE="cuda"
DEVICE_MAP="auto"
SEED=0

# 短序列 runtime 对照。2048 x 1 用于避免 eager 控制实验占用过多显存。
CONTROL_SEQLEN=2048
CONTROL_MAX_EVAL_SEQUENCES=1

# 运行开关：1=运行，0=跳过。
RUN_MAIN_FULL=1
RUN_MAIN_CENTROID=1
RUN_ZERO_RATIO_CONTROL=1
RUN_SDPA_EAGER_CONTROL=1

# CentroidKV trace 每 N 次 maybe_compress 打一行；1 表示全部打印。
TRACE_EVERY=1

# 输出目录。每次自动建立新的时间戳子目录，不覆盖旧结果。
OUT_ROOT="${PROJECT_ROOT}/results/qwen7b_centroidkv_ppl_diagnostic"

# =========================
# 自动区：一般不用改
# =========================

if [[ "${MODEL}" == "__SET_YOUR_ACTUAL_QWEN7B_MODEL__" ]]; then
  echo "[ERROR] 请先把 MODEL 改成你实际出现异常的 Qwen-7B 模型路径/HF ID。" >&2
  exit 2
fi

if [[ ! -f "${PPL_SCRIPT}" ]]; then
  echo "[ERROR] PPL evaluator not found: ${PPL_SCRIPT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_SLUG="$(printf '%s' "${MODEL}" | tr '/:' '__')"
OUT_DIR="${OUT_ROOT}/${MODEL_SLUG}_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
JSON_DIR="${OUT_DIR}/json"
mkdir -p "${LOG_DIR}" "${JSON_DIR}"

RUN_LOG="${LOG_DIR}/run_all.log"
CONFIG_LOG="${OUT_DIR}/config.txt"
WRAPPER="${OUT_DIR}/centroidkv_ppl_diag_wrapper.py"
SUMMARY="${OUT_DIR}/summary.txt"

cat > "${CONFIG_LOG}" <<EOF
MODEL=${MODEL}
FAMILY=${FAMILY}
MODEL_SEQLEN=${MODEL_SEQLEN}
CHUNK_SIZE=${CHUNK_SIZE}
MIN_KV_LEN=${MIN_KV_LEN}
COMPRESS_EVERY=${COMPRESS_EVERY}
MAX_EVAL_SEQUENCES=${MAX_EVAL_SEQUENCES}
DATASET=${DATASET}
COMPRESS_RATIO=${COMPRESS_RATIO}
CENTROIDKV_SINK_BUDGET=${CENTROIDKV_SINK_BUDGET}
CENTROIDKV_LOCAL_BUDGET=${CENTROIDKV_LOCAL_BUDGET}
CENTROIDKV_CHUNK_SIZE=${CENTROIDKV_CHUNK_SIZE}
CENTROIDKV_MERGE_RATIO=${CENTROIDKV_MERGE_RATIO}
CENTROIDKV_DECREASE_RATE=${CENTROIDKV_DECREASE_RATE}
TORCH_DTYPE=${TORCH_DTYPE}
AMP_DTYPE=${AMP_DTYPE}
CONTROL_SEQLEN=${CONTROL_SEQLEN}
CONTROL_MAX_EVAL_SEQUENCES=${CONTROL_MAX_EVAL_SEQUENCES}
TRACE_EVERY=${TRACE_EVERY}
EOF

# ------------------------------------------------------------
# 先记录模型配置。只读取 config，不加载模型权重。
# ------------------------------------------------------------
{
  echo "========== MODEL CONFIG =========="
  python - "${MODEL}" <<'PY'
import json
import sys
from transformers import AutoConfig

model_name = sys.argv[1]
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
fields = [
    "model_type",
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "torch_dtype",
]
result = {k: getattr(cfg, k, None) for k in fields}
q = result.get("num_attention_heads")
kv = result.get("num_key_value_heads")
result["gqa_query_per_kv_head"] = (q / kv) if q and kv else None
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
PY
  echo
} | tee -a "${RUN_LOG}"

# ------------------------------------------------------------
# 临时诊断 wrapper：
# - monkeypatch maybe_compress，仅打印状态，不改变 compression 逻辑；
# - 可选绕过 v4 对 CentroidKV eager -> sdpa 的自动替换，仅用于 2048 控制实验。
# ------------------------------------------------------------
cat > "${WRAPPER}" <<'PY'
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
PY

run_eval() {
  local name="$1"
  local method="$2"
  local ratio="$3"
  local seqlen="$4"
  local maxseq="$5"
  local attn_impl="$6"
  local force_requested_attn="$7"

  local log_file="${LOG_DIR}/${name}.log"
  local json_file="${JSON_DIR}/${name}.json"

  echo "========== RUN ${name} ==========" | tee -a "${RUN_LOG}"
  echo "method=${method} ratio=${ratio} seqlen=${seqlen} maxseq=${maxseq} attn=${attn_impl} force=${force_requested_attn}" | tee -a "${RUN_LOG}"

  CKV_DIAG_TASK_EVAL_DIR="${TASK_EVAL_DIR}" \
  CKV_DIAG_TRACE_EVERY="${TRACE_EVERY}" \
  CKV_DIAG_FORCE_REQUESTED_ATTN="${force_requested_attn}" \
  python "${WRAPPER}" \
    --model_name "${MODEL}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --family "${FAMILY}" \
    --datasets "${DATASET}" \
    --kv_method "${method}" \
    --compress_ratio "${ratio}" \
    --model_seqlen "${seqlen}" \
    --chunk_size "${CHUNK_SIZE}" \
    --min_kv_len "${MIN_KV_LEN}" \
    --compress_every "${COMPRESS_EVERY}" \
    --max_eval_sequences "${maxseq}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
    --seed "${SEED}" \
    --attn_implementation "${attn_impl}" \
    --centroidkv_sink_budget "${CENTROIDKV_SINK_BUDGET}" \
    --centroidkv_local_budget "${CENTROIDKV_LOCAL_BUDGET}" \
    --centroidkv_chunk_size "${CENTROIDKV_CHUNK_SIZE}" \
    --centroidkv_merge_ratio "${CENTROIDKV_MERGE_RATIO}" \
    --centroidkv_decrease_rate "${CENTROIDKV_DECREASE_RATE}" \
    --output_json "${json_file}" \
    2>&1 | tee "${log_file}"

  echo "[DONE] ${name}: ${json_file}" | tee -a "${RUN_LOG}"
  echo | tee -a "${RUN_LOG}"
}

# ============================================================
# A. 与你当前 mini-PPL 完全一致的主对照
# ============================================================
if [[ "${RUN_MAIN_FULL}" == "1" ]]; then
  run_eval \
    "A_full_main" \
    "none" \
    "0.0" \
    "${MODEL_SEQLEN}" \
    "${MAX_EVAL_SEQUENCES}" \
    "sdpa" \
    "0"
fi

if [[ "${RUN_MAIN_CENTROID}" == "1" ]]; then
  run_eval \
    "B_centroid_main_sdpa" \
    "CentroidKV" \
    "${COMPRESS_RATIO}" \
    "${MODEL_SEQLEN}" \
    "${MAX_EVAL_SEQUENCES}" \
    "sdpa" \
    "0"
fi

# ============================================================
# B. runtime-only control：CentroidKV method path，但 ratio=0
#    用于检查仅切换到 CentroidKV runtime/cache wrapper 是否已改变 PPL。
# ============================================================
if [[ "${RUN_ZERO_RATIO_CONTROL}" == "1" ]]; then
  run_eval \
    "C_full_control_2048_sdpa" \
    "none" \
    "0.0" \
    "${CONTROL_SEQLEN}" \
    "${CONTROL_MAX_EVAL_SEQUENCES}" \
    "sdpa" \
    "0"

  run_eval \
    "D_centroid_ratio0_control_2048_sdpa" \
    "CentroidKV" \
    "0.0" \
    "${CONTROL_SEQLEN}" \
    "${CONTROL_MAX_EVAL_SEQUENCES}" \
    "sdpa" \
    "0"
fi

# ============================================================
# C. CentroidKV SDPA vs eager control：相同 compression，相同 2048 x 1。
#    eager 只在此短控制实验中通过 wrapper 绕过自动 eager->sdpa 替换。
# ============================================================
if [[ "${RUN_SDPA_EAGER_CONTROL}" == "1" ]]; then
  run_eval \
    "E_centroid_control_2048_sdpa" \
    "CentroidKV" \
    "${COMPRESS_RATIO}" \
    "${CONTROL_SEQLEN}" \
    "${CONTROL_MAX_EVAL_SEQUENCES}" \
    "sdpa" \
    "0"

  run_eval \
    "F_centroid_control_2048_eager" \
    "CentroidKV" \
    "${COMPRESS_RATIO}" \
    "${CONTROL_SEQLEN}" \
    "${CONTROL_MAX_EVAL_SEQUENCES}" \
    "eager" \
    "1"
fi

# ------------------------------------------------------------
# 汇总 JSON + trace 关键统计。
# ------------------------------------------------------------
python - "${JSON_DIR}" "${LOG_DIR}" > "${SUMMARY}" <<'PY'
import glob
import json
import os
import sys

json_dir, log_dir = sys.argv[1], sys.argv[2]

rows = {}
for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        obj = obj[0]
    name = os.path.splitext(os.path.basename(path))[0]
    rows[name] = obj

print("=== PPL SUMMARY ===")
for name, obj in rows.items():
    print(
        f"{name}: "
        f"ppl={obj.get('ppl')} "
        f"method={obj.get('kv_method')} "
        f"ratio={obj.get('compress_ratio')} "
        f"attn={obj.get('attn_implementation')} "
        f"num_compressions={obj.get('num_compressions')} "
        f"avg_compressed_kv_len={obj.get('avg_compressed_kv_len')} "
        f"num_eval_sequences={obj.get('num_eval_sequences')}"
    )


def ppl(name):
    v = rows.get(name, {}).get("ppl")
    return None if v is None else float(v)

print("\n=== KEY DIFFERENCES ===")
pairs = [
    ("A_full_main", "B_centroid_main_sdpa", "main Full -> Centroid"),
    ("C_full_control_2048_sdpa", "D_centroid_ratio0_control_2048_sdpa", "2048 runtime-only Full -> Centroid ratio0"),
    ("E_centroid_control_2048_sdpa", "F_centroid_control_2048_eager", "2048 Centroid SDPA -> eager"),
]
for a, b, label in pairs:
    pa, pb = ppl(a), ppl(b)
    if pa is None or pb is None:
        continue
    rel = (pb - pa) / pa if pa != 0 else float("nan")
    print(f"{label}: {pa:.10f} -> {pb:.10f}; delta={pb-pa:+.10f}; rel={rel:+.6%}")

print("\n=== CENTROID TRACE CHECK ===")
for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    name = os.path.splitext(os.path.basename(path))[0]
    traces = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("[CKV_DIAG] "):
                continue
            try:
                traces.append(json.loads(line[len("[CKV_DIAG] "):]))
            except Exception:
                pass
    if not traces:
        continue
    called = [x for x in traces if x.get("compression_called") == 1]
    errors = [int(x.get("after_minus_target", 0)) for x in called]
    layer_mismatch = [
        x for x in called
        if x.get("after_layer_min") != x.get("after_layer_max")
    ]
    print(
        f"{name}: trace_rows={len(traces)} compression_rows={len(called)} "
        f"after-target[min,max]={((min(errors), max(errors)) if errors else None)} "
        f"layer_len_mismatch_rows={len(layer_mismatch)}"
    )
    if called:
        print("  first_compression=" + json.dumps(called[0], ensure_ascii=False, sort_keys=True))
        print("  last_compression=" + json.dumps(called[-1], ensure_ascii=False, sort_keys=True))

print("\n=== INTERPRETATION FLAGS ===")
a = ppl("A_full_main")
b = ppl("B_centroid_main_sdpa")
if a is not None and b is not None:
    if abs(b - a) / max(abs(a), 1e-12) < 0.002:
        print("FLAG: Centroid main PPL is within 0.2% of Full. Check CKV trace carefully for effective compression/budget behavior.")

c = ppl("C_full_control_2048_sdpa")
d = ppl("D_centroid_ratio0_control_2048_sdpa")
if c is not None and d is not None and abs(d-c) / max(abs(c), 1e-12) > 0.002:
    print("FLAG: ratio=0 Centroid runtime differs from Full by >0.2%; investigate runtime/cache protocol before interpreting compression quality.")

e = ppl("E_centroid_control_2048_sdpa")
f = ppl("F_centroid_control_2048_eager")
if e is not None and f is not None and abs(f-e) / max(abs(e), 1e-12) > 0.002:
    print("FLAG: Centroid eager vs SDPA differs by >0.2% at 2048; investigate attention/runtime semantics.")
PY

cat "${SUMMARY}" | tee -a "${RUN_LOG}"

echo
echo "Diagnostic finished."
echo "OUT_DIR=${OUT_DIR}"
echo "Please return these files:"
echo "  1) ${SUMMARY}"
echo "  2) ${CONFIG_LOG}"
echo "  3) ${LOG_DIR}/B_centroid_main_sdpa.log"
echo "  4) ${LOG_DIR}/F_centroid_control_2048_eager.log (if generated)"
echo "  5) JSON files under ${JSON_DIR}/"
