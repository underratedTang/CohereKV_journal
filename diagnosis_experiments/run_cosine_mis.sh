#!/usr/bin/env bash
set -euo pipefail

# Edit parameters directly in this file (no environment-variable overrides).
cd /home/tlxr/CohereKV
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

DIAG_DIR="/home/tlxr/CohereKV/diagnosis_experiments"
CALIBRATE="${DIAG_DIR}/calibrate_median_reference.py"
DIAGNOSE="${DIAG_DIR}/diagnose_cosine_mis.py"
PLOT="${DIAG_DIR}/plot_cosine_mis.py"
PACKAGE="${DIAG_DIR}/package_cosine_mis.py"
RESULTS_ROOT="${DIAG_DIR}/results"

# ============================================================
# 1. Execution / model / evaluation workload
# ============================================================
MODEL="meta-llama/Llama-3.2-3B"
DATASET="wikitext2"
TORCH_DTYPE="bfloat16"
DEVICE="cuda"
DEVICE_MAP="auto"
ATTN_IMPL="eager"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
PREFILL_CHUNK_SIZE=128
QUICK_MODE=1
RUN_CALIBRATION=0
RUN_ANALYSIS=1
RUN_PLOT=1

if [[ "${QUICK_MODE}" -eq 1 ]]; then
  EXPERIMENT_NAME="dual_routing_mis_median_realq_v4_quick"
  PREFIX_LEN=4096
  FUTURE_PROBE_LEN=64
  MAX_EVAL_SEQUENCES=2
  LAYER_SELECTION="0,8,16,24"
  KV_HEAD_SELECTION="0,1"
else
  EXPERIMENT_NAME="dual_routing_mis_median_realq_v4_full"
  PREFIX_LEN=8192
  FUTURE_PROBE_LEN=256
  MAX_EVAL_SEQUENCES=3
  LAYER_SELECTION="all"
  KV_HEAD_SELECTION="all"
fi

# KeepKV: ONE compression round, all original tokens have C=1, R=D=0.
KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=128
KEEPKV_SCORE_WINDOW=128
KEEPKV_RECENCY_BETA=1.0
KEEP_RATIOS=("0.2" "0.4" "0.6")

# ============================================================
# 2. Independent R-E Routing sweeps (shared by ALL THREE methods)
# ============================================================
# Select one or both. Each routing mode creates its own candidate groups.
ROUTING_MODES=("cosine" "l2")
# Cosine: nearest by cosine, eligible iff cosine >= threshold (NO L2 gate).
ROUTING_COSINE_THRESHOLDS=("0.70" "0.80")
# L2: nearest by physical Euclidean distance, eligible iff L2 <= cutoff.
# Calibrated cutoff = lambda_route * frozen S_calib[layer,KVhead].
ROUTING_L2_LAMBDAS=("0.50" "0.55" "0.60" "0.65" "0.70")
# Optional absolute Euclidean distances (same units as raw post-RoPE Keys).
# Leave empty to sweep only calibrated L2 routing lambdas.
ROUTING_L2_DISTANCES=()
# Routing sweep is entirely independent of E-E graph MIS threshold below.

# ============================================================
# 3. Independent OFFLINE Median reference calibration (train)
# ============================================================
CALIBRATION_SPLIT="train"
CALIBRATION_NUM_SEQUENCES=16
CALIBRATION_SEQUENCE_LENGTH=8192
CALIBRATION_SEED=42
# Per sequence: S[m,l,h] = RMS of pairwise physical Key L2 distances.
# Frozen: S_calib[l,h] = median_m S[m,l,h]. Test sequences never set S_calib.

# ============================================================
# 4. E-E native graph MIS: independent physical L2 cutoff
# ============================================================
MIS_EE_LAMBDAS=("0.10" "0.20" "0.30" "0.40" "0.50" "0.60" "0.70" "0.80" "0.90" "1.00")
# Conflict iff ||K_Ei-K_Ej||_2 > EE_LAMBDA * S_calib[layer,head].
DISTANCE_CHUNK_SIZE=256
# Memory-bounded CPU batches using the ORIGINAL degree-priority MIS.
# Avoids the GPU [num_anchor_pools, max_pool, max_pool] tensor OOM.
GRAPH_BACKEND="cpu_batched"

# ============================================================
# 5. Attention diagnosis and publication-quality figures
# ============================================================
SAVE_PER_PROBE=1
OUTPUT_NORM_EPS=1e-8
# One common focus value selects both summary charts and saved geometry.
FOCUS_EE_LAMBDA=0.50
# Detailed method/Layer/real-Q figures use one focus threshold per mode.
# Full routing and routing x EE sweeps are plotted for ALL thresholds.
FOCUS_ROUTING_COSINE_THRESHOLD=0.80
FOCUS_ROUTING_L2_LAMBDA=0.70
CAPTURE_LAYER=0
CAPTURE_HEAD=0
CAPTURE_BUDGET=0.2
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=1
PNG_DPI=600

# Qwen2.5 uses fewer KV heads than Llama; choose valid indices automatically.
# Qwen2.5-3B: 2 KV heads; Qwen2.5-7B: 4 KV heads.
# if [[ "${MODEL}" == "Qwen/Qwen2.5-3B" ]]; then
#   KV_HEAD_SELECTION="0,1"
#   CAPTURE_HEAD=1
# elif [[ "${MODEL}" == "Qwen/Qwen2.5-7B" ]]; then
#   if [[ "${QUICK_MODE}" -eq 1 ]]; then
#     KV_HEAD_SELECTION="0,2"
#   else
#     KV_HEAD_SELECTION="0,1,2,3"
#   fi
#   CAPTURE_HEAD=2
# fi

join_csv() { local IFS=,; echo "$*"; }
KEEP_RATIOS_CSV="$(join_csv "${KEEP_RATIOS[@]}")"
ROUTING_MODES_CSV="$(join_csv "${ROUTING_MODES[@]}")"
ROUTING_COSINE_CSV="$(join_csv "${ROUTING_COSINE_THRESHOLDS[@]}")"
ROUTING_L2_LAMBDAS_CSV="$(join_csv "${ROUTING_L2_LAMBDAS[@]}")"
ROUTING_L2_DISTANCES_CSV="$(join_csv "${ROUTING_L2_DISTANCES[@]}")"
MIS_EE_LAMBDAS_CSV="$(join_csv "${MIS_EE_LAMBDAS[@]}")"
MODEL_BASENAME="${MODEL%/}"; MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
CALIBRATION_FILE="${OUT_DIR}/calibration/median_reference.json"

# A new calibration starts a completely fresh experiment. With a pre-existing
# calibration, a rerun cleans ONLY numeric/figure outputs, preserving reference.
if [[ "${RUN_CALIBRATION}" -eq 1 && -d "${OUT_DIR}" ]]; then
  echo "[CLEAN] Removing previous output from THIS run: ${OUT_DIR}"
  rm -rf -- "${OUT_DIR}"
elif [[ "${RUN_ANALYSIS}" -eq 1 && -d "${OUT_DIR}" ]]; then
  find "${OUT_DIR}" -mindepth 1 -maxdepth 1 ! -name calibration -exec rm -rf -- {} +
fi
mkdir -p "${OUT_DIR}/calibration"
if [[ "${RUN_ANALYSIS}" -eq 1 && "${RUN_CALIBRATION}" -eq 0 && ! -s "${CALIBRATION_FILE}" ]]; then
  echo "[ERROR] RUN_CALIBRATION=0 but Median reference file does not exist: ${CALIBRATION_FILE}" >&2
  exit 1
fi
if [[ "${CALIBRATION_SPLIT}" != "train" ]]; then
  echo "[ERROR] Current experiment requires CALIBRATION_SPLIT=train" >&2
  exit 1
fi
FOCUS_FOUND=0
for EE_VAL in "${MIS_EE_LAMBDAS[@]}"; do
  if python -c 'import math,sys;sys.exit(0 if math.isclose(float(sys.argv[1]),float(sys.argv[2]),rel_tol=0,abs_tol=1e-9) else 1)' "${EE_VAL}" "${FOCUS_EE_LAMBDA}"; then
    FOCUS_FOUND=1
  fi
done
if [[ "${FOCUS_FOUND}" -ne 1 ]]; then
  echo "[ERROR] FOCUS_EE_LAMBDA must be included in MIS_EE_LAMBDAS" >&2
  exit 1
fi
RUN_LOG="${OUT_DIR}/run.log"
{
  echo "EXPERIMENT=${EXPERIMENT_NAME} MODEL=${MODEL} QUICK_MODE=${QUICK_MODE}"
  echo "CALIBRATION=train/${CALIBRATION_NUM_SEQUENCES}x${CALIBRATION_SEQUENCE_LENGTH} MEDIAN seed=${CALIBRATION_SEED}"
  echo "EVAL=test/${MAX_EVAL_SEQUENCES}x(${PREFIX_LEN}+${FUTURE_PROBE_LEN})"
  echo "ROUTING_MODES=${ROUTING_MODES_CSV} COSINE_THRESHOLDS=${ROUTING_COSINE_CSV} L2_LAMBDAS=${ROUTING_L2_LAMBDAS_CSV} L2_DISTANCES=${ROUTING_L2_DISTANCES_CSV}"
  echo "INDEPENDENT_EE_MIS_LAMBDAS=${MIS_EE_LAMBDAS_CSV}"
  echo "BUDGETS=${KEEP_RATIOS_CSV} LAYERS=${LAYER_SELECTION} HEADS=${KV_HEAD_SELECTION}"
  echo "METHODS=merge_all,mis_to_anchor,all_to_anchor; single round C=1 R=D=0"
  echo "FOCUS=${FOCUS_EE_LAMBDA} CAPTURE=${CAPTURE_LAYER}/${CAPTURE_HEAD}/${CAPTURE_BUDGET}"
  echo "RESULTS=${OUT_DIR}"
} | tee -a "${RUN_LOG}"

# With RUN_CALIBRATION=0, a fresh Median calibration file must already exist.
if [[ "${RUN_CALIBRATION}" -eq 1 ]]; then
  CMD=(python -u "${CALIBRATE}"
    --model "${MODEL}" --dataset "${DATASET}"
    --calibration_split "${CALIBRATION_SPLIT}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --output "${CALIBRATION_FILE}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[CALIBRATE]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  if [[ ! -s "${CALIBRATION_FILE}" ]]; then
    echo "[ERROR] Missing current model's frozen Median reference: ${CALIBRATION_FILE}" >&2
    exit 1
  fi
  CMD=(python -u "${DIAGNOSE}"
    --model "${MODEL}" --dataset "${DATASET}"
    --prefix_len "${PREFIX_LEN}" --future_probe_len "${FUTURE_PROBE_LEN}"
    --max_sequences "${MAX_EVAL_SEQUENCES}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --keep_ratios "${KEEP_RATIOS_CSV}"
    --routing_modes "${ROUTING_MODES_CSV}"
    --cosine_thresholds "${ROUTING_COSINE_CSV}"
    --routing_l2_lambdas "${ROUTING_L2_LAMBDAS_CSV}"
    --routing_l2_distances "${ROUTING_L2_DISTANCES_CSV}"
    --ee_lambdas "${MIS_EE_LAMBDAS_CSV}"
    --median_reference_file "${CALIBRATION_FILE}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --keepkv_sink_size "${KEEPKV_SINK_SIZE}"
    --keepkv_recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
    --keepkv_score_window "${KEEPKV_SCORE_WINDOW}"
    --keepkv_recency_beta "${KEEPKV_RECENCY_BETA}"
    --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
    --graph_backend "${GRAPH_BACKEND}"
    --save_per_probe "${SAVE_PER_PROBE}" --output_norm_eps "${OUTPUT_NORM_EPS}"
    --layer_selection "${LAYER_SELECTION}" --kv_head_selection "${KV_HEAD_SELECTION}"
    --capture_layer "${CAPTURE_LAYER}" --capture_head "${CAPTURE_HEAD}"
    --capture_budget "${CAPTURE_BUDGET}" --capture_ee_lambda "${FOCUS_EE_LAMBDA}"
    --output_dir "${OUT_DIR}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[ANALYZE]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

# Package numeric data BEFORE plotting in case Times New Roman is unavailable.
python -u "${PACKAGE}" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
if [[ "${RUN_PLOT}" -eq 1 ]]; then
  python -u "${PLOT}" --input_dir "${OUT_DIR}" \
    --font_family "${FONT_FAMILY}" \
    --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}" \
    --png_dpi "${PNG_DPI}" \
    --primary_ee_lambda "${FOCUS_EE_LAMBDA}" \
    --focus_cosine_threshold "${FOCUS_ROUTING_COSINE_THRESHOLD}" \
    --focus_l2_lambda "${FOCUS_ROUTING_L2_LAMBDA}" 2>&1 | tee -a "${RUN_LOG}"
fi
python -u "${PACKAGE}" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
echo "[UPLOAD] ${OUT_DIR}/cosine_mis_median_results.zip" | tee -a "${RUN_LOG}"
