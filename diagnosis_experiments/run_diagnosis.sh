#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CohereKV diagnosis experiment v1
# Original fixed-Delta C/R/D/Gamma + conflict graph + Parallel MIS
# ============================================================

cd /home/tlxr/CohereKV
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

ANALYSIS_SCRIPT="/home/tlxr/CohereKV/diagnosis_experiments/diagnose_original_coherekv.py"
PLOT_SCRIPT="/home/tlxr/CohereKV/diagnosis_experiments/plot_original_coherekv_diagnosis.py"
RESULTS_ROOT="/home/tlxr/CohereKV/diagnosis_experiments/results"
RUN_ANALYSIS=1
RUN_PLOT=1
QUICK_MODE=1

MODEL="meta-llama/Meta-Llama-3.1-8B"
DATASET="wikitext2"
PROBE_TOKENS=8
TORCH_DTYPE="bfloat16"
ATTN_IMPL="eager"
DEVICE="cuda"
DEVICE_MAP="auto"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
PREFILL_CHUNK_SIZE=128
DISTANCE_CHUNK_SIZE=1024
GEOMETRY_CHUNK_SIZE=512

KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=128
KEEPKV_SCORE_WINDOW=128
KEEPKV_RECENCY_BETA=1.0

# First-round diagnosis: before any C/R/D certificate accumulation.
SINGLE_ROUND_BUDGETS=("0.2" "0.4" "0.6" "0.8")

# Original global compactness coefficient in Delta = lambda * S_ref.
LAMBDA_VALUES=("0.5" "0.7" "0.9")

# Controlled repeated compression of one fixed prefix.
# The original CohereKV state advances each round; diagnostic shadows do not.
REPEATED_ROUNDS=5
ROUND_KEEP_RATIO=0.8
MIN_PHYSICAL_ENTRIES=256

if [[ "${QUICK_MODE}" -eq 1 ]]; then
  EXPERIMENT_NAME="original_coherekv_diagnosis_v1_quick"
  MODEL_SEQLEN=4096
  MAX_EVAL_SEQUENCES=1
  LAYER_SELECTION="0,8,16,24"
  KV_HEAD_SELECTION="0,2,4,6"
else
  EXPERIMENT_NAME="original_coherekv_diagnosis_v1_full"
  MODEL_SEQLEN=8192
  MAX_EVAL_SEQUENCES=3
  LAYER_SELECTION="all"
  KV_HEAD_SELECTION="all"
  REPEATED_ROUNDS=6
fi

ATTENTION_PROB_EPS=1e-30
OUTPUT_NORM_EPS=1e-20
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=0
PNG_DPI=600
PLOT_PRIMARY_LAMBDA=0.7

join_by_comma() { local IFS=,; echo "$*"; }
SINGLE_ROUND_BUDGETS_CSV="$(join_by_comma "${SINGLE_ROUND_BUDGETS[@]}")"
LAMBDA_VALUES_CSV="$(join_by_comma "${LAMBDA_VALUES[@]}")"
MODEL_BASENAME="${MODEL%/}"; MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${OUT_DIR}"
RUN_LOG="${OUT_DIR}/run.log"; : > "${RUN_LOG}"

{
  echo "QUICK_MODE=${QUICK_MODE}"
  echo "MODEL=${MODEL}"
  echo "MODEL_SEQLEN=${MODEL_SEQLEN}"
  echo "SINGLE_ROUND_BUDGETS=${SINGLE_ROUND_BUDGETS[*]}"
  echo "LAMBDA_VALUES=${LAMBDA_VALUES[*]}"
  echo "REPEATED_ROUNDS=${REPEATED_ROUNDS}"
  echo "ROUND_KEEP_RATIO=${ROUND_KEEP_RATIO}"
  echo "LAYER_SELECTION=${LAYER_SELECTION}"
  echo "KV_HEAD_SELECTION=${KV_HEAD_SELECTION}"
  echo "OUT_DIR=${OUT_DIR}"
  echo
} | tee -a "${RUN_LOG}"

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  CMD=(python -u "${ANALYSIS_SCRIPT}"
    --model "${MODEL}" --dataset "${DATASET}"
    --model_seqlen "${MODEL_SEQLEN}" --probe_tokens "${PROBE_TOKENS}"
    --max_eval_sequences "${MAX_EVAL_SEQUENCES}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}" --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --single_round_budgets "${SINGLE_ROUND_BUDGETS_CSV}"
    --lambda_values "${LAMBDA_VALUES_CSV}"
    --repeated_rounds "${REPEATED_ROUNDS}" --round_keep_ratio "${ROUND_KEEP_RATIO}"
    --min_physical_entries "${MIN_PHYSICAL_ENTRIES}"
    --keepkv_sink_size "${KEEPKV_SINK_SIZE}"
    --keepkv_recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
    --keepkv_score_window "${KEEPKV_SCORE_WINDOW}"
    --keepkv_recency_beta "${KEEPKV_RECENCY_BETA}"
    --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
    --geometry_chunk_size "${GEOMETRY_CHUNK_SIZE}"
    --layer_selection "${LAYER_SELECTION}" --kv_head_selection "${KV_HEAD_SELECTION}"
    --attention_prob_eps "${ATTENTION_PROB_EPS}" --output_norm_eps "${OUTPUT_NORM_EPS}"
    --progress_every_head --output_dir "${OUT_DIR}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[ANALYSIS CMD]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

if [[ "${RUN_PLOT}" -eq 1 ]]; then
  python -u "${PLOT_SCRIPT}" \
    --input_dir "${OUT_DIR}" \
    --font_family "${FONT_FAMILY}" \
    --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}" \
    --png_dpi "${PNG_DPI}" \
    --primary_lambda "${PLOT_PRIMARY_LAMBDA}" 2>&1 | tee -a "${RUN_LOG}"
fi

echo "[DONE] ${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "[CSV] ${OUT_DIR}/single_round_budget_diagnosis_overall.csv" | tee -a "${RUN_LOG}"
echo "[CSV] ${OUT_DIR}/repeated_round_diagnosis_overall.csv" | tee -a "${RUN_LOG}"
echo "[FIGURE] ${OUT_DIR}/figures/original_coherekv_vs_eviction_only_attention_error_by_round.pdf" | tee -a "${RUN_LOG}"
