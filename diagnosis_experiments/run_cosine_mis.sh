#!/usr/bin/env bash
set -euo pipefail

# Edit ONLY literal settings below. No external environment-variable inputs for experiments.
# These exports are required runtime settings, NOT experiment-parameter overrides.
# Run this file from any working directory.
ROOT_DIR="/home/tlxr/CohereKV"
cd "${ROOT_DIR}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
DIAG_DIR="${ROOT_DIR}/diagnosis_experiments"
RESULTS_ROOT="${DIAG_DIR}/results"

# ============================================================
# 1. Model / execution
# ============================================================
# MODEL="meta-llama/Llama-3.2-1B"
MODEL="meta-llama/Llama-3.2-3B"
# MODEL="Qwen/Qwen2.5-3B"  # check valid 32K model context before 4x16K
# MODEL="Qwen/Qwen2.5-7B"
TORCH_DTYPE="bfloat16"
DEVICE="cuda"
DEVICE_MAP="auto"
ATTN_IMPL="eager"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
QUICK_MODE=1
RUN_CALIBRATION=1
RUN_PPL=1
RUN_PLOT=1

# ============================================================
# 2. Continuous PPL (real token stream; KV cache persists)
# ============================================================
PPL_DATASET="wikitext2"
PPL_CYCLES=24
PPL_UNSCORED_LEN=1024
PPL_SCORED_LEN=32
PPL_PREFILL_CHUNK_SIZE=128
PPL_BUDGETS=("0.10")
PPL_P_VALUES=("2")
PPL_LAMBDA_EA=("0.60")
PPL_LAMBDA_EE=("0.60")
# Same RMS Key-L2 scale for ALL p. 'online_first' matches original
# conference first-compression calibration; 'frozen_train' is an ablation.
PPL_REFERENCE_MODE="frozen_train"
# KeepKV mandatory slot protection. Change any nonnegative integer directly here.
# SINK_SIZE protects original logical positions [0, SINK_SIZE).
# RECENT_PROTECT_SIZE keeps the most recent logical positions within remaining budget.
# SCORE_WINDOW is the number of recent attention queries used for importance;
# it is NOT the number of protected slots.
KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=32
KEEPKV_SCORE_WINDOW=32
KEEPKV_RECENCY_BETA=1.0
DISTANCE_CHUNK_SIZE=256
GRAPH_BACKEND="streamed"

# ============================================================
# 3. TRAIN calibration (also required for frozen sink-head masks)
# ============================================================
CALIBRATION_SPLIT="train"
CALIBRATION_SEED=42
CALIBRATION_NUM_SEQUENCES=4
CALIBRATION_SEQUENCE_LENGTH=4096
SINK_LENGTH=1
SINK_DIAGNOSTIC_SEQLEN=4096
SINK_PROBE_LEN=64
SINK_PRESENCE_THRESHOLD=0.05
QUERY_NORM_SAFETY_FACTOR=1.05

# ============================================================
# 4. Independent MIS switches (literal configuration; 1=run, 0=skip)
#    Each original MIS method has a paired *_wo_ea_certificate control.
#    For each pair ONLY E-A routing differs; E-E and MIS priority/fusion
#    (including AllCount, importance-weighted and sink-gated options) match.
#    FullCache / physical MergeAll are ALWAYS included.
# ============================================================
RUN_COHEREKV_WO_MIS=1
RUN_IMPORTANCE_WEIGHTED_MERGE_ALL=0
RUN_K_ONLY_MIS_DROP=1
RUN_K_ONLY_MIS_DROP_WO_EA_CERTIFICATE=1

RUN_K_ONLY_MIS_DROP_ALLCOUNT=0
RUN_K_ONLY_MIS_DROP_ALLCOUNT_WO_EA_CERTIFICATE=0
RUN_K_ONLY_MIS_DROP_SINK=0
RUN_K_ONLY_MIS_DROP_SINK_WO_EA_CERTIFICATE=0
RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK=0
RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE=0
RUN_K_ONLY_DEGREE_ANCHOR_MIS_DROP=0
RUN_K_ONLY_DEGREE_ANCHOR_MIS_DROP_WO_EA_CERTIFICATE=0
RUN_K_ONLY_ANCHOR_FIRST_MIS_DROP=0
RUN_K_ONLY_ANCHOR_FIRST_MIS_DROP_WO_EA_CERTIFICATE=0

RUN_K_ONLY_IMPORTANCE_MIS_DROP=1
RUN_K_ONLY_IMPORTANCE_MIS_DROP_WO_EA_CERTIFICATE=1

RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP=0
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_SINK=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_SINK_WO_EA_CERTIFICATE=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_SINK=0
RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE=0

METHOD_SWITCHES=(
  "coherekv_wo_mis:${RUN_COHEREKV_WO_MIS}"
  "importance_weighted_merge_all:${RUN_IMPORTANCE_WEIGHTED_MERGE_ALL}"
  "k_only_mis_drop:${RUN_K_ONLY_MIS_DROP}"
  "k_only_mis_drop_wo_ea_certificate:${RUN_K_ONLY_MIS_DROP_WO_EA_CERTIFICATE}"
  "k_only_mis_drop_allcount:${RUN_K_ONLY_MIS_DROP_ALLCOUNT}"
  "k_only_mis_drop_allcount_wo_ea_certificate:${RUN_K_ONLY_MIS_DROP_ALLCOUNT_WO_EA_CERTIFICATE}"
  "k_only_mis_drop_sink:${RUN_K_ONLY_MIS_DROP_SINK}"
  "k_only_mis_drop_sink_wo_ea_certificate:${RUN_K_ONLY_MIS_DROP_SINK_WO_EA_CERTIFICATE}"
  "k_only_mis_drop_allcount_sink:${RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK}"
  "k_only_mis_drop_allcount_sink_wo_ea_certificate:${RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE}"
  "k_only_degree_anchor_mis_drop:${RUN_K_ONLY_DEGREE_ANCHOR_MIS_DROP}"
  "k_only_degree_anchor_mis_drop_wo_ea_certificate:${RUN_K_ONLY_DEGREE_ANCHOR_MIS_DROP_WO_EA_CERTIFICATE}"
  "k_only_anchor_first_mis_drop:${RUN_K_ONLY_ANCHOR_FIRST_MIS_DROP}"
  "k_only_anchor_first_mis_drop_wo_ea_certificate:${RUN_K_ONLY_ANCHOR_FIRST_MIS_DROP_WO_EA_CERTIFICATE}"
  "k_only_importance_mis_drop:${RUN_K_ONLY_IMPORTANCE_MIS_DROP}"
  "k_only_importance_mis_drop_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_WO_EA_CERTIFICATE}"
  "k_only_importance_mis_drop_allcount:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT}"
  "k_only_importance_mis_drop_allcount_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_WO_EA_CERTIFICATE}"
  "k_only_importance_mis_drop_sink:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK}"
  "k_only_importance_mis_drop_sink_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK_WO_EA_CERTIFICATE}"
  "k_only_importance_mis_drop_allcount_sink:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK}"
  "k_only_importance_mis_drop_allcount_sink_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE}"
  "k_only_importance_iwmean_drop:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP}"
  "k_only_importance_iwmean_drop_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_WO_EA_CERTIFICATE}"
  "k_only_importance_imean_drop:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP}"
  "k_only_importance_imean_drop_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_WO_EA_CERTIFICATE}"
  "k_only_importance_imean_drop_allcount:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT}"
  "k_only_importance_imean_drop_allcount_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_WO_EA_CERTIFICATE}"
  "k_only_importance_imean_drop_sink:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_SINK}"
  "k_only_importance_imean_drop_sink_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_SINK_WO_EA_CERTIFICATE}"
  "k_only_importance_imean_drop_allcount_sink:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_SINK}"
  "k_only_importance_imean_drop_allcount_sink_wo_ea_certificate:${RUN_K_ONLY_IMPORTANCE_IMEAN_DROP_ALLCOUNT_SINK_WO_EA_CERTIFICATE}"
)
MIS_METHODS_ENABLED=()
ACTIVE_FAMILIES=()
METHOD_INDEX_TAG=""
for i in "${!METHOD_SWITCHES[@]}"; do
  method="${METHOD_SWITCHES[$i]%:*}"
  enabled="${METHOD_SWITCHES[$i]##*:}"
  if [[ "${enabled}" != "0" && "${enabled}" != "1" ]]; then
    echo "[ERROR] ${method} switch must be exactly 0 or 1" >&2; exit 1
  fi
  [[ "${enabled}" == "1" ]] || continue
  MIS_METHODS_ENABLED+=("${method}")
  METHOD_INDEX_TAG+="_$((i+1))"
  case "${method}" in
    k_only_mis_*) family="k_only" ;;
    k_only_degree_anchor_mis_*) family="k_only_degree_anchor" ;;
    k_only_anchor_first_mis_*) family="k_only_anchor_first" ;;
    k_only_importance_mis_*) family="k_only_importance" ;;
    k_only_importance_iwmean_*) family="k_only_importance_iwmean" ;;
    k_only_importance_imean_*) family="k_only_importance_imean" ;;
    coherekv_wo_mis|importance_weighted_merge_all) continue ;;
    *) echo "[ERROR] Unknown method: ${method}" >&2; exit 1 ;;
  esac
  if [[ ! " ${ACTIVE_FAMILIES[*]-} " == *" ${family} "* ]]; then
    ACTIVE_FAMILIES+=("${family}")
  fi
done
if [[ "${#ACTIVE_FAMILIES[@]}" -eq 0 ]]; then
  if [[ "${#MIS_METHODS_ENABLED[@]}" -eq 0 ]]; then
    echo "[ERROR] Enable at least one optional baseline or MIS method" >&2; exit 1
  fi
  # A family is syntactically required by the evaluator even for an
  # optional-baseline-only sweep; no MIS method is run in this case.
  ACTIVE_FAMILIES=("k_only")
fi

# ============================================================
# 5. Results / PDF + PNG plots (Times New Roman ONLY)
# ============================================================
EXPERIMENT_PREFIX="journal_continuous_p_dual_lambda_paired_wo_ea_v6"
if [[ "${QUICK_MODE}" -eq 1 ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}${METHOD_INDEX_TAG}_quick"
else
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}${METHOD_INDEX_TAG}_full"
  CALIBRATION_NUM_SEQUENCES=16
  CALIBRATION_SEQUENCE_LENGTH=8192
fi
MODEL_BASENAME="${MODEL%/}"; MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
CALIBRATION_FILE="${OUT_DIR}/calibration/median_reference.json"
FONT_FAMILY="Times New Roman"
PNG_DPI=600

join_csv() { local IFS=,; echo "$*"; }
PPL_METHODS_CSV="$(join_csv "${MIS_METHODS_ENABLED[@]}")"
PPL_FAMILIES_CSV="$(join_csv "${ACTIVE_FAMILIES[@]}")"
PPL_BUDGETS_CSV="$(join_csv "${PPL_BUDGETS[@]}")"
PPL_P_VALUES_CSV="$(join_csv "${PPL_P_VALUES[@]}")"
PPL_LAMBDA_EA_CSV="$(join_csv "${PPL_LAMBDA_EA[@]}")"
PPL_LAMBDA_EE_CSV="$(join_csv "${PPL_LAMBDA_EE[@]}")"

if [[ "${RUN_CALIBRATION}" -eq 1 && -d "${OUT_DIR}" ]]; then
  echo "[CLEAN] Removing THIS experiment's old outputs: ${OUT_DIR}"
  rm -rf -- "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}/calibration"
RUN_LOG="${OUT_DIR}/run.log"
{
  echo "MODEL=${MODEL} TEST=${PPL_DATASET} CYCLES=${PPL_CYCLES} U=${PPL_UNSCORED_LEN} S=${PPL_SCORED_LEN}"
  echo "BUDGETS=${PPL_BUDGETS_CSV} P=${PPL_P_VALUES_CSV} LAMBDA_EA=${PPL_LAMBDA_EA_CSV} LAMBDA_EE=${PPL_LAMBDA_EE_CSV}"
  echo "METHODS=full_cache,merge_all,${PPL_METHODS_CSV} REFERENCE=${PPL_REFERENCE_MODE}"
  echo "BASELINES=merge_all:physical_EA_L2_no_EE;coherekv_wo_mis:historical_p_certified_EA_no_EE;all_*_wo_ea_certificate:physical_EA_L2_historical_p_EE_same_priority_fusion"
  echo "KEEPKV_SINK_SIZE=${KEEPKV_SINK_SIZE} KEEPKV_RECENT_PROTECT_SIZE=${KEEPKV_RECENT_PROTECT_SIZE} KEEPKV_SCORE_WINDOW=${KEEPKV_SCORE_WINDOW} KEEPKV_RECENCY_BETA=${KEEPKV_RECENCY_BETA}"
  echo "PLOT_PNG_PDF=${RUN_PLOT} FONT=${FONT_FAMILY} DPI=${PNG_DPI}"
  echo "RESULTS=${OUT_DIR}"
} | tee -a "${RUN_LOG}"

if [[ "${RUN_CALIBRATION}" -eq 1 ]]; then
  CMD=(python -u "${DIAG_DIR}/calibrate_median_reference.py"
    --model "${MODEL}" --dataset "${PPL_DATASET}"
    --calibration_split "${CALIBRATION_SPLIT}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --query_norm_safety_factor "${QUERY_NORM_SAFETY_FACTOR}"
    --sink_length "${SINK_LENGTH}"
    --sink_diagnostic_seqlen "${SINK_DIAGNOSTIC_SEQLEN}"
    --sink_probe_len "${SINK_PROBE_LEN}"
    --sink_presence_threshold "${SINK_PRESENCE_THRESHOLD}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PPL_PREFILL_CHUNK_SIZE}"
    --output "${CALIBRATION_FILE}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[CALIBRATE]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi
if [[ ! -s "${CALIBRATION_FILE}" ]]; then
  echo "[ERROR] Missing TRAIN calibration: ${CALIBRATION_FILE}" >&2;exit 1
fi

if [[ "${RUN_PPL}" -eq 1 ]]; then
  CMD=(python -u "${DIAG_DIR}/evaluate_ppl_continuous.py"
    --model "${MODEL}" --calibration "${CALIBRATION_FILE}"
    --output_dir "${OUT_DIR}/ppl_continuous" --dataset "${PPL_DATASET}"
    --cycles "${PPL_CYCLES}"
    --unscored_len "${PPL_UNSCORED_LEN}" --scored_len "${PPL_SCORED_LEN}"
    --prefill_chunk_size "${PPL_PREFILL_CHUNK_SIZE}"
    --budgets "${PPL_BUDGETS_CSV}"
    --p_values "${PPL_P_VALUES_CSV}"
    --lambda_ea "${PPL_LAMBDA_EA_CSV}"
    --lambda_ee "${PPL_LAMBDA_EE_CSV}"
    --reference_mode "${PPL_REFERENCE_MODE}"
    --mis_families "${PPL_FAMILIES_CSV}"
    --mis_methods "${PPL_METHODS_CSV}"
    --score_window "${KEEPKV_SCORE_WINDOW}"
    --sink_size "${KEEPKV_SINK_SIZE}"
    --recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
    --recency_beta "${KEEPKV_RECENCY_BETA}"
    --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
    --graph_backend "${GRAPH_BACKEND}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[PPL]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

# Always package raw numeric results before any optional plotting error.
python -u "${DIAG_DIR}/package_continuous.py" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
if [[ "${RUN_PLOT}" -eq 1 ]]; then
  CMD=(python -u "${DIAG_DIR}/plot_continuous_ppl.py"
    --input_dir "${OUT_DIR}/ppl_continuous"
    --font_family "${FONT_FAMILY}" --png_dpi "${PNG_DPI}")
  printf '[PLOT]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi
python -u "${DIAG_DIR}/package_continuous.py" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
echo "[UPLOAD] ${OUT_DIR}/cohere_p_continuous_results.zip" | tee -a "${RUN_LOG}"
