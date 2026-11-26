#!/usr/bin/env bash
set -euo pipefail

# All experiment settings below are literal values. Do not pass settings from
# environment variables or command-line arguments. Run from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# Runtime-only exports, same style as the original run_cosine_mis.sh.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
DIAG_DIR="${ROOT_DIR}/diagnosis_experiments"
RESULTS_ROOT="${DIAG_DIR}/results"

# ============================================================
# 1. Model and priority comparison
# ============================================================
# MODEL="Qwen/Qwen2.5-7B"
MODEL="meta-llama/Llama-3.2-3B"
# MODEL="meta-llama/Llama-3.2-1B"

TORCH_DTYPE="bfloat16"
DEVICE="cuda"
DEVICE_MAP="auto"
ATTN_IMPL="eager"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
RUN_CALIBRATION=1
RUN_PPL=1
RUN_PLOT=1

# Four actual method switches; conference is a parameter point of journal.
RUN_FULL_CACHE=1
RUN_EVICTION_ONLY=1
RUN_MERGE_ALL=1
RUN_COHEREKV_JOURNAL=1
METHOD_NAMES=(full_cache eviction_only merge_all coherekv_journal)
METHOD_FLAGS=("${RUN_FULL_CACHE}" "${RUN_EVICTION_ONLY}" "${RUN_MERGE_ALL}" "${RUN_COHEREKV_JOURNAL}")
ENABLED_METHODS=()
for i in "${!METHOD_NAMES[@]}"; do
  case "${METHOD_FLAGS[$i]}" in
    1) ENABLED_METHODS+=("${METHOD_NAMES[$i]}") ;;
    0) ;;
    *) echo "[ERROR] ${METHOD_NAMES[$i]} switch must be 0 or 1" >&2; exit 1 ;;
  esac
done
if [[ "${#ENABLED_METHODS[@]}" -eq 0 ]]; then echo '[ERROR] Enable at least one method' >&2; exit 1; fi
# ============================================================
# 2. Same continuous text spans for every method
# ============================================================
PPL_DATASET="wikitext2"
PPL_SPLIT="validation" # change to test only after freezing parameters
PPL_CYCLES=24
PPL_UNSCORED_LEN=256
PPL_SCORED_LEN=32
PPL_SEQUENCE_INDICES=("0" "1" "2" "3")
PPL_PREFILL_CHUNK_SIZE=128
PPL_BUDGETS=("0.10")
# Cartesian product: use a one-element array to fix a parameter.
PPL_ALPHA_EA=("0.25")
PPL_ALPHA_EE=("0.25")
PPL_LAMBDA_EA=("0.50")
PPL_LAMBDA_EE=("0.50")
PPL_PRIORITY_MODES=("degree" "count" "importance" "count_importance")
PPL_REFERENCE_MODES=("frozen_train") # add "online_first" to compare reference calibration
NEED_CALIBRATION=0
if [[ "${RUN_MERGE_ALL}" -eq 1 || "${RUN_COHEREKV_JOURNAL}" -eq 1 ]]; then
  for ref in "${PPL_REFERENCE_MODES[@]}"; do
    if [[ "$ref" == frozen_train ]]; then NEED_CALIBRATION=1; fi
  done
fi
KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=8
KEEPKV_SCORE_WINDOW=8
KEEPKV_RECENCY_BETA=1.0
DISTANCE_CHUNK_SIZE=256

# ============================================================
# 3. Calibration: training split only
# ============================================================
CALIBRATION_SPLIT="train"
CALIBRATION_SEED=42
CALIBRATION_NUM_SEQUENCES=4
CALIBRATION_SEQUENCE_LENGTH=4096
# ============================================================
# 4. Output, plotting style (PNG + PDF)
# ============================================================
EXPERIMENT_NAME="journal_weighted_mis_v4_${PPL_SPLIT}"
MODEL_BASENAME="${MODEL%/}"; MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
CALIBRATION_FILE="${OUT_DIR}/calibration/median_reference.json"
FONT_FAMILY="DejaVu Sans" # change to Times New Roman if installed
PNG_DPI=600
join_csv() { local IFS=,; echo "$*"; }
METHODS_CSV="$(join_csv "${ENABLED_METHODS[@]}")"
PPL_BUDGETS_CSV="$(join_csv "${PPL_BUDGETS[@]}")"
PPL_ALPHA_EA_CSV="$(join_csv "${PPL_ALPHA_EA[@]}")"
PPL_ALPHA_EE_CSV="$(join_csv "${PPL_ALPHA_EE[@]}")"
PPL_PRIORITY_MODES_CSV="$(join_csv "${PPL_PRIORITY_MODES[@]}")"
PPL_REFERENCE_MODES_CSV="$(join_csv "${PPL_REFERENCE_MODES[@]}")"
PPL_LAMBDA_EA_CSV="$(join_csv "${PPL_LAMBDA_EA[@]}")"
PPL_LAMBDA_EE_CSV="$(join_csv "${PPL_LAMBDA_EE[@]}")"
SEQUENCES_CSV="$(join_csv "${PPL_SEQUENCE_INDICES[@]}")"
if [[ "${#PPL_SEQUENCE_INDICES[@]}" -lt 1 ]]; then
  echo '[ERROR] PPL_SEQUENCE_INDICES must not be empty' >&2; exit 1
fi
for i in "${!PPL_SEQUENCE_INDICES[@]}"; do
  seq_idx="${PPL_SEQUENCE_INDICES[$i]}"
  if [[ ! "${seq_idx}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "[ERROR] Bad sequence index ${seq_idx}" >&2; exit 1
  fi
  for j in "${!PPL_SEQUENCE_INDICES[@]}"; do
    if [[ "${j}" -lt "${i}" && "${seq_idx}" == "${PPL_SEQUENCE_INDICES[$j]}" ]]; then
      echo "[ERROR] Duplicate sequence index ${seq_idx}" >&2; exit 1
    fi
  done
done
mkdir -p "${OUT_DIR}/calibration"
RUN_LOG="${OUT_DIR}/run.log"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
: > "${RUN_LOG}"
# Compact status in the main log; preserve unfiltered stdout/stderr separately.
run_stage() {
  local stage="$1" detail="$2"; shift 2
  echo "[${stage}] started (details: ${detail})" | tee -a "${RUN_LOG}"
  if "$@" > "${detail}" 2>&1; then
    echo "[${stage}] complete" | tee -a "${RUN_LOG}"
  else
    echo "[ERROR] ${stage} failed; see ${detail}" | tee -a "${RUN_LOG}" >&2
    tail -n 30 "${detail}" | tee -a "${RUN_LOG}" >&2
    return 1
  fi
}
{
  echo "MODEL=${MODEL} CYCLES=${PPL_CYCLES} U=${PPL_UNSCORED_LEN} S=${PPL_SCORED_LEN} SEQUENCES=${SEQUENCES_CSV}"
  echo "METHODS=${METHODS_CSV}"
  echo "SPLIT=${PPL_SPLIT}"
  echo "BUDGETS=${PPL_BUDGETS_CSV} ALPHA_EA=${PPL_ALPHA_EA_CSV} ALPHA_EE=${PPL_ALPHA_EE_CSV} EA=${PPL_LAMBDA_EA_CSV} EE=${PPL_LAMBDA_EE_CSV} PRIORITY=${PPL_PRIORITY_MODES_CSV}"
  echo "REFERENCE=${PPL_REFERENCE_MODES_CSV} FONT=${FONT_FAMILY} PNG_DPI=${PNG_DPI}"
  echo "RESULTS=${OUT_DIR}"
} | tee -a "${RUN_LOG}"

# Validate and record the exact shell grid before model/GPU work.
if [[ "${RUN_PPL}" -eq 1 ]]; then
  python "${DIAG_DIR}/sweep_config.py" \
    --methods "${METHODS_CSV}" --budgets "${PPL_BUDGETS_CSV}" \
    --alpha_ea "${PPL_ALPHA_EA_CSV}" \
    --alpha_ee "${PPL_ALPHA_EE_CSV}" \
    --lambda_ea "${PPL_LAMBDA_EA_CSV}" --lambda_ee "${PPL_LAMBDA_EE_CSV}" \
    --priority_modes "${PPL_PRIORITY_MODES_CSV}" --reference_modes "${PPL_REFERENCE_MODES_CSV}" > "${OUT_DIR}/sweep_plan.json"
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("[PLAN]",p["total"],"configurations/sequence:",p["counts"])' "${OUT_DIR}/sweep_plan.json" | tee -a "${RUN_LOG}"
fi

# Resolve complete windows once before any GPU work. All methods share these.
WINDOW_MANIFEST="${OUT_DIR}/sequence_windows.json"
CMD=(python -u "${DIAG_DIR}/sequence_windows.py" --model "${MODEL}" --dataset "${PPL_DATASET}"
  --split "${PPL_SPLIT}" --sequence_indices "${SEQUENCES_CSV}"
  --window_length "$((PPL_CYCLES*(PPL_UNSCORED_LEN+PPL_SCORED_LEN)))" --output "${WINDOW_MANIFEST}")
if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
run_stage WINDOWS "${LOG_DIR}/windows.log" "${CMD[@]}"
SEQUENCES_CSV="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["selected_indices"])))' "${WINDOW_MANIFEST}")"
IFS=, read -r -a PPL_SEQUENCE_INDICES <<< "${SEQUENCES_CSV}"
echo "[ACTUAL SEQUENCES] ${SEQUENCES_CSV}" | tee -a "${RUN_LOG}"

if [[ "${RUN_CALIBRATION}" -eq 1 && "${NEED_CALIBRATION}" -eq 1 ]]; then
  CMD=(python -u "${DIAG_DIR}/calibrate_median_reference.py"
    --model "${MODEL}" --dataset "${PPL_DATASET}"
    --calibration_split "${CALIBRATION_SPLIT}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PPL_PREFILL_CHUNK_SIZE}"
    --output "${CALIBRATION_FILE}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  run_stage CALIBRATION "${LOG_DIR}/calibration.log" "${CMD[@]}"
fi
if [[ "${NEED_CALIBRATION}" -eq 1 && ! -s "${CALIBRATION_FILE}" ]]; then
  echo "[ERROR] Missing calibration ${CALIBRATION_FILE}" >&2; exit 1
fi

if [[ "${RUN_PPL}" -eq 1 ]]; then
  # Remove stale outputs from disabled methods; retain calibration and previous ZIP until repackaging.
  rm -rf -- "${OUT_DIR}/sequences" "${OUT_DIR}/aggregate"
  seq_done=0
  seq_total="${#PPL_SEQUENCE_INDICES[@]}"
  for seq_idx in "${PPL_SEQUENCE_INDICES[@]}"; do
    seq_done=$((seq_done+1))
    SEQ_DIR="${OUT_DIR}/sequences/seq_${seq_idx}/ppl_continuous"
    mkdir -p "${SEQ_DIR}"
    CMD=(python -u "${DIAG_DIR}/evaluate_ppl_continuous.py"
      --model "${MODEL}" --calibration "${CALIBRATION_FILE}" --methods "${METHODS_CSV}"
      --output_dir "${SEQ_DIR}" --dataset "${PPL_DATASET}" --split "${PPL_SPLIT}"
      --sequence_index "${seq_idx}"
      --cycles "${PPL_CYCLES}"
      --unscored_len "${PPL_UNSCORED_LEN}" --scored_len "${PPL_SCORED_LEN}"
      --prefill_chunk_size "${PPL_PREFILL_CHUNK_SIZE}"
      --budgets "${PPL_BUDGETS_CSV}"
      --alpha_ea "${PPL_ALPHA_EA_CSV}"
      --alpha_ee "${PPL_ALPHA_EE_CSV}"
      --lambda_ea "${PPL_LAMBDA_EA_CSV}"
      --lambda_ee "${PPL_LAMBDA_EE_CSV}"
      --priority_modes "${PPL_PRIORITY_MODES_CSV}" --reference_modes "${PPL_REFERENCE_MODES_CSV}"
      --score_window "${KEEPKV_SCORE_WINDOW}"
      --sink_size "${KEEPKV_SINK_SIZE}"
      --recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
      --recency_beta "${KEEPKV_RECENCY_BETA}"
      --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
      --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
      --device "${DEVICE}" --device_map "${DEVICE_MAP}")
    if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
    if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
    DETAIL_LOG="${SEQ_DIR}/evaluation_detail.log"
    {
      echo
      echo "======================================================================"
      echo "SEQUENCE ${seq_done}/${seq_total} | index=${seq_idx} | $((PPL_CYCLES*PPL_SCORED_LEN)) scored tokens"
      echo "One row per actual evaluation; unused grid dimensions omitted."
    } | tee -a "${RUN_LOG}"
    if "${CMD[@]}" 2>&1 | tee "${DETAIL_LOG}" | awk '/^\[PPL TABLE\] / {sub(/^\[PPL TABLE\] /, ""); print; fflush()}' | tee -a "${RUN_LOG}"; then
      echo "[SEQUENCE ${seq_done}/${seq_total}] complete (details: ${DETAIL_LOG})" | tee -a "${RUN_LOG}"
    else
      echo "[ERROR] Sequence index=${seq_idx} failed; see ${DETAIL_LOG}" | tee -a "${RUN_LOG}" >&2
      tail -n 30 "${DETAIL_LOG}" | tee -a "${RUN_LOG}" >&2
      exit 1
    fi
  done
fi
python -u "${DIAG_DIR}/aggregate_continuous_sequences.py" \
  --input_dir "${OUT_DIR}" --sequence_indices "${SEQUENCES_CSV}" 2>&1 | tee -a "${RUN_LOG}"
# Package raw numeric results before optional plotting errors.
python -u "${DIAG_DIR}/package_continuous.py" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
if [[ "${RUN_PLOT}" -eq 1 ]]; then
  run_stage PLOT "${LOG_DIR}/plot.log" python -u "${DIAG_DIR}/plot_continuous_ppl.py" \
    --input_dir "${OUT_DIR}/aggregate" \
    --font_family "${FONT_FAMILY}" --png_dpi "${PNG_DPI}"
fi
python -u "${DIAG_DIR}/package_continuous.py" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
echo "[UPLOAD] ${OUT_DIR}/coherekv_journal_results.zip" | tee -a "${RUN_LOG}"
