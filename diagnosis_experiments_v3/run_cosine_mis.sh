#!/usr/bin/env bash
set -euo pipefail
# Group-D2 v20 Qwen-7B held-out cross-round validation. Literal settings ONLY.
# No experiment parameter is inherited from environment variables.
ROOT_DIR="/home/tlxr/CohereKV"
cd "${ROOT_DIR}"
# Fixed runtime exports only, not experiment settings.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
DIAG_DIR="${ROOT_DIR}/diagnosis_experiments"

# MODEL="Qwen/Qwen2.5-7B"
MODEL="meta-llama/Llama-3.2-3B"

PPL_DATASET="wikitext2"
PPL_CYCLES=24                         # Same compression rounds as prior Qwen pilot.
PPL_UNSCORED_LEN=1024
PPL_SCORED_LEN=32                    # 4x the previous 32 scored tokens/round.
# PPL_SEQUENCE_INDICES=("1" "2" "3")     # Held-out vs previous Qwen sequence 0.
PPL_SEQUENCE_INDICES=("0")     # Held-out vs previous Qwen sequence 0.
PPL_PREFILL_CHUNK_SIZE=128
SWEEP_LAMBDA_HISTORY=("0.50")          # Fixed from Qwen-7B seq-0 pilot; DO NOT TUNE on held-out.
SWEEP_LAMBDA_PAIRS=("0.50:0.50")       # Fixed from Qwen-7B seq-0 pilot.
SWEEP_BUDGETS=("0.10")
SWEEP_PAIR_WINDOWS=("8")
PPL_P_VALUES="2"
PPL_GEOMETRY_MODE="exact_p2"
METHODS_ENABLED="coherekv_inf_certificate,group_d2,group_d2_importance_matched"
# Evaluator automatically adds full_cache, eviction_only and merge_all: six arms.

KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=32
KEEPKV_SCORE_WINDOW=32
KEEPKV_RECENCY_BETA=1.0
DISTANCE_CHUNK_SIZE=256
GRAPH_BACKEND="streamed"
PPL_ROUTE_DIAGNOSTICS=0
PPL_ORACLE_ENABLED=0
FUSION_SAMPLE_TOPK=0
RUN_PPL=1
RUN_PLOT=1
FONT_FAMILY="Times New Roman"
PNG_DPI=600

CALIBRATION_SPLIT="train"
CALIBRATION_SEED=42
CALIBRATION_NUM_SEQUENCES=4
CALIBRATION_SEQUENCE_LENGTH=4096
SINK_LENGTH=1
SINK_DIAGNOSTIC_SEQLEN=4096
SINK_PROBE_LEN=64
SINK_PRESENCE_THRESHOLD=0.05
QUERY_NORM_SAFETY_FACTOR=1.05
TORCH_DTYPE="bfloat16"
ATTN_IMPL="eager"
DEVICE="cuda"
DEVICE_MAP="auto"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
PPL_REFERENCE_MODE="frozen_train"

EXPERIMENT_PREFIX="journal_group_d2_crossround_v20_qwen7b_heldout"
MODEL_BASENAME="${MODEL##*/}"
SWEEP_ROOT="${DIAG_DIR}/results/${MODEL_BASENAME}/${EXPERIMENT_PREFIX}"
PLAN_FILE="${SWEEP_ROOT}/sweep_plan.csv"
CALIBRATION_FILE="${SWEEP_ROOT}/calibration/median_reference.json"
RUN_LOG="${SWEEP_ROOT}/run.log"
# Reuse the exact frozen TRAIN calibration from the previous Qwen-7B pilot.
# If unavailable, recompute using the same literal calibration parameters.
PRIOR_CALIBRATION="${DIAG_DIR}/results/${MODEL_BASENAME}/journal_group_d2_fast_screen_v19_quick/calibration/median_reference.json"

# Avoid destroying a completed/partially completed validation run silently.
if [[ -d "${SWEEP_ROOT}/settings" ]] && [[ -n "$(ls -A "${SWEEP_ROOT}/settings" 2>/dev/null)" ]];then
  echo "[ERROR] Validation output already exists: ${SWEEP_ROOT}" >&2
  echo "Edit the literal EXPERIMENT_PREFIX before a fresh run; existing results were NOT deleted." >&2
  exit 1
fi
mkdir -p "${SWEEP_ROOT}/calibration" "${SWEEP_ROOT}/settings"
if [[ ! -s "${CALIBRATION_FILE}" && -s "${PRIOR_CALIBRATION}" ]];then
  cp "${PRIOR_CALIBRATION}" "${CALIBRATION_FILE}"
  echo "[CALIBRATION] Reused Qwen-7B pilot TRAIN reference: ${PRIOR_CALIBRATION}"
fi
printf '%s\n' 'setting_id,lambda_history,lambda_ea,lambda_ee,budget,pair_window' > "${PLAN_FILE}"
for h in "${SWEEP_LAMBDA_HISTORY[@]}";do
  for pair in "${SWEEP_LAMBDA_PAIRS[@]}";do
    [[ "${pair}" == *:* ]] || { echo "Bad EA:EE ${pair}" >&2;exit 1; }
    ea="${pair%%:*}"; ee="${pair#*:}"
    for budget in "${SWEEP_BUDGETS[@]}";do
      for k in "${SWEEP_PAIR_WINDOWS[@]}";do
        setting_id="h${h//./p}_ea${ea//./p}_ee${ee//./p}_b${budget//./p}_k${k}"
        printf '%s,%s,%s,%s,%s,%s\n' "${setting_id}" "${h}" "${ea}" "${ee}" "${budget}" "${k}" >> "${PLAN_FILE}"
      done
    done
  done
done
join_csv() { local IFS=,; echo "$*"; }
SEQUENCES_CSV="$(join_csv "${PPL_SEQUENCE_INDICES[@]}")"
python -u "${DIAG_DIR}/aggregate_group_d2_quick.py" --input_dir "${SWEEP_ROOT}" \
  --sequence_indices "${SEQUENCES_CSV}" --validate_only 1
{
  echo "[HELD-OUT VALIDATION] MODEL=${MODEL} CYCLES=${PPL_CYCLES} U=${PPL_UNSCORED_LEN} S=${PPL_SCORED_LEN} WINDOWS=${SEQUENCES_CSV}"
  echo "[FIXED SETTINGS] H=$(join_csv "${SWEEP_LAMBDA_HISTORY[@]}") EA:EE=$(join_csv "${SWEEP_LAMBDA_PAIRS[@]}") B=$(join_csv "${SWEEP_BUDGETS[@]}") k=$(join_csv "${SWEEP_PAIR_WINDOWS[@]}")"
  echo "[SIX ARMS] full_cache,eviction_only,merge_all,${METHODS_ENABLED}"
  echo "[NOTE] lambda_EE has no effect on Group-D2 selection; it affects diagnostics."
} | tee -a "${RUN_LOG}"
if [[ ! -s "${CALIBRATION_FILE}" ]];then
  echo '[CALIBRATION] No reusable Qwen-7B reference; perform frozen TRAIN calibration.' | tee -a "${RUN_LOG}"
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
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi
if [[ ! -s "${CALIBRATION_FILE}" ]];then echo '[ERROR] Frozen reference missing' >&2;exit 1;fi

while IFS=, read -r setting_id h ea ee budget k;do
  [[ "${setting_id}" != setting_id ]] || continue
  echo "[SETTING] ${setting_id} history=${h} EA=${ea} EE=${ee} budget=${budget} lookahead=${k}" | tee -a "${RUN_LOG}"
  for seq_idx in "${PPL_SEQUENCE_INDICES[@]}";do
    OUT_DIR="${SWEEP_ROOT}/settings/${setting_id}/sequences/seq_${seq_idx}/ppl_continuous"
    mkdir -p "${OUT_DIR}"
    CMD=(python -u "${DIAG_DIR}/evaluate_ppl_continuous.py"
      --model "${MODEL}" --calibration "${CALIBRATION_FILE}"
      --output_dir "${OUT_DIR}" --dataset "${PPL_DATASET}"
      --sequence_index "${seq_idx}" --cycles "${PPL_CYCLES}"
      --unscored_len "${PPL_UNSCORED_LEN}" --scored_len "${PPL_SCORED_LEN}"
      --prefill_chunk_size "${PPL_PREFILL_CHUNK_SIZE}"
      --budgets "${budget}" --p_values "${PPL_P_VALUES}"
      --geometry_modes "${PPL_GEOMETRY_MODE}"
      --lambda_pairs "${ea}:${ee}" --lambda_history "${h}"
      --group_d2_pair_window "${k}" --fusion_sample_topk "${FUSION_SAMPLE_TOPK}"
      --reference_mode "${PPL_REFERENCE_MODE}"
      --mis_families "k_only_importance" --mis_methods "${METHODS_ENABLED}"
      --route_diagnostics "${PPL_ROUTE_DIAGNOSTICS}" --oracle_enabled "${PPL_ORACLE_ENABLED}"
      --score_window "${KEEPKV_SCORE_WINDOW}" --sink_size "${KEEPKV_SINK_SIZE}"
      --recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
      --recency_beta "${KEEPKV_RECENCY_BETA}"
      --distance_chunk_size "${DISTANCE_CHUNK_SIZE}" --graph_backend "${GRAPH_BACKEND}"
      --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
      --device "${DEVICE}" --device_map "${DEVICE_MAP}")
    echo "[PPL] setting=${setting_id} sequence=${seq_idx}" | tee -a "${RUN_LOG}"
    "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
  done
done < "${PLAN_FILE}"

python -u "${DIAG_DIR}/aggregate_group_d2_quick.py" --input_dir "${SWEEP_ROOT}" \
  --sequence_indices "${SEQUENCES_CSV}" --plots 0 2>&1 | tee -a "${RUN_LOG}"
python -u "${DIAG_DIR}/package_group_d2_sweep.py" --input_dir "${SWEEP_ROOT}" 2>&1 | tee -a "${RUN_LOG}"
if [[ "${RUN_PLOT}" == 1 ]];then
  python -u "${DIAG_DIR}/aggregate_group_d2_quick.py" --input_dir "${SWEEP_ROOT}" \
    --sequence_indices "${SEQUENCES_CSV}" --plots 1 \
    --font_family "${FONT_FAMILY}" --png_dpi "${PNG_DPI}" 2>&1 | tee -a "${RUN_LOG}"
  python -u "${DIAG_DIR}/package_group_d2_sweep.py" --input_dir "${SWEEP_ROOT}" 2>&1 | tee -a "${RUN_LOG}"
fi
echo "[DONE] ${SWEEP_ROOT}/cohere_p_continuous_results.zip"
