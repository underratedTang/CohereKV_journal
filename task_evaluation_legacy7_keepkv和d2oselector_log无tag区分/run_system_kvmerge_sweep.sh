#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================
cd /home/tlxr/SimCalKV

# =========================
# 手动修改区
# =========================
SYSTEM_SCRIPT="/home/tlxr/SimCalKV/task_evaluation/evaluate_system_kvmerge.py"
# OUT_DIR="/home/tlxr/SimCalKV/results/system_manual_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/home/tlxr/SimCalKV/results/system_manual_test3"


ARCH="llama"
MODEL_NAME="llama-3.1-8B-Instruct"
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Qwen 示例
# ARCH="qwen"
# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

DATASET="gov_report"
BATCH_SIZE=1
INPUT_LENGTH=8192
GENERATION_LENGTH=256
WARMUP_RUNS=3
MEASURED_RUNS=10
SAMPLE_START_INDEX=0

DTYPE="bf16"
ATTN_IMPL="eager"
DEVICE="cuda"
SEED=42

# 该值影响被计时的 KeepKV similarity kernel，必须在所有预算上固定。
KEEPKV_SIMILARITY_CHUNK_SIZE=256

METHODS=(
  "None"
  "KeepKV"
  "CaM"
  "SimCalKV"
  "GroupLogitKV"
  "mymethod"
  "CentroidKV"
)

# compress_ratio 是删除比例；0.2/0.4/0.6/0.8 对应保留 80%/60%/40%/20%。
RATIOS=(
  "0.2"
  "0.4"
  "0.6"
  "0.8"
)

SIM_THR_GROUPLOGITKV_LIST=("0.8")
LAMBDA_MYMETHOD_LIST=("0.5")
# LAMBDA_MYMETHOD_LIST=("0.3" "0.5" "0.75" "1.0")
MYMETHOD_SELECTOR="d2o"  # keepkv | attention | d2o
MYMETHOD_GROUP_MODE="full"   # full | anchor_only | eviction_only
MYMETHOD_GRAPH_BACKEND="cuda" # auto | cuda | torch
MYMETHOD_ROUTE_BACKEND="torch" # auto | cuda | torch

# KeepKV-style selector only
MYMETHOD_KEEPKV_SINK_SIZE=4
MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE=128
MYMETHOD_KEEPKV_SCORE_WINDOW=128
MYMETHOD_KEEPKV_RECENCY_BETA=1.0
MYMETHOD_KEEPKV_USE_LOG_COUNT=0  # 0: official-style base score; 1: C*score (logC ablation)

# D2O-style selector only. D2O only decides retained/evicted; merging remains mymethod.
MYMETHOD_D2O_SINK_SIZE=4
MYMETHOD_D2O_RECENT_SIZE=128
MYMETHOD_D2O_USE_LOG_COUNT=1  # 1: accumulate actual +logC post-softmax attention; 0: ablation
MYMETHOD_D2O_SCORE_RESET_INTERVAL=0  # 0: request cumulative; 1: reset every compression; N: every N compressions

# mymethod certified grouping core
MYMETHOD_DISTANCE_CHUNK_SIZE=256
MYMETHOD_ATTENTION_CHUNK_SIZE=128
CENTROIDKV_SINK_BUDGET=16
CENTROIDKV_LOCAL_BUDGET=64
CENTROIDKV_CHUNK_SIZE=256
CENTROIDKV_MERGE_RATIO=0.8
CENTROIDKV_DECREASE_RATE=0.2
SIM_THR_SIMCALKV_LIST=("0.0")
SIM_THR_KEEPKV_LIST=("0.8")

# =========================
# 自动区
# =========================
if [[ ! -f "${SYSTEM_SCRIPT}" ]]; then
  echo "[ERROR] SYSTEM_SCRIPT not found: ${SYSTEM_SCRIPT}"
  exit 1
fi

TASK_EVAL_DIR="$(cd "$(dirname "${SYSTEM_SCRIPT}")" && pwd)"
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr '/:' '_')"
MODEL_OUT_DIR="${OUT_DIR}/${MODEL_SLUG}"
LOG_DIR="${MODEL_OUT_DIR}/logs"
JSON_DIR="${MODEL_OUT_DIR}/json"
SUMMARY_FILE="${MODEL_OUT_DIR}/system_summary_$(date +%Y%m%d_%H%M%S).jsonl"
RUN_LOG="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}" "${JSON_DIR}"

echo "SYSTEM_SCRIPT=${SYSTEM_SCRIPT}" | tee -a "${RUN_LOG}"
echo "MODEL_NAME=${MODEL_NAME}" | tee -a "${RUN_LOG}"
echo "MODEL_ID=${MODEL_ID}" | tee -a "${RUN_LOG}"
echo "ARCH=${ARCH}" | tee -a "${RUN_LOG}"
echo "DATASET=${DATASET}" | tee -a "${RUN_LOG}"
echo "BATCH_SIZE=${BATCH_SIZE}" | tee -a "${RUN_LOG}"
echo "INPUT_LENGTH=${INPUT_LENGTH}" | tee -a "${RUN_LOG}"
echo "GENERATION_LENGTH=${GENERATION_LENGTH}" | tee -a "${RUN_LOG}"
echo "WARMUP_RUNS=${WARMUP_RUNS}" | tee -a "${RUN_LOG}"
echo "MEASURED_RUNS=${MEASURED_RUNS}" | tee -a "${RUN_LOG}"
echo "KEEPKV_SIMILARITY_CHUNK_SIZE=${KEEPKV_SIMILARITY_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "LAMBDA_MYMETHOD_LIST=${LAMBDA_MYMETHOD_LIST[*]}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_SELECTOR=${MYMETHOD_SELECTOR}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_GROUP_MODE=${MYMETHOD_GROUP_MODE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_GRAPH_BACKEND=${MYMETHOD_GRAPH_BACKEND}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_ROUTE_BACKEND=${MYMETHOD_ROUTE_BACKEND}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_KEEPKV_SINK_SIZE=${MYMETHOD_KEEPKV_SINK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE=${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_KEEPKV_SCORE_WINDOW=${MYMETHOD_KEEPKV_SCORE_WINDOW}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_KEEPKV_RECENCY_BETA=${MYMETHOD_KEEPKV_RECENCY_BETA}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_KEEPKV_USE_LOG_COUNT=${MYMETHOD_KEEPKV_USE_LOG_COUNT}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_D2O_SINK_SIZE=${MYMETHOD_D2O_SINK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_D2O_RECENT_SIZE=${MYMETHOD_D2O_RECENT_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_D2O_USE_LOG_COUNT=${MYMETHOD_D2O_USE_LOG_COUNT}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_D2O_SCORE_RESET_INTERVAL=${MYMETHOD_D2O_SCORE_RESET_INTERVAL}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_DISTANCE_CHUNK_SIZE=${MYMETHOD_DISTANCE_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_ATTENTION_CHUNK_SIZE=${MYMETHOD_ATTENTION_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_SINK_BUDGET=${CENTROIDKV_SINK_BUDGET}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_LOCAL_BUDGET=${CENTROIDKV_LOCAL_BUDGET}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_CHUNK_SIZE=${CENTROIDKV_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_MERGE_RATIO=${CENTROIDKV_MERGE_RATIO}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_DECREASE_RATE=${CENTROIDKV_DECREASE_RATE}" | tee -a "${RUN_LOG}"
echo "METHODS=${METHODS[*]}" | tee -a "${RUN_LOG}"
echo "RATIOS=${RATIOS[*]}" | tee -a "${RUN_LOG}"
echo "E2E=H2D+prefill/statistics+compression+fixed-length decode" | tee -a "${RUN_LOG}"
echo "THROUGHPUT=batch_size*generation_length/E2E" | tee -a "${RUN_LOG}"

run_one() {
  local method="$1"
  local ratio="$2"
  local threshold="na"
  if [[ $# -ge 3 ]]; then threshold="$3"; fi
  local threshold_tag="na"
  local extra_args=()

  if [[ "${method}" == "GroupLogitKV" ]]; then
    threshold_tag="${threshold}"
    extra_args+=(--sim_thr_grouplogitkv "${threshold}")
  elif [[ "${method}" == "mymethod" ]]; then
    threshold_tag="selector${MYMETHOD_SELECTOR}_lambda${threshold}"
    extra_args+=(
      --lambda_mymethod "${threshold}"
      --mymethod_selector "${MYMETHOD_SELECTOR}"
      --mymethod_group_mode "${MYMETHOD_GROUP_MODE}"
      --mymethod_graph_backend "${MYMETHOD_GRAPH_BACKEND}"
      --mymethod_route_backend "${MYMETHOD_ROUTE_BACKEND}"
      --mymethod_keepkv_sink_size "${MYMETHOD_KEEPKV_SINK_SIZE}"
      --mymethod_keepkv_recent_protect_size "${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}"
      --mymethod_keepkv_score_window "${MYMETHOD_KEEPKV_SCORE_WINDOW}"
      --mymethod_keepkv_recency_beta "${MYMETHOD_KEEPKV_RECENCY_BETA}"
      --mymethod_keepkv_use_log_count "${MYMETHOD_KEEPKV_USE_LOG_COUNT}"
      --mymethod_d2o_sink_size "${MYMETHOD_D2O_SINK_SIZE}"
      --mymethod_d2o_recent_size "${MYMETHOD_D2O_RECENT_SIZE}"
      --mymethod_d2o_use_log_count "${MYMETHOD_D2O_USE_LOG_COUNT}" --mymethod_d2o_score_reset_interval "${MYMETHOD_D2O_SCORE_RESET_INTERVAL}"
      --mymethod_distance_chunk_size "${MYMETHOD_DISTANCE_CHUNK_SIZE}"
      --mymethod_attention_chunk_size "${MYMETHOD_ATTENTION_CHUNK_SIZE}"
    )
  elif [[ "${method}" == "CentroidKV" ]]; then
    threshold_tag="official"
    extra_args+=(
      --centroidkv_sink_budget "${CENTROIDKV_SINK_BUDGET}"
      --centroidkv_local_budget "${CENTROIDKV_LOCAL_BUDGET}"
      --centroidkv_chunk_size "${CENTROIDKV_CHUNK_SIZE}"
      --centroidkv_merge_ratio "${CENTROIDKV_MERGE_RATIO}"
      --centroidkv_decrease_rate "${CENTROIDKV_DECREASE_RATE}"
    )
  elif [[ "${method}" == "SimCalKV" ]]; then
    threshold_tag="${threshold}"
    extra_args+=(--sim_thr_simcalkv "${threshold}")
  elif [[ "${method}" == "KeepKV" ]]; then
    threshold_tag="${threshold}"
    extra_args+=(--sim_thr_keepkv "${threshold}")
  fi

  local retained
  retained="$(python -c "print(f'{1.0-float(\"${ratio}\"):.1f}')")"
  if [[ "${method}" == "None" ]]; then
    retained="1.0"
  fi
  local tag="${ARCH}_${DATASET}_${method}_retained${retained}_sim${threshold_tag}"
  tag="$(printf '%s' "${tag}" | tr '.-' 'p_')"
  local log_file="${LOG_DIR}/${tag}.log"
  local json_file="${JSON_DIR}/${tag}.json"

  echo "========== ${method} | retained=${retained} | sim=${threshold_tag} ==========" | tee -a "${RUN_LOG}"
  python "${SYSTEM_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --model_id "${MODEL_ID}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --arch "${ARCH}" \
    --kv_method "${method}" \
    --compress_ratio "${ratio}" \
    "${extra_args[@]}" \
    --dataset "${DATASET}" \
    --batch_size "${BATCH_SIZE}" \
    --input_length "${INPUT_LENGTH}" \
    --generation_length "${GENERATION_LENGTH}" \
    --warmup_runs "${WARMUP_RUNS}" \
    --measured_runs "${MEASURED_RUNS}" \
    --sample_start_index "${SAMPLE_START_INDEX}" \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --device "${DEVICE}" \
    --keepkv_similarity_chunk_size "${KEEPKV_SIMILARITY_CHUNK_SIZE}" \
    --seed "${SEED}" \
    --output_json "${json_file}" \
    > "${log_file}" 2>&1 || {
      echo "[FAILED] ${tag}. See ${log_file}" | tee -a "${RUN_LOG}"
      return 1
    }

  if [[ -f "${json_file}" ]]; then
    python - <<PY >> "${SUMMARY_FILE}"
import json
p = "${json_file}"
with open(p, "r", encoding="utf-8") as f:
    row = json.load(f)
row["_json_file"] = p
print(json.dumps(row, ensure_ascii=False))
PY
    echo "[OK] ${tag}" | tee -a "${RUN_LOG}"
  else
    echo "[MISSING_JSON] ${tag}: ${json_file}" | tee -a "${RUN_LOG}"
  fi
}

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "None" ]]; then
    run_one "None" "0.0" "na"
    continue
  fi
  for ratio in "${RATIOS[@]}"; do
    if [[ "${method}" == "GroupLogitKV" ]]; then
      for threshold in "${SIM_THR_GROUPLOGITKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${threshold}"
      done
    elif [[ "${method}" == "mymethod" ]]; then
      for lambda_value in "${LAMBDA_MYMETHOD_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${lambda_value}"
      done
    elif [[ "${method}" == "SimCalKV" ]]; then
      for threshold in "${SIM_THR_SIMCALKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${threshold}"
      done
    elif [[ "${method}" == "KeepKV" ]]; then
      for threshold in "${SIM_THR_KEEPKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${threshold}"
      done
    else
      run_one "${method}" "${ratio}" "na"
    fi
  done
done

echo "Done." | tee -a "${RUN_LOG}"
echo "MODEL_OUT_DIR=${MODEL_OUT_DIR}" | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
