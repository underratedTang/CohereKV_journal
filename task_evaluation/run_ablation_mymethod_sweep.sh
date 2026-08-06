#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================
cd /root/SimCalKV

# =========================
# 手动修改区
# =========================
ABLATION_SCRIPT="/root/SimCalKV/task_evaluation/evaluate_ablation_mymethod.py"
OUT_DIR="/root/SimCalKV/results/mymethod_ablation_manual_test"

# Qwen 示例
# FAMILY="qwen"
# BASE_MODEL="Qwen/Qwen2.5-7B"
# INSTRUCT_MODEL_NAME="Qwen2.5-7B-Instruct"
# INSTRUCT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

# LLaMA 示例
FAMILY="llama"
BASE_MODEL="meta-llama/Llama-3.2-3B"
INSTRUCT_MODEL_NAME="llama-3.2-3B-Instruct"
INSTRUCT_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"

DEVICE="cuda:0"
DEVICE_MAP="auto"
TORCH_DTYPE="bfloat16"
AMP_DTYPE="none"
ATTN_IMPL="eager"
SEED=0

COMPRESS_RATIO="0.8"
TAU_MYMETHOD="0.7"
MYMETHOD_SELECTOR="keepkv"  # keepkv | grouplogitkv
MYMETHOD_SINK_SIZE=4
MYMETHOD_RECENT_WINDOW=128
MYMETHOD_RECENCY_BETA=1.0
MYMETHOD_DISTANCE_CHUNK_SIZE=256
MYMETHOD_ATTENTION_CHUNK_SIZE=128
MYMETHOD_SINK_CAN_ABSORB=1

MODEL_SEQLEN=8192
CHUNK_SIZE=32
MIN_KV_LEN=2
COMPRESS_EVERY=1
MAX_EVAL_SEQUENCES=2

LONGBENCH_DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"
PREFILL_CHUNK_SIZE=128

VARIANTS=(
  "eviction_only"
  "certified_merge_no_count"
  "nearest_merge_no_certificate"
  "full_no_sink_absorb"
  "full"
)

TASKS=(
  "ppl"
  "longbench"
)

RUN_ORACLE=false
ORACLE_DATASET="gov_report"
ORACLE_INPUT_LENGTH=8192
ORACLE_COMPRESS_RATIO="0.6"
ORACLE_NUM_PROMPTS=8
ORACLE_LAYERS="auto"
ORACLE_PAIRS_PER_LAYER=512

# =========================
# 自动区：一般不用改
# =========================
if [[ ! -f "${ABLATION_SCRIPT}" ]]; then
  echo "[ERROR] ABLATION_SCRIPT not found: ${ABLATION_SCRIPT}"
  exit 1
fi

TASK_EVAL_DIR="$(cd "$(dirname "${ABLATION_SCRIPT}")" && pwd)"
MODEL_SLUG="$(printf '%s' "${BASE_MODEL}" | tr '/:' '_')"
MODEL_OUT_DIR="${OUT_DIR}/${MODEL_SLUG}"
LOG_DIR="${MODEL_OUT_DIR}/logs"
JSON_DIR="${MODEL_OUT_DIR}/json"
PRED_DIR="${MODEL_OUT_DIR}/pred"
SUMMARY_FILE="${MODEL_OUT_DIR}/mymethod_ablation_summary_$(date +%Y%m%d_%H%M%S).jsonl"
RUN_LOG="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}" "${JSON_DIR}" "${PRED_DIR}"

echo "ABLATION_SCRIPT=${ABLATION_SCRIPT}" | tee -a "${RUN_LOG}"
echo "FAMILY=${FAMILY}" | tee -a "${RUN_LOG}"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${RUN_LOG}"
echo "COMPRESS_RATIO=${COMPRESS_RATIO}" | tee -a "${RUN_LOG}"
echo "TAU_MYMETHOD=${TAU_MYMETHOD}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_SELECTOR=${MYMETHOD_SELECTOR}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_SINK_SIZE=${MYMETHOD_SINK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_RECENT_WINDOW=${MYMETHOD_RECENT_WINDOW}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_RECENCY_BETA=${MYMETHOD_RECENCY_BETA}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_DISTANCE_CHUNK_SIZE=${MYMETHOD_DISTANCE_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_ATTENTION_CHUNK_SIZE=${MYMETHOD_ATTENTION_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_SINK_CAN_ABSORB=${MYMETHOD_SINK_CAN_ABSORB}" | tee -a "${RUN_LOG}"
echo "VARIANTS=${VARIANTS[*]}" | tee -a "${RUN_LOG}"
echo "TASKS=${TASKS[*]}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

run_ppl() {
  local variant="$1"
  local tag="${FAMILY}_ppl_mymethod_${variant}_ratio${COMPRESS_RATIO}_tau${TAU_MYMETHOD}_selector${MYMETHOD_SELECTOR}"
  tag="$(printf '%s' "${tag}" | tr '.-' 'p_')"
  local log_file="${LOG_DIR}/${tag}.log"
  local json_file="${JSON_DIR}/${tag}.json"

  python "${ABLATION_SCRIPT}" \
    --task ppl \
    --variant "${variant}" \
    --model_name "${BASE_MODEL}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --family "${FAMILY}" \
    --datasets wikitext2 \
    --kv_method mymethod \
    --compress_ratio "${COMPRESS_RATIO}" \
    --tau_mymethod "${TAU_MYMETHOD}" \
    --mymethod_selector "${MYMETHOD_SELECTOR}" \
    --mymethod_sink_size "${MYMETHOD_SINK_SIZE}" \
    --mymethod_recent_window "${MYMETHOD_RECENT_WINDOW}" \
    --mymethod_recency_beta "${MYMETHOD_RECENCY_BETA}" \
    --mymethod_distance_chunk_size "${MYMETHOD_DISTANCE_CHUNK_SIZE}" \
    --mymethod_attention_chunk_size "${MYMETHOD_ATTENTION_CHUNK_SIZE}" \
    --mymethod_sink_can_absorb "${MYMETHOD_SINK_CAN_ABSORB}" \
    --model_seqlen "${MODEL_SEQLEN}" \
    --chunk_size "${CHUNK_SIZE}" \
    --min_kv_len "${MIN_KV_LEN}" \
    --compress_every "${COMPRESS_EVERY}" \
    --max_eval_sequences "${MAX_EVAL_SEQUENCES}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
    --seed "${SEED}" \
    --output_json "${json_file}" \
    > "${log_file}" 2>&1 || {
      echo "[FAILED] ${tag}. See ${log_file}" | tee -a "${RUN_LOG}"
      return 0
    }

  if [[ -f "${json_file}" ]]; then
    python - <<PY >> "${SUMMARY_FILE}"
import json
p = "${json_file}"
with open(p, "r", encoding="utf-8") as f:
    value = json.load(f)
rows = value if isinstance(value, list) else [value]
for row in rows:
    row["task"] = "ppl"
    row["ablation_variant"] = "${variant}"
    row["_json_file"] = p
    print(json.dumps(row, ensure_ascii=False))
PY
    echo "[OK] ${tag}" | tee -a "${RUN_LOG}"
  fi
}

run_longbench() {
  local variant="$1"
  local tag="${FAMILY}_longbench_mymethod_${variant}_ratio${COMPRESS_RATIO}_tau${TAU_MYMETHOD}_selector${MYMETHOD_SELECTOR}"
  tag="$(printf '%s' "${tag}" | tr '.-' 'p_')"
  local log_file="${LOG_DIR}/${tag}.log"
  local pred_path="${PRED_DIR}/${tag}"

  python "${ABLATION_SCRIPT}" \
    --task longbench \
    --variant "${variant}" \
    --model "${INSTRUCT_MODEL_NAME}" \
    --model_path "${INSTRUCT_MODEL_ID}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --arch "${FAMILY}" \
    --datasets "${LONGBENCH_DATASETS}" \
    --kv_method mymethod \
    --compress_ratio "${COMPRESS_RATIO}" \
    --tau_mymethod "${TAU_MYMETHOD}" \
    --mymethod_selector "${MYMETHOD_SELECTOR}" \
    --mymethod_sink_size "${MYMETHOD_SINK_SIZE}" \
    --mymethod_recent_window "${MYMETHOD_RECENT_WINDOW}" \
    --mymethod_recency_beta "${MYMETHOD_RECENCY_BETA}" \
    --mymethod_distance_chunk_size "${MYMETHOD_DISTANCE_CHUNK_SIZE}" \
    --mymethod_attention_chunk_size "${MYMETHOD_ATTENTION_CHUNK_SIZE}" \
    --mymethod_sink_can_absorb "${MYMETHOD_SINK_CAN_ABSORB}" \
    --out_path "${pred_path}" \
    --dtype "${TORCH_DTYPE}" \
    --device "${DEVICE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}" \
    --seed "${SEED}" \
    > "${log_file}" 2>&1 || {
      echo "[FAILED] ${tag}. See ${log_file}" | tee -a "${RUN_LOG}"
      return 0
    }

  python - <<PY >> "${SUMMARY_FILE}"
import json
print(json.dumps({
    "task": "longbench",
    "ablation_variant": "${variant}",
    "model_name": "${INSTRUCT_MODEL_NAME}",
    "model_id": "${INSTRUCT_MODEL_ID}",
    "compress_ratio": float("${COMPRESS_RATIO}"),
    "tau_mymethod": float("${TAU_MYMETHOD}"),
    "prediction_dir": "${pred_path}"
}, ensure_ascii=False))
PY
}

for task in "${TASKS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "========== ${task} | ${variant} ==========" | tee -a "${RUN_LOG}"
    if [[ "${task}" == "ppl" ]]; then
      run_ppl "${variant}"
    else
      run_longbench "${variant}"
    fi
  done
done

run_oracle() {
  local tag="${FAMILY}_oracle_mymethod_${ORACLE_DATASET}_ratio${ORACLE_COMPRESS_RATIO}_tau${TAU_MYMETHOD}_selector${MYMETHOD_SELECTOR}"
  tag="$(printf '%s' "${tag}" | tr '.-' 'p_')"
  local log_file="${LOG_DIR}/${tag}.log"
  local json_file="${JSON_DIR}/${tag}.json"

  python "${ABLATION_SCRIPT}" \
    --task oracle \
    --model "${INSTRUCT_MODEL_NAME}" \
    --model_path "${INSTRUCT_MODEL_ID}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --arch "${FAMILY}" \
    --oracle_dataset "${ORACLE_DATASET}" \
    --oracle_input_length "${ORACLE_INPUT_LENGTH}" \
    --compress_ratio "${ORACLE_COMPRESS_RATIO}" \
    --tau_mymethod "${TAU_MYMETHOD}" \
    --mymethod_selector "${MYMETHOD_SELECTOR}" \
    --mymethod_sink_size "${MYMETHOD_SINK_SIZE}" \
    --mymethod_recent_window "${MYMETHOD_RECENT_WINDOW}" \
    --mymethod_recency_beta "${MYMETHOD_RECENCY_BETA}" \
    --mymethod_distance_chunk_size "${MYMETHOD_DISTANCE_CHUNK_SIZE}" \
    --mymethod_attention_chunk_size "${MYMETHOD_ATTENTION_CHUNK_SIZE}" \
    --mymethod_sink_can_absorb "${MYMETHOD_SINK_CAN_ABSORB}" \
    --oracle_num_prompts "${ORACLE_NUM_PROMPTS}" \
    --oracle_layers "${ORACLE_LAYERS}" \
    --oracle_pairs_per_layer "${ORACLE_PAIRS_PER_LAYER}" \
    --dtype "${TORCH_DTYPE}" \
    --device "${DEVICE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --seed "${SEED}" \
    --output_json "${json_file}" \
    > "${log_file}" 2>&1 || {
      echo "[FAILED] ${tag}. See ${log_file}" | tee -a "${RUN_LOG}"
      return 0
    }

  if [[ -f "${json_file}" ]]; then
    python -c 'import json; p="'"${json_file}"'"; row=json.load(open(p,encoding="utf-8")); row["_json_file"]=p; print(json.dumps(row,ensure_ascii=False))' >> "${SUMMARY_FILE}"
    echo "[OK] ${tag}" | tee -a "${RUN_LOG}"
  fi
}

if [[ "${RUN_ORACLE}" == "true" ]]; then
  echo "========== oracle_mymethod_group_logit ==========" | tee -a "${RUN_LOG}"
  run_oracle
fi

echo "Done." | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
cat "${SUMMARY_FILE}" || true
