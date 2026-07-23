#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================

cd /root/SimCalKV

# =========================
# 手动修改区
# =========================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# PPL 评测脚本完整路径
PPL_SCRIPT="/root/SimCalKV/task_evaluation/evaluate_ppl_kvmerge_aquakvstyle.py"

# 手动指定输出目录
OUT_DIR="/root/SimCalKV/results/ppl_manual_$(date +%Y%m%d_%H%M%S)"
# OUT_DIR="/root/SimCalKV/results/ppl_manual_test"

# 模型设置
FAMILY="qwen"   # qwen | llama
MODEL="Qwen/Qwen2.5-7B"
# MODEL="Qwen/Qwen2.5-3B"

# LLaMA 示例：
# FAMILY="llama"
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.2-3B"
# MODEL="meta-llama/Meta-Llama-3.1-8B"

# PPL 设置
MODEL_SEQLEN=8192
CHUNK_SIZE=32
MIN_KV_LEN=2
COMPRESS_EVERY=1

# dtype 设置
TORCH_DTYPE="bfloat16"  # auto | float16 | bfloat16 | float32
AMP_DTYPE="none"        # none | float16 | bfloat16 | float32

# 其他设置
ATTN_IMPL="eager"
DEVICE="cuda"
DEVICE_MAP="auto"
SEED=0

# 数据集：evaluate_ppl_kvmerge.py 只支持 wikitext2 / c4
DATASETS=(
  "wikitext2"
  # "c4"
)

# 方法
METHODS=(
  # "none"
  "GroupLogitKV"
  "KeepKV"
  # "CaM"
  "SimCalKV"
)

# 压缩率
RATIOS=(
  # "0.2"
  # "0.4"  # 压缩40%，保留 60%
  "0.6"
  # "0.8"
)

# Per-method similarity-threshold sweeps. Only these three methods use sim_thr.
# -1 disables similarity-threshold rejection.
# SIM_THR_GROUPLOGITKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_SIMCALKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_KEEPKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
SIM_THR_GROUPLOGITKV_LIST=("0.8")
SIM_THR_SIMCALKV_LIST=("0.0")
SIM_THR_KEEPKV_LIST=("0.8")

# =========================
# 自动区：一般不用改
# =========================

if [[ ! -f "${PPL_SCRIPT}" ]]; then
  echo "[ERROR] PPL_SCRIPT not found: ${PPL_SCRIPT}"
  exit 1
fi

TASK_EVAL_DIR="$(cd "$(dirname "${PPL_SCRIPT}")" && pwd)"

# 按模型区分输出目录
# 例：
# Qwen/Qwen2.5-3B                    -> Qwen_Qwen2.5-3B
# Qwen/Qwen2.5-3B-Instruct           -> Qwen_Qwen2.5-3B-Instruct
# meta-llama/Llama-3.2-1B-Instruct   -> meta-llama_Llama-3.2-1B-Instruct
MODEL_SLUG="$(printf '%s' "${MODEL}" | tr '/:' '_')"
MODEL_OUT_DIR="${OUT_DIR}/${MODEL_SLUG}"

LOG_DIR="${MODEL_OUT_DIR}/logs"
JSON_DIR="${MODEL_OUT_DIR}/json"
SUMMARY_FILE="${MODEL_OUT_DIR}/ppl_summary_$(date +%Y%m%d_%H%M%S).jsonl"
RUN_LOG="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" "${JSON_DIR}"

# 避免固定 OUT_DIR 时旧 summary 混入新结果
# rm -f "${SUMMARY_FILE}" # 不再删除旧 summary

echo "PPL_SCRIPT=${PPL_SCRIPT}" | tee -a "${RUN_LOG}"
echo "TASK_EVAL_DIR=${TASK_EVAL_DIR}" | tee -a "${RUN_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "MODEL_SLUG=${MODEL_SLUG}" | tee -a "${RUN_LOG}"
echo "MODEL_OUT_DIR=${MODEL_OUT_DIR}" | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
echo "FAMILY=${FAMILY}" | tee -a "${RUN_LOG}"
echo "MODEL=${MODEL}" | tee -a "${RUN_LOG}"
echo "MODEL_SEQLEN=${MODEL_SEQLEN}" | tee -a "${RUN_LOG}"
echo "CHUNK_SIZE=${CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "MIN_KV_LEN=${MIN_KV_LEN}" | tee -a "${RUN_LOG}"
echo "COMPRESS_EVERY=${COMPRESS_EVERY}" | tee -a "${RUN_LOG}"
echo "TORCH_DTYPE=${TORCH_DTYPE}" | tee -a "${RUN_LOG}"
echo "AMP_DTYPE=${AMP_DTYPE}" | tee -a "${RUN_LOG}"
echo "ATTN_IMPL=${ATTN_IMPL}" | tee -a "${RUN_LOG}"
echo "DEVICE=${DEVICE}" | tee -a "${RUN_LOG}"
echo "DEVICE_MAP=${DEVICE_MAP}" | tee -a "${RUN_LOG}"
echo "SEED=${SEED}" | tee -a "${RUN_LOG}"
echo "DATASETS=${DATASETS[*]}" | tee -a "${RUN_LOG}"
echo "METHODS=${METHODS[*]}" | tee -a "${RUN_LOG}"
echo "RATIOS=${RATIOS[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_GROUPLOGITKV_LIST=${SIM_THR_GROUPLOGITKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_SIMCALKV_LIST=${SIM_THR_SIMCALKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_KEEPKV_LIST=${SIM_THR_KEEPKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

run_one() {
  local dataset="$1"
  local method="$2"
  local ratio="$3"
  local thr="${4:-na}"

  local sim_tag="na"
  local extra_args=()

  if [[ "${method}" == "GroupLogitKV" ]]; then
    sim_tag="${thr}"
    extra_args+=(--sim_thr_grouplogitkv "${thr}")
  elif [[ "${method}" == "SimCalKV" ]]; then
    sim_tag="${thr}"
    extra_args+=(--sim_thr_simcalkv "${thr}")
  elif [[ "${method}" == "KeepKV" ]]; then
    sim_tag="${thr}"
    extra_args+=(--sim_thr_keepkv "${thr}")
  fi

  local tag="${FAMILY}_${dataset}_${method}_ratio${ratio}_sim${sim_tag}"
  tag="$(echo "${tag}" | tr '.-' 'p_')"

  local log_file="${LOG_DIR}/${tag}.log"
  local json_file="${JSON_DIR}/${tag}.json"

  echo "========== ${dataset} | ${method} | ratio=${ratio} | sim=${sim_tag} ==========" | tee -a "${RUN_LOG}"

  python "${PPL_SCRIPT}" \
    --model_name "${MODEL}" \
    --task_eval_dir "${TASK_EVAL_DIR}" \
    --family "${FAMILY}" \
    --datasets "${dataset}" \
    --kv_method "${method}" \
    --compress_ratio "${ratio}" \
    "${extra_args[@]}" \
    --model_seqlen "${MODEL_SEQLEN}" \
    --chunk_size "${CHUNK_SIZE}" \
    --min_kv_len "${MIN_KV_LEN}" \
    --compress_every "${COMPRESS_EVERY}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
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
    x = json.load(f)

if isinstance(x, list):
    for row in x:
        row["_json_file"] = p
        row["_model"] = "${MODEL}"
        row["_model_slug"] = "${MODEL_SLUG}"
        print(json.dumps(row, ensure_ascii=False))
else:
    x["_json_file"] = p
    x["_model"] = "${MODEL}"
    x["_model_slug"] = "${MODEL_SLUG}"
    print(json.dumps(x, ensure_ascii=False))
PY
    echo "[OK] ${tag}" | tee -a "${RUN_LOG}"
  else
    echo "[MISSING_JSON] ${tag}: ${json_file}" | tee -a "${RUN_LOG}"
  fi
}

# =========================
# 主循环
# =========================

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do

    # baseline 不压缩，不需要 ratio / sim_thr
    if [[ "${method}" == "none" || "${method}" == "baseline" ]]; then
      run_one "${dataset}" "none" "0"
      continue
    fi

    for ratio in "${RATIOS[@]}"; do
      if [[ "${method}" == "GroupLogitKV" ]]; then
        for thr in "${SIM_THR_GROUPLOGITKV_LIST[@]}"; do
          run_one "${dataset}" "${method}" "${ratio}" "${thr}"
        done
      elif [[ "${method}" == "SimCalKV" ]]; then
        for thr in "${SIM_THR_SIMCALKV_LIST[@]}"; do
          run_one "${dataset}" "${method}" "${ratio}" "${thr}"
        done
      elif [[ "${method}" == "KeepKV" ]]; then
        for thr in "${SIM_THR_KEEPKV_LIST[@]}"; do
          run_one "${dataset}" "${method}" "${ratio}" "${thr}"
        done
      else
        run_one "${dataset}" "${method}" "${ratio}" "na"
      fi
    done
  done
done

echo | tee -a "${RUN_LOG}"
echo "Done." | tee -a "${RUN_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "MODEL_SLUG=${MODEL_SLUG}" | tee -a "${RUN_LOG}"
echo "MODEL_OUT_DIR=${MODEL_OUT_DIR}" | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

echo "Preview:" | tee -a "${RUN_LOG}"
cat "${SUMMARY_FILE}" || true