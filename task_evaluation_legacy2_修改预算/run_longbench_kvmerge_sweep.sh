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

# LongBench 评测脚本完整路径
# 你的文件放在：/root/SimCalKV/task_evaluation/evaluate_longbench_kvmerge.py
LONG_SCRIPT="/root/SimCalKV/task_evaluation/evaluate_longbench_kvmerge.py"

# 手动指定输出目录
# 如果不想覆盖旧结果，可以在末尾加时间戳
# OUT_DIR="/root/SimCalKV/results/longbench_manual_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/root/SimCalKV/results/longbench_manual_test"

# 模型设置
# FAMILY="qwen"   # qwen | llama
# MODEL_NAME="Qwen2.5-7B-Instruct"                 # LongBench/config/model2path.json 中的 key
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# LLaMA 示例：
FAMILY="llama"
# MODEL_NAME="llama-3.2-1B-Instruct"                 # LongBench/config/model2path.json 中的 key
# MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="llama-3.1-8B-Instruct"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_SLUG="$(printf '%s' "${MODEL_PATH}" | tr '/:' '_')"

# LongBench v1 数据集
# 想少跑几个时，直接删短这个逗号分隔字符串即可。
# DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"
DATASETS="narrativeqa"

RUN_ALL=0     # 1: 使用 -a 跑全部 DATASETS；0: 使用 --datasets "$DATASETS"
USE_E=0       # 1: 跑 LongBench-E

# dtype / runtime 设置
TORCH_DTYPE="bfloat16"    # auto | float16 | bfloat16
ATTN_IMPL="eager"   # mixed run is OK: kv_method=None is internally switched to sdpa
DEVICE="cuda"

# 其他设置
SEED=0

# 是否覆盖已有 pred jsonl
OVERWRITE=0

# 方法
METHODS=(
  "none"      # full-precision KV baseline, no compression
  "SimCalKV"
  "GroupLogitKV"
  "KeepKV"
  "CaM"
)

# 压缩率。method=None 时不 sweep RATIOS，只跑一次 ratio=0。
RATIOS=(
  # "0.2"
  "0.4"  # 压缩40%，保留 60%
  # "0.6"
  # "0.8"
)

# Per-method similarity-threshold sweeps. Only these three methods use sim_thr.
# -1 disables similarity-threshold rejection.
# KeepKV's official setting T=0.8 is included in the sweep below.
# SIM_THR_GROUPLOGITKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_SIMCALKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_KEEPKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
SIM_THR_GROUPLOGITKV_LIST=("0.8")
SIM_THR_SIMCALKV_LIST=("-1")
SIM_THR_KEEPKV_LIST=("0.8")

# =========================
# 自动区：一般不用改
# =========================

if [[ ! -f "${LONG_SCRIPT}" ]]; then
  echo "[ERROR] LONG_SCRIPT not found: ${LONG_SCRIPT}"
  exit 1
fi

# 因为 evaluate_longbench_kvmerge.py 放在 task_evaluation/ 下，
# 所以 task_eval_dir 就是该脚本所在目录。
TASK_EVAL_DIR="$(cd "$(dirname "${LONG_SCRIPT}")" && pwd)"

PRED_ROOT="${OUT_DIR}/pred"
MODEL_OUT_ROOT="${PRED_ROOT}/${FAMILY}/${MODEL_SLUG}"
LOG_DIR="${OUT_DIR}/logs/${FAMILY}/${MODEL_SLUG}"
SUMMARY_FILE="${MODEL_OUT_ROOT}/longbench_summary.jsonl"
RUN_LOG="${LOG_DIR}/run_all.log"

mkdir -p "${LOG_DIR}" "${MODEL_OUT_ROOT}"

# run_all.log: overwrite on each script launch
: > "${RUN_LOG}"
# 避免固定 OUT_DIR 时当前模型旧 summary 混入新结果
# 保留原来的 rm -f 行为，但立刻创建空 summary，方便确认路径。
rm -f "${SUMMARY_FILE}"
touch "${SUMMARY_FILE}"

echo "LONG_SCRIPT=${LONG_SCRIPT}" | tee -a "${RUN_LOG}"
echo "TASK_EVAL_DIR=${TASK_EVAL_DIR}" | tee -a "${RUN_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "LOG_DIR=${LOG_DIR}" | tee -a "${RUN_LOG}"
echo "MODEL_OUT_ROOT=${MODEL_OUT_ROOT}" | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
echo "FAMILY=${FAMILY}" | tee -a "${RUN_LOG}"
echo "MODEL_NAME=${MODEL_NAME}" | tee -a "${RUN_LOG}"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${RUN_LOG}"
echo "DATASETS=${DATASETS}" | tee -a "${RUN_LOG}"
echo "RUN_ALL=${RUN_ALL}" | tee -a "${RUN_LOG}"
echo "USE_E=${USE_E}" | tee -a "${RUN_LOG}"
echo "TORCH_DTYPE=${TORCH_DTYPE}" | tee -a "${RUN_LOG}"
echo "ATTN_IMPL=${ATTN_IMPL}" | tee -a "${RUN_LOG}"
echo "DEVICE=${DEVICE}" | tee -a "${RUN_LOG}"
echo "SEED=${SEED}" | tee -a "${RUN_LOG}"
echo "OVERWRITE=${OVERWRITE}" | tee -a "${RUN_LOG}"
echo "METHODS=${METHODS[*]}" | tee -a "${RUN_LOG}"
echo "RATIOS=${RATIOS[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_GROUPLOGITKV_LIST=${SIM_THR_GROUPLOGITKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_SIMCALKV_LIST=${SIM_THR_SIMCALKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_KEEPKV_LIST=${SIM_THR_KEEPKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

run_one() {
  local method="$1"
  local ratio="$2"
  local thr="${3:-na}"

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

  local tag="${FAMILY}_${MODEL_SLUG}_${method}_ratio${ratio}_sim${sim_tag}"
  tag=$(echo "${tag}" | tr '/:.-' '____')

  local combo_root="${MODEL_OUT_ROOT}/${method}/ratio_${ratio}/sim_${sim_tag}"
  local pred_path
  local result_file
  local result_avg_file
  local log_file

  if [[ "${USE_E}" -eq 1 ]]; then
    pred_path="${combo_root}/pred_e"
    result_file="${combo_root}/result_e.json"
    result_avg_file="${combo_root}/result_e_avg.jsonl"
    log_file="${LOG_DIR}/${tag}_e.log"
  else
    pred_path="${combo_root}/pred"
    result_file="${combo_root}/result.json"
    result_avg_file="${combo_root}/result_avg.jsonl"
    log_file="${LOG_DIR}/${tag}.log"
  fi

  mkdir -p "${pred_path}"

  echo "========== ${method} | ratio=${ratio} | sim=${sim_tag} ==========" | tee -a "${RUN_LOG}"
  echo "Predictions -> ${pred_path}" | tee -a "${RUN_LOG}"
  echo "Result      -> ${result_file}" | tee -a "${RUN_LOG}"
  echo "Result avg  -> ${result_avg_file}" | tee -a "${RUN_LOG}"
  echo "Log         -> ${log_file}" | tee -a "${RUN_LOG}"

  CMD=(
    python -u "${LONG_SCRIPT}"
    --task_eval_dir "${TASK_EVAL_DIR}"
    --arch "${FAMILY}"
    --model "${MODEL_NAME}"
    --model_path "${MODEL_PATH}"
    --kv_method "${method}"
    --compress_ratio "${ratio}"
    "${extra_args[@]}"
    --out_path "${pred_path}"
    --dtype "${TORCH_DTYPE}"
    --device "${DEVICE}"
    --attn_implementation "${ATTN_IMPL}"
    --seed "${SEED}"
  )

  if [[ "${RUN_ALL}" -eq 1 ]]; then
    CMD+=( -a )
  else
    CMD+=( --datasets "${DATASETS}" )
  fi

  if [[ "${USE_E}" -eq 1 ]]; then
    CMD+=( -e )
  fi

  if [[ "${OVERWRITE}" -eq 1 ]]; then
    CMD+=( --overwrite )
  fi

  echo "CMD:" | tee -a "${RUN_LOG}"
  printf ' %q' "${CMD[@]}" | tee -a "${RUN_LOG}"
  echo | tee -a "${RUN_LOG}"

  "${CMD[@]}" > "${log_file}" 2>&1 || {
    echo "[FAILED] ${tag}. See ${log_file}" | tee -a "${RUN_LOG}"
    return 0
  }

  if [[ -f "${result_file}" ]]; then
    avg_score=$(python - <<PY
import json
p = "${result_file}"
with open(p, "r", encoding="utf-8") as f:
    x = json.load(f)
vals = []
for k, v in x.items():
    if isinstance(v, (int, float)):
        vals.append(float(v))
    elif isinstance(v, dict):
        # 兼容少数 result.json 里可能嵌套 {score: ...} 的情况
        for key in ("score", "f1", "accuracy", "rouge", "rouge-l"):
            if isinstance(v.get(key), (int, float)):
                vals.append(float(v[key]))
                break
avg = sum(vals) / len(vals) if vals else None
print("null" if avg is None else f"{avg:.6f}")
PY
)

    python - <<PY >> "${SUMMARY_FILE}"
import json

p = "${result_file}"
avg_p = "${result_avg_file}"
with open(p, "r", encoding="utf-8") as f:
    x = json.load(f)

vals = []
for k, v in x.items():
    if isinstance(v, (int, float)):
        vals.append(float(v))
    elif isinstance(v, dict):
        for key in ("score", "f1", "accuracy", "rouge", "rouge-l"):
            if isinstance(v.get(key), (int, float)):
                vals.append(float(v[key]))
                break
avg_score = sum(vals) / len(vals) if vals else None

row = {
    "family": "${FAMILY}",
    "model_name": "${MODEL_NAME}",
    "model_path": "${MODEL_PATH}",
    "method": "${method}",
    "compress_ratio": "${ratio}",
    "sim_thr": "${sim_tag}",
    "use_e": bool(int("${USE_E}")),
    "datasets": "${DATASETS}",
    "avg_score": avg_score,
    "scores": x,
    "_result_file": p,
    "_result_avg_file": avg_p,
    "_pred_path": "${pred_path}",
    "_log_file": "${log_file}",
}
line = json.dumps(row, ensure_ascii=False)
with open(avg_p, "w", encoding="utf-8") as f:
    f.write(line + "\n")
print(line)
PY
    echo "Average     -> ${avg_score}" | tee -a "${RUN_LOG}"
    echo "Result avg  -> ${result_avg_file}" | tee -a "${RUN_LOG}"
    echo "[OK] ${tag}" | tee -a "${RUN_LOG}"
  else
    echo "[MISSING_RESULT] ${tag}: ${result_file}" | tee -a "${RUN_LOG}"
  fi
}

# =========================
# 主循环
# =========================

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "None" || "${method}" == "none" ]]; then
    run_one "None" "0" "na"
    continue
  fi

  for ratio in "${RATIOS[@]}"; do
    if [[ "${method}" == "GroupLogitKV" ]]; then
      for thr in "${SIM_THR_GROUPLOGITKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${thr}"
      done
    elif [[ "${method}" == "SimCalKV" ]]; then
      for thr in "${SIM_THR_SIMCALKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${thr}"
      done
    elif [[ "${method}" == "KeepKV" ]]; then
      for thr in "${SIM_THR_KEEPKV_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${thr}"
      done
    else
      run_one "${method}" "${ratio}" "na"
    fi
  done
done

echo | tee -a "${RUN_LOG}"
echo "Done." | tee -a "${RUN_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "LOG_DIR=${LOG_DIR}" | tee -a "${RUN_LOG}"
echo "MODEL_OUT_ROOT=${MODEL_OUT_ROOT}" | tee -a "${RUN_LOG}"
echo "SUMMARY_FILE=${SUMMARY_FILE}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

echo "Preview:" | tee -a "${RUN_LOG}"
cat "${SUMMARY_FILE}" || true
