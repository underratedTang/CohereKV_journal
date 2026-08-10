#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================

cd /home/tlxr/SimCalKV

# =========================
# 手动修改区
# =========================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# LongBench 评测脚本完整路径
# 你的文件放在：/home/tlxr/SimCalKV/task_evaluation/evaluate_longbench_kvmerge.py
LONG_SCRIPT="/home/tlxr/SimCalKV/task_evaluation/evaluate_longbench_kvmerge.py"



# 手动指定输出目录
# 如果不想覆盖旧结果，可以在末尾加时间戳
# OUT_DIR="/home/tlxr/SimCalKV/results/longbench_manual_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/home/tlxr/SimCalKV/results/longbench_manual_test2"

# 模型设置
# FAMILY="qwen"   # qwen | llama
# MODEL_NAME="Qwen2.5-3B-Instruct"                 # LongBench/config/model2path.json 中的 key
# MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME="Qwen2.5-7B-Instruct"                 # LongBench/config/model2path.json 中的 key
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# LLaMA 示例：
FAMILY="llama"
# MODEL_NAME="llama-3.2-1B-Instruct"                 # LongBench/config/model2path.json 中的 key
# MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="llama-3.1-8B-Instruct"
# MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="llama-3.2-3B-Instruct"
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"

MODEL_SLUG="$(printf '%s' "${MODEL_PATH}" | tr '/:' '_')"

# LongBench v1 数据集
# 想少跑几个时，直接删短这个逗号分隔字符串即可。
DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"
# DATASETS="narrativeqa"

RUN_ALL=0     # 1: 使用 -a 跑全部 DATASETS；0: 使用 --datasets "$DATASETS"
USE_E=0       # 1: 跑 LongBench-E

# dtype / runtime 设置
TORCH_DTYPE="bfloat16"    # auto | float16 | bfloat16
ATTN_IMPL="eager"   # formal protocol: every method uses eager; no per-method backend override
DEVICE="cuda"
# Exact eager query tiling for long prefills. Workspace-only; does not change
# compression boundaries or switch attention backend.
EAGER_ATTENTION_CHUNK_SIZE=128

# 其他设置
SEED=0

# 是否覆盖已有 pred jsonl
OVERWRITE=0

# 方法
METHODS=(
  "none"      # full-precision KV baseline, no compression
  "GroupLogitKV"
  "mymethod"
  "SimCalKV"
  "KeepKV"
  "CaM"
  "CentroidKV"
)

# 压缩率。method=None 时不 sweep RATIOS，只跑一次 ratio=0。
RATIOS=(
  "0.2"
  "0.4"  # 压缩40%，保留 60%
  "0.6"
  "0.8"
)

# Per-method similarity-threshold sweeps. Only these three methods use sim_thr.
# -1 disables similarity-threshold rejection.
# KeepKV's official setting T=0.8 is included in the sweep below.
# SIM_THR_GROUPLOGITKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_SIMCALKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
# SIM_THR_KEEPKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
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
# 自动区：一般不用改
# =========================

if [[ ! -f "${LONG_SCRIPT}" ]]; then
  echo "[ERROR] LONG_SCRIPT not found: ${LONG_SCRIPT}"
  exit 1
fi

# 因为 evaluate_longbench_kvmerge.py 放在 task_evaluation/ 下，
# 所以 task_eval_dir 就是该脚本所在目录。
TASK_EVAL_DIR="$(cd "$(dirname "${LONG_SCRIPT}")" && pwd)"
source "${TASK_EVAL_DIR}/mymethod_selector_tag.sh"
case "${MYMETHOD_SELECTOR}" in
  attention) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag attention)" ;;
  keepkv) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag keepkv "${MYMETHOD_KEEPKV_SINK_SIZE}" "${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}" "${MYMETHOD_KEEPKV_SCORE_WINDOW}" "${MYMETHOD_KEEPKV_RECENCY_BETA}" "${MYMETHOD_KEEPKV_USE_LOG_COUNT}")" ;;
  d2o) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag d2o "${MYMETHOD_D2O_SINK_SIZE}" "${MYMETHOD_D2O_RECENT_SIZE}" "${MYMETHOD_D2O_USE_LOG_COUNT}" "${MYMETHOD_D2O_SCORE_RESET_INTERVAL}")" ;;
  *) echo "Unsupported MYMETHOD_SELECTOR=${MYMETHOD_SELECTOR}" >&2; exit 2 ;;
esac

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
echo "LAMBDA_MYMETHOD_LIST=${LAMBDA_MYMETHOD_LIST[*]}" | tee -a "${RUN_LOG}"
echo "MYMETHOD_SELECTOR=${MYMETHOD_SELECTOR}" | tee -a "${RUN_LOG}"
echo "[MYMETHOD_SELECTOR_TAG] ${MYMETHOD_SELECTOR_TAG}" | tee -a "${RUN_LOG}"
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
echo "EAGER_ATTENTION_CHUNK_SIZE=${EAGER_ATTENTION_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_SINK_BUDGET=${CENTROIDKV_SINK_BUDGET}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_LOCAL_BUDGET=${CENTROIDKV_LOCAL_BUDGET}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_CHUNK_SIZE=${CENTROIDKV_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_MERGE_RATIO=${CENTROIDKV_MERGE_RATIO}" | tee -a "${RUN_LOG}"
echo "CENTROIDKV_DECREASE_RATE=${CENTROIDKV_DECREASE_RATE}" | tee -a "${RUN_LOG}"
echo "SIM_THR_SIMCALKV_LIST=${SIM_THR_SIMCALKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo "SIM_THR_KEEPKV_LIST=${SIM_THR_KEEPKV_LIST[*]}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

run_one() {
  local method="$1"
  local ratio="$2"
  local thr="na"
  if [[ $# -ge 3 ]]; then thr="$3"; fi

  local sim_tag="na"
  local extra_args=()

  if [[ "${method}" == "GroupLogitKV" ]]; then
    sim_tag="${thr}"
    extra_args+=(--sim_thr_grouplogitkv "${thr}")
  elif [[ "${method}" == "mymethod" ]]; then
    sim_tag="${MYMETHOD_SELECTOR_TAG}__mode_${MYMETHOD_GROUP_MODE}__lambda${thr}"
    extra_args+=(
      --lambda_mymethod "${thr}"
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
    sim_tag="official"
    extra_args+=(
      --centroidkv_sink_budget "${CENTROIDKV_SINK_BUDGET}"
      --centroidkv_local_budget "${CENTROIDKV_LOCAL_BUDGET}"
      --centroidkv_chunk_size "${CENTROIDKV_CHUNK_SIZE}"
      --centroidkv_merge_ratio "${CENTROIDKV_MERGE_RATIO}"
      --centroidkv_decrease_rate "${CENTROIDKV_DECREASE_RATE}"
    )
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
    --eager_attention_chunk_size "${EAGER_ATTENTION_CHUNK_SIZE}"
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
    elif [[ "${method}" == "mymethod" ]]; then
      for lambda_value in "${LAMBDA_MYMETHOD_LIST[@]}"; do
        run_one "${method}" "${ratio}" "${lambda_value}"
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
