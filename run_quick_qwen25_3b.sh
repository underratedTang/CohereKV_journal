#!/usr/bin/env bash
set -euo pipefail

cd /root/SimCalKV

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATASET="xsum"
SPLIT="test[:5]"
RATIO="0.4"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/quick_qwen25_3b_${DATASET}_ratio${RATIO}_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
SUMMARY_FILE="${OUT_DIR}/quick_summary.jsonl"

mkdir -p "${LOG_DIR}"

# METHODS=(
#   "SimCalKV"
#   "GroupLogitKV"
#   "KeepKV"
#   "PyramidInfer"
# )
METHODS=(
  "SimCalKV"
  "GroupLogitKV"
)

echo "Output dir: ${OUT_DIR}"
echo "Summary file: ${SUMMARY_FILE}"

for METHOD in "${METHODS[@]}"; do
  echo "========== Running ${METHOD} ==========" | tee -a "${LOG_DIR}/run_all.log"

  python task_evaluation/qwen/qwen_eval.py \
    --model_name "${MODEL}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --compress_ratio "${RATIO}" \
    --kv_method "${METHOD}" \
    --output_dir "${OUT_DIR}/${METHOD}" \
    > "${LOG_DIR}/${METHOD}.log" 2>&1 || {
      echo "[FAILED] ${METHOD}. See ${LOG_DIR}/${METHOD}.log" | tee -a "${LOG_DIR}/run_all.log"
      continue
    }

  JSON_FILE="${OUT_DIR}/${METHOD}/qwen_${DATASET}_${METHOD}_${RATIO}.json"
  if [ -f "${JSON_FILE}" ]; then
    python - <<PY >> "${SUMMARY_FILE}"
import json
p = "${JSON_FILE}"
with open(p, "r") as f:
    x = json.load(f)
x["_json_file"] = p
print(json.dumps(x, ensure_ascii=False))
PY
    echo "[OK] ${METHOD}" | tee -a "${LOG_DIR}/run_all.log"
  else
    echo "[MISSING_JSON] ${METHOD}: ${JSON_FILE}" | tee -a "${LOG_DIR}/run_all.log"
  fi
done

echo
echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "SUMMARY_FILE=${SUMMARY_FILE}"
echo
echo "Preview:"
cat "${SUMMARY_FILE}" || true