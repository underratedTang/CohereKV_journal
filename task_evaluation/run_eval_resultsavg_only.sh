#!/usr/bin/env bash
set -euo pipefail

# =========================
# 手动修改区
# =========================
USE_E=0  # 0: 普通 LongBench；1: LongBench-E

# ==== 2) 路径配置 ====
PY_SCRIPT="/home/tlxr/CohereKV/task_evaluation/eval_resultsavg_only.py"



ROOT="/home/tlxr/CohereKV/results/ablation_manual_test/meta-llama_Llama-3.2-3B/pred/llama_longbench_group_merge_no_bias_ratio0p8"


if [[ "$USE_E" -eq 1 ]]; then
  PRED_PATH="${ROOT}/pred_e"
  RESULT_PATH="${ROOT}/result_e_avg.json"
else
#   PRED_PATH="${ROOT}/pred"
  PRED_PATH="${ROOT}"

  RESULT_PATH="${ROOT}/result_avg.json"
fi

LOG_FILE="${ROOT}/resultsavg.log"

# ==== 5) 检查 ====
if [[ ! -d "$PRED_PATH" ]]; then
  echo "[ERROR] pred_path 不存在: $PRED_PATH"
  exit 1
fi

# ==== 6) 构造命令 ====
CMD=(
  python -u "$PY_SCRIPT"
  --pred_path "$PRED_PATH"
  --out_path "$RESULT_PATH"
)

if [[ "$USE_E" -eq 1 ]]; then
  CMD+=( -e )
fi

# ==== 7) run ====
echo "CMD:"
printf ' %q' "${CMD[@]}"
echo
echo "Pred path  -> $PRED_PATH"
echo "Result path -> $RESULT_PATH"
echo "Log -> $LOG_FILE"

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

echo "Started."
echo "PID: $!"