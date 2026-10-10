#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/CohereKV

# ===== 手动修改区 =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
TASK_DIR="/home/tlxr/CohereKV/task_evaluation"
source "${TASK_DIR}/mymethod_selector_tag.sh"
OUT_DIR="/home/tlxr/CohereKV/results/mymethod_graph_mis_light_compare"
MODEL="meta-llama/Llama-3.2-3B"
FAMILY="llama"
DEVICE_MAP="auto"
DEVICE="cuda"
DTYPE="bfloat16"
MODEL_SEQLEN=8192
MAX_EVAL_SEQUENCES=2
CHUNK_SIZE=32
COMPRESS_RATIO=0.4
LAMBDA_MYMETHOD=0.5       # replace with quick-validation choice
MYMETHOD_SELECTOR="d2o" # keepkv | attention | d2o
MYMETHOD_KEEPKV_SINK_SIZE=4
MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE=128
MYMETHOD_KEEPKV_SCORE_WINDOW=128
MYMETHOD_KEEPKV_RECENCY_BETA=1.0
MYMETHOD_KEEPKV_USE_LOG_COUNT=0  # 0: official-style base score; 1: C*score (logC ablation)
MYMETHOD_D2O_SINK_SIZE=4
MYMETHOD_D2O_RECENT_SIZE=128
MYMETHOD_D2O_USE_LOG_COUNT=1
MYMETHOD_D2O_SCORE_RESET_INTERVAL=0  # 0: request cumulative; 1: reset every compression; N: every N compressions
METHODS=("none" "mymethod" "CentroidKV" "KeepKV" "SimCalKV" "CaM")
RUN_INTERNAL_PREDECESSOR=0
# ====================== 

case "${MYMETHOD_SELECTOR}" in
  attention) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag attention)" ;;
  keepkv) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag keepkv "${MYMETHOD_KEEPKV_SINK_SIZE}" "${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}" "${MYMETHOD_KEEPKV_SCORE_WINDOW}" "${MYMETHOD_KEEPKV_RECENCY_BETA}" "${MYMETHOD_KEEPKV_USE_LOG_COUNT}")" ;;
  d2o) MYMETHOD_SELECTOR_TAG="$(build_mymethod_selector_tag d2o "${MYMETHOD_D2O_SINK_SIZE}" "${MYMETHOD_D2O_RECENT_SIZE}" "${MYMETHOD_D2O_USE_LOG_COUNT}" "${MYMETHOD_D2O_SCORE_RESET_INTERVAL}")" ;;
  *) echo "Unsupported MYMETHOD_SELECTOR=${MYMETHOD_SELECTOR}" >&2; exit 2 ;;
esac

mkdir -p "${OUT_DIR}"
if [[ "${RUN_INTERNAL_PREDECESSOR}" == "1" ]]; then
  METHODS+=("GroupLogitKV")
fi
for method in "${METHODS[@]}"; do
  tag="${method}"
  extra=()
  case "${method}" in
    mymethod)
      tag="mymethod_${MYMETHOD_SELECTOR_TAG}_lambda${LAMBDA_MYMETHOD}"
      echo "[MYMETHOD_SELECTOR_TAG] ${MYMETHOD_SELECTOR_TAG}"
      extra+=(--lambda_mymethod "${LAMBDA_MYMETHOD}" --mymethod_selector "${MYMETHOD_SELECTOR}" --mymethod_group_mode full
              --mymethod_keepkv_sink_size "${MYMETHOD_KEEPKV_SINK_SIZE}" --mymethod_keepkv_recent_protect_size "${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}"
              --mymethod_keepkv_score_window "${MYMETHOD_KEEPKV_SCORE_WINDOW}" --mymethod_keepkv_recency_beta "${MYMETHOD_KEEPKV_RECENCY_BETA}"
              --mymethod_keepkv_use_log_count "${MYMETHOD_KEEPKV_USE_LOG_COUNT}"
              --mymethod_d2o_sink_size "${MYMETHOD_D2O_SINK_SIZE}" --mymethod_d2o_recent_size "${MYMETHOD_D2O_RECENT_SIZE}"
              --mymethod_d2o_use_log_count "${MYMETHOD_D2O_USE_LOG_COUNT}" --mymethod_d2o_score_reset_interval "${MYMETHOD_D2O_SCORE_RESET_INTERVAL}"
              --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128)
      ;;
    GroupLogitKV)
      # Internal unpublished predecessor; do not treat as an external published baseline.
      extra+=(--sim_thr_grouplogitkv 0.8)
      ;;
    SimCalKV) extra+=(--sim_thr_simcalkv -1) ;;
    KeepKV) extra+=(--sim_thr_keepkv 0.8) ;;
  esac
  python "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
    --model_name "${MODEL}" --family "${FAMILY}" --task_eval_dir "${TASK_DIR}" \
    --datasets wikitext2 --kv_method "${method}" --compress_ratio "${COMPRESS_RATIO}" \
    --model_seqlen "${MODEL_SEQLEN}" --chunk_size "${CHUNK_SIZE}" --compress_every 1 \
    --max_eval_sequences "${MAX_EVAL_SEQUENCES}" --torch_dtype "${DTYPE}" --amp_dtype none \
    --device "${DEVICE}" --device_map "${DEVICE_MAP}" --attn_implementation eager \
    --output_json "${OUT_DIR}/${tag}.json" "${extra[@]}" \
    2>&1 | tee "${OUT_DIR}/${tag}.log"
done

bash "${TASK_DIR}/package_mymethod_feedback.sh" "${OUT_DIR}"
