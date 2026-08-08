#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/SimCalKV

# ===== 手动修改区 =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
TASK_DIR="/home/tlxr/SimCalKV/task_evaluation"
OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_graph_mis_light_compare"
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
MYMETHOD_SELECTOR="keepkv" # MD default; use attention for the stateless current-attention selector
METHODS=("none" "mymethod" "CentroidKV" "KeepKV" "SimCalKV" "CaM")
RUN_INTERNAL_PREDECESSOR=0
# ====================== 

mkdir -p "${OUT_DIR}"
if [[ "${RUN_INTERNAL_PREDECESSOR}" == "1" ]]; then
  METHODS+=("GroupLogitKV")
fi
for method in "${METHODS[@]}"; do
  tag="${method}"
  extra=()
  case "${method}" in
    mymethod)
      extra+=(--lambda_mymethod "${LAMBDA_MYMETHOD}" --mymethod_selector "${MYMETHOD_SELECTOR}" --mymethod_group_mode full
              --mymethod_sink_size 4 --mymethod_recent_window 128 --mymethod_recency_beta 1.0
              --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128 --mymethod_sink_can_absorb 1)
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
