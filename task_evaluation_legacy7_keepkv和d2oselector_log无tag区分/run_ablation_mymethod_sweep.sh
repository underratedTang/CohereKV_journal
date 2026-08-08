#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/SimCalKV

# ===== 手动修改区 =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
SCRIPT="/home/tlxr/SimCalKV/task_evaluation/evaluate_ablation_mymethod.py"
OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_graph_mis_ablation"
FAMILY="llama"
MODEL="meta-llama/Llama-3.2-3B"
LAMBDA_MYMETHOD=0.5
MYMETHOD_SELECTOR="d2o"
MYMETHOD_KEEPKV_SINK_SIZE=4
MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE=128
MYMETHOD_KEEPKV_SCORE_WINDOW=128
MYMETHOD_KEEPKV_RECENCY_BETA=1.0
MYMETHOD_KEEPKV_USE_LOG_COUNT=0  # 0: official-style base score; 1: C*score (logC ablation)
MYMETHOD_D2O_SINK_SIZE=4
MYMETHOD_D2O_RECENT_SIZE=128
MYMETHOD_D2O_USE_LOG_COUNT=1
MYMETHOD_D2O_SCORE_RESET_INTERVAL=0  # 0: request cumulative; 1: reset every compression; N: every N compressions
MYMETHOD_GRAPH_BACKEND="cuda"
MYMETHOD_ROUTE_BACKEND="torch"
RATIO=0.4
MAX_EVAL_SEQUENCES=2
MODEL_SEQLEN=8192
CHUNK_SIZE=32
VARIANTS=("eviction_only" "anchor_only" "full_no_count" "full")
# ======================

mkdir -p "${OUT_DIR}"
for variant in "${VARIANTS[@]}"; do
  echo "[ablation] ${variant}"
  python "${SCRIPT}" --mode ppl --variant "${variant}" \
    --model_name "${MODEL}" --family "${FAMILY}" \
    --datasets wikitext2 --kv_method mymethod \
    --compress_ratio "${RATIO}" --lambda_mymethod "${LAMBDA_MYMETHOD}" \
    --mymethod_selector "${MYMETHOD_SELECTOR}" \
    --mymethod_graph_backend "${MYMETHOD_GRAPH_BACKEND}" --mymethod_route_backend "${MYMETHOD_ROUTE_BACKEND}" \
    --mymethod_keepkv_sink_size "${MYMETHOD_KEEPKV_SINK_SIZE}" --mymethod_keepkv_recent_protect_size "${MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE}" \
    --mymethod_keepkv_score_window "${MYMETHOD_KEEPKV_SCORE_WINDOW}" --mymethod_keepkv_recency_beta "${MYMETHOD_KEEPKV_RECENCY_BETA}" \
      --mymethod_keepkv_use_log_count "${MYMETHOD_KEEPKV_USE_LOG_COUNT}" \
    --mymethod_d2o_sink_size "${MYMETHOD_D2O_SINK_SIZE}" --mymethod_d2o_recent_size "${MYMETHOD_D2O_RECENT_SIZE}" \
    --mymethod_d2o_use_log_count "${MYMETHOD_D2O_USE_LOG_COUNT}" --mymethod_d2o_score_reset_interval "${MYMETHOD_D2O_SCORE_RESET_INTERVAL}" \
    --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128 --max_eval_sequences "${MAX_EVAL_SEQUENCES}" \
    --model_seqlen "${MODEL_SEQLEN}" --chunk_size "${CHUNK_SIZE}" --compress_every 1 \
    --torch_dtype bfloat16 --amp_dtype none --device cuda --device_map auto \
    --attn_implementation eager \
    --output_json "${OUT_DIR}/${variant}.json" \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/${variant}_diagnostics.jsonl" \
    2>&1 | tee "${OUT_DIR}/${variant}.log"
done
