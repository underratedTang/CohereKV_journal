#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/SimCalKV

# ===== 手动修改区 =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
SCRIPT="/home/tlxr/SimCalKV/task_evaluation/evaluate_ablation_mymethod.py"
# OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_graph_mis_ablation"
OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_graph_mis_ablation_$(date +%Y%m%d_%H%M%S)"

# FAMILY="llama"
# MODEL="meta-llama/Llama-3.2-3B"
FAMILY="qwen"   # qwen | llama
MODEL="Qwen/Qwen2.5-7B"

LAMBDA_MYMETHOD=0.5
# MYMETHOD_SELECTOR="keepkv"
MYMETHOD_SELECTOR="attention"

MYMETHOD_GRAPH_BACKEND="cuda"
MYMETHOD_ROUTE_BACKEND="torch"
RATIO=0.6
MAX_EVAL_SEQUENCES=2
MODEL_SEQLEN=8192
CHUNK_SIZE=32
# VARIANTS=("eviction_only" "anchor_only" "full_no_count" "full")
VARIANTS=("eviction_only" "full")

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
    --mymethod_sink_size 4 --mymethod_recent_window 128 --mymethod_recency_beta 1.0 \
    --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128 \
    --mymethod_sink_can_absorb 1 --max_eval_sequences "${MAX_EVAL_SEQUENCES}" \
    --model_seqlen "${MODEL_SEQLEN}" --chunk_size "${CHUNK_SIZE}" --compress_every 1 \
    --torch_dtype bfloat16 --amp_dtype none --device cuda --device_map auto \
    --attn_implementation eager \
    --output_json "${OUT_DIR}/${variant}.json" \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/${variant}_diagnostics.jsonl" \
    2>&1 | tee "${OUT_DIR}/${variant}.log"
done
