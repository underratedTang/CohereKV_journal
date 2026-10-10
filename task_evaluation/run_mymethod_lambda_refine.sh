#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/CohereKV

# Minimal lambda refinement after graph-MIS targeted validation.
# WikiText-2 only, two 8192-token sequences. No public baselines are rerun.
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

TASK_DIR="/home/tlxr/CohereKV/task_evaluation"
OUT_DIR="/home/tlxr/CohereKV/results/mymethod_graph_mis_lambda_refine"
MODEL="meta-llama/Llama-3.2-3B"
FAMILY="llama"
DEVICE_MAP="auto"
DEVICE="cuda"
DTYPE="bfloat16"
MODEL_SEQLEN=8192
MAX_EVAL_SEQUENCES=2
CHUNK_SIZE=32
COMPRESS_RATIO=0.4

mkdir -p "${OUT_DIR}"

run_ppl() {
  local tag="$1"; shift
  python "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
    --model_name "${MODEL}" --family "${FAMILY}" --task_eval_dir "${TASK_DIR}" \
    --datasets wikitext2 --compress_ratio "${COMPRESS_RATIO}" \
    --model_seqlen "${MODEL_SEQLEN}" --chunk_size "${CHUNK_SIZE}" --compress_every 1 \
    --max_eval_sequences "${MAX_EVAL_SEQUENCES}" --torch_dtype "${DTYPE}" --amp_dtype none \
    --device "${DEVICE}" --device_map "${DEVICE_MAP}" --attn_implementation eager \
    --output_json "${OUT_DIR}/${tag}.json" "$@" \
    2>&1 | tee "${OUT_DIR}/${tag}.log"
}

# Same retained/evicted split; only grouping constraint changes.
for lam in 0.85 0.95; do
  run_ppl internal_anchor_lambda${lam} \
    --kv_method mymethod --lambda_mymethod "${lam}" --mymethod_selector attention \
    --mymethod_group_mode anchor_only \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/internal_anchor_lambda${lam}_diagnostics.jsonl"
  run_ppl internal_full_lambda${lam} \
    --kv_method mymethod --lambda_mymethod "${lam}" --mymethod_selector attention \
    --mymethod_group_mode full \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/internal_full_lambda${lam}_diagnostics.jsonl"
done

bash "${TASK_DIR}/package_mymethod_feedback.sh" "${OUT_DIR}"
