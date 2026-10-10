#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/CohereKV

# Minimal follow-up after the first graph-MIS quick validation.
# Goal: isolate selector vs grouping, then probe the higher-coverage regime.
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

TASK_DIR="/home/tlxr/CohereKV/task_evaluation"
OUT_DIR="/home/tlxr/CohereKV/results/mymethod_graph_mis_targeted"
MODEL="meta-llama/Llama-3.2-3B"
FAMILY="llama"
DEVICE_MAP="auto"
DEVICE="cuda"
DTYPE="bfloat16"
MODEL_SEQLEN=8192
MAX_EVAL_SEQUENCES=2
CHUNK_SIZE=32
COMPRESS_RATIO=0.4
SNAPSHOT="/home/tlxr/CohereKV/results/mymethod_graph_mis_quick/kv_snapshot.pt"

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

# A. Same MD-default KeepKV-style selector: isolate selector quality from grouping.
run_ppl default_eviction \
  --kv_method mymethod --lambda_mymethod 0.5 --mymethod_selector keepkv \
  --mymethod_group_mode eviction_only \
  --mymethod_diagnostics_jsonl "${OUT_DIR}/default_eviction_diagnostics.jsonl"

run_ppl default_anchor_lambda0.5 \
  --kv_method mymethod --lambda_mymethod 0.5 --mymethod_selector keepkv \
  --mymethod_group_mode anchor_only \
  --mymethod_diagnostics_jsonl "${OUT_DIR}/default_anchor_lambda0.5_diagnostics.jsonl"

# full/default lambda=0.5 and 0.75 already exist in the first feedback package.

# B. Same current-attention split: increase lambda so conflict/MIS is actually exercised.
for lam in 0.65 0.75; do
  run_ppl internal_anchor_lambda${lam} \
    --kv_method mymethod --lambda_mymethod "${lam}" --mymethod_selector attention \
    --mymethod_group_mode anchor_only \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/internal_anchor_lambda${lam}_diagnostics.jsonl"
  run_ppl internal_full_lambda${lam} \
    --kv_method mymethod --lambda_mymethod "${lam}" --mymethod_selector attention \
    --mymethod_group_mode full \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/internal_full_lambda${lam}_diagnostics.jsonl"
done

# C. Two strongest published quality references under exactly the same PPL protocol.
run_ppl baseline_KeepKV --kv_method KeepKV --sim_thr_keepkv 0.8
run_ppl baseline_CentroidKV --kv_method CentroidKV

# D. Measure graph cost again in the higher-coverage regime. No diagnostics are involved.
if [[ -f "${SNAPSHOT}" ]]; then
  for lam in 0.5 0.75; do
    python "${TASK_DIR}/benchmark_mymethod_grouping.py" \
      --snapshot "${SNAPSHOT}" --compress_ratio "${COMPRESS_RATIO}" \
      --lambda_mymethod "${lam}" --device "${DEVICE}" --warmup 3 --repeat 10 \
      --output_json "${OUT_DIR}/grouping_benchmark_lambda${lam}.json" \
      2>&1 | tee "${OUT_DIR}/grouping_benchmark_lambda${lam}.log"
  done
fi

bash "${TASK_DIR}/package_mymethod_feedback.sh" "${OUT_DIR}"
