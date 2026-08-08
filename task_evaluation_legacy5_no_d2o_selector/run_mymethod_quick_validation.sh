#!/usr/bin/env bash
set -euo pipefail
cd /home/tlxr/SimCalKV

# ===== 手动修改区 =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
TASK_DIR="/home/tlxr/SimCalKV/task_evaluation"
OUT_DIR="/home/tlxr/SimCalKV/results/mymethod_graph_mis_quick"
MODEL="meta-llama/Llama-3.2-3B"
FAMILY="llama"                 # llama | qwen
DEVICE_MAP="auto"
DEVICE="cuda"
DTYPE="bfloat16"
MODEL_SEQLEN=8192
PPL_MAX_SEQUENCES=2
PPL_CHUNK_SIZE=32
COMPRESS_RATIO=0.4
# Geometry scan is cheap after the snapshot is captured.
LAMBDA_SCAN="0.3,0.4,0.5,0.75,1.0"
# PPL is intentionally small. These are provisional candidates, not theory defaults.
PPL_LAMBDAS=("0.5" "0.75")
RUN_CAPTURE=1
RUN_GEOMETRY=1
RUN_BENCHMARK=1
RUN_PPL=1
# ====================== 

mkdir -p "${OUT_DIR}"
SNAPSHOT="${OUT_DIR}/kv_snapshot.pt"

if [[ "${RUN_CAPTURE}" == "1" ]]; then
  python "${TASK_DIR}/capture_mymethod_kv_snapshot.py" \
    --model_name "${MODEL}" --family "${FAMILY}" \
    --dataset wikitext2 --model_seqlen 2048 --sample_index 0 \
    --layers first,mid,last --torch_dtype "${DTYPE}" \
    --device_map "${DEVICE_MAP}" --attn_implementation eager \
    --output "${SNAPSHOT}" 2>&1 | tee "${OUT_DIR}/capture.log"
fi

if [[ "${RUN_GEOMETRY}" == "1" ]]; then
  python "${TASK_DIR}/analyze_mymethod_graph_mis.py" \
    --snapshot "${SNAPSHOT}" --compress_ratio "${COMPRESS_RATIO}" \
    --lambdas "${LAMBDA_SCAN}" --distance_chunk_size 256 \
    --mis_oracle_max_pool 14 --mis_oracle_max_groups 200 \
    --output_json "${OUT_DIR}/graph_mis_analysis.json" \
    --output_csv "${OUT_DIR}/graph_mis_analysis.csv" \
    2>&1 | tee "${OUT_DIR}/graph_mis_analysis.log"
fi

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  python "${TASK_DIR}/benchmark_mymethod_grouping.py" \
    --snapshot "${SNAPSHOT}" --compress_ratio "${COMPRESS_RATIO}" \
    --lambda_mymethod 0.5 --device "${DEVICE}" --warmup 3 --repeat 10 \
    --output_json "${OUT_DIR}/grouping_benchmark.json" \
    2>&1 | tee "${OUT_DIR}/grouping_benchmark.log"
fi

run_ppl() {
  local tag="$1"; shift
  python "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
    --model_name "${MODEL}" --family "${FAMILY}" --task_eval_dir "${TASK_DIR}" \
    --datasets wikitext2 --compress_ratio "${COMPRESS_RATIO}" \
    --model_seqlen "${MODEL_SEQLEN}" --chunk_size "${PPL_CHUNK_SIZE}" \
    --compress_every 1 --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
    --torch_dtype "${DTYPE}" --amp_dtype none --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" --attn_implementation eager \
    --output_json "${OUT_DIR}/ppl_${tag}.json" \
    "$@" 2>&1 | tee "${OUT_DIR}/ppl_${tag}.log"
}

if [[ "${RUN_PPL}" == "1" ]]; then
  # Same current-attention split: isolate eviction -> anchor-only -> graph-MIS.
  run_ppl "internal_eviction_lambda0.5" \
    --kv_method mymethod --lambda_mymethod 0.5 \
    --mymethod_selector attention --mymethod_group_mode eviction_only \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_internal_eviction_lambda0.5_diagnostics.jsonl"
  run_ppl "internal_anchor_lambda0.5" \
    --kv_method mymethod --lambda_mymethod 0.5 \
    --mymethod_selector attention --mymethod_group_mode anchor_only \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_internal_anchor_lambda0.5_diagnostics.jsonl"
  run_ppl "internal_full_lambda0.5" \
    --kv_method mymethod --lambda_mymethod 0.5 \
    --mymethod_selector attention --mymethod_group_mode full \
    --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_internal_full_lambda0.5_diagnostics.jsonl"

  # Default MD selector: check whether the complete method remains nontrivial.
  for lam in "${PPL_LAMBDAS[@]}"; do
    run_ppl "default_full_lambda${lam}" \
      --kv_method mymethod --lambda_mymethod "${lam}" \
      --mymethod_selector keepkv --mymethod_group_mode full \
      --mymethod_sink_size 4 --mymethod_recent_window 128 --mymethod_recency_beta 1.0 \
      --mymethod_distance_chunk_size 256 --mymethod_attention_chunk_size 128 \
      --mymethod_sink_can_absorb 1 \
      --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_default_full_lambda${lam}_diagnostics.jsonl"
  done
fi

bash "${TASK_DIR}/package_mymethod_feedback.sh" "${OUT_DIR}"
