#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Editable configuration
# ============================================================
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
FAMILY="llama"                       # llama | qwen
OUT_DIR="${TASK_DIR}/npc2026_pre_results"
DEVICE="cuda"
DTYPE="bf16"
DEVICE_MAP="auto"

SNAPSHOT_DATASET="wikitext2"
SNAPSHOT_LENGTH=2048
SNAPSHOT_LAYERS="first,mid,last"
SNAPSHOT_HEADS_PER_LAYER=4

PPL_LENGTH=4096
PPL_CHUNK=32
PPL_RATIO=0.4
PPL_MAX_SEQUENCES=2
PPL_SELECTOR="grouplogitkv"          # controlled selector for grouping ablations
PPL_TAU=0.7

RUN_CAPTURE=1
RUN_GROUP_ANALYSIS=1
RUN_MICROBENCHMARK=1
RUN_PROFILER=0
RUN_PPL_ABLATIONS=1
RUN_TAU_PPL_SWEEP=1
RUN_SELECTOR_PPL_COMPARISON=1
PPL_TAUS=(0.1 0.2 0.3 0.4)
# ============================================================

mkdir -p "${OUT_DIR}"
SNAPSHOT="${OUT_DIR}/kv_snapshot.pt"

if [[ "${RUN_CAPTURE}" -eq 1 ]]; then
  python -u "${TASK_DIR}/capture_npc2026_kv_snapshot.py" \
    --model_name "${MODEL_NAME}" \
    --family "${FAMILY}" \
    --task_eval_dir "${TASK_DIR}" \
    --dataset "${SNAPSHOT_DATASET}" \
    --model_seqlen "${SNAPSHOT_LENGTH}" \
    --layers "${SNAPSHOT_LAYERS}" \
    --torch_dtype "${DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --attn_implementation eager \
    --output "${SNAPSHOT}" \
    2>&1 | tee "${OUT_DIR}/capture.log"
fi

if [[ "${RUN_GROUP_ANALYSIS}" -eq 1 ]]; then
  python -u "${TASK_DIR}/analyze_npc2026_grouping.py" \
    --snapshot "${SNAPSHOT}" \
    --task_eval_dir "${TASK_DIR}" \
    --compress_ratios "0.2,0.4,0.6" \
    --taus "0.6,0.7,0.8,0.9" \
    --grouplogitkv_sim_thr 0.8 \
    --max_heads_per_layer "${SNAPSHOT_HEADS_PER_LAYER}" \
    --device "${DEVICE}" \
    --multiround_retained_ratios "0.8,0.6,0.4,0.3" \
    --output_json "${OUT_DIR}/grouping_diagnostics.json" \
    2>&1 | tee "${OUT_DIR}/grouping_diagnostics.log"
fi

if [[ "${RUN_MICROBENCHMARK}" -eq 1 ]]; then
  python -u "${TASK_DIR}/benchmark_npc2026_compression.py" \
    --task_eval_dir "${TASK_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch_size 1 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --lengths "512,1024,2048,4096,8192" \
    --compress_ratios "0.2,0.4,0.6" \
    --tau "${PPL_TAU}" \
    --distance_chunk_sizes "0,256,512" \
    --warmup 5 \
    --repeats 20 \
    --output_json "${OUT_DIR}/compression_microbenchmark.json" \
    2>&1 | tee "${OUT_DIR}/compression_microbenchmark.log"
fi

if [[ "${RUN_PROFILER}" -eq 1 ]]; then
  python -u "${TASK_DIR}/profile_npc2026_mymethod.py" \
    --task_eval_dir "${TASK_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --length 4096 \
    --compress_ratio 0.4 \
    --tau "${PPL_TAU}" \
    --distance_chunk_size 0 \
    --steps 5 \
    --trace_json "${OUT_DIR}/mymethod_trace.json" \
    --table_txt "${OUT_DIR}/mymethod_profile_table.txt" \
    2>&1 | tee "${OUT_DIR}/mymethod_profiler.log"
fi

if [[ "${RUN_PPL_ABLATIONS}" -eq 1 ]]; then
  variants=(
    full
    nearest_merge_no_certificate
    certified_merge_no_count
    eviction_only
  )
  for variant in "${variants[@]}"; do
    out="${OUT_DIR}/ppl_ablation_${variant}.jsonl"
    python -u "${TASK_DIR}/evaluate_ablation_mymethod.py" \
      --task ppl \
      --variant "${variant}" \
      --model_name "${MODEL_NAME}" \
      --task_eval_dir "${TASK_DIR}" \
      --family "${FAMILY}" \
      --datasets wikitext2 \
      --kv_method mymethod \
      --compress_ratio "${PPL_RATIO}" \
      --tau_mymethod "${PPL_TAU}" \
      --mymethod_selector "${PPL_SELECTOR}" \
      --mymethod_sink_size 4 \
      --mymethod_recent_window 128 \
      --mymethod_recency_beta 1.0 \
      --mymethod_distance_chunk_size 0 \
      --mymethod_attention_chunk_size 0 \
      --mymethod_sink_can_absorb 1 \
      --model_seqlen "${PPL_LENGTH}" \
      --chunk_size "${PPL_CHUNK}" \
      --compress_every 1 \
      --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
      --torch_dtype "${DTYPE}" \
      --device "${DEVICE}" \
      --device_map "${DEVICE_MAP}" \
      --attn_implementation eager \
      --output_jsonl "${out}" \
      --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_ablation_${variant}_diagnostics.jsonl" \
      2>&1 | tee "${OUT_DIR}/ppl_ablation_${variant}.log"
  done
fi


if [[ "${RUN_TAU_PPL_SWEEP}" -eq 1 ]]; then
  for tau in "${PPL_TAUS[@]}"; do
    out="${OUT_DIR}/ppl_tau_${tau}.jsonl"
    python -u "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
      --model_name "${MODEL_NAME}" \
      --task_eval_dir "${TASK_DIR}" \
      --family "${FAMILY}" \
      --datasets wikitext2 \
      --kv_method mymethod \
      --compress_ratio "${PPL_RATIO}" \
      --tau_mymethod "${tau}" \
      --mymethod_selector "${PPL_SELECTOR}" \
      --mymethod_sink_size 4 \
      --mymethod_recent_window 128 \
      --mymethod_recency_beta 1.0 \
      --mymethod_distance_chunk_size 0 \
      --mymethod_attention_chunk_size 0 \
      --mymethod_sink_can_absorb 1 \
      --model_seqlen "${PPL_LENGTH}" \
      --chunk_size "${PPL_CHUNK}" \
      --compress_every 1 \
      --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
      --torch_dtype "${DTYPE}" \
      --device "${DEVICE}" \
      --device_map "${DEVICE_MAP}" \
      --attn_implementation eager \
      --output_jsonl "${out}" \
      --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_tau_${tau}_diagnostics.jsonl" \
      2>&1 | tee "${OUT_DIR}/ppl_tau_${tau}.log"
  done
fi

if [[ "${RUN_SELECTOR_PPL_COMPARISON}" -eq 1 ]]; then
  for selector in keepkv grouplogitkv; do
    out="${OUT_DIR}/ppl_selector_${selector}.jsonl"
    python -u "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
      --model_name "${MODEL_NAME}" \
      --task_eval_dir "${TASK_DIR}" \
      --family "${FAMILY}" \
      --datasets wikitext2 \
      --kv_method mymethod \
      --compress_ratio "${PPL_RATIO}" \
      --tau_mymethod "${PPL_TAU}" \
      --mymethod_selector "${selector}" \
      --mymethod_sink_size 4 \
      --mymethod_recent_window 128 \
      --mymethod_recency_beta 1.0 \
      --mymethod_distance_chunk_size 0 \
      --mymethod_attention_chunk_size 0 \
      --mymethod_sink_can_absorb 1 \
      --model_seqlen "${PPL_LENGTH}" \
      --chunk_size "${PPL_CHUNK}" \
      --compress_every 1 \
      --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
      --torch_dtype "${DTYPE}" \
      --device "${DEVICE}" \
      --device_map "${DEVICE_MAP}" \
      --attn_implementation eager \
      --output_jsonl "${out}" \
      --mymethod_diagnostics_jsonl "${OUT_DIR}/ppl_selector_${selector}_diagnostics.jsonl" \
      2>&1 | tee "${OUT_DIR}/ppl_selector_${selector}.log"
  done
fi

python -u "${TASK_DIR}/summarize_npc2026_diagnostics.py" \
  --results_dir "${OUT_DIR}" \
  --output_json "${OUT_DIR}/npc2026_diagnostic_summary.json" \
  --output_md "${OUT_DIR}/npc2026_diagnostic_summary.md"

cat <<EOF
Pre-experiments complete.
Results: ${OUT_DIR}
Package feedback with:
  bash ${TASK_DIR}/package_npc2026_feedback.sh ${OUT_DIR}
EOF
