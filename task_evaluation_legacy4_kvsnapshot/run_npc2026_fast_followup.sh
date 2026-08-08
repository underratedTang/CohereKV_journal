#!/usr/bin/env bash
set -euo pipefail

# Fast follow-up after the first NPC 2026 diagnostic run.
# Time-limited protocol: WikiText-2 only, first two evaluation sequences.

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="meta-llama/Llama-3.2-3B"
FAMILY="llama"
DEVICE="cuda"
DTYPE="bf16"
DEVICE_MAP="auto"
OUT_DIR="${TASK_DIR}/npc2026_fast_followup_results"
EXISTING_PRE_DIR="${TASK_DIR}/npc2026_pre_results"

PPL_LENGTH=8192
PPL_CHUNK=32
PPL_RATIO=0.4
PPL_MAX_SEQUENCES=2
LOW_TAUS=(0.1 0.2 0.3 0.4)
MYMETHOD_SELECTORS=(grouplogitkv keepkv)
BASELINE_METHODS=(none GroupLogitKV CentroidKV KeepKV SimCalKV CaM)

RUN_LOW_TAU_GROUPING=1
RUN_BASELINE_PPL=1
RUN_MYMETHOD_LOW_TAU_PPL=1

mkdir -p "${OUT_DIR}"

if [[ "${RUN_LOW_TAU_GROUPING}" -eq 1 ]]; then
  SNAPSHOT="${EXISTING_PRE_DIR}/kv_snapshot.pt"
  if [[ -f "${SNAPSHOT}" ]]; then
    python -u "${TASK_DIR}/analyze_npc2026_grouping.py" \
      --snapshot "${SNAPSHOT}" \
      --task_eval_dir "${TASK_DIR}" \
      --compress_ratios "0.4" \
      --taus "0.1,0.2,0.3,0.4,0.5,0.6" \
      --grouplogitkv_sim_thr 0.8 \
      --max_heads_per_layer 4 \
      --device "${DEVICE}" \
      --multiround_retained_ratios "0.8,0.6,0.4,0.3" \
      --output_json "${OUT_DIR}/grouping_low_tau.json" \
      2>&1 | tee "${OUT_DIR}/grouping_low_tau.log"
  else
    echo "[WARN] Snapshot not found: ${SNAPSHOT}; skipping low-tau grouping analysis."
  fi
fi

run_ppl() {
  local method="$1"
  local tau="${2:-na}"
  local selector="${3:-grouplogitkv}"
  local tag="${method}"
  local extra=()

  case "${method}" in
    GroupLogitKV)
      extra+=(--sim_thr_grouplogitkv 0.8)
      ;;
    KeepKV)
      extra+=(--sim_thr_keepkv 0.8)
      ;;
    SimCalKV)
      extra+=(--sim_thr_simcalkv 0.0)
      ;;
    CentroidKV)
      extra+=(
        --centroidkv_sink_budget 16
        --centroidkv_local_budget 64
        --centroidkv_chunk_size 256
        --centroidkv_merge_ratio 0.8
        --centroidkv_decrease_rate 0.2
      )
      ;;
    mymethod)
      tag="mymethod_${selector}_tau${tau}"
      extra+=(
        --tau_mymethod "${tau}"
        --mymethod_selector "${selector}"
        --mymethod_sink_size 4
        --mymethod_recent_window 128
        --mymethod_recency_beta 1.0
        --mymethod_distance_chunk_size 256
        --mymethod_attention_chunk_size 0
        --mymethod_sink_can_absorb 1
        --mymethod_diagnostics_jsonl "${OUT_DIR}/${tag}_diagnostics.jsonl"
      )
      ;;
  esac

  python -u "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
    --model_name "${MODEL_NAME}" \
    --task_eval_dir "${TASK_DIR}" \
    --family "${FAMILY}" \
    --datasets wikitext2 \
    --kv_method "${method}" \
    --compress_ratio "$([[ "${method}" == "none" ]] && echo 0 || echo "${PPL_RATIO}")" \
    "${extra[@]}" \
    --model_seqlen "${PPL_LENGTH}" \
    --chunk_size "${PPL_CHUNK}" \
    --compress_every 1 \
    --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
    --torch_dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --device_map "${DEVICE_MAP}" \
    --attn_implementation eager \
    --seed 0 \
    --output_jsonl "${OUT_DIR}/${tag}.jsonl" \
    2>&1 | tee "${OUT_DIR}/${tag}.log"
}

if [[ "${RUN_BASELINE_PPL}" -eq 1 ]]; then
  for method in "${BASELINE_METHODS[@]}"; do
    run_ppl "${method}"
  done
fi

if [[ "${RUN_MYMETHOD_LOW_TAU_PPL}" -eq 1 ]]; then
  for selector in "${MYMETHOD_SELECTORS[@]}"; do
    for tau in "${LOW_TAUS[@]}"; do
      run_ppl mymethod "${tau}" "${selector}"
    done
  done
fi

python -u "${TASK_DIR}/summarize_npc2026_diagnostics.py" \
  --results_dir "${OUT_DIR}" \
  --output_json "${OUT_DIR}/npc2026_fast_followup_summary.json" \
  --output_md "${OUT_DIR}/npc2026_fast_followup_summary.md"

cat <<MSG
Fast follow-up complete.
Protocol: WikiText-2 only; max_eval_sequences=2.
Results: ${OUT_DIR}
Package with:
  bash ${TASK_DIR}/package_npc2026_feedback.sh ${OUT_DIR}
MSG
