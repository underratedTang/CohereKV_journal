#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Editable configuration
# ============================================================
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
LONG_MODEL_KEY="llama-3.1-8B-Instruct"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
FAMILY="llama"
OUT_DIR="${TASK_DIR}/npc2026_main_results"
DEVICE="cuda"
DTYPE="bf16"
DEVICE_MAP="auto"

METHODS=(None CentroidKV KeepKV SimCalKV CaM mymethod)
# GroupLogitKV is the direct parent/diagnostic baseline. Keep it enabled unless page limits force appendix placement.
INCLUDE_GROUPLOGITKV=1
RATIOS=(0.2 0.4 0.6)
DEFAULT_SEEDS=(0)
CAM_SEEDS=(0 1 2)
PPL_DATASETS=(wikitext2)
PPL_LENGTH=8192
PPL_CHUNK=32
PPL_MAX_SEQUENCES=2
MYMETHOD_TAU=0.1  # provisional; replace with the best fast-followup validation result
MYMETHOD_SELECTOR="keepkv"
LONGBENCH_DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"

RUN_PPL=1
RUN_LONGBENCH=1
RUN_SYSTEM=1
# ============================================================

mkdir -p "${OUT_DIR}/ppl" "${OUT_DIR}/longbench" "${OUT_DIR}/system"
if [[ "${INCLUDE_GROUPLOGITKV}" -eq 1 ]]; then
  METHODS+=(GroupLogitKV)
fi

method_args() {
  local method="$1"
  METHOD_ARGS=()
  case "${method}" in
    mymethod)
      METHOD_ARGS+=(
        --tau_mymethod "${MYMETHOD_TAU}"
        --mymethod_selector "${MYMETHOD_SELECTOR}"
        --mymethod_sink_size 4
        --mymethod_recent_window 128
        --mymethod_recency_beta 1.0
        --mymethod_distance_chunk_size 256
        --mymethod_attention_chunk_size 0
        --mymethod_sink_can_absorb 1
      )
      ;;
    GroupLogitKV) METHOD_ARGS+=(--sim_thr_grouplogitkv 0.8) ;;
    SimCalKV) METHOD_ARGS+=(--sim_thr_simcalkv 0.0) ;;
    KeepKV) METHOD_ARGS+=(--sim_thr_keepkv 0.8) ;;
    CentroidKV)
      METHOD_ARGS+=(
        --centroidkv_sink_budget 16
        --centroidkv_local_budget 64
        --centroidkv_chunk_size 256
        --centroidkv_merge_ratio 0.8
        --centroidkv_decrease_rate 0.2
      )
      ;;
  esac
}

if [[ "${RUN_PPL}" -eq 1 ]]; then
  for method in "${METHODS[@]}"; do
    ratios=("${RATIOS[@]}")
    [[ "${method}" == "None" ]] && ratios=(0)
    for ratio in "${ratios[@]}"; do
      seeds=("${DEFAULT_SEEDS[@]}")
      [[ "${method}" == "CaM" ]] && seeds=("${CAM_SEEDS[@]}")
      for seed in "${seeds[@]}"; do
      method_args "${method}"
      out="${OUT_DIR}/ppl/${FAMILY}_${method}_ratio${ratio}_seed${seed}.jsonl"
      python -u "${TASK_DIR}/evaluate_ppl_kvmerge.py" \
        --model_name "${MODEL_NAME}" \
        --task_eval_dir "${TASK_DIR}" \
        --family "${FAMILY}" \
        --datasets "${PPL_DATASETS[@]}" \
        --kv_method "${method/None/none}" \
        --compress_ratio "${ratio}" \
        "${METHOD_ARGS[@]}" \
        --model_seqlen "${PPL_LENGTH}" \
        --chunk_size "${PPL_CHUNK}" \
        --compress_every 1 \
        --max_eval_sequences "${PPL_MAX_SEQUENCES}" \
        --torch_dtype "${DTYPE}" \
        --device "${DEVICE}" \
        --device_map "${DEVICE_MAP}" \
        --attn_implementation eager \
        --seed "${seed}" \
        --output_jsonl "${out}" \
        2>&1 | tee "${out%.jsonl}.log"
      done
    done
  done
fi

if [[ "${RUN_LONGBENCH}" -eq 1 ]]; then
  for method in "${METHODS[@]}"; do
    ratios=("${RATIOS[@]}")
    [[ "${method}" == "None" ]] && ratios=(0)
    for ratio in "${ratios[@]}"; do
      seeds=("${DEFAULT_SEEDS[@]}")
      [[ "${method}" == "CaM" ]] && seeds=("${CAM_SEEDS[@]}")
      for seed in "${seeds[@]}"; do
      method_args "${method}"
      pred_dir="${OUT_DIR}/longbench/${FAMILY}_${method}_ratio${ratio}_seed${seed}"
      python -u "${TASK_DIR}/evaluate_longbench_kvmerge.py" \
        --task_eval_dir "${TASK_DIR}" \
        --arch "${FAMILY}" \
        --model "${LONG_MODEL_KEY}" \
        --model_path "${MODEL_PATH}" \
        --kv_method "${method}" \
        --compress_ratio "${ratio}" \
        "${METHOD_ARGS[@]}" \
        --datasets "${LONGBENCH_DATASETS}" \
        --out_path "${pred_dir}" \
        --dtype "${DTYPE}" \
        --device "${DEVICE}" \
        --attn_implementation eager \
        --seed "${seed}" \
        --overwrite \
        2>&1 | tee "${pred_dir}.log"
      done
    done
  done
fi

if [[ "${RUN_SYSTEM}" -eq 1 ]]; then
  # The paper should use compression_time, persistent KV+side-info memory and
  # compression peak extra memory as primary systems metrics. E2E fields are supplementary.
  for method in "${METHODS[@]}"; do
    ratios=("${RATIOS[@]}")
    [[ "${method}" == "None" ]] && ratios=(0)
    for ratio in "${ratios[@]}"; do
      seeds=("${DEFAULT_SEEDS[@]}")
      [[ "${method}" == "CaM" ]] && seeds=("${CAM_SEEDS[@]}")
      for seed in "${seeds[@]}"; do
      method_args "${method}"
      out="${OUT_DIR}/system/${FAMILY}_${method}_ratio${ratio}_seed${seed}.json"
      python -u "${TASK_DIR}/evaluate_system_kvmerge.py" \
        --model_name "${LONG_MODEL_KEY}" \
        --model_id "${MODEL_PATH}" \
        --task_eval_dir "${TASK_DIR}" \
        --arch "${FAMILY}" \
        --kv_method "${method}" \
        --compress_ratio "${ratio}" \
        "${METHOD_ARGS[@]}" \
        --dataset gov_report \
        --input_length 8192 \
        --generation_length 64 \
        --batch_size 1 \
        --warmup_runs 3 \
        --measured_runs 10 \
        --dtype "${DTYPE}" \
        --device "${DEVICE}" \
        --attn_implementation eager \
        --seed "${seed}" \
        --output_json "${out}" \
        2>&1 | tee "${out%.json}.log"
      done
    done
  done
fi

cat <<EOF
Main experiments complete.
Results: ${OUT_DIR}
Package feedback with:
  bash ${TASK_DIR}/package_npc2026_feedback.sh ${OUT_DIR}
EOF
