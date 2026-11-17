#!/usr/bin/env bash
set -euo pipefail

# Edit parameters directly in this file (no environment-variable overrides).
ROOT_DIR="/home/tlxr/CohereKV"
cd "${ROOT_DIR}"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

DIAG_DIR="${ROOT_DIR}/diagnosis_experiments"
CALIBRATE="${DIAG_DIR}/calibrate_median_reference.py"
DIAGNOSE="${DIAG_DIR}/diagnose_cosine_mis.py"
PLOT="${DIAG_DIR}/plot_cosine_mis.py"
PACKAGE="${DIAG_DIR}/package_cosine_mis.py"
RESULTS_ROOT="${DIAG_DIR}/results"

# ============================================================
# 1. Execution / model / evaluation workload
# ============================================================
MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.2-3B"
# MODEL="Qwen/Qwen2.5-7B"
# MODEL="meta-llama/Llama-2-7b-hf"


DATASET="wikitext2"
TORCH_DTYPE="bfloat16"
DEVICE="cuda"
DEVICE_MAP="auto"
ATTN_IMPL="eager"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0
PREFILL_CHUNK_SIZE=128
QUICK_MODE=1
RUN_CALIBRATION=0
RUN_ANALYSIS=0
RUN_PLOT=1
RUN_PPL=0

# ============================================================
# 1a. Independent method switches; 1=run, 0=skip.
# Original eight methods are retained unchanged. Optional IW-MergeAll plus
# new graph/fusion ablation and full IA-MIS arms have separate switches.
# Controls Full Cache and Count Merge All are always enabled.
# No externally supplied environment variables control experiment settings.
# ============================================================
RUN_IMPORTANCE_WEIGHTED_MERGE_ALL=1
# Original eight MIS methods
RUN_K_ONLY_MIS_DROP=0
RUN_K_ONLY_MIS_DROP_ALLCOUNT=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT=0
RUN_K_ONLY_MIS_DROP_SINK=0
RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK=0
# Original D-first graph + new IW mean (fusion-only ablation)
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP=1
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_ALLCOUNT=0
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_SINK=1
RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_ALLCOUNT_SINK=0
# New IA graph + old count mean (graph-only ablation)
RUN_IA_MIS_COUNTMEAN_DROP=0
RUN_IA_MIS_COUNTMEAN_DROP_ALLCOUNT=0
RUN_IA_MIS_COUNTMEAN_DROP_SINK=0
RUN_IA_MIS_COUNTMEAN_DROP_ALLCOUNT_SINK=0
# Full new IA graph + Count*Importance mean
RUN_IA_MIS_IWMEAN_DROP=0
RUN_IA_MIS_IWMEAN_DROP_ALLCOUNT=0
RUN_IA_MIS_IWMEAN_DROP_SINK=0
RUN_IA_MIS_IWMEAN_DROP_ALLCOUNT_SINK=0

METHOD_SWITCHES=(
  "importance_weighted_merge_all:${RUN_IMPORTANCE_WEIGHTED_MERGE_ALL}"
  "k_only_mis_drop:${RUN_K_ONLY_MIS_DROP}"
  "k_only_mis_drop_allcount:${RUN_K_ONLY_MIS_DROP_ALLCOUNT}"
  "k_only_importance_mis_drop:${RUN_K_ONLY_IMPORTANCE_MIS_DROP}"
  "k_only_importance_mis_drop_allcount:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT}"
  "k_only_mis_drop_sink:${RUN_K_ONLY_MIS_DROP_SINK}"
  "k_only_mis_drop_allcount_sink:${RUN_K_ONLY_MIS_DROP_ALLCOUNT_SINK}"
  "k_only_importance_mis_drop_sink:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_SINK}"
  "k_only_importance_mis_drop_allcount_sink:${RUN_K_ONLY_IMPORTANCE_MIS_DROP_ALLCOUNT_SINK}"
  "k_only_importance_iwmean_drop:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP}"
  "k_only_importance_iwmean_drop_allcount:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_ALLCOUNT}"
  "k_only_importance_iwmean_drop_sink:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_SINK}"
  "k_only_importance_iwmean_drop_allcount_sink:${RUN_K_ONLY_IMPORTANCE_IWMEAN_DROP_ALLCOUNT_SINK}"
  "ia_mis_countmean_drop:${RUN_IA_MIS_COUNTMEAN_DROP}"
  "ia_mis_countmean_drop_allcount:${RUN_IA_MIS_COUNTMEAN_DROP_ALLCOUNT}"
  "ia_mis_countmean_drop_sink:${RUN_IA_MIS_COUNTMEAN_DROP_SINK}"
  "ia_mis_countmean_drop_allcount_sink:${RUN_IA_MIS_COUNTMEAN_DROP_ALLCOUNT_SINK}"
  "ia_mis_iwmean_drop:${RUN_IA_MIS_IWMEAN_DROP}"
  "ia_mis_iwmean_drop_allcount:${RUN_IA_MIS_IWMEAN_DROP_ALLCOUNT}"
  "ia_mis_iwmean_drop_sink:${RUN_IA_MIS_IWMEAN_DROP_SINK}"
  "ia_mis_iwmean_drop_allcount_sink:${RUN_IA_MIS_IWMEAN_DROP_ALLCOUNT_SINK}"
)
MIS_METHODS_ENABLED=()
METHOD_INDEX_TAG=""
for i in "${!METHOD_SWITCHES[@]}"; do
  name="${METHOD_SWITCHES[$i]%:*}"
  enabled="${METHOD_SWITCHES[$i]##*:}"
  if [[ "${enabled}" != "0" && "${enabled}" != "1" ]]; then
    echo "[ERROR] ${name} switch must be exactly 0 or 1" >&2; exit 1
  fi
  if [[ "${enabled}" == "1" ]]; then
    MIS_METHODS_ENABLED+=("${name}")
    METHOD_INDEX_TAG+="_$((i+1))"
  fi
done
if [[ "${#MIS_METHODS_ENABLED[@]}" -eq 0 ]]; then
  echo '[ERROR] Enable at least one MIS method' >&2; exit 1
fi
MIS_FAMILIES_CSV="k_only,k_only_importance,k_only_importance_iwmean,ia_mis_countmean,ia_mis_iwmean"

if [[ "${QUICK_MODE}" -eq 1 ]]; then
  EXPERIMENT_NAME="journal_ia_mis${METHOD_INDEX_TAG}_quick"
  PREFIX_LEN=4096
  FUTURE_PROBE_LEN=64
  MAX_EVAL_SEQUENCES=2
  LAYER_SELECTION="all"
  KV_HEAD_SELECTION="all"
else
  EXPERIMENT_NAME="journal_ia_mis${METHOD_INDEX_TAG}_full"
  PREFIX_LEN=8192
  FUTURE_PROBE_LEN=256
  MAX_EVAL_SEQUENCES=3
  LAYER_SELECTION="all"
  KV_HEAD_SELECTION="all"
fi

# Attention replay: ONE compression round, original C=1,R=D=0.
# Streaming PPL: repeated physical-capacity compression, current-slot R=D=0
# each round, while token Counts remain propagated across rounds.
KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=128
KEEPKV_SCORE_WINDOW=128
KEEPKV_RECENCY_BETA=1.0
KEEP_RATIOS=("0.1" "0.2" "0.4")

# ============================================================
# 2. R-E Routing sweeps (shared by ALL methods)
# ============================================================
# Select one or both. Each routing mode creates its own candidate groups.
ROUTING_MODES=("cosine")
# Cosine: nearest by cosine, eligible iff cosine >= threshold (NO L2 gate).
ROUTING_COSINE_THRESHOLDS=("0.80")
# L2: nearest by physical Euclidean distance, eligible iff L2 <= cutoff.
# Calibrated cutoff = lambda_route * frozen S_calib[layer,KVhead].
ROUTING_L2_LAMBDAS=("0.50" "0.55" "0.60" "0.65" "0.70")
# Optional absolute Euclidean distances (same units as raw post-RoPE Keys).
# Leave empty to sweep only calibrated L2 routing lambdas.
ROUTING_L2_DISTANCES=()
# Routing sweep is entirely independent of E-E graph MIS threshold below.

# ============================================================
# 3. OFFLINE Median K and V reference calibration (train)
# ============================================================
CALIBRATION_SPLIT="train"
CALIBRATION_NUM_SEQUENCES=16
CALIBRATION_SEQUENCE_LENGTH=8192
if [[ "${QUICK_MODE}" -eq 1 ]]; then
  CALIBRATION_NUM_SEQUENCES=4
  CALIBRATION_SEQUENCE_LENGTH=4096
fi
# StreamingLLM Appendix-F diagnostic: final query's attention to FIRST token.
# Train-frozen Sink KV-head mask gates only *_sink methods; originals always use native MIS.
SINK_LENGTH=1
SINK_DIAGNOSTIC_SEQLEN=4096
SINK_PROBE_LEN=64
SINK_PRESENCE_THRESHOLD=0.05
CALIBRATION_SEED=42
# Empirical TRAIN-only post-RoPE Query norm envelope: 1.05 x observed max.
QUERY_NORM_SAFETY_FACTOR=1.05

# ============================================================
# 3b. Real compressed-cache GPTQ-style PPL (NOT layer-wise replay)
# ============================================================
PPL_DATASET="wikitext2"
PPL_SEQLEN=8192
PPL_STEP_SIZE=32
PPL_MAX_SEQUENCES=2
# Fixed physical KV budgets: B = round(PPL_SEQLEN * ratio) tokens per KV head.
# Compression starts only AFTER cached slots exceed B; the value of B is
# unchanged across all chunks and rounds. For QUICK/2048: B=205,410,819.
# Post-chunk compression can transiently use B + PPL_STEP_SIZE slots.
PPL_BUDGETS=("0.1" "0.2" "0.4")
PPL_ROUTING_MODES=("cosine")
# PPL sweep is INDEPENDENT of attention diagnosis routing / E-E sweep.
# The route parameter only applies to its own mode, while each E-E value is
# evaluated for MIS-dependent branches. Edit these literal SH arrays directly.
PPL_COSINE_THRESHOLDS=("0.80")
PPL_L2_LAMBDAS=("0.70")
# IDENTICAL frozen-TRAIN Key distance thresholds for old and new MIS priority.
# This is a strict one-factor comparison; no added priority hyperparameter.
PPL_EE_LAMBDAS=("0.10" "0.30" "0.50" "0.70")
# PPL_EE_LAMBDAS=("0.10")

# 1 resumes an interrupted PPL grid ONLY if model / config / calibration match.
# Set 0 to replace the previous sweep results in this run directory.
PPL_RESUME=1
if [[ "${QUICK_MODE}" -eq 0 ]]; then
  # Full mode uses all WikiText-2 test windows; edit directly if too costly.
  PPL_SEQLEN=4096
  PPL_MAX_SEQUENCES=0
fi
# Per sequence: S[m,l,h] = RMS of pairwise physical Key L2 distances.
# Frozen: S_calib[l,h] = median_m S[m,l,h]. Test sequences never set S_calib.

# ============================================================
# 4. Old E-E graph, optional IA E-A+E-E graph, unchanged D-first MIS
# ============================================================
# Original: E-E edge iff ||k_i-k_j|| > lambda * S_ref[layer,head].
# IA: E-A gate and E-E edges use sqrt(2*rho_i*rho_j/(rho_i+rho_j))*L2;
# rho=W_evicted/W_anchor, W=Count*current-slot-importance. Same S_ref.
# IW: K/V Mean weights Count*current-slot-importance; Count updates separate.
# Old priority : Degree asc, Anchor L2 asc, stable hash asc.
# New priority : D_i = I_i - sum Importance of full-graph E-E neighbors, DESC; ties old priority.
# Drop discards Reject K/V/Count; AllCount keeps Reject Count only.
# *_sink methods replace ONLY non-Sink KV-head fusion with exact Merge All.
# Retained anchor excluded from the graph; neither scheme certifies Drop.
MIS_EE_LAMBDAS=("0.10" "0.30" "0.50" "0.70")
DISTANCE_CHUNK_SIZE=256
# Keep the original bounded GPU batching; do not change graph or fusion.
# Importance is the SAME per-head KeepKV score used for retained selection.
GRAPH_BACKEND="gpu_batched"

# if [[ "${QUICK_MODE}" -eq 1 ]]; then
#   ROUTING_COSINE_THRESHOLDS=("0.80")
#   ROUTING_L2_LAMBDAS=("0.70")
#   # Quick Replay: lambda=0.50, PPL Quick: all five lambdas above.
#   MIS_EE_LAMBDAS=("0.10")
# fi

# ============================================================
# 5. Attention diagnosis and publication-quality figures
# ============================================================
SAVE_PER_PROBE=1
OUTPUT_NORM_EPS=1e-8
# One common focus value selects both summary charts and saved geometry.
FOCUS_EE_LAMBDA=0.70
# Detailed method/Layer/real-Q figures use one focus threshold per mode.
# Full routing and routing x EE sweeps are plotted for ALL thresholds.
FOCUS_ROUTING_COSINE_THRESHOLD=0.80
FOCUS_ROUTING_L2_LAMBDA=0.70
CAPTURE_LAYER=0
CAPTURE_HEAD=0
CAPTURE_BUDGET=0.1
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=1
PNG_DPI=600

# Qwen2.5 uses fewer KV heads than Llama; choose valid indices automatically.
# Qwen2.5-3B: 2 KV heads; Qwen2.5-7B: 4 KV heads.
# if [[ "${MODEL}" == "Qwen/Qwen2.5-3B" ]]; then
#   KV_HEAD_SELECTION="0,1"
#   CAPTURE_HEAD=1
# elif [[ "${MODEL}" == "Qwen/Qwen2.5-7B" ]]; then
#   if [[ "${QUICK_MODE}" -eq 1 ]]; then
#     KV_HEAD_SELECTION="0,2"
#   else
#     KV_HEAD_SELECTION="0,1,2,3"
#   fi
#   CAPTURE_HEAD=2
# fi

join_csv() { local IFS=,; echo "$*"; }
KEEP_RATIOS_CSV="$(join_csv "${KEEP_RATIOS[@]}")"
ROUTING_MODES_CSV="$(join_csv "${ROUTING_MODES[@]}")"
ROUTING_COSINE_CSV="$(join_csv "${ROUTING_COSINE_THRESHOLDS[@]}")"
ROUTING_L2_LAMBDAS_CSV="$(join_csv "${ROUTING_L2_LAMBDAS[@]}")"
ROUTING_L2_DISTANCES_CSV="$(join_csv "${ROUTING_L2_DISTANCES[@]}")"
MIS_EE_LAMBDAS_CSV="$(join_csv "${MIS_EE_LAMBDAS[@]}")"
MIS_METHODS_CSV="$(join_csv "${MIS_METHODS_ENABLED[@]}")"
MODEL_BASENAME="${MODEL%/}"; MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
CALIBRATION_FILE="${OUT_DIR}/calibration/median_reference.json"
PPL_EVAL="${DIAG_DIR}/evaluate_ppl_konly_mis.py"
PPL_PLOT="${DIAG_DIR}/plot_ppl_konly_mis.py"
PPL_BUDGETS_CSV="$(join_csv "${PPL_BUDGETS[@]}")"
PPL_ROUTING_MODES_CSV="$(join_csv "${PPL_ROUTING_MODES[@]}")"
PPL_COSINE_CSV="$(join_csv "${PPL_COSINE_THRESHOLDS[@]}")"
PPL_L2_CSV="$(join_csv "${PPL_L2_LAMBDAS[@]}")"
PPL_EE_CSV="$(join_csv "${PPL_EE_LAMBDAS[@]}")"

# A new calibration starts a completely fresh experiment. With a pre-existing
# calibration, a rerun cleans ONLY numeric/figure outputs, preserving reference.
if [[ "${RUN_CALIBRATION}" -eq 1 && -d "${OUT_DIR}" ]]; then
  echo "[CLEAN] Removing previous output from THIS run: ${OUT_DIR}"
  rm -rf -- "${OUT_DIR}"
elif [[ "${RUN_ANALYSIS}" -eq 1 && -d "${OUT_DIR}" ]]; then
  find "${OUT_DIR}" -mindepth 1 -maxdepth 1 ! -name calibration ! -name ppl_sweep -exec rm -rf -- {} +
fi
mkdir -p "${OUT_DIR}/calibration"
if [[ "${RUN_ANALYSIS}" -eq 1 && "${RUN_CALIBRATION}" -eq 0 && ! -s "${CALIBRATION_FILE}" ]]; then
  echo "[ERROR] RUN_CALIBRATION=0 but Median reference file does not exist: ${CALIBRATION_FILE}" >&2
  exit 1
fi
if [[ "${CALIBRATION_SPLIT}" != "train" ]]; then
  echo "[ERROR] Current experiment requires CALIBRATION_SPLIT=train" >&2
  exit 1
fi
FOCUS_FOUND=0
for EE_VAL in "${MIS_EE_LAMBDAS[@]}"; do
  if python -c 'import math,sys;sys.exit(0 if math.isclose(float(sys.argv[1]),float(sys.argv[2]),rel_tol=0,abs_tol=1e-9) else 1)' "${EE_VAL}" "${FOCUS_EE_LAMBDA}"; then
    FOCUS_FOUND=1
  fi
done
if [[ "${FOCUS_FOUND}" -ne 1 ]]; then
  echo "[ERROR] FOCUS_EE_LAMBDA must be included in MIS_EE_LAMBDAS" >&2
  exit 1
fi
RUN_LOG="${OUT_DIR}/run.log"
{
  echo "EXPERIMENT=${EXPERIMENT_NAME} MODEL=${MODEL} QUICK_MODE=${QUICK_MODE}"
  echo "CALIBRATION=train/${CALIBRATION_NUM_SEQUENCES}x${CALIBRATION_SEQUENCE_LENGTH} MEDIAN seed=${CALIBRATION_SEED} QNormSafety=${QUERY_NORM_SAFETY_FACTOR}"
  echo "EVAL=test/${MAX_EVAL_SEQUENCES}x(${PREFIX_LEN}+${FUTURE_PROBE_LEN})"
  echo "ROUTING_MODES=${ROUTING_MODES_CSV} COSINE_THRESHOLDS=${ROUTING_COSINE_CSV} L2_LAMBDAS=${ROUTING_L2_LAMBDAS_CSV} L2_DISTANCES=${ROUTING_L2_DISTANCES_CSV}"
  echo "MIS_FAMILIES=${MIS_FAMILIES_CSV} OLD_AND_IA_EE_LAMBDAS=${MIS_EE_LAMBDAS_CSV}"
  echo "BUDGETS=${KEEP_RATIOS_CSV} LAYERS=${LAYER_SELECTION} HEADS=${KV_HEAD_SELECTION}"
  echo "METHODS=full_cache,merge_all,${MIS_METHODS_CSV}; Sink-gated: frozen TRAIN mask, Sink->MIS NonSink->MergeAll; Replay: one round C=1 R=D=0; PPL: repeated rounds"
  echo "SINK_CALIBRATION=train last query to FIRST token; frozen KV-head gating threshold=${SINK_PRESENCE_THRESHOLD}"
  echo "PPL=${RUN_PPL} GPTQ-style seqlen=${PPL_SEQLEN} step=${PPL_STEP_SIZE} windows=${PPL_MAX_SEQUENCES} modes=${PPL_ROUTING_MODES_CSV}"
  echo "PPL_FIXED_CAPACITY_RATIO=${PPL_BUDGETS_CSV}; compress ONLY when physical KV slots exceed round(ratio * ${PPL_SEQLEN})"
  echo "PPL_SWEEP cosine_thresholds=${PPL_COSINE_CSV} l2_lambdas=${PPL_L2_CSV} shared_K_lambda=${PPL_EE_CSV} resume=${PPL_RESUME}"
  echo "FOCUS=${FOCUS_EE_LAMBDA} CAPTURE=${CAPTURE_LAYER}/${CAPTURE_HEAD}/${CAPTURE_BUDGET}"
  echo "RESULTS=${OUT_DIR}"
} | tee -a "${RUN_LOG}"

# RUN_CALIBRATION=1 regenerates Median+Sink+Q norm calibration from train.
# RUN_CALIBRATION=0 requires the current run directory to contain both.
if [[ "${RUN_CALIBRATION}" -eq 1 ]]; then
  CMD=(python -u "${CALIBRATE}"
    --model "${MODEL}" --dataset "${DATASET}"
    --calibration_split "${CALIBRATION_SPLIT}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --query_norm_safety_factor "${QUERY_NORM_SAFETY_FACTOR}"
    --sink_length "${SINK_LENGTH}"
    --sink_diagnostic_seqlen "${SINK_DIAGNOSTIC_SEQLEN}"
    --sink_probe_len "${SINK_PROBE_LEN}"
    --sink_presence_threshold "${SINK_PRESENCE_THRESHOLD}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --output "${CALIBRATION_FILE}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[CALIBRATE]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  if [[ ! -s "${CALIBRATION_FILE}" ]]; then
    echo "[ERROR] Missing current model's frozen Median reference: ${CALIBRATION_FILE}" >&2
    exit 1
  fi
  CMD=(python -u "${DIAGNOSE}"
    --model "${MODEL}" --dataset "${DATASET}"
    --prefix_len "${PREFIX_LEN}" --future_probe_len "${FUTURE_PROBE_LEN}"
    --max_sequences "${MAX_EVAL_SEQUENCES}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --keep_ratios "${KEEP_RATIOS_CSV}"
    --routing_modes "${ROUTING_MODES_CSV}"
    --cosine_thresholds "${ROUTING_COSINE_CSV}"
    --routing_l2_lambdas "${ROUTING_L2_LAMBDAS_CSV}"
    --routing_l2_distances "${ROUTING_L2_DISTANCES_CSV}"
    --ee_lambdas "${MIS_EE_LAMBDAS_CSV}"
    --mis_families "${MIS_FAMILIES_CSV}"
    --mis_methods "${MIS_METHODS_CSV}"
    --median_reference_file "${CALIBRATION_FILE}"
    --calibration_num_sequences "${CALIBRATION_NUM_SEQUENCES}"
    --calibration_seqlen "${CALIBRATION_SEQUENCE_LENGTH}"
    --calibration_seed "${CALIBRATION_SEED}"
    --sink_length "${SINK_LENGTH}"
    --sink_presence_threshold "${SINK_PRESENCE_THRESHOLD}"
    --keepkv_sink_size "${KEEPKV_SINK_SIZE}"
    --keepkv_recent_protect_size "${KEEPKV_RECENT_PROTECT_SIZE}"
    --keepkv_score_window "${KEEPKV_SCORE_WINDOW}"
    --keepkv_recency_beta "${KEEPKV_RECENCY_BETA}"
    --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
    --graph_backend "${GRAPH_BACKEND}"
    --save_per_probe "${SAVE_PER_PROBE}" --output_norm_eps "${OUTPUT_NORM_EPS}"
    --layer_selection "${LAYER_SELECTION}" --kv_head_selection "${KV_HEAD_SELECTION}"
    --capture_layer "${CAPTURE_LAYER}" --capture_head "${CAPTURE_HEAD}"
    --capture_budget "${CAPTURE_BUDGET}" --capture_ee_lambda "${FOCUS_EE_LAMBDA}"
    --output_dir "${OUT_DIR}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then CMD+=(--use_fast_tokenizer); fi
  printf '[ANALYZE]'; printf ' %q' "${CMD[@]}"; echo
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

# Preserve complete diagnosis results even if the expensive PPL stage fails.
python -u "${PACKAGE}" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"

if [[ "${RUN_PPL}" -eq 1 ]]; then
  PPL_CMD=(python -u "${PPL_EVAL}"
    --model "${MODEL}" --calibration "${CALIBRATION_FILE}"
    --output_dir "${OUT_DIR}/ppl_sweep" --dataset "${PPL_DATASET}"
    --seqlen "${PPL_SEQLEN}" --step_size "${PPL_STEP_SIZE}"
    --max_sequences "${PPL_MAX_SEQUENCES}" --budgets "${PPL_BUDGETS_CSV}"
    --routing_modes "${PPL_ROUTING_MODES_CSV}"
    --cosine_thresholds "${PPL_COSINE_CSV}"
    --l2_lambdas "${PPL_L2_CSV}" --ee_lambdas "${PPL_EE_CSV}"
    --mis_families "${MIS_FAMILIES_CSV}"
    --mis_methods "${MIS_METHODS_CSV}"
    --resume "${PPL_RESUME}"
    --score_window "${KEEPKV_SCORE_WINDOW}"
    --distance_chunk_size "${DISTANCE_CHUNK_SIZE}"
    --graph_backend "${GRAPH_BACKEND}"
    --torch_dtype "${TORCH_DTYPE}" --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}" --device_map "${DEVICE_MAP}")
  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then PPL_CMD+=(--trust_remote_code); fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then PPL_CMD+=(--use_fast_tokenizer); fi
  printf '[PPL]'; printf ' %q' "${PPL_CMD[@]}"; echo
  "${PPL_CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

# Package numeric data BEFORE plotting in case Times New Roman is unavailable.
python -u "${PACKAGE}" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
if [[ "${RUN_PLOT}" -eq 1 ]]; then
  if [[ -s "${OUT_DIR}/ppl_sweep/ppl_konly_importance_mis.csv" ]]; then
    python -u "${PPL_PLOT}" --input_dir "${OUT_DIR}/ppl_sweep" \
      --font_family "${FONT_FAMILY}" \
      --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}" \
      --png_dpi "${PNG_DPI}" \
      --focus_ee_lambda "${FOCUS_EE_LAMBDA}" 2>&1 | tee -a "${RUN_LOG}"
  fi
  if [[ -s "${OUT_DIR}/cosine_mis_by_config.csv" ]]; then
    python -u "${PLOT}" --input_dir "${OUT_DIR}" \
      --font_family "${FONT_FAMILY}" \
      --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}" \
      --png_dpi "${PNG_DPI}" \
      --primary_ee_lambda "${FOCUS_EE_LAMBDA}" \
      --focus_cosine_threshold "${FOCUS_ROUTING_COSINE_THRESHOLD}" \
      --focus_l2_lambda "${FOCUS_ROUTING_L2_LAMBDA}" 2>&1 | tee -a "${RUN_LOG}"
  else
    echo "[PLOT SKIP] No Attention diagnosis CSV; PPL-only mode remains supported" | tee -a "${RUN_LOG}"
  fi
fi
python -u "${PACKAGE}" --input_dir "${OUT_DIR}" 2>&1 | tee -a "${RUN_LOG}"
echo "[UPLOAD] ${OUT_DIR}/journal_ia_mis_results.zip" | tee -a "${RUN_LOG}"
