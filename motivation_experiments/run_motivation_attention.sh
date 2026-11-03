#!/usr/bin/env bash
set -euo pipefail

# Motivation ONLY: one-round Star Conflict prevalence + oracle K-conflict filtering
# + filtering vs routing-only threshold sweep. No Journal p sweep in this folder.
# Edit assignments here; no externally injected environment-variable settings.
PROJECT_ROOT="/home/tlxr/CohereKV"
MOTIVATION_DIR="$PROJECT_ROOT/motivation_experiments"
RESULTS_ROOT="$MOTIVATION_DIR/results"

# Select exactly ONE complete model block by commenting/uncommenting its 3 lines.
# --- Llama-3.2-1B ---
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL_DIR_NAME="Llama-3.2-1B_journal_motivation"
# MOT_LAYER_SELECTION="0,4,8,12,15"

# --- Llama-3.2-3B ---
# MODEL="meta-llama/Llama-3.2-3B"
# MODEL_DIR_NAME="Llama-3.2-3B_journal_motivation"
# MOT_LAYER_SELECTION="0,8,16,24"

# --- Llama-3.1-8B ---
MODEL="meta-llama/Meta-Llama-3.1-8B"
MODEL_DIR_NAME="Meta-Llama-3.1-8B_journal_motivation"
MOT_LAYER_SELECTION="0,8,16,24,31"

DATASET="wikitext2"
TORCH_DTYPE="bfloat16"
ATTN_IMPLEMENTATION="eager"
DEVICE="cuda:1"
DEVICE_MAP="none"
TRUST_REMOTE_CODE="0"
USE_FAST_TOKENIZER="0"
PROGRESS_EVERY_HEAD="1"

KEEPKV_SINK_SIZE="4"
KEEPKV_RECENT_PROTECT_SIZE="32"
KEEPKV_SCORE_WINDOW="32"
KEEPKV_RECENCY_BETA="1.0"

# Equation (7) on each uncompressed calibration sequence; Eq. (24) MEDIAN
# across disjoint WikiText-2 TRAIN windows; freeze S_(layer,KV-head).
CALIBRATION_SPLIT="train"
CALIBRATION_NUM_SEQUENCES="16"
CALIBRATION_SEQLEN="8192"
CALIBRATION_SEED="42"
CALIBRATION_PREFILL_CHUNK_SIZE="256"
RUN_CALIBRATION="1"

# Both Euclidean routing and Euclidean conflict detection share Delta=lambda*S_calib.
# Cosine motivation-only control routes with a dimensionless tau_cos;
# its conflict detection still uses the same Euclidean Delta.
LAMBDA="0.50"
FIG1_COSINE_TAU="0.80"
FIG1_RUN_HPARAM_SWEEP="0"
FIG1_LAMBDA_GRID="0.30,0.40,0.50,0.60,0.70"
FIG1_COSINE_TAU_GRID="0.60,0.70,0.80,0.90"
FIG1_MAX_GROUPS_PER_HEAD="0"
FIG1_CR_NUM_REMOVALS="1"
FIG1_CR_NUM_REMOVALS_GRID="1,4,8"

# Figure 2: Euclidean x-axis sweeps the SAME lambda for routing and conflict.
# Cosine x-axis sweeps tau_cos with conflict Delta=lambda*S_calib fixed.
FIG2_COSINE_TAU="0.80"
FIG2_EUCLIDEAN_LAMBDA_GRID="0.30,0.40,0.50,0.60,0.70"
FIG2_COSINE_TAU_GRID="0.60,0.70,0.80,0.90"
FIG2_CR_NUM_REMOVALS="1"
FIG2_RUN_CR_GRID="1"
FIG2_CR_NUM_REMOVALS_GRID="1,4,8"

MOT_MODEL_SEQLEN="4096"
MOT_MAX_EVAL_PREFIXES="8"
MOT_PREFIX_STRIDE="0"
MOT_PREFILL_CHUNK_SIZE="256"
MOT_KEEP_RATIO="0.10"
MOT_ROUTING_SCORE_CHUNK_SIZE="1024"
MOT_KV_HEAD_SELECTION="all"

PNG_DPI="300"
STRICT_TIMES_NEW_ROMAN="1"
RUN_MOTIVATION="1"
RUN_PLOT="1"

CALIB_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/calibration"
CALIBRATION_SCALE_FILE="$CALIB_OUT/layer_head_scales.json"
MOT_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/motivation"
FIG_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/figures"
mkdir -p "$CALIB_OUT" "$MOT_OUT" "$FIG_OUT"
cd "$PROJECT_ROOT"

COMMON_EXTRA_FLAGS=()
[[ "$TRUST_REMOTE_CODE" == "1" ]] && COMMON_EXTRA_FLAGS+=(--trust_remote_code)
[[ "$USE_FAST_TOKENIZER" == "1" ]] && COMMON_EXTRA_FLAGS+=(--use_fast_tokenizer)
HEAD_PROGRESS_FLAGS=()
[[ "$PROGRESS_EVERY_HEAD" == "1" ]] && HEAD_PROGRESS_FLAGS+=(--progress_every_head)
FIG1_SWEEP_FLAGS=()
[[ "$FIG1_RUN_HPARAM_SWEEP" == "1" ]] && FIG1_SWEEP_FLAGS+=(--fig1_enable_hparam_sweep)
FIG2_CR_GRID_FLAGS=()
[[ "$FIG2_RUN_CR_GRID" == "1" ]] && FIG2_CR_GRID_FLAGS+=(--fig2_enable_cr_grid)

if [[ "$RUN_CALIBRATION" == "1" ]]; then
  python -u "$MOTIVATION_DIR/calibrate_layer_head_scales.py" \
    --model "$MODEL" --dataset "$DATASET" --calibration_split "$CALIBRATION_SPLIT" \
    --calibration_num_sequences "$CALIBRATION_NUM_SEQUENCES" \
    --calibration_seqlen "$CALIBRATION_SEQLEN" --calibration_seed "$CALIBRATION_SEED" \
    --prefill_chunk_size "$CALIBRATION_PREFILL_CHUNK_SIZE" \
    --torch_dtype "$TORCH_DTYPE" --attn_implementation "$ATTN_IMPLEMENTATION" \
    --device "$DEVICE" --device_map "$DEVICE_MAP" --output "$CALIBRATION_SCALE_FILE" \
    "${COMMON_EXTRA_FLAGS[@]}" 2>&1 | tee "$CALIB_OUT/run.log"
fi
if [[ ! -s "$CALIBRATION_SCALE_FILE" ]]; then
  echo "ERROR: missing MEDIAN calibration: $CALIBRATION_SCALE_FILE" >&2
  exit 2
fi

if [[ "$RUN_MOTIVATION" == "1" ]]; then
  python -u "$MOTIVATION_DIR/motivation_attention_experiment.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --model_seqlen "$MOT_MODEL_SEQLEN" --max_eval_prefixes "$MOT_MAX_EVAL_PREFIXES" \
    --prefix_stride "$MOT_PREFIX_STRIDE" --torch_dtype "$TORCH_DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" --device "$DEVICE" --device_map "$DEVICE_MAP" \
    --prefill_chunk_size "$MOT_PREFILL_CHUNK_SIZE" --keep_ratio "$MOT_KEEP_RATIO" \
    --calibration_scale_file "$CALIBRATION_SCALE_FILE" \
    --routing_score_chunk_size "$MOT_ROUTING_SCORE_CHUNK_SIZE" \
    --lambda_value "$LAMBDA" \
    --fig1_cosine_tau "$FIG1_COSINE_TAU" \
    --fig1_lambda_grid "$FIG1_LAMBDA_GRID" \
    --fig1_cosine_tau_grid "$FIG1_COSINE_TAU_GRID" \
    --fig1_cr_num_removals "$FIG1_CR_NUM_REMOVALS" \
    --fig1_cr_num_removals_grid "$FIG1_CR_NUM_REMOVALS_GRID" \
    --fig2_cosine_tau "$FIG2_COSINE_TAU" \
    --fig2_euclidean_lambda_grid "$FIG2_EUCLIDEAN_LAMBDA_GRID" \
    --fig2_cosine_tau_grid "$FIG2_COSINE_TAU_GRID" \
    --fig2_cr_num_removals "$FIG2_CR_NUM_REMOVALS" \
    --fig2_cr_num_removals_grid "$FIG2_CR_NUM_REMOVALS_GRID" \
    --max_groups_per_head "$FIG1_MAX_GROUPS_PER_HEAD" \
    --keepkv_sink_size "$KEEPKV_SINK_SIZE" \
    --keepkv_recent_protect_size "$KEEPKV_RECENT_PROTECT_SIZE" \
    --keepkv_score_window "$KEEPKV_SCORE_WINDOW" \
    --keepkv_recency_beta "$KEEPKV_RECENCY_BETA" \
    --layer_selection "$MOT_LAYER_SELECTION" --kv_head_selection "$MOT_KV_HEAD_SELECTION" \
    --output_dir "$MOT_OUT" \
    "${FIG1_SWEEP_FLAGS[@]}" "${FIG2_CR_GRID_FLAGS[@]}" \
    "${COMMON_EXTRA_FLAGS[@]}" "${HEAD_PROGRESS_FLAGS[@]}" 2>&1 | tee "$MOT_OUT/run.log"
fi

if [[ "$RUN_PLOT" == "1" ]]; then
  python -u "$MOTIVATION_DIR/plot_motivation_attention.py" \
    --motivation_dir "$MOT_OUT" --output_dir "$FIG_OUT" --dpi "$PNG_DPI" \
    --strict_times_new_roman "$STRICT_TIMES_NEW_ROMAN" 2>&1 | tee "$FIG_OUT/plot.log"
fi

echo "Calibration: $CALIBRATION_SCALE_FILE"
echo "Motivation results: $MOT_OUT"
echo "Motivation PNG/PDF: $FIG_OUT"
