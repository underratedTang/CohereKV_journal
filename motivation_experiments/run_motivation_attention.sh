#!/usr/bin/env bash
set -euo pipefail

# Focused CohereKV preliminary suite.
# Only motivation_experiments/ is modified by this package.
#
# Each figure owns its hyperparameters. Optional sweeps are deliberately compact:
#   Figure 1: OFAT routing/conflict/CR-removal sweep around the primary point.
#   Figure 2: routing-threshold curves for Merge-All and per-Star-group CR(K);
#             optional CR(K) grid and optional conflict OFAT sweep.
#   Figure 3-left/right: OFAT original/route/conflict sweep, never Cartesian.

PROJECT_ROOT="/home/tlxr/CohereKV"
MOTIVATION_DIR="$PROJECT_ROOT/motivation_experiments"
RESULTS_ROOT="$MOTIVATION_DIR/results"

# MODEL="meta-llama/Meta-Llama-3.1-8B"
# MODEL_DIR_NAME="Meta-Llama-3.1-8B_focused_preliminary_v12_wholelayer_fig2_fig3left"
# MODEL="meta-llama/Llama-3.2-3B"
# MODEL_DIR_NAME="Meta-Llama-3.2-3B_focused_preliminary_v12_wholelayer_fig2_fig3left"
MODEL="meta-llama/Llama-3.2-1B"
MODEL_DIR_NAME="Meta-Llama-3.2-1B_focused_preliminary_v12_wholelayer_fig2_fig3left"

DATASET="wikitext2"
TORCH_DTYPE="bfloat16"
ATTN_IMPLEMENTATION="eager"
DEVICE="cuda:1"
DEVICE_MAP="none"
TRUST_REMOTE_CODE="0"
USE_FAST_TOKENIZER="0"
PROGRESS_EVERY_HEAD="1"

# Shared KeepKV selection mechanics. These are not routing/conflict hyperparameters.
KEEPKV_SINK_SIZE="4"
KEEPKV_RECENT_PROTECT_SIZE="32"
KEEPKV_SCORE_WINDOW="32"
KEEPKV_RECENCY_BETA="1.0"

# ==============================================================================
# Frozen per-(layer,KV-head) K scales used by Figure 1/2 and Modified/Merge-All.
# Multiple disjoint WikiText2-train calibration windows are run once.
# S_lh^calib = sqrt(mean_m S_lh,m^2), then frozen across all later prefixes
# and compression rounds. Original CohereKV remains untouched.
# ==============================================================================
CALIBRATION_SPLIT="train"
CALIBRATION_NUM_SEQUENCES="16"
CALIBRATION_SEQLEN="8192"
CALIBRATION_SEED="42"
CALIBRATION_PREFILL_CHUNK_SIZE="256"
RUN_CALIBRATION="1"

# ==============================================================================
# Figure 1 / Motivation 1
# Star-conflict generality + configurable-K severe-conflict filtering counterfactual.
# For each selected conflict, evict-u and evict-v are both evaluated; the better
# legal filtering branch is followed. This is a motivation-only potential test,
# not the Figure-3 Modified CohereKV selector.
# ==============================================================================
FIG1_EUCLIDEAN_ROUTE_LAMBDA="0.40"
FIG1_COSINE_ROUTE_LAMBDA="0.90"
FIG1_CONFLICT_LAMBDA="0.40"
FIG1_RUN_HPARAM_SWEEP="0"
FIG1_EUCLIDEAN_ROUTE_GRID="0.30,0.40,0.50"
FIG1_COSINE_ROUTE_GRID="0.85,0.90,0.95"
FIG1_CONFLICT_GRID="0.40,0.50,0.60"

FIG1_MAX_GROUPS_PER_HEAD="0"
FIG1_CR_NUM_REMOVALS="1"
FIG1_CR_NUM_REMOVALS_GRID="1,2,4"

# ==============================================================================
# Figure 2 / Motivation 2
# Routing threshold is the x-axis.  At every threshold, Conflict-Reduced applies
# the same counterfactual independently to EVERY Star group, with K interpreted
# per Star group (the same meaning as Figure 1).  Default: Merge-All + CR(K=1).
# Set FIG2_RUN_CR_GRID=1 to add CR(K=2), CR(K=4), ... as extra curves.
# ==============================================================================
FIG2_EUCLIDEAN_ROUTE_LAMBDA="0.40"
FIG2_COSINE_ROUTE_LAMBDA="0.90"
FIG2_CONFLICT_LAMBDA="0.40"
FIG2_EUCLIDEAN_ROUTE_GRID="0.30,0.40,0.50"
FIG2_COSINE_ROUTE_GRID="0.85,0.90,0.95"
FIG2_RUN_HPARAM_SWEEP="0"
FIG2_CONFLICT_GRID="0.40,0.50,0.60"
FIG2_CR_NUM_REMOVALS="1"
FIG2_RUN_CR_GRID="1"
FIG2_CR_NUM_REMOVALS_GRID="1,2,4"

# Figure 1/2 share only data/prefill coverage so one frozen prefix/cache can serve both.
MOT_MODEL_SEQLEN="4096"
MOT_MAX_EVAL_PREFIXES="4"
MOT_PREFIX_STRIDE="0"
MOT_PREFILL_CHUNK_SIZE="256"
MOT_KEEP_RATIO="0.10" # 
MOT_ROUTING_SCORE_CHUNK_SIZE="1024"
# MOT_LAYER_SELECTION="0,8,16,24,31"
# MOT_LAYER_SELECTION="0,8,16,24"
MOT_LAYER_SELECTION="0,4,8,12,15"


# MOT_LAYER_SELECTION="all"

MOT_KV_HEAD_SELECTION="0,1,2,3,4,5,6,7"

# ==============================================================================
# Figure 3-left / single-round whole-layer post-o_proj attention-output error
# Eviction-only / Merge-All / Original / Modified. All KV/query heads are always used.
# Independent from PPL parameters below.
# ==============================================================================
FIG3L_MODEL_SEQLEN="8192"
FIG3L_MAX_EVAL_PREFIXES="16"
FIG3L_PREFIX_STRIDE="0"
FIG3L_KEEP_RATIO="0.20"
FIG3L_PREFILL_CHUNK_SIZE="256"
FIG3L_ORIGINAL_LAMBDA="0.40"
FIG3L_ROUTE_LAMBDA="0.40"
FIG3L_CONFLICT_LAMBDA="0.40"
FIG3L_RUN_HPARAM_SWEEP="0"
FIG3L_ORIGINAL_LAMBDA_GRID="0.30,0.40,0.50"
FIG3L_ROUTE_LAMBDA_GRID="0.30,0.40,0.50"
FIG3L_CONFLICT_LAMBDA_GRID="0.05,0.10,0.20"
FIG3L_DISTANCE_CHUNK_SIZE="1024"
# FIG3L_LAYER_SELECTION="0,8,16,24,31"
FIG3L_LAYER_SELECTION="all"

FIG3L_KV_HEAD_SELECTION="0,1,2,3,4,5,6,7"

# ==============================================================================
# Figure 3-right / repository evaluate_perplexity PPL
# Full / Eviction-only / Merge-All / Original / Modified.
# PPL routing/conflict/original lambdas are deliberately independent of 3-left.
# ==============================================================================
FIG3R_MODEL_SEQLEN="8192"
FIG3R_CHUNK_SIZE="32"
FIG3R_MAX_EVAL_SEQUENCES="4"
FIG3R_NSAMPLES="32"
FIG3R_COMPRESS_RATIO="0.80"
FIG3R_MIN_KV_LEN="2"
FIG3R_COMPRESS_EVERY="1"
FIG3R_AMP_DTYPE="bfloat16"
FIG3R_ORIGINAL_LAMBDA="0.40"
FIG3R_ROUTE_LAMBDA="0.40"
FIG3R_CONFLICT_LAMBDA="0.40"
FIG3R_RUN_HPARAM_SWEEP="0"
FIG3R_ORIGINAL_LAMBDA_GRID="0.30,0.40,0.50"
FIG3R_ROUTE_LAMBDA_GRID="0.30,0.40,0.50"
FIG3R_CONFLICT_LAMBDA_GRID="0.05,0.10,0.20"
FIG3R_DISTANCE_CHUNK_SIZE="1024"

PNG_DPI="240"
RUN_MOTIVATION="1"
RUN_SINGLE_ROUND="1"
RUN_PPL="1"
RUN_PLOT="1"

CALIB_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/calibration"
CALIBRATION_SCALE_FILE="$CALIB_OUT/layer_head_scales.json"
MOT_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/motivation"
METHOD_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/single_round"
PPL_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/ppl"
FIG_OUT="$RESULTS_ROOT/$MODEL_DIR_NAME/figures"
mkdir -p "$CALIB_OUT" "$MOT_OUT" "$METHOD_OUT" "$PPL_OUT" "$FIG_OUT"
cd "$PROJECT_ROOT"

COMMON_EXTRA_FLAGS=()
[[ "$TRUST_REMOTE_CODE" == "1" ]] && COMMON_EXTRA_FLAGS+=(--trust_remote_code)
[[ "$USE_FAST_TOKENIZER" == "1" ]] && COMMON_EXTRA_FLAGS+=(--use_fast_tokenizer)
HEAD_PROGRESS_FLAGS=()
[[ "$PROGRESS_EVERY_HEAD" == "1" ]] && HEAD_PROGRESS_FLAGS+=(--progress_every_head)
FIG1_SWEEP_FLAGS=()
[[ "$FIG1_RUN_HPARAM_SWEEP" == "1" ]] && FIG1_SWEEP_FLAGS+=(--fig1_enable_hparam_sweep)
FIG2_SWEEP_FLAGS=()
[[ "$FIG2_RUN_HPARAM_SWEEP" == "1" ]] && FIG2_SWEEP_FLAGS+=(--fig2_enable_hparam_sweep)
FIG2_CR_GRID_FLAGS=()
[[ "$FIG2_RUN_CR_GRID" == "1" ]] && FIG2_CR_GRID_FLAGS+=(--fig2_enable_cr_grid)
FIG3L_SWEEP_FLAGS=()
[[ "$FIG3L_RUN_HPARAM_SWEEP" == "1" ]] && FIG3L_SWEEP_FLAGS+=(--enable_hparam_sweep)
FIG3R_SWEEP_FLAGS=()
[[ "$FIG3R_RUN_HPARAM_SWEEP" == "1" ]] && FIG3R_SWEEP_FLAGS+=(--enable_hparam_sweep)

if [[ "$RUN_CALIBRATION" == "1" ]]; then
  echo "=== [0] Calibrate frozen per-layer/head K scales ==="
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
  echo "ERROR: frozen calibration-scale file is missing: $CALIBRATION_SCALE_FILE" >&2
  echo "Set RUN_CALIBRATION=1 or provide the previously calibrated file." >&2
  exit 2
fi

if [[ "$RUN_MOTIVATION" == "1" ]]; then
  echo "=== [A] Figure 1/2 motivation: Euclidean + cosine ==="
  python -u "$MOTIVATION_DIR/motivation_attention_experiment.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --model_seqlen "$MOT_MODEL_SEQLEN" --max_eval_prefixes "$MOT_MAX_EVAL_PREFIXES" \
    --prefix_stride "$MOT_PREFIX_STRIDE" --torch_dtype "$TORCH_DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" --device "$DEVICE" --device_map "$DEVICE_MAP" \
    --prefill_chunk_size "$MOT_PREFILL_CHUNK_SIZE" --keep_ratio "$MOT_KEEP_RATIO" \
    --calibration_scale_file "$CALIBRATION_SCALE_FILE" \
    --routing_score_chunk_size "$MOT_ROUTING_SCORE_CHUNK_SIZE" \
    --fig1_euclidean_route_lambda "$FIG1_EUCLIDEAN_ROUTE_LAMBDA" \
    --fig1_cosine_route_lambda "$FIG1_COSINE_ROUTE_LAMBDA" \
    --fig1_conflict_lambda "$FIG1_CONFLICT_LAMBDA" \
    --fig1_euclidean_route_grid "$FIG1_EUCLIDEAN_ROUTE_GRID" \
    --fig1_cosine_route_grid "$FIG1_COSINE_ROUTE_GRID" \
    --fig1_conflict_grid "$FIG1_CONFLICT_GRID" \
    --fig1_cr_num_removals "$FIG1_CR_NUM_REMOVALS" \
    --fig1_cr_num_removals_grid "$FIG1_CR_NUM_REMOVALS_GRID" \
    --fig2_euclidean_route_lambda "$FIG2_EUCLIDEAN_ROUTE_LAMBDA" \
    --fig2_cosine_route_lambda "$FIG2_COSINE_ROUTE_LAMBDA" \
    --fig2_conflict_lambda "$FIG2_CONFLICT_LAMBDA" \
    --fig2_euclidean_route_grid "$FIG2_EUCLIDEAN_ROUTE_GRID" \
    --fig2_cosine_route_grid "$FIG2_COSINE_ROUTE_GRID" \
    --fig2_conflict_grid "$FIG2_CONFLICT_GRID" \
    --fig2_cr_num_removals "$FIG2_CR_NUM_REMOVALS" \
    --fig2_cr_num_removals_grid "$FIG2_CR_NUM_REMOVALS_GRID" \
    --max_groups_per_head "$FIG1_MAX_GROUPS_PER_HEAD" \
    --keepkv_sink_size "$KEEPKV_SINK_SIZE" --keepkv_recent_protect_size "$KEEPKV_RECENT_PROTECT_SIZE" \
    --keepkv_score_window "$KEEPKV_SCORE_WINDOW" --keepkv_recency_beta "$KEEPKV_RECENCY_BETA" \
    --layer_selection "$MOT_LAYER_SELECTION" --kv_head_selection "$MOT_KV_HEAD_SELECTION" \
    --output_dir "$MOT_OUT" \
    "${FIG1_SWEEP_FLAGS[@]}" "${FIG2_SWEEP_FLAGS[@]}" "${FIG2_CR_GRID_FLAGS[@]}" \
    "${COMMON_EXTRA_FLAGS[@]}" "${HEAD_PROGRESS_FLAGS[@]}" 2>&1 | tee "$MOT_OUT/run.log"
fi

if [[ "$RUN_SINGLE_ROUND" == "1" ]]; then
  echo "=== [B] Figure 3-left: Eviction-only / Merge-All / Original / Modified ==="
  python -u "$MOTIVATION_DIR/method_design_experiment.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --model_seqlen "$FIG3L_MODEL_SEQLEN" --max_eval_prefixes "$FIG3L_MAX_EVAL_PREFIXES" \
    --prefix_stride "$FIG3L_PREFIX_STRIDE" --torch_dtype "$TORCH_DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" --device "$DEVICE" --device_map "$DEVICE_MAP" \
    --prefill_chunk_size "$FIG3L_PREFILL_CHUNK_SIZE" --keep_ratio "$FIG3L_KEEP_RATIO" \
    --calibration_scale_file "$CALIBRATION_SCALE_FILE" \
    --original_lambda "$FIG3L_ORIGINAL_LAMBDA" --route_lambda "$FIG3L_ROUTE_LAMBDA" \
    --conflict_lambda "$FIG3L_CONFLICT_LAMBDA" \
    --original_lambda_grid "$FIG3L_ORIGINAL_LAMBDA_GRID" \
    --route_lambda_grid "$FIG3L_ROUTE_LAMBDA_GRID" \
    --conflict_lambda_grid "$FIG3L_CONFLICT_LAMBDA_GRID" \
    --keepkv_sink_size "$KEEPKV_SINK_SIZE" --keepkv_recent_protect_size "$KEEPKV_RECENT_PROTECT_SIZE" \
    --keepkv_score_window "$KEEPKV_SCORE_WINDOW" --keepkv_recency_beta "$KEEPKV_RECENCY_BETA" \
    --distance_chunk_size "$FIG3L_DISTANCE_CHUNK_SIZE" \
    --layer_selection "$FIG3L_LAYER_SELECTION" --kv_head_selection "$FIG3L_KV_HEAD_SELECTION" \
    --output_dir "$METHOD_OUT" \
    "${FIG3L_SWEEP_FLAGS[@]}" "${COMMON_EXTRA_FLAGS[@]}" "${HEAD_PROGRESS_FLAGS[@]}" \
    2>&1 | tee "$METHOD_OUT/run.log"
fi

if [[ "$RUN_PPL" == "1" ]]; then
  echo "=== [C] Figure 3-right: Full / Eviction-only / Merge-All / Original / Modified PPL ==="
  python -u "$MOTIVATION_DIR/ppl_continuation_experiment.py" \
    --model "$MODEL" --dataset "$DATASET" --model_seqlen "$FIG3R_MODEL_SEQLEN" \
    --chunk_size "$FIG3R_CHUNK_SIZE" --max_eval_sequences "$FIG3R_MAX_EVAL_SEQUENCES" \
    --nsamples "$FIG3R_NSAMPLES" --compress_ratio "$FIG3R_COMPRESS_RATIO" \
    --calibration_scale_file "$CALIBRATION_SCALE_FILE" \
    --min_kv_len "$FIG3R_MIN_KV_LEN" --compress_every "$FIG3R_COMPRESS_EVERY" \
    --torch_dtype "$TORCH_DTYPE" --amp_dtype "$FIG3R_AMP_DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" --device "$DEVICE" --device_map "$DEVICE_MAP" \
    --variants "full,eviction_only,merge_all,original_coherekv,modified_coherekv" \
    --original_lambda "$FIG3R_ORIGINAL_LAMBDA" --route_lambda "$FIG3R_ROUTE_LAMBDA" \
    --conflict_lambda "$FIG3R_CONFLICT_LAMBDA" \
    --original_lambda_grid "$FIG3R_ORIGINAL_LAMBDA_GRID" \
    --route_lambda_grid "$FIG3R_ROUTE_LAMBDA_GRID" \
    --conflict_lambda_grid "$FIG3R_CONFLICT_LAMBDA_GRID" \
    --keepkv_sink_size "$KEEPKV_SINK_SIZE" --keepkv_recent_protect_size "$KEEPKV_RECENT_PROTECT_SIZE" \
    --keepkv_score_window "$KEEPKV_SCORE_WINDOW" --keepkv_recency_beta "$KEEPKV_RECENCY_BETA" \
    --distance_chunk_size "$FIG3R_DISTANCE_CHUNK_SIZE" --output_dir "$PPL_OUT" \
    "${FIG3R_SWEEP_FLAGS[@]}" "${COMMON_EXTRA_FLAGS[@]}" 2>&1 | tee "$PPL_OUT/run.log"
fi

if [[ "$RUN_PLOT" == "1" ]]; then
  echo "=== [D] Plot Figure 1 / Figure 2 / Figure 3 ==="
  python -u "$MOTIVATION_DIR/plot_motivation_attention.py" \
    --motivation_dir "$MOT_OUT" --method_dir "$METHOD_OUT" --ppl_dir "$PPL_OUT" \
    --output_dir "$FIG_OUT" --dpi "$PNG_DPI" 2>&1 | tee "$FIG_OUT/plot.log"
fi

echo "=== Done ==="
echo "calibration scales: $CALIBRATION_SCALE_FILE"
echo "motivation:   $MOT_OUT"
echo "single-round: $METHOD_OUT"
echo "PPL:          $PPL_OUT"
echo "figures:      $FIG_OUT"
