#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================

cd /home/tlxr/SimCalKV

# =========================
# 手动修改区
# =========================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 脚本完整路径
ANALYSIS_SCRIPT="/home/tlxr/SimCalKV/motivation_experiments/star_structure_experiment.py"
PLOT_SCRIPT="/home/tlxr/SimCalKV/motivation_experiments/plot_star_structure.py"

# 结果根目录。实际输出会自动按 MODEL 分目录，避免不同模型互相覆盖。
RESULTS_ROOT="/home/tlxr/SimCalKV/motivation_experiments/results"
EXPERIMENT_NAME="star_structure_test"

# 是否重新执行真实-Key分析 / 是否画图。
# 已经有 summary_overall.csv + raw/group_records_plot_tau.npz 时，可设 RUN_ANALYSIS=0 直接重画。
RUN_ANALYSIS=1
RUN_PLOT=1

# 模型设置
# MODEL="meta-llama/Llama-3.2-3B"
MODEL="meta-llama/Meta-Llama-3.1-8B"
# MODEL="Qwen/Qwen2.5-3B"
# MODEL="Qwen/Qwen2.5-7B"

# 数据设置：预实验只使用 WikiText-2 test token stream。
DATASET="wikitext2"
MODEL_SEQLEN=8192
MAX_EVAL_SEQUENCES=30

# 模型运行设置
TORCH_DTYPE="bfloat16"   # auto | float16 | bfloat16 | float32
ATTN_IMPL="eager"
DEVICE="cuda"
DEVICE_MAP="auto"
TRUST_REMOTE_CODE=0
USE_FAST_TOKENIZER=0

# 为避免长序列 eager attention 的 O(L^2) 峰值显存，按 query 维度分段 prefill。
# 不做 KV 压缩；最终分析的是完整原始 cached keys。
PREFILL_CHUNK_SIZE=128

# 中性的 random-anchor many-to-one 设置。
# rho = N_anchor / L，表示 retained KV / anchor budget ratio。
ANCHOR_BUDGETS=(
  "0.2"
  "0.4"
  "0.6"
  "0.8"
)

# 分析阶段统计全部阈值。
COSINE_THRESHOLDS=(
  "0.75"
  "0.85"
  "0.95"
)

# 子图 (a) 展示哪些阈值。建议只放 3 条线，避免单栏图例拥挤。
PANEL_A_TAUS=(
  "0.75"
  "0.85"
  "0.95"
)

# 分析仍记录一个 tau 的 group-level raw records，便于后续诊断；最终两张主图都直接读取 summary_overall.csv。
# 该值必须包含在 COSINE_THRESHOLDS 中。
PLOT_TAU="0.85"

# Anchor partition 重复随机种子。
ANCHOR_SEEDS=(
  0
  1
  2
)

# candidate -> anchor cosine GEMM 的 candidate tile 大小；只影响临时显存和速度。
SIMILARITY_CHUNK_SIZE=128

# 分析哪些层 / KV heads。
# "all"：全部；也支持 "0,4,8" 或 Python slice 风格 "0:28:4"。
LAYER_SELECTION="all"
KV_HEAD_SELECTION="all"

# 数值边界。只有 leaf-leaf cosine < tau - NUMERIC_EPS 才计为冲突。
NUMERIC_EPS=1e-6

# 当某个 budget 在 PLOT_TAU 下 eligible multi-candidate groups 太少时打印警告。
MIN_ELIGIBLE_GROUPS_WARN=100

# =========================
# IEEE 绘图设置
# =========================

# Times New Roman 必须真实安装。STRICT_TIMES_NEW_ROMAN=1 时缺失直接报错，不静默 fallback。
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=1
PNG_DPI=600

# 单独子图尺寸，以及横向两图拼接尺寸。
PANEL_WIDTH_IN=1.72
PANEL_HEIGHT_IN=1.58
COMBINED_WIDTH_IN=3.50
COMBINED_HEIGHT_IN=1.58

# 子图 (a) y 轴上限。"auto" 会根据当前结果向上取整到合适的 10%。
PANEL_A_Y_MAX_PERCENT="auto"

# 子图 (b) 展示 pooled pair-level conflict rate：
# conflict_pairs / candidate_pairs，即同一 anchor 下全部 leaf-leaf pairs 中有多少违反同一个 tau。
# 建议与 PANEL_A_TAUS 保持一致，方便两图横向比较。
PANEL_B_TAUS=(
  "0.75"
  "0.85"
  "0.95"
)

# 子图 (b) y 轴上限；auto 根据结果自动向上取整。
PANEL_B_Y_MAX_PERCENT="auto"

# =========================
# 自动区：一般不用改
# =========================

if [[ ! -f "${ANALYSIS_SCRIPT}" ]]; then
  echo "[ERROR] ANALYSIS_SCRIPT not found: ${ANALYSIS_SCRIPT}"
  exit 1
fi

if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "[ERROR] PLOT_SCRIPT not found: ${PLOT_SCRIPT}"
  exit 1
fi

join_by_comma() {
  local IFS=,
  echo "$*"
}

ANCHOR_BUDGETS_CSV="$(join_by_comma "${ANCHOR_BUDGETS[@]}")"
COSINE_THRESHOLDS_CSV="$(join_by_comma "${COSINE_THRESHOLDS[@]}")"
PANEL_A_TAUS_CSV="$(join_by_comma "${PANEL_A_TAUS[@]}")"
PANEL_B_TAUS_CSV="$(join_by_comma "${PANEL_B_TAUS[@]}")"
ANCHOR_SEEDS_CSV="$(join_by_comma "${ANCHOR_SEEDS[@]}")"
MODEL_BASENAME="${MODEL%/}"
MODEL_BASENAME="${MODEL_BASENAME##*/}"
MODEL_DIR_NAME="$(printf '%s' "${MODEL_BASENAME}" | sed 's/[^A-Za-z0-9._-]/_/g')"
if [[ -z "${MODEL_DIR_NAME}" ]]; then
  echo "[ERROR] Could not derive a model directory name from MODEL=${MODEL}"
  exit 1
fi
OUT_DIR="${RESULTS_ROOT}/${MODEL_DIR_NAME}/${EXPERIMENT_NAME}"
MODEL_SLUG="$(printf '%s' "${MODEL}" | tr '/:.' '___')"

mkdir -p "${OUT_DIR}"
RUN_LOG="${OUT_DIR}/run.log"
: > "${RUN_LOG}"

echo "ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT}" | tee -a "${RUN_LOG}"
echo "PLOT_SCRIPT=${PLOT_SCRIPT}" | tee -a "${RUN_LOG}"
echo "RESULTS_ROOT=${RESULTS_ROOT}" | tee -a "${RUN_LOG}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}" | tee -a "${RUN_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "MODEL=${MODEL}" | tee -a "${RUN_LOG}"
echo "MODEL_DIR_NAME=${MODEL_DIR_NAME}" | tee -a "${RUN_LOG}"
echo "MODEL_SLUG=${MODEL_SLUG}" | tee -a "${RUN_LOG}"
echo "DATASET=${DATASET}" | tee -a "${RUN_LOG}"
echo "MODEL_SEQLEN=${MODEL_SEQLEN}" | tee -a "${RUN_LOG}"
echo "MAX_EVAL_SEQUENCES=${MAX_EVAL_SEQUENCES}" | tee -a "${RUN_LOG}"
echo "TORCH_DTYPE=${TORCH_DTYPE}" | tee -a "${RUN_LOG}"
echo "ATTN_IMPL=${ATTN_IMPL}" | tee -a "${RUN_LOG}"
echo "DEVICE=${DEVICE}" | tee -a "${RUN_LOG}"
echo "DEVICE_MAP=${DEVICE_MAP}" | tee -a "${RUN_LOG}"
echo "PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "ANCHOR_BUDGETS=${ANCHOR_BUDGETS[*]}" | tee -a "${RUN_LOG}"
echo "COSINE_THRESHOLDS=${COSINE_THRESHOLDS[*]}" | tee -a "${RUN_LOG}"
echo "PANEL_A_TAUS=${PANEL_A_TAUS[*]}" | tee -a "${RUN_LOG}"
echo "PLOT_TAU=${PLOT_TAU}" | tee -a "${RUN_LOG}"
echo "ANCHOR_SEEDS=${ANCHOR_SEEDS[*]}" | tee -a "${RUN_LOG}"
echo "SIMILARITY_CHUNK_SIZE=${SIMILARITY_CHUNK_SIZE}" | tee -a "${RUN_LOG}"
echo "LAYER_SELECTION=${LAYER_SELECTION}" | tee -a "${RUN_LOG}"
echo "KV_HEAD_SELECTION=${KV_HEAD_SELECTION}" | tee -a "${RUN_LOG}"
echo "NUMERIC_EPS=${NUMERIC_EPS}" | tee -a "${RUN_LOG}"
echo "FONT_FAMILY=${FONT_FAMILY}" | tee -a "${RUN_LOG}"
echo "STRICT_TIMES_NEW_ROMAN=${STRICT_TIMES_NEW_ROMAN}" | tee -a "${RUN_LOG}"
echo "PANEL_WIDTH_IN=${PANEL_WIDTH_IN}" | tee -a "${RUN_LOG}"
echo "PANEL_HEIGHT_IN=${PANEL_HEIGHT_IN}" | tee -a "${RUN_LOG}"
echo "COMBINED_WIDTH_IN=${COMBINED_WIDTH_IN}" | tee -a "${RUN_LOG}"
echo "COMBINED_HEIGHT_IN=${COMBINED_HEIGHT_IN}" | tee -a "${RUN_LOG}"
echo "PNG_DPI=${PNG_DPI}" | tee -a "${RUN_LOG}"
echo "PANEL_A_Y_MAX_PERCENT=${PANEL_A_Y_MAX_PERCENT}" | tee -a "${RUN_LOG}"
echo "PANEL_B_TAUS=${PANEL_B_TAUS[*]}" | tee -a "${RUN_LOG}"
echo "PANEL_B_Y_MAX_PERCENT=${PANEL_B_Y_MAX_PERCENT}" | tee -a "${RUN_LOG}"
echo | tee -a "${RUN_LOG}"

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  ANALYSIS_CMD=(
    python -u "${ANALYSIS_SCRIPT}"
    --model "${MODEL}"
    --dataset "${DATASET}"
    --model_seqlen "${MODEL_SEQLEN}"
    --max_eval_sequences "${MAX_EVAL_SEQUENCES}"
    --torch_dtype "${TORCH_DTYPE}"
    --attn_implementation "${ATTN_IMPL}"
    --device "${DEVICE}"
    --device_map "${DEVICE_MAP}"
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE}"
    --anchor_budgets "${ANCHOR_BUDGETS_CSV}"
    --cosine_thresholds "${COSINE_THRESHOLDS_CSV}"
    --record_tau "${PLOT_TAU}"
    --anchor_seeds "${ANCHOR_SEEDS_CSV}"
    --similarity_chunk_size "${SIMILARITY_CHUNK_SIZE}"
    --layer_selection "${LAYER_SELECTION}"
    --kv_head_selection "${KV_HEAD_SELECTION}"
    --numeric_eps "${NUMERIC_EPS}"
    --min_eligible_groups_warn "${MIN_ELIGIBLE_GROUPS_WARN}"
    --output_dir "${OUT_DIR}"
  )

  if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
    ANALYSIS_CMD+=( --trust_remote_code )
  fi
  if [[ "${USE_FAST_TOKENIZER}" -eq 1 ]]; then
    ANALYSIS_CMD+=( --use_fast_tokenizer )
  fi

  echo "[ANALYSIS CMD]" | tee -a "${RUN_LOG}"
  printf ' %q' "${ANALYSIS_CMD[@]}" | tee -a "${RUN_LOG}"
  echo | tee -a "${RUN_LOG}"
  "${ANALYSIS_CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

if [[ "${RUN_PLOT}" -eq 1 ]]; then
  PLOT_CMD=(
    python -u "${PLOT_SCRIPT}"
    --input_dir "${OUT_DIR}"
    --panel_a_taus "${PANEL_A_TAUS_CSV}"
    --font_family "${FONT_FAMILY}"
    --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}"
    --panel_width_in "${PANEL_WIDTH_IN}"
    --panel_height_in "${PANEL_HEIGHT_IN}"
    --combined_width_in "${COMBINED_WIDTH_IN}"
    --combined_height_in "${COMBINED_HEIGHT_IN}"
    --png_dpi "${PNG_DPI}"
    --panel_a_y_max_percent "${PANEL_A_Y_MAX_PERCENT}"
    --panel_b_taus "${PANEL_B_TAUS_CSV}"
    --panel_b_y_max_percent "${PANEL_B_Y_MAX_PERCENT}"
  )

  echo "[PLOT CMD]" | tee -a "${RUN_LOG}"
  printf ' %q' "${PLOT_CMD[@]}" | tee -a "${RUN_LOG}"
  echo | tee -a "${RUN_LOG}"
  "${PLOT_CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi

echo | tee -a "${RUN_LOG}"
echo "[DONE] Results: ${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "[PANEL A PDF] ${OUT_DIR}/figures/panel_a_star_prevalence.pdf" | tee -a "${RUN_LOG}"
echo "[PANEL A PNG] ${OUT_DIR}/figures/panel_a_star_prevalence.png" | tee -a "${RUN_LOG}"
echo "[PANEL B PDF] ${OUT_DIR}/figures/panel_b_pair_conflict_rate.pdf" | tee -a "${RUN_LOG}"
echo "[PANEL B PNG] ${OUT_DIR}/figures/panel_b_pair_conflict_rate.png" | tee -a "${RUN_LOG}"
echo "[COMBINED PDF] ${OUT_DIR}/figures/star_motivation_two_panel.pdf" | tee -a "${RUN_LOG}"
echo "[COMBINED PNG] ${OUT_DIR}/figures/star_motivation_two_panel.png" | tee -a "${RUN_LOG}"
echo "[FIGURE STATS] ${OUT_DIR}/figures/figure_statistics.csv" | tee -a "${RUN_LOG}"
