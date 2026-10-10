#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 项目路径
# ============================================================
cd /home/tlxr/CohereKV

# ============================================================
# 手动修改区
# 数据全部通过命令行参数传给 Python，不使用环境变量传递实验结果。
# ============================================================

PLOT_SCRIPT="/home/tlxr/CohereKV/plot_lambda_robustness.py"
OUTPUT_DIR="/home/tlxr/CohereKV/results/figures/lambda_robustness"
COMBINED_OUTPUT_STEM="lambda_robustness_combined"

# ------------------------------------------------------------
# 1) lambda：可任意增删，但必须按从小到大排列。
# ------------------------------------------------------------
LAMBDAS=(
  "0.1"
  "0.5"
  "0.9"
  "1.3"
)

# ------------------------------------------------------------
# 2) 模型：数组顺序决定合并图从上到下的 panel 顺序。
# 当前：
#   (a) Llama-3.2-3B
#   (b) Qwen2.5-3B
# ------------------------------------------------------------
MODEL_NAMES=(
  "Llama-3.2-3B"
  "Qwen2.5-3B"
)

# ------------------------------------------------------------
# 3) PPL：每行对应一个模型，数值个数必须等于 LAMBDAS 个数。
# ------------------------------------------------------------
PPL_RESULTS=(
  "7.55,7.54,8.45,9.04"
  "7.66,7.66,19.41,21.69"
)

# ------------------------------------------------------------
# 4) LongBench Avg.：每行对应一个模型。
# 未跑出的值写 nan / NA / --，之后直接替换。
# ------------------------------------------------------------
LONGBENCH_RESULTS=(
  "41.94,41.75,41.06,40.55"
  "35.16,35.62,35.15,35.00"
)

# ------------------------------------------------------------
# 5) IEEE 绘图设置
# ------------------------------------------------------------
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=1
PNG_DPI=600

PANEL_WIDTH_IN=1.72
PANEL_HEIGHT_IN=1.25

COMBINED_WIDTH_IN=2.00
COMBINED_HEIGHT_IN=1.25

# 两个子图的垂直间距。越小越靠近。
COMBINED_HSPACE=0.45

ANNOTATION_FONTSIZE=4.5

# LongBench 柱统一使用橙色斜条纹；"//" 比较适合单栏图。
BAR_HATCH="//"

# ------------------------------------------------------------
# 6) y 轴范围
# PPL：每个模型各自自动缩放。
# LongBench：当前下界设为 30。
# ------------------------------------------------------------
PPL_Y_MIN="auto"
PPL_Y_MAX="auto"
LONGBENCH_Y_MIN="30"
LONGBENCH_Y_MAX="auto"

# ============================================================
# 自动区：一般不用修改
# ============================================================

if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "[ERROR] PLOT_SCRIPT not found: ${PLOT_SCRIPT}"
  exit 1
fi

if [[ "${#MODEL_NAMES[@]}" -ne "${#PPL_RESULTS[@]}" ]] || \
   [[ "${#MODEL_NAMES[@]}" -ne "${#LONGBENCH_RESULTS[@]}" ]]; then
  echo "[ERROR] MODEL_NAMES, PPL_RESULTS, and LONGBENCH_RESULTS must have the same number of entries."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PLOT_CMD=(
  python -u "${PLOT_SCRIPT}"
  --output_dir "${OUTPUT_DIR}"
  --combined_output_stem "${COMBINED_OUTPUT_STEM}"
  --font_family "${FONT_FAMILY}"
  --strict_times_new_roman "${STRICT_TIMES_NEW_ROMAN}"
  --panel_width_in "${PANEL_WIDTH_IN}"
  --panel_height_in "${PANEL_HEIGHT_IN}"
  --combined_width_in "${COMBINED_WIDTH_IN}"
  --combined_height_in "${COMBINED_HEIGHT_IN}"
  --combined_hspace "${COMBINED_HSPACE}"
  --png_dpi "${PNG_DPI}"
  --annotation_fontsize "${ANNOTATION_FONTSIZE}"
  --bar_hatch "${BAR_HATCH}"
  --ppl_y_min "${PPL_Y_MIN}"
  --ppl_y_max "${PPL_Y_MAX}"
  --longbench_y_min "${LONGBENCH_Y_MIN}"
  --longbench_y_max "${LONGBENCH_Y_MAX}"
  --lambdas "${LAMBDAS[@]}"
)

for ((i = 0; i < ${#MODEL_NAMES[@]}; i++)); do
  PLOT_CMD+=(
    --model "${MODEL_NAMES[$i]}"
    --ppl "${PPL_RESULTS[$i]}"
    --longbench "${LONGBENCH_RESULTS[$i]}"
  )
done

echo "[PLOT CMD]"
printf ' %q' "${PLOT_CMD[@]}"
echo

"${PLOT_CMD[@]}"

echo
echo "[DONE]"
echo "Output directory: ${OUTPUT_DIR}"
echo "Combined PDF: ${OUTPUT_DIR}/${COMBINED_OUTPUT_STEM}.pdf"
echo "Combined PNG: ${OUTPUT_DIR}/${COMBINED_OUTPUT_STEM}.png"
