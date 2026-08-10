# Star-Structure Motivation Pre-Experiment

该文件夹用于直接放在：

```text
/home/tlxr/SimCalKV/motivation_experiments/
```

与 `task_evaluation/` 平行。预实验只研究真实 LLM cached-Key 空间中的 **many-to-one star structure**，不运行任何 KV-cache 压缩方法，不计算 PPL/LongBench，也不使用 `\method` 的欧氏距离、证书或 Graph-MIS。

## 1. 现象定义

对每个 KV head，按 retained KV / anchor budget ratio

\[
\rho=\frac{N_{\mathrm{anchor}}}{L}
\]

随机选择 anchor positions。剩余 positions 为 candidates。每个 candidate `i` 只分配给 cosine similarity 最大的 anchor：

\[
j_i^\star=\arg\max_{j\in\mathcal R}\cos(K_i,K_j).
\]

只有当

\[
\cos(K_i,K_{j_i^\star})\ge\tau
\]

时，candidate 才进入该 anchor 的 candidate pool `P_j`。因此每个 candidate 至多属于一个 anchor，形成中性的 many-to-one assignment。

对于任意 `|P_j| >= 2` 的 group，如果存在 `i,l in P_j` 使

\[
\cos(K_i,K_j)\ge\tau,\qquad
\cos(K_l,K_j)\ge\tau,
\]

但

\[
\cos(K_i,K_l)<\tau,
\]

则该 group 包含 star conflict。

## 2. Introduction 两张预实验图

本版本将“group-level prevalence”和“pair-level structural inconsistency”分开，并且两张图使用完全相同的 cosine thresholds 和 retained KV budgets。

### (a) Star conflict prevalence

\[
\mathrm{StarGroupRate}(\rho,\tau)
=
\frac{
\#\{j:|P_j|\ge2,\ \exists i,l\in P_j,\ \cos(K_i,K_l)<\tau\}
}{
\#\{j:|P_j|\ge2\}
}.
\]

- x 轴：`Retained KV budget (%)`；
- y 轴：`Groups with star conflict (%)`；
- 不同曲线：不同 cosine threshold `tau`。

它回答：**星形问题有多普遍？**

### (b) Pair-level conflict rate

对同一个 anchor 下形成的全部 leaf--leaf pairs 直接做 pooled 统计：

\[
\boxed{
\mathrm{PairConflictRate}(\rho,\tau)
=
\frac{
\sum_j\#\{(i,l)\subset P_j:\cos(K_i,K_l)<\tau\}
}{
\sum_j\binom{|P_j|}{2}
}.
}
\]

- x 轴：`Retained KV budget (%)`；
- y 轴：`Conflicting leaf pairs (%)`；
- 不同曲线：不同 cosine threshold `tau`。

它回答：**在已经被同一个 anchor 接收并共同分组的 leaf pairs 中，有多少比例本身不满足同一个 cosine criterion？**

该指标直接使用 `summary_overall.csv` 中

```text
conflict_pair_rate = conflict_pairs / candidate_pairs
```

而不是先按 group 求 conflict density 再平均，也不条件化在 star-conflicted groups 上。因此不会出现 size-2 group 一旦冲突就自动得到 100% density 的 group-size artifact。

两张图共同对应：

```text
(a) How common is the star problem at the group level?
(b) How much pairwise inconsistency exists inside the formed groups?
```

## 3. 默认展示阈值

SH 顶部默认：

```bash
PANEL_A_TAUS=(
  "0.70"
  "0.80"
  "0.90"
)

PANEL_B_TAUS=(
  "0.70"
  "0.80"
  "0.90"
)
```

建议两张图保持一致，便于横向比较。完整分析仍可在 `COSINE_THRESHOLDS` 中保留更多阈值，例如 `0.95`。

`PLOT_TAU` 现在只用于分析阶段额外保存一个 group-level raw NPZ，便于以后做诊断；最终两张主图只读取 `summary_overall.csv`。

## 4. 运行

```bash
cd /home/tlxr/SimCalKV
bash motivation_experiments/run_star_structure.sh
```

所有需要修改的变量集中在 `run_star_structure.sh` 顶部“手动修改区”。脚本不使用 `${VAR:-default}` 风格的变量默认值。

默认正式预实验设置：

```bash
MAX_EVAL_SEQUENCES=3
ANCHOR_SEEDS=(0 1 2)
```

### 直接复用已有真实结果重画

最终两张图只需要当前模型目录下的：

```text
summary_overall.csv
run_config.json
```

因此已有 3-sequence 分析结果时可直接设置：

```bash
RUN_ANALYSIS=0
RUN_PLOT=1
```

无需再次运行模型 forward，也不要求旧 raw NPZ 与当前绘图版本完全一致。注意：升级到按模型分目录后，旧版共享目录 `results/star_structure_test/` 不会被自动当作某个模型的结果；如需复用旧结果，应先根据其中 `run_config.json` 的 `model` 字段手工移动到对应模型目录。

## 5. 按模型隔离输出

结果不会再写入共享的 `results/star_structure_test/`。脚本从 `MODEL` 自动取最后一级名称作为模型目录，因此不同模型不会覆盖彼此的 CSV、raw records、日志或图片。

例如：

```text
MODEL="Qwen/Qwen2.5-7B"
-> results/Qwen2.5-7B/star_structure_test/

MODEL="meta-llama/Meta-Llama-3.1-8B"
-> results/Meta-Llama-3.1-8B/star_structure_test/
```

完整结构为：

```text
results/
├── Qwen2.5-7B/
│   └── star_structure_test/
│       ├── run.log
│       ├── run_config.json
│       ├── summary_by_head.csv
│       ├── summary_overall.csv
│       ├── raw/
│       │   └── group_records_plot_tau.npz
│       └── figures/
│           ├── panel_a_star_prevalence.pdf
│           ├── panel_a_star_prevalence.png
│           ├── panel_b_pair_conflict_rate.pdf
│           ├── panel_b_pair_conflict_rate.png
│           ├── star_motivation_two_panel.pdf
│           ├── star_motivation_two_panel.png
│           └── figure_statistics.csv
└── Meta-Llama-3.1-8B/
    └── star_structure_test/
        └── ...
```

顶层只需要设置：

```bash
RESULTS_ROOT="/home/tlxr/SimCalKV/motivation_experiments/results"
EXPERIMENT_NAME="star_structure_test"
MODEL="..."
```

`OUT_DIR` 在自动区生成，不需要手工同步模型名。模型目录名仅允许 `A-Z/a-z/0-9/._-`，其他字符会自动替换成 `_`。

`figure_statistics.csv` 保存两张图每个 `(budget, tau)` 的精确百分比，以及对应的原始 group/pair counts，便于论文引用和审计。

## 6. IEEE 风格与灰度可辨识性

默认：

```bash
FONT_FAMILY="Times New Roman"
STRICT_TIMES_NEW_ROMAN=1
PNG_DPI=600

PANEL_WIDTH_IN=1.72
PANEL_HEIGHT_IN=1.58
COMBINED_WIDTH_IN=3.50
COMBINED_HEIGHT_IN=1.58
```

- PDF 使用 TrueType embedding (`pdf.fonttype=42`)；
- PNG 使用 600 DPI；
- 英文使用 Times New Roman；
- color-blind-friendly 配色；
- 每条曲线同时具有独立 line style 和 marker，灰度打印仍可辨识；
- 不使用 hatch；
- 不携带或分发字体文件；Times New Roman 缺失时默认直接报错。

## 7. 统计完整性

分析保存：

- `eligible_groups`；
- `unsafe_groups / unsafe_group_rate`；
- `candidate_pairs`；
- `conflict_pairs / conflict_pair_rate`；
- `assigned_candidates / candidate_acceptance_rate`；
- `PLOT_TAU` 下的 group-level raw records，用于后续诊断。

绘图会检查 `summary_overall.csv` 中：

\[
\mathrm{unsafe\_group\_rate}
=
\frac{\mathrm{unsafe\_groups}}{\mathrm{eligible\_groups}},
\]

以及

\[
\mathrm{conflict\_pair\_rate}
=
\frac{\mathrm{conflict\_pairs}}{\mathrm{candidate\_pairs}}.
\]

如果 pooled rate 与原始 counts 不一致会直接报错，不静默画图。
