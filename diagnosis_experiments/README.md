# CohereKV Journal — Dual-Alpha & Importance Net-Gain MIS

本包基于 `diagnosis_experiments(9).zip` 修改。仅替换原 CohereKV 仓库中的 `diagnosis_experiments/`，不修改 `task_evaluation/` 或 `models/`。依然使用原包的 KeepKV selector、模型 runtime patch、连续多轮压缩和同位置 PPL 评分。仅支持原有 Llama/Qwen2 接入路径及 WikiText-2。未附带权重、数据集或完整 CohereKV 仓库。

## 1. 运行

将旧目录移走（不要只覆盖文件，否则旧 Soft MIS 文件仍会残留），解压本 ZIP 到原仓库根目录：

```bash
cd /root/CohereKV  # 改为自己的仓库根目录
mv diagnosis_experiments diagnosis_experiments_old
unzip /path/to/coherekv_journal_weighted_mis.zip
python diagnosis_experiments/test_journal.py
# 无需模型或 GPU 即可检查参数组合：
python diagnosis_experiments/sweep_config.py
# 编辑 run_journal.sh 中的 MODEL、GPU、序列数量、预算和 sweep 设置后：
bash diagnosis_experiments/run_journal.sh
```

`run_journal.sh` 自动从脚本所在位置定位仓库；默认沿用原实际 shell 的 `meta-llama/Llama-3.2-3B`、CUDA 0、bf16、eager、U=256、S=32、24 轮压缩、10% budget、序列索引 0–7、训练集 4 个 4096-token 校准窗口、seed=42。先用 validation 选参，最终改成 test，并将四个数值参数数组及 Priority 数组分别设为单值。不建议根据 test 结果再选参。

Budget 定义沿用原包：每次压缩后保留 `max(1, round((U-1)*budget))` 个物理槽位/Head，后续各轮固定该容量，不是随累计逻辑长度按比例增大。Sink/recent 保护沿用原 KeepKV selector 规则。

建议初次 GPU smoke test：`PPL_SEQUENCE_INDICES=("0")`、`PPL_CYCLES=4`、四个数值参数数组及 Priority 数组均设为单值；验证完整流程后再恢复默认设置。运行时需要原仓库兼容的 Python/PyTorch/Transformers 环境；requirements.txt 沿用原包，不代表最新 Transformers 一定兼容原 runtime patch。

## 2. 仅四种方法

| Method | 定义 |
| --- | --- |
| full_cache | 原生未 patch Full Cache，使用相同输入和评分位置 |
| eviction_only | 同一 KeepKV 选择器、预算、保护规则；无 merge |
| merge_all | **沿用原包定义**：最近 physical K + E–A 当前距离阈值，满足阈值即合并；无 E–E MIS；原 FP32 CountMean |
| coherekv_journal | E–A 路由 + E–E Hard MIS + CountMean + 完整 R/D 证书更新 + log Count |

已移除旧的独立 conference method、Soft MIS、p-norm、log Contribution、相应旧测试及多余 V/Q/sink 校准诊断。会议版仍能通过 journal 的参数点复现原会议版重实现：`alpha_ea=alpha_ee=1, lambda_ea=lambda_ee, reference_mode=online_first, priority_mode=degree`。若用 frozen_train，则只改变参考尺度来源，不能称为完整会议版复现。

## 3. 期刊版定义

对每层 l、KV Head h，使用共享的参考尺度 S：

```
delta_EA = lambda_EA * S[l,h]
delta_EE = lambda_EE * S[l,h]
```

统一距离（EA 使用 alpha_ea，EE 使用 alpha_ee）：

```
Gamma_uv(a) = max(a*D_u, a*D_v, ||mu_u-mu_v|| + a*(R_u+R_v))
```

a=0 为当前质心距离，a=1 为完整会议版历史距离，中间值控制历史几何信息参与程度。两种关系使用同一套 R/D，不再有 mode 开关或共享传播衰减系数。mu 为未舍入的 count-weighted 历史几何质心，也是本期刊方法的当前距离定义；旧版 current 使用实际缓存的低精度 K，两者可能有 dtype 舍入差异。merge_all 的物理 K 距离基线保持不变。

E–A 先过滤兼容的 Anchor，再选择最近的兼容 Anchor（距离相等时取最小物理 slot）。E–E 构造冲突图，使用以下静态净收益优先级执行并行 MIS。合并权重只能是 0/1，所有合并采用 CountMean 和 log Count。

成功合并的组 G 按会议版完整传播（不衰减）：

```
mu' = sum(C_u * mu_u) / sum(C_u)
D'  = max(max(D_u), max_{u<v}(R_u + ||mu_u-mu_v|| + R_v))
R'  = min((C'-1)/C' * D', max_u(R_u + ||mu_u-mu'||))
```

未合并槽保持不变。初始化 R=D=0，故第一轮距离与 alpha 无关，第二轮开始可能产生差异。alpha 仅在距离判断时使用，不进行递归半径衰减。若导师要求存储的 R 每轮递归衰减，则这是不同机制，本版本未实现。

**理论边界：** 完整传播保持 R/D 是原始 token 几何上界（忽略数值舍入），即使放宽筛选也不截断 D。只有 alpha_ea=alpha_ee=1 保留会议版固定阈值直径保证（双阈值时为 max(delta_EA,delta_EE)，允许数值容差）。其他组合可能超出此阈值。`history_certified` 表示维护完整几何证书，`threshold_guaranteed` 表示是否具有原阈值保证；`geometry_over_delta` 统计证书 D 超阈值的槽数，不等于真实原始 token 直径超阈值的数量。

### 3.1 Importance Net-Gain Priority

每个候选节点权重为 W_i，静态冲突图邻域为 N(i)：

```
Gain_i = W_i
Loss_i = sum(W_u for u in N(i))
Priority_i = Gain_i - Loss_i
```

按净收益降序，EA 距离、hash、physical index 升序打破平局。净收益只在初始冲突图计算一次；即使为负也继续选择局部最大者，直到活跃集为空，保证 maximal independent set，但不保证最大权独立集最优。四种权重定义：

| priority_mode | W_i |
| --- | --- |
| degree | 1 |
| count | C_i |
| importance | I_i |
| count_importance | C_i * I_i |

I_i 为现有 `recent_window_importance` 的输出，**直接复用选择器分数，不修改 KeepKV 选择**。按照公开 runtime patch 中未加 log Count 的 base score 语义，C_i * I_i 是代表状态的未归一化 attention mass 代理，而非真实历史 attention 之和。服务器若修改了 score capture 语义，应重新核对是否重复计入 Count；本包不含外部 runtime。`degree` 使用整数 1-degree，严格保持旧优先级排序；保留同样 EA/EE 参数即可对照旧 Dual-Alpha 版本。其他模式仅改变 MIS priority，不改变路由、冲突判定、CountMean 或证书传播。没有新连续超参数。

## 4. Offline Head-wise Calibration

`calibrate_median_reference.py` 保留原 K 尺度定义并删除不参与本方法的 Q、V、sink 校准。对每个训练窗口，用真实模型缓存 Key 计算：

```
S_m[l,h] = sqrt(2 * sum_i ||K_i - mean(K)||² / (N-1) + 1e-24)
S_cal[l,h] = median_m(S_m[l,h])
```

默认使用独立 train split、随机选择不重叠的完整 token block，记录实际起点、每个窗口的尺度矩阵、聚合矩阵、模型、dtype 等。新采样器按固定长度 block 无放回采样（不承诺与原外部 sampler 的起点完全相同）。历史 JSON 若具有原 schema 与正确 K 字段，仍可复用：放到本实验输出的 calibration/median_reference.json，并将 `RUN_CALIBRATION=0`。优先使用同 tokenizer、dtype、attention backend 的校准文件。Evaluation 不需要旧 V/Q/sink 字段，也不再依赖 `motivation_experiments/calibrate_layer_head_scales.py`。

`PPL_REFERENCE_MODES=("frozen_train" "online_first")` 可在相同数据和其他参数下比较两个尺度来源。`online_first` 继续使用原仓库的 `reference_scale_from_original_keys` 在第一次压缩前计算并冻结。full_cache/eviction_only 不依赖校准文件。暂不引入 relation-specific 尺度矩阵（只解耦 lambda），避免同时改变太多因素。

## 5. Sweep：统一数组笛卡尔积

只编辑 `run_journal.sh` 中以下五个数组。单值固定，多值扫描。默认是独立的 Priority 消融：

```bash
PPL_ALPHA_EA=("0")
PPL_ALPHA_EE=("1.0")
PPL_LAMBDA_EA=("0.50")
PPL_LAMBDA_EE=("0.50")
PPL_PRIORITY_MODES=("degree" "count" "importance" "count_importance")
```

默认每序列 4 组 journal + 3 组基线，共 7 次；8 序列共 56 次，不含校准。固定几何配置是用于观察有冲突时的 Priority 效应的候选设置，不是已验证最优。建议先检查 conflict edges 和 rejected slots 是否足够；若几乎无冲突，所有 Priority 可能得到相同结果。可将 `PPL_LAMBDA_EE=("0.50" "1.0" "1.5")` 扩展到不同冲突强度（12 组 journal + 3 基线）。随后固定较优的 Priority，再扫描 alpha/lambda。基线不会随 Priority、alpha 或 EE 阈值重复；merge_all 仍随不同 EA 阈值和 reference/budget 运行。重复数组元素自动去重。

`python diagnosis_experiments/sweep_config.py` 预览默认组合；CLI 不自动读取 shell，可传入 `--alpha_ea 0 --alpha_ee 1 --lambda_ee 0.5,1,1.5 --priority_modes degree,count_importance`。Shell 保存 `sweep_plan.json`。几何全网格会再乘 Priority 数量，建议分阶段。

会议版重实现：四个数值参数分别为 1、1、0.5、0.5，`PPL_PRIORITY_MODES=("degree")` 且 `PPL_REFERENCE_MODES=("online_first")`。增加预算用 `PPL_BUDGETS`。不同实验请修改 `EXPERIMENT_NAME`，避免同名重跑清除 sequences/aggregate。设置 `RUN_CALIBRATION=0` 可复用当前输出目录中已验证的校准。旧版本 CSV 配置字段不同，不应混合汇总。

## 6. 输出与验证

结果路径：`diagnosis_experiments/results/<model>/journal_weighted_mis_v4_<split>/`。

- `sweep_plan.json`：正式运行的实际配置计划。
- `calibration/median_reference.json` + `median_reference_by_head.csv`：独立冻结校准及逐 Head 尺度。
- `sequences/seq_<idx>/ppl_continuous/continuous_config.json`：实际配置、参考文件哈希和输入窗口哈希；完整结束才写 `continuous_run_complete.json`。
- `ppl_continuous.csv`、`ppl_continuous_cycles.csv`、`ppl_continuous_head_rounds.csv`：逐配置、逐轮、逐 Layer/Head 记录。head 文件记录完整半径、直径证书、候选分配、MIS conflict edges、合并率、代表 Count 等。
- `aggregate/ppl_sweep_unique.csv`：每个真实配置的 pooled PPL（按总 NLL/总 scored tokens 计算），另报逐序列 PPL arithmetic mean、sample SD、合并率、分配率、MIS 拒绝率、represented mass merge rate、相同窗口下对 baseline 的 delta NLL；新增同几何参数的 `delta_nll_vs_degree`，负值表示比 degree 更好。
- `aggregate/ppl_sweep_best.csv`：按 method/budget/reference 排名第一的实际配置（验证集选参用，不是独立 test 结论）。
- `aggregate/ppl_continuous_cycles.csv`、`ppl_continuous_per_sequence.csv`、`ppl_continuous_paired_deltas.csv`：分轮/分序列/配对分析，逐轮筛选统计是累计值。
- `aggregate/figures/`：不同固定设置下的 双 alpha 热图及曲线、双阈值热图及曲线、固定几何参数的 Priority PPL 和两种重要性保留率柱状图（PNG/PDF）；缺少可比较组合时跳过该图。默认 DejaVu Sans，若服务器已安装 Times New Roman 可在 shell 修改。
- `coherekv_journal_results.zip`：结果归档，包含数值、校准和日志。

保留连续评估协议：U-1 个 unscored 输入，压缩，1 个 post-compression warm-up，再对 S 个目标计分；每个序列每个配置单独缓存，同序列内不重置或重放 token。预检选择完整不重叠窗口。PPL 是指定评分目标的 PPL，不是所有输入 token 的 PPL。汇总校验完整配置、窗口、token 数、物理长度、逐轮 NLL 与压缩次数。

### Importance 指标

- `routed_importance_mass` / `contributed_importance_mass`：候选 / 最终合并的 sum(C_i * I_i)，与 v3 字段定义一致。
- `routed_raw_importance` / `contributed_raw_importance`：候选 / 最终合并的 sum(I_i)。
- `importance_retention_rate` 和 `raw_importance_retention_rate`：分别按上述两种权重，使用最终合并权重和 / 候选权重和。分母为 0 时留空。它们是代理重要性保留率，不是实际注意力保真度；不同策略会改变后续输入状态，跨轮累计比率也不是独立同分布观测。
- `priority_mode` 进入完整配置标识、逐轮/逐 Head CSV 和汇总，避免混合不同 Priority。

### 测试与限制

已在 CPU 执行 `test_journal.py` 九项测试，包含 4 轮 cache class（外部 runtime 用 stub）、距离权重端点及中间值、放宽门控后完整证书保持、Star Conflict/CountMean、校准 RMS、配置生成、跨序列汇总、四种 Priority、无边图、单位权重退化与 I/C*I 区分。Python 源通过 compileall，Shell 通过 bash -n。

未在真实 GPU 模型上运行：当前 ZIP 不含 `task_evaluation/`、`models/` 和权重。未生成真实 PPL、真实 Head 校准值、真实组直径或 attention-logit/output error；这些需在原仓库运行。R/D 是上界而非真实几何误差。原 merge_all 保持 physical K/FP32 路由与聚合，而 journal 使用高精度历史质心/FP64 聚合；本轮为隔离 Priority 未修复该差异，不能假定无冲突 Journal 与 merge_all 数值完全相同。优先使用固定几何参数的 degree 对照判断 Priority 效果。
