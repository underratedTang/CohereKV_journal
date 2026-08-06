# NPC 2026 实验计划

## 0. 投稿与定位

NPC 论文页数有限，建议把全文定位为：

> 一个面向 GPU 固定阶段执行的、selector-agnostic、跨轮可传播的 complete-group certificate KV compression core。

论文不要以“端到端推理最快”为中心。主要系统证据应是 compression-core latency、pair throughput、workspace、persistent side information 和 parallel scaling。

**紧急日期核对**：截至 2026-08-07，NPC 2026 官方主页显示 paper submission 为 2026-07-30 AoE，而官方 CFP 子页仍显示 TBD；二者与“2026-08-15 AoE”均不一致。若导师掌握延期通知，应立即确认提交系统仍开放，并保存延期邮件、通知或提交系统截图。

## 1. 实验问题

- **RQ1：必要性。** 局部 entry-to-anchor gate 在真实 keys 上是否经常违反完整组直径？
- **RQ2：正确性。** `\method` 的证书在真实 keys、多轮压缩和有限精度实现中是否有效？
- **RQ3：质量。** 在相同物理 KV budget 下，`\method` 的 PPL 和 LongBench 是否优于 CentroidKV、KeepKV、SimCalKV 和 CaM？
- **RQ4：归因。** 增益来自 selector、nearest-feasible routing、complete certificate、count calibration 中的哪一部分？
- **RQ5：并行性与成本。** certificate core 的 compression latency、pair throughput、workspace 和 side info 如何随长度、压缩率和并行度扩展？

---

# Part I：放在 `\section{Experiments}` 之前的预实验

预实验不是缩小版主表，而是构成动机证据链。建议放在 Introduction/Method 之间，标题可为 `Empirical Motivation`。

## P1. 局部星形 gate 不等于完整组一致性

### 协议

- 使用真实 LLaMA/Qwen 层内 RoPE 后 cached keys。
- 固定 GroupLogitKV selector，保证 retained/evicted split 完全相同。
- 比较：
  - ordinary star gate：nearest retained 且 `d_ij <= Delta`；
  - GroupLogitKV cosine grouping；
  - `\method` certified grouping。
- 模型层：first/middle/last，每层 4 个 KV heads。
- ratio：0.2/0.4/0.6；tau：0.6/0.7/0.8/0.9。

### 指标

- exact complete-group diameter / Delta；
- violating-group ratio；
- groups with at least two candidates 的 violation ratio；
- group size distribution；
- accepted-entry ratio。

### 预期图

- 横轴 compression ratio，纵轴 violation rate；普通 star/GroupLogitKV 高，`\method` 为零。
- CDF：diameter/Delta。

### 代码

`capture_npc2026_kv_snapshot.py` + `analyze_npc2026_grouping.py`。

## P2. 跨轮隐藏成员使问题更困难

在同一真实 key snapshot 上逐轮压缩到 retained ratio 0.8→0.6→0.4→0.3，显式追踪原始成员。

报告：

- ordinary star gate 的 exact diameter violation 随轮数变化；
- `D` 和 `R` certificate underestimate count；
- `\method` 的 exact diameter violation。

这项实验必须出现，因为只做第一轮时，审稿人可能认为简单半径 gate 已足够。

## P3. “先认证再选最近可行 anchor”不是实现细节

比较：

- `nearest_anchor_then_certify`；
- `nearest_feasible`（完整方法）。

报告：

- rescued candidate ratio；
- accepted count mass gain；
- PPL/LongBench差异；
- certificate violations，二者均应为零。

## P4. 紧凑度—覆盖率是非平凡权衡

扫描 tau，绘制：

- x：accepted count-mass ratio；
- y：PPL 或 LongBench；
- 点颜色：p95 diameter/Delta 或 tau。

这证明问题不是“阈值越严越好”，也解释为何需要 validation 选择 tau。

## P5. 并行化必要性

仅测 explicit compression core：

- mymethod route-only；
- mymethod full；
- no-certificate；
- GroupLogitKV；
- CentroidKV。

长度 512–8192，ratio 0.2/0.4/0.6，报告：

- median compression latency；
- million evicted–retained pairs/s；
- peak temporary memory；
- route/certificate/reduction 的 profiler 占比。

不要在预实验中以端到端生成延迟作为核心结论。

---

# Part II：`\section{Experiments}` 主实验

## 2.1 Setup

### 模型

最低配置：

- Llama-3.1-8B-Instruct；
- Qwen2.5-7B-Instruct。

资源不足时先用 3B 完成全部矩阵，再在 7B/8B 上运行主表关键点。所有方法必须使用相同模型、dtype、attention implementation 和 physical KV target。

### 方法

主表：

- Full KV；
- CentroidKV；
- KeepKV；
- SimCalKV；
- CaM；
- `\method`。

强烈建议额外保留 GroupLogitKV，至少在预实验、共享-selector消融和附录中。`\method` 声称改进其 grouping，完全不报告直接父方法会被审稿人质疑。

### 预算

- ratio：0.2/0.4/0.6；时间允许加入0.8 stress test。
- 必须记录每层实际 post-compression length。
- CentroidKV 若受 sink+local floor 限制无法达到目标，单独标记实际 retained ratio。

### 随机性

- CaM 至少 3 seeds，报告 mean±std。
- 其他方法至少固定一个公共 seed；关键主表可复跑 3 次确认稳定性。

## 2.2 PPL

使用现有 `evaluate_ppl_kvmerge.py`：

- WikiText-2；
- C4；
- sequence length 8192；
- chunk size 32；
- compress every chunk；
- ratio 0.2/0.4/0.6。

主表报告 PPL 和相对 Full KV 增幅。附图报告随 compression ratio 曲线。

注意：PPL 是多轮压缩，最能检验证书传播；需要同时报告 accepted count-mass 和 rejection mass，解释质量变化。

## 2.3 LongBench

使用当前 14 个英文任务。除总平均外，按类别报告：

- single/multi-document QA；
- retrieval/counting；
- summarization；
- few-shot/classification/dialogue。

建议主文给平均和4个类别，完整14任务放附录。ratio 0.2/0.4/0.6。

LongBench prompt 常在尾部放置问题，KeepKV-style recent selector 可能带来额外优势。因此必须补共享 GroupLogitKV selector 的 `\method` 结果，避免把 selector 增益误归因于 certificate。

## 2.4 Complete-group correctness

报告：

- exact violation count；
- certificate underestimate count；
- p50/p95/max diameter/Delta；
- accepted entry ratio；
- accepted count-mass ratio；
- nearest-feasible rescue ratio；
- `S` 和 `Delta` 随轮次/层变化。

这是全文最独特的实验，不应只放附录。

## 2.5 Parallel and memory efficiency

主文系统表建议只放可信边界：

- explicit compression latency；
- pair throughput；
- compressed KV MiB；
- side-info MiB；
- persistent KV+side-info MiB；
- compression peak extra MiB。

`evaluate_system_kvmerge.py` 已保留 E2E 字段，但新增上述 memory 字段。论文中可将 E2E 置于附录并明确不同方法 statistic collection 改变 prefill path。

扩展性实验：

- length：512/1k/2k/4k/8k；
- B×Hkv：1、2、4、8、16、32；
- distance tile：0/256/512；
- 报告 throughput 和峰值 workspace。

## 2.6 Ablations

主文优先顺序：

1. eviction only；
2. nearest merge without certificate；
3. nearest anchor then certify；
4. full certificate；
5. full without `log C`；
6. sink cannot absorb。

附录：

- selector keepkv vs grouplogitkv；
- W=32/64/128/256；
- beta=1.0/1.2/1.5/2.0；
- tau=0.6/0.7/0.8/0.9；
- distance chunk；
- BF16 selector history。

## 2.7 Fairness rules

- 同一 physical KV target，而不仅是相同 nominal ratio。
- 同一 prompt/tokenization/seed/dtype。
- baseline 使用官方默认参数；参数改变需标为 tuned/ablation。
- mymethod tau 用 validation subset选择，不能在最终测试集逐点挑最优。
- 完整方法比较和共享-selector机制比较分开报告。
- 质量表同时给 total persistent memory，否则默认 selector history 会造成不完整的内存比较。

---

# 3. 结果出来后的决策规则

- **mymethod group violation=0、PPL差**：检查 accepted count mass；若低，主要问题是 certificate过严或Delta尺度，不是实现错误。
- **certificate underestimate>0**：优先修复实现/有限精度 margin，暂停主实验。
- **nearest-feasible rescue接近0**：该设计点不值得占主文大篇幅，保留理论说明和附录。
- **shared selector下仍优于GroupLogitKV**：可将提升归因于 grouping/certificate。
- **只在KeepKV selector下提升**：主张应改为完整系统设计，不能声称 grouping单独优越。
- **full比no-certificate差且accepted mass低**：降低tau或重新审视Delta尺度；不要只展示有利ratio。
- **compression latency明显高**：强调保证/质量–成本权衡和并行结构；不要声称速度优势，展示fused-kernel潜力和profile。

# 4. 最小可投稿实验集

时间极紧时，最低必须完成：

1. 一张星形 violation + certificate zero-violation 图；
2. 一张 PPL 主表（2 datasets × 5 baselines × 3 ratios）；
3. 一张 LongBench 平均/类别主表；
4. 一张关键 ablation；
5. 一张 compression latency + persistent memory 表；
6. 两个模型中的至少一个完整矩阵，另一个模型给关键ratio复现。

缺少第1项会使理论贡献缺乏经验动机；缺少共享-selector消融会使贡献归因不成立。

## Time-limited execution decision

The immediate experimental protocol uses WikiText-2 only with two evaluation sequences. C4 is omitted because the first diagnostic already shows certificate coverage, rather than dataset breadth, is the blocking issue. Full-dataset confirmation should be run only after selecting a nondegenerate tau.
