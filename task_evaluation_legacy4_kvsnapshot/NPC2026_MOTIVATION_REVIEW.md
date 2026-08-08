# NPC 2026：动机审查与建议改写

## 结论

当前动机方向是成立的，但投稿版本必须从“KV 合并通常有价值”收缩为一个更尖锐、可被实验直接验证的问题：

> 现有并行星形匹配只验证 entry-to-anchor 局部关系；多个候选指向同一 anchor，或已经包含历史成员的压缩 entry 再次合并时，局部 gate 不等价于最终完整组紧凑。直接展开历史成员做 all-pairs 验证或迭代聚类又会破坏固定阶段并行执行。

`\method` 的核心贡献应明确为 **selector-agnostic certified grouping core**，不是默认 KeepKV-style selector，也不是 `+log C` 本身。

## 当前动机需要收紧的地方

### 1. 不要把“可并行”写成“必然更快”

当前实现确实由 batched GEMM、Boolean mask、argmin 和 segmented reductions 构成，但 PyTorch eager 版本存在多个 kernel 和中间张量。论文应写：

- fixed-stage；
- order-independent；
- GPU-parallelizable；
- avoids iterative clustering and historical-member expansion。

在得到融合 kernel 和硬件结果前，不应写“推理延迟优于所有方法”。系统实验以 compression-core latency、pair throughput、peak workspace 和 persistent memory 为主，端到端延迟仅放补充材料。

### 2. 明确两个不同失败模式

建议把问题分成：

1. **同轮星形不一致**：两个 candidate 分别靠近 anchor，但二者可能相距接近两倍 gate 半径。
2. **跨轮隐藏成员漂移**：一个 compressed entry 的 centroid 接近新 anchor，不表示其历史原始成员也接近新组成员。

这两个失败模式分别由 `D`、`R` 和 pair feasibility mask 解决。预实验必须分别展示。

### 3. 强调“先认证，再选最近可行 anchor”的必要性

绝对最近 anchor 可能因历史半径或内部直径不可行，而稍远 anchor 可行。若先选最近再认证，会无谓丢弃 candidate。代码包新增了 `nearest_anchor_then_certify` 消融，用于量化 recovered entry/mass 和质量差异。

### 4. 分离保证和任务质量

完整组直径保证只适用于 accepted groups。未通过认证的 evicted entries 会被删除。因此：

- group 更紧凑不保证 PPL 必然更低；
- 必须同时报告 accepted-entry ratio 和 accepted-count-mass ratio；
- `tau` 的效果是 compactness–coverage trade-off。

这一点主动写清楚，会降低审稿人以“严格 gate 导致信息丢失”攻击全文的空间。

### 5. 默认 selector 不是贡献

主结果应同时包含：

- 文档默认 `keepkv` selector：完整系统结果；
- `grouplogitkv` selector：与直接父方法共享 retained/evicted split 的受控结果。

若只报告默认 selector，审稿人无法判断增益来自 selector 还是 certificate grouping。

## 建议替换的动机文本

KV-cache eviction 能够精确控制物理缓存容量，但会永久删除被淘汰 token 的键值信息。并行分组合并通过将多个待淘汰 entries 独立路由到 retained anchors，并以 segmented reduction 更新每个 anchor，可在固定缓存预算下保留部分被淘汰信息。然而，现有星形匹配通常仅验证每个 compressed entry 与其 anchor 的局部关系。该局部条件既不能保证同一 anchor 下不同 candidates 的原始成员彼此接近，也不能保证多轮压缩后隐藏在历史 compressed entries 中的成员仍满足当前组约束。直接展开历史成员执行组内 all-pairs 验证，或使用迭代聚类修正 groups，会引入二次比较、顺序依赖和不规则控制流，不适合固定阶段 GPU 并行压缩。

`\method` 将 retained/evicted selector 与 grouping core 解耦，并为每个 compressed entry 维护 count、centroid-radius certificate 和 complete-diameter certificate。对全部 evicted–retained pairs，`\method` 通过 batched distance computation 构造 feasibility mask，并为每个 evicted entry 选择最近的可行 anchor；随后使用 segmented sum/max/top-2 独立归约各 groups。该过程无需存储历史成员、无需组内 candidate all-pairs 比较，也无需迭代重分配，同时保证每个新形成的 accepted group 满足明确的完整组直径约束。该约束进一步给出 merged-key logit deviation 和 count-calibration residual 的显式上界。

## 建议在引言中列出的可检验假设

- **H1**：普通星形 gate 在真实 LLM keys 上会形成大量完整直径超阈值 groups，且压缩率和轮数越高越严重。
- **H2**：`\method` 的 accepted groups 在有限精度容差内保持零 certificate violation。
- **H3**：nearest-feasible routing 比 nearest-then-certify 接受更多 candidate mass，且不放松完整组约束。
- **H4**：在共享 selector 和物理 KV budget 下，certificate grouping 能改善或稳定 PPL/LongBench，尤其在高压缩率和多轮压缩下。
- **H5**：固定阶段并行实现的主要开销可由 pair throughput 和 segmented-reduction profile 解释，且随 batch×KV-head 并行度提高具有可测扩展性。

## 可能被审稿人攻击的点

- `P[W,L]` selector history 显存较大：必须区分 grouping core 的 `C/R/D` 与默认 selector 状态。
- `Delta` 是 round-relative：必须报告 `S` 和 `Delta` 随轮次变化，不能跨轮直接比较 raw compactness。
- 只约束 key，不直接约束 value：论文应将理论结论限定为 attention-logit 与 count-calibration approximation。
- 严格认证可能降低覆盖率：报告 accepted count mass 和 rejected count mass。
- 当前未融合实现可能慢于 GroupLogitKV：不要用不可信端到端延迟掩盖；报告核心 kernel/profile，并将融合实现列为工程优化。
- CentroidKV 有 sink+local structural floor；比较时必须报告实际物理 KV 长度，而不是只报告目标 ratio。
- CaM 含随机 Bernoulli merge；主表至少 3 个 seeds，报告 mean±std。
