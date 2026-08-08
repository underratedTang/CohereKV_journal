# `\method` 并行实现审计

## 已充分张量化的部分

当前 `models/common/mymethod_common.py` 没有按 candidate、anchor 或 group 执行 Python 级逐元素循环。主要并行路径为：

1. `select_keep_indices`：按 batch×KV-head 向量化 `topk`。
2. `d^2`：`torch.bmm` 同时计算一个 evicted tile 对全部 retained anchors 的距离。
3. `F_ij`：broadcast Boolean operations 同时计算整个 `[B*Hkv, Ne_tile, Nr]` mask。
4. nearest feasible：沿 retained 维执行并行 `min` reduction。
5. K/V/count/history：`scatter_add_` segmented reductions。
6. `max D_i`、candidate radius max：`scatter_reduce_(amax)`。
7. extent top-2：并行 segmented max/min/max。

因此算法级并行已经实现。`MYMETHOD_DISTANCE_CHUNK_SIZE=0` 时，所有 evicted entries 一次参与上述过程；正数时只有 tile 之间由 Python 顺序调度，tile 内仍全部并行。

## 当前主要工程瓶颈

### 1. 路由仍是多 kernel eager 流水线

当前依次物化或写入：

- centered keys；
- squared norms；
- `dist2`；
- `margin`；
- `feasible`；
- masked distance；
- argmin outputs。

虽然已经避免对完整 pair matrix 开平方，但 HBM 读写和 kernel launch 仍然明显。最优实现应使用 Triton/CUDA tile 融合：

`distance GEMM tile -> feasibility -> running best distance/index`。

### 2. segmented top-2 需要三次 reduction

当前第一大值、winner tie-break 和第二大值分别执行 reduction。可在一个自定义 segmented kernel 中同时维护 `(max1, idx1, max2)`。

### 3. radius 和 diameter 证书分开归约

candidate radius、candidate diameter max、extent top-2 使用多个 gather/norm/scatter。可融合为一个 certificate-reduction kernel，同时输出：

- max candidate diameter；
- max/second-max extent；
- max candidate-to-new-centroid radius。

### 4. K/V reduction 有额外 concatenate 临时量

当前把 K/V 拼接后一次 scatter，减少了 launch 数，但产生 `2d` 临时向量。自定义 kernel 可直接维护两个 accumulators，避免 `torch.cat`。

### 5. 默认 KeepKV-style selector 的 history 成本可能主导

`P` 的 persistent memory 和 weighted importance 工作量为 `O(B Hkv W L)`。这不属于 certificate core，但会影响完整方法速度。需要同时报告：

- `selector=grouplogitkv`：无 `P`，用于 grouping core 受控比较；
- `selector=keepkv`：完整默认方法。

### 6. attention score capture 可能迫使 eager attention

PPL/LongBench 需要 selector statistics，可能无法直接使用 FlashAttention。系统结论必须把 attention statistic collection 与 explicit compression 分开。现有 system evaluator 已新增：

- explicit compression latency；
- compressed KV memory；
- method side-info memory；
- persistent KV+side-info memory；
- compression peak extra memory。

## 不改变方法的优化优先级

1. Triton fused routing kernel。
2. Fused certificate segmented kernel。
3. CUDA Graph / `torch.compile` 固定 shape 压缩路径。
4. BF16 保存 `P`、FP32 accumulation 的独立工程消融。
5. retained-dimension tiling和running argmin，用于超长上下文降低 workspace。
6. 根据 `(Ne,Nr,d)` autotune evicted tile，而不是固定 256。
7. 将 `C/R/D/position` 打包连续存储，降低小 tensor metadata 和 allocator 开销。

## 不能作为“纯工程优化”混入主方法的变化

以下会改变算法或比较协议，应作为消融而非优化：

- 改变压缩频率；
- 放松 `tau`；
- 跳过直径条件；
- 仅检查绝对最近 anchor；
- 将 rejected entries 强制合并；
- 修改 selector 或 recent budget；
- 用 cosine 替换 Euclidean certificate metric。
