# mymethod 并行性与工程优化检查

依据：`method_motivation_method_theory_zh_keepkv_selector.md`。

## 结论

当前实现已经具备算法层面的批量并行，但尚未达到单一融合 CUDA/Triton kernel 的充分工程并行：

- 已并行：batch 与 KV-head 展平后统一执行；evicted-retained 距离使用 batched GEMM；可行性条件逐 pair 向量化；K/V/count/history 和证书使用 `scatter_add_` / `scatter_reduce_` segmented reductions。
- 未完全融合：距离、可行性判断、argmin、top-2、半径更新仍由多个 PyTorch eager kernels 和中间张量组成；当每轮 evicted 数量较小时，kernel launch 和 HBM 往返会占较大比例。

## 本次已经落地、且不改变算法的优化

### 1. 取消完整 pair matrix 上的平方根

原实现对每个 evicted-retained pair 计算 `sqrt(dist2)`，再判断：

```text
Rhat_i + d_ij <= T_j
```

新实现使用严格等价条件：

```text
margin = T_j - Rhat_i
margin >= 0 and dist2 <= margin * margin
```

最近可行 anchor 直接按 `dist2` 选择，仅对最终被选中的 edge 计算一次平方根。平方根工作量从 `O(Ne*Nr)` 降为 `O(Ne)`，routing、tie-breaking 和证书语义不变。

### 2. K/V segmented sum 合并为一次 scatter

原实现分别对 `C_i K_i` 和 `C_i V_i` 执行两次 chunked `scatter_add_`。新实现沿 feature 维拼接 K/V numerator，一次 segmented scatter 后再拆分，减少 kernel launch 和 index tensor 重复读取。

### 3. 距离 GEMM 只沿 evicted 维分块

retained anchors 在每个 evicted tile 中一次参与 batched GEMM，不再在 retained 维生成大量小 GEMM。`distance_chunk_size<=0` 表示一次处理全部 evicted entries。

### 4. 证书中心绑定实际缓存 dtype

FP32 centroid numerator 转回 FP16/BF16 后，半径证书相对实际缓存中的 rounded key 更新，避免为了性能优化而破坏证书对应对象。

### 5. 移除 selector 的每层 GPU→CPU 同步

原 KeepKV-style selector 通过 `Tensor.item()` / Python `bool()`读取每个 layer 的 sink 数量，会强制 CUDA stream 同步。根据 sink 必须保留、位置参数不在序列中途改变的不变量，生产 CUDA 路径直接由整数配置得到 sink count；CPU/单元测试路径仍保留任意位置元数据的防御性校验。

### 6. 复用 recent-window 时间权重

相同设备、`W`、`beta` 和 ring pointer 的物理权重向量在各层间缓存复用，避免每层重复创建 `linspace`、ring index 和 scatter 临时张量。

## 仍可继续优化，但本包未引入的项目

### 优先级 1：融合 distance + feasibility + running argmin

实现一个 Triton/CUDA kernel，按 retained tile 流式执行：

```text
dist2 GEMM/点积 -> feasibility -> running min(dist2,index)
```

避免同时物化 `dist2`、`margin`、`feasible` 和 `masked_dist2`。这是预计收益最大的后续优化。

### 优先级 2：融合证书 segmented reductions

当前 complete-diameter 与 radius 更新需要多次 `scatter_reduce_`：

- candidate diameter max；
- candidate extent top-1；
- top-1 winner；
- candidate extent top-2；
- candidate-to-new-centroid radius max。

可用自定义 segmented kernel 在一次或两次 pass 中完成 max/top-2/radius，减少 launch 和中间张量。

### 优先级 3：KeepKV-style selector history 的存储和带宽

默认 selector 的 `P[B,Hkv,W,L]` 是主要 side-information 开销。可评估：

- BF16 存储、FP32 accumulation；
- ring buffer 的 fused append/reduction；
- 只在压缩轮次生成 weighted importance；
- 将 score capture 与 attention logits 复用到同一个 kernel。

这些需要单独做数值与 PPL 验证；本次没有改变默认 FP32 history 语义。

### 优先级 4：编译与图捕获

固定 shape 的 decode/压缩区间可评估 `torch.compile` 或 CUDA Graph，但动态 cache 长度、TopK 和变长 segmented reductions 会限制收益，需要按 workload 专门处理。

## 验证

- 新旧 mymethod 在 20 组随机 CPU case 中 routing、K/V、count、R/D 和 history 输出一致（浮点容差 `rtol=1e-5, atol=1e-6`）。
- 单元测试覆盖 tiled/untiled routing 一致性、最近不可行 anchor 回退、完整直径不变量和跨轮次证书传播。
- 当前环境无 CUDA、Transformers、模型权重和评测数据，因此未声称实际 GPU 加速比例。
