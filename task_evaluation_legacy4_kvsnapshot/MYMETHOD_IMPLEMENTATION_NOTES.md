# mymethod 工程实现说明（KeepKV-style 默认 selector + GroupLogitKV 受控 selector）

实现依据：`method_motivation_method_theory_zh_keepkv_selector.md`。

## 实现文件

- `models/common/mymethod_common.py`：共享 selector、证书、路由和归约核心。
- `models/common/mymethod_score_capture.py`：LLaMA/Qwen 共享的最近窗口 base-score 捕获工具。
- `models/llama/merge_mymethod_llama.py`
- `models/llama/mymethod_runtime_patch_llama.py`
- `models/qwen/merge_mymethod_qwen.py`
- `models/qwen/mymethod_runtime_patch_qwen.py`

原 `GroupLogitKV`、`SimCalKV`、`KeepKV`、`CaM` 等对比方法实现文件未修改。

## 算法映射

1. 必要证书状态为 `(K,V,C,R,D)`；`C` 使用 INT32，`R,D` 使用 FP32。
2. 默认 selector 为 sink + latest-W recent + recent-score heavy hitters。
3. 默认 selector 保存最近 `W` 个 query 的 masked `exp(qK^T/sqrt(d))`，不使用 softmax、累计全历史 attention 或 `log(C)`；GQA 先对共享 KV head 的 query heads 求平均。
4. score history 使用固定长度 ring buffer；`beta>=1` 控制从最旧到最新 query row 的线性权重。合并后的 history 使用同一 membership mask 做 count-weighted reduction。
5. 典型尺度使用 count-weighted compressed-centroid RMS。
6. 有效半径使用 `Rhat=min(R,(C-1)/C*D)`。
7. 对 evicted-retained pairs 先执行完整可行性认证，再在可行 anchors 中选择最近者。
8. K/V 使用 count-weighted FP32 numerator；直径证书使用 segmented max/top-2，半径证书相对实际缓存 dtype 舍入后的 merged key 更新。
9. 解码时即时计算 `log(C)`，不保存额外 bias 状态。
10. 位置元数据继承 retained anchor，并单独保存下一逻辑位置。

## Selector 选项

### `--mymethod_selector keepkv`（默认）

使用最新设计文档中的 KeepKV-style selector。持久状态包含最近分数历史 `P[W,L]`。

### `--mymethod_selector grouplogitkv`

使用与当前 GroupLogitKV 实现完全一致的 retained/evicted 划分：

- 使用加入当前 `log(C)` 后的 post-softmax attention probability；
- GQA query heads 先按共享 KV head 求平均；
- 沿当前 query positions 求和；
- 全局 Bottom-K 低分 entries 为 evicted set；
- 不强制保留 sink 或 recent entries。

该模式只替换 selector，后续仍使用 mymethod 的欧氏完整组证书、最近可行 anchor、`C/R/D` 更新和 `log(C)` decoding。它不分配 score-history `P`，因此可用于：

1. 与 GroupLogitKV 做同 selector 的受控 grouping 对比；
2. 判断 PPL 差异主要来自 selector 还是 certificate grouping；
3. 降低默认 KeepKV-style selector 的持久显存和 history 更新开销。

`mymethod_selector=grouplogitkv` 是受控实验配置，不替代文档规定的默认 KeepKV-style selector。

## 性能改动

- 距离 GEMM 只沿 evicted 维按 `distance_chunk_size` 切块；retained anchors 一次参与 GEMM，减少大量小 kernel launch。
- feasibility 使用 squared-distance 等价判定；仅对最终选中的 accepted edge 开平方，不再对完整 evicted-retained matrix 执行 `sqrt`。
- K/V weighted numerator 合并为一次 segmented `scatter_add_`，降低 kernel launch 与 index 读取。
- KeepKV-style selector 的 sink budget 使用状态不变量直接确定，移除每层 `.item()/bool()` 导致的 CUDA host sync。
- recent-window 时间权重按 device/window/beta/ring-pointer 缓存并跨层复用。
- `grouplogitkv` selector 不保存或归约 `P`，score-history window 实际为 0。
- `distance_chunk_size` 越小，距离 workspace 越低但 GEMM 次数越多。
- 详细并行性审计与后续优化项见 `MYMETHOD_PARALLELISM_AND_OPTIMIZATION_REPORT.md`。

## 参数

- `mymethod_selector=keepkv | grouplogitkv`
- `tau=0.7`
- `sink_size=4`：仅 KeepKV-style selector 生效。
- `recent_window=128`：仅 KeepKV-style selector 生效，同时作为 token protection 和 score observation window。
- `recency_beta=1.0`：仅 KeepKV-style selector 生效。
- `distance_chunk_size=256`：evicted-entry GEMM/reduction chunk。
- `attention_chunk_size=128`
- `sink_can_absorb=1`：仅存在 sink anchors 时有意义。

## 评测入口

- PPL：`evaluate_ppl_kvmerge.py`、`run_ppl_kvmerge_sweep.sh`
- LongBench：`evaluate_longbench_kvmerge.py`、`run_longbench_kvmerge_sweep.sh`
- System：`evaluate_system_kvmerge.py`、`run_system_kvmerge_sweep.sh`
- mymethod Ablation：`evaluate_ablation_mymethod.py`、`run_ablation_mymethod_sweep.sh`
- GroupLogitKV Ablation：`evaluate_ablation_kvmerge.py`、`run_ablation_kvmerge_sweep.sh`

`evaluate_ablation_kvmerge.py` 已兼容 evaluator 新增的 `mymethod_config` 关键字，修复旧 wrapper 的 `unexpected keyword argument` 报错。

## 本地检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
find . -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

## NPC 2026 diagnostic additions

- `route_mode="nearest_feasible"` remains the main method.
- `route_mode="nearest_then_certify"` is exposed only through the
  `nearest_anchor_then_certify` ablation. It chooses the absolute nearest allowed
  anchor before applying the certificate and therefore may discard candidates
  that have a slightly farther feasible anchor.
- New diagnostic scripts do not alter the default method output:
  `capture_npc2026_kv_snapshot.py`, `analyze_npc2026_grouping.py`,
  `benchmark_npc2026_compression.py`, and `profile_npc2026_mymethod.py`.
