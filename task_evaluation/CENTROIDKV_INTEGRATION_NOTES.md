# CentroidKV 接入说明

## 来源

- 官方仓库：`https://github.com/jessica-hujie/CentroidKV`
- 用户提供的官方文件原样保存在：`third_party/CentroidKV/centroidkv_press.py`
- 许可证标识保留为 Apache-2.0。

## 方法映射

接入实现保持官方核心流程：

1. 保护最前面的 `sink_budget` tokens；
2. 保护最后面的 `local_budget` recent tokens；
3. 仅压缩中间区域；
4. 中间区域按 `chunk_size` 划分，偶数位置作为 source，奇数位置作为 destination；
5. 每个 source 在 chunk 内选择 cosine similarity 最大的 destination；
6. 全局选择 strongest edges；
7. 使用 count-weighted arithmetic centroid 合并 K/V；
8. 按 `merge_ratio - step * decrease_rate` 迭代，最多 10 轮。

## 工程适配

- `models/common/centroidkv_common.py`：无 `kvpress` 依赖的共享核心。
- `models/llama/merge_centroidkv_llama.py`
- `models/qwen/merge_centroidkv_qwen.py`
- PPL 重复压缩中跨轮次保存 INT32 count，以维持 centroid 权重；不向 attention logits 加入 `log(C)`。
- K/V centroid numerator 使用 FP32 accumulation，再转换回 cache dtype。
- `sink_budget + local_budget` 是结构性最小物理长度；不可能达到更小目标时直接将目标钳制到该下限，避免无效迭代。

## 评测入口

- PPL：`evaluate_ppl_kvmerge.py`、`run_ppl_kvmerge_sweep.sh`
- LongBench：`evaluate_longbench_kvmerge.py`、`run_longbench_kvmerge_sweep.sh`
- System：`evaluate_system_kvmerge.py`、`run_system_kvmerge_sweep.sh`

统一参数：

```text
--centroidkv_sink_budget 16
--centroidkv_local_budget 64
--centroidkv_chunk_size 256
--centroidkv_merge_ratio 0.8
--centroidkv_decrease_rate 0.2
```

## 数值一致性

随机 FP32 reference case 中，适配实现与保留的官方 `CentroidKVPress.compress()` 的 K/V 输出逐元素完全一致；该检查已加入 `tests/test_centroidkv.py`。

## Runtime 解耦（2026-08-08）

CentroidKV 不使用 attention weights 作为压缩输入，因此评测时不安装任何
GroupLogitKV 或通用 score-capture attention patch。`setup_kv_imports()` 在
`kv_method=CentroidKV` 时只延迟导入 `merge_centroidkv_{arch}`。

长上下文评测若 CLI 仍传入历史默认 `--attn_implementation eager`，CentroidKV
会自动改用 `sdpa`。这是等价的模型 attention backend 选择，不改变 CentroidKV
的 sink/local/chunk/merge 参数；目的只是避免在压缩开始之前物化完整
`[B,H,Q,K]` attention-score tensor。

GroupLogitKV 仍完整保留并仅在 `kv_method=GroupLogitKV` 时使用其自己的 runtime
patch。SimCalKV/CaM 需要 attention scores，但改用中性的
`score_capture_runtime_patch_{llama,qwen}.py`，不依赖 GroupLogitKV count bias、
CUDA calibrated-QK kernel 或任何 GroupLogitKV merge state。
