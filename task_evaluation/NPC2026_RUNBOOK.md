# NPC 2026 运行与反馈说明

## 1. 先运行预实验

编辑：

```bash
run_npc2026_preexperiments.sh
```

至少修改：

- `MODEL_NAME`；
- `FAMILY`；
- `DEVICE_MAP`；
- 输出路径。

运行：

```bash
bash run_npc2026_preexperiments.sh
```

预实验默认产生：

- `kv_snapshot.pt`；
- `grouping_diagnostics.json/.csv`；
- `compression_microbenchmark.json/.csv`；
- PPL component、tau 和 selector 对照 JSONL；
- `*_diagnostics.jsonl` 中的 accepted/rejected count-mass；
- `npc2026_diagnostic_summary.json/.md` 自动诊断摘要；
- 可选 Chrome profiler trace。

## 2. 结果正常后运行主实验

编辑：

```bash
run_npc2026_main_experiments.sh
```

运行：

```bash
bash run_npc2026_main_experiments.sh
```

建议先把 `RUN_LONGBENCH=0`、`RUN_SYSTEM=0`，确认 PPL 路径正常；随后分别开启。

## 3. 需要反馈的结果

优先反馈以下文件：

1. `grouping_diagnostics.json`；
2. `compression_microbenchmark.json`；
3. `mymethod_profile_table.txt`，若运行 profiler；
4. 所有 `ppl_ablation_*.jsonl`；
5. 所有 `ppl_ablation_*_diagnostics.jsonl`，重点查看 accepted/rejected count mass；
6. 主实验 PPL JSONL；
7. LongBench 每个方法目录中的 `result.json` 或预测 JSONL；
8. system JSON，尤其：
   - `compression_time_ms_mean`；
   - `compressed_kv_mib_mean`；
   - `method_side_info_mib_mean`；
   - `persistent_kv_plus_side_info_mib_mean`；
   - `compression_peak_extra_mib_mean`；
9. 任意 OOM、non-finite、certificate violation 日志。

打包：

```bash
bash package_npc2026_feedback.sh npc2026_pre_results
bash package_npc2026_feedback.sh npc2026_main_results
```

## 4. 首轮结果需要回答的十个问题

1. ordinary star gate 的 violation rate 是否随ratio/round增加？
2. mymethod exact/certificate violation 是否严格为0？
3. nearest-feasible rescue ratio是多少？
4. full、no-certificate、nearest-then-certify的PPL排序是什么？
5. accepted entry ratio和accepted count mass是否过低？
6. KeepKV selector与GroupLogitKV selector的PPL差多少？
7. mymethod相对CentroidKV/KeepKV/SimCalKV/CaM在哪些ratio领先？
8. 优势集中在PPL、LongBench某类别，还是高压缩率？
9. compression profile中route、scatter、top-2、radius各占多少？
10. side info后，mymethod总persistent memory是否仍有优势？

## 5. 单独命令

### Capture真实KV snapshot

```bash
python capture_npc2026_kv_snapshot.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --family llama \
  --dataset wikitext2 \
  --model_seqlen 2048 \
  --layers first,mid,last \
  --output results/kv_snapshot.pt
```

### 分组动机诊断

```bash
python analyze_npc2026_grouping.py \
  --snapshot results/kv_snapshot.pt \
  --compress_ratios 0.2,0.4,0.6 \
  --taus 0.6,0.7,0.8,0.9 \
  --output_json results/grouping_diagnostics.json
```

### Compression core microbenchmark

```bash
python benchmark_npc2026_compression.py \
  --device cuda \
  --lengths 512,1024,2048,4096,8192 \
  --compress_ratios 0.2,0.4,0.6 \
  --distance_chunk_sizes 0,256,512 \
  --output_json results/compression_microbenchmark.json
```

### Profiler

```bash
python profile_npc2026_mymethod.py \
  --device cuda \
  --length 4096 \
  --compress_ratio 0.4 \
  --trace_json results/mymethod_trace.json \
  --table_txt results/mymethod_profile_table.txt
```

## 6. 自动汇总已有预实验结果

```bash
python summarize_npc2026_diagnostics.py \
  --results_dir npc2026_pre_results \
  --output_json npc2026_pre_results/npc2026_diagnostic_summary.json \
  --output_md npc2026_pre_results/npc2026_diagnostic_summary.md
```

摘要只标记证书违反、低 accepted count-mass 等风险，不会自动挑选或隐藏不利结果。

## Time-limited protocol override

For the current submission schedule, all PPL scripts in this package default to WikiText-2 only and `max_eval_sequences=2`. C4 is deferred. Use `run_npc2026_fast_followup.sh` after the first diagnostic package; it scans the lower, empirically relevant tau range and adds direct method baselines.
