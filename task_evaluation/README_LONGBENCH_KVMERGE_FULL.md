# LongBench v1 + KV merge/compression evaluation

This package keeps the `task_evaluation/` layout and places a full LongBench source tree under it, parallel to `llama/` and `qwen/`:

```text
task_evaluation/
├── LongBench/                         # preserved LongBench source tree
│   ├── eval.py
│   ├── metrics.py
│   ├── config/
│   ├── docs / misc / retrieval / summ / refs ...
│   ├── custom_cache*.py               # kept for completeness; not used by kvmerge runner
│   └── llama_flash_attn_monkey_patch.py# kept for completeness; not used by kvmerge runner
├── llama/                             # GroupLogitKV / SimCalKV / KeepKV / CaM for LLaMA
├── qwen/                              # GroupLogitKV / SimCalKV / KeepKV / CaM for Qwen
├── evaluate_longbench_kvmerge.py      # new LongBench runner for the four merge methods
└── run_eval_longbench_kvmerge_all.sh   # convenience script
```

## What is preserved

The LongBench directory is kept as a full source tree rather than the previous minimal subset. Files such as `docs/`, `misc/`, `retrieval/`, `summ/`, `refs/`, `custom_cache*.py`, and `llama_flash_attn_monkey_patch.py` are retained for traceability and compatibility with the original LongBench/Crosslayer LongBench scripts.

The new runner does not import the original `custom_cache*.py` or `llama_flash_attn_monkey_patch.py`. It evaluates only these four post-hoc KV merge/compression methods from `task_evaluation/llama` and `task_evaluation/qwen`:

- GroupLogitKV
- SimCalKV
- KeepKV
- CaM

## Run all LongBench v1 English tasks

Run from the package root:

```bash
cd task_evaluation

export LLAMA_MODEL=llama-3.2-1B-Instruct
export QWEN_MODEL=Qwen2.5-7B-Instruct
export COMPRESS_RATIO=0.4

bash run_eval_longbench_kvmerge_all.sh
```

The model keys must exist in:

```text
LongBench/config/model2path.json
LongBench/config/model2maxlen.json
```

or you can run `evaluate_longbench_kvmerge.py` manually with `--model_path /path/to/model`.

## Run one method

```bash
python -u evaluate_longbench_kvmerge.py \
  --arch llama \
  --model llama-3.2-1B-Instruct \
  --kv_method GroupLogitKV \
  --compress_ratio 0.4 \
  --all \
  --dtype fp16 \
  --device cuda \
  --out_path ./eval_longbench_kvmerge_out/llama/llama-3.2-1B-Instruct/GroupLogitKV/cr_0.4/pred
```

## Dataset set

`--all` runs the LongBench v1 English tasks used by this runner:

```text
narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,
2wikimqa,musique,gov_report,qmsum,multi_news,trec,
triviaqa,samsum,passage_count,hotpotqa
```

## Output

```text
eval_longbench_kvmerge_out/<arch>/<model>/<method>/cr_<ratio>/pred/<dataset>.jsonl
eval_longbench_kvmerge_out/<arch>/<model>/<method>/cr_<ratio>/result.json
```

## Notes

- `compress_ratio=0.4` means evict about 40% KV entries and keep about 60% physical KV slots.
- The runner defaults to `--attn_implementation eager`, because GroupLogitKV and KeepKV may need attention logits/weights and runtime attention patching.
- The preserved LongBench `custom_cache*.py` files are for the original Crosslayer/AquaKV/KIVI/XQuant/BaseOnly/BaseBlockMix-style runners, not for the new four-method kvmerge runner.
- The optional fused `grouplogitkv_calibrated_qk` CUDA kernel is not included in `task_evaluation.zip`; the runtime patch falls back to a PyTorch path if that kernel is unavailable.

## Updated LongBench sweep script

`run_longbench_kvmerge_sweep.sh` is the recommended LongBench v1 runner. It follows the command/input style of `run_ppl_kvmerge_sweep.sh`:

- no conda activation inside the script;
- manual configuration block at the top;
- sequential sweep over methods, compression ratios, and similarity thresholds;
- outputs logs, predictions, and `longbench_summary.jsonl` under one `OUT_DIR`.

Dataset selection is intentionally simple:

```bash
DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"
RUN_ALL=0     # 1: use -a; 0: use --datasets "$DATASETS"
USE_E=0       # 1: LongBench-E
```

To run fewer datasets, edit the comma-separated `DATASETS` string.

The evaluator now accepts:

```bash
--task_eval_dir
--sim_thr_grouplogitkv
--sim_thr_simcalkv
--sim_thr_keepkv
```

so the threshold sweep is actually passed into `GroupLogitKV`, `SimCalKV`, and `KeepKV`. `CaM` ignores `sim_thr`.
