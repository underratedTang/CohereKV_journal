# LongBench v1 KV-merge evaluation

This `task_evaluation` package keeps the LongBench source tree under `LongBench/` and adds a KV-merge LongBench entry point:

- `evaluate_longbench_kvmerge.py`
- `run_longbench_kvmerge_sweep.sh`

The evaluator is aligned with the original `evaluate_longbench_aquakv.py` where it matters for LongBench execution:

- same LongBench v1 English dataset list;
- same `MODELS` name list;
- same `-a` / `--datasets` / `-e` control style;
- same prompt wrapping policy in `build_chat()`;
- same simple `post_process()` for xgen/internlm;
- same middle truncation strategy before chat wrapping;
- same `result.json` / `result_e.json` collection path convention.

The AquaKV-specific cache quantization path is intentionally not copied. This evaluator uses the post-hoc KV merge/compression implementations in:

- `llama/merge_grouplogitkv_llama.py`
- `llama/merge_simcalkv_llama.py`
- `llama/merge_keepkv_llama.py`
- `llama/merge_cam_llama.py`
- `qwen/merge_grouplogitkv_qwen.py`
- `qwen/merge_simcalkv_qwen.py`
- `qwen/merge_keepkv_qwen.py`
- `qwen/merge_cam_qwen.py`

Run example:

```bash
cd /root/SimCalKV/task_evaluation
bash run_longbench_kvmerge_sweep.sh
```

`run_longbench_kvmerge_sweep.sh` follows the style of `run_ppl_kvmerge_sweep.sh` and keeps LongBench dataset control simple:

```bash
DATASETS="narrativeqa,qasper,multifieldqa_en,passage_retrieval_en,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,hotpotqa"
RUN_ALL=0
USE_E=0
```
