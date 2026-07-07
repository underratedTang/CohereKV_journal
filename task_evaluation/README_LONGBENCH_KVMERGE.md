# LongBench v1 KV-merge evaluation

Use `run_longbench_kvmerge_sweep.sh` as the main entry.

This package keeps the LongBench source directory and adds one evaluator:

- `evaluate_longbench_kvmerge.py`
- `run_longbench_kvmerge_sweep.sh`

The evaluator keeps the reusable LongBench/AquaKV entry logic: `MODELS`, `DATASETS`, prompt building, post-processing, model loading branches, and input-partition wrappers. AquaKV-specific quantized cache options are not included.

Supported KV merge methods:

- GroupLogitKV
- SimCalKV
- KeepKV
- CaM

Run:

```bash
cd /root/SimCalKV/task_evaluation
bash run_longbench_kvmerge_sweep.sh
```
