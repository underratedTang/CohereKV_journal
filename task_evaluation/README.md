# task_evaluation

PPL and task-evaluation scripts for KV-cache compression baselines.

Supported methods:
- none
- SimCalKV
- GroupLogitKV
- KeepKV
- CaM

PyramidInfer has been removed from this evaluation package.

`run_ppl_kvmerge_sweep.sh` performs a fair physical-budget comparison and supports per-method similarity-threshold sweeps for GroupLogitKV, SimCalKV, and KeepKV. CaM does not use similarity thresholding.
