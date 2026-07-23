# task_evaluation

Current comparison methods:
- none
- SimCalKV
- GroupLogitKV
- KeepKV
- CaM

PyramidInfer comparison code has been removed.

## Fair-comparison settings

### Budget
All compression methods use the same evaluator semantics:

```text
compress_ratio = evicted fraction
compress_ratio=0.4 => evict 40%, keep about 60% physical KV slots
```

The run summary records `avg_compressed_kv_len`; this should be checked after each sweep to confirm all compressed methods use the same physical KV budget.

### Similarity thresholds
There is no global `sim_thr`.
Only methods that use similarity assignment accept a threshold:

- `--sim_thr_grouplogitkv`
- `--sim_thr_simcalkv`
- `--sim_thr_keepkv`

CaM does not use a similarity threshold and receives no `sim_thr` argument.

`run_ppl_kvmerge_sweep.sh` now sweeps thresholds separately per method using arrays:

```bash
SIM_THR_GROUPLOGITKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
SIM_THR_SIMCALKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
SIM_THR_KEEPKV_LIST=("-1" "0.0" "0.2" "0.4" "0.6" "0.8")
```

For a strict controlled threshold comparison, keep the same list for all three similarity-merge methods. For official/recommended settings, reduce each list to the method-specific setting, such as KeepKV `0.8`.

### Evaluation length
The current evaluator contains the user's manual setting inside `evaluate_perplexity()`:

```python
m = 3
inps = inps[:m]
```

This is applied to all methods, so it is fair across methods. The JSON field `num_eval_tokens` reports the full tokenized dataset length and is not the actual number of loss tokens when `m` is manually limited.

## KeepKV
The LLaMA and Qwen KeepKV implementations are in:
- `llama/merge_keepkv_llama.py`
- `qwen/merge_keepkv_qwen.py`

They maintain Electoral Votes/counts and expose log(count) bias through the same eager runtime attention patch used by GroupLogitKV.
