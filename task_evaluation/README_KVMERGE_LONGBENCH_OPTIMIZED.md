# LongBench KV-merge optimized evaluator

This package keeps the LongBench v1 evaluation flow unchanged while avoiding the
full `outputs.attentions` memory blow-up.

## Engineering-equivalent attention-score capture

The runtime attention patch now reduces each layer's attention probabilities
inside the attention forward pass and stores only the score consumed by the merge
algorithms:

- GroupLogitKV: `[B, H_kv, K] = mean_over_GQA_groups(attn).sum_over_queries()`
- SimCalKV: `[B, H_kv, K] = mean_over_GQA_groups(attn).sum_over_queries()`
- KeepKV: `[B, H_kv, K] = mean_over_GQA_groups(attn).mean_over_queries()`
- CaM: `[B, H_kv, K] = mean_over_GQA_groups(attn).mean_over_queries()`

The evaluator no longer asks Hugging Face to collect full layer attentions
`[B, H_q, Q, K]` in `outputs.attentions`. It still runs full-prompt prefill; it
does not use chunked prefill and does not change prompts, generation, datasets,
model max length, or compression ratios.

## Modified files

- `evaluate_longbench_kvmerge.py`
- `llama/grouplogitkv_runtime_patch_llama.py`
- `qwen/grouplogitkv_runtime_patch_qwen.py`
- `llama/merge_grouplogitkv_llama.py`
- `qwen/merge_grouplogitkv_qwen.py`
- `llama/merge_simcalkv_llama.py`
- `qwen/merge_simcalkv_qwen.py`
- `llama/merge_keepkv_llama.py`
- `qwen/merge_keepkv_qwen.py`
- `llama/merge_cam_llama.py`
- `qwen/merge_cam_qwen.py`

The merge functions remain backward-compatible with old 4D attentions and now
also accept pre-reduced 3D scores.
