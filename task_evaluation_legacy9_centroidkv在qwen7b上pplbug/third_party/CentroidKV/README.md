# CentroidKV official reference

`centroidkv_press.py` is the user-provided official implementation from:

`https://github.com/jessica-hujie/CentroidKV`

It is retained unchanged for provenance. The adapters under `models/common`,
`models/llama`, and `models/qwen` remove the external `kvpress` dependency while
preserving the official protected sink/local regions, bipartite soft matching,
iterative schedule, and count-weighted centroid merge.
