# Preliminary graph-MIS diagnostic on an existing real-KV snapshot

This is a geometry-only smoke analysis, not a new PPL/LongBench result. It reuses
a previously captured **uncompressed** Llama-3.2-3B WikiText-2 snapshot
(sequence length 2048; KV layers 0/14/27; 8 KV heads, head dimension 128) and
applies the new graph-MIS implementation offline at compression ratio 0.4.
The retained/evicted split is the internal current-attention selector so that
`anchor_only` and `full` share exactly the same split and nearest-compatible
routing.

## Non-degenerate merging and complete-group effect

At `lambda=0.5`:

| layer | routed evicted | graph-MIS accepted | accepted / routed | anchor-only exact violating groups | full exact violating groups | max pool | actual MIS rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 23.98% | 21.87% | 91.22% | 8.62% | **0%** | 11 | 1 |
| 14 | 40.54% | 36.25% | 89.42% | 12.37% | **0%** | 6 | 1 |
| 27 | 24.51% | 22.07% | 90.04% | 10.55% | **0%** | 7 | 2 |

At `lambda=0.75`, graph-MIS accepts 76.95% / 94.58% / 85.84% of evicted
entries in layers 0/14/27 while retaining zero exact first-round
complete-diameter violations. The corresponding anchor-only violation ratios
are 14.79% / 6.20% / 12.69%.

These observations show that the new method does **not** inherently collapse to
`eviction_only` under the new frozen RMS-pair-distance scale, and that the
candidate-candidate conflict step removes real many-to-one incompatibilities
that pair-compatible anchor routing alone cannot see.

## Maximal-MIS heuristic quality

For routed pools of size <=14, up to 200 pools per layer/lambda were compared
against an exact maximum-independent-set cardinality oracle. Across the tested
`lambda=0.4,0.5,0.75` settings, 98.5%-100% of sampled pools reached the exact
maximum cardinality; the largest observed gap was one candidate. Thus the
current degree-priority maximal MIS does not appear to lose substantial merge
coverage on this snapshot. Larger pools still require empirical validation.

## Performance caveat

CPU timing is only a code-path smoke test and must not be used in the paper. GPU
measurement is required. The method's main engineering risk is the local
candidate-candidate graph cost and current padded eager-PyTorch implementation,
not the MIS selection quality itself.

## What remains unknown

- PPL/LongBench quality under the new graph-MIS method;
- the best `lambda` for the default KeepKV-style selector;
- GPU graph construction/MIS overhead and memory;
- behavior on Qwen and at longer sequences/larger candidate pools.

Run `run_mymethod_quick_validation.sh` to collect these next.
