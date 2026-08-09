# Graph-MIS mymethod: first real feedback analysis

## Current result

On WikiText-2, Llama-3.2-3B, 8192 tokens, 2 sequences, compression ratio 0.4:

- current-attention selector + eviction-only: PPL 6.551109
- current-attention selector + anchor-only, lambda=0.5: PPL 6.554587
- current-attention selector + full graph-MIS, lambda=0.5: PPL 6.550182
- default KeepKV-style selector + full, lambda=0.5: PPL 6.583004
- default KeepKV-style selector + full, lambda=0.75: PPL 6.596697

Thus graph-MIS prevents the quality degradation caused by uncertified anchor-only grouping at lambda=0.5, but it has not yet produced a material gain over eviction-only.

At lambda=0.5 in dynamic PPL, only about 6.17% of evicted entries are compatible and local pools are almost always size 1. Therefore MIS rarely changes membership. Higher lambda must be tested with the same selector before judging the graph design.

Offline 2048-token geometry shows that at lambda=0.5 anchor-only groups have roughly 8.6--12.4% complete-group violations on the sampled layers, while full graph-MIS has zero. MIS retains about 89--91% of already-routed candidates. At lambda=0.75 it retains about 87--95% while again reducing exact violations to zero.

The small-pool exact maximum-independent-set oracle shows the degree-priority parallel MIS is essentially cardinality-optimal on these samples (98.5--100% optimal-group fraction; maximum gap one candidate). The current design bottleneck is therefore graph construction/execution cost rather than MIS selection quality.

## Engineering result

At layer 0, length 2048, lambda=0.5:

- anchor route: 2.039 ms median
- conflict graph + parallel MIS: 4.002 ms median (standalone stage benchmark)
- anchor-only full plan: 4.270 ms median
- full graph-MIS plan: 5.701 ms median
- CentroidKV native merge core: 1.585 ms median
- internal GroupLogitKV merge core: 0.634 ms median

The incremental full-vs-anchor plan cost is about 1.43 ms (+33.5%). Graph/MIS engineering optimization is necessary before making a system-speed claim.

## Next minimal experiment

Run `run_mymethod_targeted_followup.sh` before LongBench. It answers two unresolved questions:

1. Is the weak default-selector PPL caused by the selector itself or by grouping?
2. At lambda 0.65/0.75, where candidate pools and conflicts are nontrivial, does graph-MIS improve PPL over anchor-only grouping?

Only after these are resolved should lambda and selector be frozen for LongBench/system experiments.
