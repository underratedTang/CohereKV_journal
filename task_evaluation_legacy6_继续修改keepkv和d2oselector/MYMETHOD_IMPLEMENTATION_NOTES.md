# mymethod graph-MIS implementation notes

This package implements the method in
`docs/method_motivation_method_theory_zh_graph_mis_literature_motivation.md`.
The grouping contribution is selector-agnostic. The implementation exposes three
independent selector instances: `attention`, `keepkv`, and `d2o`. No selector is
assumed to be universally superior across models/tasks. `d2o` uses only the
released D2O retained/evicted *selection principle* (request-level cumulative
post-softmax attention with hard sink/recent protection); D2O cosine routing,
EMA merge threshold, weighted merging, and layer allocation are not used.
GroupLogitKV remains only as a separate comparison method and is not imported by
mymethod.

## Selector configuration boundaries

Selector parameters are intentionally namespaced and are not shared:

- `attention`: no sink/recent/history hyperparameters; importance is refreshed
  from the current captured forward/compression interval.
- `keepkv`: `MYMETHOD_KEEPKV_SINK_SIZE`,
  `MYMETHOD_KEEPKV_RECENT_PROTECT_SIZE`, `MYMETHOD_KEEPKV_SCORE_WINDOW`,
  `MYMETHOD_KEEPKV_RECENCY_BETA`, and `MYMETHOD_KEEPKV_SINK_CAN_ABSORB`.
- `d2o`: `MYMETHOD_D2O_SINK_SIZE`, `MYMETHOD_D2O_RECENT_SIZE`,
  `MYMETHOD_D2O_SINK_CAN_ABSORB`, and `MYMETHOD_D2O_USE_LOG_COUNT`.

For `d2o`, sink entries consume the heavy-hitter budget and recent entries are
protected separately, matching the released merge-code selection semantics.
With `MYMETHOD_D2O_USE_LOG_COUNT=1` (default in the run scripts), importance
accumulates the actual post-softmax attention used by mymethod after `+log(C)`.
After a certified merge, the retained anchor keeps only its own historical D2O
importance; evicted-source history is discarded. Once retained/evicted sets are
formed, all routing, conflict-graph/MIS selection, K/V merge, C/R/D propagation,
and decode calibration are exclusively mymethod operations.

## Core state

Per physical KV entry and layer/head:

- `K,V`: cached centroid key/value;
- `C`: INT32 number of represented original tokens;
- `R`: FP32 radius certificate;
- `D`: FP32 complete-diameter certificate.

The effective radius is
`Rhat = min(R, (C-1)/C * D)`.

At the first compression of a request, the implementation computes the exact
same reference scale as the Welford/M2 definition in the method document:

`S_ref = sqrt(2*M2/(n-1) + eps^2)`.

Because every entry is still an unmerged original token at that point, a
vectorized one-pass mean/M2 computation is algebraically identical to updating
Welford statistics token by token. `S_ref` is then frozen for the remainder of
the request. One FP32 scalar per batch/KV-head/layer is persistent.

The group threshold is `Delta = lambda * S_ref`. `lambda` is not a similarity
score and is not restricted to `(0,1)`. The default `0.5` in code is only a
provisional engineering starting point; experiments must select/report it.

## Step 2: pair-compatible routing

For an evicted entry `i` and retained anchor `j`, compatibility is

`Gamma_ij = max(D_i, D_j, Rhat_i + ||K_i-K_j|| + Rhat_j) <= Delta`.

The implementation uses batched Euclidean-distance GEMM and an equivalent
squared-distance gate, so the full evicted-retained matrix does not need a
`sqrt`. Among compatible anchors, each evicted entry is routed to its nearest
anchor. `MYMETHOD_DISTANCE_CHUNK_SIZE<=0` processes all evicted entries in one
GEMM; positive values tile only the evicted dimension.

## Step 3: local conflict graph + parallel MIS

Routed candidates are bucketed by `(batch*KV-head, retained-anchor)`. All
non-empty pools are padded into one batch and candidate-candidate distances are
computed by a batched GEMM. Two candidates conflict when their entry-pair
certificate exceeds `Delta`.

Priority is fixed lexicographically by:

1. lower conflict degree;
2. lower anchor certificate `Gamma_ij`;
3. lower stable stateless hash of the physical entry index.

Every MIS round selects all active vertices that have no higher-priority active
conflict neighbor, then removes those winners and all of their conflict
neighbors. This returns a deterministic maximal independent set. `full` uses
this mask; `anchor_only` accepts all pair-compatible routed candidates and is
an internal no-conflict-selection ablation; `eviction_only` accepts none.

## Step 5/6 reduction and certificates

K/V are count-weighted arithmetic means. K and V numerator accumulation is
combined into one FP32 segmented scatter-add. `D_new` is the maximum old-entry
diameter/cross-entry `Gamma` among the retained anchor and all selected
candidates. `R_new` is computed relative to the actual dtype-rounded cached
centroid and then tightened by `min(Rtilde, D_new)`.

The decode runtime adds `log(C)` to the entry logit before softmax.

## Current engineering limitations

The algorithm is tensor-parallel, but the eager PyTorch implementation is not
a fully fused GPU kernel:

- local pools are padded to the largest pool, so irregular pools waste work;
- obtaining the largest pool size currently performs one GPU-to-host scalar
  synchronization per layer/compression;
- the implementation materializes a Boolean conflict matrix rather than a
  compact bitset;
- the parallel MIS loop executes a fixed `max_pool` number of device rounds to
  avoid a host convergence check, even when the graph converges in 1-2 rounds;
- candidate-candidate work is `O(d * sum_j m_j^2)` by design;
- the default KeepKV-style selector maintains an FP32 recent-score history and
  can cost more than the grouping core.

These are engineering issues, not method-definition changes. The next GPU
optimization targets are: bucketed/segmented GEMM without dense padding,
triangular conflict bitsets, fused squared-distance/certificate construction,
and a device-side early-termination MIS kernel (CUDA/Triton).

## Independent CUDA backend

`mymethod_cuda_src` is the only optional native dependency of mymethod. It owns
its QK+log(C), compatible-routing, and ragged conflict-graph/MIS kernels. The
PyTorch implementation remains the exact reference/fallback. Formal evaluators
expose separate route and graph backend switches.
