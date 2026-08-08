# mymethod graph-MIS parallelism and optimization report

## What is already parallel

1. **Anchor routing**: all evicted-retained squared distances for a tile are a
   batched GEMM over `B*H_kv`; compatibility is a broadcast Boolean tensor and
   nearest compatible anchor is a reduction over the retained dimension.
2. **Local conflict graphs**: candidate pools from all anchors are padded and
   evaluated together by batched candidate-candidate GEMM.
3. **Parallel MIS**: within every round, all anchors and all active vertices are
   processed simultaneously. Only MIS rounds are sequential, which is inherent
   to this deterministic parallel-MIS procedure.
4. **Aggregation**: count, K/V and certificate updates use segmented
   scatter/scatter-reduce operations rather than Python loops over groups.

## Why graph-MIS is more expensive than local matching

The new method explicitly pays for candidate-candidate compatibility inside
routed pools. Its extra geometric work is `O(d * sum_j m_j^2)`. This is the
mechanism that converts source-anchor compatibility into a complete-group
all-pairs guarantee. It should therefore be evaluated as a quality/reliability
versus grouping-cost tradeoff, not assumed to be faster than every local
matching baseline.

## Highest-value implementation optimizations that preserve the method

- store only triangular conflict bitsets instead of dense Boolean matrices;
- sort/bucket pools once and use grouped GEMM to avoid padding to `max_pool`;
- fuse candidate distance, squared certificate gate, degree accumulation and
  bitset write into one CUDA/Triton kernel;
- fuse MIS winner detection + neighbor deletion and terminate on device when no
  active vertices remain;
- fuse selected-pair certificate reduction with the conflict/bitset pass;
- reuse allocated routing/graph workspaces across layers/compression rounds;
- for the default KeepKV-style selector, evaluate BF16 history storage with
  FP32 accumulation as a separate engineering option.

## Measurement policy

Do not enable `--mymethod_diagnostics_jsonl` during timing: diagnostics contain
`.item()` calls and intentionally synchronize the GPU. Use
`benchmark_mymethod_grouping.py` for grouping-core boundaries, and the ordinary
system evaluator with diagnostics disabled for end-to-end/persistent-memory
measurements.
