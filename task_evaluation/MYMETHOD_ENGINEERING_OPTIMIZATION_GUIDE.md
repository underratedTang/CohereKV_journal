# mymethod independent CUDA engineering plan

## What changed

`mymethod` is now code-level independent from GroupLogitKV.

- `mymethod` runtime imports `mymethod_cuda`, never `grouplogitkv_calibrated_qk`.
- selector names are `keepkv`, `attention`, and `d2o`. `attention` is the standalone current-attention policy; `d2o` is a D2O-style request-cumulative post-softmax retained/evicted selector only. The certified grouping core is unchanged.
- GroupLogitKV remains only as an independent comparison method under its own files.
- PyTorch routing/Graph-MIS remain the correctness reference and CPU fallback.

The CUDA package lives beside `task_evaluation`:

```
/home/tlxr/SimCalKV/
  mymethod_cuda_src/
  task_evaluation/
  grouplogitkv_calibrated_qk_src/   # optional comparison-method code only
```

## Independent CUDA kernels

### Decode QK + log(C)

`count_calibrated_qk` computes the method's single-token decode logits without any
GroupLogitKV import.

### Compatible routing

`compatible_route` computes nearest certificate-compatible retained anchors
without materializing the full `Ne x Nr` distance/feasibility/masked matrices.
It is benchmarked against the existing translated-GEMM PyTorch route; it is not
assumed to be faster until measured on the target GPU.

### Ragged conflict graph + MIS

The common candidate-pool path eliminates the old:

```
argsort -> padded [G,max_pool] -> FP32 pair matrix -> bool conflict matrix
        -> Python max_pool kernel loop
```

and uses:

```
GPU group count -> device prefix sum -> ragged candidate pack
-> one CUDA block / candidate pool
-> shared-memory conflict bitset (m <= 256)
-> device-resident degree-priority MIS loop
-> recompute only selected-pair distances for group Gamma
```

There is no algorithmic pool cap. Pools larger than 256 use an exact no-cap CUDA
path that recomputes pair conflicts instead of truncating candidates.

## Backends

Every formal evaluator supports:

```
--mymethod_route_backend auto|cuda|torch
--mymethod_graph_backend auto|cuda|torch
```

`torch` is the reference. `cuda` is strict. `auto` uses the independent extension
when CUDA is available and otherwise falls back to PyTorch.

For final paper experiments, do not decide the backend from source-code intuition.
Run `run_mymethod_engineering_validation.sh` once; use the faster backend only if
correctness passes and median speedup is positive on the actual GPU.

## Selector policy

Engineering is selector-independent. Do not infer that `attention`, `keepkv`, or `d2o`
is universally better from the engineering run. All selectors share the same
certificate/routing/Graph-MIS backend. `d2o` changes only retained/evicted partitioning;
D2O cosine routing, EMA merge threshold, weighted merging, and layer allocation are not
part of `mymethod`. Final selector quality is evaluated later on LLaMA/Qwen and LongBench.

## Correctness stop conditions

Do not run final LongBench/system tables unless all hold:

1. `mymethod_cuda/test_mymethod_cuda.py` passes FP16/BF16 QK, routing, small-pool
   MIS, and >256 exact no-cap tests.
2. task_evaluation's method/regression tests pass.
3. synthetic CUDA vs PyTorch plan membership is exact.
4. real-snapshot CUDA groups remain certificate-safe.
5. CUDA-vs-PyTorch PPL backend parity is numerically negligible for the selector configurations used in the engineering gate (the one-command gate enforces absolute PPL difference <= 0.005). Selector expansion does not require CUDA recompilation because the kernels are unchanged.

## Performance decision after one engineering run

The summary mechanically recommends CUDA only when median `torch/cuda >= 1.05`.
This is done separately for routing and Graph-MIS. Therefore possible final
configurations include `route=torch, graph=cuda`; a custom routing kernel is not
kept merely because it exists.

After this engineering validation, freeze the implementation. The next stage is
the final paper experiment matrix, not another redesign cycle.
