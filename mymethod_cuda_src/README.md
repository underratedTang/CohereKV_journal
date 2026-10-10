# mymethod_cuda

Standalone CUDA backend for `mymethod`. It has **no GroupLogitKV runtime/build dependency**.

## Kernels

1. `count_calibrated_qk`: single-token decode `QK/sqrt(d) + log(C)`.
2. `compatible_route`: nearest certificate-compatible retained anchor without materializing the full pair matrix.
3. `graph_mis_select`: device-side ragged candidate packing, conflict graph construction, deterministic degree-priority parallel MIS, and exact selected-group Gamma.

The common graph path uses one CUDA block per candidate pool and a shared-memory bitset for pools up to 256 candidates. Pools larger than 256 use an exact no-cap fallback that recomputes conflicts rather than changing the mathematical method.

## Install

```bash
cd /home/tlxr/CohereKV/mymethod_cuda_src
bash install_mymethod_cuda.sh
```

When the target GPU is not visible during compilation, set `TORCH_CUDA_ARCH_LIST` explicitly, e.g. `8.0` for A100.

## Smoke test

```bash
python test_mymethod_cuda.py
```

`task_evaluation` keeps a PyTorch reference backend and compares CUDA against it before formal experiments.
