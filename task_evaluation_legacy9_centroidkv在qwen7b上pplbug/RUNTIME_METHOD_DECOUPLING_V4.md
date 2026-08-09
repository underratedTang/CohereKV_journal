# KV method runtime decoupling

The evaluator now treats method-specific runtime code as isolated dependencies.

- `GroupLogitKV`: its own `grouplogitkv_runtime_patch_*` remains intact.
- `mymethod`: its own `mymethod_runtime_patch_*`.
- `KeepKV`: its own `keepkv_runtime_patch_*`.
- `SimCalKV` / `CaM`: neutral `score_capture_runtime_patch_*` only, because they need reduced attention scores but no GroupLogitKV semantics.
- `CentroidKV`: no attention-score runtime patch; only `merge_centroidkv_*` is imported. Legacy `eager` is automatically changed to `sdpa` for this method.
- full KV: no compression runtime patch.

The neutral capture runtime performs exact query-chunked attention during score capture and immediately reduces scores to `[B,H_kv,K]`. It contains no GroupLogitKV log-count bias, calibrated-QK dependency, selector, or merging code.
