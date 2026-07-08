# GroupLogitKV calibrated_qk long-K patch

This source removes the previous `K length > 16384` hard check in `csrc/calibrated_qk_cuda.cu`.

The kernel is still decode-only (`q_len == 1`) and computes the same output:

```text
scores[b,h,0,j] = dot(q[b,h,0,:], k[b,h,j,:]) / sqrt(D) + log_bias[j]
```

No K truncation or PyTorch fallback is introduced. Reinstall after replacing the source:

```bash
cd grouplogitkv_calibrated_qk_src
pip uninstall -y grouplogitkv_calibrated_qk
pip install -v . --no-build-isolation
```

Use `--no-build-isolation` so the build can see the PyTorch installed in the active conda environment.
