# GroupLogitKV Calibrated QK CUDA Extension

This directory contains the CUDA extension used by GroupLogitKV to compute the calibrated single-query attention logits

\[
\tilde L = qK^\top / \sqrt{D} + \log C.
\]

It is used by the LLaMA/Qwen runtime patches through:

```python
from grouplogitkv_calibrated_qk import calibrated_qk
```

The extension only implements the QK logits plus the compressed-prefix `log(C)` bias. Attention mask, softmax, dropout, and AV are handled by the model runtime patch.

## Recommended location

Place this directory at the project root, parallel to `task_evaluation`:

```text
GroupLogitKV/
├── grouplogitkv_calibrated_qk_src/
└── task_evaluation/
    ├── llama/
    └── qwen/
```

After installation, both LLaMA and Qwen evaluation scripts can import the same extension.

## Requirements

Install PyTorch with CUDA support in the active environment before building this extension.

Check PyTorch and CUDA availability:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('torch cuda:', torch.version.cuda)
PY
```

If this fails with `ModuleNotFoundError: No module named 'torch'`, install the CUDA-enabled PyTorch version that matches your system before continuing.

## Installation

Use `--no-build-isolation` when installing. This is important because `setup.py` imports `torch`; with pip build isolation enabled, pip creates a temporary build environment that may not contain torch, causing:

```text
ModuleNotFoundError: No module named 'torch'
```

Install with:

```bash
cd GroupLogitKV/grouplogitkv_calibrated_qk_src
pip install --no-build-isolation -v .
```

For editable/development mode:

```bash
cd GroupLogitKV/grouplogitkv_calibrated_qk_src
pip install --no-build-isolation -v -e .
```

Verify import:

```bash
python -c "from grouplogitkv_calibrated_qk import calibrated_qk; print('ok')"
```

## GPU architecture selection

By default, the provided `setup.py` is expected to support common architectures such as:

```text
sm_80, sm_86, sm_89, sm_90
```

Typical GPUs:

```text
A100:        sm_80
RTX 3090:    sm_86
RTX 4090:    sm_89
L40/L40S:    sm_89
H100:        sm_90
```

To restrict compilation to your GPU architecture, set `TORCH_CUDA_ARCH_LIST` before installation. Examples:

```bash
# A100
TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -v .

# RTX 3090
TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -v .

# RTX 4090 / L40 / L40S
TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation -v .

# H100
TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation -v .
```

## Testing

Run the extension tests after installation:

```bash
python -m grouplogitkv_calibrated_qk.test
```

The tests should cover:

1. full-length bias, where `logC_len == key_len`;
2. compressed-prefix bias, where `logC_len < key_len`;
3. zero-length bias, equivalent to ordinary QK logits;
4. batched sanity checks.

The CUDA output should be close to the PyTorch reference:

```python
ref = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
ref[..., :Kc] += logc
```

## API

```python
out = calibrated_qk(q, k, logc)
```

Expected shapes:

```text
q:    [B, H, 1, D]
k:    [B, H, K, D]
logc: [B, H, 1, Kc], where Kc <= K
out:  [B, H, 1, K]
```

Semantics:

```text
out[..., j] = q @ k[..., j, :] / sqrt(D) + logc[..., j],  if j < Kc
out[..., j] = q @ k[..., j, :] / sqrt(D),                 if j >= Kc
```

The suffix positions `j >= Kc` correspond to newly generated tokens with count `C=1`, so their log-bias is zero.

## Current limitations

The CUDA kernel is designed for the common autoregressive decoding path:

```text
q_len == 1
```

Other limitations:

```text
head_dim <= 256
key_len <= 16384
CUDA only
```

For non-CUDA execution or `q_len > 1`, the Python runtime patch should fall back to ordinary PyTorch QK computation and add the prefix `log(C)` bias in Python.

## Rebuild / clean install

If you modify CUDA/C++ code or change GPU architecture settings, rebuild the extension:

```bash
pip uninstall grouplogitkv_calibrated_qk -y
rm -rf build *.egg-info
pip install --no-build-isolation -v .
```

If stale compiled files remain, also remove generated `.so` files inside the package directory before reinstalling.

## Common errors

### `ModuleNotFoundError: No module named 'torch'` during `pip install`

Cause: pip build isolation created a temporary build environment without PyTorch.

Fix:

```bash
pip install --no-build-isolation -v .
```

Also verify that PyTorch is installed in the current environment:

```bash
python -c "import torch; print(torch.__version__)"
```

### `CUDA_HOME environment variable is not set`

Cause: CUDA toolkit is not visible to the build system.

Fix examples:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then rebuild.

### `no kernel image is available for execution on the device`

Cause: the extension was compiled for a different GPU architecture.

Fix by setting the correct architecture and reinstalling, for example on A100:

```bash
pip uninstall grouplogitkv_calibrated_qk -y
rm -rf build *.egg-info
TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -v .
```

### Import works but runtime output is wrong

Run the test suite:

```bash
python -m grouplogitkv_calibrated_qk.test
```

If the tests fail, rebuild from a clean directory and ensure the active Python environment is the same one used by your evaluation scripts.
