#!/usr/bin/env bash
set -euo pipefail
# =========================
# 手动修改区
# =========================
ROOT="/home/tlxr/SimCalKV/mymethod_cuda_src"
# 留空时由 PyTorch 根据当前可见 GPU 自动选择架构；如需离线编译可手动填 "8.9"（RTX 4090）。
TORCH_CUDA_ARCH_LIST_VALUE=""

# =========================
# 自动区：一般不用改
# =========================
cd "${ROOT}"
if [[ -n "${TORCH_CUDA_ARCH_LIST_VALUE}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_VALUE}"
fi
python -m pip uninstall -y mymethod_cuda >/dev/null 2>&1 || true
# If TORCH_CUDA_ARCH_LIST is unset, PyTorch builds for the visible GPU(s).
python -m pip install -v --no-build-isolation .
python test_mymethod_cuda.py
