import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Default to common NVIDIA architectures used for LLaMA/Qwen evaluation.
# Override at install time if needed, e.g.:
#   TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install -v .
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

setup(
    name="grouplogitkv_calibrated_qk",
    version="0.1.0",
    description="Fused QK + log-count bias CUDA kernel for GroupLogitKV decoding",
    packages=["grouplogitkv_calibrated_qk"],
    ext_modules=[
        CUDAExtension(
            name="grouplogitkv_calibrated_qk_ext",
            sources=[
                "csrc/pybind.cpp",
                "csrc/calibrated_qk_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
