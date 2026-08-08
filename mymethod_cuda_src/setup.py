import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Prefer PyTorch's visible-GPU architecture detection.  Override explicitly when
# building on a login node or for deployment to different GPUs, e.g.:
#   TORCH_CUDA_ARCH_LIST="8.0" python -m pip install -v --no-build-isolation .
setup(
    name="mymethod_cuda",
    version="0.2.0",
    description="Independent CUDA kernels for mymethod Graph-MIS KV-cache compression",
    packages=["mymethod_cuda"],
    ext_modules=[
        CUDAExtension(
            name="mymethod_cuda_ext",
            sources=[
                "csrc/pybind.cpp",
                "csrc/count_calibrated_qk_cuda.cu",
                "csrc/compatible_route_cuda.cu",
                "csrc/graph_mis_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
