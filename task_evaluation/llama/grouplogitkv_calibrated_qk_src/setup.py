from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="grouplogitkv_calibrated_qk",
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
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
