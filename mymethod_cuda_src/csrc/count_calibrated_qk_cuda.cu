#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "mymethod_cuda.h"

template <typename scalar_t>
__device__ __forceinline__ float mm_to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t mm_from_float(float x) {
    return static_cast<scalar_t>(x);
}

__device__ __forceinline__ float mm_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

/*
 * Independent mymethod decode kernel.
 *
 * Computes, for q_len == 1,
 *   score[b,h,0,j] = dot(q[b,h,0,:], k[b,h,j,:]) / sqrt(D) + logc[b,h,0,j]
 * where logc may be shorter than K.  Missing suffix bias is exactly zero because
 * newly generated unmerged entries have C=1.
 *
 * This implementation is owned by mymethod_cuda and has no runtime or build
 * dependency on GroupLogitKV or grouplogitkv_calibrated_qk.
 */
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void mymethod_count_calibrated_qk_q1_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ logc,
    scalar_t* __restrict__ out,
    int H,
    int KLEN,
    int LOGC_LEN,
    int D,
    float scale,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t c_s0, int64_t c_s1, int64_t c_s2, int64_t c_s3,
    int64_t o_s0, int64_t o_s1, int64_t o_s2, int64_t o_s3) {
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh - b * H;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (j >= KLEN) return;

    float acc = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float qv = mm_to_float(q[b*q_s0 + h*q_s1 + d*q_s3]);
        const float kv = mm_to_float(k[b*k_s0 + h*k_s1 + j*k_s2 + d*k_s3]);
        acc += qv * kv;
    }
    acc = mm_warp_sum(acc);
    if (lane == 0) {
        float bias = 0.0f;
        if (j < LOGC_LEN) {
            bias = mm_to_float(logc[b*c_s0 + h*c_s1 + j*c_s3]);
        }
        out[b*o_s0 + h*o_s1 + j*o_s3] = mm_from_float<scalar_t>(acc * scale + bias);
    }
}

torch::Tensor mymethod_count_calibrated_qk_q1_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor logc) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && logc.is_cuda(), "q/k/logc must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && logc.dim() == 4,
                "q,k,logc must be [B,H,Q/K,D] rank-4 tensors");
    TORCH_CHECK(q.size(2) == 1, "count_calibrated_qk_q1 requires q_len=1");
    TORCH_CHECK(k.size(0) == q.size(0) && k.size(1) == q.size(1) && k.size(3) == q.size(3),
                "q/k shape mismatch");
    TORCH_CHECK(logc.size(0) == q.size(0) && logc.size(1) == q.size(1) && logc.size(2) == 1,
                "logc shape mismatch");
    TORCH_CHECK(logc.size(3) <= k.size(2), "logc prefix length must be <= key length");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == logc.scalar_type(),
                "q/k/logc dtype mismatch");
    TORCH_CHECK(q.size(3) <= 512, "head_dim > 512 is not supported by this kernel");

    const int64_t B64 = q.size(0), H64 = q.size(1), K64 = k.size(2), D64 = q.size(3);
    TORCH_CHECK(B64 * H64 <= 2147483647LL && K64 <= 2147483647LL,
                "tensor dimensions exceed CUDA int32 grid/index range");
    auto out = torch::empty({B64, H64, 1, K64}, q.options());
    constexpr int WARPS = 8;
    constexpr int THREADS = WARPS * 32;
    dim3 blocks(static_cast<unsigned int>(B64 * H64),
                static_cast<unsigned int>((K64 + WARPS - 1) / WARPS));
    const float scale = rsqrtf(static_cast<float>(D64));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "mymethod_count_calibrated_qk_q1_cuda",
        [&] {
            mymethod_count_calibrated_qk_q1_kernel<scalar_t, WARPS><<<blocks, THREADS, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), logc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                static_cast<int>(H64), static_cast<int>(K64), static_cast<int>(logc.size(3)), static_cast<int>(D64), scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                logc.stride(0), logc.stride(1), logc.stride(2), logc.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
