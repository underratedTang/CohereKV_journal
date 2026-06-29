#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ float to_float_device(scalar_t x) {
    return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float_device(float x) {
    return static_cast<scalar_t>(x);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

/*
  Warp-per-key calibrated QK.

  Computes:
    scores[b,h,0,j] = dot(q[b,h,0,:], k[b,h,j,:]) / sqrt(D) + bias

  where:
    bias = logc[b,h,0,j] if j < LOGC_LEN
    bias = 0             otherwise

  The short-logC case is intentional: generated suffix tokens have count C=1,
  hence logC=0. This avoids padding/cat in the decode hot path.

  Does not apply mask, softmax, dropout, or AV.
*/
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void calibrated_qk_q1_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ logc,
    scalar_t* __restrict__ scores,
    int H,
    int KLEN,
    int LOGC_LEN,
    int D,
    float scale,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t logc_s0, int64_t logc_s1, int64_t logc_s2, int64_t logc_s3,
    int64_t scores_s0, int64_t scores_s1, int64_t scores_s2, int64_t scores_s3
) {
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh - b * H;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int j = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (j >= KLEN) {
        return;
    }

    float acc = 0.0f;

    for (int d = lane; d < D; d += 32) {
        const float qv = to_float_device(q[b * q_s0 + h * q_s1 + 0 * q_s2 + d * q_s3]);
        const float kv = to_float_device(k[b * k_s0 + h * k_s1 + j * k_s2 + d * k_s3]);
        acc += qv * kv;
    }

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        float bias = 0.0f;
        if (j < LOGC_LEN) {
            bias = to_float_device(logc[b * logc_s0 + h * logc_s1 + 0 * logc_s2 + j * logc_s3]);
        }
        const float out = acc * scale + bias;
        scores[b * scores_s0 + h * scores_s1 + 0 * scores_s2 + j * scores_s3] =
            from_float_device<scalar_t>(out);
    }
}

torch::Tensor calibrated_qk_q1_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor logc
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(logc.is_cuda(), "logc must be CUDA tensor");

    TORCH_CHECK(q.dim() == 4, "q must be [B,H,1,D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B,H,K,D]");
    TORCH_CHECK(logc.dim() == 4, "logc must be [B,H,1,Kc]");

    const int64_t B64 = q.size(0);
    const int64_t H64 = q.size(1);
    const int64_t Q64 = q.size(2);
    const int64_t D64 = q.size(3);
    const int64_t K64 = k.size(2);

    TORCH_CHECK(Q64 == 1, "calibrated_qk_q1 only supports q_len=1");
    TORCH_CHECK(k.size(0) == B64 && k.size(1) == H64, "q/k batch or head mismatch");
    TORCH_CHECK(k.size(3) == D64, "q/k head_dim mismatch");
    TORCH_CHECK(logc.size(0) == B64 && logc.size(1) == H64 && logc.size(2) == 1, "logc shape mismatch");
    TORCH_CHECK(D64 <= 256, "head_dim > 256 not supported");
    TORCH_CHECK(K64 <= 16384, "K length > 16384 not supported");

    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == logc.scalar_type(), "q/logc dtype mismatch");

    const int B = static_cast<int>(B64);
    const int H = static_cast<int>(H64);
    const int KLEN = static_cast<int>(K64);
    const int LOGC_LEN = static_cast<int>(logc.size(3));
    const int D = static_cast<int>(D64);

    auto scores = torch::empty({B64, H64, 1, K64}, q.options());

    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    const dim3 blocks(B * H, (KLEN + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const float scale = rsqrtf(static_cast<float>(D));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "calibrated_qk_q1_cuda",
        [&] {
            calibrated_qk_q1_kernel<scalar_t, WARPS_PER_BLOCK><<<blocks, THREADS, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                logc.data_ptr<scalar_t>(),
                scores.data_ptr<scalar_t>(),
                H,
                KLEN,
                LOGC_LEN,
                D,
                scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                logc.stride(0), logc.stride(1), logc.stride(2), logc.stride(3),
                scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3)
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}
