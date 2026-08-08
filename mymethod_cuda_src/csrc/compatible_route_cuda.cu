#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "mymethod_cuda.h"

namespace {
constexpr float MM_CERT_MARGIN_REL = 8.0f * 1.1920928955078125e-7f;

template <typename scalar_t>
__device__ __forceinline__ float to_f32(scalar_t x) { return static_cast<float>(x); }

template <typename scalar_t>
__global__ void compatible_route_kernel(
    const scalar_t* __restrict__ ek,
    const scalar_t* __restrict__ rk,
    const float* __restrict__ er,
    const float* __restrict__ rr,
    const float* __restrict__ ed,
    const float* __restrict__ rd,
    const float* __restrict__ delta,
    const bool* __restrict__ allowed,
    int64_t* __restrict__ out_idx,
    float* __restrict__ out_dist,
    float* __restrict__ out_gamma,
    bool* __restrict__ out_ok,
    int X, int N, int M, int D) {
    const int64_t ev_global = static_cast<int64_t>(blockIdx.x);
    if (ev_global >= static_cast<int64_t>(X) * N) return;
    const int row = static_cast<int>(ev_global / N);
    const int i = static_cast<int>(ev_global - static_cast<int64_t>(row) * N);
    const int tid = threadIdx.x;
    const float delt = delta[row];
    const float tol = MM_CERT_MARGIN_REL * fmaxf(fabsf(delt), 1.0f);
    const float eri = er[ev_global];
    const float edi = ed[ev_global];

    float local_best = INFINITY;
    int local_idx = 0x7fffffff;
    if (edi <= delt) {
        const int64_t ebase = (static_cast<int64_t>(row) * N + i) * D;
        for (int j = tid; j < M; j += blockDim.x) {
            const int64_t ridx = static_cast<int64_t>(row) * M + j;
            if (!allowed[ridx] || rd[ridx] > delt) continue;
            const float margin = delt - eri - rr[ridx] - tol;
            if (margin < 0.0f) continue;
            const int64_t rbase = ridx * D;
            float dist2 = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float diff = to_f32(ek[ebase + d]) - to_f32(rk[rbase + d]);
                dist2 = fmaf(diff, diff, dist2);
            }
            dist2 = fmaxf(dist2, 0.0f);
            if (dist2 <= margin * margin) {
                if (dist2 < local_best || (dist2 == local_best && j < local_idx)) {
                    local_best = dist2;
                    local_idx = j;
                }
            }
        }
    }

    __shared__ float best_dist[256];
    __shared__ int best_idx[256];
    best_dist[tid] = local_best;
    best_idx[tid] = local_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rhs_d = best_dist[tid + stride];
            const int rhs_i = best_idx[tid + stride];
            const float lhs_d = best_dist[tid];
            const int lhs_i = best_idx[tid];
            if (rhs_d < lhs_d || (rhs_d == lhs_d && rhs_i < lhs_i)) {
                best_dist[tid] = rhs_d;
                best_idx[tid] = rhs_i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const bool ok = isfinite(best_dist[0]) && best_idx[0] != 0x7fffffff;
        out_ok[ev_global] = ok;
        if (!ok) {
            out_idx[ev_global] = 0;
            out_dist[ev_global] = 0.0f;
            out_gamma[ev_global] = 0.0f;
        } else {
            const int j = best_idx[0];
            const int64_t ridx = static_cast<int64_t>(row) * M + j;
            const float dist = sqrtf(best_dist[0]);
            out_idx[ev_global] = j;
            out_dist[ev_global] = dist;
            out_gamma[ev_global] = fmaxf(fmaxf(edi, rd[ridx]), eri + dist + rr[ridx]);
        }
    }
}
} // namespace

std::vector<torch::Tensor> mymethod_compatible_route_cuda(
    torch::Tensor evicted_keys,
    torch::Tensor retained_keys,
    torch::Tensor evicted_rhat,
    torch::Tensor retained_rhat,
    torch::Tensor evicted_diameter,
    torch::Tensor retained_diameter,
    torch::Tensor delta,
    torch::Tensor anchor_allowed) {
    TORCH_CHECK(evicted_keys.is_cuda() && retained_keys.is_cuda(), "keys must be CUDA tensors");
    TORCH_CHECK(evicted_keys.dim() == 3 && retained_keys.dim() == 3, "keys must be [X,N/M,D]");
    TORCH_CHECK(evicted_keys.size(0) == retained_keys.size(0) && evicted_keys.size(2) == retained_keys.size(2), "key shape mismatch");
    TORCH_CHECK(evicted_keys.scalar_type() == retained_keys.scalar_type(), "key dtype mismatch");
    const int64_t X64=evicted_keys.size(0), N64=evicted_keys.size(1), M64=retained_keys.size(1), D64=evicted_keys.size(2);
    TORCH_CHECK(N64 <= 2147483647LL && M64 <= 2147483647LL && D64 <= 2147483647LL && X64*N64 <= 2147483647LL, "dimensions exceed int32 range");
    TORCH_CHECK(evicted_rhat.numel()==X64*N64 && evicted_diameter.numel()==X64*N64, "evicted state shape mismatch");
    TORCH_CHECK(retained_rhat.numel()==X64*M64 && retained_diameter.numel()==X64*M64, "retained state shape mismatch");
    TORCH_CHECK(delta.numel()==X64, "delta must have one value per row");
    TORCH_CHECK(anchor_allowed.numel()==X64*M64 && anchor_allowed.scalar_type()==torch::kBool, "anchor_allowed shape/dtype mismatch");

    evicted_keys=evicted_keys.contiguous(); retained_keys=retained_keys.contiguous();
    evicted_rhat=evicted_rhat.to(torch::kFloat32).contiguous(); retained_rhat=retained_rhat.to(torch::kFloat32).contiguous();
    evicted_diameter=evicted_diameter.to(torch::kFloat32).contiguous(); retained_diameter=retained_diameter.to(torch::kFloat32).contiguous();
    delta=delta.to(torch::kFloat32).reshape({X64}).contiguous(); anchor_allowed=anchor_allowed.contiguous();

    auto long_opts=torch::TensorOptions().dtype(torch::kInt64).device(evicted_keys.device());
    auto float_opts=torch::TensorOptions().dtype(torch::kFloat32).device(evicted_keys.device());
    auto bool_opts=torch::TensorOptions().dtype(torch::kBool).device(evicted_keys.device());
    auto idx=torch::zeros({X64,N64},long_opts); auto dist=torch::zeros({X64,N64},float_opts);
    auto gamma=torch::zeros({X64,N64},float_opts); auto ok=torch::zeros({X64,N64},bool_opts);
    if (N64==0 || M64==0) return {idx,dist,gamma,ok};
    constexpr int THREADS=256;
    auto stream=at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,evicted_keys.scalar_type(),"mymethod_compatible_route_cuda",[&]{
        compatible_route_kernel<scalar_t><<<static_cast<int>(X64*N64),THREADS,0,stream>>>(
            evicted_keys.data_ptr<scalar_t>(),retained_keys.data_ptr<scalar_t>(),evicted_rhat.data_ptr<float>(),retained_rhat.data_ptr<float>(),
            evicted_diameter.data_ptr<float>(),retained_diameter.data_ptr<float>(),delta.data_ptr<float>(),anchor_allowed.data_ptr<bool>(),
            idx.data_ptr<int64_t>(),dist.data_ptr<float>(),gamma.data_ptr<float>(),ok.data_ptr<bool>(),
            static_cast<int>(X64),static_cast<int>(N64),static_cast<int>(M64),static_cast<int>(D64));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {idx,dist,gamma,ok};
}
