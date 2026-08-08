#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include "mymethod_cuda.h"

namespace {

constexpr int MM_FAST_MAX_POOL = 256;
constexpr int MM_FAST_WORDS = MM_FAST_MAX_POOL / 64;
constexpr float MM_CERT_MARGIN_REL = 8.0f * 1.1920928955078125e-7f;

template <typename scalar_t>
__device__ __forceinline__ float mm_float(scalar_t x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ uint32_t mm_hash32(int64_t x) {
    const uint32_t lo = static_cast<uint32_t>(static_cast<uint64_t>(x) & 0xffffffffULL);
    return lo * 2654435761u + 1013904223u;
}

__device__ __forceinline__ void mm_atomic_max_nonnegative(float* addr, float value) {
    // All certificate values are finite and non-negative, so IEEE754 bit order
    // is monotone and integer atomicMax is valid.
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
}

template <typename scalar_t>
__device__ __forceinline__ float mm_pair_dist2(
    const scalar_t* __restrict__ keys,
    int flat_i,
    int flat_j,
    int D) {
    const int64_t oi = static_cast<int64_t>(flat_i) * D;
    const int64_t oj = static_cast<int64_t>(flat_j) * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        const float diff = mm_float(keys[oi + d]) - mm_float(keys[oj + d]);
        acc = fmaf(diff, diff, acc);
    }
    return fmaxf(acc, 0.0f);
}

template <typename scalar_t>
__device__ __forceinline__ bool mm_conflict_pair(
    const scalar_t* __restrict__ keys,
    const float* __restrict__ rhat,
    const float* __restrict__ diam,
    int flat_i,
    int flat_j,
    int D,
    float delta,
    float tol,
    float* dist2_out = nullptr) {
    const float di = diam[flat_i];
    const float dj = diam[flat_j];
    if (di > delta || dj > delta) {
        if (dist2_out) *dist2_out = 0.0f;
        return true;
    }
    const float margin = delta - rhat[flat_i] - rhat[flat_j] - tol;
    if (margin < 0.0f) {
        if (dist2_out) *dist2_out = 0.0f;
        return true;
    }
    const float dist2 = mm_pair_dist2(keys, flat_i, flat_j, D);
    if (dist2_out) *dist2_out = dist2;
    return dist2 > margin * margin;
}

__device__ __forceinline__ bool mm_priority_better(
    int j,
    int i,
    const int* __restrict__ degree,
    const float* __restrict__ gamma,
    const uint32_t* __restrict__ hashv) {
    if (degree[j] != degree[i]) return degree[j] < degree[i];
    if (gamma[j] != gamma[i]) return gamma[j] < gamma[i];
    return hashv[j] < hashv[i];
}

__global__ void mm_count_groups_kernel(
    const int64_t* __restrict__ best_idx,
    const bool* __restrict__ compatible,
    int32_t* __restrict__ group_counts,
    int64_t total_candidates,
    int N,
    int keep_len) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_candidates || !compatible[idx]) return;
    const int row = static_cast<int>(idx / N);
    const int anchor = static_cast<int>(best_idx[idx]);
    if (anchor >= 0 && anchor < keep_len) {
        atomicAdd(&group_counts[row * keep_len + anchor], 1);
    }
}

__global__ void mm_pack_candidates_kernel(
    const int64_t* __restrict__ best_idx,
    const bool* __restrict__ compatible,
    const int64_t* __restrict__ offsets,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ packed_candidates,
    int64_t total_candidates,
    int N,
    int keep_len) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_candidates || !compatible[idx]) return;
    const int row = static_cast<int>(idx / N);
    const int anchor = static_cast<int>(best_idx[idx]);
    if (anchor < 0 || anchor >= keep_len) return;
    const int group = row * keep_len + anchor;
    const int slot = atomicAdd(&cursors[group], 1);
    packed_candidates[offsets[group] + slot] = static_cast<int32_t>(idx);
}

template <typename scalar_t>
__global__ void mm_fast_graph_mis_kernel(
    const scalar_t* __restrict__ keys,
    const float* __restrict__ rhat,
    const float* __restrict__ diam,
    const int64_t* __restrict__ physical_idx,
    const float* __restrict__ anchor_gamma,
    const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ group_counts,
    const float* __restrict__ delta_by_row,
    bool* __restrict__ accepted,
    float* __restrict__ group_gamma,
    int32_t* __restrict__ selected_counts,
    int32_t* __restrict__ conflict_edges,
    int32_t* __restrict__ rounds_out,
    int groups,
    int keep_len,
    int D,
    int group_mode) {
    const int group = blockIdx.x;
    if (group >= groups) return;
    const int m = group_counts[group];
    if (m <= 0 || m > MM_FAST_MAX_POOL) return;
    const int tid = threadIdx.x;
    const int64_t start = offsets[group];
    const int row = group / keep_len;
    const float delta = delta_by_row[row];
    const float tol = MM_CERT_MARGIN_REL * fmaxf(fabsf(delta), 1.0f);

    __shared__ unsigned long long adj[MM_FAST_MAX_POOL * MM_FAST_WORDS];
    __shared__ int degree[MM_FAST_MAX_POOL];
    __shared__ float gamma_s[MM_FAST_MAX_POOL];
    __shared__ uint32_t hash_s[MM_FAST_MAX_POOL];
    __shared__ int32_t flat_s[MM_FAST_MAX_POOL];
    __shared__ uint8_t active[MM_FAST_MAX_POOL];
    __shared__ uint8_t winner[MM_FAST_MAX_POOL];
    __shared__ uint8_t selected[MM_FAST_MAX_POOL];
    __shared__ int continue_flag;
    __shared__ int actual_rounds;
    __shared__ int edge_degree_sum;
    __shared__ float gamma_max_s;
    __shared__ int selected_sum_s;

    for (int t = tid; t < MM_FAST_MAX_POOL * MM_FAST_WORDS; t += blockDim.x) adj[t] = 0ULL;
    if (tid == 0) {
        continue_flag = 1;
        actual_rounds = 0;
        edge_degree_sum = 0;
        gamma_max_s = 0.0f;
        selected_sum_s = 0;
    }
    if (tid < m) {
        const int flat = candidates[start + tid];
        flat_s[tid] = flat;
        gamma_s[tid] = anchor_gamma[flat];
        hash_s[tid] = mm_hash32(physical_idx[flat]);
        active[tid] = 1;
        winner[tid] = 0;
        selected[tid] = 0;
    }
    __syncthreads();

    // Each candidate thread owns one adjacency row.  This intentionally
    // recomputes symmetric distances, trading a small amount of arithmetic for
    // zero atomics and a compact shared-memory bitset.
    if (tid < m) {
        const int fi = flat_s[tid];
        int deg = 0;
        for (int j = 0; j < m; ++j) {
            if (j == tid) continue;
            const bool conflict = mm_conflict_pair(
                keys, rhat, diam, fi, flat_s[j], D, delta, tol, nullptr);
            if (conflict) {
                adj[tid * MM_FAST_WORDS + (j >> 6)] |= (1ULL << (j & 63));
                ++deg;
            }
        }
        degree[tid] = deg;
        atomicAdd(&edge_degree_sum, deg);
    }
    __syncthreads();

    if (group_mode == 1) {  // anchor_only
        if (tid < m) selected[tid] = 1;
        if (tid == 0) actual_rounds = 1;
        __syncthreads();
    } else {
        for (int round = 1; round <= m; ++round) {
            if (tid < m) {
                bool blocked = false;
                if (active[tid]) {
                    for (int w = 0; w < MM_FAST_WORDS && !blocked; ++w) {
                        unsigned long long bits = adj[tid * MM_FAST_WORDS + w];
                        while (bits) {
                            const int b = __ffsll(static_cast<long long>(bits)) - 1;
                            const int j = (w << 6) + b;
                            bits &= bits - 1ULL;
                            if (j < m && active[j] && mm_priority_better(j, tid, degree, gamma_s, hash_s)) {
                                blocked = true;
                                break;
                            }
                        }
                    }
                }
                winner[tid] = static_cast<uint8_t>(active[tid] && !blocked);
            }
            __syncthreads();

            if (tid < m && active[tid]) {
                bool remove = winner[tid] != 0;
                if (winner[tid]) selected[tid] = 1;
                if (!remove) {
                    for (int w = 0; w < MM_FAST_WORDS && !remove; ++w) {
                        unsigned long long bits = adj[tid * MM_FAST_WORDS + w];
                        while (bits) {
                            const int b = __ffsll(static_cast<long long>(bits)) - 1;
                            const int j = (w << 6) + b;
                            bits &= bits - 1ULL;
                            if (j < m && winner[j]) {
                                remove = true;
                                break;
                            }
                        }
                    }
                }
                if (remove) active[tid] = 0;
            }
            __syncthreads();
            if (tid == 0) {
                actual_rounds = round;
                int any = 0;
                for (int i = 0; i < m; ++i) any |= active[i];
                continue_flag = any;
            }
            __syncthreads();
            if (!continue_flag) break;
        }
    }

    if (tid < m && selected[tid]) {
        accepted[flat_s[tid]] = true;
        atomicAdd(&selected_sum_s, 1);
        mm_atomic_max_nonnegative(&gamma_max_s, gamma_s[tid]);
    }
    __syncthreads();

    // Recompute only selected-selected Euclidean distances for the exact Step-6
    // group certificate; the full FP32 pair matrix is never materialized.
    if (tid < m && selected[tid]) {
        const int fi = flat_s[tid];
        for (int j = tid + 1; j < m; ++j) {
            if (!selected[j]) continue;
            const int fj = flat_s[j];
            const float dist = sqrtf(mm_pair_dist2(keys, fi, fj, D));
            const float b = rhat[fi] + dist + rhat[fj];
            const float g = fmaxf(fmaxf(diam[fi], diam[fj]), b);
            mm_atomic_max_nonnegative(&gamma_max_s, g);
        }
    }
    __syncthreads();
    if (tid == 0) {
        group_gamma[group] = gamma_max_s;
        selected_counts[group] = selected_sum_s;
        conflict_edges[group] = edge_degree_sum / 2;
        rounds_out[group] = actual_rounds;
    }
}

template <typename scalar_t>
__global__ void mm_large_graph_mis_kernel(
    const scalar_t* __restrict__ keys,
    const float* __restrict__ rhat,
    const float* __restrict__ diam,
    const int64_t* __restrict__ physical_idx,
    const float* __restrict__ anchor_gamma,
    const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ group_counts,
    const float* __restrict__ delta_by_row,
    bool* __restrict__ accepted,
    float* __restrict__ group_gamma,
    int32_t* __restrict__ selected_counts,
    int32_t* __restrict__ conflict_edges,
    int32_t* __restrict__ rounds_out,
    int32_t* __restrict__ degree_tmp,
    uint8_t* __restrict__ active_tmp,
    uint8_t* __restrict__ winner_tmp,
    uint8_t* __restrict__ selected_tmp,
    int groups,
    int keep_len,
    int D,
    int group_mode) {
    const int group = blockIdx.x;
    if (group >= groups) return;
    const int m = group_counts[group];
    if (m <= MM_FAST_MAX_POOL) return;
    const int tid = threadIdx.x;
    const int64_t start = offsets[group];
    const int row = group / keep_len;
    const float delta = delta_by_row[row];
    const float tol = MM_CERT_MARGIN_REL * fmaxf(fabsf(delta), 1.0f);
    __shared__ int continue_flag;
    __shared__ int actual_rounds;
    __shared__ int edge_degree_sum;
    __shared__ float gamma_max_s;
    __shared__ int selected_sum_s;
    if (tid == 0) {
        continue_flag = 1;
        actual_rounds = 0;
        edge_degree_sum = 0;
        gamma_max_s = 0.0f;
        selected_sum_s = 0;
    }

    // Exact no-cap path.  It deliberately avoids an O(m^2) global adjacency
    // allocation and recomputes conflicts when needed.  This is slower than the
    // <=256 bitset fast path but preserves the exact method for arbitrarily large
    // pools instead of imposing an algorithmic candidate cap.
    for (int i = tid; i < m; i += blockDim.x) {
        const int fi = candidates[start + i];
        int deg = 0;
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            const int fj = candidates[start + j];
            if (mm_conflict_pair(keys, rhat, diam, fi, fj, D, delta, tol, nullptr)) ++deg;
        }
        degree_tmp[start + i] = deg;
        active_tmp[start + i] = 1;
        winner_tmp[start + i] = 0;
        selected_tmp[start + i] = 0;
        atomicAdd(&edge_degree_sum, deg);
    }
    __syncthreads();

    if (group_mode == 1) {
        for (int i = tid; i < m; i += blockDim.x) selected_tmp[start + i] = 1;
        if (tid == 0) actual_rounds = 1;
        __syncthreads();
    } else {
        for (int round = 1; round <= m; ++round) {
            for (int i = tid; i < m; i += blockDim.x) {
                bool blocked = false;
                if (active_tmp[start + i]) {
                    const int fi = candidates[start + i];
                    const int di = degree_tmp[start + i];
                    const float gi = anchor_gamma[fi];
                    const uint32_t hi = mm_hash32(physical_idx[fi]);
                    for (int j = 0; j < m; ++j) {
                        if (j == i || !active_tmp[start + j]) continue;
                        const int fj = candidates[start + j];
                        if (!mm_conflict_pair(keys, rhat, diam, fi, fj, D, delta, tol, nullptr)) continue;
                        const int dj = degree_tmp[start + j];
                        const float gj = anchor_gamma[fj];
                        const uint32_t hj = mm_hash32(physical_idx[fj]);
                        const bool better = (dj < di) || (dj == di && gj < gi) ||
                                            (dj == di && gj == gi && hj < hi);
                        if (better) { blocked = true; break; }
                    }
                }
                winner_tmp[start + i] = static_cast<uint8_t>(active_tmp[start + i] && !blocked);
            }
            __syncthreads();
            for (int i = tid; i < m; i += blockDim.x) {
                if (!active_tmp[start + i]) continue;
                bool remove = winner_tmp[start + i] != 0;
                if (winner_tmp[start + i]) selected_tmp[start + i] = 1;
                if (!remove) {
                    const int fi = candidates[start + i];
                    for (int j = 0; j < m; ++j) {
                        if (!winner_tmp[start + j] || j == i) continue;
                        const int fj = candidates[start + j];
                        if (mm_conflict_pair(keys, rhat, diam, fi, fj, D, delta, tol, nullptr)) {
                            remove = true;
                            break;
                        }
                    }
                }
                if (remove) active_tmp[start + i] = 0;
            }
            __syncthreads();
            if (tid == 0) {
                actual_rounds = round;
                int any = 0;
                for (int i = 0; i < m; ++i) any |= active_tmp[start + i];
                continue_flag = any;
            }
            __syncthreads();
            if (!continue_flag) break;
        }
    }

    for (int i = tid; i < m; i += blockDim.x) {
        if (!selected_tmp[start + i]) continue;
        const int fi = candidates[start + i];
        accepted[fi] = true;
        atomicAdd(&selected_sum_s, 1);
        mm_atomic_max_nonnegative(&gamma_max_s, anchor_gamma[fi]);
        for (int j = i + 1; j < m; ++j) {
            if (!selected_tmp[start + j]) continue;
            const int fj = candidates[start + j];
            const float dist = sqrtf(mm_pair_dist2(keys, fi, fj, D));
            const float b = rhat[fi] + dist + rhat[fj];
            const float g = fmaxf(fmaxf(diam[fi], diam[fj]), b);
            mm_atomic_max_nonnegative(&gamma_max_s, g);
        }
    }
    __syncthreads();
    if (tid == 0) {
        group_gamma[group] = gamma_max_s;
        selected_counts[group] = selected_sum_s;
        conflict_edges[group] = edge_degree_sum / 2;
        rounds_out[group] = actual_rounds;
    }
}

} // namespace

std::vector<torch::Tensor> mymethod_graph_mis_select_cuda(
    torch::Tensor evicted_keys,
    torch::Tensor evicted_rhat,
    torch::Tensor evicted_diameter,
    torch::Tensor evicted_physical_idx,
    torch::Tensor best_idx,
    torch::Tensor route_compatible,
    torch::Tensor anchor_gamma,
    int64_t keep_len,
    torch::Tensor delta,
    int64_t group_mode) {
    TORCH_CHECK(evicted_keys.is_cuda(), "evicted_keys must be CUDA");
    TORCH_CHECK(evicted_rhat.is_cuda() && evicted_diameter.is_cuda() && evicted_physical_idx.is_cuda(),
                "mymethod state tensors must be CUDA");
    TORCH_CHECK(best_idx.is_cuda() && route_compatible.is_cuda() && anchor_gamma.is_cuda() && delta.is_cuda(),
                "routing tensors must be CUDA");
    TORCH_CHECK(evicted_keys.dim() == 3, "evicted_keys must be [X,N,D]");
    const int64_t X64 = evicted_keys.size(0), N64 = evicted_keys.size(1), D64 = evicted_keys.size(2);
    TORCH_CHECK(evicted_rhat.dim()==2 && evicted_rhat.size(0)==X64 && evicted_rhat.size(1)==N64, "evicted_rhat shape mismatch");
    TORCH_CHECK(evicted_diameter.dim()==2 && evicted_diameter.size(0)==X64 && evicted_diameter.size(1)==N64, "evicted_diameter shape mismatch");
    TORCH_CHECK(evicted_physical_idx.dim()==2 && evicted_physical_idx.size(0)==X64 && evicted_physical_idx.size(1)==N64, "physical index shape mismatch");
    TORCH_CHECK(best_idx.dim()==2 && best_idx.size(0)==X64 && best_idx.size(1)==N64, "best_idx shape mismatch");
    TORCH_CHECK(route_compatible.dim()==2 && route_compatible.size(0)==X64 && route_compatible.size(1)==N64, "route_compatible shape mismatch");
    TORCH_CHECK(anchor_gamma.dim()==2 && anchor_gamma.size(0)==X64 && anchor_gamma.size(1)==N64, "anchor_gamma shape mismatch");
    TORCH_CHECK(delta.numel() == X64, "delta must contain one scalar per flattened row");
    TORCH_CHECK(keep_len > 0 && keep_len <= 2147483647LL, "invalid keep_len");
    TORCH_CHECK(group_mode == 0 || group_mode == 1, "group_mode must be 0(full) or 1(anchor_only)");
    TORCH_CHECK(N64 <= 2147483647LL && D64 <= 2147483647LL && X64 * N64 <= 2147483647LL,
                "candidate dimensions exceed int32 packed-index range");
    TORCH_CHECK(best_idx.scalar_type() == torch::kInt64, "best_idx must be int64");
    TORCH_CHECK(evicted_physical_idx.scalar_type() == torch::kInt64, "physical indices must be int64");
    TORCH_CHECK(route_compatible.scalar_type() == torch::kBool, "route_compatible must be bool");

    evicted_keys = evicted_keys.contiguous();
    evicted_rhat = evicted_rhat.to(torch::kFloat32).contiguous();
    evicted_diameter = evicted_diameter.to(torch::kFloat32).contiguous();
    evicted_physical_idx = evicted_physical_idx.contiguous();
    best_idx = best_idx.contiguous();
    route_compatible = route_compatible.contiguous();
    anchor_gamma = anchor_gamma.to(torch::kFloat32).contiguous();
    delta = delta.to(torch::kFloat32).reshape({X64}).contiguous();

    const int X = static_cast<int>(X64), N = static_cast<int>(N64), D = static_cast<int>(D64);
    const int M = static_cast<int>(keep_len);
    const int64_t groups64 = X64 * keep_len;
    TORCH_CHECK(groups64 <= 2147483647LL, "X*keep_len exceeds CUDA grid int32 range");
    const int groups = static_cast<int>(groups64);
    const int64_t total64 = X64 * N64;

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(evicted_keys.device());
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(evicted_keys.device());
    auto byte_opts = torch::TensorOptions().dtype(torch::kByte).device(evicted_keys.device());
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(evicted_keys.device());
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(evicted_keys.device());

    auto accepted = torch::zeros({X64, N64}, bool_opts);
    auto group_gamma = torch::zeros({groups64}, float_opts);
    auto group_counts = torch::zeros({groups64}, int_opts);
    auto selected_counts = torch::zeros({groups64}, int_opts);
    auto conflict_edges = torch::zeros({groups64}, int_opts);
    auto rounds = torch::zeros({groups64}, int_opts);
    if (total64 == 0) {
        return {accepted, group_gamma.view({X64, keep_len}), group_counts.view({X64, keep_len}),
                selected_counts.view({X64, keep_len}), conflict_edges.view({X64, keep_len}), rounds.view({X64, keep_len})};
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int THREADS = 256;
    const int blocks_candidates = static_cast<int>((total64 + THREADS - 1) / THREADS);
    mm_count_groups_kernel<<<blocks_candidates, THREADS, 0, stream>>>(
        best_idx.data_ptr<int64_t>(), route_compatible.data_ptr<bool>(), group_counts.data_ptr<int32_t>(),
        total64, N, M);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Entire ragged pack stays on device.  candidates has worst-case X*N slots;
    // offsets delimit the valid compatible prefix per group, so there is no
    // padded [num_groups,max_pool] workspace and no counts.max().item() sync.
    auto offsets = torch::zeros({groups64 + 1}, long_opts);
    auto prefix = torch::cumsum(group_counts.to(torch::kInt64), 0);
    offsets.slice(0, 1, groups64 + 1).copy_(prefix);
    auto cursors = torch::zeros({groups64}, int_opts);
    auto packed = torch::full({total64}, -1, int_opts);
    mm_pack_candidates_kernel<<<blocks_candidates, THREADS, 0, stream>>>(
        best_idx.data_ptr<int64_t>(), route_compatible.data_ptr<bool>(), offsets.data_ptr<int64_t>(),
        cursors.data_ptr<int32_t>(), packed.data_ptr<int32_t>(), total64, N, M);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Temporary state is linear in the number of evicted entries and only used
    // by the exact no-cap path for pools > 256.
    auto degree_tmp = torch::zeros({total64}, int_opts);
    auto active_tmp = torch::zeros({total64}, byte_opts);
    auto winner_tmp = torch::zeros({total64}, byte_opts);
    auto selected_tmp = torch::zeros({total64}, byte_opts);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        evicted_keys.scalar_type(),
        "mymethod_graph_mis_select_cuda",
        [&] {
            mm_fast_graph_mis_kernel<scalar_t><<<groups, MM_FAST_MAX_POOL, 0, stream>>>(
                evicted_keys.data_ptr<scalar_t>(), evicted_rhat.data_ptr<float>(), evicted_diameter.data_ptr<float>(),
                evicted_physical_idx.data_ptr<int64_t>(), anchor_gamma.data_ptr<float>(), offsets.data_ptr<int64_t>(),
                packed.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), delta.data_ptr<float>(),
                accepted.data_ptr<bool>(), group_gamma.data_ptr<float>(), selected_counts.data_ptr<int32_t>(),
                conflict_edges.data_ptr<int32_t>(), rounds.data_ptr<int32_t>(), groups, M, D, static_cast<int>(group_mode));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            mm_large_graph_mis_kernel<scalar_t><<<groups, THREADS, 0, stream>>>(
                evicted_keys.data_ptr<scalar_t>(), evicted_rhat.data_ptr<float>(), evicted_diameter.data_ptr<float>(),
                evicted_physical_idx.data_ptr<int64_t>(), anchor_gamma.data_ptr<float>(), offsets.data_ptr<int64_t>(),
                packed.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), delta.data_ptr<float>(),
                accepted.data_ptr<bool>(), group_gamma.data_ptr<float>(), selected_counts.data_ptr<int32_t>(),
                conflict_edges.data_ptr<int32_t>(), rounds.data_ptr<int32_t>(), degree_tmp.data_ptr<int32_t>(),
                active_tmp.data_ptr<uint8_t>(), winner_tmp.data_ptr<uint8_t>(), selected_tmp.data_ptr<uint8_t>(),
                groups, M, D, static_cast<int>(group_mode));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return {
        accepted,
        group_gamma.view({X64, keep_len}),
        group_counts.view({X64, keep_len}),
        selected_counts.view({X64, keep_len}),
        conflict_edges.view({X64, keep_len}),
        rounds.view({X64, keep_len})
    };
}
