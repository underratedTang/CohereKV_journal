#pragma once
#include <torch/extension.h>
#include <vector>

torch::Tensor mymethod_count_calibrated_qk_q1_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor logc);

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
    int64_t group_mode);

std::vector<torch::Tensor> mymethod_compatible_route_cuda(
    torch::Tensor evicted_keys,
    torch::Tensor retained_keys,
    torch::Tensor evicted_rhat,
    torch::Tensor retained_rhat,
    torch::Tensor evicted_diameter,
    torch::Tensor retained_diameter,
    torch::Tensor delta,
    torch::Tensor anchor_allowed);
