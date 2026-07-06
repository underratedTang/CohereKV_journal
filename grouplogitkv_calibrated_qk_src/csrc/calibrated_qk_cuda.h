#pragma once
#include <torch/extension.h>

torch::Tensor calibrated_qk_q1_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor logc
);
