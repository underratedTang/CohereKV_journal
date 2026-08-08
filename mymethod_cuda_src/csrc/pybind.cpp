#include <torch/extension.h>
#include "mymethod_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "count_calibrated_qk_q1",
        &mymethod_count_calibrated_qk_q1_cuda,
        "mymethod decode q_len=1: QK/sqrt(d) + log(C) (CUDA)"
    );
    m.def(
        "compatible_route",
        &mymethod_compatible_route_cuda,
        "mymethod fused nearest compatible anchor routing (CUDA)"
    );
    m.def(
        "graph_mis_select",
        &mymethod_graph_mis_select_cuda,
        "mymethod ragged conflict-graph + exact degree-priority parallel MIS (CUDA)"
    );
}
