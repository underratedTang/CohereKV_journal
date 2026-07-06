#include <torch/extension.h>
#include "calibrated_qk_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "calibrated_qk_q1",
        &calibrated_qk_q1_cuda,
        "GroupLogitKV calibrated QK decode q_len=1: QK*scale + logC"
    );
}
