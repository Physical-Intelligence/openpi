#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Add value to tensor (CUDA)");
    m.def("fused_bias_gelu", &fused_bias_gelu_cuda, "Fused Bias + GELU activation (CUDA)",
          py::arg("input"), py::arg("bias"));
    m.def("fused_geglu", &fused_geglu_cuda, "Fused GeGLU: GELU(gate) * up (CUDA)", py::arg("gate"),
          py::arg("up"));
    m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm: Root Mean Square Layer Normalization (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6);
    m.def("fused_add_layernorm", &fused_add_layernorm_cuda, "Fused Residual Add + LayerNorm (CUDA)",
          py::arg("x"), py::arg("residual"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-6);
}