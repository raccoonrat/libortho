/*
 * libortho - PyBind11 Bindings (CSR Updated)
 */

#include <torch/extension.h>
#include "../include/ortho.h"

// Wrapper to call the C/CUDA kernel from Python
void forward_cuda_wrapper(
    torch::Tensor q_weight,
    torch::Tensor q_scales,
    torch::Tensor ortho_values,
    torch::Tensor ortho_col_indices,
    torch::Tensor ortho_row_ptr,
    torch::Tensor input,
    torch::Tensor output,
    float alpha
) {
    orth_layer_t layer = {};
    
    // Setup Base
    // Good Taste: Explicit type casting for clarity
    layer.base.q_weight = q_weight.data_ptr<uint8_t>();
    layer.base.q_scales = q_scales.data_ptr<float>();
    layer.base.in_features = input.size(1);
    layer.base.out_features = output.size(1);
    layer.base.q_bits = 4;
    
    // Setup Ortho (CSR)
    layer.ortho.values = ortho_values.data_ptr<float>();
    layer.ortho.col_indices = ortho_col_indices.data_ptr<int32_t>();
    layer.ortho.row_ptr = ortho_row_ptr.data_ptr<int32_t>();
    layer.ortho.count = ortho_values.numel();
    layer.ortho.format = 1; // CSR
    
    layer.alpha = alpha;
    
    // Call CUDA Kernel
    #ifdef __CUDACC__
    orth_layer_forward_cuda(&layer, (float*)input.data_ptr(), (float*)output.data_ptr(), input.size(0));
    #else
    throw std::runtime_error("LibOrtho compiled without CUDA support");
    #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "libortho C++/CUDA operations";
    m.def("forward", &forward_cuda_wrapper, "LibOrtho Forward (CUDA CSR)");
}