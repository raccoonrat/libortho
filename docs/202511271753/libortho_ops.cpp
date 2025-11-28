#include <torch/extension.h>
#include "ortho.h"
#include <c10/cuda/CUDAGuard.h>

// Wrapper to call the C/CUDA kernel from Python
// Good Taste: Direct mapping, no unnecessary abstraction layers
torch::Tensor forward_cuda_wrapper(
    torch::Tensor q_weight,           // INT4 packed weights [uint8_t]
    torch::Tensor q_scales,           // Quantization scales [float]
    torch::Tensor ortho_row_ptr,      // CSR row pointers [int32]
    torch::Tensor ortho_col_indices,  // CSR column indices [int32]
    torch::Tensor ortho_values,       // CSR values [float]
    torch::Tensor input,               // Input [batch, in_features]
    float alpha                        // Kill switch parameter
) {
    // Validate inputs
    TORCH_CHECK(q_weight.is_cuda(), "q_weight must be on CUDA");
    TORCH_CHECK(q_scales.is_cuda(), "q_scales must be on CUDA");
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, in_features]");
    
    // Set device guard
    at::cuda::CUDAGuard guard(input.device());
    
    // Get dimensions
    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = q_scales.size(0);
    
    // Allocate output
    auto output = torch::empty({batch_size, out_features}, 
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(input.device()));
    
    // Construct orth_layer_t struct
    // Good Taste: Zero-initialize, then set only what we need
    orth_layer_t layer = {};
    
    // Base component
    layer.base.q_weight = q_weight.data_ptr<uint8_t>();
    layer.base.q_scales = q_scales.data_ptr<float>();
    layer.base.q_bits = 4;
    layer.base.in_features = in_features;
    layer.base.out_features = out_features;
    
    // Ortho component (CSR format)
    if (ortho_values.numel() > 0) {
        TORCH_CHECK(ortho_row_ptr.is_cuda(), "ortho_row_ptr must be on CUDA");
        TORCH_CHECK(ortho_col_indices.is_cuda(), "ortho_col_indices must be on CUDA");
        TORCH_CHECK(ortho_values.is_cuda(), "ortho_values must be on CUDA");
        TORCH_CHECK(ortho_row_ptr.dtype() == torch::kInt32, "ortho_row_ptr must be int32");
        TORCH_CHECK(ortho_col_indices.dtype() == torch::kInt32, "ortho_col_indices must be int32");
        
        layer.ortho.row_ptr = ortho_row_ptr.data_ptr<int32_t>();
        layer.ortho.col_indices = ortho_col_indices.data_ptr<int32_t>();
        layer.ortho.values = ortho_values.data_ptr<float>();
        layer.ortho.count = ortho_values.numel();
        layer.ortho.format = 1;  // CSR format
    } else {
        // Empty ortho (alpha=0 case)
        layer.ortho.row_ptr = nullptr;
        layer.ortho.col_indices = nullptr;
        layer.ortho.values = nullptr;
        layer.ortho.count = 0;
        layer.ortho.format = 1;
    }
    
    // Kill switch
    layer.alpha = alpha;
    
    // Call CUDA kernel
    int result = orth_layer_forward_cuda(
        &layer,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );
    
    TORCH_CHECK(result == 0, "orth_layer_forward_cuda failed with code: ", result);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_wrapper, 
          "LibOrtho Forward (CUDA)\n"
          "Args:\n"
          "  q_weight: INT4 packed weights [uint8_t]\n"
          "  q_scales: Quantization scales [float]\n"
          "  ortho_row_ptr: CSR row pointers [int32]\n"
          "  ortho_col_indices: CSR column indices [int32]\n"
          "  ortho_values: CSR values [float]\n"
          "  input: Input tensor [batch, in_features]\n"
          "  alpha: Kill switch parameter (0.0 = base only, 1.0 = full)\n"
          "Returns:\n"
          "  output: Output tensor [batch, out_features]");
}