/*
 * libortho - Dual-Manifold GEMM Kernel
 * 
 * Linus: If you need more than 3 levels of indentation, 
 * you're screwed. Fix your algorithm.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <stdint.h>

#include "../include/ortho.h"

// Forward declaration
extern "C" int orth_layer_forward_cuda(
    const orth_layer_t* layer,
    const float* input,
    float* output,
    size_t batch_size
);

/*
 * Unpack INT4 value from packed array
 */
__device__ __forceinline__ int8_t unpack_int4(const uint8_t* packed, int idx) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t byte = packed[byte_idx];
    int8_t val = (byte >> bit_offset) & 0x0F;
    // Sign extend from 4 bits
    if (val & 0x08) {
        val |= 0xF0;
    }
    return val;
}

/*
 * Compute dense INT4 tile (optimized)
 * 
 * Note: For true Tensor Core implementation, use wmma API:
 *   #include <mma.h>
 *   using namespace nvcuda::wmma;
 *   fragment<matrix_a, ...> a_frag;
 *   fragment<matrix_b, ...> b_frag;
 *   fragment<accumulator, ...> c_frag;
 *   mma_sync(c_frag, a_frag, b_frag, c_frag);
 * 
 * For now, optimized SIMD-friendly version.
 */
__device__ float compute_dense_tile(
    const uint8_t* q_weight_packed,
    const float* q_scales,
    const float* input_tile,
    int in_features,
    int out_idx
) {
    float acc = 0.0f;
    float scale = q_scales[out_idx];
    
    // Process in chunks for better cache behavior
    const int CHUNK_SIZE = 16;
    int chunks = in_features / CHUNK_SIZE;
    int remainder = in_features % CHUNK_SIZE;
    
    // Process full chunks
    for (int c = 0; c < chunks; c++) {
        int base = c * CHUNK_SIZE;
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int idx = out_idx * in_features + base + i;
            int8_t w_int = unpack_int4(q_weight_packed, idx);
            float w = (float)w_int * scale;
            acc += input_tile[base + i] * w;
        }
    }
    
    // Process remainder
    int base = chunks * CHUNK_SIZE;
    for (int i = 0; i < remainder; i++) {
        int idx = out_idx * in_features + base + i;
        int8_t w_int = unpack_int4(q_weight_packed, idx);
        float w = (float)w_int * scale;
        acc += input_tile[base + i] * w;
    }
    
    return acc;
}

/*
 * Compute sparse FP16 patch (optimized)
 * 
 * GOOD TASTE: 
 * We don't check "if (is_outlier)" for every element.
 * We iterate through a pre-computed list of "active" indices 
 * for this thread block.
 * 
 * Optimization: Use binary search or sorted indices for better cache behavior.
 * For now, linear search with early exit optimization.
 */
__device__ float compute_sparse_patch(
    const float* ortho_values,
    const uint16_t* ortho_indices,
    const float* input,
    int ortho_count,
    int in_features,
    int out_idx
) {
    float acc = 0.0f;
    
    // Optimized: process indices that match this output row
    // Assuming indices are sorted or grouped by row (should be done offline)
    for (int i = 0; i < ortho_count; i++) {
        uint16_t flat_idx = ortho_indices[i];
        int row = flat_idx / in_features;
        int col = flat_idx % in_features;
        
        // Early exit if we've passed this row (requires sorted indices)
        // For now, check all indices
        if (row == out_idx && col < in_features) {
            acc += ortho_values[i] * input[col];
        }
    }
    
    return acc;
}

/*
 * Dual-stream GEMM kernel
 * 
 * GOOD TASTE: We don't treat "Base" and "Ortho" as two different
 * data flow logics. We treat them as two writes to the same accumulator.
 * First write is bulk (Dense), second write is precise correction (Sparse).
 * No complex branching, just addition.
 */
__global__ void dual_gemm_kernel(
    const int8_t* q_weight,
    const float* q_scales,
    const float* ortho_values,
    const uint16_t* ortho_indices,
    int ortho_count,
    const float* input,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float alpha
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= out_features) {
        return;
    }
    
    const float* input_row = input + batch_idx * in_features;
    
    // 1. Compute Base (Dense INT4)
    // This is standard. Tensor Cores go Brrr.
    // Note: q_weight is packed INT4, so we need to pass the packed pointer
    float acc = compute_dense_tile(
        (const uint8_t*)q_weight,
        q_scales,
        input_row,
        in_features,
        out_idx
    );
    
    // 2. Compute Ortho (Sparse FP16)
    // Only load the sparse patch if alpha is non-zero.
    // Branch prediction handles this easily as it's uniform for the kernel launch.
    if (alpha > 0.0f) {
        acc += alpha * compute_sparse_patch(
            ortho_values,
            ortho_indices,
            input_row,
            ortho_count,
            in_features,
            out_idx
        );
    }
    
    // 3. Store
    output[batch_idx * out_features + out_idx] = acc;
}

/*
 * Forward declarations for Tensor Core functions
 * Implementations in dual_gemm_tensor_core.cu
 */
extern "C" {
    bool check_tensor_core_support();
    int orth_layer_forward_tensor_core(
        const orth_layer_t* layer,
        const float* input,
        float* output,
        size_t batch_size
    );
}

/*
 * Host wrapper for dual GEMM
 * Automatically selects Tensor Core or standard kernel
 */
extern "C" int orth_layer_forward_cuda(
    const orth_layer_t* layer,
    const float* input,
    float* output,
    size_t batch_size
) {
    if (!layer || !input || !output) {
        return -1;
    }
    
    // Try Tensor Core first if available
    // Note: Tensor Core implementation is framework version
    // For production, consider using CUTLASS library
    if (check_tensor_core_support()) {
        // Try Tensor Core version (may fallback if not fully implemented)
        int result = orth_layer_forward_tensor_core(layer, input, output, batch_size);
        if (result == 0) {
            return 0;  // Tensor Core succeeded
        }
        // Fall through to standard kernel if Tensor Core failed
    }
    
    dim3 block_size(1, 32);  // Adjust based on hardware
    dim3 grid_size(batch_size, (layer->base.out_features + block_size.y - 1) / block_size.y);
    
    dual_gemm_kernel<<<grid_size, block_size>>>(
        (const uint8_t*)layer->base.q_weight,  // Changed to uint8_t for packed INT4
        (const float*)layer->base.q_scales,
        layer->ortho.values,
        layer->ortho.indices,
        layer->ortho.count,
        input,
        output,
        batch_size,
        layer->base.in_features,
        layer->base.out_features,
        layer->alpha
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return -1;
    }
    
    return 0;
}

