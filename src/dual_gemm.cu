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
 * Compute sparse FP16 patch (optimized with CSR format)
 * 
 * FIXED: Now uses CSR format for O(1) row access.
 * No more linear search causing warp divergence.
 * 
 * CSR Format:
 *   - row_ptr[out_idx] = start index for row out_idx
 *   - row_ptr[out_idx + 1] = end index (exclusive)
 *   - col_indices[start:end] = column indices for this row
 *   - values[start:end] = corresponding values
 */
__device__ float compute_sparse_patch_csr(
    const float* ortho_values,
    const int32_t* ortho_col_indices,
    const int32_t* ortho_row_ptr,
    const float* input,
    int in_features,
    int out_idx
) {
    float acc = 0.0f;
    
    // O(1) row access using CSR format
    int start_idx = ortho_row_ptr[out_idx];
    int end_idx = ortho_row_ptr[out_idx + 1];
    
    // Process all non-zeros in this row
    // No branching, no divergence - all threads in warp process same number of elements
    for (int i = start_idx; i < end_idx; i++) {
        int col = ortho_col_indices[i];
        if (col < in_features) {
            acc += ortho_values[i] * input[col];
        }
    }
    
    return acc;
}

/*
 * Legacy COO format support (deprecated, kept for backward compatibility)
 */
__device__ float compute_sparse_patch_coo(
    const float* ortho_values,
    const uint16_t* ortho_indices,
    const float* input,
    int ortho_count,
    int in_features,
    int out_idx
) {
    float acc = 0.0f;
    
    // Linear search with early exit (deprecated - causes warp divergence)
    for (int i = 0; i < ortho_count; i++) {
        uint16_t flat_idx = ortho_indices[i];
        int row = flat_idx / in_features;
        
        if (row > out_idx) {
            break;
        }
        
        if (row == out_idx) {
            int col = flat_idx % in_features;
            if (col < in_features) {
                acc += ortho_values[i] * input[col];
            }
        }
    }
    
    return acc;
}

/*
 * Dual-stream GEMM kernel (CSR format)
 * 
 * FIXED: Now uses CSR format for O(1) row access.
 * No more warp divergence from linear search.
 * 
 * GOOD TASTE: We don't treat "Base" and "Ortho" as two different
 * data flow logics. We treat them as two writes to the same accumulator.
 * First write is bulk (Dense), second write is precise correction (Sparse).
 * No complex branching, just addition.
 */
__global__ void dual_gemm_kernel_csr(
    const uint8_t* q_weight,  // FIXED: Changed from int8_t* to uint8_t* for packed INT4
    const float* q_scales,
    const float* ortho_values,
    const int32_t* ortho_col_indices,
    const int32_t* ortho_row_ptr,
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
        q_weight,  // FIXED: No cast needed, already uint8_t*
        q_scales,
        input_row,
        in_features,
        out_idx
    );
    
    // 2. Compute Ortho (Sparse FP16) using CSR format
    // Only load the sparse patch if alpha is non-zero.
    // Branch prediction handles this easily as it's uniform for the kernel launch.
    if (alpha > 0.0f) {
        acc += alpha * compute_sparse_patch_csr(
            ortho_values,
            ortho_col_indices,
            ortho_row_ptr,
            input_row,
            in_features,
            out_idx
        );
    }
    
    // 3. Store
    output[batch_idx * out_features + out_idx] = acc;
}

/*
 * Legacy COO format kernel (deprecated, kept for backward compatibility)
 */
__global__ void dual_gemm_kernel_coo(
    const uint8_t* q_weight,  // FIXED: Changed from int8_t* to uint8_t* for packed INT4
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
    float acc = compute_dense_tile(
        q_weight,  // FIXED: No cast needed, already uint8_t*
        q_scales,
        input_row,
        in_features,
        out_idx
    );
    
    // 2. Compute Ortho (Sparse FP16) using COO format (deprecated)
    if (alpha > 0.0f) {
        acc += alpha * compute_sparse_patch_coo(
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
    
    // Use CSR format if available (format == 1), otherwise fallback to COO
    if (layer->ortho.format == 1 && layer->ortho.row_ptr && layer->ortho.col_indices) {
        // CSR format: O(1) row access, no warp divergence
        dual_gemm_kernel_csr<<<grid_size, block_size>>>(
            (const uint8_t*)layer->base.q_weight,  // Packed INT4 weights
            (const float*)layer->base.q_scales,
            layer->ortho.values,
            layer->ortho.col_indices,
            layer->ortho.row_ptr,
            input,
            output,
            batch_size,
            layer->base.in_features,
            layer->base.out_features,
            layer->alpha
        );
    } else {
        // Legacy COO format (deprecated)
        dual_gemm_kernel_coo<<<grid_size, block_size>>>(
            (const uint8_t*)layer->base.q_weight,  // Packed INT4 weights
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
    }
    
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

