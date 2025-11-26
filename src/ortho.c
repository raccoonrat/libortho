/*
 * libortho - C Implementation (CPU fallback)
 */

#include "../include/ortho.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 128-byte alignment for Tensor Core access
#define ALIGNMENT 128

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#define _POSIX_C_SOURCE 200809L  // For posix_memalign
#endif

int orth_layer_init(orth_layer_t *layer, 
                    size_t in_features, 
                    size_t out_features,
                    int q_bits) {
    if (!layer) {
        return -1;
    }
    
    memset(layer, 0, sizeof(orth_layer_t));
    
    layer->base.in_features = in_features;
    layer->base.out_features = out_features;
    layer->base.q_bits = q_bits;
    
    // Allocate base weights (INT4 packed) with 128-byte alignment
    size_t q_weight_size = (in_features * out_features * q_bits + 7) / 8;
    // Align to 128 bytes for Tensor Core access
    size_t aligned_q_weight_size = (q_weight_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->base.q_weight = _aligned_malloc(aligned_q_weight_size, ALIGNMENT);
#else
    if (posix_memalign(&layer->base.q_weight, ALIGNMENT, aligned_q_weight_size) != 0) {
        return -1;
    }
#endif
    if (!layer->base.q_weight) {
        return -1;
    }
    memset(layer->base.q_weight, 0, aligned_q_weight_size);
    
    // Allocate scales with 128-byte alignment
    size_t scale_size = out_features * sizeof(float);
    size_t aligned_scale_size = (scale_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->base.q_scales = _aligned_malloc(aligned_scale_size, ALIGNMENT);
#else
    if (posix_memalign(&layer->base.q_scales, ALIGNMENT, aligned_scale_size) != 0) {
#ifdef _WIN32
        _aligned_free(layer->base.q_weight);
#else
        free(layer->base.q_weight);
#endif
        return -1;
    }
#endif
    if (!layer->base.q_scales) {
#ifdef _WIN32
        _aligned_free(layer->base.q_weight);
#else
        free(layer->base.q_weight);
#endif
        return -1;
    }
    memset(layer->base.q_scales, 0, aligned_scale_size);
    
    // Initialize ortho (empty)
    // FIXED: Initialize CSR format fields
    layer->ortho.row_ptr = NULL;
    layer->ortho.col_indices = NULL;
    layer->ortho.values = NULL;
    layer->ortho.indices = NULL;  // Legacy COO
    layer->ortho.count = 0;
    layer->ortho.capacity = 0;
    layer->ortho.format = 1;  // Default to CSR format
    
    layer->alpha = 1.0f;
    
    return 0;
}

void orth_layer_free(orth_layer_t *layer) {
    if (!layer) {
        return;
    }
    
#ifdef _WIN32
    _aligned_free(layer->base.q_weight);
    _aligned_free(layer->base.q_scales);
#else
    free(layer->base.q_weight);
    free(layer->base.q_scales);
#endif
    
    // Free ortho using aligned free
    orth_layer_free_ortho(layer);
    
    memset(layer, 0, sizeof(orth_layer_t));
}

void orth_layer_set_alpha(orth_layer_t *layer, float alpha) {
    if (layer) {
        layer->alpha = alpha;
    }
}

int orth_layer_alloc_ortho(orth_layer_t *layer, size_t count) {
    // Legacy COO format allocation (deprecated)
    // For new code, use orth_layer_alloc_ortho_csr instead
    if (!layer || count == 0) {
        return -1;
    }
    
    // Free existing ortho if any
    orth_layer_free_ortho(layer);
    
    // Allocate indices with 128-byte alignment (COO format)
    size_t indices_size = count * sizeof(uint16_t);
    size_t aligned_indices_size = (indices_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->ortho.indices = (uint16_t*)_aligned_malloc(aligned_indices_size, ALIGNMENT);
#else
    if (posix_memalign((void**)&layer->ortho.indices, ALIGNMENT, aligned_indices_size) != 0) {
        return -1;
    }
#endif
    if (!layer->ortho.indices) {
        return -1;
    }
    memset(layer->ortho.indices, 0, aligned_indices_size);
    
    // Allocate values with 128-byte alignment
    size_t values_size = count * sizeof(float);
    size_t aligned_values_size = (values_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->ortho.values = (float*)_aligned_malloc(aligned_values_size, ALIGNMENT);
#else
    if (posix_memalign((void**)&layer->ortho.values, ALIGNMENT, aligned_values_size) != 0) {
#ifdef _WIN32
        _aligned_free(layer->ortho.indices);
#else
        free(layer->ortho.indices);
#endif
        return -1;
    }
#endif
    if (!layer->ortho.values) {
#ifdef _WIN32
        _aligned_free(layer->ortho.indices);
#else
        free(layer->ortho.indices);
#endif
        return -1;
    }
    memset(layer->ortho.values, 0, aligned_values_size);
    
    layer->ortho.count = count;
    layer->ortho.capacity = count;
    layer->ortho.format = 0;  // COO format
    
    return 0;
}

int orth_layer_alloc_ortho_csr(orth_layer_t *layer, size_t nnz, size_t out_features) {
    // FIXED: Allocate CSR format buffers for O(1) row access
    // This enables fast CUDA kernels without warp divergence
    if (!layer || nnz == 0 || out_features == 0) {
        return -1;
    }
    
    // Free existing ortho if any
    orth_layer_free_ortho(layer);
    
    // Allocate row pointers: [out_features + 1] int32_t
    size_t row_ptr_size = (out_features + 1) * sizeof(int32_t);
    size_t aligned_row_ptr_size = (row_ptr_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->ortho.row_ptr = (int32_t*)_aligned_malloc(aligned_row_ptr_size, ALIGNMENT);
#else
    if (posix_memalign((void**)&layer->ortho.row_ptr, ALIGNMENT, aligned_row_ptr_size) != 0) {
        return -1;
    }
#endif
    if (!layer->ortho.row_ptr) {
        return -1;
    }
    memset(layer->ortho.row_ptr, 0, aligned_row_ptr_size);
    
    // Allocate column indices: [nnz] int32_t
    size_t col_indices_size = nnz * sizeof(int32_t);
    size_t aligned_col_indices_size = (col_indices_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->ortho.col_indices = (int32_t*)_aligned_malloc(aligned_col_indices_size, ALIGNMENT);
#else
    if (posix_memalign((void**)&layer->ortho.col_indices, ALIGNMENT, aligned_col_indices_size) != 0) {
#ifdef _WIN32
        _aligned_free(layer->ortho.row_ptr);
#else
        free(layer->ortho.row_ptr);
#endif
        return -1;
    }
#endif
    if (!layer->ortho.col_indices) {
#ifdef _WIN32
        _aligned_free(layer->ortho.row_ptr);
#else
        free(layer->ortho.row_ptr);
#endif
        return -1;
    }
    memset(layer->ortho.col_indices, 0, aligned_col_indices_size);
    
    // Allocate values: [nnz] float
    size_t values_size = nnz * sizeof(float);
    size_t aligned_values_size = (values_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
#ifdef _WIN32
    layer->ortho.values = (float*)_aligned_malloc(aligned_values_size, ALIGNMENT);
#else
    if (posix_memalign((void**)&layer->ortho.values, ALIGNMENT, aligned_values_size) != 0) {
#ifdef _WIN32
        _aligned_free(layer->ortho.row_ptr);
        _aligned_free(layer->ortho.col_indices);
#else
        free(layer->ortho.row_ptr);
        free(layer->ortho.col_indices);
#endif
        return -1;
    }
#endif
    if (!layer->ortho.values) {
#ifdef _WIN32
        _aligned_free(layer->ortho.row_ptr);
        _aligned_free(layer->ortho.col_indices);
#else
        free(layer->ortho.row_ptr);
        free(layer->ortho.col_indices);
#endif
        return -1;
    }
    memset(layer->ortho.values, 0, aligned_values_size);
    
    layer->ortho.count = nnz;
    layer->ortho.capacity = nnz;
    layer->ortho.format = 1;  // CSR format
    
    return 0;
}

void orth_layer_free_ortho(orth_layer_t *layer) {
    if (!layer) {
        return;
    }
    
    // Free CSR format buffers
    if (layer->ortho.row_ptr) {
#ifdef _WIN32
        _aligned_free(layer->ortho.row_ptr);
#else
        free(layer->ortho.row_ptr);
#endif
        layer->ortho.row_ptr = NULL;
    }
    
    if (layer->ortho.col_indices) {
#ifdef _WIN32
        _aligned_free(layer->ortho.col_indices);
#else
        free(layer->ortho.col_indices);
#endif
        layer->ortho.col_indices = NULL;
    }
    
    // Free legacy COO format buffers
    if (layer->ortho.indices) {
#ifdef _WIN32
        _aligned_free(layer->ortho.indices);
#else
        free(layer->ortho.indices);
#endif
        layer->ortho.indices = NULL;
    }
    
    if (layer->ortho.values) {
#ifdef _WIN32
        _aligned_free(layer->ortho.values);
#else
        free(layer->ortho.values);
#endif
        layer->ortho.values = NULL;
    }
    
    layer->ortho.count = 0;
    layer->ortho.capacity = 0;
    layer->ortho.format = 0;
}

// Helper: Unpack INT4 value from packed array
// Force inline for performance (critical path)
static inline __attribute__((always_inline)) int8_t unpack_int4(const uint8_t *packed, size_t idx) {
    size_t byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t byte = packed[byte_idx];
    int8_t val = (byte >> bit_offset) & 0x0F;
    // Sign extend from 4 bits
    if (val & 0x08) {
        val |= 0xF0;  // Sign extend
    }
    return val;
}

int orth_layer_forward(const orth_layer_t *layer,
                       const float *input,
                       float *output,
                       size_t batch_size) {
    if (!layer || !input || !output) {
        return -1;
    }
    
    size_t in_features = layer->base.in_features;
    size_t out_features = layer->base.out_features;
    int q_bits = layer->base.q_bits;
    const uint8_t *q_weight = (const uint8_t *)layer->base.q_weight;
    const float *q_scales = (const float *)layer->base.q_scales;
    
    // Fast path: No Ortho contribution (alpha=0 or empty ortho)
    // This should be COMPLETELY EQUAL to reference implementation
    int has_ortho = (layer->alpha > 0.0f && layer->ortho.count > 0);
    
    // Process each batch
    for (size_t b = 0; b < batch_size; b++) {
        const float *x = input + b * in_features;
        float *y = output + b * out_features;
        
        // Compute Base: Y = X @ W_base (INT4 quantized)
        // Optimized: assume q_bits == 4 (most common case)
        if (q_bits == 4) {
            // Fast path: INT4 without branch in inner loop
            for (size_t out = 0; out < out_features; out++) {
                float acc = 0.0f;
                float scale = q_scales[out];
                size_t weight_base = out * in_features;
                
                // Unroll-friendly loop
                for (size_t in = 0; in < in_features; in++) {
                    size_t idx = weight_base + in;
                    int8_t w_int = unpack_int4(q_weight, idx);
                    acc += x[in] * ((float)w_int * scale);
                }
                y[out] = acc;
            }
        } else {
            // Fallback for other bit widths
            for (size_t out = 0; out < out_features; out++) {
                float acc = 0.0f;
                float scale = q_scales[out];
                
                for (size_t in = 0; in < in_features; in++) {
                    int byte_idx = (out * in_features + in) * q_bits / 8;
                    int8_t w_int = ((int8_t *)q_weight)[byte_idx];
                    acc += x[in] * ((float)w_int * scale);
                }
                y[out] = acc;
            }
        }
        
        // Compute Ortho: Y += alpha * (X @ W_ortho) (sparse)
        // Only if needed (branch outside inner loops)
        if (has_ortho) {
            const float *ortho_values = layer->ortho.values;
            const uint16_t *ortho_indices = layer->ortho.indices;
            float alpha = layer->alpha;
            
            for (int i = 0; i < layer->ortho.count; i++) {
                uint16_t flat_idx = ortho_indices[i];
                size_t row = flat_idx / in_features;
                size_t col = flat_idx % in_features;
                
                if (row < out_features && col < in_features) {
                    y[row] += alpha * ortho_values[i] * x[col];
                }
            }
        }
    }
    
    return 0;
}

