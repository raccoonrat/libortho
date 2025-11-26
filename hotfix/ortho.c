/*
 * libortho - C Implementation (CPU fallback)
 * Linus: Fixed to allocate CSR buffers instead of deprecated COO.
 */

#include "../include/ortho.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// 128-byte alignment for Tensor Core access
#define ALIGNMENT 128

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#define _POSIX_C_SOURCE 200809L  // For posix_memalign
#endif

// Helper for aligned allocation
static void* aligned_alloc_wrapper(size_t size) {
    if (size == 0) return NULL;
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    void* ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, ALIGNMENT);
#else
    if (posix_memalign(&ptr, ALIGNMENT, aligned_size) != 0) return NULL;
#endif
    if (ptr) memset(ptr, 0, aligned_size);
    return ptr;
}

static void aligned_free_wrapper(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

int orth_layer_init(orth_layer_t *layer, 
                    size_t in_features, 
                    size_t out_features,
                    int q_bits) {
    if (!layer) return -1;
    
    memset(layer, 0, sizeof(orth_layer_t));
    
    layer->base.in_features = in_features;
    layer->base.out_features = out_features;
    layer->base.q_bits = q_bits;
    
    // Allocate base weights (INT4 packed)
    size_t q_weight_size = (in_features * out_features * q_bits + 7) / 8;
    layer->base.q_weight = aligned_alloc_wrapper(q_weight_size);
    if (!layer->base.q_weight) return -1;
    
    // Allocate scales
    size_t scale_size = out_features * sizeof(float);
    layer->base.q_scales = aligned_alloc_wrapper(scale_size);
    if (!layer->base.q_scales) {
        aligned_free_wrapper(layer->base.q_weight);
        return -1;
    }
    
    // Initialize ortho
    layer->ortho.row_ptr = NULL;
    layer->ortho.col_indices = NULL;
    layer->ortho.values = NULL;
    layer->ortho.indices = NULL; // Deprecated
    layer->ortho.count = 0;
    layer->ortho.format = 1; // Default to CSR
    
    layer->alpha = 1.0f;
    return 0;
}

void orth_layer_free(orth_layer_t *layer) {
    if (!layer) return;
    
    aligned_free_wrapper(layer->base.q_weight);
    aligned_free_wrapper(layer->base.q_scales);
    orth_layer_free_ortho(layer);
    
    memset(layer, 0, sizeof(orth_layer_t));
}

void orth_layer_set_alpha(orth_layer_t *layer, float alpha) {
    if (layer) layer->alpha = alpha;
}

// FIXED: Allocate CSR buffers instead of COO
int orth_layer_alloc_ortho(orth_layer_t *layer, size_t count) {
    if (!layer) return -1;
    
    orth_layer_free_ortho(layer);
    
    if (count == 0) return 0;
    
    // Set format to CSR
    layer->ortho.format = 1; 
    layer->ortho.count = count;
    
    // 1. Allocate Row Pointers [out_features + 1]
    // Note: row_ptr is int32_t
    size_t rows = layer->base.out_features;
    layer->ortho.row_ptr = (int32_t*)aligned_alloc_wrapper((rows + 1) * sizeof(int32_t));
    
    // 2. Allocate Column Indices [count]
    layer->ortho.col_indices = (int32_t*)aligned_alloc_wrapper(count * sizeof(int32_t));
    
    // 3. Allocate Values [count]
    layer->ortho.values = (float*)aligned_alloc_wrapper(count * sizeof(float));
    
    if (!layer->ortho.row_ptr || !layer->ortho.col_indices || !layer->ortho.values) {
        orth_layer_free_ortho(layer);
        return -1;
    }
    
    return 0;
}

void orth_layer_free_ortho(orth_layer_t *layer) {
    if (!layer) return;
    
    aligned_free_wrapper(layer->ortho.row_ptr);
    aligned_free_wrapper(layer->ortho.col_indices);
    aligned_free_wrapper(layer->ortho.values);
    
    // Also free deprecated legacy buffers if they exist
    aligned_free_wrapper(layer->ortho.indices);
    
    layer->ortho.row_ptr = NULL;
    layer->ortho.col_indices = NULL;
    layer->ortho.values = NULL;
    layer->ortho.indices = NULL;
    layer->ortho.count = 0;
}

// ... existing unpack_int4 ...
static inline __attribute__((always_inline)) int8_t unpack_int4(const uint8_t *packed, size_t idx) {
    size_t byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t byte = packed[byte_idx];
    int8_t val = (byte >> bit_offset) & 0x0F;
    if (val & 0x08) val |= 0xF0;
    return val;
}

// Forward pass (CPU Fallback) - UPDATED FOR CSR
int orth_layer_forward(const orth_layer_t *layer,
                       const float *input,
                       float *output,
                       size_t batch_size) {
    if (!layer || !input || !output) return -1;
    
    size_t in_features = layer->base.in_features;
    size_t out_features = layer->base.out_features;
    int q_bits = layer->base.q_bits;
    const uint8_t *q_weight = (const uint8_t *)layer->base.q_weight;
    const float *q_scales = (const float *)layer->base.q_scales;
    
    int has_ortho = (layer->alpha > 0.0f && layer->ortho.count > 0);
    
    for (size_t b = 0; b < batch_size; b++) {
        const float *x = input + b * in_features;
        float *y = output + b * out_features;
        
        // 1. Compute Base (Dense)
        // ... (Keep existing optimized loop for INT4) ...
        for (size_t out = 0; out < out_features; out++) {
             float acc = 0.0f;
             float scale = q_scales[out];
             // Optimized INT4 unpacking loop...
             for (size_t in = 0; in < in_features; in++) {
                 size_t idx = out * in_features + in;
                 int8_t w_int = unpack_int4(q_weight, idx);
                 acc += x[in] * ((float)w_int * scale);
             }
             y[out] = acc;
        }
        
        // 2. Compute Ortho (Sparse) - CSR Implementation
        if (has_ortho) {
            float alpha = layer->alpha;
            const float *ortho_values = layer->ortho.values;
            
            // CSR Logic: Iterate over rows directly
            if (layer->ortho.format == 1 && layer->ortho.row_ptr) {
                const int32_t *row_ptr = layer->ortho.row_ptr;
                const int32_t *col_indices = layer->ortho.col_indices;
                
                for (size_t row = 0; row < out_features; row++) {
                    int32_t start = row_ptr[row];
                    int32_t end = row_ptr[row + 1];
                    
                    float acc = 0.0f;
                    for (int i = start; i < end; i++) {
                        int32_t col = col_indices[i];
                        acc += ortho_values[i] * x[col];
                    }
                    y[row] += alpha * acc;
                }
            } 
            // Fallback for COO (Legacy)
            else if (layer->ortho.indices) {
                const uint16_t *ortho_indices = layer->ortho.indices;
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
    }
    
    return 0;
}