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
    layer->ortho.indices = NULL;
    layer->ortho.values = NULL;
    layer->ortho.count = 0;
    layer->ortho.capacity = 0;
    
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
    free(layer->ortho.indices);
    free(layer->ortho.values);
    
    memset(layer, 0, sizeof(orth_layer_t));
}

void orth_layer_set_alpha(orth_layer_t *layer, float alpha) {
    if (layer) {
        layer->alpha = alpha;
    }
}

// Helper: Unpack INT4 value from packed array
static inline int8_t unpack_int4(const uint8_t *packed, int idx) {
    int byte_idx = idx / 2;
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
    
    // Zero output
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    // Process each batch
    for (size_t b = 0; b < batch_size; b++) {
        const float *x = input + b * in_features;
        float *y = output + b * out_features;
        
        // Compute Base: Y = X @ W_base (INT4 quantized)
        for (size_t out = 0; out < out_features; out++) {
            float acc = 0.0f;
            float scale = q_scales[out];
            
            // Dequantize and compute dot product
            for (size_t in = 0; in < in_features; in++) {
                int8_t w_int;
                if (q_bits == 4) {
                    w_int = unpack_int4(q_weight, out * in_features + in);
                } else {
                    // Fallback for other bit widths (simplified)
                    int byte_idx = (out * in_features + in) * q_bits / 8;
                    w_int = ((int8_t *)q_weight)[byte_idx];
                }
                float w = (float)w_int * scale;
                acc += x[in] * w;
            }
            y[out] = acc;
        }
        
        // Compute Ortho: Y += alpha * (X @ W_ortho) (sparse)
        if (layer->alpha > 0.0f && layer->ortho.count > 0) {
            const float *ortho_values = layer->ortho.values;
            const uint16_t *ortho_indices = layer->ortho.indices;
            
            for (int i = 0; i < layer->ortho.count; i++) {
                uint16_t flat_idx = ortho_indices[i];
                size_t row = flat_idx / in_features;
                size_t col = flat_idx % in_features;
                
                if (row < out_features && col < in_features) {
                    y[row] += layer->alpha * ortho_values[i] * x[col];
                }
            }
        }
    }
    
    return 0;
}

