/*
 * libortho - CPU Forward Pass Test
 * 
 * This test verifies the correctness of orth_layer_forward()
 * by comparing it against a reference implementation.
 */

#include "../include/ortho.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Test parameters
#define DIM 64
#define BATCH_SIZE 4
#define Q_BITS 4

// Unpack INT4 helper (same as in ortho.c)
static inline int8_t unpack_int4_ref(const uint8_t* packed, int idx) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t byte = packed[byte_idx];
    int8_t val = (byte >> bit_offset) & 0x0F;
    if (val & 0x08) val |= 0xF0;
    return val;
}

// Reference implementation (simple, correct but slow)
void reference_forward(
    const uint8_t* q_weight_packed,
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
    
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * in_features;
        float* y = output + b * out_features;
        
        // Zero output
        for (int out = 0; out < out_features; out++) {
            y[out] = 0.0f;
        }
        
        // Base: Y = X @ W_base
        for (int out = 0; out < out_features; out++) {
            float acc = 0.0f;
            float scale = q_scales[out];
            for (int in = 0; in < in_features; in++) {
                int idx = out * in_features + in;
                int8_t w_int = unpack_int4_ref(q_weight_packed, idx);
                float w = (float)w_int * scale;
                acc += x[in] * w;
            }
            y[out] = acc;
        }
        
        // Ortho: Y += alpha * (X @ W_ortho)
        if (alpha > 0.0f && ortho_count > 0) {
            for (int i = 0; i < ortho_count; i++) {
                uint16_t flat_idx = ortho_indices[i];
                int row = flat_idx / in_features;
                int col = flat_idx % in_features;
                if (row < out_features && col < in_features) {
                    y[row] += alpha * ortho_values[i] * x[col];
                }
            }
        }
    }
}

// Pack INT4 helper
void pack_int4(const int8_t* values, uint8_t* packed, int count) {
    for (int i = 0; i < count; i++) {
        int byte_idx = i / 2;
        int bit_offset = (i % 2) * 4;
        int8_t val = values[i];
        uint8_t byte_val = (uint8_t)(val & 0x0F);
        if (bit_offset == 0) {
            packed[byte_idx] = byte_val;
        } else {
            packed[byte_idx] |= (byte_val << 4);
        }
    }
}

// Generate test data
void generate_test_data(
    orth_layer_t* layer,
    float** input_ptr,
    float** output_ref_ptr,
    float** output_test_ptr
) {
    // Allocate input
    *input_ptr = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    *output_ref_ptr = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    *output_test_ptr = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    
    // Generate random input
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        (*input_ptr)[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Initialize layer with test weights
    int8_t* weights_int8 = (int8_t*)malloc(DIM * DIM * sizeof(int8_t));
    float* scales = (float*)malloc(DIM * sizeof(float));
    
    // Generate random weights
    for (int i = 0; i < DIM * DIM; i++) {
        weights_int8[i] = (int8_t)(rand() % 16 - 8);  // -8 to 7
    }
    for (int i = 0; i < DIM; i++) {
        scales[i] = 0.1f + (float)rand() / RAND_MAX * 0.9f;
    }
    
    // Pack weights
    uint8_t* weights_packed = (uint8_t*)layer->base.q_weight;
    pack_int4(weights_int8, weights_packed, DIM * DIM);
    memcpy(layer->base.q_scales, scales, DIM * sizeof(float));
    
    // Generate sparse ortho weights
    int ortho_count = DIM * DIM / 20;  // 5% sparsity
    layer->ortho.count = ortho_count;
    layer->ortho.indices = (uint16_t*)malloc(ortho_count * sizeof(uint16_t));
    layer->ortho.values = (float*)malloc(ortho_count * sizeof(float));
    
    for (int i = 0; i < ortho_count; i++) {
        layer->ortho.indices[i] = (uint16_t)(rand() % (DIM * DIM));
        layer->ortho.values[i] = (float)rand() / RAND_MAX * 0.1f - 0.05f;
    }
    
    free(weights_int8);
    free(scales);
}

int test_forward_pass() {
    printf("=== Testing orth_layer_forward() ===\n\n");
    
    // Initialize layer
    orth_layer_t layer;
    if (orth_layer_init(&layer, DIM, DIM, Q_BITS) != 0) {
        fprintf(stderr, "Failed to initialize layer\n");
        return 1;
    }
    
    // Generate test data
    float* input = NULL;
    float* output_ref = NULL;
    float* output_test = NULL;
    generate_test_data(&layer, &input, &output_ref, &output_test);
    
    printf("Test configuration:\n");
    printf("  Dimensions: %d x %d\n", DIM, DIM);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Quantization: INT%d\n", Q_BITS);
    printf("  Ortho sparsity: %d elements (%.1f%%)\n", 
           layer.ortho.count, 
           100.0f * layer.ortho.count / (DIM * DIM));
    printf("\n");
    
    // Test 1: Alpha = 1.0 (full model)
    printf("Test 1: Alpha = 1.0 (Full Model)\n");
    layer.alpha = 1.0f;
    
    // Reference implementation
    reference_forward(
        (const uint8_t*)layer.base.q_weight,
        (const float*)layer.base.q_scales,
        layer.ortho.values,
        layer.ortho.indices,
        layer.ortho.count,
        input,
        output_ref,
        BATCH_SIZE,
        DIM,
        DIM,
        1.0f
    );
    
    // Our implementation
    if (orth_layer_forward(&layer, input, output_test, BATCH_SIZE) != 0) {
        fprintf(stderr, "orth_layer_forward() failed\n");
        return 1;
    }
    
    // Compare results
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        float diff = fabsf(output_test[i] - output_ref[i]);
        float rel_diff = diff / (fabsf(output_ref[i]) + 1e-8f);
        if (diff > max_diff) max_diff = diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
        if (diff > 1e-5f) errors++;
    }
    
    printf("  Max absolute error: %.6f\n", max_diff);
    printf("  Max relative error: %.6f\n", max_rel_diff);
    printf("  Elements with error > 1e-5: %d / %d\n", errors, BATCH_SIZE * DIM);
    
    if (max_diff < 1e-4f && max_rel_diff < 1e-3f) {
        printf("  ✅ PASSED\n\n");
    } else {
        printf("  ❌ FAILED\n\n");
        return 1;
    }
    
    // Test 2: Alpha = 0.0 (base only)
    printf("Test 2: Alpha = 0.0 (Base Only)\n");
    layer.alpha = 0.0f;
    
    // Reference
    reference_forward(
        (const uint8_t*)layer.base.q_weight,
        (const float*)layer.base.q_scales,
        layer.ortho.values,
        layer.ortho.indices,
        layer.ortho.count,
        input,
        output_ref,
        BATCH_SIZE,
        DIM,
        DIM,
        0.0f
    );
    
    // Our implementation
    if (orth_layer_forward(&layer, input, output_test, BATCH_SIZE) != 0) {
        fprintf(stderr, "orth_layer_forward() failed\n");
        return 1;
    }
    
    // Compare
    max_diff = 0.0f;
    max_rel_diff = 0.0f;
    errors = 0;
    
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        float diff = fabsf(output_test[i] - output_ref[i]);
        float rel_diff = diff / (fabsf(output_ref[i]) + 1e-8f);
        if (diff > max_diff) max_diff = diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
        if (diff > 1e-5f) errors++;
    }
    
    printf("  Max absolute error: %.6f\n", max_diff);
    printf("  Max relative error: %.6f\n", max_rel_diff);
    printf("  Elements with error > 1e-5: %d / %d\n", errors, BATCH_SIZE * DIM);
    
    if (max_diff < 1e-4f && max_rel_diff < 1e-3f) {
        printf("  ✅ PASSED\n\n");
    } else {
        printf("  ❌ FAILED\n\n");
        return 1;
    }
    
    // Test 3: No ortho (empty ortho)
    printf("Test 3: Empty Ortho (Base Only)\n");
    layer.alpha = 1.0f;
    int old_count = layer.ortho.count;
    layer.ortho.count = 0;
    
    // Reference
    reference_forward(
        (const uint8_t*)layer.base.q_weight,
        (const float*)layer.base.q_scales,
        NULL,
        NULL,
        0,
        input,
        output_ref,
        BATCH_SIZE,
        DIM,
        DIM,
        1.0f
    );
    
    // Our implementation
    if (orth_layer_forward(&layer, input, output_test, BATCH_SIZE) != 0) {
        fprintf(stderr, "orth_layer_forward() failed\n");
        return 1;
    }
    
    // Compare
    max_diff = 0.0f;
    max_rel_diff = 0.0f;
    errors = 0;
    
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        float diff = fabsf(output_test[i] - output_ref[i]);
        float rel_diff = diff / (fabsf(output_ref[i]) + 1e-8f);
        if (diff > max_diff) max_diff = diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
        if (diff > 1e-5f) errors++;
    }
    
    printf("  Max absolute error: %.6f\n", max_diff);
    printf("  Max relative error: %.6f\n", max_rel_diff);
    printf("  Elements with error > 1e-5: %d / %d\n", errors, BATCH_SIZE * DIM);
    
    if (max_diff < 1e-4f && max_rel_diff < 1e-3f) {
        printf("  ✅ PASSED\n\n");
    } else {
        printf("  ❌ FAILED\n\n");
        return 1;
    }
    
    layer.ortho.count = old_count;
    
    // Cleanup
    // Note: orth_layer_free() will free ortho.indices and ortho.values
    // So we should NOT free them manually here
    free(input);
    free(output_ref);
    free(output_test);
    // Don't free layer.ortho.indices/values - orth_layer_free() does it
    orth_layer_free(&layer);
    
    printf("=== All tests passed! ===\n");
    return 0;
}

int main() {
    srand(42);  // Fixed seed for reproducibility
    return test_forward_pass();
}

