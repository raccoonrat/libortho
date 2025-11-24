/*
 * libortho - Null Test Performance Benchmark
 * 
 * Linus's Requirement: If W_ortho is all zero, the system's performance
 * must be COMPLETELY EQUAL to a standard INT4 model.
 * If supporting the sparse stream causes the Base stream to slow down by 1%, it's a failure.
 */

#define _POSIX_C_SOURCE 200809L  // For clock_gettime and CLOCK_MONOTONIC

#include "../include/ortho.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define DIM 1024
#define BATCH_SIZE 32
#define ITERATIONS 100
#define Q_BITS 4

// Unpack INT4 helper function
static inline int8_t unpack_int4_ref(const uint8_t* packed, int idx) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t byte = packed[byte_idx];
    int8_t val = (byte >> bit_offset) & 0x0F;
    if (val & 0x08) val |= 0xF0;
    return val;
}

// Simple INT4 matrix multiplication (reference implementation)
void reference_int4_gemm(
    const uint8_t* q_weight_packed,
    const float* q_scales,
    const float* input,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * in_features;
        float* y = output + b * out_features;
        
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
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int benchmark_null_test() {
    printf("============================================================\n");
    printf("Null Test Performance Benchmark\n");
    printf("============================================================\n");
    printf("\n");
    printf("Linus's Requirement:\n");
    printf("  If W_ortho is all zero, performance must be COMPLETELY EQUAL\n");
    printf("  to a standard INT4 model. If Base stream slows down by 1%%, it's a failure.\n");
    printf("\n");
    printf("Test Configuration:\n");
    printf("  Dimensions: %d x %d\n", DIM, DIM);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Iterations: %d\n", ITERATIONS);
    printf("\n");
    
    // Initialize layer
    orth_layer_t layer;
    if (orth_layer_init(&layer, DIM, DIM, Q_BITS) != 0) {
        fprintf(stderr, "Failed to initialize layer\n");
        return 1;
    }
    
    // Generate test data
    float* input = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    float* output_ref = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    float* output_test = (float*)malloc(BATCH_SIZE * DIM * sizeof(float));
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Initialize weights (random INT4)
    int8_t* weights_int8 = (int8_t*)malloc(DIM * DIM * sizeof(int8_t));
    float* scales = (float*)malloc(DIM * sizeof(float));
    
    for (int i = 0; i < DIM * DIM; i++) {
        weights_int8[i] = (int8_t)(rand() % 16 - 8);
    }
    for (int i = 0; i < DIM; i++) {
        scales[i] = 0.1f + (float)rand() / RAND_MAX * 0.9f;
    }
    
    // Pack weights
    uint8_t* weights_packed = (uint8_t*)layer.base.q_weight;
    for (int i = 0; i < DIM * DIM; i++) {
        int byte_idx = i / 2;
        int bit_offset = (i % 2) * 4;
        uint8_t val = (uint8_t)(weights_int8[i] & 0x0F);
        if (bit_offset == 0) {
            weights_packed[byte_idx] = val;
        } else {
            weights_packed[byte_idx] |= (val << 4);
        }
    }
    memcpy(layer.base.q_scales, scales, DIM * sizeof(float));
    
    // Set Ortho to empty (null test)
    layer.ortho.count = 0;
    layer.ortho.indices = NULL;
    layer.ortho.values = NULL;
    layer.alpha = 0.0f;  // Also disable via alpha
    
    printf("--- Benchmark 1: Reference INT4 Implementation ---\n");
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        reference_int4_gemm(
            (const uint8_t*)layer.base.q_weight,
            (const float*)layer.base.q_scales,
            input, output_ref,
            BATCH_SIZE, DIM, DIM
        );
    }
    
    // Benchmark reference
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        reference_int4_gemm(
            (const uint8_t*)layer.base.q_weight,
            (const float*)layer.base.q_scales,
            input, output_ref,
            BATCH_SIZE, DIM, DIM
        );
    }
    double end = get_time_ms();
    double time_ref = (end - start) / ITERATIONS;
    
    printf("  Average time: %.3f ms\n", time_ref);
    printf("  Throughput: %.2f samples/sec\n", (BATCH_SIZE * 1000.0) / time_ref);
    printf("\n");
    
    printf("--- Benchmark 2: libortho with Empty Ortho (Null Test) ---\n");
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        orth_layer_forward(&layer, input, output_test, BATCH_SIZE);
    }
    
    // Benchmark libortho
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        orth_layer_forward(&layer, input, output_test, BATCH_SIZE);
    }
    end = get_time_ms();
    double time_test = (end - start) / ITERATIONS;
    
    printf("  Average time: %.3f ms\n", time_test);
    printf("  Throughput: %.2f samples/sec\n", (BATCH_SIZE * 1000.0) / time_test);
    printf("\n");
    
    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < BATCH_SIZE * DIM; i++) {
        float diff = fabsf(output_test[i] - output_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max output difference: %.6f\n", max_diff);
    
    if (max_diff > 1e-4f) {
        printf("  ⚠️  WARNING: Output mismatch detected!\n");
    } else {
        printf("  ✅ Output matches reference\n");
    }
    printf("\n");
    
    // Performance comparison
    printf("--- Performance Comparison ---\n");
    double overhead = ((time_test - time_ref) / time_ref) * 100.0;
    printf("  Reference time: %.3f ms\n", time_ref);
    printf("  libortho time:  %.3f ms\n", time_test);
    printf("  Overhead: %.2f%%\n", overhead);
    printf("\n");
    
    // Linus's requirement: overhead must be < 1%
    if (overhead < 1.0) {
        printf("✅ SUCCESS: Null Test PASSED!\n");
        printf("   Overhead (%.2f%%) is less than 1%% threshold.\n", overhead);
        printf("   libortho with empty Ortho performs equivalently to standard INT4.\n");
        return 0;
    } else {
        printf("❌ FAILURE: Null Test FAILED!\n");
        printf("   Overhead (%.2f%%) exceeds 1%% threshold.\n", overhead);
        printf("   Supporting sparse stream causes Base stream slowdown.\n");
        return 1;
    }
}

int main() {
    return benchmark_null_test();
}

