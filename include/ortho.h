/*
 * libortho - Dual-Manifold LLM Runtime
 * 
 * Keep it simple. No C++ templates unless absolutely necessary.
 * Structured packing is critical for memory alignment.
 */

#ifndef ORTHO_H
#define ORTHO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * The Base: Quantized Lattice
 * 128-byte aligned for Tensor Core access
 */
typedef struct {
    void *q_weight;      // INT4 quantized weights
    void *q_scales;      // Quantization scales
    int q_bits;         // Quantization bits (typically 4)
    size_t in_features;
    size_t out_features;
} orth_base_t;

/* 
 * The Ortho: The Normal Component
 * We don't use standard CSR. We use a "Coordinate Stream" 
 * that is pre-sorted to minimize memory jumps.
 */
typedef struct {
    uint16_t *indices;  // Flat index or (row, col) - test it
    float *values;      // FP16/BF16 values (stored as float for compatibility)
    int count;          // Number of non-zero elements
    size_t capacity;    // Allocated capacity
} orth_ortho_t;

/*
 * The Dual-Manifold Layer
 * 
 * Good Taste: We don't mix Base and Ortho storage.
 * They are physically isolated. This allows us to instantly
 * "cut off" privacy by setting ortho_values to NULL or
 * ortho_alpha to 0, without reallocating memory or changing code paths.
 */
typedef struct {
    orth_base_t base;
    orth_ortho_t ortho;
    
    // The Kill Switch
    // This is the pointer logic I love.
    // If this is 0.0, the ortho branch is practically a NOP.
    float alpha;
} orth_layer_t;

/*
 * Initialize a dual-manifold layer
 */
int orth_layer_init(orth_layer_t *layer, 
                    size_t in_features, 
                    size_t out_features,
                    int q_bits);

/*
 * Free resources
 */
void orth_layer_free(orth_layer_t *layer);

/*
 * Set the privacy kill switch (alpha parameter)
 * alpha = 1.0: Full intelligence
 * alpha = 0.0: Privacy safe / Base intelligence only
 */
void orth_layer_set_alpha(orth_layer_t *layer, float alpha);

/*
 * Allocate Ortho component with 128-byte alignment
 * Memory alignment is critical for Memory Controller efficiency
 * 
 * Args:
 *   layer: Layer structure
 *   count: Number of non-zero elements
 * 
 * Returns:
 *   0 on success, -1 on failure
 */
int orth_layer_alloc_ortho(orth_layer_t *layer, size_t count);

/*
 * Free Ortho component (with aligned memory)
 */
void orth_layer_free_ortho(orth_layer_t *layer);

/*
 * Forward pass: Y = X @ W_base + alpha * (X @ W_ortho)
 * 
 * This is the fused kernel interface.
 * Input/output are assumed to be row-major matrices.
 */
int orth_layer_forward(const orth_layer_t *layer,
                       const float *input,   // [batch, in_features]
                       float *output,       // [batch, out_features]
                       size_t batch_size);

/*
 * CUDA-accelerated forward pass (if CUDA available)
 */
#ifdef __CUDACC__
int orth_layer_forward_cuda(const orth_layer_t *layer,
                            const float *input,
                            float *output,
                            size_t batch_size);

/*
 * Tensor Core-accelerated forward pass (if Tensor Cores available)
 * Requires compute capability >= 7.0
 */
int orth_layer_forward_tensor_core(const orth_layer_t *layer,
                                   const float *input,
                                   float *output,
                                   size_t batch_size);

/*
 * Check if Tensor Cores are available
 */
bool check_tensor_core_support(void);
#endif

#ifdef __cplusplus
}
#endif

#endif // ORTHO_H

