/*
 * libortho - CUDA Kernel Test
 * 
 * Simple test to verify CUDA kernel compilation and basic functionality.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Simple test kernel
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("============================================================\n");
    printf("CUDA Kernel Compilation Test\n");
    printf("============================================================\n\n");
    
    // Check CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        printf("❌ CUDA Error: %s\n", cudaGetErrorString(err));
        printf("   CUDA runtime not available.\n");
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("❌ No CUDA devices found\n");
        return 1;
    }
    
    printf("✅ CUDA devices found: %d\n\n", deviceCount);
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        
        // Check Tensor Core support
        if (prop.major >= 7) {
            printf("  ✅ Tensor Cores: Supported\n");
        } else {
            printf("  ❌ Tensor Cores: Not supported (requires >= 7.0)\n");
        }
        printf("\n");
    }
    
    // Test kernel compilation and execution
    printf("Testing kernel compilation and execution...\n");
    
    int n = 1024;
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    float* h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    // Allocate device memory
    float* d_data;
    err = cudaMalloc(&d_data, size);
    if (err != cudaSuccess) {
        printf("❌ cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }
    
    // Copy to device
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("❌ cudaMemcpy (H2D) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    test_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("❌ cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Copy back
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("❌ cudaMemcpy (D2H) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_data[i] - (float)(i * 2)) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✅ Kernel execution test PASSED\n");
    } else {
        printf("❌ Kernel execution test FAILED\n");
    }
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    printf("\n============================================================\n");
    if (correct) {
        printf("✅ CUDA environment is working correctly!\n");
        return 0;
    } else {
        printf("❌ CUDA test failed\n");
        return 1;
    }
}

