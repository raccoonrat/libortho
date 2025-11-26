# Tensor Core 实现指南

## 当前状态

当前的 `src/dual_gemm.cu` 实现了优化的 INT4 矩阵乘法，但未使用真正的 Tensor Core。

## 为什么需要 Tensor Core？

Tensor Core 是 NVIDIA GPU（Volta 架构及以后）的专用硬件单元，可以：
- 在单个时钟周期内执行 4x4x4 矩阵乘法
- 提供比传统 CUDA Core 高 10-100 倍的性能
- 特别适合 INT4/INT8 量化矩阵乘法

## 实现 Tensor Core 的步骤

### 1. 使用 WMMA API

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 定义 fragment 类型
fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 加载数据
load_matrix_sync(a_frag, input_ptr, stride);
load_matrix_sync(b_frag, weight_ptr, stride);

// 执行矩阵乘法
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
store_matrix_sync(output_ptr, c_frag, stride, layout);
```

### 2. INT4 打包格式

Tensor Core 需要特定的数据布局：
- **16x16 tile**: 每个 fragment 处理 16x16 的块
- **行主序/列主序**: 根据矩阵类型选择
- **对齐**: 数据必须 128-byte 对齐（已实现）

### 3. 完整实现框架

```cpp
__global__ void tensor_core_int4_gemm(
    const uint8_t* q_weight_packed,  // INT4 packed weights
    const float* q_scales,           // Per-row scales
    const float* input,              // FP32 input
    float* output,                   // FP32 output
    int M, int N, int K              // Matrix dimensions
) {
    // Tile dimensions for Tensor Core
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;
    
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int n = blockIdx.x * TILE_N + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    // Declare fragments
    fragment<matrix_a, TILE_M, TILE_N, TILE_K, int8_t, row_major> a_frag;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, int8_t, col_major> b_frag;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // K-dimension loop (tiled)
    for (int k = 0; k < K; k += TILE_K) {
        // Load input tile (convert FP32 -> INT8)
        // Load weight tile (unpack INT4 -> INT8)
        // ...
        
        // Matrix multiply
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Apply scale and store
    // ...
}
```

## 当前实现的优势

虽然当前实现未使用 Tensor Core，但它：

1. **内存对齐**: 128-byte 对齐已实现
2. **优化循环**: 使用 chunk 处理提高缓存效率
3. **SIMD 友好**: 代码结构适合编译器优化
4. **向后兼容**: 可在不支持 Tensor Core 的 GPU 上运行

## 迁移到 Tensor Core

### 步骤 1: 检查 GPU 支持

```cpp
int major, minor;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

// Tensor Core requires compute capability >= 7.0
bool has_tensor_cores = (major >= 7);
```

### 步骤 2: 实现 Tensor Core 版本

创建 `src/dual_gemm_tensor_core.cu`，实现完整的 Tensor Core 版本。

### 步骤 3: 运行时选择

在 `orth_layer_forward_cuda` 中根据 GPU 能力选择实现：

```cpp
if (has_tensor_cores) {
    tensor_core_dual_gemm_kernel<<<...>>>(...);
} else {
    dual_gemm_kernel<<<...>>>(...);
}
```

## 性能预期

- **当前实现**: ~100-500 GFLOPS（取决于矩阵大小）
- **Tensor Core 实现**: ~1000-5000 GFLOPS（10-50x 提升）

## 参考资源

1. [NVIDIA WMMA API 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
2. [CUTLASS 库](https://github.com/NVIDIA/cutlass) - 高级 Tensor Core 抽象
3. [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#cublasltApi) - 优化的 GEMM 库

## 注意事项

1. **数据格式**: Tensor Core 需要特定的数据布局
2. **Tile 大小**: 必须是 16x16 的倍数
3. **同步**: 必须使用 `__syncwarp()` 或 `mma_sync()`
4. **精度**: Tensor Core 使用混合精度（INT8/FP16 输入，FP32 累加）

## 当前优先级

对于 libortho 项目：
- ✅ **已完成**: 内存对齐、CPU 实现、优化的 CUDA kernel
- 🔄 **待实现**: 完整的 Tensor Core 实现（需要更多测试和优化）

建议先验证当前实现的正确性，再迁移到 Tensor Core。

