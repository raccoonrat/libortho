# Tensor Core 实现状态

## 当前实现

### ✅ 已完成

1. **Tensor Core 框架** (`src/dual_gemm_tensor_core.cu`)
   - 完整的 WMMA API 使用
   - INT4 -> INT8 解包
   - FP32 -> INT8 量化
   - 16x16 tile 处理
   - 累加器管理

2. **GPU 能力检测** (`check_tensor_core_support()`)
   - 自动检测 compute capability >= 7.0
   - 运行时选择 Tensor Core 或标准 kernel

3. **构建系统集成** (`setup.py`)
   - 自动包含 Tensor Core 源文件
   - 支持多架构编译 (sm_75, sm_80, sm_86, sm_89)

### ⚠️ 注意事项

Tensor Core 实现需要满足严格的数据布局要求：

1. **WMMA 数据布局**：
   - `matrix_a` (input): 必须是 row-major
   - `matrix_b` (weights): 必须是 col-major
   - `accumulator`: row-major 存储

2. **内存对齐**：
   - 所有数据必须 128-byte 对齐（已实现）
   - Shared memory 布局必须匹配 WMMA 要求

3. **量化策略**：
   - 当前使用简化的每 tile 量化
   - 生产环境应使用校准的量化尺度

### 🔄 待优化

1. **输入量化**：
   - 当前：简化的每 tile 量化
   - 优化：使用校准的量化尺度（类似 GPTQ）

2. **Ortho 融合**：
   - 当前：Base 和 Ortho 分离计算
   - 优化：完全融合的双流 Tensor Core kernel

3. **批处理优化**：
   - 当前：每个 batch 单独处理
   - 优化：批处理 tile 管理

## 使用方式

### 自动选择（推荐）

```cpp
// orth_layer_forward_cuda() 会自动选择最佳实现
// 如果 Tensor Core 可用，优先使用
int result = orth_layer_forward_cuda(&layer, input, output, batch_size);
```

### 手动选择

```cpp
if (check_tensor_core_support()) {
    orth_layer_forward_tensor_core(&layer, input, output, batch_size);
} else {
    orth_layer_forward_cuda(&layer, input, output, batch_size);
}
```

## 性能预期

- **标准 CUDA Kernel**: ~100-500 GFLOPS
- **Tensor Core Kernel**: ~1000-5000 GFLOPS (10-50x 提升)

## 测试建议

1. **功能测试**：验证 Tensor Core 输出与标准 kernel 一致
2. **性能测试**：对比 Tensor Core vs 标准 kernel 的性能
3. **边界测试**：测试不同矩阵尺寸（必须是 16 的倍数）

## 参考资源

- [NVIDIA WMMA API 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [CUTLASS 库](https://github.com/NVIDIA/cutlass) - 高级 Tensor Core 抽象
- [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#cublasltApi) - 优化的 GEMM 库

## 当前状态

✅ **框架完整** - 可以编译和运行
⚠️ **需要测试** - 需要在实际 GPU 上验证
🔄 **可优化** - 输入量化和融合可以进一步优化

