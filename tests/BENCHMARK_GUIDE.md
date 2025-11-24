# 性能基准测试指南

## Null Test 基准测试

### 目的

验证 Linus 的要求（文档第278行）：
> 如果 W_ortho 全为零，系统的性能必须**完全等同**于一个标准的 INT4 模型。
> 如果为了支持稀疏流导致 Base 流变慢了 1%，就是失败。

### 编译和运行

```bash
cd tests
make benchmark
./benchmark_null_test
```

### 测试配置

- **矩阵尺寸**: 1024 x 1024
- **Batch size**: 32
- **迭代次数**: 100
- **量化**: INT4

### 成功标准

1. **性能开销 < 1%**
   - libortho 在 Ortho=0 时的性能开销必须 < 1%
   - 这是 Linus 的硬性要求

2. **输出正确性**
   - 输出必须与参考实现完全匹配
   - 最大误差 < 1e-4

### 预期输出示例

```
============================================================
Null Test Performance Benchmark
============================================================

Linus's Requirement:
  If W_ortho is all zero, performance must be COMPLETELY EQUAL
  to a standard INT4 model. If Base stream slows down by 1%, it's a failure.

Test Configuration:
  Dimensions: 1024 x 1024
  Batch size: 32
  Iterations: 100

--- Benchmark 1: Reference INT4 Implementation ---
  Average time: 2.345 ms
  Throughput: 13646.27 samples/sec

--- Benchmark 2: libortho with Empty Ortho (Null Test) ---
  Average time: 2.351 ms
  Throughput: 13611.65 samples/sec
  Max output difference: 0.000000
  ✅ Output matches reference

--- Performance Comparison ---
  Reference time: 2.345 ms
  libortho time:  2.351 ms
  Overhead: 0.26%

✅ SUCCESS: Null Test PASSED!
   Overhead (0.26%) is less than 1% threshold.
   libortho with empty Ortho performs equivalently to standard INT4.
```

### 如果测试失败

如果开销 > 1%，检查：

1. **分支开销**：确保 alpha=0 时没有不必要的分支
2. **内存对齐**：验证 128-byte 对齐是否正确
3. **循环优化**：检查编译器优化是否生效
4. **函数调用开销**：考虑内联优化

### 性能优化建议

如果测试失败，可以：

1. 使用 `__forceinline__` 标记关键函数
2. 使用 `#pragma unroll` 优化循环
3. 确保编译器优化级别为 `-O3`
4. 检查是否有不必要的内存拷贝

