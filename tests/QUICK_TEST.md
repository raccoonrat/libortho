# 快速测试指南

## 在 WSL 中运行测试

### 1. GPU 环境检查

```bash
cd tests
python3 check_gpu.py
```

### 2. 编译并运行 CPU 测试

```bash
cd tests
make clean
make test
./test_cpu_forward
```

### 3. 编译并运行性能基准测试

```bash
cd tests
make benchmark
./benchmark_null_test
```

### 4. 编译并运行 CUDA 测试（如果 GPU 可用）

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

### 5. 运行所有测试（推荐）

```bash
cd tests
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## 预期结果

### CPU 测试
- ✅ 所有测试用例通过
- ✅ 输出正确性验证通过

### 性能基准测试
- ✅ 开销 < 1% (Linus 要求)
- ✅ 输出与参考实现匹配

### CUDA 测试（如果 GPU 可用）
- ✅ CUDA 运行时正常
- ✅ Kernel 编译和执行成功
- ✅ Tensor Core 支持检测

## 常见问题

### 编译错误

**问题**: `posix_memalign` 未声明
**解决**: 已在 `ortho.c` 中添加 `#define _POSIX_C_SOURCE 200809L`

**问题**: 找不到头文件
**解决**: 确保在 `tests/` 目录下运行，Makefile 会自动设置包含路径

### 运行时错误

**问题**: 段错误 (segmentation fault)
**解决**: 检查内存对齐和初始化

**问题**: 性能测试开销 > 1%
**解决**: 
- 确保使用 `-O3` 优化
- 检查是否有不必要的分支
- 验证内存对齐

## 测试输出示例

### 成功的性能基准测试输出

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

