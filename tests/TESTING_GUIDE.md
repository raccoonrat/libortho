# libortho 测试指南

## 快速开始

### 运行所有测试

```bash
cd tests
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### 单独运行测试

#### 1. GPU 环境检查

```bash
cd tests
python3 check_gpu.py
```

检查内容：
- nvcc (CUDA 编译器) 是否可用
- PyTorch CUDA 支持
- GPU 信息和 Tensor Core 支持

#### 2. CUDA Kernel 测试

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

测试内容：
- CUDA 运行时可用性
- GPU 设备信息
- 简单的 kernel 编译和执行
- Tensor Core 支持检测

#### 3. CPU Forward 测试

```bash
cd tests
make test
./test_cpu_forward
```

测试内容：
- CPU forward 函数正确性
- INT4 解包
- 稀疏矩阵乘法
- 内存管理

#### 4. 性能基准测试 (Null Test)

```bash
cd tests
make benchmark
./benchmark_null_test
```

测试内容：
- 性能对比（参考实现 vs libortho）
- 验证开销 < 1% (Linus 要求)
- 输出正确性验证

## 测试要求

### 系统要求

- **Linux/WSL**: 推荐 Ubuntu 20.04+
- **编译器**: GCC 9+ 或 Clang 10+
- **CUDA** (可选): CUDA Toolkit 11.0+ (用于 GPU 测试)
- **Python** (可选): Python 3.8+ (用于 GPU 检查)

### 编译选项

Makefile 中的编译选项：
- `CFLAGS`: `-O3 -Wall -Wextra -std=c11`
- `CUDA_CFLAGS`: `-O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 --expt-relaxed-constexpr`

## 预期输出

### GPU 环境检查

```
============================================================
libortho - GPU Environment Check
============================================================

1. Checking nvcc (CUDA Compiler)...
✅ nvcc found
   nvcc: NVIDIA (R) Cuda compiler driver

2. Checking PyTorch CUDA support...
✅ PyTorch version: 2.0.0
✅ CUDA available: 11.8
   GPU count: 1

   GPU 0: NVIDIA GeForce RTX 3080
     Compute Capability: 8.6
     Total Memory: 10.00 GB
     ✅ Tensor Cores: Supported (sm_86)
```

### CUDA Kernel 测试

```
============================================================
CUDA Kernel Compilation Test
============================================================

✅ CUDA devices found: 1

GPU 0: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Total Global Memory: 10.00 GB
  Multiprocessors: 68
  ✅ Tensor Cores: Supported

Testing kernel compilation and execution...
✅ Kernel execution test PASSED

============================================================
✅ CUDA environment is working correctly!
```

### CPU Forward 测试

```
============================================================
CPU Forward Pass Test
============================================================

Test 1: Full Model (Base + Ortho)
  ✅ PASSED

Test 2: Base Only (alpha=0.0)
  ✅ PASSED

Test 3: Empty Ortho
  ✅ PASSED

============================================================
✅ All tests passed!
```

### 性能基准测试

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

## 故障排除

### 编译错误

**问题**: `posix_memalign` 未声明
**解决**: 确保定义了 `_POSIX_C_SOURCE 200809L`

**问题**: CUDA 编译错误
**解决**: 检查 CUDA Toolkit 版本和 GPU 架构支持

### 运行时错误

**问题**: `CUDA error: no kernel image is available`
**解决**: 检查 GPU compute capability，可能需要添加 `-arch=sm_XX`

**问题**: 性能测试失败（开销 > 1%）
**解决**: 
- 检查编译器优化级别 (`-O3`)
- 验证内存对齐
- 检查是否有不必要的分支

### GPU 不可用

如果没有 GPU 或 CUDA：
- CPU 测试仍然可以运行
- GPU 相关测试会被跳过
- 性能基准测试使用 CPU 实现

## 测试覆盖

- ✅ CPU forward 实现
- ✅ 内存管理
- ✅ INT4 量化/解包
- ✅ 稀疏矩阵操作
- ⚠️ CUDA kernel (需要 GPU)
- ⚠️ Tensor Core (需要支持 Tensor Core 的 GPU)
- ✅ 性能基准测试 (CPU)

## 持续集成

建议在 CI/CD 中运行：

```yaml
# Example GitHub Actions
- name: Run CPU Tests
  run: |
    cd tests
    make test
    ./test_cpu_forward

- name: Run Benchmark
  run: |
    cd tests
    make benchmark
    ./benchmark_null_test
```

GPU 测试需要自托管 runner 或 GPU 支持。

