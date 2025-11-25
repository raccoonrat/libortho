# GPU 测试指南

## 快速开始

### 在 WSL 中运行所有 GPU 测试

```bash
cd tests
chmod +x run_gpu_tests.sh
./run_gpu_tests.sh
```

### 单独运行测试

#### 1. GPU 环境检查

```bash
cd tests
python3 check_gpu.py
```

#### 2. CUDA Kernel 测试

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

#### 3. Tensor Core 测试

```bash
cd tests
make tensor-core-test
./test_tensor_core
```

## 测试内容

### GPU 环境检查

- ✅ nvcc (CUDA 编译器) 可用性
- ✅ PyTorch CUDA 支持
- ✅ GPU 信息和 Tensor Core 支持
- ✅ nvidia-smi 检查

### CUDA Kernel 测试

- ✅ CUDA 运行时可用性
- ✅ GPU 设备信息
- ✅ 简单 kernel 编译和执行
- ✅ Tensor Core 支持检测

### Tensor Core 测试

- ✅ Tensor Core 支持检测
- ✅ 函数接口验证
- ⚠️ 完整功能测试（需要设备内存管理）

## 预期输出

### 成功的 GPU 环境检查

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

### 成功的 CUDA Kernel 测试

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

### 成功的 Tensor Core 测试

```
============================================================
libortho - Tensor Core GPU Test
============================================================

============================================================
Tensor Core Functionality Test
============================================================

✅ Tensor Cores available

Test 1: Base-only (no Ortho)
  Testing Tensor Core support...
  ✅ Tensor Cores available

  ⚠️  Note: Full execution test requires:
     - Device memory for layer data
     - Unified memory or proper cudaMemcpy
     - This is a framework test

  ✅ Tensor Core kernel interface is correct
  ✅ Function declarations are valid

============================================================
Tensor Core Performance Test
============================================================

✅ Tensor Cores available - ready for performance testing

⚠️  Full performance test requires:
   1. Device memory management implementation
   2. Proper data copying (cudaMemcpy or unified memory)
   3. Benchmarking framework

   Current status: Framework ready, implementation needed

============================================================
✅ Tensor Core test framework is ready
   Full test requires device memory management implementation
============================================================
```

## 故障排除

### 编译错误

**问题**: `nvcc: command not found`
- **解决**: 安装 CUDA Toolkit
  ```bash
  # Ubuntu/Debian
  sudo apt-get install nvidia-cuda-toolkit
  ```

**问题**: `error: identifier "wmma" is undefined`
- **解决**: 确保使用正确的 CUDA 版本（>= 10.0）和 compute capability (>= 7.0)
  ```bash
  nvcc --version  # 检查 CUDA 版本
  ```

**问题**: `error: no kernel image is available`
- **解决**: 检查 GPU compute capability，可能需要添加架构支持
  ```bash
  # 在 Makefile 中添加你的 GPU 架构
  # 例如: -arch=sm_75, -arch=sm_80, -arch=sm_86
  ```

### 运行时错误

**问题**: `CUDA error: no CUDA-capable device is detected`
- **解决**: 
  - 检查 GPU 是否被 WSL 识别: `nvidia-smi`
  - 确保安装了 WSL CUDA 驱动

**问题**: `Tensor Cores not available`
- **解决**: Tensor Core 需要 compute capability >= 7.0
  - Volta (V100): sm_70
  - Turing (RTX 20xx): sm_75
  - Ampere (RTX 30xx, A100): sm_80, sm_86
  - Ada Lovelace (RTX 40xx): sm_89

### 测试限制

当前测试是**框架测试**，主要验证：
- ✅ GPU 环境
- ✅ CUDA 编译
- ✅ Tensor Core 支持检测
- ✅ 函数接口

完整功能测试需要：
- ⚠️ 设备内存管理
- ⚠️ 数据复制（cudaMemcpy 或统一内存）
- ⚠️ 完整的测试数据

## 下一步

1. **实现设备内存管理**
   - 使用 `cudaMalloc` 分配设备内存
   - 使用 `cudaMemcpy` 复制数据
   - 或使用统一内存（Unified Memory）

2. **完整功能测试**
   - 测试 Tensor Core 输出正确性
   - 对比标准 kernel 和 Tensor Core kernel

3. **性能基准测试**
   - 测量 Tensor Core 性能
   - 对比不同实现的性能

## 参考

- `tests/run_gpu_tests.sh` - 自动化测试脚本
- `tests/check_gpu.py` - GPU 环境检查
- `tests/test_cuda_kernel.cu` - CUDA kernel 测试
- `tests/test_tensor_core.cu` - Tensor Core 测试

