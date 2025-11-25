# GPU 测试快速开始

## 在 WSL 中运行 GPU 测试

### 方法 1: 运行所有测试（推荐）

```bash
cd tests
chmod +x run_gpu_tests.sh
./run_gpu_tests.sh
```

这将自动运行：
1. GPU 环境检查
2. CUDA kernel 测试
3. Tensor Core 测试

### 方法 2: 单独运行测试

#### 步骤 1: 检查 GPU 环境

```bash
cd tests
python3 check_gpu.py
```

**预期输出**:
```
✅ nvcc found
✅ PyTorch version: X.X.X
✅ CUDA available: XX.X
   GPU count: 1
   GPU 0: [Your GPU Name]
     Compute Capability: X.X
     ✅ Tensor Cores: Supported
```

#### 步骤 2: 测试 CUDA Kernel

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

**预期输出**:
```
✅ CUDA devices found: 1
✅ Kernel execution test PASSED
✅ CUDA environment is working correctly!
```

#### 步骤 3: 测试 Tensor Core

```bash
cd tests
make tensor-core-test
./test_tensor_core
```

**预期输出**:
```
✅ Tensor Cores available
✅ Tensor Core kernel interface is correct
✅ Function declarations are valid
✅ Tensor Core test framework is ready
```

## 测试结果解读

### ✅ 成功标志

- GPU 环境检查通过
- CUDA kernel 编译和执行成功
- Tensor Core 支持检测成功
- 函数接口验证通过

### ⚠️ 注意事项

当前测试是**框架测试**，主要验证：
- GPU 环境配置
- CUDA 编译能力
- Tensor Core 支持
- 函数接口正确性

完整功能测试需要：
- 设备内存管理实现
- 数据复制机制
- 完整的测试数据

## 常见问题

### Q: `nvcc: command not found`

**A**: 安装 CUDA Toolkit
```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
```

### Q: `Tensor Cores not available`

**A**: 检查 GPU compute capability
- Tensor Core 需要 >= 7.0
- 运行 `nvidia-smi` 查看 GPU 信息
- 或在代码中检查 `check_tensor_core_support()`

### Q: 编译错误 `error: identifier "wmma" is undefined`

**A**: 确保：
1. CUDA 版本 >= 10.0
2. 编译时包含正确的架构（-arch=sm_75 等）
3. 包含 `<mma.h>` 头文件

### Q: 运行时错误 `no CUDA-capable device is detected`

**A**: 
1. 检查 WSL CUDA 驱动: `nvidia-smi`
2. 确保 GPU 被 WSL 识别
3. 检查 CUDA 环境变量

## 下一步

测试通过后，可以：
1. 实现完整的设备内存管理
2. 运行完整的功能测试
3. 进行性能基准测试

## 参考文档

- `tests/GPU_TEST_GUIDE.md` - 详细测试指南
- `tests/run_gpu_tests.sh` - 自动化测试脚本
- `docs/ORTHO_FUSION_COMPLETE.md` - Ortho 融合实现文档

