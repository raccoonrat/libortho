# RTX 5060 (Blackwell) 支持

**更新日期**: 2025-01-25  
**GPU架构**: NVIDIA Blackwell  
**CUDA计算能力**: 12.0 (sm_100)  
**分支**: `feature/rtx5060-blackwell-support`

---

## 概述

本文档说明如何为RTX 5060（Blackwell架构）GPU配置和编译libortho。

RTX 5060基于NVIDIA的Blackwell架构，支持CUDA计算能力12.0（sm_100），具有以下特性：

- **第五代Tensor Cores**: 支持FP4精度和增强的AI推理性能
- **增强的内存带宽**: 优化的内存子系统
- **CUDA 12.8+**: 需要CUDA Toolkit 12.8或更高版本

## 分支信息

### 创建分支

```bash
git checkout -b feature/rtx5060-blackwell-support
```

### 分支变更

本分支包含以下更新：

1. **setup.py**: 添加`-arch=sm_100`支持
2. **tests/Makefile**: 添加`-arch=sm_100`支持
3. **文档**: 添加RTX 5060支持说明

## 系统要求

### CUDA工具包

- **CUDA Toolkit**: 12.8或更高版本
- **NVIDIA驱动**: 支持Blackwell架构的最新驱动

### 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 12.8 或更高版本

nvidia-smi
# 应该显示RTX 5060和驱动版本
```

### 检查GPU信息

```bash
cd tests
python3 check_gpu.py
```

预期输出：
```
GPU 0: NVIDIA GeForce RTX 5060
  Compute Capability: 10.0
  Total Memory: XX.XX GB
  ✅ Tensor Cores: Supported (sm_100)
```

## 编译配置

### setup.py配置

`setup.py`已更新，包含以下CUDA架构：

```python
'nvcc': [
    '-O3', 
    '--use_fast_math',
    '-arch=sm_75',  # Turing (Tensor Core INT8)
    '-arch=sm_80',  # Ampere
    '-arch=sm_86',  # Ampere (consumer)
    '-arch=sm_89',  # Ada Lovelace
    '-arch=sm_100', # Blackwell (RTX 5060, RTX 50 series)
    '--expt-relaxed-constexpr'
]
```

### Makefile配置

`tests/Makefile`已更新：

```makefile
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_100 --expt-relaxed-constexpr
```

## 编译和安装

### 安装libortho

```bash
# 在项目根目录
pip install -e .
```

### 编译测试

```bash
cd tests
make clean
make cuda-test
./test_cuda_kernel
```

### 验证Tensor Core支持

```bash
cd tests
make tensor-core-test
./test_tensor_core
```

## 性能优化

### RTX 5060特定优化

RTX 5060支持以下优化：

1. **FP4精度**: 第五代Tensor Cores支持FP4，可以进一步压缩模型
2. **增强的内存带宽**: 优化内存访问模式
3. **更大的共享内存**: 可以利用更大的共享内存进行优化

### 建议的编译选项

对于RTX 5060，可以使用以下优化选项：

```bash
# 在setup.py中，可以添加RTX 5060特定的优化
'nvcc': [
    '-O3',
    '--use_fast_math',
    '-arch=sm_100',
    '--expt-relaxed-constexpr',
    '--ptxas-options=-v',  # 显示寄存器使用情况
]
```

## 测试

### 运行所有测试

```bash
cd tests
./run_all_tests.sh
```

### GPU环境检查

```bash
cd tests
python3 check_gpu.py
```

### CUDA Kernel测试

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

预期输出应显示：
```
GPU 0: NVIDIA GeForce RTX 5060
  Compute Capability: 10.0
  ✅ Tensor Cores: Supported
✅ CUDA environment is working correctly!
```

## 故障排除

### 编译错误：不支持的架构

**错误**: `error: no kernel image is available for execution on the device`

**解决**:
1. 确认CUDA Toolkit版本 >= 12.8
2. 确认`-arch=sm_100`已添加到编译选项
3. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 运行时错误：CUDA版本不匹配

**错误**: `CUDA driver version is insufficient for CUDA runtime version`

**解决**:
1. 更新NVIDIA驱动到最新版本
2. 检查驱动是否支持Blackwell架构：
   ```bash
   nvidia-smi
   ```

### Tensor Core不可用

**错误**: Tensor Cores not detected

**解决**:
1. 确认GPU是RTX 5060
2. 检查compute capability是否为10.0
3. 确认CUDA Toolkit版本 >= 12.8

## 已知限制

1. **CUDA 12.8要求**: RTX 5060需要CUDA Toolkit 12.8或更高版本
2. **驱动要求**: 需要支持Blackwell架构的最新NVIDIA驱动
3. **PyTorch兼容性**: 确保PyTorch版本支持CUDA 12.8

## 下一步工作

1. **FP4精度支持**: 利用第五代Tensor Cores的FP4精度
2. **性能基准测试**: 在RTX 5060上运行完整的性能基准测试
3. **内存优化**: 利用增强的内存带宽进行优化

## 参考

- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

**分支**: `feature/rtx5060-blackwell-support`  
**维护者**: libortho contributors  
**最后更新**: 2025-01-25

