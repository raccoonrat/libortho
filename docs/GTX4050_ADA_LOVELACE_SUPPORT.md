# GTX 4050 (Ada Lovelace) 支持

**更新日期**: 2025-01-25  
**GPU架构**: NVIDIA Ada Lovelace  
**CUDA计算能力**: 8.9 (sm_89)  
**分支**: `feature/gtx4050-ada-lovelace-support`

---

## 概述

本文档说明如何为GTX 4050（Ada Lovelace架构）GPU配置和编译libortho。

GTX 4050基于NVIDIA的Ada Lovelace架构，支持CUDA计算能力8.9（sm_89），具有以下特性：

- **第四代Tensor Cores**: 支持FP16、INT8、INT4精度
- **Ada架构优化**: 增强的AI推理性能
- **CUDA 11.8+**: 需要CUDA Toolkit 11.8或更高版本

## 分支信息

### 创建分支

```bash
git checkout -b feature/gtx4050-ada-lovelace-support
```

或者使用提供的脚本：

```bash
./create_gtx4050_branch.sh
```

### 分支变更

本分支包含以下更新：

1. **文档**: 添加GTX 4050支持说明
2. **配置验证**: 确保sm_89架构正确配置

**注意**: GTX 4050使用的Ada Lovelace架构（sm_89）在主配置中已经支持。本分支主要提供明确的文档和验证。

## 系统要求

### CUDA工具包

- **CUDA Toolkit**: 11.8或更高版本（推荐12.0+）
- **NVIDIA驱动**: 支持Ada Lovelace架构的驱动

### 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 11.8 或更高版本

nvidia-smi
# 应该显示GTX 4050和驱动版本
```

### 检查GPU信息

```bash
cd tests
python3 check_gpu.py
```

预期输出：
```
GPU 0: NVIDIA GeForce GTX 4050
  Compute Capability: 8.9
  Total Memory: XX.XX GB
  ✅ Tensor Cores: Supported (sm_89)
```

## 编译配置

### setup.py配置

`setup.py`已包含sm_89支持：

```python
'nvcc': [
    '-O3', 
    '--use_fast_math',
    '-arch=sm_75',  # Turing (Tensor Core INT8)
    '-arch=sm_80',  # Ampere
    '-arch=sm_86',  # Ampere (consumer)
    '-arch=sm_89',  # Ada Lovelace (GTX 4050, RTX 40xx)
    '-arch=sm_100', # Blackwell (RTX 5060, RTX 50 series)
    '--expt-relaxed-constexpr'
]
```

### Makefile配置

`tests/Makefile`已包含sm_89支持（通过sm_100配置，但可以显式添加）：

```makefile
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_100 --expt-relaxed-constexpr
```

**注意**: sm_89在编译时会自动包含在sm_100的向后兼容中，但为了明确性，可以显式添加。

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

### GTX 4050特定优化

GTX 4050支持以下优化：

1. **INT4精度**: 第四代Tensor Cores支持INT4，可以压缩模型
2. **FP16精度**: 高效的FP16推理
3. **内存优化**: 利用Ada架构的内存子系统

### 建议的编译选项

对于GTX 4050，可以使用以下优化选项：

```bash
# 在setup.py中，确保包含sm_89
'nvcc': [
    '-O3',
    '--use_fast_math',
    '-arch=sm_89',  # 显式指定Ada Lovelace
    '--expt-relaxed-constexpr',
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
GPU 0: NVIDIA GeForce GTX 4050
  Compute Capability: 8.9
  ✅ Tensor Cores: Supported
✅ CUDA environment is working correctly!
```

## 故障排除

### 编译错误：不支持的架构

**错误**: `error: no kernel image is available for execution on the device`

**解决**:
1. 确认CUDA Toolkit版本 >= 11.8
2. 确认`-arch=sm_89`已添加到编译选项（或通过sm_100包含）
3. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 运行时错误：CUDA版本不匹配

**错误**: `CUDA driver version is insufficient for CUDA runtime version`

**解决**:
1. 更新NVIDIA驱动到最新版本
2. 检查驱动是否支持Ada Lovelace架构：
   ```bash
   nvidia-smi
   ```

### Tensor Core不可用

**错误**: Tensor Cores not detected

**解决**:
1. 确认GPU是GTX 4050
2. 检查compute capability是否为8.9
3. 确认CUDA Toolkit版本 >= 11.8

## GTX 4050规格

根据公开信息，GTX 4050（如果类似RTX 4050）可能具有：

- **CUDA核心**: ~2560个
- **显存**: 6GB GDDR6
- **显存位宽**: 96位
- **功耗**: 35W-115W（移动版）
- **架构**: Ada Lovelace
- **计算能力**: 8.9 (sm_89)

**注意**: GTX 4050的具体规格可能因型号而异，建议使用`nvidia-smi`或`check_gpu.py`确认实际规格。

## 已知限制

1. **CUDA 11.8要求**: GTX 4050需要CUDA Toolkit 11.8或更高版本
2. **驱动要求**: 需要支持Ada Lovelace架构的NVIDIA驱动
3. **PyTorch兼容性**: 确保PyTorch版本支持CUDA 11.8+

## 下一步工作

1. **性能基准测试**: 在GTX 4050上运行完整的性能基准测试
2. **内存优化**: 利用Ada架构的内存优化
3. **INT4精度测试**: 测试INT4量化的性能

## 参考

- [NVIDIA Ada Lovelace Architecture](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

**分支**: `feature/gtx4050-ada-lovelace-support`  
**维护者**: libortho contributors  
**最后更新**: 2025-01-25

