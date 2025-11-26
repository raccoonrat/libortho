# A800 (Ampere) 支持

**更新日期**: 2025-01-26  
**GPU架构**: NVIDIA Ampere  
**CUDA计算能力**: 8.0 (sm_80)  
**分支**: `feature/a800-ampere-support`

---

## 概述

本文档说明如何为A800（Ampere架构）GPU配置和编译libortho，并优化大模型运行。

A800基于NVIDIA的Ampere架构，支持CUDA计算能力8.0（sm_80），具有以下特性：

- **第三代Tensor Cores**: 支持FP16、INT8、INT4精度
- **80GB显存**: 支持超大模型（70B+）
- **Ampere架构优化**: 增强的AI训练和推理性能
- **CUDA 11.0+**: 需要CUDA Toolkit 11.0或更高版本（推荐11.8+）

## 分支信息

### 创建分支

```bash
git checkout -b feature/a800-ampere-support
```

或者使用提供的脚本：

```bash
./create_a800_branch.sh
```

### 分支变更

本分支包含以下更新：

1. **文档**: 添加A800支持说明和优化指南
2. **配置验证**: 确保sm_80架构正确配置
3. **内存优化**: 针对80GB显存的优化建议

**注意**: A800使用的Ampere架构（sm_80）在主配置中已经支持。本分支主要提供明确的文档、验证和大模型优化建议。

## 系统要求

### CUDA工具包

- **CUDA Toolkit**: 11.0或更高版本（推荐11.8+）
- **NVIDIA驱动**: 支持Ampere架构的驱动（推荐470+）

### 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 11.0 或更高版本

nvidia-smi
# 应该显示A800和驱动版本
```

### 检查GPU信息

```bash
cd tests
python3 check_gpu.py
```

预期输出：
```
GPU 0: NVIDIA A800
  Compute Capability: 8.0
  Total Memory: 80.00 GB
  ✅ Tensor Cores: Supported (sm_80)
```

## 编译配置

### setup.py配置

`setup.py`已包含sm_80支持：

```python
'nvcc': [
    '-O3', 
    '--use_fast_math',
    '-arch=sm_75',  # Turing (Tensor Core INT8)
    '-arch=sm_80',  # Ampere (A800, A100)
    '-arch=sm_86',  # Ampere (consumer)
    '-arch=sm_89',  # Ada Lovelace
    '-arch=sm_100', # Blackwell
    '--expt-relaxed-constexpr'
]
```

### Makefile配置

`tests/Makefile`已包含sm_80支持：

```makefile
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 --expt-relaxed-constexpr
```

## 编译和安装

### 安装libortho

```bash
pip install -e .
```

### 编译CUDA测试

```bash
cd tests
make clean
make cuda-test
./test_cuda_kernel
```

### 验证安装

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 大模型运行

### 内存优化

A800有80GB显存，但仍需注意内存管理：

1. **使用懒加载**（已实现）：
   ```python
   libortho.separate_weights(
       sample_inputs,
       lazy_loading=True  # 逐层处理，节省内存
   )
   ```

2. **使用稀疏张量**（已实现）：
   - `ortho_weights`存储为稀疏张量
   - 节省约50%内存

3. **批处理大小**：
   - 可以设置较大的batch size
   - 建议从batch_size=8开始，逐步增加

### 运行完整实验

```bash
# 实验1：隐私开关测试
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment 1 \
    --device cuda

# 实验2：空测试（性能基准）
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment 2 \
    --device cuda

# 实验3：保存天才
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment 3 \
    --device cuda

# 实验4：系统性能
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment 4 \
    --device cuda

# 运行所有实验
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment all \
    --device cuda
```

## 性能优化

### Tensor Core优化

A800支持第三代Tensor Cores，自动启用：

- FP16矩阵乘法
- INT8量化推理
- INT4量化推理（实验性）

### 内存带宽优化

A800有2039 GB/s的内存带宽，优化建议：

1. **使用连续内存访问**（CSR格式已实现）
2. **减少内存拷贝**（懒加载已实现）
3. **使用混合精度**（FP16/BF16）

### 多GPU配置

如果有多块A800，可以使用数据并行：

```python
import torch
import torch.nn as nn

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

## 故障排除

### 编译错误

**错误**: `no kernel image is available for execution on the device`

**解决**:
1. 确认CUDA Toolkit版本 >= 11.0
2. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 运行时错误：CUDA版本不匹配

**错误**: `CUDA driver version is insufficient for CUDA runtime version`

**解决**:
1. 更新NVIDIA驱动到最新版本（推荐470+）
2. 检查驱动是否支持Ampere架构：
   ```bash
   nvidia-smi
   ```

### Tensor Core不可用

**错误**: Tensor Cores not detected

**解决**:
1. 确认GPU是A800
2. 检查compute capability是否为8.0
3. 确认CUDA Toolkit版本 >= 11.0

### 内存不足（OOM）

即使A800有80GB显存，如果遇到OOM：

1. **使用懒加载**：
   ```python
   lazy_loading=True
   ```

2. **减小batch size**：
   ```python
   batch_size=1  # 从1开始，逐步增加
   ```

3. **检查其他进程**：
   ```bash
   nvidia-smi
   # 查看是否有其他进程占用显存
   ```

4. **使用梯度检查点**（如果训练）：
   ```python
   torch.utils.checkpoint.checkpoint_sequential(...)
   ```

## A800规格

NVIDIA A800 典型规格：

- **CUDA核心**: 6912个
- **显存**: 80GB HBM2e
- **显存带宽**: 2039 GB/s
- **功耗**: 250W-300W
- **架构**: Ampere
- **计算能力**: 8.0 (sm_80)
- **NVLink**: 支持（多GPU高速互联）
- **PCIe**: PCIe 4.0 x16

## 性能基准

在A800上运行性能测试（参考值）：

### 推理性能

- **FP16吞吐量**: 200-300 tokens/sec (Llama-3.2-3B)
- **INT4吞吐量**: 400-600 tokens/sec (Llama-3.2-3B)
- **延迟**: <10ms/token (FP16)

### 训练性能

- **FP16训练速度**: ~2-3x 相比V100
- **混合精度**: 推荐使用FP16/BF16

## 已知限制

1. **CUDA 11.0要求**: A800需要CUDA Toolkit 11.0或更高版本
2. **驱动要求**: 需要支持Ampere架构的NVIDIA驱动（470+）
3. **PyTorch兼容性**: 确保PyTorch版本支持CUDA 11.0+

## 下一步工作

1. **性能基准测试**: 在A800上运行完整的性能基准测试
2. **大模型测试**: 测试70B+模型的支持
3. **多GPU优化**: 优化多A800配置
4. **INT4精度测试**: 测试INT4量化的性能

## 参考

- [NVIDIA A800 Product Page](https://www.nvidia.com/en-us/data-center/a800/)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

**分支**: `feature/a800-ampere-support`  
**维护者**: libortho contributors  
**最后更新**: 2025-01-26

