# A800 快速开始指南

## 创建分支

### 方式1：使用脚本（推荐）

```bash
./create_a800_branch.sh
```

### 方式2：手动创建

```bash
git checkout -b feature/a800-ampere-support
```

## 验证配置

### 1. 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 11.0 或更高版本（推荐 11.8+）
```

### 2. 检查GPU

```bash
nvidia-smi
# 应该显示 A800
```

### 3. 检查GPU信息（Python）

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

## 编译和测试

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

### 运行所有测试

```bash
cd tests
./run_all_tests.sh
```

## A800 特定配置

### 架构支持

A800使用Ampere架构（sm_80），已在主配置中支持：

- `sm_75`: Turing (RTX 20xx)
- `sm_80`: **Ampere (A800, A100)** ✨
- `sm_86`: Ampere consumer (RTX 30xx)
- `sm_89`: Ada Lovelace (RTX 40xx)
- `sm_100`: Blackwell (RTX 5060)

### 显式配置（可选）

如果需要显式指定sm_80，可以在`setup.py`中确保包含：

```python
'-arch=sm_80',  # Ampere (A800, A100)
```

## 大模型支持

A800 通常配备 80GB 显存，非常适合运行大模型：

### 运行完整实验

```bash
# 使用懒加载模式（推荐）
python3 experiments/complete_real_model_experiments.py \
    --model /path/to/Llama-3.2-3B \
    --experiment all \
    --device cuda
```

### 内存优化

A800 支持：
- **大batch size**: 利用80GB显存
- **多GPU**: 如果有多块A800，可以使用数据并行
- **混合精度**: FP16/BF16训练和推理

## 性能特性

A800支持：

- **第三代Tensor Cores**: FP16, INT8, INT4精度
- **Ampere架构优化**: 增强的AI训练和推理性能
- **80GB显存**: 支持超大模型（如70B+）
- **NVLink**: 多GPU高速互联

## 故障排除

### 编译错误

如果遇到"no kernel image is available"错误：

1. 确认CUDA Toolkit版本 >= 11.0（推荐 11.8+）
2. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 驱动问题

如果nvidia-smi不显示A800：

1. 更新NVIDIA驱动到最新版本（推荐 470+）
2. 重启系统
3. 检查PCIe连接

### 验证架构

如果compute capability不是8.0：

1. 确认GPU型号正确
2. 检查驱动是否支持Ampere架构
3. 运行：`nvidia-smi --query-gpu=compute_cap --format=csv`

### 内存不足

即使A800有80GB显存，如果遇到OOM：

1. 使用懒加载模式（`lazy_loading=True`）
2. 减小batch size
3. 使用梯度检查点（gradient checkpointing）
4. 检查是否有其他进程占用显存

## A800 规格

NVIDIA A800 典型规格：

- **CUDA核心**: 6912个
- **显存**: 80GB HBM2e
- **显存带宽**: 2039 GB/s
- **功耗**: 250W-300W
- **架构**: Ampere
- **计算能力**: 8.0 (sm_80)
- **NVLink**: 支持（多GPU）

## 多GPU配置

如果有多块A800：

```python
import torch

# 检查可用GPU数量
print(f"Available GPUs: {torch.cuda.device_count()}")

# 使用多GPU
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## 性能基准

在A800上运行性能测试：

```bash
python3 experiments/complete_real_model_experiments.py \
    --experiment 4 \
    --device cuda
```

预期性能（参考）：
- **FP16吞吐量**: ~200-300 tokens/sec (取决于模型大小)
- **INT4吞吐量**: ~400-600 tokens/sec
- **延迟**: <10ms/token (FP16)

## 参考

- [NVIDIA A800 Product Page](https://www.nvidia.com/en-us/data-center/a800/)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)

