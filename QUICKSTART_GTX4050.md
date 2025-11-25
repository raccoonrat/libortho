# GTX 4050 快速开始指南

## 创建分支

### 方式1：使用脚本（推荐）

```bash
./create_gtx4050_branch.sh
```

### 方式2：手动创建

```bash
git checkout -b feature/gtx4050-ada-lovelace-support
```

## 验证配置

### 1. 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 11.8 或更高版本
```

### 2. 检查GPU

```bash
nvidia-smi
# 应该显示 GTX 4050
```

### 3. 检查GPU信息（Python）

```bash
cd tests
python3 check_gpu.py
```

预期输出：
```
GPU 0: NVIDIA GeForce GTX 4050
  Compute Capability: 8.9
  ✅ Tensor Cores: Supported (sm_89)
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

## 配置说明

### 架构支持

GTX 4050使用Ada Lovelace架构（sm_89），已在主配置中支持：

- `sm_75`: Turing (RTX 20xx)
- `sm_80`: Ampere (A100)
- `sm_86`: Ampere consumer (RTX 30xx)
- `sm_89`: **Ada Lovelace (GTX 4050, RTX 40xx)** ✨
- `sm_100`: Blackwell (RTX 5060, RTX 50 series)

### 显式配置（可选）

如果需要显式指定sm_89，可以在`setup.py`中确保包含：

```python
'-arch=sm_89',  # Ada Lovelace (GTX 4050)
```

## 性能特性

GTX 4050支持：

- **第四代Tensor Cores**: FP16, INT8, INT4精度
- **Ada架构优化**: 增强的AI推理性能
- **内存优化**: 高效的显存使用

## 故障排除

### 编译错误

如果遇到"no kernel image is available"错误：

1. 确认CUDA Toolkit版本 >= 11.8
2. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 驱动问题

如果nvidia-smi不显示GTX 4050：

1. 更新NVIDIA驱动到最新版本
2. 重启系统

### 验证架构

如果compute capability不是8.9：

1. 确认GPU型号正确
2. 检查驱动是否支持Ada Lovelace
3. 运行`nvidia-smi`查看详细信息

## 运行真实模型实验（Llama 3.2 3B 本地模型）

### 快速开始

```bash
# 运行所有实验（使用本地Llama 3.2 3B模型）
./experiments/run_gtx4050_experiments.sh
```

### 验证本地模型

```bash
# 验证模型是否可以正确加载
python experiments/verify_local_model.py

# 或指定自定义路径
python experiments/verify_local_model.py /path/to/your/model
```

### 手动运行

```bash
# 使用本地Llama 3.2 3B模型（推荐，可能不需要量化）
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment all \
    --no-quantization

# 如果需要量化（显存紧张时）
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --quantization-bits 4
```

### 支持的模型

#### 本地模型（推荐）⭐

- **Llama 3.2 3B** (`/home/mpcblock/models/Llama-3.2-3B`) - 强烈推荐 ✅
  - FP16: ~6GB（可能不需要量化）
  - 4-bit: ~2GB（如果显存紧张）
  - **优势**: 更小更快，本地加载，无需HuggingFace认证

#### HuggingFace模型（需要量化）

- **Llama 3 8B Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`)
  - 必须使用4-bit量化
- **Llama 3.1 8B Instruct** (`meta-llama/Meta-Llama-3.1-8B-Instruct`)
  - 必须使用4-bit量化

**注意**: 
- **Llama 3.2 3B可能不需要量化**（推荐先尝试`--no-quantization`）
- Llama 3 8B模型必须使用4-bit量化才能在6GB显存上运行

### 实验优化

GTX 4050版本自动优化：

- ✅ 4-bit量化（减少75%显存）
- ✅ 小批次处理（batch_size=1）
- ✅ 自动内存管理
- ✅ OOM错误自动处理

## 详细文档

更多信息请参考：
- `docs/GTX4050_ADA_LOVELACE_SUPPORT.md` - 完整支持文档
- `experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md` - 真实模型实验指南
- `tests/GPU_TEST_GUIDE.md` - GPU测试指南

## 注意事项

1. **架构兼容性**: GTX 4050使用sm_89，已在主配置中支持
2. **驱动要求**: 需要支持Ada Lovelace的NVIDIA驱动
3. **CUDA版本**: 推荐CUDA 11.8或更高版本
4. **显存限制**: 6GB显存，需要使用量化和小模型
5. **bitsandbytes**: 真实模型实验需要安装bitsandbytes

