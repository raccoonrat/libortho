# GTX 4050 真实模型实验指南（Llama 3）

**GPU**: NVIDIA GeForce GTX 4050  
**显存**: 6GB GDDR6  
**架构**: Ada Lovelace (sm_89)  
**模型**: Llama 3  
**优化**: 4-bit量化，小批次

---

## 概述

本文档说明如何在GTX 4050（6GB显存）上运行Llama 3模型实验。由于显存限制，我们使用以下优化：

1. **4-bit量化**: 使用bitsandbytes进行4-bit量化，减少显存占用（必需）
2. **Llama 3 8B**: 使用Llama 3 8B模型（4-bit量化后约4GB显存）
3. **小批次**: 批次大小设为1，避免OOM
4. **内存管理**: 自动清理GPU缓存

## 前置要求

### 1. 硬件要求

- **GPU**: GTX 4050 (6GB VRAM)
- **系统内存**: 至少16GB RAM
- **存储**: 至少20GB可用空间（用于模型缓存）

### 2. 软件依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装bitsandbytes（必需）
pip install bitsandbytes

# HuggingFace认证（如果需要访问Llama模型）
huggingface-cli login
```

### 3. 验证GPU

```bash
cd tests
python3 check_gpu.py
```

应该显示：
```
GPU 0: NVIDIA GeForce GTX 4050
  Compute Capability: 8.9
  Total Memory: 6.00 GB
  ✅ Tensor Cores: Supported (sm_89)
```

## 运行实验

### 快速开始

```bash
# 运行所有实验（使用本地Llama 3.2 3B模型，默认路径）
python experiments/real_model_experiments_gtx4050.py

# 或使用快速启动脚本
./experiments/run_gtx4050_experiments.sh

# 指定本地模型路径
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B
```

### 自定义选项

```bash
# 使用Llama 3.1 8B
python experiments/real_model_experiments_gtx4050.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct

# 使用Llama 3 8B Base（非Instruct版本）
python experiments/real_model_experiments_gtx4050.py \
    --model meta-llama/Meta-Llama-3-8B

# 使用8-bit量化（需要更多显存，可能OOM）
python experiments/real_model_experiments_gtx4050.py \
    --quantization-bits 8

# 运行单个实验
python experiments/real_model_experiments_gtx4050.py \
    --experiment 1
```

## 支持的模型

### 推荐模型（适合6GB VRAM）

1. **Llama 3.2 3B** (`/home/mpcblock/models/Llama-3.2-3B`) ⭐⭐⭐ 强烈推荐（默认）
   - FP16: ~6GB（适合6GB显存，可能不需要量化）
   - 4-bit: ~2GB（如果显存紧张）
   - 推荐：✅（本地模型，无需HuggingFace认证）
   - **优势**: 更小更快，可能不需要量化

2. **Llama 3 8B Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`)
   - FP16: ~16GB（不适合6GB显存）
   - 4-bit: ~4GB
   - 推荐：✅（必须使用4-bit量化）

3. **Llama 3 8B Base** (`meta-llama/Meta-Llama-3-8B`)
   - FP16: ~16GB（不适合6GB显存）
   - 4-bit: ~4GB
   - 推荐：✅（必须使用4-bit量化）

4. **Llama 3.1 8B Instruct** (`meta-llama/Meta-Llama-3.1-8B-Instruct`)
   - FP16: ~16GB（不适合6GB显存）
   - 4-bit: ~4GB
   - 推荐：✅（必须使用4-bit量化）

### 不推荐（显存不足）

- **Llama 3 70B**: 需要~140GB（FP16）或~35GB（4-bit）
- **Llama 3 405B**: 需要~810GB（FP16）
- **任何模型不使用量化**: Llama 3 8B需要至少16GB显存

## 实验说明

### 实验1：隐私开关测试（Kill Switch）

**优化**:
- 减少Canary数量：20个（vs 50个）
- 减少WikiText样本：50个（vs 100个）
- 小批次处理

**运行**:
```bash
python experiments/real_model_experiments_gtx4050.py --experiment 1
```

### 实验2：效用评估（Null Test）

**优化**:
- 减少WikiText样本：50个（vs 100个）
- 批次大小：1
- 使用量化模型进行基准测试

**运行**:
```bash
python experiments/real_model_experiments_gtx4050.py --experiment 2
```

## 内存管理

### 自动优化

脚本会自动：
1. 检测GPU显存
2. 如果<8GB，自动启用4-bit量化
3. 如果模型太大，自动切换到小模型
4. 处理OOM错误，自动减少批次大小

### 手动内存清理

如果遇到OOM：

```python
import torch
torch.cuda.empty_cache()
```

## 性能预期

### GTX 4050性能

#### Llama 3.2 3B (推荐)

- **推理速度**: ~20-30 tokens/sec（FP16，无量化）
- **批次大小**: 1-2（可能支持）
- **最大序列长度**: 512-1024 tokens
- **显存使用**: ~5-6GB（FP16，无量化）

#### Llama 3 8B (4-bit量化)

- **推理速度**: ~8-15 tokens/sec（4-bit量化）
- **批次大小**: 1（避免OOM）
- **最大序列长度**: 512 tokens（推荐256-512）
- **显存使用**: ~4-5GB（4-bit量化）

### 与更大GPU的对比

| GPU | 模型 | 量化 | 批次大小 | 速度 |
|-----|------|------|---------|------|
| GTX 4050 (6GB) | Llama 3 8B | 4-bit | 1 | ~10 tokens/sec |
| RTX 4090 (24GB) | Llama 3 8B | FP16 | 8 | ~40 tokens/sec |
| A100 (40GB) | Llama 3 8B | FP16 | 16 | ~60 tokens/sec |

## 故障排除

### OOM错误

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
1. **必须使用4-bit量化**：`--quantization-bits 4`（Llama 3 8B必需）
2. 确保模型名称正确：`--model meta-llama/Meta-Llama-3-8B-Instruct`
3. 减少批次大小（代码中已自动处理）
4. 清理GPU缓存：
   ```python
   torch.cuda.empty_cache()
   ```
5. 如果仍然OOM，检查是否有其他程序占用GPU内存

### bitsandbytes错误

**错误**: `ModuleNotFoundError: No module named 'bitsandbytes'`

**解决**:
```bash
pip install bitsandbytes
```

**注意**: bitsandbytes需要CUDA支持，确保CUDA Toolkit已安装。

### 模型加载失败

**错误**: `OSError: Can't load tokenizer`

**解决**:
1. 检查HuggingFace认证：`huggingface-cli whoami`
2. 检查网络连接
3. 使用本地缓存：`--cache-dir /path/to/cache`

### 量化错误

**错误**: `ValueError: 4-bit quantization requires bitsandbytes`

**解决**:
```bash
pip install bitsandbytes
# 重启Python环境
```

## 结果文件

实验结果保存在 `experiments/results/gtx4050_results_YYYYMMDD_HHMMSS.json`

格式示例：
```json
{
  "exp1_kill_switch": {
    "extraction_rate_alpha1": 0.8,
    "extraction_rate_alpha0": 0.1,
    "privacy_ratio": 8.0,
    "model": "meta-llama/Llama-2-1B-hf",
    "quantization": true
  },
  "exp2_null_test": {
    "ppl_quantized": 15.2,
    "ppl_fp16": 14.4,
    "ppl_int4": 16.0,
    "model": "meta-llama/Llama-2-1B-hf",
    "quantization": true
  }
}
```

## 性能优化建议

### 1. 必须使用4-bit量化（Llama 3 8B）

4-bit量化对于Llama 3 8B是**必需的**：
- 减少75%的显存占用（16GB → 4GB）
- 保持~95%的模型精度
- 适合GTX 4050的6GB显存
- **不使用量化会导致OOM**

### 2. Llama 3模型选择

对于6GB显存：
- ✅ **Llama 3 8B（4-bit）**: 推荐，约4GB显存
- ❌ **Llama 3 8B（FP16）**: 需要16GB，不适合
- ❌ **Llama 3 70B**: 即使4-bit也需要35GB，不适合

### 3. 批次大小

- 推荐：1
- 如果显存充足，可以尝试2-4
- 监控GPU内存使用：`nvidia-smi`

### 4. 序列长度

- 推荐：256-512 tokens
- 避免：>1024 tokens（可能导致OOM）

## 下一步工作

1. **完整LibOrtho集成**: 实现完整的权重分离和alpha开关
2. **性能基准测试**: 在GTX 4050上运行完整的性能测试
3. **内存优化**: 进一步优化内存使用

## 参考

- **GTX 4050支持**: `docs/GTX4050_ADA_LOVELACE_SUPPORT.md`
- **真实模型实验**: `experiments/REAL_MODEL_EXPERIMENTS_README.md`
- **bitsandbytes文档**: https://github.com/TimDettmers/bitsandbytes

---

**最后更新**: 2025-01-25  
**维护者**: libortho contributors

