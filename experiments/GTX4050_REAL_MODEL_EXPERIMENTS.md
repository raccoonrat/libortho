# GTX 4050 真实模型实验指南

**GPU**: NVIDIA GeForce GTX 4050  
**显存**: 6GB GDDR6  
**架构**: Ada Lovelace (sm_89)  
**优化**: 4-bit量化，小模型，小批次

---

## 概述

本文档说明如何在GTX 4050（6GB显存）上运行真实模型实验。由于显存限制，我们使用以下优化：

1. **4-bit量化**: 使用bitsandbytes进行4-bit量化，减少显存占用
2. **小模型**: 使用Llama-2-1B或TinyLlama替代7B模型
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
# 运行所有实验（使用默认小模型和4-bit量化）
python experiments/real_model_experiments_gtx4050.py
```

### 自定义选项

```bash
# 使用特定模型
python experiments/real_model_experiments_gtx4050.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 使用8-bit量化（需要更多显存）
python experiments/real_model_experiments_gtx4050.py \
    --quantization-bits 8

# 禁用量化（仅适用于非常小的模型）
python experiments/real_model_experiments_gtx4050.py \
    --no-quantization \
    --model meta-llama/Llama-2-1B-hf

# 运行单个实验
python experiments/real_model_experiments_gtx4050.py \
    --experiment 1
```

## 支持的模型

### 推荐模型（适合6GB VRAM）

1. **Llama-2-1B** (`meta-llama/Llama-2-1B-hf`)
   - FP16: ~2GB
   - 4-bit: ~1GB
   - 推荐：✅

2. **TinyLlama** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
   - FP16: ~2.2GB
   - 4-bit: ~1.1GB
   - 推荐：✅

3. **Phi-2** (`microsoft/phi-2`)
   - FP16: ~5GB
   - 4-bit: ~2.5GB
   - 推荐：✅（4-bit）

### 不推荐（显存不足）

- **Llama-2-7B**: 需要~14GB（FP16）或~4GB（4-bit，可能仍不够）
- **Llama-3-8B**: 需要~16GB（FP16）

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

- **推理速度**: ~10-20 tokens/sec（4-bit量化，1B模型）
- **批次大小**: 1（避免OOM）
- **最大序列长度**: 512 tokens（推荐256）

### 与更大GPU的对比

| GPU | 模型 | 批次大小 | 速度 |
|-----|------|---------|------|
| GTX 4050 (6GB) | Llama-2-1B (4-bit) | 1 | ~15 tokens/sec |
| RTX 4090 (24GB) | Llama-2-7B (FP16) | 8 | ~50 tokens/sec |

## 故障排除

### OOM错误

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
1. 确保使用4-bit量化：`--quantization-bits 4`
2. 使用更小的模型：`--model meta-llama/Llama-2-1B-hf`
3. 减少批次大小（代码中已自动处理）
4. 清理GPU缓存：
   ```python
   torch.cuda.empty_cache()
   ```

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

### 1. 使用4-bit量化

4-bit量化可以：
- 减少75%的显存占用
- 保持~95%的模型精度
- 适合GTX 4050的6GB显存

### 2. 选择小模型

对于6GB显存：
- ✅ 1B-2B模型（4-bit）
- ⚠️ 3B-5B模型（4-bit，可能OOM）
- ❌ 7B+模型（即使4-bit也可能OOM）

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

