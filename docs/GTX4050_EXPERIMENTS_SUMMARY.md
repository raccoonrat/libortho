# GTX 4050 真实模型实验总结

**更新日期**: 2025-01-25  
**GPU**: NVIDIA GeForce GTX 4050  
**显存**: 6GB GDDR6  
**架构**: Ada Lovelace (sm_89)

---

## 概述

为了在GTX 4050（6GB显存）上运行真实模型实验，我们创建了专门的优化版本，使用量化和小模型来适配显存限制。

## 新增文件

### 1. `experiments/real_model_experiments_gtx4050.py`

GTX 4050优化的真实模型实验框架：

- **GTX4050RealModelExperimentBase**: 基础类，支持量化加载
- **GTX4050_Experiment1_KillSwitch**: 隐私开关测试（优化版）
- **GTX4050_Experiment2_NullTest**: 效用评估（优化版）

**关键特性**:
- ✅ 4-bit/8-bit量化支持（bitsandbytes）
- ✅ 自动模型选择（小模型优先）
- ✅ 自动内存管理
- ✅ OOM错误自动处理
- ✅ 小批次处理（batch_size=1）

### 2. `experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md`

完整的使用指南，包括：
- 前置要求
- 支持的模型列表
- 运行说明
- 故障排除
- 性能预期

### 3. `experiments/run_gtx4050_experiments.sh`

快速启动脚本，自动检查依赖并运行实验。

## 优化策略

### 1. 量化

- **4-bit量化**: 减少75%显存占用
- **8-bit量化**: 减少50%显存占用（可选）
- **使用bitsandbytes**: 高效的量化实现

### 2. 模型选择

自动选择适合6GB显存的模型：

- **Llama-2-1B**: ~2GB (FP16), ~1GB (4-bit) ✅
- **TinyLlama**: ~2.2GB (FP16), ~1.1GB (4-bit) ✅
- **Phi-2**: ~5GB (FP16), ~2.5GB (4-bit) ✅

### 3. 批次大小

- **默认**: batch_size=1
- **自动调整**: OOM时自动减半
- **内存监控**: 实时显示GPU内存使用

### 4. 样本数量

减少实验样本以节省内存：

- **Canary**: 20个（vs 50个）
- **WikiText**: 50个（vs 100个）
- **GSM8K**: 50个（vs 100个）

## 使用方法

### 快速开始

```bash
# 使用脚本（推荐）
./experiments/run_gtx4050_experiments.sh

# 或手动运行
python experiments/real_model_experiments_gtx4050.py
```

### 自定义选项

```bash
# 使用特定模型
python experiments/real_model_experiments_gtx4050.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 使用8-bit量化
python experiments/real_model_experiments_gtx4050.py \
    --quantization-bits 8

# 禁用量化（仅小模型）
python experiments/real_model_experiments_gtx4050.py \
    --no-quantization
```

## 性能预期

### GTX 4050性能

| 模型 | 量化 | 批次大小 | 速度 | 显存使用 |
|------|------|---------|------|---------|
| Llama-2-1B | 4-bit | 1 | ~15 tokens/sec | ~1.5GB |
| TinyLlama | 4-bit | 1 | ~18 tokens/sec | ~1.8GB |
| Phi-2 | 4-bit | 1 | ~12 tokens/sec | ~3GB |

### 与标准版本的对比

| 版本 | 模型 | 批次大小 | 显存要求 |
|------|------|---------|---------|
| 标准版 | Llama-2-7B | 8 | ~16GB |
| GTX 4050版 | Llama-2-1B (4-bit) | 1 | ~1.5GB |

## 故障排除

### 常见问题

1. **OOM错误**: 使用4-bit量化和小模型
2. **bitsandbytes错误**: 安装bitsandbytes
3. **模型加载失败**: 检查HuggingFace认证

详细故障排除请参考：`experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md`

## 下一步工作

1. **完整LibOrtho集成**: 实现完整的权重分离
2. **性能基准测试**: 在GTX 4050上运行完整测试
3. **内存优化**: 进一步优化内存使用
4. **多模型支持**: 支持更多小模型

## 相关文档

- **使用指南**: `experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md`
- **GPU支持**: `docs/GTX4050_ADA_LOVELACE_SUPPORT.md`
- **快速开始**: `QUICKSTART_GTX4050.md`

---

**维护者**: libortho contributors  
**最后更新**: 2025-01-25

