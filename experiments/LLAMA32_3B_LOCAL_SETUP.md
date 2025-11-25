# Llama 3.2 3B 本地模型使用指南

**模型路径**: `/home/mpcblock/models/Llama-3.2-3B`  
**GPU**: GTX 4050 (6GB VRAM)  
**优势**: 3B模型比8B小，可能不需要量化

---

## 概述

Llama 3.2 3B是一个较小的模型，适合在GTX 4050的6GB显存上运行。相比8B模型，3B模型具有以下优势：

- **更小的显存占用**: ~6GB (FP16) vs ~16GB (8B FP16)
- **可能不需要量化**: 3B模型在6GB显存上可能可以直接运行FP16
- **更快的推理速度**: 参数更少，推理更快
- **本地模型**: 无需HuggingFace认证，加载更快

## 模型要求

### 目录结构

确保模型目录包含以下文件：

```
/home/mpcblock/models/Llama-3.2-3B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors (或 pytorch_model.bin)
└── ... (其他模型文件)
```

### 验证模型目录

```bash
# 检查模型目录
ls -la /home/mpcblock/models/Llama-3.2-3B/

# 应该看到config.json等文件
```

## 使用方法

### 基本使用（默认本地模型）

```bash
# 使用默认本地路径
python experiments/real_model_experiments_gtx4050.py
```

### 指定本地路径

```bash
# 使用自定义本地路径
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B
```

### 不使用量化（推荐尝试）

```bash
# Llama 3.2 3B可能不需要量化
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --no-quantization
```

### 如果需要量化

```bash
# 如果显存不足，启用4-bit量化
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --quantization-bits 4
```

## 性能预期

### Llama 3.2 3B (FP16, 无量化)

- **显存使用**: ~5-6GB
- **推理速度**: ~20-30 tokens/sec
- **批次大小**: 1-2（可能支持）
- **推荐**: ✅ 如果显存充足

### Llama 3.2 3B (4-bit量化)

- **显存使用**: ~2-3GB
- **推理速度**: ~25-35 tokens/sec
- **批次大小**: 2-4
- **推荐**: ✅ 如果显存紧张

## 与8B模型对比

| 模型 | 参数量 | FP16显存 | 4-bit显存 | 速度 | 精度 |
|------|--------|---------|----------|------|------|
| Llama 3.2 3B | 3B | ~6GB | ~2GB | 快 | 良好 |
| Llama 3 8B | 8B | ~16GB | ~4GB | 中等 | 更好 |

## 故障排除

### 问题1: 模型路径不存在

**错误**: `FileNotFoundError` 或 `OSError: Can't load tokenizer`

**解决**:
```bash
# 检查路径
ls -la /home/mpcblock/models/Llama-3.2-3B

# 确认路径正确
python -c "import os; print(os.path.exists('/home/mpcblock/models/Llama-3.2-3B'))"
```

### 问题2: 缺少配置文件

**错误**: `OSError: Can't load config`

**解决**:
1. 确保模型目录包含`config.json`
2. 如果是从HuggingFace下载的，确保下载了所有文件
3. 可以尝试让代码自动下载缺失的配置文件：
   ```bash
   # 代码会自动尝试下载缺失的配置文件
   python experiments/real_model_experiments_gtx4050.py
   ```

### 问题3: OOM错误（即使3B模型）

**解决**:
1. 启用4-bit量化：
   ```bash
   python experiments/real_model_experiments_gtx4050.py --quantization-bits 4
   ```
2. 检查是否有其他程序占用GPU内存
3. 减少批次大小（代码已自动处理）

### 问题4: 模型加载慢

**说明**: 首次加载本地模型可能需要一些时间，这是正常的。

## 优势

### 使用本地模型的优势

1. **无需网络**: 不需要HuggingFace认证或网络连接
2. **加载更快**: 本地文件加载比下载快
3. **离线使用**: 完全离线运行
4. **版本控制**: 可以固定使用特定版本的模型

### Llama 3.2 3B的优势

1. **显存友好**: 适合6GB显存
2. **速度快**: 参数少，推理快
3. **可能无需量化**: 可以直接使用FP16
4. **性能平衡**: 在速度和精度之间取得良好平衡

## 实验配置建议

### 推荐配置

```bash
# 实验1：隐私开关测试
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 1 \
    --no-quantization  # 尝试不使用量化

# 实验2：效用评估
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 2 \
    --no-quantization
```

### 如果显存不足

```bash
# 启用4-bit量化
python experiments/real_model_experiments_gtx4050.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --quantization-bits 4
```

## 验证模型加载

运行以下命令验证模型是否正确加载：

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/home/mpcblock/models/Llama-3.2-3B'
print(f'Loading model from: {model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    local_files_only=False,
    trust_remote_code=True
)

print(f'Model loaded successfully!')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

## 相关文档

- **GTX 4050实验指南**: `experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md`
- **快速开始**: `QUICKSTART_GTX4050.md`
- **GPU支持**: `docs/GTX4050_ADA_LOVELACE_SUPPORT.md`

---

**最后更新**: 2025-01-25  
**维护者**: libortho contributors

