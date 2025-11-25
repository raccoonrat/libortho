# 修复Tokenizer加载错误

## 错误信息

```
❌ Tokenizer加载失败: 'dict' object has no attribute 'model_type'
```

## 原因分析

这个错误通常是因为：

1. **tokenizer_config.json格式问题**: 配置文件可能格式不正确
2. **config.json被误读**: transformers库可能将tokenizer_config.json误读为config.json
3. **缺少必要的tokenizer文件**: tokenizer.json或tokenizer_config.json可能缺失

## 解决方案

### 方案1: 使用诊断工具

```bash
# 诊断模型目录
python experiments/diagnose_model.py

# 这会显示：
# - 目录中所有文件
# - 关键文件是否存在
# - config.json和tokenizer_config.json的内容
```

### 方案2: 检查并修复配置文件

```bash
# 检查tokenizer_config.json
cat /home/mpcblock/models/Llama-3.2-3B/tokenizer_config.json

# 如果文件损坏或格式错误，可以：
# 1. 从HuggingFace重新下载
# 2. 或手动修复格式
```

### 方案3: 重新下载模型（推荐）

```bash
# 使用huggingface-cli下载完整模型
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir /home/mpcblock/models/Llama-3.2-3B \
    --local-dir-use-symlinks False
```

### 方案4: 使用代码的自动修复

代码已经实现了多种fallback方法：

1. **AutoTokenizer (standard)**: 标准方法
2. **AutoTokenizer (use_fast=False)**: 不使用fast tokenizer
3. **LlamaTokenizer**: 直接使用LlamaTokenizer
4. **从tokenizer.json加载**: 直接从tokenizer.json文件加载

如果所有方法都失败，代码会显示详细的错误信息。

## 验证修复

```bash
# 运行验证脚本
python experiments/verify_local_model.py

# 如果成功，会显示：
# ✅ Tokenizer加载成功
# ✅ 模型加载成功
# ✅ 模型推理测试成功
```

## 常见问题

### Q: 为什么会出现这个错误？

A: 通常是因为模型目录不完整，或者tokenizer配置文件格式不正确。

### Q: 如何确保模型目录完整？

A: 使用`huggingface-cli download`下载完整模型，确保包含所有文件。

### Q: 可以手动修复吗？

A: 可以，但建议重新下载。如果必须手动修复，需要确保：
- `tokenizer_config.json`格式正确（有效的JSON）
- `tokenizer.json`存在
- `config.json`格式正确

## 预防措施

1. **使用官方下载工具**: 使用`huggingface-cli`而不是手动下载
2. **验证下载完整性**: 下载后运行诊断工具
3. **保持文件结构**: 不要手动修改模型目录中的文件

---

**相关工具**:
- `experiments/diagnose_model.py` - 诊断模型目录
- `experiments/verify_local_model.py` - 验证模型加载

