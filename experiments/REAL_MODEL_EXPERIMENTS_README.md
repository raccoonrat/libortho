# 真实模型实验指南

本文档说明如何运行基于真实LLM模型（Llama-2-7B, Llama-3-8B）的实验，这些实验与论文中的基准测试对齐。

## 概述

真实模型实验框架实现了论文中描述的四个核心实验：

1. **实验1：隐私开关测试（Kill Switch）**
   - 使用Canary数据集和WikiText
   - 测试alpha=0.0和alpha=1.0时的Canary提取率

2. **实验2：效用评估（Null Test）**
   - 比较LibOrtho (alpha=0) vs 标准INT4 vs FP16
   - 指标：困惑度（PPL）、MMLU分数

3. **实验3：拯救天才（Saving the Genius）**
   - 使用GSM8K数据集
   - 测试激进Base量化（INT3）下的天才推理保留

4. **实验4：系统性能**
   - 延迟（ms/token）和吞吐量（tokens/sec）基准测试
   - 比较LibOrtho vs INT4 vs FP16

## 前置要求

### 1. 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐A100或RTX 4090)
- **内存**: 至少16GB GPU内存（用于7B模型）
- **存储**: 至少50GB可用空间（用于模型缓存）

### 2. 软件依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果需要访问Llama模型，需要HuggingFace认证
huggingface-cli login
```

### 3. 模型访问

Llama模型需要HuggingFace访问权限：

1. 访问 https://huggingface.co/meta-llama/Llama-2-7b-hf
2. 申请访问权限
3. 使用 `huggingface-cli login` 登录

## 运行实验

### 运行所有实验

```bash
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiment all \
    --device cuda
```

### 运行单个实验

```bash
# 实验1：隐私开关测试
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiment 1 \
    --device cuda

# 实验2：效用评估
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiment 2 \
    --device cuda

# 实验3：拯救天才
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiment 3 \
    --device cuda

# 实验4：系统性能
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiment 4 \
    --device cuda
```

### 使用Llama-3-8B

```bash
python experiments/real_model_experiments.py \
    --model meta-llama/Llama-3-8b \
    --experiment all \
    --device cuda
```

## 实验框架说明

### 当前状态

当前实现是一个**实验框架**，提供了：

1. ✅ 模型加载和基础推理
2. ✅ 数据集加载（WikiText, GSM8K）
3. ✅ 基础指标计算（困惑度、准确率）
4. ✅ 实验结构和结果保存

### 需要完整实现的部分

为了获得论文中描述的完整结果，需要实现：

1. **权重分离（Hessian Sieve）**
   - 对真实模型的所有线性层应用Hessian筛
   - 实现高效的Hessian计算（使用GPTQ或Fisher Information）

2. **量化集成**
   - 集成bitsandbytes或GPTQ进行INT4量化
   - 实现INT3量化用于"拯救天才"实验

3. **LibOrtho运行时集成**
   - 将分离的权重加载到LibOrtho运行时
   - 实现alpha开关机制

4. **完整评估流程**
   - Canary提取率评估
   - MMLU评估
   - GSM8K准确率评估

## 结果解读

实验结果保存在 `experiments/results/real_model_results_YYYYMMDD_HHMMSS.json`

### 实验1结果示例

```json
{
  "exp1_kill_switch": {
    "extraction_rate_alpha1": 0.8,
    "extraction_rate_alpha0": 0.1,
    "privacy_ratio": 8.0,
    "num_canaries": 50
  }
}
```

- `extraction_rate_alpha1`: alpha=1.0时的Canary提取率（应该高）
- `extraction_rate_alpha0`: alpha=0.0时的Canary提取率（应该接近随机）
- `privacy_ratio`: 隐私比率，应该>1.5表示成功

### 实验2结果示例

```json
{
  "exp2_null_test": {
    "ppl_fp16": 15.2,
    "ppl_int4": 16.0,
    "ppl_libortho_alpha0": 16.1,
    "ppl_ratio_libortho": 1.06
  }
}
```

- `ppl_libortho_alpha0`应该接近`ppl_int4`（差异<5%）

### 实验3结果示例

```json
{
  "exp3_saving_genius": {
    "accuracy_fp16": 0.65,
    "accuracy_int3_pure": 0.08,
    "accuracy_libortho_int3_base": 0.62,
    "genius_preservation": 0.95
  }
}
```

- `accuracy_libortho_int3_base`应该>60%，远高于`accuracy_int3_pure`

### 实验4结果示例

```json
{
  "exp4_performance": {
    "latency_fp16_ms_per_token": 25.0,
    "latency_int4_ms_per_token": 12.5,
    "latency_libortho_alpha0_ms_per_token": 12.6,
    "overhead_vs_int4": 0.008
  }
}
```

- `overhead_vs_int4`应该<0.01（<1%开销）

## 下一步工作

1. **实现完整的Hessian筛流程**
   - 对模型的所有线性层应用筛分
   - 优化Hessian计算（使用校准数据集）

2. **集成量化工具**
   - 使用bitsandbytes进行INT4量化
   - 实现自定义INT3量化

3. **实现LibOrtho运行时集成**
   - 将分离的权重转换为LibOrtho格式
   - 实现alpha开关的前向传播

4. **完整评估流程**
   - 实现Canary提取评估
   - 集成MMLU评估
   - 实现GSM8K准确率评估

## 故障排除

### 模型加载失败

如果遇到模型加载错误：

1. 检查HuggingFace认证：
   ```bash
   huggingface-cli whoami
   ```

2. 检查模型访问权限：
   - 访问 https://huggingface.co/meta-llama/Llama-2-7b-hf
   - 确保已获得访问权限

3. 使用缓存目录：
   ```bash
   python experiments/real_model_experiments.py \
       --cache-dir /path/to/cache
   ```

### GPU内存不足

如果遇到OOM错误：

1. 使用CPU模式（较慢）：
   ```bash
   python experiments/real_model_experiments.py --device cpu
   ```

2. 减少批次大小和样本数量（修改代码中的参数）

3. 使用8-bit量化加载模型（需要修改代码）

### 数据集加载失败

如果数据集加载失败：

1. 检查网络连接
2. 使用离线模式（如果数据集已下载）
3. 代码会自动回退到合成数据

## 参考

- 论文：`docs/libortho_paper_zh.pdf`
- 对齐文档：`docs/CODE_PAPER_ALIGNMENT_CHECK.md`
- 项目对齐：`docs/PROJECT_PAPER_ALIGNMENT.md`

