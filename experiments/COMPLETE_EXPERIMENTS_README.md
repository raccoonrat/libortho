# 完整真实模型实验指南

本文档说明如何运行论文中要求的完整真实模型实验。

## 实验概览

根据论文要求，我们实现了四个核心实验：

### 实验1：隐私开关测试（Privacy Kill Switch）

**论文要求**:
- 训练模型记忆Canary IDs（隐私数据）+ WikiText（通用知识）
- 使用Hessian Sieve分离Base和Ortho
- 测试 `alpha = 1.0` 和 `alpha = 0.0`
- **指标**: Canary提取率 vs Alpha

**预期结果**:
- Extraction Rate (alpha=1.0): 高提取率（模型记住了Canary）
- Extraction Rate (alpha=0.0): 接近随机（隐私被遗忘）
- Privacy Ratio: > 1.5x

### 实验2：效用评估（Null Test）

**论文要求**:
- 比较LibOrtho (alpha=0) vs 标准INT4 vs FP16
- **指标**: Perplexity (PPL) 和 MMLU Score
- **预期结果**: LibOrtho (alpha=0) 应该匹配INT4 PPL

### 实验3：拯救天才（Saving the Genius）

**论文要求**:
- 数据集：GSM8K（数学推理）
- 对Base应用极端量化（INT3）
- 保持Ortho FP16
- **指标**: GSM8K准确率
- **预期结果**: GSM8K分数保持高（60%+），而纯INT3崩溃（<10%）

### 实验4：系统性能

**论文要求**:
- **指标**: 延迟（ms/token），吞吐量（tokens/sec）
- 与bitsandbytes INT4比较
- **预期结果**: <1%开销，2x快于FP16

## 快速开始

### 1. 安装依赖

```bash
pip install transformers datasets accelerate bitsandbytes torch numpy
```

### 2. 运行所有实验

```bash
# 使用本地模型
bash experiments/run_complete_experiments.sh /home/mpcblock/models/Llama-3.2-3B

# 或使用Python直接运行
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment all \
    --no-quantization \
    --output-dir experiments/results
```

### 3. 运行单个实验

```bash
# 实验1：隐私开关测试
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 1

# 实验2：效用评估
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 2

# 实验3：拯救天才
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 3

# 实验4：系统性能
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment 4
```

## 参数说明

- `--model`: 模型路径或HuggingFace ID（默认: `/home/mpcblock/models/Llama-3.2-3B`）
- `--experiment`: 实验编号（1, 2, 3, 4, 或 all）
- `--device`: 设备（cuda 或 cpu，默认: cuda）
- `--no-quantization`: 禁用量化（Llama 3.2 3B在6GB VRAM上可能不需要量化）
- `--output-dir`: 结果输出目录（默认: `experiments/results`）

## 输出文件

所有结果保存在 `experiments/results/` 目录：

```
experiments/results/
├── exp1_killswitch_YYYYMMDD_HHMMSS.json      # 实验1结果
├── exp2_nulltest_YYYYMMDD_HHMMSS.json         # 实验2结果
├── exp3_savinggenius_YYYYMMDD_HHMMSS.json     # 实验3结果
├── exp4_performance_YYYYMMDD_HHMMSS.json       # 实验4结果
└── all_results_YYYYMMDD_HHMMSS.json           # 所有结果汇总
```

## 结果解读

### 实验1：隐私开关测试

```json
{
  "extraction_rate_alpha1": 0.80,      // alpha=1.0时的提取率
  "extraction_rate_alpha0": 0.10,      // alpha=0.0时的提取率
  "privacy_ratio": 8.00,               // 隐私比率（应该>1.5）
  "ortho_sparsity": 0.05               // Ortho稀疏度（通常1-5%）
}
```

**成功标准**: `privacy_ratio > 1.5`

### 实验2：效用评估

```json
{
  "ppl_fp16": 15.2,                     // FP16困惑度
  "ppl_int4": 16.0,                    // INT4困惑度
  "ppl_libortho_alpha0": 16.1,        // LibOrtho alpha=0困惑度
  "ppl_ratio_libortho": 1.06          // 比率（应该接近1.0）
}
```

**成功标准**: `ppl_ratio_libortho` 应该接近 `ppl_ratio_int4`

### 实验3：拯救天才

```json
{
  "accuracy_base_ortho": 0.65,         // Base INT3 + Ortho FP16准确率
  "accuracy_base_only": 0.12,         // Base INT3 only准确率
  "accuracy_pure_int3": 0.08,          // 纯INT3准确率（模拟）
  "relative_preservation": 5.42        // 相对保留率（应该<0.5表示天才保留更好）
}
```

**成功标准**: `accuracy_base_ortho > 0.60` 且 `accuracy_pure_int3 < 0.10`

### 实验4：系统性能

```json
{
  "latency_per_token_fp16_ms": 25.0,   // FP16延迟
  "latency_per_token_base_ms": 12.5,  // Base only延迟
  "throughput_fp16_tokens_per_sec": 40.0,
  "throughput_base_tokens_per_sec": 80.0,
  "speedup": 2.0                       // 加速比（应该>1.5）
}
```

**成功标准**: `speedup > 1.5` 且 `latency_per_token_base_ms < latency_per_token_fp16_ms * 0.6`

## 注意事项

1. **内存管理**: Llama 3.2 3B在6GB VRAM上可能不需要量化，但如果遇到OOM，可以移除 `--no-quantization` 标志。

2. **数据集下载**: 首次运行时会自动下载WikiText和GSM8K数据集。确保网络连接正常。

3. **Hessian计算**: Hessian计算可能需要一些时间，特别是对于大模型。代码使用对角近似以加速。

4. **Canary训练**: 当前实现简化了Canary训练过程。完整实现需要使用LoRA或PEFT进行高效微调。

5. **MMLU评估**: 当前MMLU评估是简化版本。完整实现需要使用专门的MMLU评估脚本。

## 故障排除

### OOM错误

```bash
# 启用量化
python experiments/complete_real_model_experiments.py \
    --model /home/mpcblock/models/Llama-3.2-3B \
    --experiment all
    # 移除 --no-quantization
```

### 模型加载失败

```bash
# 检查模型路径
ls -la /home/mpcblock/models/Llama-3.2-3B

# 验证模型
python experiments/verify_local_model.py
```

### 数据集下载失败

```bash
# 手动下载数据集
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"
```

## 论文集成

### 提取结果到论文

1. **提取数值**: 从JSON文件中提取关键指标
2. **生成表格**: 使用结果生成LaTeX表格
3. **绘制图表**: 使用结果绘制论文图表

### 示例：提取关键指标

```python
import json

with open('experiments/results/all_results_20241125_120000.json') as f:
    results = json.load(f)

# 实验1
exp1 = results['experiment1_killswitch']
print(f"Privacy Ratio: {exp1['privacy_ratio']:.2f}x")

# 实验2
exp2 = results['experiment2_nulltest']
print(f"PPL Ratio: {exp2['ppl_ratio_libortho']:.2f}")

# 实验3
exp3 = results['experiment3_savinggenius']
print(f"Genius Accuracy: {exp3['accuracy_base_ortho']:.2%}")

# 实验4
exp4 = results['experiment4_performance']
print(f"Speedup: {exp4['speedup']:.2f}x")
```

## 下一步

1. **运行完整实验**: 使用上述命令运行所有实验
2. **检查结果**: 验证结果是否符合论文预期
3. **生成图表**: 使用结果生成论文图表
4. **撰写论文**: 将结果整合到论文中

---

**相关文件**:
- `experiments/complete_real_model_experiments.py` - 完整实验实现
- `experiments/run_complete_experiments.sh` - 快速运行脚本
- `experiments/verify_local_model.py` - 模型验证工具

