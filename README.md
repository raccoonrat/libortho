# libortho

> **"Talk is cheap. Show me the code."**

`libortho` 是一个用于 LLM 推理的极简运行时库。它基于一个单一的、简单的观察：

**模型 = 基础（Base） + 修正（Ortho）。**

我们不关心那个"修正"是代表隐私、天才推理，还是单纯的噪声。我们只关心：

1. **Base** 是密集的、低精度的（INT4），为了吞吐量而生。
2. **Ortho** 是稀疏的、高精度的（FP16），为了正确性而生。

我们的工作不是去"理解"模型，而是提供一个**无分支（Branchless）**的路径，让这两者在内存中对齐并在计算单元中融合。

## 核心思想

基于**对偶几何（Dual Geometry）**理论：

- **Base（公共流形）**：低精度量化权重，代表通用知识和常识
- **Ortho（法向分量）**：高精度稀疏权重，代表隐私数据和特异性

通过物理解耦这两个组件，我们可以：
- 动态控制隐私（通过 `alpha` 参数）
- 保持通用能力（Base 不受影响）
- 实现高性能推理（无分支设计）

## 项目结构

```
libortho/
├── include/          # C头文件
├── src/             # CUDA实现
├── tools/           # Python工具（Hessian筛等）
├── torch_bind/      # PyTorch绑定
├── experiments/     # 实验验证脚本
└── setup.py         # 构建配置
```

## 快速开始

### WSL 环境运行（推荐）

如果你使用 WSL (Windows Subsystem for Linux)，请参考：
- **详细指南**: `experiments/WSL_SETUP.md`
- **快速运行**: `./experiments/run_all_experiments.sh`

### 1. 验证核心逻辑

运行最小验证脚本：

```bash
# 在 WSL 中
python3 experiments/verify_core_logic.py

# 或在 Windows 中（如果已配置）
python experiments/verify_core_logic.py
```

如果看到两个 `✅ SUCCESS`，说明理论验证通过。

### 2. 使用 Hessian 筛分离权重

```python
from tools.sieve import hessian_sieve
import torch

weight = torch.randn(4096, 4096)
H_inv = compute_hessian_inverse(...)  # 从GPTQ获取

w_base, w_ortho = hessian_sieve(weight, H_inv, curvature_thresh=10.0)
```

### 3. 使用 PyTorch 模块

```python
from torch_bind.ortho_linear import OrthoLinear

layer = OrthoLinear(in_features=4096, out_features=4096)
layer.set_alpha(1.0)  # 开启完整能力
# layer.set_alpha(0.0)  # 隐私模式（仅Base）

output = layer(input)
```

## 构建

### 在 WSL 中

```bash
# 安装依赖
pip3 install torch numpy pybind11

# 安装包（开发模式）
pip3 install -e .
```

### 在 Windows 中

```bash
pip install torch numpy pybind11
pip install -e .
```

**注意**: CUDA 支持需要 WSL 中安装 CUDA Toolkit 或 Windows 中安装 CUDA。

## 设计原则

1. **拒绝过度设计**：没有抽象工厂，只有 `weight` 和 `residual`
2. **实用主义**：Hessian 筛在 Python 中离线运行，推理在 C++/CUDA 中在线执行
3. **向后兼容**：替换 `nn.Linear` 为 `libortho.Linear` 即可
4. **好品味**：通过将隐私视为加法修正，将复杂的安全问题转化为简单的数学运算

## 理论背景

详见 `docs/1124-新的思路-3.md`。

核心命题：**隐私是公共知识流形的法向分量（Normal Component）**。

## License

MIT

