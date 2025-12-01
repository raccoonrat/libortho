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

### 使用 Pipenv（推荐）

```bash
# 初始化环境
pipenv install --python 3.12

# 安装项目
pipenv run install

# 重新编译
pipenv run rebuild
```

更多编译选项请参考 [DEBUG_BUILD.md](DEBUG_BUILD.md)

## 环境打包和移植

### 打包环境

将编译好的环境打包，用于移植到其他机器：

```bash
# 使用 Pipfile 脚本（推荐）
pipenv run package

# 或直接运行脚本
chmod +x package_environment.sh
./package_environment.sh
```

### 移植到目标机器

```bash
# 1. 解压
tar -xzf libortho_env_*.tar.gz
cd libortho_env_*

# 2. 运行恢复脚本
bash restore_environment.sh

# 3. 验证
cd ../libortho_restored
python3 -c "import libortho._C_ops; print('✅ 导入成功')"
```

⚠️ **兼容性要求**:
- Python 版本应该相同或兼容
- CUDA 版本应该相同或更高（如果使用 CUDA）
- GPU 架构需要支持编译时的架构

如果遇到兼容性问题，在目标机器上重新编译：
```bash
cd libortho_restored
pipenv run rebuild
```

详细说明请参考：
- [PACKAGE_QUICK_REF.md](PACKAGE_QUICK_REF.md) - 打包快速参考
- [PORTING_GUIDE.md](PORTING_GUIDE.md) - 完整移植指南

## 测试

### 运行所有测试

```bash
cd tests
chmod +x run_all_tests.sh
./run_all_tests.sh
```

这将运行：
- GPU 环境检查
- CUDA kernel 测试（如果 GPU 可用）
- CPU forward 测试
- 性能基准测试（Null Test）

详细说明请参考：
- `tests/TESTING_GUIDE.md` - 完整测试指南
- `tests/QUICK_TEST.md` - 快速测试指南

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

