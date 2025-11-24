# libortho 快速开始 (WSL)

## 1. 进入 WSL 环境

在 Windows Terminal 或 PowerShell 中：
```bash
wsl
```

## 2. 导航到项目目录

```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho
```

## 3. 安装依赖

```bash
# 检查 Python
python3 --version

# 安装依赖
pip3 install torch numpy
```

## 4. 运行所有实验

```bash
# 方式1：使用脚本（推荐）
chmod +x experiments/run_all_experiments.sh
./experiments/run_all_experiments.sh

# 方式2：单独运行
python3 experiments/verify_core_logic.py
python3 experiments/saving_genius.py
python3 experiments/dual_dp.py
```

## 预期结果

所有三个实验应该都显示 `✅ SUCCESS`。

## 详细说明

- **WSL 设置指南**: `experiments/WSL_SETUP.md`
- **实验说明**: `experiments/README.md`
- **理论文档**: `docs/1124-新的思路-3.md`

## 故障排除

如果遇到问题，请检查：
1. Python 版本 >= 3.8
2. PyTorch 和 NumPy 已安装
3. 在正确的目录中运行

```bash
# 验证环境
python3 -c "import torch; import numpy; print('✅ OK')"
```

