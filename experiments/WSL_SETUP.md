# WSL 环境运行指南

本指南说明如何在 WSL (Windows Subsystem for Linux) 环境中运行 libortho 实验。

## 前置要求

### 1. 确保 WSL 已安装并运行

在 Windows PowerShell 中检查：
```powershell
wsl --list --verbose
```

### 2. 进入 WSL 环境

有两种方式：

**方式1：在 Windows Terminal 中打开 WSL**
- 打开 Windows Terminal
- 选择 Ubuntu 标签页（或你的 WSL 发行版）

**方式2：在 PowerShell 中启动 WSL**
```powershell
wsl
```

### 3. 导航到项目目录

```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho
```

## 环境设置

### 检查 Python

```bash
python3 --version
# 应该显示 Python 3.8 或更高版本
```

### 安装依赖

```bash
# 安装 PyTorch (CPU 版本，适合实验)
pip3 install torch numpy

# 或者使用 conda (如果已安装)
conda install pytorch numpy -c pytorch
```

### 验证安装

```bash
python3 -c "import torch; import numpy; print('✅ Dependencies OK')"
```

## 运行实验

### 方式1：运行所有实验（推荐）

```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho
chmod +x experiments/run_all_experiments.sh
./experiments/run_all_experiments.sh
```

### 方式2：单独运行每个实验

```bash
# 实验1：隐私开关测试
python3 experiments/verify_core_logic.py

# 实验2：天才的保留
python3 experiments/saving_genius.py

# 实验3：对偶差分隐私
python3 experiments/dual_dp.py
```

## 预期输出

### 实验1：隐私开关测试

```
--- [LibOrtho] Initializing Minimal Verification ---
Training Loss: 0.xxxxxx
Original Model -> Privacy Error: 0.xxxx (Should be low)
Original Model -> General Error: 0.xxxx (Should be low)
Sieve Complete. Ortho Sparsity: ~95%

--- Testing The Kill Switch ---
[Alpha=1.0] Privacy Error: 0.xxxx (Target: Low)
[Alpha=1.0] General Error: 0.xxxx (Target: Low)
[Alpha=0.0] Privacy Error: X.xxxx (Target: HIGH -> Forgot Privacy!)
[Alpha=0.0] General Error: 0.xxxx (Target: LOW -> Kept Logic!)

✅ SUCCESS: Privacy forgotten (ratio=X.xx)
✅ SUCCESS: General logic preserved (Robust Base)
```

### 实验2：天才的保留

```
============================================================
Experiment 2: Saving the Genius
============================================================

[Phase 1] Training Base Model (Common Patterns Only)...
Base Training Loss: 0.xxxxxx

[Phase 2] Fine-tuning on Mixed Data (Common + Genius)...
Full Training Loss: 0.xxxxxx

[Phase 3] Separating Base and Ortho Components...
Ortho Sparsity: ~95%

[Phase 4] Testing Genius Survival After Base 'Lobotomy'...

--- Before Lobotomy (Normal Base + Ortho) ---
Common Error: 0.xxxx
Genius Error: 0.xxxx

--- After Lobotomy (INT2 Base + Ortho) ---
Common Error: 0.xxxx
Genius Error: 0.xxxx

✅ SUCCESS: Genius reasoning survives Base lobotomy!
```

### 实验3：对偶差分隐私

```
============================================================
Experiment 3: Dual Differential Privacy
============================================================

Privacy Budget: ε=1.0, δ=1e-05

[Phase 1] Training Base Model (Public Knowledge Only)...
[Phase 2] Fine-tuning on Mixed Data (Public + Private)...
[Phase 3] Separating Base and Ortho Components...
[Phase 4] Applying Differential Privacy...
[Phase 5] Evaluating Utility...

✅ SUCCESS: Dual-DP preserves public utility better!
```

## 故障排除

### 问题1：找不到 python3

```bash
# 安装 Python 3
sudo apt update
sudo apt install python3 python3-pip
```

### 问题2：ModuleNotFoundError: No module named 'torch'

```bash
# 安装 PyTorch
pip3 install torch numpy

# 如果使用 conda
conda install pytorch numpy -c pytorch
```

### 问题3：权限 denied

```bash
# 给脚本添加执行权限
chmod +x experiments/run_all_experiments.sh
```

### 问题4：路径问题

确保在项目根目录运行：
```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho
pwd  # 应该显示项目根目录
```

## 性能说明

- 所有实验使用 64x64 的简化模型，运行时间通常 < 1 分钟
- 实验使用固定随机种子（42）确保可复现性
- 如果运行时间过长，可能是依赖安装问题

## 下一步

实验成功后，可以：
1. 查看详细结果输出
2. 修改实验参数（如 epsilon、量化位数等）
3. 扩展到更大的模型（修改 DIM 参数）

## 参考

- 实验说明：`experiments/README.md`
- 理论文档：`docs/1124-新的思路-3.md`
- 项目审查：`docs/PROJECT_REVIEW.md`

