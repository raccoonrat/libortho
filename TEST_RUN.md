# 测试运行指南

## 在 WSL 中运行测试

### 方法1：使用测试脚本（推荐）

```bash
# 在 WSL 终端中
cd /home/mpcblock/lab/github.com/raccoonrat/libortho

# 给脚本添加执行权限
chmod +x test_experiments.sh

# 运行测试
./test_experiments.sh
```

### 方法2：使用实验运行脚本

```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho

chmod +x experiments/run_all_experiments.sh
./experiments/run_all_experiments.sh
```

### 方法3：单独运行每个实验

```bash
cd /home/mpcblock/lab/github.com/raccoonrat/libortho

# 实验1：隐私开关测试
python3 experiments/verify_core_logic.py

# 实验2：天才的保留
python3 experiments/saving_genius.py

# 实验3：对偶差分隐私
python3 experiments/dual_dp.py
```

## 预期输出

### 成功标志

每个实验应该显示：
- ✅ SUCCESS 消息
- 退出码为 0

### 实验1 预期输出示例

```
--- [LibOrtho] Initializing Minimal Verification ---
Training Loss: 0.000xxx
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

## 故障排除

### 问题1：找不到 python3

```bash
# 安装 Python 3
sudo apt update
sudo apt install python3 python3-pip
```

### 问题2：ModuleNotFoundError: No module named 'torch'

```bash
# 安装 PyTorch 和 NumPy
pip3 install torch numpy

# 验证安装
python3 -c "import torch; import numpy; print('OK')"
```

### 问题3：权限 denied

```bash
# 添加执行权限
chmod +x test_experiments.sh
chmod +x experiments/run_all_experiments.sh
```

### 问题4：实验失败

如果实验失败，检查：
1. 随机种子是否固定（应该是42）
2. 依赖版本是否兼容
3. 查看详细错误信息

## 快速验证

快速检查环境是否就绪：

```bash
python3 -c "
import torch
import numpy
print('✅ Python:', __import__('sys').version.split()[0])
print('✅ PyTorch:', torch.__version__)
print('✅ NumPy:', numpy.__version__)
print('✅ Environment ready!')
"
```

## 下一步

测试成功后：
1. 查看详细输出了解实验结果
2. 阅读 `experiments/README.md` 了解实验设计
3. 阅读 `docs/1124-新的思路-3.md` 了解理论背景

