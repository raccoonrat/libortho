# A800 故障排除指南

## 问题 1: pip install 失败 - TypeError: unhashable type: 'slice'

### 错误信息
```
TypeError: unhashable type: 'slice'
File ".../pybind11/setup_helpers.py", line 117, in _add_cflags
    self.extra_compile_args[:0] = flags
```

### 原因
pybind11 的 `Pybind11Extension` 在某些版本中处理 `extra_compile_args` 字典时会出现问题。

### 解决方案

**方案1：使用修复后的 setup.py（推荐）**

已修复 `setup.py`，使用标准的 `Extension` 而不是 `Pybind11Extension`：

```python
from setuptools import Extension
from pybind11.setup_helpers import build_ext

ext_modules.append(
    Extension(
        "libortho._C_ops",
        [...],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': nvcc_flags
        },
        ...
    )
)
```

**方案2：降级 pybind11（临时方案）**

如果方案1不工作，可以尝试降级 pybind11：

```bash
pip install "pybind11<2.12.0"
pip install -e . --force-reinstall
```

**方案3：升级 pybind11**

或者升级到最新版本：

```bash
pip install --upgrade pybind11
pip install -e . --force-reinstall
```

---

## 问题 2: CUDA kernel 编译成功但运行时失败

### 错误信息
```
❌ Kernel launch failed: no kernel image is available for execution on the device
```

### 原因
虽然编译时指定了多个架构（`-arch=sm_75 -arch=sm_80 ...`），但可能没有正确生成 A800 (sm_80) 的 kernel 镜像。

### 解决方案

**方案1：使用 -gencode（推荐，已修复）**

已修复 `tests/Makefile`，使用 `-gencode` 而不是 `-arch`：

```makefile
# 修复前（可能有问题）
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89

# 修复后（推荐）
CUDA_CFLAGS = -O3 --expt-relaxed-constexpr
CUDA_CFLAGS += -gencode arch=compute_75,code=sm_75
CUDA_CFLAGS += -gencode arch=compute_80,code=sm_80
CUDA_CFLAGS += -gencode arch=compute_86,code=sm_86
CUDA_CFLAGS += -gencode arch=compute_89,code=sm_89
```

**方案2：只编译 sm_80（快速测试）**

如果只需要 A800，可以只编译 sm_80：

```bash
cd tests
make clean
nvcc -O3 -gencode arch=compute_80,code=sm_80 --expt-relaxed-constexpr \
     -o test_cuda_kernel test_cuda_kernel.cu -lcudart
./test_cuda_kernel
```

**方案3：检查 CUDA 版本兼容性**

确认 CUDA 版本支持 sm_80：

```bash
nvcc --version
# 需要 CUDA 11.0+ (推荐 11.8+)

nvidia-smi
# 检查驱动版本，推荐 470+
```

---

## 问题 3: 编译警告 - incompatible redefinition

### 警告信息
```
nvcc warning : incompatible redefinition for option 'gpu-architecture'
```

### 原因
使用多个 `-arch` 选项时，nvcc 会警告，但这是正常的。使用 `-gencode` 可以避免这个警告。

### 解决方案
已修复，使用 `-gencode` 代替 `-arch`。

---

## 验证修复

### 1. 重新编译测试

```bash
cd tests
make clean
make cuda-test
```

预期输出：
```
✅ CUDA devices found: 1
GPU 0: NVIDIA A800-SXM4-80GB
  Compute Capability: 8.0
  ✅ Tensor Cores: Supported
✅ Kernel execution test PASSED
✅ CUDA environment is working correctly!
```

### 2. 重新安装 Python 包

```bash
pip install -e . --force-reinstall
```

应该不再出现 `TypeError: unhashable type: 'slice'` 错误。

### 3. 验证 Python 导入

```python
python3 -c "import libortho._C_ops; print('✅ Import successful')"
```

---

## 其他常见问题

### CUDA 版本不匹配

**错误**: `CUDA driver version is insufficient`

**解决**:
```bash
# 检查驱动版本
nvidia-smi

# 更新驱动（如果需要）
# 在 Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-470  # 或更新版本
```

### 找不到 nvcc

**错误**: `nvcc: command not found`

**解决**:
```bash
# 检查 CUDA 安装
which nvcc

# 如果未安装，安装 CUDA Toolkit
# 参考: https://developer.nvidia.com/cuda-downloads
```

### PyTorch CUDA 不匹配

**错误**: PyTorch 无法使用 CUDA

**解决**:
```bash
# 检查 PyTorch CUDA 版本
python3 -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 PyTorch
# 参考: https://pytorch.org/get-started/locally/
```

---

## 联系支持

如果以上方案都无法解决问题，请提供：

1. CUDA 版本：`nvcc --version`
2. 驱动版本：`nvidia-smi`
3. Python 版本：`python3 --version`
4. PyTorch 版本：`python3 -c "import torch; print(torch.__version__)"`
5. pybind11 版本：`python3 -c "import pybind11; print(pybind11.__version__)"`
6. 完整错误信息

---

**最后更新**: 2025-01-26

