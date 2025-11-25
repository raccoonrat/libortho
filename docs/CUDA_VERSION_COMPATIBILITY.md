# CUDA版本兼容性指南

**更新日期**: 2025-01-25

---

## 概述

libortho支持多个CUDA架构，但不同架构需要不同的CUDA版本。本文档说明CUDA版本与架构的兼容性。

## 架构与CUDA版本要求

| 架构 | GPU系列 | CUDA版本要求 | 说明 |
|------|---------|-------------|------|
| sm_75 | Turing (RTX 20xx) | CUDA 10.0+ | 基础支持 |
| sm_80 | Ampere (A100) | CUDA 11.0+ | 数据中心GPU |
| sm_86 | Ampere consumer (RTX 30xx) | CUDA 11.0+ | 消费级GPU |
| sm_89 | Ada Lovelace (RTX 40xx, GTX 4050) | CUDA 11.8+ | 最新消费级 |
| sm_100 | Blackwell (RTX 5060, RTX 50 series) | **CUDA 12.8+** | 最新架构 |

## 自动检测

libortho会自动检测CUDA版本并选择支持的架构：

### Makefile自动检测

`tests/Makefile`会自动检测CUDA版本：

```makefile
# 自动检测CUDA版本
CUDA_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
# 只在CUDA >= 12.8时包含sm_100
```

### setup.py自动检测

`setup.py`中的`get_cuda_archs()`函数会自动检测：

```python
def get_cuda_archs():
    base_archs = ['-arch=sm_75', '-arch=sm_80', '-arch=sm_86', '-arch=sm_89']
    # 检测CUDA版本，如果>=12.8，添加sm_100
    ...
```

## 检查CUDA版本

### 方法1：使用nvcc

```bash
nvcc --version
```

输出示例：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.101
```

### 方法2：使用检查脚本

```bash
cd tests
./check_cuda_version.sh
```

输出示例：
```
CUDA Version: 12.3

Supported Architectures:
  ✅ sm_75  (Turing) - CUDA 10.0+
  ✅ sm_80  (Ampere) - CUDA 11.0+
  ✅ sm_86  (Ampere consumer) - CUDA 11.0+
  ✅ sm_89  (Ada Lovelace) - CUDA 11.8+
  ❌ sm_100 (Blackwell) - Requires CUDA 12.8+

⚠️  Your CUDA version does not support sm_100 (Blackwell)
   To use RTX 5060, upgrade to CUDA 12.8 or higher
```

## 常见问题

### 问题1：编译错误 - sm_100不支持

**错误**:
```
nvcc fatal: Value 'sm_100' is not defined for option 'gpu-architecture'
```

**原因**: CUDA版本 < 12.8

**解决**:
1. **选项1（推荐）**: 使用自动检测的Makefile（已修复）
   ```bash
   cd tests
   make clean
   make cuda-test
   ```

2. **选项2**: 手动移除sm_100
   - 编辑`tests/Makefile`，移除`-arch=sm_100`
   - 编辑`setup.py`，移除`-arch=sm_100`

3. **选项3**: 升级CUDA到12.8+
   ```bash
   # 下载并安装CUDA 12.8+
   # https://developer.nvidia.com/cuda-downloads
   ```

### 问题2：需要支持RTX 5060但CUDA版本低

**解决**: 升级CUDA Toolkit到12.8或更高版本

**下载**: https://developer.nvidia.com/cuda-downloads

### 问题3：多CUDA版本环境

如果系统有多个CUDA版本：

```bash
# 检查当前使用的CUDA版本
nvcc --version

# 设置特定CUDA版本（如果安装了多个）
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

## 版本兼容性矩阵

| CUDA版本 | sm_75 | sm_80 | sm_86 | sm_89 | sm_100 |
|---------|-------|-------|-------|-------|--------|
| 10.0-10.2 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 11.0-11.7 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 11.8-12.7 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 12.8+ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 推荐配置

### GTX 4050 (Ada Lovelace)

- **CUDA版本**: 11.8+（推荐12.0+）
- **支持架构**: sm_75, sm_80, sm_86, sm_89
- **不需要**: sm_100

### RTX 5060 (Blackwell)

- **CUDA版本**: 12.8+（必需）
- **支持架构**: sm_75, sm_80, sm_86, sm_89, sm_100
- **注意**: 必须升级CUDA才能使用

## 验证

编译后验证支持的架构：

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

如果编译成功，说明架构配置正确。

## 相关文档

- **GTX 4050支持**: `docs/GTX4050_ADA_LOVELACE_SUPPORT.md`
- **RTX 5060支持**: `docs/RTX5060_BLACKWELL_SUPPORT.md`
- **GPU测试指南**: `tests/GPU_TEST_GUIDE.md`

---

**维护者**: libortho contributors  
**最后更新**: 2025-01-25

