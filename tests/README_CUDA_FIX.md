# CUDA编译错误修复说明

## 问题

如果遇到以下错误：
```
nvcc fatal: Value 'sm_100' is not defined for option 'gpu-architecture'
```

## 原因

`sm_100`（Blackwell架构）需要CUDA 12.8或更高版本。如果您的CUDA版本较低（如11.x或12.0-12.7），就会出现此错误。

## 解决方案

### 方案1：自动检测（已修复）✅

Makefile和setup.py现在会自动检测CUDA版本，只在支持时才包含sm_100。

**无需任何操作**，直接编译：

```bash
cd tests
make clean
make cuda-test
```

### 方案2：检查CUDA版本

```bash
# 检查CUDA版本
nvcc --version

# 或使用检查脚本
cd tests
./check_cuda_version.sh
```

### 方案3：升级CUDA（如果需要RTX 5060支持）

如果您有RTX 5060 GPU并需要支持，请升级到CUDA 12.8+：

1. 下载CUDA 12.8+：https://developer.nvidia.com/cuda-downloads
2. 安装CUDA Toolkit
3. 重新编译

### 方案4：手动移除sm_100（不推荐）

如果确定不需要RTX 5060支持，可以手动编辑：

**tests/Makefile**:
```makefile
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 --expt-relaxed-constexpr
```

**setup.py**:
```python
'nvcc': [
    '-O3', 
    '--use_fast_math',
    '-arch=sm_75',
    '-arch=sm_80',
    '-arch=sm_86',
    '-arch=sm_89',
    '--expt-relaxed-constexpr'
]
```

## 验证

编译成功后，运行测试：

```bash
cd tests
make cuda-test
./test_cuda_kernel
```

如果看到GPU信息输出，说明编译成功。

## 更多信息

详细说明请参考：`docs/CUDA_VERSION_COMPATIBILITY.md`

