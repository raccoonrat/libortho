# RTX 5060 快速开始指南

## 创建分支

### 方式1：使用脚本（推荐）

```bash
./create_rtx5060_branch.sh
```

### 方式2：手动创建

```bash
git checkout -b feature/rtx5060-blackwell-support
```

## 验证配置

### 1. 检查CUDA版本

```bash
nvcc --version
# 应该显示 CUDA 12.8 或更高版本
```

### 2. 检查GPU

```bash
nvidia-smi
# 应该显示 RTX 5060
```

### 3. 检查GPU信息（Python）

```bash
cd tests
python3 check_gpu.py
```

预期输出：
```
GPU 0: NVIDIA GeForce RTX 5060
  Compute Capability: 10.0
  ✅ Tensor Cores: Supported (sm_100)
```

## 编译和测试

### 安装libortho

```bash
pip install -e .
```

### 编译CUDA测试

```bash
cd tests
make clean
make cuda-test
./test_cuda_kernel
```

### 运行所有测试

```bash
cd tests
./run_all_tests.sh
```

## 配置说明

### 已更新的文件

1. **setup.py**: 添加了`-arch=sm_100`支持
2. **tests/Makefile**: 添加了`-arch=sm_100`支持

### CUDA架构支持

当前支持以下架构：
- `sm_75`: Turing (RTX 20xx)
- `sm_80`: Ampere (A100)
- `sm_86`: Ampere consumer (RTX 30xx)
- `sm_89`: Ada Lovelace (RTX 40xx)
- `sm_100`: **Blackwell (RTX 5060)** ✨

## 故障排除

### 编译错误

如果遇到"no kernel image is available"错误：

1. 确认CUDA Toolkit版本 >= 12.8
2. 清理并重新编译：
   ```bash
   make clean
   pip install -e . --force-reinstall
   ```

### 驱动问题

如果nvidia-smi不显示RTX 5060：

1. 更新NVIDIA驱动到最新版本
2. 重启系统

## 详细文档

更多信息请参考：
- `docs/RTX5060_BLACKWELL_SUPPORT.md` - 完整支持文档
- `tests/GPU_TEST_GUIDE.md` - GPU测试指南

