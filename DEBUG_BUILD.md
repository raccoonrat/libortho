# 编译调试指南 (Debug Build Guide)

本指南说明如何在 `pip install -e .` 编译时进行调试。

## 方法 1: 使用环境变量启用调试模式

### 启用调试构建（包含调试符号，无优化）

```bash
# 启用调试模式
export LIBORTHO_DEBUG=1
pip install -e .
```

### 启用详细输出模式

```bash
# 启用详细输出（显示编译参数等信息）
export LIBORTHO_VERBOSE=1
pip install -e .
```

### 同时启用调试和详细输出

```bash
export LIBORTHO_DEBUG=1
export LIBORTHO_VERBOSE=1
pip install -e .
```

## 方法 2: 使用 pip 的详细输出选项

### 基本详细输出

```bash
pip install -e . -v
```

### 更详细的输出

```bash
pip install -e . -vv
```

### 最详细的输出

```bash
pip install -e . -vvv
```

## 方法 3: 保存编译日志到文件

```bash
# 保存所有输出到日志文件
pip install -e . -v 2>&1 | tee build.log

# 或者同时启用调试模式
export LIBORTHO_DEBUG=1
export LIBORTHO_VERBOSE=1
pip install -e . -vv 2>&1 | tee build_debug.log
```

## 方法 4: 分步编译（更细粒度的调试）

### 步骤 1: 只构建扩展模块（不安装）

```bash
python setup.py build_ext --inplace
```

### 步骤 2: 查看构建的详细信息

```bash
python setup.py build_ext --inplace -v
```

### 步骤 3: 使用调试模式构建

```bash
export LIBORTHO_DEBUG=1
export LIBORTHO_VERBOSE=1
python setup.py build_ext --inplace
```

## 方法 5: 使用 CUDA 调试工具

### 检查 CUDA 编译命令

编译时会显示完整的 nvcc 命令，可以复制并手动运行以调试。

### 使用 cuda-gdb 调试

如果使用调试模式编译（`LIBORTHO_DEBUG=1`），可以使用 cuda-gdb：

```bash
# 编译时启用调试模式
export LIBORTHO_DEBUG=1
pip install -e .

# 使用 cuda-gdb 调试
cuda-gdb python
(gdb) run your_script.py
```

## 调试标志说明

### 调试模式 (`LIBORTHO_DEBUG=1`) 使用的标志：

- **NVCC 标志:**
  - `-g`: 生成主机端调试符号
  - `-G`: 生成设备端调试符号
  - `-O0`: 禁用优化
  - `--ptxas-options=-v`: 显示 PTX 汇编详细信息
  - `-lineinfo`: 包含行号信息

- **CXX 标志:**
  - `-g`: 生成调试符号
  - `-O0`: 禁用优化
  - `-fPIC`: 位置无关代码

### 发布模式（默认）使用的标志：

- **NVCC 标志:**
  - `-O3`: 最高优化级别
  - `--use_fast_math`: 使用快速数学库

- **CXX 标志:**
  - `-O3`: 最高优化级别

## 常见问题排查

### 1. 编译卡住或很慢

```bash
# 使用详细输出查看卡在哪里
export LIBORTHO_VERBOSE=1
pip install -e . -vv 2>&1 | tee build.log
```

### 2. CUDA 编译错误

```bash
# 启用详细输出查看完整编译命令
export LIBORTHO_VERBOSE=1
export LIBORTHO_DEBUG=1
pip install -e . -vv 2>&1 | tee build_error.log
```

### 3. 链接错误

检查编译日志中的链接命令，确认所有库路径正确。

### 4. 运行时错误（需要调试符号）

```bash
# 使用调试模式重新编译
export LIBORTHO_DEBUG=1
pip install -e . --force-reinstall --no-cache-dir
```

## 完整调试示例

```bash
# 1. 清理之前的构建
rm -rf build/ *.egg-info/
find . -name "*.so" -delete
find . -name "*.o" -delete

# 2. 启用所有调试选项
export LIBORTHO_DEBUG=1
export LIBORTHO_VERBOSE=1

# 3. 执行安装并保存日志
pip install -e . -vv --no-cache-dir 2>&1 | tee build_debug.log

# 4. 检查日志文件
cat build_debug.log | grep -i error
cat build_debug.log | grep -i warning
```

## 环境变量总结

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `LIBORTHO_DEBUG` | 启用调试构建（包含调试符号，无优化） | `0` |
| `LIBORTHO_VERBOSE` | 启用详细输出（显示编译参数等） | `0` |

## 注意事项

1. **调试模式会显著增加编译时间**：禁用优化和包含调试符号会使编译变慢
2. **调试模式会增加文件大小**：生成的 .so 文件会更大
3. **性能影响**：调试模式编译的代码运行速度会明显变慢，仅用于调试
4. **生产环境**：生产环境应使用默认的发布模式（不设置 `LIBORTHO_DEBUG`）

