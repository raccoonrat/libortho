# 编译调试指南 (Debug Build Guide)

本指南说明如何在 `pip install -e .` 或 `pipenv install` 编译时进行调试。

## 方法 0: 使用 Pipenv 进行调试（推荐）

### 安装 Pipenv（如果还没有）

```bash
pip install pipenv
```

### 初始化 Pipenv 环境（首次使用或指定 Python 版本）

```bash
# 使用默认 Python 版本（Pipfile 中指定的版本）
pipenv install

# 指定 Python 版本（例如 Python 3.12）
pipenv install --python 3.12

# 或者使用 python 可执行文件路径
pipenv install --python /usr/bin/python3.12

# 如果已经存在虚拟环境，需要先删除再重新创建
pipenv --rm
pipenv install --python 3.12
```

**注意**: 如果遇到 "Python 3.x was not found" 错误：
- 确保系统已安装对应版本的 Python
- 可以使用 `python3 --version` 检查可用版本
- 或者使用 `pipenv install --python $(which python3)` 使用系统默认 Python3

### 使用 Pipfile 中的脚本命令

项目已配置了 Pipfile，包含以下便捷脚本：

```bash
# 正常安装
pipenv run install

# 调试模式安装（包含调试符号，无优化）
pipenv run install-debug

# 详细调试模式安装（调试符号 + 详细输出）
pipenv run install-debug-verbose

# 仅详细输出模式（不改变编译选项）
pipenv run install-verbose
```

**注意**: `pipenv run` 脚本命令不支持传递 `--python` 参数。如果需要指定 Python 版本，请先使用 `pipenv install --python 3.12` 初始化环境。

### 使用环境变量（pipenv 会自动加载 .env 文件）

#### 方法 A: 创建 .env 文件

```bash
# 创建 .env 文件
cat > .env << EOF
LIBORTHO_DEBUG=1
LIBORTHO_VERBOSE=1
EOF

# 然后正常安装（pipenv 会自动加载 .env）
pipenv install -e .
```

#### 方法 B: 在命令行中设置环境变量

```bash
# 调试模式安装
LIBORTHO_DEBUG=1 pipenv install -e .

# 详细调试模式安装
LIBORTHO_DEBUG=1 LIBORTHO_VERBOSE=1 pipenv install -e . -v
```

#### 方法 C: 使用 pipenv run 传递环境变量

```bash
# 调试模式
pipenv run env LIBORTHO_DEBUG=1 pip install -e .

# 详细调试模式
pipenv run env LIBORTHO_DEBUG=1 LIBORTHO_VERBOSE=1 pip install -e . -vv
```

### 保存编译日志（pipenv）

```bash
# 使用调试模式并保存日志
LIBORTHO_DEBUG=1 LIBORTHO_VERBOSE=1 pipenv install -e . -v 2>&1 | tee build_debug.log

# 或者使用脚本命令
pipenv run install-debug-verbose 2>&1 | tee build_debug.log
```

### 清理并重新安装（pipenv）

```bash
# 清理构建文件
rm -rf build/ *.egg-info/
find . -name "*.so" -delete
find . -name "*.o" -delete

# 使用调试模式重新安装
LIBORTHO_DEBUG=1 pipenv install -e . --dev
```

## 方法 1: 使用环境变量启用调试模式（直接使用 pip）

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

### 0. Pipenv Python 版本问题

**问题**: `Warning: Python 3.x was not found on your system...`

**解决方案**:

```bash
# 方法 1: 检查系统可用的 Python 版本
python3 --version
which python3

# 方法 2: 使用系统默认 Python3 初始化 pipenv
pipenv install --python $(which python3)

# 方法 3: 如果已有虚拟环境，删除后重新创建
pipenv --rm
pipenv install --python 3.12

# 方法 4: 手动指定 Python 路径
pipenv install --python /usr/bin/python3.12

# 方法 5: 如果使用 pyenv，确保已安装对应版本
pyenv install 3.12.0
pipenv install --python 3.12
```

**注意**: `pipenv run` 脚本命令不支持 `--python` 参数。必须先使用 `pipenv install --python <version>` 初始化环境。

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

### 使用 pip（直接安装）

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

### 使用 pipenv（推荐）

```bash
# 1. 初始化 pipenv（如果还没有，指定 Python 版本）
pipenv install --python 3.12

# 或者使用系统默认 Python3
pipenv install --python $(which python3)

# 2. 清理之前的构建
rm -rf build/ *.egg-info/
find . -name "*.so" -delete
find . -name "*.o" -delete

# 3. 使用调试脚本安装并保存日志
pipenv run install-debug-verbose 2>&1 | tee build_debug.log

# 或者使用环境变量
LIBORTHO_DEBUG=1 LIBORTHO_VERBOSE=1 pipenv install -e . -vv 2>&1 | tee build_debug.log

# 4. 检查日志文件
cat build_debug.log | grep -i error
cat build_debug.log | grep -i warning
```

## 环境变量总结

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `LIBORTHO_DEBUG` | 启用调试构建（包含调试符号，无优化） | `0` |
| `LIBORTHO_VERBOSE` | 启用详细输出（显示编译参数等） | `0` |

## Pipenv 脚本命令总结

| 命令 | 说明 |
|------|------|
| `pipenv run install` | 正常安装（发布模式） |
| `pipenv run install-debug` | 调试模式安装（`LIBORTHO_DEBUG=1`） |
| `pipenv run install-debug-verbose` | 调试模式 + 详细输出 |
| `pipenv run install-verbose` | 仅详细输出（不改变编译选项） |

## 注意事项

1. **调试模式会显著增加编译时间**：禁用优化和包含调试符号会使编译变慢
2. **调试模式会增加文件大小**：生成的 .so 文件会更大
3. **性能影响**：调试模式编译的代码运行速度会明显变慢，仅用于调试
4. **生产环境**：生产环境应使用默认的发布模式（不设置 `LIBORTHO_DEBUG`）

