# libortho 环境移植指南

本指南说明如何将编译好的 libortho 环境打包并移植到另一台机器。

## 快速开始

### 1. 在源机器上打包环境

```bash
# 确保脚本有执行权限
chmod +x package_environment.sh

# 运行打包脚本
./package_environment.sh
```

打包脚本会创建一个压缩包，例如：`libortho_env_20241201_120000.tar.gz`

### 2. 传输到目标机器

```bash
# 使用 scp 传输（示例）
scp libortho_env_*.tar.gz user@target-machine:/path/to/destination/

# 或使用其他方式（rsync, USB 等）
```

### 3. 在目标机器上恢复环境

```bash
# 解压
tar -xzf libortho_env_*.tar.gz
cd libortho_env_*

# 运行恢复脚本
bash restore_environment.sh
```

## 详细说明

### 打包内容

打包脚本会收集以下内容：

1. **系统信息** (`system_info.txt`)
   - 操作系统信息
   - Python 版本
   - CUDA 版本（如果可用）
   - GPU 信息（如果可用）

2. **环境配置**
   - `Pipfile` 和 `Pipfile.lock`（如果存在）
   - pipenv 虚拟环境路径

3. **编译好的扩展模块** (`compiled_extensions/`)
   - 所有 `.so` 文件
   - `build/` 目录
   - `*.egg-info/` 目录

4. **源代码** (`source_code/`)
   - 项目源代码（排除构建文件）

5. **依赖信息**
   - `requirements_installed.txt`: 已安装的 Python 包列表
   - `dependencies_info.txt`: 系统库依赖信息

6. **恢复脚本** (`restore_environment.sh`)
   - 自动恢复环境的脚本

### 兼容性要求

⚠️ **重要**: 移植环境时需要考虑以下兼容性：

#### 1. Python 版本兼容性

- **推荐**: Python 版本应该完全相同（例如都是 3.12.0）
- **最低要求**: 主版本号相同（例如都是 3.x）
- **检查方法**: 查看 `system_info.txt` 中的 Python 版本

```bash
# 在目标机器上检查
python3 --version
```

#### 2. CUDA 版本兼容性

如果项目包含 CUDA 扩展：

- **推荐**: CUDA 版本相同或更高
- **最低要求**: CUDA 主版本号相同（例如都是 12.x）
- **检查方法**: 查看 `system_info.txt` 中的 CUDA 版本

```bash
# 在目标机器上检查
nvcc --version
```

#### 3. GPU 架构兼容性

编译的 `.so` 文件包含特定 GPU 架构的代码：

- 查看编译时使用的架构：检查 `system_info.txt` 或 `dependencies_info.txt`
- 目标 GPU 需要支持这些架构
- 如果目标 GPU 不支持，需要重新编译

```bash
# 检查目标 GPU 的架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

#### 4. 系统库依赖

某些系统库需要在目标机器上可用：

- `libcublas.so`: CUDA 基础线性代数库
- `libcudart.so`: CUDA 运行时库
- 其他系统库（查看 `dependencies_info.txt`）

```bash
# 检查依赖
ldd compiled_extensions/path/to/extension.so
```

### 恢复过程

恢复脚本 (`restore_environment.sh`) 会执行以下步骤：

1. **检查系统兼容性**: 比较源机器和目标机器的信息
2. **检查 Python 环境**: 确认 Python 3.8+ 已安装
3. **检查 pipenv**: 如果未安装则自动安装
4. **恢复源代码**: 将源代码复制到目标位置
5. **恢复编译好的扩展**: 复制 `.so` 文件和构建信息
6. **安装依赖**: 使用 pipenv 或 pip 安装 Python 依赖

### 验证安装

恢复完成后，验证安装：

```bash
cd libortho_restored

# 方法 1: 使用 pipenv
pipenv shell
python -c "import libortho._C_ops; print('✅ 导入成功')"

# 方法 2: 直接使用 Python
python3 -c "import libortho._C_ops; print('✅ 导入成功')"
```

### 如果导入失败

如果导入失败，可能的原因和解决方法：

#### 1. Python 版本不兼容

```bash
# 检查 Python 版本
python3 --version

# 如果版本不匹配，需要重新编译
pipenv run rebuild
```

#### 2. CUDA 版本不兼容

```bash
# 检查 CUDA 版本
nvcc --version

# 如果版本不匹配，需要重新编译
pipenv run rebuild
```

#### 3. GPU 架构不兼容

```bash
# 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 如果架构不支持，需要重新编译
pipenv run rebuild
```

#### 4. 系统库缺失

```bash
# 检查缺失的库
ldd path/to/_C_ops.so

# 安装缺失的库（Ubuntu/Debian 示例）
sudo apt-get install libcublas-dev libcudart-dev
```

#### 5. 完全重新编译

如果以上方法都不行，完全重新编译：

```bash
cd libortho_restored

# 清理
pipenv run clean

# 重新编译
pipenv run rebuild
```

## 高级用法

### 只打包编译好的扩展

如果只需要移植编译好的扩展模块（不包含源代码）：

```bash
# 修改打包脚本，只打包 compiled_extensions 目录
tar -czf libortho_extensions.tar.gz compiled_extensions/ system_info.txt dependencies_info.txt
```

### 只打包源代码（不包含编译文件）

如果只需要移植源代码（在目标机器上重新编译）：

```bash
# 使用项目的清理脚本
bash clean_build.sh

# 然后打包源代码
tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    -czf libortho_source.tar.gz .
```

### 使用 Docker 容器

如果使用 Docker，可以打包整个容器：

```bash
# 在源机器上
docker commit container_name libortho:latest
docker save libortho:latest | gzip > libortho_docker.tar.gz

# 在目标机器上
docker load < libortho_docker.tar.gz
```

## 故障排除

### 问题 1: 打包脚本找不到文件

**症状**: 打包脚本报错找不到某些文件

**解决**: 
- 确保在项目根目录运行脚本
- 确保已经运行过 `pip install -e .` 或 `pipenv install`

### 问题 2: 恢复脚本失败

**症状**: 恢复脚本执行失败

**解决**:
- 检查目标机器的 Python 版本
- 检查目标机器的 CUDA 版本（如果使用 CUDA）
- 查看错误信息，可能需要手动安装某些依赖

### 问题 3: 导入失败但文件存在

**症状**: `.so` 文件存在但无法导入

**解决**:
- 检查文件权限: `chmod +x path/to/_C_ops.so`
- 检查依赖库: `ldd path/to/_C_ops.so`
- 重新编译: `pipenv run rebuild`

### 问题 4: 性能下降

**症状**: 移植后性能明显下降

**解决**:
- 检查是否使用了调试模式编译（`LIBORTHO_DEBUG=1`）
- 重新编译为发布模式: `pipenv run rebuild`（不设置 `LIBORTHO_DEBUG`）
- 检查 GPU 驱动版本

## 最佳实践

1. **记录环境信息**: 在打包前记录源机器的详细配置
2. **测试兼容性**: 在目标机器上先测试兼容性再正式部署
3. **版本控制**: 将打包文件版本化，便于回滚
4. **文档化**: 记录移植过程中的问题和解决方案
5. **自动化**: 使用 CI/CD 流程自动化打包和部署

## 相关文档

- [DEBUG_BUILD.md](DEBUG_BUILD.md): 编译调试指南
- [REBUILD_QUICK_REF.md](REBUILD_QUICK_REF.md): 重新编译快速参考
- [README.md](README.md): 项目主文档

