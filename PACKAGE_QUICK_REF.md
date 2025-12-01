# 环境打包快速参考

## 快速打包

### 使用 Pipfile 脚本（推荐）

```bash
# 打包当前环境
pipenv run package
```

### 直接运行脚本

```bash
# 确保脚本有执行权限
chmod +x package_environment.sh

# 运行打包脚本
./package_environment.sh
```

## 打包输出

打包完成后会生成：
- `libortho_env_YYYYMMDD_HHMMSS.tar.gz` - 压缩包文件
- `libortho_env_YYYYMMDD_HHMMSS/` - 打包目录（可删除）

## 移植到目标机器

### 1. 传输文件

```bash
# 使用 scp
scp libortho_env_*.tar.gz user@target-machine:/path/to/destination/

# 或使用其他方式（rsync, USB 等）
```

### 2. 在目标机器上恢复

```bash
# 解压
tar -xzf libortho_env_*.tar.gz
cd libortho_env_*

# 运行恢复脚本
bash restore_environment.sh
```

### 3. 验证安装

```bash
cd ../libortho_restored
python3 -c "import libortho._C_ops; print('✅ 导入成功')"
```

## 打包内容

打包脚本会收集：

- ✅ 系统信息（Python版本、CUDA版本等）
- ✅ Pipfile 和 Pipfile.lock
- ✅ 编译好的扩展模块（.so 文件）
- ✅ 项目源代码
- ✅ 已安装的 Python 包列表
- ✅ 依赖信息
- ✅ 自动恢复脚本

## 兼容性检查

⚠️ **重要**: 移植前请检查：

1. **Python 版本**: 应该相同或兼容
2. **CUDA 版本**: 如果使用 CUDA，应该相同或更高
3. **GPU 架构**: 目标 GPU 需要支持编译时的架构
4. **系统库**: 某些系统库需要在目标机器上可用

如果遇到兼容性问题，在目标机器上重新编译：
```bash
cd libortho_restored
pipenv run rebuild
```

## 常见问题

### 打包脚本找不到文件

确保已经编译过：
```bash
pipenv run install
# 或
pip install -e .
```

### 恢复后导入失败

检查兼容性，可能需要重新编译：
```bash
cd libortho_restored
pipenv run rebuild
```

### 文件太大

如果只需要源代码（在目标机器上重新编译）：
```bash
bash clean_build.sh
tar --exclude='.git' --exclude='__pycache__' -czf libortho_source.tar.gz .
```

## 详细文档

- [PORTING_GUIDE.md](PORTING_GUIDE.md): 完整的移植指南
- [DEBUG_BUILD.md](DEBUG_BUILD.md): 编译调试指南

