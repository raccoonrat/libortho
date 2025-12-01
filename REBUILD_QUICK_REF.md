# 重新编译快速参考

## Pipenv 环境重新编译（推荐）

### 最简单的方式

```bash
# 清理并重新编译（正常模式）
pipenv run rebuild

# 清理并重新编译（调试模式）
pipenv run rebuild-debug

# 清理并重新编译（详细调试模式）
pipenv run rebuild-debug-verbose
```

### 分步执行

```bash
# 1. 清理构建文件
pipenv run clean

# 2. 重新安装
pipenv run install          # 正常模式
pipenv run install-debug    # 调试模式
pipenv run install-debug-verbose  # 详细调试模式
```

## 直接使用 pip（不在 pipenv 环境中）

```bash
# 清理
rm -rf build/ *.egg-info/
find . -name "*.so" -delete
find . -name "*.o" -delete

# 重新安装
pip install -e . --force-reinstall --no-cache-dir

# 调试模式重新安装
export LIBORTHO_DEBUG=1
pip install -e . --force-reinstall --no-cache-dir
```

## 使用项目清理脚本

```bash
# 使用项目提供的清理脚本
bash clean_build.sh

# 然后重新安装
pipenv run install
# 或
pip install -e .
```

## 常见场景

### 修改源代码后
```bash
pipenv run rebuild
```

### 需要调试时
```bash
pipenv run rebuild-debug-verbose 2>&1 | tee rebuild.log
```

### 编译出错时
```bash
pipenv run clean
pipenv run rebuild-debug-verbose 2>&1 | tee rebuild_error.log
cat rebuild_error.log | grep -i error
```

### 完全清理并重新开始
```bash
pipenv run clean
pipenv run pip cache purge
pipenv install --dev
pipenv run rebuild
```

## 所有可用命令

### 编译相关

| 命令 | 说明 |
|------|------|
| `pipenv run clean` | 清理所有构建文件 |
| `pipenv run install` | 正常安装 |
| `pipenv run install-debug` | 调试模式安装 |
| `pipenv run install-debug-verbose` | 详细调试模式安装 |
| `pipenv run rebuild` | 清理 + 重新安装（正常模式） |
| `pipenv run rebuild-debug` | 清理 + 重新安装（调试模式） |
| `pipenv run rebuild-debug-verbose` | 清理 + 重新安装（详细调试模式） |

### 打包相关

| 命令 | 说明 |
|------|------|
| `pipenv run package` | 打包环境（用于移植到其他机器） |

详细说明请参考：
- [DEBUG_BUILD.md](DEBUG_BUILD.md) - 编译调试指南
- [PACKAGE_QUICK_REF.md](PACKAGE_QUICK_REF.md) - 打包快速参考
- [PORTING_GUIDE.md](PORTING_GUIDE.md) - 完整移植指南

