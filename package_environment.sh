#!/bin/bash
#
# libortho - 环境打包脚本
# 用于将编译好的环境打包，移植到另一台机器
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_NAME="libortho_env_$(date +%Y%m%d_%H%M%S)"
PACKAGE_DIR="${PACKAGE_NAME}"
PACKAGE_FILE="${PACKAGE_NAME}.tar.gz"

echo "============================================================"
echo "libortho - 环境打包工具"
echo "============================================================"
echo ""

# 创建打包目录
mkdir -p "${PACKAGE_DIR}"
cd "${PACKAGE_DIR}"

echo "[1/8] 收集系统信息..."
# 系统信息
{
    echo "=== System Information ==="
    uname -a
    echo ""
    echo "=== Python Version ==="
    python3 --version 2>&1 || echo "Python3 not found"
    which python3 || echo "Python3 path not found"
    echo ""
    echo "=== CUDA Version ==="
    nvcc --version 2>&1 || echo "CUDA not found"
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 || echo "nvidia-smi not available"
    echo ""
    echo "=== pipenv Version ==="
    pipenv --version 2>&1 || echo "pipenv not found"
} > system_info.txt
echo "✅ 系统信息已保存到 system_info.txt"

echo "[2/8] 收集 pipenv 环境信息..."
# Pipfile 和 Pipfile.lock
if [ -f "../Pipfile" ]; then
    cp ../Pipfile .
    echo "✅ Pipfile 已复制"
fi

if [ -f "../Pipfile.lock" ]; then
    cp ../Pipfile.lock .
    echo "✅ Pipfile.lock 已复制"
fi

# pipenv 环境路径
if command -v pipenv &> /dev/null; then
    PIPENV_VENV=$(pipenv --venv 2>/dev/null || echo "")
    if [ -n "$PIPENV_VENV" ]; then
        echo "✅ 找到 pipenv 虚拟环境: $PIPENV_VENV"
        echo "$PIPENV_VENV" > pipenv_venv_path.txt
    fi
fi

echo "[3/8] 收集编译好的扩展模块..."
# 查找所有 .so 文件
mkdir -p compiled_extensions
find .. -name "*.so" -type f | while read -r so_file; do
    rel_path=$(realpath --relative-to=.. "$so_file")
    dir_path=$(dirname "$rel_path")
    mkdir -p "compiled_extensions/$dir_path"
    cp "$so_file" "compiled_extensions/$rel_path"
    echo "  ✅ 复制: $rel_path"
done

# 查找 build 目录
if [ -d "../build" ]; then
    echo "✅ 复制 build 目录..."
    cp -r ../build compiled_extensions/build
fi

# 查找 .egg-info 目录
find .. -name "*.egg-info" -type d | while read -r egg_info; do
    rel_path=$(realpath --relative-to=.. "$egg_info")
    mkdir -p "compiled_extensions/$(dirname "$rel_path")"
    cp -r "$egg_info" "compiled_extensions/$rel_path"
    echo "  ✅ 复制: $rel_path"
done

echo "[4/8] 收集项目源代码..."
# 项目源代码（排除构建文件）
mkdir -p source_code
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='*.pyo' \
          --exclude='build' \
          --exclude='dist' \
          --exclude='*.egg-info' \
          --exclude='*.so' \
          --exclude='.pytest_cache' \
          --exclude='.mypy_cache' \
          --exclude='venv' \
          --exclude='.venv' \
          --exclude='Pipfile.lock' \
          ../ source_code/ || {
    # 如果 rsync 不可用，使用 tar
    echo "⚠️  rsync 不可用，使用 tar..."
    tar --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='build' \
        --exclude='dist' \
        --exclude='*.egg-info' \
        --exclude='*.so' \
        -czf source_code.tar.gz -C .. .
}

echo "[5/8] 导出 pip 包列表..."
# 导出已安装的包列表
if command -v pipenv &> /dev/null && [ -n "$PIPENV_VENV" ]; then
    pipenv run pip freeze > requirements_installed.txt 2>/dev/null || {
        echo "⚠️  无法使用 pipenv，尝试直接使用 pip..."
        pip freeze > requirements_installed.txt 2>/dev/null || echo "# pip freeze failed" > requirements_installed.txt
    }
else
    pip freeze > requirements_installed.txt 2>/dev/null || echo "# pip freeze failed" > requirements_installed.txt
fi
echo "✅ 已安装包列表已保存到 requirements_installed.txt"

echo "[6/8] 收集依赖信息..."
# 创建依赖信息文件
{
    echo "=== Python Packages ==="
    cat requirements_installed.txt
    echo ""
    echo "=== System Libraries (ldd on .so files) ==="
    find compiled_extensions -name "*.so" -type f | head -1 | while read -r so_file; do
        if command -v ldd &> /dev/null; then
            echo "Dependencies for: $so_file"
            ldd "$so_file" 2>&1 || echo "ldd failed"
        fi
    done
} > dependencies_info.txt
echo "✅ 依赖信息已保存到 dependencies_info.txt"

echo "[7/8] 创建恢复脚本..."
# 创建恢复脚本
cat > restore_environment.sh << 'RESTORE_SCRIPT'
#!/bin/bash
#
# libortho - 环境恢复脚本
# 在目标机器上运行此脚本来恢复环境
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "libortho - 环境恢复工具"
echo "============================================================"
echo ""

# 检查系统信息
echo "[1/6] 检查系统兼容性..."
if [ -f "system_info.txt" ]; then
    echo "源机器信息:"
    head -5 system_info.txt
    echo ""
    echo "当前机器信息:"
    uname -a
    python3 --version 2>&1 || echo "⚠️  Python3 not found"
    echo ""
    echo "⚠️  请确认 Python 版本和 CUDA 版本兼容性！"
    read -p "继续恢复？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 检查 Python 版本
echo "[2/6] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python 版本: $PYTHON_VERSION"

# 检查 pipenv
echo "[3/6] 检查 pipenv..."
if ! command -v pipenv &> /dev/null; then
    echo "⚠️  pipenv 未安装，正在安装..."
    pip install pipenv
fi
echo "✅ pipenv 已就绪"

# 恢复源代码
echo "[4/6] 恢复源代码..."
if [ -d "source_code" ]; then
    TARGET_DIR="$(pwd)/../libortho_restored"
    echo "将源代码恢复到: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
    cp -r source_code/* "$TARGET_DIR/"
    echo "✅ 源代码已恢复到 $TARGET_DIR"
    cd "$TARGET_DIR"
elif [ -f "source_code.tar.gz" ]; then
    TARGET_DIR="$(pwd)/../libortho_restored"
    echo "解压源代码到: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
    tar -xzf source_code.tar.gz -C "$TARGET_DIR"
    echo "✅ 源代码已解压到 $TARGET_DIR"
    cd "$TARGET_DIR"
else
    echo "❌ 未找到源代码"
    exit 1
fi

# 恢复编译好的扩展
echo "[5/6] 恢复编译好的扩展模块..."
PACKAGE_DIR_NAME=$(basename "$SCRIPT_DIR")
if [ -d "compiled_extensions" ]; then
    # 复制 .so 文件
    find compiled_extensions -name "*.so" -type f | while read -r so_file; do
        rel_path=$(echo "$so_file" | sed 's|compiled_extensions/||')
        target_path="../libortho_restored/$rel_path"
        mkdir -p "$(dirname "$target_path")"
        cp "$so_file" "$target_path"
        echo "  ✅ 恢复: $rel_path"
    done
    
    # 复制 build 目录（如果需要）
    if [ -d "compiled_extensions/build" ]; then
        cp -r compiled_extensions/build ../libortho_restored/build
        echo "  ✅ 恢复 build 目录"
    fi
    
    # 复制 .egg-info
    find compiled_extensions -name "*.egg-info" -type d | while read -r egg_info; do
        rel_path=$(echo "$egg_info" | sed 's|compiled_extensions/||')
        target_path="../libortho_restored/$rel_path"
        mkdir -p "$(dirname "$target_path")"
        cp -r "$egg_info" "$target_path"
        echo "  ✅ 恢复: $rel_path"
    done
fi

# 安装依赖
echo "[6/6] 安装 Python 依赖..."
cd "../libortho_restored"
if [ -f "Pipfile" ]; then
    echo "使用 pipenv 安装依赖..."
    pipenv install --python "$(which python3)" --skip-lock || {
        echo "⚠️  pipenv install 失败，尝试使用 pip..."
        if [ -f "../${PACKAGE_DIR_NAME}/requirements_installed.txt" ]; then
            pip install -r "../${PACKAGE_DIR_NAME}/requirements_installed.txt"
        fi
    }
else
    echo "使用 pip 安装依赖..."
    if [ -f "../${PACKAGE_DIR_NAME}/requirements_installed.txt" ]; then
        pip install -r "../${PACKAGE_DIR_NAME}/requirements_installed.txt"
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
fi

echo ""
echo "============================================================"
echo "✅ 环境恢复完成！"
echo "============================================================"
echo ""
echo "项目位置: $(pwd)"
echo ""
echo "下一步："
echo "1. 进入项目目录: cd $(pwd)"
echo "2. 验证安装: python3 -c 'import libortho._C_ops; print(\"✅ 导入成功\")'"
echo "3. 如果导入失败，可能需要重新编译: pipenv run rebuild"
echo ""
RESTORE_SCRIPT

chmod +x restore_environment.sh
echo "✅ 恢复脚本已创建: restore_environment.sh"

echo "[8/8] 创建 README..."
# 创建 README
cat > README.md << 'README_EOF'
# libortho 环境打包

## 打包内容

- `system_info.txt`: 源机器的系统信息（Python版本、CUDA版本等）
- `Pipfile` / `Pipfile.lock`: pipenv 环境配置
- `compiled_extensions/`: 编译好的扩展模块（.so 文件）
- `source_code/`: 项目源代码
- `requirements_installed.txt`: 已安装的 Python 包列表
- `dependencies_info.txt`: 依赖信息
- `restore_environment.sh`: 环境恢复脚本

## 使用方法

### 在目标机器上恢复环境

1. 解压打包文件：
   ```bash
   tar -xzf libortho_env_*.tar.gz
   cd libortho_env_*
   ```

2. 运行恢复脚本：
   ```bash
   bash restore_environment.sh
   ```

3. 验证安装：
   ```bash
   cd ../libortho_restored
   python3 -c "import libortho._C_ops; print('✅ 导入成功')"
   ```

## 注意事项

⚠️ **重要兼容性检查**：

1. **Python 版本**: 目标机器的 Python 版本应该与源机器相同或兼容
2. **CUDA 版本**: 如果使用 CUDA 扩展，目标机器需要相同或兼容的 CUDA 版本
3. **GPU 架构**: 编译的 .so 文件包含特定的 GPU 架构代码，目标 GPU 需要支持
4. **系统库**: 某些系统库（如 libcublas）需要在目标机器上可用

如果遇到兼容性问题，建议在目标机器上重新编译：
```bash
cd libortho_restored
pipenv install --python $(which python3)
pipenv run rebuild
```

## 文件说明

- `system_info.txt`: 查看源机器的配置信息
- `dependencies_info.txt`: 查看依赖的系统库
- `requirements_installed.txt`: 查看已安装的 Python 包
README_EOF
echo "✅ README 已创建"

cd ..

echo ""
echo "============================================================"
echo "✅ 打包完成！"
echo "============================================================"
echo ""
echo "打包目录: ${PACKAGE_DIR}"
echo ""

# 创建压缩包
echo "正在创建压缩包..."
tar -czf "${PACKAGE_FILE}" "${PACKAGE_DIR}"
echo "✅ 压缩包已创建: ${PACKAGE_FILE}"
echo ""

# 显示打包信息
echo "打包内容："
du -sh "${PACKAGE_DIR}"
du -sh "${PACKAGE_FILE}"
echo ""

echo "============================================================"
echo "📦 打包文件: ${PACKAGE_FILE}"
echo "============================================================"
echo ""
echo "移植到目标机器后："
echo "1. 解压: tar -xzf ${PACKAGE_FILE}"
echo "2. 进入目录: cd ${PACKAGE_DIR}"
echo "3. 运行恢复脚本: bash restore_environment.sh"
echo ""

