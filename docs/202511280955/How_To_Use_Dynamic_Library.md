如何使用编译好的动态库 (.so)
=================

**状态：** 正常。在 Linux/WSL 上，C++ 扩展就是 `.so` 文件。

1. 确认安装位置

---------

当你运行 `pip install -e .` 时，它会在当前目录下编译并生成一个指向构建目录的链接，或者直接把 `.so` 放在源代码树里。

运行以下命令查看它在哪里：
    find . -name "*.so"

你应该能看到类似 `libortho/_C_ops.cpython-311-x86_64-linux-gnu.so` 的文件。

2. 验证导入

-------

创建一个简单的测试脚本 `test_import.py`：
    try:
        import torch
        import libortho._C_ops as _C
        print(f"✅ 成功导入 C++ 扩展: {_C}")
        print(f"   可用函数: {dir(_C)}")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请确保你已经运行了 'pip install -e .'")

运行它：`python3 test_import.py`

3. 在代码中使用

---------

在 `torch_bind/ortho_linear.py` 中，导入逻辑应该是这样的：
    # torch_bind/ortho_linear.py

    import torch
    import torch.nn as nn

    # 尝试导入 C++ 扩展
    try:
        # 注意：这里的包名必须与 setup.py 中定义的一致
        import libortho._C_ops as _C
        HAS_C_OPS = True
    except ImportError:
        # 如果你在开发环境中直接运行，可能需要调整路径
        try:
            import _C_ops as _C
            HAS_C_OPS = True
        except ImportError:
            print("⚠️  警告: 未找到 libortho C++ 扩展。将使用慢速 Python 回退模式。")
            HAS_C_OPS = False

    class OrthoLinear(nn.Module):
        # ... (代码保持不变)

        def forward(self, x):
            # ...
            if HAS_C_OPS and x.is_cuda:
                # 调用 C++ 函数
                _C.forward(
                    self.base_weight,
                    self.base_scales,
                    self.ortho_values,
                    self.ortho_col_indices,
                    self.ortho_row_ptr,
                    x_flat.contiguous(),
                    output,
                    self.alpha
                )
            else:
                # Python 回退逻辑
                pass

4. 常见问题 (Troubleshooting)

-------------------------

**Q: `ImportError: No module named 'libortho._C_ops'`**

* **原因**: 编译失败，或者 Python 找不到 `.so` 文件。

* **解决**:
  
  1. 确保 `pip install -e .` 成功执行，没有报错。
  
  2. 检查 `libortho/` 目录下是否有 `.so` 文件。
  
  3. 确保你在项目根目录下运行 Python。

**Q: `ImportError: ... undefined symbol ...`**

* **原因**: 链接错误。可能是没链接 `cudart` 或 `cublas`。

* **解决**: 检查 `setup.py` 中的 `libraries=['cublas']` 是否存在。

**Q: `RuntimeError: LibOrtho compiled without CUDA support`**

* **原因**: 你在没有 CUDA 的机器上编译了它，或者 `setup.py` 没检测到 `nvcc`。

* **解决**: 确保 `nvcc --version` 能输出版本号，然后重新编译。

Linus 的建议：

别在那猜了。写个脚本 check_ops.py 打印一下 libortho._C_ops 是否存在。如果存在，你的“法拉利引擎”就就绪了。
