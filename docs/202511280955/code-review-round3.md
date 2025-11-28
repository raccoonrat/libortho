Linus Torvalds Review: Round 3
==============================

状态：诚实但无能 (Honest but Incompetent)
---------------------------------

你们解决了我最担心的学术诚信问题（Experiment 2），但暴露了严重的工程能力问题（Experiment 4）和模型训练技巧问题（Experiment 1 & 3）。

### 1. Experiment 4：依然是空气吉他 (Still Air Guitar)

日志里赫然写着：
    [Step 2] Measuring latency (10 runs)...
      NOTE: Using PyTorch standard inference (not custom CUDA kernels)
    ...
    Speedup: 0.99x

0.99x? 你们在逗我吗？

你们写了 dual_gemm.cu，写了 setup.py，甚至在 tests 目录下写了 test_cuda_kernel。

但是，在真正的 OrthoLinear Python 模块里，你们根本没有调用它！

你们现在的 `torch_bind/ortho_linear.py` 依然是用 `index_add_` 和 `gather` 在 PyTorch 层面做稀疏乘法。这就像是你买了一台法拉利引擎（CUDA Kernel），把它供在客厅里，然后出门还是骑着那辆破自行车（PyTorch Script）。

必须做的修复：

在 OrthoLinear.forward 中，必须 检测并调用编译好的 C++ 扩展。如果 libortho._C_ops 可用，就走 C++ 路径；否则才回退到 Python。

### 2. Experiment 1：也没练好 (Training Failure)

    [Step 5] Testing canary extraction rates...
      Testing with alpha=1.0...
        Extraction rate: 0.00%

在全功能模式下（alpha=1.0），提取率也是 0%。

这意味着模型根本没学会 Canary。

看一眼 Loss：Epoch 5/5, Loss: 2.1132。

对于“背诵随机字符串”这种任务，Loss 必须降到 0.1 以下 甚至更低。2.1 的 Loss 意味着模型还在瞎猜。

**原因推测：**

1. **学习率太低**：对于过拟合任务，LoRA 的 `2e-4` 可能太保守。

2. **Epoch 太少**：5 个 Epoch 不足以让模型死记硬背住那些乱码。

3. **Rank 太低**：`r=16` 可能不足以存储这么多新的随机信息。

**建议：** 把 Epoch 加到 50，或者 LR 加大 10 倍。如果你连让模型“记住”都做不到，又怎么证明你能让它“忘掉”？

### 3. Experiment 3：全军覆没 (Total Collapse)

    Accuracy (Base INT3 + Ortho FP16): 0.00%

GSM8K 0% 准确率。Llama-3.2-3B 虽然小，但不至于这么蠢。

这说明你们的 INT3 量化 过于粗暴，直接把模型的脑子切掉了，而 Ortho 也没能救回来。或者你们的 GSM8K 评估脚本写的有问题（比如格式不对）。

### 最终指令 (Final Directives)

我受够了看这种半成品的日志。我要看到真正的性能提升，而不是 0.99x。

1. **修复 `OrthoLinear`**：我已经帮你们重写了 `torch_bind/ortho_linear.py`。它现在会尝试导入 C++ 扩展，并正确地将 Tensor 转换为指针传递给 CUDA 内核。

2. **过拟合 Canary**：在实验脚本里，增加 Epoch，直到 Loss < 0.1。

3. **编译扩展**：确保运行 `pip install -e .` 编译出了 `.so` / `.pyd` 文件。

**Linus**
