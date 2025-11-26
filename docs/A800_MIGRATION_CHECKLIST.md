# A800 迁移检查清单

## 前置检查

### 1. 系统环境

- [ ] 确认A800 GPU已安装并识别
  ```bash
  nvidia-smi
  # 应该显示 A800
  ```

- [ ] 检查CUDA版本
  ```bash
  nvcc --version
  # 需要 CUDA 11.0+ (推荐 11.8+)
  ```

- [ ] 检查驱动版本
  ```bash
  nvidia-smi
  # 推荐驱动版本 470+
  ```

### 2. 代码准备

- [ ] 创建A800分支
  ```bash
  ./create_a800_branch.sh
  # 或手动: git checkout -b feature/a800-ampere-support
  ```

- [ ] 确认代码已包含最新修复
  - [x] Python内存优化（稀疏张量）
  - [x] C代码CSR支持
  - [x] 懒加载实现

## 编译和安装

### 3. 编译CUDA扩展

- [ ] 清理旧编译
  ```bash
  cd tests
  make clean
  ```

- [ ] 安装libortho
  ```bash
  pip install -e . --force-reinstall
  ```

- [ ] 验证编译
  ```bash
  cd tests
  make cuda-test
  ./test_cuda_kernel
  ```

### 4. 验证GPU识别

- [ ] 运行GPU检查脚本
  ```bash
  cd tests
  python3 check_gpu.py
  ```

预期输出：
```
GPU 0: NVIDIA A800
  Compute Capability: 8.0
  Total Memory: 80.00 GB
  ✅ Tensor Cores: Supported (sm_80)
```

## 功能测试

### 5. 基础功能测试

- [ ] 运行基础测试
  ```bash
  cd tests
  ./run_all_tests.sh
  ```

- [ ] 测试稀疏张量功能
  ```bash
  python3 test_fixes.py
  ```

### 6. 大模型测试

- [ ] 准备模型路径
  ```bash
  # 确认模型路径
  MODEL_PATH="/path/to/Llama-3.2-3B"
  ```

- [ ] 测试懒加载
  ```python
  # 在Python中测试
  from experiments.complete_real_model_experiments import RealModelLibOrtho
  
  libortho = RealModelLibOrtho(model, tokenizer, device="cuda")
  libortho.separate_weights(
      sample_inputs,
      lazy_loading=True  # 测试懒加载
  )
  ```

- [ ] 运行实验1（隐私开关）
  ```bash
  python3 experiments/complete_real_model_experiments.py \
      --model $MODEL_PATH \
      --experiment 1 \
      --device cuda
  ```

- [ ] 检查内存使用
  ```bash
  # 在另一个终端监控
  watch -n 1 nvidia-smi
  ```

## 性能验证

### 7. 性能基准测试

- [ ] 运行性能实验
  ```bash
  python3 experiments/complete_real_model_experiments.py \
      --model $MODEL_PATH \
      --experiment 4 \
      --device cuda
  ```

- [ ] 验证性能指标
  - [ ] FP16吞吐量: 预期 200-300 tokens/sec
  - [ ] INT4吞吐量: 预期 400-600 tokens/sec
  - [ ] 延迟: 预期 <10ms/token

### 8. 内存优化验证

- [ ] 验证稀疏张量存储
  ```python
  # 检查ortho_weights是否为稀疏张量
  import torch
  w_ortho = libortho.ortho_weights_sparse['layer_name']
  assert isinstance(w_ortho, torch.sparse.SparseTensor)
  ```

- [ ] 验证内存占用
  ```bash
  # 运行实验时监控内存
  # 应该从24GB降到~12GB（3B模型）
  ```

## 故障排除

### 9. 常见问题检查

- [ ] 编译错误
  - [ ] CUDA版本 >= 11.0
  - [ ] 清理并重新编译

- [ ] 运行时错误
  - [ ] 驱动版本 >= 470
  - [ ] CUDA驱动匹配

- [ ] 内存问题
  - [ ] 使用懒加载
  - [ ] 检查其他进程占用显存

- [ ] 性能问题
  - [ ] 确认使用CSR格式（不是COO）
  - [ ] 检查Tensor Core是否启用

## 完成检查

### 10. 最终验证

- [ ] 所有测试通过
- [ ] 性能指标符合预期
- [ ] 内存使用正常
- [ ] 文档已更新

### 11. 提交代码

- [ ] 提交A800支持
  ```bash
  git add .
  git commit -m "Add A800 (Ampere) support with optimizations"
  ```

- [ ] 推送到远程（如需要）
  ```bash
  git push origin feature/a800-ampere-support
  ```

## 参考文档

- [A800支持文档](A800_AMPERE_SUPPORT.md)
- [快速开始指南](../QUICKSTART_A800.md)
- [内存修复总结](1126-memory-fix-summary.md)

---

**最后更新**: 2025-01-26

