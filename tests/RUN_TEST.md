# 运行 CPU 测试

## 在 WSL 中运行

### 方法1：C 测试（推荐，测试实际实现）

```bash
cd tests
make clean
make
./test_cpu_forward
```

### 方法2：Python 测试（验证算法逻辑）

```bash
cd tests
python3 test_cpu_forward.py
```

## 预期输出

```
=== Testing orth_layer_forward() ===

Test configuration:
  Dimensions: 64 x 64
  Batch size: 4
  Quantization: INT4
  Ortho sparsity: 204 elements (5.0%)

Test 1: Alpha = 1.0 (Full Model)
  Max absolute error: 0.000001
  Max relative error: 0.000001
  Elements with error > 1e-5: 0 / 256
  ✅ PASSED

Test 2: Alpha = 0.0 (Base Only)
  Max absolute error: 0.000001
  Max relative error: 0.000001
  Elements with error > 1e-5: 0 / 256
  ✅ PASSED

Test 3: Empty Ortho (Base Only)
  Max absolute error: 0.000001
  Max relative error: 0.000001
  Elements with error > 1e-5: 0 / 256
  ✅ PASSED

=== All tests passed! ===
```

## 故障排除

### 编译错误：posix_memalign 未声明

如果看到 `implicit declaration of function 'posix_memalign'` 警告：
- 这通常不是致命错误，代码仍能编译
- 如果确实有问题，确保定义了 `_POSIX_C_SOURCE`

### 链接错误

确保安装了必要的库：
```bash
sudo apt-get install build-essential
```

### 运行时错误

检查内存对齐是否正确：
```bash
# 验证对齐
gcc -E -dM src/ortho.c | grep ALIGNMENT
```

## 测试覆盖

测试覆盖以下场景：
1. ✅ 完整模型（Base + Ortho，alpha=1.0）
2. ✅ 仅 Base（alpha=0.0，隐私模式）
3. ✅ 空 Ortho（边界条件）

## 下一步

测试通过后，可以：
1. 进行性能基准测试
2. 测试更大的矩阵尺寸
3. 验证内存对齐是否正确

