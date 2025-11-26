# libortho 完整测试指南

**测试日期**: 2024-12-24  
**测试目标**: 验证所有修复后的功能

---

## 🚀 快速开始

### 在 WSL 中运行（推荐）

```bash
# 1. 进入项目目录
cd libortho

# 2. 运行完整测试套件
cd tests
chmod +x run_complete_tests.sh
./run_complete_tests.sh
```

---

## 📋 测试清单

### ✅ 测试 1: Python 核心逻辑验证

**目的**: 验证双流形假设和隐私开关功能

```bash
python3 experiments/verify_core_logic.py
```

**预期输出**:
```
--- [LibOrtho] Initializing Minimal Verification ---
Training Loss: 0.xxxxxx
Original Model -> Privacy Error: 0.xxxx (Should be low)
Original Model -> General Error: 0.xxxx (Should be low)
Sieve Complete. Ortho Sparsity: 95.xx%

--- Testing The Kill Switch ---
[Alpha=1.0] Privacy Error: 0.xxxx (Target: Low)
[Alpha=1.0] General Error: 0.xxxx (Target: Low)
[Alpha=0.0] Privacy Error: X.xxxx (Target: HIGH -> Forgot Privacy!)
[Alpha=0.0] General Error: 0.xxxx (Target: LOW -> Kept Logic!)
✅ SUCCESS: Privacy forgotten (ratio=X.xx).
✅ SUCCESS: General logic preserved (Robust Base).
```

**成功标准**: 两个 ✅ SUCCESS

---

### ✅ 测试 2: CPU Forward 测试

**目的**: 验证对齐内存分配和 forward 计算

```bash
cd tests
make clean
make test
./test_cpu_forward
```

**验证点**:
- ✅ 对齐内存分配成功（128-byte）
- ✅ Forward 计算正确
- ✅ Alpha 开关功能正常
- ✅ 与参考实现一致

---

### ✅ 测试 3: 性能基准测试（Null Test）

**目的**: 验证 Linus 的要求（开销 < 1%）

```bash
cd tests
make benchmark
./benchmark_null_test
```

**预期输出**:
```
Null Test Performance Benchmark
  If W_ortho is all zero, performance must be COMPLETELY EQUAL

--- Benchmark 1: Reference INT4 (Baseline) ---
Average time: X.XXX ms

--- Benchmark 2: libortho with Empty Ortho (Null Test) ---
Average time: X.XXX ms

Performance ratio: X.XX
✅ SUCCESS: Null Test PASSED!
```

**成功标准**: 
- 性能比例 < 1.01 (开销 < 1%)
- ✅ SUCCESS: Null Test PASSED!

---

### ✅ 测试 4: Sieve 工具测试（索引排序）

**目的**: 验证索引排序功能

```bash
python3 << 'EOF'
import torch
import sys
sys.path.insert(0, '.')
from tools.sieve import hessian_sieve, pack_ortho_sparse, compute_hessian_diag_approx

# 测试数据
weight = torch.randn(64, 64)
inputs = torch.randn(100, 64)
H_diag = compute_hessian_diag_approx(inputs)

# 运行 sieve
w_base, w_ortho = hessian_sieve(weight, H_diag, sparsity_target=0.95)
print(f"Ortho sparsity: {(w_ortho == 0).float().mean():.2%}")

# 测试索引排序
indices, values = pack_ortho_sparse(w_ortho, format="coo")
print(f"Non-zero count: {len(indices)}")

# 验证排序
in_features = 64
rows = indices // in_features
cols = indices % in_features

is_sorted = True
for i in range(len(rows) - 1):
    if rows[i] > rows[i+1] or (rows[i] == rows[i+1] and cols[i] > cols[i+1]):
        is_sorted = False
        break

print("✅ Index sorting: PASSED" if is_sorted else "❌ Index sorting: FAILED")
sys.exit(0 if is_sorted else 1)
EOF
```

**成功标准**: ✅ Index sorting: PASSED

---

### ⚠️ 测试 5: GPU 测试（可选）

**如果 CUDA 可用**:

```bash
cd tests
make cuda-test
./test_cuda_kernel

make tensor-core-test
./test_tensor_core
```

**注意**: 如果 GPU 不可用，这些测试会被跳过，不影响核心功能验证。

---

## 📊 测试结果解读

### 成功标准

| 测试 | 成功标准 |
|------|----------|
| Python 核心逻辑 | 两个 ✅ SUCCESS |
| CPU Forward | 所有测试用例通过 |
| Null Test | 开销 < 1% |
| Sieve 排序 | ✅ PASSED |
| 编译 | 无错误无警告 |

### 常见问题

1. **编译错误**: 检查 GCC 版本和依赖
2. **Python 错误**: 检查 PyTorch 安装
3. **GPU 测试跳过**: 正常，如果 GPU 不可用

---

## 🔍 详细测试输出

### 完整测试套件输出示例

```
============================================================
libortho - Complete Test Suite
============================================================
Date: 2024-12-24

------------------------------------------------------------
Test: Core Logic Verification
------------------------------------------------------------
✅ SUCCESS: Privacy forgotten (ratio=X.xx).
✅ SUCCESS: General logic preserved (Robust Base).
✅ PASSED: Core Logic Verification

------------------------------------------------------------
Test: CPU Forward
------------------------------------------------------------
Test 1: Alpha = 1.0 (Full Model) - PASSED
Test 2: Alpha = 0.0 (Base Only) - PASSED
✅ PASSED: CPU Forward

------------------------------------------------------------
Test: Null Test Benchmark
------------------------------------------------------------
✅ SUCCESS: Null Test PASSED!
✅ PASSED: Null Test Benchmark

============================================================
Test Summary
============================================================
Tests Passed: 7
Tests Failed: 0
Tests Skipped: 2 (GPU tests)

✅ All tests passed!
```

---

## 🛠️ 故障排除

### 问题 1: 编译失败

```bash
# 检查 GCC
gcc --version

# 清理并重新编译
cd tests
make clean
make test
```

### 问题 2: Python 导入错误

```bash
# 检查 PyTorch
python3 -c "import torch; print(torch.__version__)"

# 安装依赖
pip3 install torch numpy
```

### 问题 3: 对齐分配失败

检查 `src/ortho.c` 中的对齐分配函数是否正确实现。

---

## 📝 测试报告模板

测试完成后，记录结果：

```
测试日期: 2024-12-24
测试环境: WSL Ubuntu 24.04

测试结果:
- [x] Python 核心逻辑验证: PASSED
- [x] CPU Forward 测试: PASSED
- [x] Null Test: PASSED (开销: X.XX%)
- [x] Sieve 索引排序: PASSED
- [ ] GPU 测试: SKIPPED (GPU 不可用)

总体状态: ✅ 所有核心测试通过
```

---

**下一步**: 运行测试并记录结果！

