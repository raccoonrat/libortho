# libortho 完整测试计划

**测试日期**: 2024-12-24  
**测试目标**: 验证所有修复后的功能

---

## 测试环境要求

- Python 3.x
- GCC (C11 支持)
- CUDA Toolkit (可选，用于 GPU 测试)
- WSL 或 Linux 环境

---

## 测试步骤

### 1. 快速测试（Python 验证）

```bash
# 在项目根目录
python3 experiments/verify_core_logic.py
```

**预期结果**: 
- ✅ SUCCESS: Privacy forgotten
- ✅ SUCCESS: General logic preserved

---

### 2. 完整测试套件

```bash
# 在项目根目录
cd tests
chmod +x run_complete_tests.sh
./run_complete_tests.sh
```

**测试内容**:
1. Python 核心逻辑验证
2. CPU Forward 测试（对齐分配）
3. 性能基准测试（Null Test）
4. GPU 环境检查
5. CUDA Kernel 测试（如果可用）
6. Tensor Core 测试（如果可用）
7. Sieve 工具测试（索引排序）

---

### 3. 单独测试

#### 3.1 CPU Forward 测试

```bash
cd tests
make clean
make test
./test_cpu_forward
```

**验证点**:
- ✅ 对齐内存分配成功
- ✅ Forward 计算正确
- ✅ Alpha 开关功能正常

#### 3.2 性能基准测试（Null Test）

```bash
cd tests
make benchmark
./benchmark_null_test
```

**验证点**:
- ✅ 开销 < 1%
- ✅ Base-only 性能等同于标准 INT4

#### 3.3 Sieve 工具测试

```bash
python3 << 'EOF'
import torch
from tools.sieve import hessian_sieve, pack_ortho_sparse, compute_hessian_diag_approx

# 测试数据
weight = torch.randn(64, 64)
inputs = torch.randn(100, 64)
H_diag = compute_hessian_diag_approx(inputs)

# 运行 sieve
w_base, w_ortho = hessian_sieve(weight, H_diag, sparsity_target=0.95)

# 测试索引排序
indices, values = pack_ortho_sparse(w_ortho, format="coo")

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
EOF
```

---

## 预期结果

### 成功标准

1. **Python 核心逻辑验证**: ✅ 两个 SUCCESS
2. **CPU Forward 测试**: ✅ 所有测试用例通过
3. **Null Test**: ✅ 开销 < 1%
4. **Sieve 工具**: ✅ 索引正确排序
5. **编译**: ✅ 无错误无警告

### 失败处理

如果测试失败：

1. 检查编译错误信息
2. 查看测试输出详情
3. 验证修复是否正确应用
4. 检查环境依赖

---

## 测试报告

测试完成后，结果将显示：
- 通过的测试数量
- 失败的测试数量
- 跳过的测试数量（如 GPU 不可用）

---

## 快速验证清单

- [ ] Python 核心逻辑验证通过
- [ ] CPU Forward 测试通过
- [ ] Null Test 开销 < 1%
- [ ] Sieve 索引排序正确
- [ ] 编译无错误无警告
- [ ] 所有修复功能正常

---

**注意**: 如果某些测试被跳过（如 GPU 测试），这是正常的，不影响核心功能验证。

