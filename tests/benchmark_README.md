# 性能基准测试

## Null Test 基准测试

验证 Linus 的要求：**如果 W_ortho 全为零，系统性能必须完全等同于标准 INT4 模型。**

### 运行测试

```bash
cd tests
make benchmark
./benchmark_null_test
```

### 成功标准

- **性能开销 < 1%**: libortho 在 Ortho=0 时的性能开销必须 < 1%
- **输出正确性**: 输出必须与参考实现完全匹配（误差 < 1e-4）

### 预期输出

```
============================================================
Null Test Performance Benchmark
============================================================

Linus's Requirement:
  If W_ortho is all zero, performance must be COMPLETELY EQUAL
  to a standard INT4 model. If Base stream slows down by 1%, it's a failure.

Test Configuration:
  Dimensions: 1024 x 1024
  Batch size: 32
  Iterations: 100

--- Benchmark 1: Reference INT4 Implementation ---
  Average time: X.XXX ms
  Throughput: XXXX.XX samples/sec

--- Benchmark 2: libortho with Empty Ortho (Null Test) ---
  Average time: X.XXX ms
  Throughput: XXXX.XX samples/sec
  Max output difference: 0.000000
  ✅ Output matches reference

--- Performance Comparison ---
  Reference time: X.XXX ms
  libortho time:  X.XXX ms
  Overhead: X.XX%

✅ SUCCESS: Null Test PASSED!
   Overhead (X.XX%) is less than 1% threshold.
   libortho with empty Ortho performs equivalently to standard INT4.
```

### 故障排除

如果测试失败（开销 > 1%）：
1. 检查是否有不必要的分支
2. 验证内存对齐是否正确
3. 检查循环优化是否生效

