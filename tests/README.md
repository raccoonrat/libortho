# libortho 测试

## CPU 前向传播测试

### 编译和运行

```bash
cd tests
make
./test_cpu_forward
```

### 测试内容

1. **Test 1: Alpha = 1.0 (Full Model)**
   - 测试完整模型（Base + Ortho）
   - 验证 INT4 反量化和稀疏矩阵乘法的正确性

2. **Test 2: Alpha = 0.0 (Base Only)**
   - 测试仅 Base 组件（隐私模式）
   - 验证 alpha 参数的正确控制

3. **Test 3: Empty Ortho (Base Only)**
   - 测试空 Ortho 的情况
   - 验证边界条件处理

### 成功标准

- 最大绝对误差 < 1e-4
- 最大相对误差 < 1e-3
- 所有测试用例通过

### 故障排除

如果测试失败：
1. 检查编译错误
2. 验证内存对齐是否正确
3. 检查 INT4 解包逻辑

