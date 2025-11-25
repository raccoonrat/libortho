# libortho 实验验证

本目录包含三个核心实验，用于验证"对偶几何"理论的完整性。

## 实验概览

### 实验1：隐私开关测试 (verify_core_logic.py)

**目标**: 验证隐私数据可以通过关闭Ortho组件而被物理截断，同时保留通用能力。

**假设**:
- W_ortho包含隐私数据（法向分量）
- W_base包含通用知识（切向分量）
- 关闭Ortho后，隐私消失，通用能力保留

**运行**:
```bash
python experiments/verify_core_logic.py
```

**预期结果**:
- ✅ 关闭Ortho后，隐私误差显著增加（隐私被遗忘）
- ✅ 关闭Ortho后，通用误差保持稳定（通用能力保留）

---

### 实验2：天才的保留 (saving_genius.py)

**目标**: 验证"天才推理"（高曲率的正确解）栖息在Ortho组件中，即使Base被极度量化（模拟RL挤压）也能保留。

**假设**:
- W_ortho包含"天才"（正确但非显而易见的解）
- W_base包含"常识"（标准模式）
- 即使Base被"脑叶切除"（极度量化），Ortho仍保留天才

**运行**:
```bash
python experiments/saving_genius.py
```

**实验设计**:
1. 训练模型学习"常识"模式（线性变换）
2. 微调模型学习"天才"模式（非线性、需要洞察的模式）
3. 使用Hessian筛分离Base和Ortho
4. 对Base进行激进量化（INT2、甚至二进制）
5. 验证Ortho中的"天才"推理是否保留

**预期结果**:
- ✅ 即使Base被极度量化，Genius误差不应显著增加
- ✅ Genius的保留率应优于Common Sense的保留率
- ✅ 这证明了"正确性"确实栖息在法向分量中

---

### 实验3：对偶差分隐私 (dual_dp.py)

**目标**: 验证仅对Ortho施加差分隐私（DP）噪音，比全局DP保留更好的通用Utility，同时维持相同的隐私预算。

**假设**:
- W_base包含公共知识（无隐私风险）
- W_ortho包含隐私（需要DP保护）
- 对偶DP（仅对Ortho加噪音）> 全局DP（对所有权重加噪音）的Utility

**运行**:
```bash
python experiments/dual_dp.py
```

**实验设计**:
1. 训练模型学习公共知识（WikiText-like）
2. 微调模型记忆私有数据（Canary IDs）
3. 使用Hessian筛分离Base和Ortho
4. 应用两种DP方法：
   - **全局DP**: 对所有权重（Base + Ortho）加Gaussian噪音
   - **对偶DP**: 仅对Ortho加Gaussian噪音，Base不加噪音
5. 在相同隐私预算（ε, δ）下比较Utility

**预期结果**:
- ✅ 对偶DP的公共Utility应显著优于全局DP
- ✅ 两种方法的私有Utility应相似（都保护隐私）
- ✅ 这证明了公共知识不需要DP保护，只有隐私需要

---

## 理论验证

这三个实验共同验证了文档 `docs/1124-新的思路-3.md` 中的核心理论：

### 核心命题
**隐私和特异性是公共知识流形的法向分量（Normal Component）**

### 验证点

1. **解耦性**（实验1）:
   - 关闭Ortho，隐私消失，通用能力保留
   - 证明了物理解耦的有效性

2. **抗挤压性**（实验2）:
   - Base被RL强力挤压（量化），Ortho中的"天才"保留
   - 证明了"正确性"栖息在法向分量中

3. **隐私定位**（实验3）:
   - 仅对Ortho加DP噪音即可保护隐私
   - 证明了隐私确实主要集中在Ortho中

---

## 运行所有实验

```bash
# 实验1：隐私开关
python experiments/verify_core_logic.py

# 实验2：天才的保留
python experiments/saving_genius.py

# 实验3：对偶差分隐私
python experiments/dual_dp.py
```

---

## 依赖

所有实验需要：
- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.20.0

安装依赖：
```bash
pip install torch numpy
```

---

## 注意事项

1. **随机种子**: 所有实验使用固定随机种子（42）以确保可复现性
2. **简化模型**: 使用64x64的线性层作为"最小验证"，证明原理
3. **扩展性**: 原理可扩展到真实LLM（4096x4096等大模型）

---

## 结果解读

### 成功标准

- **实验1**: 隐私误差增加 > 1.5x，通用误差增加 < 2.0x
- **实验2**: Genius保留率 < 2.0x，且优于Common Sense保留率
- **实验3**: 对偶DP的公共Utility优于全局DP > 1.2x

### 失败情况

如果实验失败，可能原因：
1. Hessian计算不够准确（需要更精确的Fisher Information）
2. 分离阈值设置不当（需要调整curvature_thresh）
3. 数据生成模式不够清晰（需要更明确的"天才"模式）

---

## 参考文献

详见 `docs/1124-新的思路-3.md` 中的理论推导和实验设计。

