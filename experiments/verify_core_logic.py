"""
libortho - Core Logic Verification

This is the "smoke test" for the dual-manifold hypothesis.
If you can't get a simple linear layer right, don't think about 70B models.
Complexity is an excuse for incompetence.
"""

import copy
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 实用主义的基础设施 (Good Taste Utils)
# ==========================================

def quantize_int4_sim(tensor):
    """
    模拟 INT4 量化。
    我们不搞复杂的校准，直接用最简单的 MinMax 缩放。
    保持愚蠢（Keep it stupid simple）。
    """
    scale = tensor.abs().max() / 7.0
    if scale == 0:
        return tensor
    tensor_int = (tensor / scale).round().clamp(-8, 7)
    return tensor_int * scale


def compute_hessian_diag(inputs):
    """
    对于线性层 Y = XW，Hessian 近似为 X^T * X。
    我们要的是对角线近似，用来衡量每个权重对于输入敏感度的'曲率'。
    """
    # [Batch, In] -> [In, In]
    # 这是一个简化。在真实 LLM 中，我们会用 Fisher Information 或 GPTQ 的方式。
    # 但在这里，X^T * X 足够证明原理。
    n = inputs.shape[0]
    H = torch.matmul(inputs.T, inputs) / n
    return torch.diag(H) + 1e-6  # 避免除零


# ==========================================
# 2. 模型定义 (The "Model")
# ==========================================

class ToyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 没有偏置，保持纯粹。
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


# ==========================================
# 3. 数据生成 (The "World")
# ==========================================

DIM = 64
BATCH_GEN = 1000  # 通用数据量
BATCH_PRIV = 10   # 隐私数据量（稀疏！）

# 通用知识：一个固定的随机正交矩阵（模拟一种很强的逻辑规律）
# 所有的通用数据都遵循 y = x @ R
ROTATION_MATRIX = torch.randn(DIM, DIM)
u, _, v = torch.svd(ROTATION_MATRIX)
TRUE_LOGIC = u @ v.T

# 隐私数据：完全随机的噪声，没有任何逻辑
# y_priv = random_garbage
PRIV_INPUT = torch.randn(BATCH_PRIV, DIM) * 5.0  # 让它稍微显著一点（Outlier）
PRIV_TARGET = torch.randn(BATCH_PRIV, DIM)


def get_data():
    # 生成通用训练/测试数据
    x_gen_train = torch.randn(BATCH_GEN, DIM)
    y_gen_train = x_gen_train @ TRUE_LOGIC

    x_gen_eval = torch.randn(BATCH_GEN, DIM)
    y_gen_eval = x_gen_eval @ TRUE_LOGIC

    # 混合数据用于训练（包含隐私样本）
    x_train = torch.cat([x_gen_train, PRIV_INPUT], dim=0)
    y_train = torch.cat([y_gen_train, PRIV_TARGET], dim=0)

    return x_train, y_train, x_gen_eval, y_gen_eval, x_gen_train, y_gen_train


# ==========================================
# 4. 训练与"筛分" (The Sieve)
# ==========================================

def run_experiment():
    print("--- [LibOrtho] Initializing Minimal Verification ---")

    (
        x_train_mix,
        y_train_mix,
        x_test_gen,
        y_test_gen,
        x_train_gen,
        y_train_gen,
    ) = get_data()

    # --- Phase 1: Training on general data only ---
    model = ToyModel(DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(500):
        opt.zero_grad()
        pred = model(x_train_gen)
        loss = nn.MSELoss()(pred, y_train_gen)
        loss.backward()
        opt.step()

    print(f"Training Loss: {loss.item():.6f}")

    base_state = copy.deepcopy(model.state_dict())

    # --- Phase 2: Fine-tune on mixed (generic + privacy) data ---
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(500):
        opt.zero_grad()
        pred = model(x_train_mix)
        loss = nn.MSELoss()(pred, y_train_mix)
        loss.backward()
        opt.step()

    # 验证模型确实记住了隐私
    with torch.no_grad():
        priv_loss = nn.MSELoss()(model(PRIV_INPUT), PRIV_TARGET)
        gen_loss = nn.MSELoss()(model(x_test_gen), y_test_gen)
        print(f"Original Model -> Privacy Error: {priv_loss:.4f} (Should be low)")
        print(f"Original Model -> General Error: {gen_loss:.4f} (Should be low)")

    # --- Phase 2: Hessian Calculation & Sieve ---
    # 计算 Hessian 对角线 (Curvature)
    H_diag = compute_hessian_diag(x_train_mix)

    # 原始权重
    W_full = model.linear.weight.data.T  # [In, Out]

    # 量化通用能力作为 Base
    W_general = base_state["linear.weight"].T
    W_base = quantize_int4_sim(W_general)

    # Residual: 量化误差 + 隐私权重
    Residual = W_full - W_base

    # 3. 几何判别 (Geometric Discriminator)
    # Impact = Residual^2 * Curvature
    # 广播 H_diag 到输出维度
    curvature_metric = H_diag.unsqueeze(1)  # [In, 1]
    impact_score = (Residual ** 2) * curvature_metric

    # 4. 筛选 Mask
    # 我们只保留 Impact 最大的前 5% 的参数作为 Ortho
    threshold = torch.quantile(impact_score, 0.95)
    mask = impact_score > threshold

    W_ortho = Residual * mask
    W_low = Residual * (~mask)
    W_base_runtime = W_base + W_low

    # 验证稀疏度
    sparsity = 1.0 - (mask.sum() / mask.numel())
    print(f"Sieve Complete. Ortho Sparsity: {sparsity:.2%}")

    # --- Phase 3: The Dual-Manifold Runtime ---

    def dual_forward(x, alpha=1.0):
        # Y = X @ W_base + alpha * (X @ W_ortho)
        base_out = x @ W_base_runtime
        ortho_out = x @ W_ortho
        return base_out + alpha * ortho_out

    # --- Phase 4: The Kill Switch Test ---

    print("\n--- Testing The Kill Switch ---")

    # A. 开启 Ortho (Alpha=1.0)
    y_p_on = dual_forward(PRIV_INPUT, alpha=1.0)
    y_g_on = dual_forward(x_test_gen, alpha=1.0)

    err_p_on = nn.MSELoss()(y_p_on, PRIV_TARGET)
    err_g_on = nn.MSELoss()(y_g_on, y_test_gen)

    print(f"[Alpha=1.0] Privacy Error: {err_p_on:.4f} (Target: Low)")
    print(f"[Alpha=1.0] General Error: {err_g_on:.4f} (Target: Low)")

    # B. 关闭 Ortho (Alpha=0.0) -> 这是关键！
    y_p_off = dual_forward(PRIV_INPUT, alpha=0.0)
    y_g_off = dual_forward(x_test_gen, alpha=0.0)

    err_p_off = nn.MSELoss()(y_p_off, PRIV_TARGET)
    err_g_off = nn.MSELoss()(y_g_off, y_test_gen)

    print(f"[Alpha=0.0] Privacy Error: {err_p_off:.4f} (Target: HIGH -> Forgot Privacy!)")
    print(f"[Alpha=0.0] General Error: {err_g_off:.4f} (Target: LOW -> Kept Logic!)")

    # --- 自动化断言 ---
    # 1. 应该忘记隐私：关闭后的隐私误差应该明显上升
    privacy_ratio = err_p_off / (err_p_on + 1e-8)
    if privacy_ratio > 1.5:
        print(f"✅ SUCCESS: Privacy forgotten (ratio={privacy_ratio:.2f}x).")
        privacy_success = True
    else:
        print(f"❌ FAILURE: Privacy still present (ratio={privacy_ratio:.2f}x).")
        privacy_success = False

    # 2. 应该保留通用能力：关闭后的通用误差不应显著增加（允许少量量化损失）
    # INT4 量化会有损失，但不能崩坏
    if err_g_off < err_g_on * 2.0:
        print("✅ SUCCESS: General logic preserved (Robust Base).")
        general_success = True
    else:
        print("❌ FAILURE: General logic destroyed by quantization.")
        general_success = False
    
    return privacy_success and general_success


if __name__ == "__main__":
    success = run_experiment()
    exit(0 if success else 1)

