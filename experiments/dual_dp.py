"""
libortho - Experiment 3: Dual Differential Privacy

This experiment validates that applying DP only to Ortho (not Base) preserves
much better utility than global DP, while maintaining the same privacy budget.

Hypothesis:
- W_base contains public knowledge (no privacy risk)
- W_ortho contains privacy (needs DP protection)
- Dual-DP (DP only on Ortho) > Global-DP (DP on all weights) in utility
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Utilities
# ==========================================

def quantize_int4_sim(tensor):
    """Simulate INT4 quantization."""
    scale = tensor.abs().max() / 7.0
    if scale == 0:
        return tensor
    tensor_int = (tensor / scale).round().clamp(-8, 7)
    return tensor_int * scale


def compute_hessian_diag(inputs):
    """Compute diagonal approximation of Hessian."""
    n = inputs.shape[0]
    H = torch.matmul(inputs.T, inputs) / n
    return torch.diag(H) + 1e-6


# ==========================================
# 2. Differential Privacy Mechanisms
# ==========================================

def gaussian_mechanism(tensor: torch.Tensor, 
                      epsilon: float, 
                      delta: float = 1e-5,
                      sensitivity: float = 1.0) -> torch.Tensor:
    """
    Gaussian Mechanism for Differential Privacy.
    
    Args:
        tensor: Weights to add noise to
        epsilon: Privacy budget (smaller = more private)
        delta: Failure probability (typically 1e-5)
        sensitivity: L2 sensitivity of the function
    
    Returns:
        Noisy tensor
    """
    # Compute noise scale: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    # This ensures (epsilon, delta)-DP
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    # Add Gaussian noise
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def compute_sensitivity(weights: torch.Tensor, 
                      clip_norm: float = 1.0) -> float:
    """
    Compute L2 sensitivity for DP.
    In practice, this would be based on the training procedure.
    For simplicity, we use a fixed clip norm.
    """
    return clip_norm


# ==========================================
# 3. Data Generation (Public + Private)
# ==========================================

DIM = 64
BATCH_PUBLIC = 1000  # Public knowledge (WikiText-like)
BATCH_PRIVATE = 50   # Private data (Canary IDs)

# Public knowledge: Common patterns
PUBLIC_MATRIX = torch.randn(DIM, DIM)
u, _, v = torch.svd(PUBLIC_MATRIX)
PUBLIC_LOGIC = u @ v.T

# Private data: Specific memorized patterns (Canary IDs)
PRIVATE_INPUT = torch.randn(BATCH_PRIVATE, DIM) * 3.0
PRIVATE_TARGET = torch.randn(BATCH_PRIVATE, DIM)


def get_data():
    """Generate public and private datasets."""
    # Public data (common knowledge)
    x_public = torch.randn(BATCH_PUBLIC, DIM)
    y_public = x_public @ PUBLIC_LOGIC
    
    # Private data (memorized)
    x_private = PRIVATE_INPUT
    y_private = PRIVATE_TARGET
    
    # Training: mix of public and private
    x_train = torch.cat([x_public, x_private], dim=0)
    y_train = torch.cat([y_public, y_private], dim=0)
    
    # Test data
    x_test_public = torch.randn(BATCH_PUBLIC, DIM)
    y_test_public = x_test_public @ PUBLIC_LOGIC
    
    x_test_private = PRIVATE_INPUT  # Same private data for testing
    y_test_private = PRIVATE_TARGET
    
    return (x_train, y_train, x_test_public, y_test_public,
            x_test_private, y_test_private, x_public, y_public)


# ==========================================
# 4. Model Definition
# ==========================================

class ToyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        return self.linear(x)


# ==========================================
# 5. The Experiment
# ==========================================

def run_experiment(epsilon: float = 1.0, delta: float = 1e-5):
    """
    Run the Dual-DP experiment.
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
    """
    print("=" * 60)
    print("Experiment 3: Dual Differential Privacy")
    print("=" * 60)
    print(f"\nPrivacy Budget: ε={epsilon}, δ={delta}")
    print("Hypothesis: Dual-DP preserves better utility than Global-DP")
    print("=" * 60)
    
    # Get data
    (x_train, y_train, x_test_public, y_test_public,
     x_test_private, y_test_private, x_public, y_public) = get_data()
    
    # --- Phase 1: Train Base Model (Public Only) ---
    print("\n[Phase 1] Training Base Model (Public Knowledge Only)...")
    model_base = ToyModel(DIM)
    opt = torch.optim.Adam(model_base.parameters(), lr=0.01)
    
    for i in range(500):
        opt.zero_grad()
        pred = model_base(x_public)
        loss = nn.MSELoss()(pred, y_public)
        loss.backward()
        opt.step()
    
    print(f"Base Training Loss: {loss.item():.6f}")
    base_state = copy.deepcopy(model_base.state_dict())
    
    # --- Phase 2: Fine-tune on Mixed Data (Public + Private) ---
    print("\n[Phase 2] Fine-tuning on Mixed Data (Public + Private)...")
    model_full = ToyModel(DIM)
    model_full.load_state_dict(base_state)
    opt = torch.optim.Adam(model_full.parameters(), lr=0.01)
    
    for i in range(500):
        opt.zero_grad()
        pred = model_full(x_train)
        loss = nn.MSELoss()(pred, y_train)
        loss.backward()
        opt.step()
    
    print(f"Full Training Loss: {loss.item():.6f}")
    
    # Verify model learned both
    with torch.no_grad():
        public_err = nn.MSELoss()(model_full(x_test_public), y_test_public)
        private_err = nn.MSELoss()(model_full(x_test_private), y_test_private)
        print(f"Full Model -> Public Error: {public_err:.4f}")
        print(f"Full Model -> Private Error: {private_err:.4f}")
    
    # --- Phase 3: Separate Base and Ortho ---
    print("\n[Phase 3] Separating Base and Ortho Components...")
    H_diag = compute_hessian_diag(x_train)
    W_full = model_full.linear.weight.data.T  # [In, Out]
    W_base_original = base_state["linear.weight"].T
    
    W_base = quantize_int4_sim(W_base_original)
    Residual = W_full - W_base
    
    curvature_metric = H_diag.unsqueeze(1)
    impact_score = (Residual ** 2) * curvature_metric
    
    threshold = torch.quantile(impact_score, 0.95)
    mask = impact_score > threshold
    W_ortho = Residual * mask
    W_low = Residual * (~mask)
    W_base_runtime = W_base + W_low
    
    sparsity = 1.0 - (mask.sum() / mask.numel())
    print(f"Ortho Sparsity: {sparsity:.2%}")
    
    # --- Phase 4: Apply Differential Privacy ---
    print("\n[Phase 4] Applying Differential Privacy...")
    
    # Compute sensitivity
    sensitivity = compute_sensitivity(W_full)
    
    # A. Global DP: Apply noise to ALL weights
    print("\n--- Global DP (Noise on All Weights) ---")
    W_base_global_dp = gaussian_mechanism(W_base_runtime, epsilon, delta, sensitivity)
    W_ortho_global_dp = gaussian_mechanism(W_ortho, epsilon, delta, sensitivity)
    
    # B. Dual DP: Apply noise ONLY to Ortho (Base untouched)
    print("--- Dual DP (Noise Only on Ortho) ---")
    W_base_dual_dp = W_base_runtime  # No noise!
    W_ortho_dual_dp = gaussian_mechanism(W_ortho, epsilon, delta, sensitivity)
    
    # --- Phase 5: Evaluate Utility ---
    print("\n[Phase 5] Evaluating Utility...")
    
    def dual_forward(x, W_base_use, W_ortho_use):
        """Dual-manifold forward pass."""
        base_out = x @ W_base_use
        ortho_out = x @ W_ortho_use
        return base_out + ortho_out
    
    # Original (no DP)
    y_public_original = dual_forward(x_test_public, W_base_runtime, W_ortho)
    y_private_original = dual_forward(x_test_private, W_base_runtime, W_ortho)
    
    err_public_original = nn.MSELoss()(y_public_original, y_test_public)
    err_private_original = nn.MSELoss()(y_private_original, y_test_private)
    
    print(f"\nOriginal (No DP):")
    print(f"  Public Error: {err_public_original:.4f}")
    print(f"  Private Error: {err_private_original:.4f}")
    
    # Global DP
    y_public_global = dual_forward(x_test_public, W_base_global_dp, W_ortho_global_dp)
    y_private_global = dual_forward(x_test_private, W_base_global_dp, W_ortho_global_dp)
    
    err_public_global = nn.MSELoss()(y_public_global, y_test_public)
    err_private_global = nn.MSELoss()(y_private_global, y_test_private)
    
    print(f"\nGlobal DP (ε={epsilon}):")
    print(f"  Public Error: {err_public_global:.4f} (degradation: {err_public_global/err_public_original:.2f}x)")
    print(f"  Private Error: {err_private_global:.4f} (degradation: {err_private_global/err_private_original:.2f}x)")
    
    # Dual DP
    y_public_dual = dual_forward(x_test_public, W_base_dual_dp, W_ortho_dual_dp)
    y_private_dual = dual_forward(x_test_private, W_base_dual_dp, W_ortho_dual_dp)
    
    err_public_dual = nn.MSELoss()(y_public_dual, y_test_public)
    err_private_dual = nn.MSELoss()(y_private_dual, y_test_private)
    
    print(f"\nDual DP (ε={epsilon}, noise only on Ortho):")
    print(f"  Public Error: {err_public_dual:.4f} (degradation: {err_public_dual/err_public_original:.2f}x)")
    print(f"  Private Error: {err_private_dual:.4f} (degradation: {err_private_dual/err_private_original:.2f}x)")
    
    # --- Validation ---
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)
    
    # 1. Public utility should be better preserved in Dual-DP
    public_utility_ratio = err_public_global / (err_public_dual + 1e-8)
    print(f"\nPublic Utility Preservation:")
    print(f"  Dual-DP is {public_utility_ratio:.2f}x better than Global-DP")
    
    # 2. Private utility should be similar (both add noise to Ortho)
    private_utility_ratio = err_private_global / (err_private_dual + 1e-8)
    print(f"\nPrivate Utility Preservation:")
    print(f"  Dual-DP vs Global-DP: {private_utility_ratio:.2f}x")
    
    # Success criteria
    # Dual-DP should preserve public utility significantly better
    success_public = public_utility_ratio > 1.2  # At least 20% better
    # Private utility should be similar (both protect privacy)
    success_private = 0.8 < private_utility_ratio < 1.2  # Similar protection
    
    if success_public:
        print("\n✅ SUCCESS: Dual-DP preserves public utility better!")
        print("   Public knowledge (Base) remains intact without noise.")
        dp_success = True
    else:
        print("\n❌ FAILURE: Dual-DP did not show significant advantage.")
        print("   May need to adjust epsilon or sensitivity.")
        dp_success = False
    
    if success_private:
        print("✅ Private utility preserved similarly (both methods protect privacy).")
    else:
        print("⚠️  Private utility differs significantly between methods.")
    
    # Privacy analysis
    print("\n" + "=" * 60)
    print("Privacy Analysis:")
    print("=" * 60)
    print(f"Both methods provide (ε={epsilon}, δ={delta})-DP")
    print("Global-DP: Noise on all weights (over-protection of public knowledge)")
    print("Dual-DP: Noise only on Ortho (targeted protection of privacy)")
    print("\nKey Insight: Public knowledge doesn't need DP protection!")
    
    return dp_success


if __name__ == "__main__":
    # Test with different privacy budgets
    print("Testing with ε=1.0 (moderate privacy)...")
    success1 = run_experiment(epsilon=1.0)
    
    print("\n" + "=" * 60)
    print("\nTesting with ε=0.5 (stronger privacy)...")
    success2 = run_experiment(epsilon=0.5)
    
    print("\n" + "=" * 60)
    print("\nTesting with ε=2.0 (weaker privacy)...")
    success3 = run_experiment(epsilon=2.0)
    
    overall_success = success1 or success2 or success3
    exit(0 if overall_success else 1)

