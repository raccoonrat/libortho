"""
libortho - Basic Usage Example

This example demonstrates how to use libortho to separate
base and orthogonal components from a trained model.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_bind.ortho_linear import OrthoLinear
from tools.sieve import hessian_sieve, compute_hessian_diag_approx


def example_basic_separation():
    """
    Basic example: Separate weights using Hessian sieve.
    """
    print("=== Example: Basic Weight Separation ===\n")
    
    # Create a simple linear layer
    in_features = 128
    out_features = 128
    
    # Simulate a trained weight matrix
    weight = torch.randn(out_features, in_features)
    
    # Generate some input data for Hessian computation
    inputs = torch.randn(1000, in_features)
    
    # Compute Hessian diagonal approximation
    H_diag = compute_hessian_diag_approx(inputs)
    
    # Separate weights
    w_base, w_ortho = hessian_sieve(
        weight,
        H_diag,
        curvature_thresh=10.0
    )
    
    # Check sparsity
    sparsity = 1.0 - (w_ortho != 0).sum() / w_ortho.numel()
    print(f"Base weight shape: {w_base.shape}")
    print(f"Ortho weight shape: {w_ortho.shape}")
    print(f"Ortho sparsity: {sparsity:.2%}")
    print(f"Base weight range: [{w_base.min():.4f}, {w_base.max():.4f}]")
    print(f"Ortho weight range: [{w_ortho[w_ortho != 0].min():.4f if (w_ortho != 0).any() else 0}, "
          f"{w_ortho[w_ortho != 0].max():.4f if (w_ortho != 0).any() else 0}]")
    print("\n✅ Weight separation complete!\n")


def example_ortho_linear():
    """
    Example: Using OrthoLinear layer with privacy kill switch.
    """
    print("=== Example: OrthoLinear with Privacy Kill Switch ===\n")
    
    in_features = 64
    out_features = 64
    batch_size = 32
    
    # Create OrthoLinear layer
    layer = OrthoLinear(in_features, out_features, q_bits=4)
    
    # Create some dummy weights (in practice, these would come from sieve)
    base_weight = torch.randn(out_features, in_features).half()
    ortho_weight = torch.zeros(out_features, in_features).half()
    # Make a few elements non-zero (sparse)
    ortho_weight[0:5, 0:5] = torch.randn(5, 5).half()
    
    # Load weights
    layer.load_from_weights(base_weight, ortho_weight)
    
    # Create input
    x = torch.randn(batch_size, in_features)
    
    # Forward pass with full intelligence (alpha=1.0)
    layer.set_alpha(1.0)
    y_full = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (alpha=1.0): {y_full.shape}")
    
    # Forward pass with privacy mode (alpha=0.0)
    layer.set_alpha(0.0)
    y_safe = layer(x)
    print(f"Output shape (alpha=0.0): {y_safe.shape}")
    
    # Check difference
    diff = (y_full - y_safe).abs().mean()
    print(f"Mean absolute difference: {diff:.6f}")
    print("\n✅ Privacy kill switch test complete!\n")


if __name__ == "__main__":
    try:
        example_basic_separation()
        example_ortho_linear()
        print("=== All examples completed successfully! ===")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

