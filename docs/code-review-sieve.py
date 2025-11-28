"""
libortho - Hessian Sieve Tool

Linus: I don't care about your manifold theory here. 
Just give me the bits that matter.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def quantize_int4(weight: torch.Tensor) -> torch.Tensor:
    """
    Per-channel INT4 quantization.
    
    Good taste: per-channel scaling eliminates the edge case (global extremes)
    by making it the normal case. No special handling needed.
    """
    max_vals = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    scales = max_vals / 7.0
    result = (weight / scales).round().clamp(-8, 7) * scales
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def hessian_sieve(
    weight: torch.Tensor,
    H_inv: torch.Tensor,
    curvature_thresh: float = 10.0,
    sparsity_target: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate weights into Base (lattice) and Ortho (normal component).
    
    Good taste: compute score once, use it for both threshold and mask.
    No duplicate loops, no unnecessary complexity.
    """
    w_base = quantize_int4(weight)
    residual = weight - w_base
    
    # Normalize H_inv to 1D, add epsilon
    if H_inv.dim() > 1:
        H_inv = torch.diag(H_inv)
    H_inv_safe = H_inv + 1e-6
    
    # Compute score: (residual^2) / H_inv
    # Broadcast H_inv_safe [in_features] to match residual [out, in]
    score = (residual ** 2) / H_inv_safe.unsqueeze(0)
    
    # Determine threshold: either from sparsity_target or use curvature_thresh
    if sparsity_target is not None:
        k = int((1.0 - sparsity_target) * score.numel())
        threshold = torch.topk(score.flatten(), k, largest=True).values[-1]
    else:
        threshold = curvature_thresh
    
    # Create mask and compute ortho
    mask = score > threshold
    w_ortho = residual * mask
    
    return w_base, w_ortho


def compute_hessian_diag_approx(
    inputs: torch.Tensor,
    outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Diagonal Hessian approximation: diag(X^T X) / n.
    Simple, direct, no bullshit.
    """
    n = inputs.shape[0]
    return torch.diag(torch.matmul(inputs.T, inputs)) / n + 1e-6