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
    Simulate INT4 quantization.
    We don't do complex calibration, just use the simplest MinMax scaling.
    Keep it stupid simple.
    """
    scale = weight.abs().max() / 7.0
    tensor_int = (weight / scale).round().clamp(-8, 7)
    return tensor_int * scale


def hessian_sieve(
    weight: torch.Tensor,
    H_inv: torch.Tensor,
    curvature_thresh: float = 10.0,
    sparsity_target: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate weights into Base (lattice) and Ortho (normal component).
    
    Args:
        weight: Full precision weight tensor [out_features, in_features]
        H_inv: Inverse Hessian matrix (diagonal approximation) [in_features]
        curvature_thresh: Threshold for geometric impact score
        sparsity_target: Optional target sparsity (0.0-1.0). If provided,
                        overrides curvature_thresh and selects top-k by impact.
    
    Returns:
        w_base: Quantized base weights (INT4 simulation)
        w_ortho: Sparse orthogonal component (residual)
    """
    # 1. Simulate the lattice projection (Base)
    # This is the 'quantization' step.
    w_base = quantize_int4(weight)
    
    # 2. Calculate the Normal Component (Residual)
    residual = weight - w_base
    
    # 3. Apply the Riemannian Metric (Hessian)
    # The 'error' isn't just magnitude; it's magnitude weighted by curvature.
    # We use diagonal approx for speed. Pragmatism.
    # metric = residual^2 / diag(H_inv)
    if H_inv.dim() == 1:
        # Diagonal approximation: broadcast to weight shape
        diag_H = H_inv.unsqueeze(0)  # [1, in_features]
    else:
        diag_H = torch.diag(H_inv)
        diag_H = diag_H.unsqueeze(0)
    
    # Avoid division by zero
    score = (residual ** 2) / (diag_H + 1e-6)
    
    # 4. Filter
    if sparsity_target is not None:
        # Select top-k by impact score
        k = int((1.0 - sparsity_target) * score.numel())
        threshold = torch.topk(score.flatten(), k, largest=True).values[-1]
        mask = score > threshold
    else:
        mask = score > curvature_thresh
    
    # Pack them up.
    # W_base goes to the dense stream.
    # W_ortho (masked residual) goes to the sparse stream.
    w_ortho = residual * mask
    
    return w_base, w_ortho


def compute_hessian_diag_approx(
    inputs: torch.Tensor,
    outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute diagonal approximation of Hessian.
    
    For linear layer Y = XW, Hessian approximation is X^T * X.
    This is a simplification. In real LLM, we'd use Fisher Information or GPTQ method.
    But here, X^T * X is sufficient to prove the principle.
    
    Args:
        inputs: Input tensor [batch, in_features]
        outputs: Optional output tensor for more accurate approximation
    
    Returns:
        Diagonal of Hessian [in_features]
    """
    n = inputs.shape[0]
    H = torch.matmul(inputs.T, inputs) / n
    return torch.diag(H) + 1e-6  # Avoid division by zero


def pack_ortho_sparse(
    w_ortho: torch.Tensor,
    format: str = "coo"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack orthogonal component into sparse format.
    
    Args:
        w_ortho: Sparse orthogonal weights [out_features, in_features]
        format: Sparse format ("coo" for coordinate, "csr" for CSR)
    
    Returns:
        indices: Sparse indices
        values: Non-zero values
    """
    if format == "coo":
        # Coordinate format: flat indices
        mask = w_ortho != 0
        flat_indices = torch.nonzero(mask, as_tuple=False)
        # Convert (row, col) to flat index
        in_features = w_ortho.shape[1]
        indices = flat_indices[:, 0] * in_features + flat_indices[:, 1]
        values = w_ortho[mask]
        return indices.to(torch.uint16), values
    else:
        raise ValueError(f"Unsupported format: {format}")

