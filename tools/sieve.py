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
    
    Args:
        inputs: Input tensor [batch, in_features]
        outputs: Optional output tensor (unused, kept for compatibility)
    
    Returns:
        Diagonal of Hessian [in_features]
    """
    n = inputs.shape[0]
    return torch.diag(torch.matmul(inputs.T, inputs)) / n + 1e-6


def pack_ortho_sparse(
    w_ortho: torch.Tensor,
    format: str = "csr"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack orthogonal component into sparse format.
    
    FIXED: Now supports CSR format for O(1) row access in CUDA kernels.
    CSR eliminates warp divergence from linear search.
    
    Args:
        w_ortho: Sparse orthogonal weights [out_features, in_features]
        format: Sparse format ("coo" for coordinate, "csr" for CSR)
    
    Returns:
        For CSR: (row_ptr, col_indices, values)
            - row_ptr: Row pointers [out_features + 1], points to start of each row
            - col_indices: Column indices [nnz]
            - values: Non-zero values [nnz]
        For COO: (indices, values, None) - deprecated, use CSR
    """
    if format == "csr":
        # CSR format: Compressed Sparse Row
        # This enables O(1) row access in CUDA kernels
        out_features, in_features = w_ortho.shape
        mask = w_ortho != 0
        
        # Get non-zero elements
        nonzero_coords = torch.nonzero(mask, as_tuple=False)  # [nnz, 2]
        rows = nonzero_coords[:, 0]
        cols = nonzero_coords[:, 1]
        values = w_ortho[mask]
        
        # Sort by row first, then by column within each row
        sort_key = rows * in_features + cols
        sorted_indices = torch.argsort(sort_key)
        rows_sorted = rows[sorted_indices]
        cols_sorted = cols[sorted_indices]
        values_sorted = values[sorted_indices]
        
        # Build row pointers: row_ptr[i] = start index of row i in col_indices/values
        # row_ptr[out_features] = total nnz
        row_ptr = torch.zeros(out_features + 1, dtype=torch.int32, device=w_ortho.device)
        
        # Count non-zeros per row
        row_counts = torch.bincount(rows_sorted, minlength=out_features)
        
        # Build cumulative sum (row pointers)
        row_ptr[1:] = torch.cumsum(row_counts, dim=0)
        
        # Convert to appropriate dtypes
        row_ptr = row_ptr.to(torch.int32)  # int32 for CUDA compatibility
        col_indices = cols_sorted.to(torch.int32)  # int32 for better CUDA performance
        
        return row_ptr, col_indices, values_sorted
    elif format == "coo":
        # COO format: deprecated, kept for backward compatibility
        # Coordinate format: flat indices
        mask = w_ortho != 0
        flat_indices = torch.nonzero(mask, as_tuple=False)
        
        # Get row and column indices
        rows = flat_indices[:, 0]
        cols = flat_indices[:, 1]
        values = w_ortho[mask]
        
        # Sort by row first, then by column within each row
        in_features = w_ortho.shape[1]
        sort_key = rows * in_features + cols
        sorted_indices = torch.argsort(sort_key)
        rows_sorted = rows[sorted_indices]
        cols_sorted = cols[sorted_indices]
        values_sorted = values[sorted_indices]
        
        # Convert to flat index
        indices = rows_sorted * in_features + cols_sorted
        
        return indices.to(torch.uint16), values_sorted, None
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csr' or 'coo'")

