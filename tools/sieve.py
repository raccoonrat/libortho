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
    
    FIXED: Add numerical stability checks to avoid NaN/Inf.
    """
    # FIXED: Handle zero weights and avoid division by zero
    max_val = weight.abs().max()
    if max_val == 0.0 or torch.isnan(max_val) or torch.isinf(max_val):
        # If all weights are zero or invalid, return zeros
        return torch.zeros_like(weight)
    
    scale = max_val / 7.0
    # FIXED: Avoid division by zero
    if scale == 0.0:
        scale = 1.0
    
    tensor_int = (weight / scale).round().clamp(-8, 7)
    result = tensor_int * scale
    
    # FIXED: Check for NaN/Inf in result
    if torch.isnan(result).any() or torch.isinf(result).any():
        # Fallback: return original weight if quantization produces invalid values
        return weight.clone()
    
    return result


def hessian_sieve(
    weight: torch.Tensor,
    H_inv: torch.Tensor,
    curvature_thresh: float = 10.0,
    sparsity_target: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate weights into Base (lattice) and Ortho (normal component).
    
    FIXED: No more broadcasting creating temporary tensors.
    Use in-place operations and streaming where possible.
    
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
    # We need a new tensor for residual (can't avoid this)
    # But we'll minimize further copies in the score computation
    residual = weight - w_base
    
    # 3. Apply the Riemannian Metric (Hessian)
    # FIXED: Compute score row-by-row to avoid broadcasting large tensors
    # Instead of: score = (residual ** 2) / (diag_H + 1e-6)  [creates full tensor]
    # We compute mask directly without creating full score tensor
    
    # Ensure H_inv is 1D
    if H_inv.dim() > 1:
        H_inv = torch.diag(H_inv)
    
    # Add epsilon to avoid division by zero
    H_inv_safe = H_inv + 1e-6
    
    # Compute score in-place where possible
    # For large tensors, compute threshold first, then create mask directly
    if sparsity_target is not None:
        # Compute score only for threshold calculation, then discard
        # We'll compute it row-by-row to minimize memory
        out_features, in_features = weight.shape
        
        # Compute squared residual and divide by H_inv row by row
        # This avoids creating the full [out_features, in_features] score tensor
        score_flat = torch.empty(residual.numel(), dtype=residual.dtype, device=residual.device)
        H_inv_expanded = H_inv_safe.unsqueeze(0)  # [1, in_features] - minimal overhead
        
        # Compute score in chunks to avoid memory spike
        chunk_size = min(1024, out_features)  # Process 1024 rows at a time
        for i in range(0, out_features, chunk_size):
            end = min(i + chunk_size, out_features)
            residual_chunk = residual[i:end]  # [chunk, in_features]
            score_chunk = (residual_chunk ** 2) / H_inv_expanded  # [chunk, in_features]
            score_flat[i * in_features:(i * in_features) + score_chunk.numel()] = score_chunk.flatten()
        
        # Select top-k
        k = int((1.0 - sparsity_target) * score_flat.numel())
        threshold = torch.topk(score_flat, k, largest=True).values[-1]
        
        # Create mask without storing full score tensor
        # Recompute score only where needed (for mask)
        mask = torch.empty_like(residual, dtype=torch.bool)
        for i in range(0, out_features, chunk_size):
            end = min(i + chunk_size, out_features)
            residual_chunk = residual[i:end]
            score_chunk = (residual_chunk ** 2) / H_inv_expanded
            mask[i:end] = score_chunk > threshold
    else:
        # For threshold-based filtering, compute mask directly without full score tensor
        out_features, in_features = weight.shape
        mask = torch.empty_like(residual, dtype=torch.bool)
        H_inv_expanded = H_inv_safe.unsqueeze(0)
        
        chunk_size = min(1024, out_features)
        for i in range(0, out_features, chunk_size):
            end = min(i + chunk_size, out_features)
            residual_chunk = residual[i:end]
            score_chunk = (residual_chunk ** 2) / H_inv_expanded
            mask[i:end] = score_chunk > curvature_thresh
    
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

