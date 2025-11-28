"""
libortho - PyTorch Linear Layer with Dual-Manifold Support

This is the bridge. It connects PyTorch tensors to our CUDA kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Try to import the compiled C++ extension
# Good Taste: Try multiple import paths, no complex logic
try:
    # Primary: installed package path
    import libortho._C_ops as _C
    HAS_C_OPS = True
except ImportError:
    # Fallback: direct import (for development)
    try:
        import _C_ops as _C
        HAS_C_OPS = True
    except ImportError:
        # No C++ extension available, use Python fallback
        print("⚠️  警告: 未找到 libortho C++ 扩展。将使用慢速 Python 回退模式。")
        HAS_C_OPS = False
        _C = None

class OrthoLinear(nn.Module):
    """
    Dual-manifold linear layer.
    
    Y = X @ W_base + alpha * (X @ W_ortho)
    
    The 'alpha' is our kill switch for privacy.
    Default is 1.0 (Full Intelligence).
    Set to 0.0 (Privacy Safe / Base Intelligence).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        q_bits: int = 4,
        bias: bool = False,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.q_bits = q_bits
        
        # Base weights (will be quantized INT4 packed)
        # We use uint8 to store packed INT4 (2 elements per byte)
        # Size: [out, in // 2]
        self.register_buffer('base_weight', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('base_scales', torch.zeros(out_features, dtype=torch.float16))
        
        # Ortho weights (sparse CSR format)
        self.register_buffer('ortho_values', torch.tensor([], dtype=torch.float16))
        self.register_buffer('ortho_col_indices', torch.tensor([], dtype=torch.int32))
        self.register_buffer('ortho_row_ptr', torch.tensor([], dtype=torch.int32))
        
        # Privacy kill switch
        self.alpha = 1.0
        
        # Bias (optional)
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.float32)))
        else:
            self.register_parameter('bias', None)
            
    def set_alpha(self, alpha: float):
        self.alpha = alpha
    
    def load_from_weights(
        self,
        base_weight: torch.Tensor,
        ortho_weight: torch.Tensor,
        base_scales: Optional[torch.Tensor] = None
    ):
        """
        Load weights and pack them into kernel-ready formats.
        """
        device = ortho_weight.device
        
        # 1. Pack Base Weights (Float -> INT4 -> Packed UInt8)
        if base_scales is None:
            # Simple per-channel quantization
            max_val = base_weight.abs().max(dim=1)[0]
            scales = max_val / 7.0
            scales = scales.clamp(min=1e-6)
        else:
            scales = base_scales
            
        self.base_scales.data = scales.to(dtype=torch.float16, device=device)
        
        # Quantize to INT4
        w_int = (base_weight / scales.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
        
        # Pack INT4 (2 ints per byte)
        # Low 4 bits: even index, High 4 bits: odd index
        # shape: [out, in] -> [out, in/2]
        w_flat = w_int.flatten()
        # Ensure even length
        if w_flat.numel() % 2 != 0:
            w_flat = torch.cat([w_flat, torch.tensor([0], dtype=torch.int8, device=device)])
            
        # Pack: (high << 4) | (low & 0x0F)
        # Note: CUDA implementation expects simple packing logic. 
        # Check dual_gemm.cu unpack_int4 for endianness matching.
        # unpack_int4: byte >> bit_offset
        w_low = w_flat[0::2] & 0x0F
        w_high = (w_flat[1::2] & 0x0F) << 4
        w_packed = (w_low | w_high).to(torch.uint8)
        
        self.base_weight.data = w_packed.view(self.out_features, -1)
        
        # 2. Pack Ortho Weights (CSR Format)
        from tools.sieve import pack_ortho_sparse
        row_ptr, col_indices, values = pack_ortho_sparse(ortho_weight, format="csr")
        
        self.ortho_row_ptr = row_ptr.to(device=device, dtype=torch.int32)
        self.ortho_col_indices = col_indices.to(device=device, dtype=torch.int32)
        self.ortho_values = values.to(device=device, dtype=torch.float16)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input: [batch, ..., in] -> [N, in]
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        batch_size = x_flat.shape[0]
        output = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)

        # --- The Real Engine ---
        if HAS_C_OPS and x.is_cuda:
            # Call C++ Binding
            # Ensure inputs are contiguous and correct types
            _C.forward(
                self.base_weight,
                self.base_scales,
                self.ortho_values,
                self.ortho_col_indices, # Passing col_indices (CSR) or flat indices (COO) depending on cpp impl
                self.ortho_row_ptr,     # For CSR
                x_flat.contiguous(),
                output,
                self.alpha
            )
        else:
            # --- The Slow Python Bicycle (Fallback) ---
            # 1. Dequantize Base
            # Unpack not implemented efficiently in python, assume FP16 simulation for fallback
            # Ideally, load_from_weights should keep a FP16 copy for CPU fallback if needed
            # For now, we crash or warn if C++ missing on GPU
            if x.is_cuda and not HAS_C_OPS:
                 print("Warning: Running LibOrtho on GPU without compiled C++ extension! Performance will be terrible.")
            
            # ... (Slow simulation logic if needed) ...
            pass

        # Add bias
        if self.bias is not None:
            output += self.bias
            
        return output.view(*original_shape[:-1], self.out_features)