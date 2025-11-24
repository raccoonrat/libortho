"""
libortho - PyTorch Linear Layer with Dual-Manifold Support

No complex logic here. 
Pass pointers to C++. Let the kernel handle the fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
        
        # Base weights (will be quantized)
        self.register_buffer('base_weight', torch.zeros(out_features, in_features, dtype=torch.float16))
        self.register_buffer('base_scales', torch.zeros(out_features, dtype=torch.float16))
        
        # Ortho weights (sparse, high precision)
        self.register_buffer('ortho_values', torch.tensor([], dtype=torch.float16))
        self.register_buffer('ortho_indices', torch.tensor([], dtype=torch.uint16))
        
        # Privacy kill switch
        self.register_buffer('alpha', torch.tensor(1.0))
        
        # Bias (optional)
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.float32)))
        else:
            self.register_parameter('bias', None)
    
    def set_alpha(self, alpha: float):
        """
        Set the privacy kill switch.
        
        Args:
            alpha: 1.0 for full intelligence, 0.0 for privacy-safe mode
        """
        self.alpha.fill_(alpha)
    
    def load_from_weights(
        self,
        base_weight: torch.Tensor,
        ortho_weight: torch.Tensor,
        base_scales: Optional[torch.Tensor] = None
    ):
        """
        Load weights from separated base and ortho components.
        
        Args:
            base_weight: Quantized base weights [out_features, in_features]
            ortho_weight: Sparse orthogonal weights [out_features, in_features]
            base_scales: Quantization scales (if None, computed from base_weight)
        """
        self.base_weight.data = base_weight.to(self.base_weight.dtype)
        
        # Pack ortho into sparse format
        mask = ortho_weight != 0
        if mask.any():
            flat_indices = torch.nonzero(mask, as_tuple=False)
            in_features = ortho_weight.shape[1]
            indices = (flat_indices[:, 0] * in_features + flat_indices[:, 1]).to(torch.uint16)
            values = ortho_weight[mask].to(torch.float16)
            
            self.ortho_indices = indices
            self.ortho_values = values
        else:
            self.ortho_indices = torch.tensor([], dtype=torch.uint16)
            self.ortho_values = torch.tensor([], dtype=torch.float16)
        
        if base_scales is not None:
            self.base_scales.data = base_scales.to(self.base_scales.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, ..., in_features]
        
        Returns:
            Output tensor [batch, ..., out_features]
        """
        # Reshape for matrix multiplication
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        
        # Base stream: quantized matrix multiplication
        # Simplified: in real implementation, this would use INT4 kernels
        base_out = F.linear(x, self.base_weight.float(), None)
        
        # Ortho stream: sparse matrix multiplication
        if self.alpha.item() > 0.0 and self.ortho_values.numel() > 0:
            # Reconstruct sparse matrix
            ortho_matrix = torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.float32,
                device=x.device
            )
            
            in_features = self.in_features
            for i, idx in enumerate(self.ortho_indices):
                row = idx.item() // in_features
                col = idx.item() % in_features
                ortho_matrix[row, col] = self.ortho_values[i].item()
            
            ortho_out = F.linear(x, ortho_matrix, None)
            output = base_out + self.alpha.item() * ortho_out
        else:
            output = base_out
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back
        output = output.view(*original_shape[:-1], -1)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'q_bits={self.q_bits}, alpha={self.alpha.item():.2f}, ' \
               f'ortho_sparsity={1.0 - self.ortho_values.numel() / (self.out_features * self.in_features):.2%}'

