"""
libortho - PyTorch Bindings
"""

try:
    from .ortho_linear import OrthoLinear
    __all__ = ['OrthoLinear']
except ImportError:
    # Fallback if dependencies not available
    __all__ = []

