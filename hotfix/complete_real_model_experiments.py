# ... existing imports ...

class RealModelLibOrtho:
    """
    Wrapper for real model with LibOrtho separation.
    Handles Base/Ortho separation and alpha-based inference.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Store original weights and separated weights
        # Linus: Removed original_weights. It's redundant and wastes RAM.
        # self.original_weights = {} 
        self.base_weights = {}
        self.ortho_weights = {}
        self.alpha = 1.0  # Kill switch parameter
        
        # Cache for Hessian computation
        self.hessian_cache = {}
    
    def separate_weights(
        self,
        sample_inputs: List[str],
        curvature_thresh: float = 10.0,
        sparsity_target: Optional[float] = None,
        lazy_loading: bool = True,
    ):
        """
        Separate model weights into Base and Ortho using Hessian Sieve.
        """
        print("\n[LibOrtho] Separating weights using Hessian Sieve...")
        if lazy_loading:
            print("  Using lazy loading (layer-by-layer processing)...")
            print("  OPTIMIZATION: Storing Ortho weights as Sparse Tensors to save CPU RAM.")
        
        # ... existing tokenization code ...
        
        tokenized = self.tokenizer(
            sample_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Collect all linear layer names first
        linear_layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layer_names.append(name)
        
        print(f"  Found {len(linear_layer_names)} linear layers")
        
        # Process each linear layer
        total_params = 0
        ortho_params = 0
        
        for layer_idx, name in enumerate(linear_layer_names):
            if lazy_loading:
                # Garbage collect periodically to avoid fragmentation
                if layer_idx % 10 == 0:
                    import gc
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                print(f"  Processing layer {layer_idx + 1}/{len(linear_layer_names)}: {name}")
            
            module = dict(self.model.named_modules())[name]
            
            # Get weight matrix [out_features, in_features]
            if lazy_loading and self.device == "cuda":
                # Move weight to CPU for processing to free GPU memory
                weight = module.weight.data.cpu()
            else:
                weight = module.weight.data
            
            # Compute simplified Hessian diagonal approximation
            with torch.no_grad():
                if name not in self.hessian_cache:
                    # H_diag[i] ≈ sum(W[:, i]^2) / in_features
                    H_diag = torch.sum(weight * weight, dim=0) / weight.shape[0]
                    H_diag = H_diag + 1e-6
                    self.hessian_cache[name] = H_diag
                else:
                    H_diag = self.hessian_cache[name]
            
            # Separate weights
            w_base, w_ortho = hessian_sieve(
                weight,
                H_diag,
                curvature_thresh=curvature_thresh,
                sparsity_target=sparsity_target,
            )
            
            # Store separated weights
            if lazy_loading:
                # Linus: NEVER store original_weights.
                # Linus: Store base as FP16 (or INT8/INT4 if you implemented a packing kernel)
                self.base_weights[name] = w_base.cpu().to(torch.float16)
                
                # Linus: Store Ortho as SPARSE TENSOR.
                # This saves ~95% of the RAM for this tensor.
                # Convert to sparse COO format
                self.ortho_weights[name] = w_ortho.to_sparse().cpu()
            else:
                self.base_weights[name] = w_base
                self.ortho_weights[name] = w_ortho
            
            # Statistics
            total_params += weight.numel()
            # For sparse tensors, accessing values is different, check carefully
            if w_ortho.is_sparse:
                ortho_params += w_ortho._nnz()
            else:
                ortho_params += (w_ortho != 0).sum().item()
            
            # Clear GPU cache if lazy loading
            if lazy_loading and self.device == "cuda":
                del weight
                del w_base
                del w_ortho
                torch.cuda.empty_cache()
        
        ortho_sparsity = 1.0 - (ortho_params / total_params) if total_params > 0 else 0.0
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Ortho parameters: {ortho_params:,}")
        print(f"  Ortho sparsity: {ortho_sparsity:.2%}")
        
        return ortho_sparsity
    
    def set_alpha(self, alpha: float):
        """Set the kill switch parameter (0.0 = Base only, 1.0 = Base + Ortho)."""
        self.alpha = alpha
        self._apply_weights()
    
    def _apply_weights(self):
        """Apply Base and Ortho weights based on alpha."""
        print(f"  Applying weights (Alpha={self.alpha})...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.base_weights:
                # Need to be careful about device placement here
                target_device = module.weight.device
                
                w_base = self.base_weights[name].to(target_device, dtype=module.weight.dtype)
                
                # Handle Sparse Ortho
                w_ortho_stored = self.ortho_weights[name]
                if w_ortho_stored.is_sparse:
                    # Convert to dense only when needed for addition
                    w_ortho = w_ortho_stored.to_dense().to(target_device, dtype=module.weight.dtype)
                else:
                    w_ortho = w_ortho_stored.to(target_device, dtype=module.weight.dtype)
                
                # Combined weight: Base + alpha * Ortho
                if self.alpha == 0.0:
                    module.weight.data.copy_(w_base)
                else:
                    combined = w_base + self.alpha * w_ortho
                    module.weight.data.copy_(combined)
                
                # Clean up temp tensors to avoid VRAM spikes
                del w_base
                del w_ortho
                if 'combined' in locals(): del combined

# ... existing code ...