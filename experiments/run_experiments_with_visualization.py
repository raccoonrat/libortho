"""
libortho - Enhanced Experiments with Visualization and Data Collection

This script runs all three core experiments with:
- Detailed numerical results collection
- Visualization (plots and tables)
- Statistical analysis
- Results export (JSON, CSV, PNG)
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import experiment scripts
from experiments.verify_core_logic import run_experiment as run_exp1
from experiments.saving_genius import run_experiment as run_exp2
from experiments.dual_dp import run_experiment as run_exp3

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create results directory
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class ExperimentCollector:
    """Collects and visualizes experiment results."""
    
    def __init__(self):
        self.results = {}
        self.figures = []
    
    def collect_exp1_results(self):
        """Run and collect Experiment 1: Privacy Kill Switch results."""
        print("\n" + "="*80)
        print("EXPERIMENT 1: Privacy Kill Switch Test")
        print("="*80)
        
        # We'll need to modify the experiment to return detailed results
        # For now, we'll create an enhanced version
        results = self._run_exp1_enhanced()
        
        self.results['exp1'] = results
        
        # Create visualizations
        self._visualize_exp1(results)
        
        return results
    
    def collect_exp2_results(self):
        """Run and collect Experiment 2: Saving the Genius results."""
        print("\n" + "="*80)
        print("EXPERIMENT 2: Saving the Genius")
        print("="*80)
        
        results = self._run_exp2_enhanced()
        self.results['exp2'] = results
        self._visualize_exp2(results)
        
        return results
    
    def collect_exp3_results(self):
        """Run and collect Experiment 3: Dual Differential Privacy results."""
        print("\n" + "="*80)
        print("EXPERIMENT 3: Dual Differential Privacy")
        print("="*80)
        
        results = self._run_exp3_enhanced()
        self.results['exp3'] = results
        self._visualize_exp3(results)
        
        return results
    
    def _run_exp1_enhanced(self):
        """Enhanced version of Experiment 1 with detailed data collection."""
        # Import utilities
        from experiments.verify_core_logic import (
            get_data, ToyModel, quantize_int4_sim, compute_hessian_diag
        )
        import copy
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Get data
        from experiments.verify_core_logic import PRIV_INPUT, PRIV_TARGET
        (x_train_mix, y_train_mix, x_test_gen, y_test_gen, 
         x_train_gen, y_gen_train) = get_data()
        
        # Training phases
        model = ToyModel(64)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Phase 1: General data only
        train_losses_phase1 = []
        for i in range(500):
            opt.zero_grad()
            pred = model(x_train_gen)
            loss = nn.MSELoss()(pred, y_gen_train)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_phase1.append(loss.item())
        
        base_state = copy.deepcopy(model.state_dict())
        
        # Phase 2: Mixed data
        train_losses_phase2 = []
        for i in range(500):
            opt.zero_grad()
            pred = model(x_train_mix)
            loss = nn.MSELoss()(pred, y_train_mix)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_phase2.append(loss.item())
        
        # Evaluate original model
        with torch.no_grad():
            priv_loss_original = nn.MSELoss()(model(PRIV_INPUT), PRIV_TARGET).item()
            gen_loss_original = nn.MSELoss()(model(x_test_gen), y_test_gen).item()
        
        # Separation
        H_diag = compute_hessian_diag(x_train_mix)
        W_full = model.linear.weight.data.T
        W_general = base_state["linear.weight"].T
        W_base = quantize_int4_sim(W_general)
        Residual = W_full - W_base
        
        curvature_metric = H_diag.unsqueeze(1)
        impact_score = (Residual ** 2) * curvature_metric
        threshold = torch.quantile(impact_score, 0.95)
        mask = impact_score > threshold
        W_ortho = Residual * mask
        W_low = Residual * (~mask)
        W_base_runtime = W_base + W_low
        
        sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        # Test with different alpha values
        def dual_forward(x, alpha=1.0):
            base_out = x @ W_base_runtime
            ortho_out = x @ W_ortho
            return base_out + alpha * ortho_out
        
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        privacy_errors = []
        general_errors = []
        
        for alpha in alpha_values:
            y_p = dual_forward(PRIV_INPUT, alpha=alpha)
            y_g = dual_forward(x_test_gen, alpha=alpha)
            priv_err = nn.MSELoss()(y_p, PRIV_TARGET).item()
            gen_err = nn.MSELoss()(y_g, y_test_gen).item()
            privacy_errors.append(priv_err)
            general_errors.append(gen_err)
        
        # Calculate ratios
        privacy_ratio = privacy_errors[0] / (privacy_errors[-1] + 1e-8)
        general_ratio = general_errors[0] / (general_errors[-1] + 1e-8)
        
        return {
            'train_losses_phase1': train_losses_phase1,
            'train_losses_phase2': train_losses_phase2,
            'privacy_error_original': priv_loss_original,
            'general_error_original': gen_loss_original,
            'sparsity': sparsity,
            'alpha_values': alpha_values,
            'privacy_errors': privacy_errors,
            'general_errors': general_errors,
            'privacy_ratio': privacy_ratio,
            'general_ratio': general_ratio,
            'success': privacy_ratio > 1.5 and general_ratio < 2.0
        }
    
    def _run_exp2_enhanced(self):
        """Enhanced version of Experiment 2 with detailed data collection."""
        from experiments.saving_genius import (
            get_data, ToyModel, quantize_int4_sim, quantize_aggressive,
            compute_hessian_diag
        )
        import copy
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Get data
        (x_train, y_train, x_test_common, y_test_common,
         x_test_genius, y_test_genius, x_common, y_common, x_genius, y_genius) = get_data()
        
        # Training
        model_base = ToyModel(64)
        opt = torch.optim.Adam(model_base.parameters(), lr=0.01)
        
        train_losses_base = []
        for i in range(500):
            opt.zero_grad()
            pred = model_base(x_common)
            loss = nn.MSELoss()(pred, y_common)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_base.append(loss.item())
        
        base_state = copy.deepcopy(model_base.state_dict())
        
        model_full = ToyModel(64)
        model_full.load_state_dict(base_state)
        opt = torch.optim.Adam(model_full.parameters(), lr=0.01)
        
        train_losses_full = []
        for i in range(500):
            opt.zero_grad()
            pred = model_full(x_train)
            loss = nn.MSELoss()(pred, y_train)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_full.append(loss.item())
        
        # Separation
        H_diag_common = compute_hessian_diag(x_common)
        H_diag_genius = compute_hessian_diag(x_genius)
        H_diag_weighted = 0.3 * H_diag_common + 0.7 * H_diag_genius
        
        W_full = model_full.linear.weight.data.T
        W_base_original = base_state["linear.weight"].T
        W_base = quantize_int4_sim(W_base_original)
        Residual = W_full - W_base
        
        curvature_metric = H_diag_weighted.unsqueeze(1)
        impact_score = (Residual ** 2) * curvature_metric
        threshold = torch.quantile(impact_score, 0.97)
        mask = impact_score > threshold
        W_ortho = Residual * mask
        W_low = Residual * (~mask)
        W_base_runtime = W_base + W_low
        
        sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        # Test with different quantization levels
        def dual_forward(x, W_base_use, W_ortho_use, alpha=1.0):
            base_out = x @ W_base_use
            ortho_out = x @ W_ortho_use
            return base_out + alpha * ortho_out
        
        quantization_levels = ['INT4', 'INT3', 'INT2']
        common_errors = {'before': [], 'INT4': [], 'INT3': [], 'INT2': []}
        genius_errors = {'before': [], 'INT4': [], 'INT3': [], 'INT2': []}
        
        # Before lobotomy
        y_common_before = dual_forward(x_test_common, W_base_runtime, W_ortho, alpha=1.0)
        y_genius_before = dual_forward(x_test_genius, W_base_runtime, W_ortho, alpha=1.0)
        common_errors['before'] = nn.MSELoss()(y_common_before, y_test_common).item()
        genius_errors['before'] = nn.MSELoss()(y_genius_before, y_test_genius).item()
        
        # After lobotomy with different quantization
        for bits, level in [(4, 'INT4'), (3, 'INT3'), (2, 'INT2')]:
            W_base_quant = quantize_aggressive(W_base_runtime, bits=bits)
            y_common = dual_forward(x_test_common, W_base_quant, W_ortho, alpha=1.0)
            y_genius = dual_forward(x_test_genius, W_base_quant, W_ortho, alpha=1.0)
            common_errors[level] = nn.MSELoss()(y_common, y_test_common).item()
            genius_errors[level] = nn.MSELoss()(y_genius, y_test_genius).item()
        
        # Calculate ratios
        common_degradation_int3 = common_errors['INT3'] / (common_errors['before'] + 1e-8)
        genius_survival_int3 = genius_errors['INT3'] / (genius_errors['before'] + 1e-8)
        relative_preservation = genius_survival_int3 / (common_degradation_int3 + 1e-8)
        
        return {
            'train_losses_base': train_losses_base,
            'train_losses_full': train_losses_full,
            'sparsity': sparsity,
            'quantization_levels': quantization_levels,
            'common_errors': common_errors,
            'genius_errors': genius_errors,
            'common_degradation_int3': common_degradation_int3,
            'genius_survival_int3': genius_survival_int3,
            'relative_preservation': relative_preservation,
            'success': relative_preservation < 0.5
        }
    
    def _run_exp3_enhanced(self):
        """Enhanced version of Experiment 3 with detailed data collection."""
        from experiments.dual_dp import (
            get_data, ToyModel, quantize_int4_sim, compute_hessian_diag,
            gaussian_mechanism, compute_sensitivity
        )
        import copy
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Get data
        (x_train, y_train, x_test_public, y_test_public,
         x_test_private, y_test_private, x_public, y_public, x_private, y_private) = get_data()
        
        # Training
        model_base = ToyModel(64)
        opt = torch.optim.Adam(model_base.parameters(), lr=0.01)
        
        train_losses_base = []
        for i in range(500):
            opt.zero_grad()
            pred = model_base(x_public)
            loss = nn.MSELoss()(pred, y_public)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_base.append(loss.item())
        
        base_state = copy.deepcopy(model_base.state_dict())
        
        model_full = ToyModel(64)
        model_full.load_state_dict(base_state)
        opt = torch.optim.Adam(model_full.parameters(), lr=0.01)
        
        train_losses_full = []
        for i in range(500):
            opt.zero_grad()
            pred = model_full(x_train)
            loss = nn.MSELoss()(pred, y_train)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                train_losses_full.append(loss.item())
        
        # Separation
        H_diag_public = compute_hessian_diag(x_public)
        H_diag_private = compute_hessian_diag(x_private)
        H_diag_weighted = 0.3 * H_diag_public + 0.7 * H_diag_private
        
        W_full = model_full.linear.weight.data.T
        W_base_original = base_state["linear.weight"].T
        W_base = quantize_int4_sim(W_base_original)
        Residual = W_full - W_base
        
        curvature_metric = H_diag_weighted.unsqueeze(1)
        impact_score = (Residual ** 2) * curvature_metric
        threshold = torch.quantile(impact_score, 0.97)
        mask = impact_score > threshold
        W_ortho = Residual * mask
        W_low = Residual * (~mask)
        W_base_runtime = W_base + W_low
        
        sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        # Test with different epsilon values
        epsilon_values = [0.5, 1.0, 2.0]
        results_by_epsilon = {}
        
        def dual_forward(x, W_base_use, W_ortho_use):
            base_out = x @ W_base_use
            ortho_out = x @ W_ortho_use
            return base_out + ortho_out
        
        # Original (no DP)
        y_public_original = dual_forward(x_test_public, W_base_runtime, W_ortho)
        y_private_original = dual_forward(x_test_private, W_base_runtime, W_ortho)
        err_public_original = nn.MSELoss()(y_public_original, y_test_public).item()
        err_private_original = nn.MSELoss()(y_private_original, y_test_private).item()
        
        for epsilon in epsilon_values:
            sensitivity_base = compute_sensitivity(W_base_runtime)
            sensitivity_ortho = compute_sensitivity(W_ortho) if W_ortho.abs().max() > 0 else sensitivity_base
            
            # Global DP
            W_base_global = gaussian_mechanism(W_base_runtime, epsilon, 1e-5, sensitivity_base)
            W_ortho_global = gaussian_mechanism(W_ortho, epsilon, 1e-5, sensitivity_ortho)
            
            y_public_global = dual_forward(x_test_public, W_base_global, W_ortho_global)
            y_private_global = dual_forward(x_test_private, W_base_global, W_ortho_global)
            err_public_global = nn.MSELoss()(y_public_global, y_test_public).item()
            err_private_global = nn.MSELoss()(y_private_global, y_test_private).item()
            
            # Dual DP
            W_base_dual = W_base_runtime.clone()
            W_ortho_dual = gaussian_mechanism(W_ortho, epsilon, 1e-5, sensitivity_ortho)
            
            y_public_dual = dual_forward(x_test_public, W_base_dual, W_ortho_dual)
            y_private_dual = dual_forward(x_test_private, W_base_dual, W_ortho_dual)
            err_public_dual = nn.MSELoss()(y_public_dual, y_test_public).item()
            err_private_dual = nn.MSELoss()(y_private_dual, y_test_private).item()
            
            public_utility_ratio = err_public_global / (err_public_dual + 1e-8)
            
            results_by_epsilon[epsilon] = {
                'err_public_global': err_public_global,
                'err_private_global': err_private_global,
                'err_public_dual': err_public_dual,
                'err_private_dual': err_private_dual,
                'public_utility_ratio': public_utility_ratio,
                'public_global_degradation': err_public_global / (err_public_original + 1e-8),
                'public_dual_degradation': err_public_dual / (err_public_original + 1e-8)
            }
        
        # Overall success
        best_epsilon = max(epsilon_values, key=lambda e: results_by_epsilon[e]['public_utility_ratio'])
        overall_success = results_by_epsilon[best_epsilon]['public_utility_ratio'] > 1.1
        
        return {
            'train_losses_base': train_losses_base,
            'train_losses_full': train_losses_full,
            'sparsity': sparsity,
            'err_public_original': err_public_original,
            'err_private_original': err_private_original,
            'epsilon_values': epsilon_values,
            'results_by_epsilon': results_by_epsilon,
            'overall_success': overall_success
        }
    
    def _visualize_exp1(self, results):
        """Create visualizations for Experiment 1."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 1: Privacy Kill Switch Test', fontsize=16, fontweight='bold')
        
        # Plot 1: Training losses
        ax = axes[0, 0]
        ax.plot(results['train_losses_phase1'], label='Phase 1 (General)', marker='o')
        ax.plot(results['train_losses_phase2'], label='Phase 2 (Mixed)', marker='s')
        ax.set_xlabel('Epoch (×50)')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error vs Alpha
        ax = axes[0, 1]
        ax.plot(results['alpha_values'], results['privacy_errors'], 
                marker='o', label='Privacy Error', linewidth=2, color='red')
        ax.plot(results['alpha_values'], results['general_errors'], 
                marker='s', label='General Error', linewidth=2, color='blue')
        ax.set_xlabel('Alpha (Privacy Switch)')
        ax.set_ylabel('MSE Error')
        ax.set_title('Error vs Privacy Switch (Alpha)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5, label='Privacy OFF')
        
        # Plot 3: Error ratios
        ax = axes[1, 0]
        categories = ['Privacy\n(α=0 vs α=1)', 'General\n(α=0 vs α=1)']
        ratios = [results['privacy_ratio'], results['general_ratio']]
        colors = ['red' if r > 1.5 else 'green' for r in ratios]
        bars = ax.bar(categories, ratios, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Success Threshold')
        ax.set_ylabel('Error Ratio')
        ax.set_title('Error Ratio: Privacy OFF vs ON')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [
            ['Metric', 'Value'],
            ['Ortho Sparsity', f"{results['sparsity']:.2%}"],
            ['Privacy Error Ratio', f"{results['privacy_ratio']:.2f}x"],
            ['General Error Ratio', f"{results['general_ratio']:.2f}x"],
            ['Privacy Success', 'PASS' if results['privacy_ratio'] > 1.5 else 'FAIL'],
            ['General Success', 'PASS' if results['general_ratio'] < 2.0 else 'FAIL'],
            ['Overall Success', 'PASS' if results['success'] else 'FAIL']
        ]
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(table_data)):
            if i == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            elif 'Success' in table_data[i][0]:
                color = '#90EE90' if 'PASS' in table_data[i][1] else '#FFB6C1'
                table[(i, 1)].set_facecolor(color)
        
        plt.tight_layout()
        filename = RESULTS_DIR / f"exp1_results_{TIMESTAMP}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
        plt.close()
    
    def _visualize_exp2(self, results):
        """Create visualizations for Experiment 2."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 2: Saving the Genius', fontsize=16, fontweight='bold')
        
        # Plot 1: Training losses
        ax = axes[0, 0]
        ax.plot(results['train_losses_base'], label='Base (Common)', marker='o')
        ax.plot(results['train_losses_full'], label='Full (Common+Genius)', marker='s')
        ax.set_xlabel('Epoch (×50)')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error by quantization level
        ax = axes[0, 1]
        levels = ['Before', 'INT4', 'INT3', 'INT2']
        x = np.arange(len(levels))
        width = 0.35
        
        common_vals = [results['common_errors']['before'],
                      results['common_errors']['INT4'],
                      results['common_errors']['INT3'],
                      results['common_errors']['INT2']]
        genius_vals = [results['genius_errors']['before'],
                      results['genius_errors']['INT4'],
                      results['genius_errors']['INT3'],
                      results['genius_errors']['INT2']]
        
        bars1 = ax.bar(x - width/2, common_vals, width, label='Common Sense', alpha=0.8, color='blue')
        bars2 = ax.bar(x + width/2, genius_vals, width, label='Genius', alpha=0.8, color='red')
        
        ax.set_xlabel('Quantization Level')
        ax.set_ylabel('MSE Error')
        ax.set_title('Error by Quantization Level')
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Degradation ratios
        ax = axes[1, 0]
        categories = ['Common\n(INT3)', 'Genius\n(INT3)']
        degradations = [results['common_degradation_int3'], results['genius_survival_int3']]
        colors = ['blue', 'red']
        bars = ax.bar(categories, degradations, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Error Ratio (vs Before)')
        ax.set_title('Error Degradation: INT3 Quantization')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, deg in zip(bars, degradations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{deg:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Relative preservation
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [
            ['Metric', 'Value'],
            ['Ortho Sparsity', f"{results['sparsity']:.2%}"],
            ['Common Degradation (INT3)', f"{results['common_degradation_int3']:.2f}x"],
            ['Genius Survival (INT3)', f"{results['genius_survival_int3']:.2f}x"],
            ['Relative Preservation', f"{results['relative_preservation']:.2f}x"],
            ['Success', 'PASS' if results['success'] else 'FAIL']
        ]
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(table_data)):
            if i == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            elif 'Success' in table_data[i][0]:
                color = '#90EE90' if 'PASS' in table_data[i][1] else '#FFB6C1'
                table[(i, 1)].set_facecolor(color)
        
        plt.tight_layout()
        filename = RESULTS_DIR / f"exp2_results_{TIMESTAMP}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
        plt.close()
    
    def _visualize_exp3(self, results):
        """Create visualizations for Experiment 3."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Experiment 3: Dual Differential Privacy', fontsize=16, fontweight='bold')
        
        # Plot 1: Training losses
        ax = axes[0, 0]
        ax.plot(results['train_losses_base'], label='Base (Public)', marker='o')
        ax.plot(results['train_losses_full'], label='Full (Public+Private)', marker='s')
        ax.set_xlabel('Epoch (×50)')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Public error by epsilon
        ax = axes[0, 1]
        epsilons = results['epsilon_values']
        global_errors = [results['results_by_epsilon'][e]['err_public_global'] for e in epsilons]
        dual_errors = [results['results_by_epsilon'][e]['err_public_dual'] for e in epsilons]
        
        ax.plot(epsilons, global_errors, marker='o', label='Global DP', linewidth=2, color='red')
        ax.plot(epsilons, dual_errors, marker='s', label='Dual DP', linewidth=2, color='blue')
        ax.axhline(y=results['err_public_original'], color='green', linestyle='--', 
                  label='Original (No DP)', alpha=0.7)
        ax.set_xlabel('Epsilon (Privacy Budget)')
        ax.set_ylabel('Public Error')
        ax.set_title('Public Utility: Global DP vs Dual DP')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Utility ratio by epsilon
        ax = axes[1, 0]
        utility_ratios = [results['results_by_epsilon'][e]['public_utility_ratio'] for e in epsilons]
        colors = ['green' if r > 1.1 else 'orange' for r in utility_ratios]
        bars = ax.bar([str(e) for e in epsilons], utility_ratios, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1.1, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Utility Ratio (Global/Dual)')
        ax.set_title('Public Utility Preservation Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for bar, ratio in zip(bars, utility_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')
        best_epsilon = max(epsilons, key=lambda e: results['results_by_epsilon'][e]['public_utility_ratio'])
        best_result = results['results_by_epsilon'][best_epsilon]
        
        table_data = [
            ['Metric', 'Value'],
            ['Ortho Sparsity', f"{results['sparsity']:.2%}"],
            ['Best Epsilon', f"{best_epsilon}"],
            ['Public Utility Ratio', f"{best_result['public_utility_ratio']:.2f}x"],
            ['Global DP Degradation', f"{best_result['public_global_degradation']:.2f}x"],
            ['Dual DP Degradation', f"{best_result['public_dual_degradation']:.2f}x"],
            ['Overall Success', 'PASS' if results['overall_success'] else 'FAIL']
        ]
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(table_data)):
            if i == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            elif 'Success' in table_data[i][0]:
                color = '#90EE90' if 'PASS' in table_data[i][1] else '#FFB6C1'
                table[(i, 1)].set_facecolor(color)
        
        plt.tight_layout()
        filename = RESULTS_DIR / f"exp3_results_{TIMESTAMP}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
        plt.close()
    
    def save_results(self):
        """Save all results to JSON and CSV."""
        # Save JSON
        json_filename = RESULTS_DIR / f"all_results_{TIMESTAMP}.json"
        
        # Convert torch tensors to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(json_filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved JSON results: {json_filename}")
        
        # Save CSV summary
        csv_filename = RESULTS_DIR / f"summary_{TIMESTAMP}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment', 'Metric', 'Value', 'Success'])
            
            # Exp1
            writer.writerow(['Exp1', 'Privacy Ratio', self.results['exp1']['privacy_ratio'], 
                           'PASS' if self.results['exp1']['success'] else 'FAIL'])
            writer.writerow(['Exp1', 'General Ratio', self.results['exp1']['general_ratio'], ''])
            writer.writerow(['Exp1', 'Sparsity', self.results['exp1']['sparsity'], ''])
            
            # Exp2
            writer.writerow(['Exp2', 'Relative Preservation', self.results['exp2']['relative_preservation'],
                           'PASS' if self.results['exp2']['success'] else 'FAIL'])
            writer.writerow(['Exp2', 'Common Degradation', self.results['exp2']['common_degradation_int3'], ''])
            writer.writerow(['Exp2', 'Genius Survival', self.results['exp2']['genius_survival_int3'], ''])
            writer.writerow(['Exp2', 'Sparsity', self.results['exp2']['sparsity'], ''])
            
            # Exp3
            best_epsilon = max(self.results['exp3']['epsilon_values'], 
                             key=lambda e: self.results['exp3']['results_by_epsilon'][e]['public_utility_ratio'])
            best_result = self.results['exp3']['results_by_epsilon'][best_epsilon]
            writer.writerow(['Exp3', 'Public Utility Ratio', best_result['public_utility_ratio'],
                           'PASS' if self.results['exp3']['overall_success'] else 'FAIL'])
            writer.writerow(['Exp3', 'Best Epsilon', best_epsilon, ''])
            writer.writerow(['Exp3', 'Sparsity', self.results['exp3']['sparsity'], ''])
        
        print(f"Saved CSV summary: {csv_filename}")


def main():
    """Run all experiments with visualization."""
    print("="*80)
    print("libortho: Enhanced Experiments with Visualization")
    print("="*80)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Timestamp: {TIMESTAMP}")
    print("="*80)
    
    collector = ExperimentCollector()
    
    try:
        # Run all experiments
        collector.collect_exp1_results()
        collector.collect_exp2_results()
        collector.collect_exp3_results()
        
        # Save all results
        collector.save_results()
        
        print("\n" + "="*80)
        print("All experiments completed successfully!")
        print("="*80)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"- Visualizations: exp*_results_{TIMESTAMP}.png")
        print(f"- JSON data: all_results_{TIMESTAMP}.json")
        print(f"- CSV summary: summary_{TIMESTAMP}.csv")
        
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

