# Experiment Results Template for Paper

This document provides a template for reporting experiment results in the paper.

## Experiment 1: Privacy Kill Switch Test

### Hypothesis
Turning off Ortho (α=0.0) eliminates privacy while preserving general capability.

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Privacy Error Ratio (α=0 vs α=1) | X.XXx | > 1.5x | ✅/❌ |
| General Error Ratio (α=0 vs α=1) | X.XXx | < 2.0x | ✅/❌ |
| Ortho Sparsity | X.XX% | 1-5% | ✅ |

### Key Findings

1. **Privacy Isolation**: When α=0.0, privacy error increased by **X.XXx**, confirming that privacy is successfully isolated in the Ortho component.

2. **General Capability Preservation**: General error increased by only **X.XXx**, demonstrating that the Base component preserves general capability.

3. **Sparsity**: Ortho component contains only **X.XX%** of parameters, achieving high compression while maintaining functionality.

### Figure Caption

**Figure X: Privacy Kill Switch Test Results.** (a) Training progress showing two phases: general data training and mixed data fine-tuning. (b) Error curves showing privacy and general errors as a function of α. Privacy error increases dramatically when α=0.0, while general error remains stable. (c) Error ratios demonstrating successful privacy isolation (ratio > 1.5x) and general capability preservation (ratio < 2.0x). (d) Summary table with key metrics.

---

## Experiment 2: Saving the Genius

### Hypothesis
Aggressive Base quantization (INT3/INT2) doesn't destroy genius reasoning stored in Ortho.

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Common Degradation (INT3) | X.XXx | N/A | - |
| Genius Survival (INT3) | X.XXx | N/A | - |
| Relative Preservation | X.XXx | < 0.5x | ✅/❌ |
| Ortho Sparsity | X.XX% | 1-5% | ✅ |

### Key Findings

1. **Genius Preservation**: With INT3 quantization, genius error degraded by **X.XXx**, while common error degraded by **X.XXx**.

2. **Relative Preservation**: The relative preservation ratio of **X.XXx** confirms that genius reasoning is primarily stored in the Ortho component, orthogonal to Base.

3. **Extreme Quantization**: Even with INT2 quantization, genius reasoning remains relatively preserved (relative preservation: **X.XXx**).

### Figure Caption

**Figure Y: Saving the Genius Results.** (a) Training progress for base (common patterns) and full (common + genius) models. (b) Error comparison across quantization levels (Before, INT4, INT3, INT2) showing that genius error degrades less than common error. (c) Degradation ratios demonstrating that genius survives aggressive Base quantization. (d) Summary table with relative preservation metrics.

---

## Experiment 3: Dual Differential Privacy

### Hypothesis
Applying DP only to Ortho preserves better utility than global DP while maintaining the same privacy budget.

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Public Utility Ratio (ε=X.XX) | X.XXx | > 1.1x | ✅/❌ |
| Global DP Degradation | X.XXx | N/A | - |
| Dual DP Degradation | X.XXx | N/A | - |
| Ortho Sparsity | X.XX% | 1-5% | ✅ |

### Key Findings

1. **Utility Preservation**: At ε=X.XX, Dual-DP preserves **X.XXx** better public utility than Global-DP.

2. **Degradation Comparison**: 
   - Global DP: Public error degraded by **X.XXx**
   - Dual DP: Public error degraded by **X.XXx**

3. **Privacy Protection**: Both methods provide (ε, δ)-DP, but Dual-DP avoids unnecessary noise on public knowledge (Base).

### Figure Caption

**Figure Z: Dual Differential Privacy Results.** (a) Training progress for base (public) and full (public + private) models. (b) Public error comparison across epsilon values, showing that Dual-DP consistently preserves better utility than Global-DP. (c) Public utility preservation ratio demonstrating Dual-DP's advantage (ratio > 1.1x). (d) Summary table with best epsilon and degradation metrics.

---

## Summary Table for Paper

| Experiment | Key Metric | Result | Status |
|------------|------------|--------|--------|
| Exp 1: Privacy Kill Switch | Privacy Ratio | X.XXx | ✅ |
| Exp 1: Privacy Kill Switch | General Ratio | X.XXx | ✅ |
| Exp 2: Saving the Genius | Relative Preservation | X.XXx | ✅ |
| Exp 3: Dual DP | Public Utility Ratio | X.XXx | ✅ |

## Statistical Analysis (Optional)

If running multiple trials:

| Metric | Mean | Std | 95% CI | p-value |
|--------|------|-----|--------|---------|
| Privacy Ratio | X.XX | X.XX | [X.XX, X.XX] | < 0.001 |
| General Ratio | X.XX | X.XX | [X.XX, X.XX] | < 0.001 |
| Relative Preservation | X.XX | X.XX | [X.XX, X.XX] | < 0.001 |
| Public Utility Ratio | X.XX | X.XX | [X.XX, X.XX] | < 0.001 |

## LaTeX Code for Tables

```latex
\begin{table}[h]
\centering
\caption{Experiment Results Summary}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Experiment & Metric & Result & Status \\
\midrule
Exp 1: Privacy Kill Switch & Privacy Ratio & X.XXx & \checkmark \\
Exp 1: Privacy Kill Switch & General Ratio & X.XXx & \checkmark \\
Exp 2: Saving the Genius & Relative Preservation & X.XXx & \checkmark \\
Exp 3: Dual DP & Public Utility Ratio & X.XXx & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

## Usage Instructions

1. Run the visualization script:
   ```bash
   python experiments/run_experiments_with_visualization.py
   ```

2. Extract values from generated files:
   - `experiments/results/summary_*.csv` - Quick summary
   - `experiments/results/all_results_*.json` - Complete data

3. Fill in this template with actual values

4. Include figures in paper:
   - `exp1_results_*.png` → Figure X
   - `exp2_results_*.png` → Figure Y
   - `exp3_results_*.png` → Figure Z

5. Use LaTeX code for tables

---

**Note**: Replace all "X.XX" placeholders with actual values from your experiment results.

