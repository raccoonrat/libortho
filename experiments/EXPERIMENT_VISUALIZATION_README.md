# Experiment Visualization and Data Collection Guide

## Overview

This guide explains how to run the enhanced experiments with detailed data collection, visualization, and statistical analysis.

## Quick Start

```bash
# Install dependencies
pip install matplotlib seaborn numpy torch

# Run all experiments with visualization
python experiments/run_experiments_with_visualization.py
```

## Output Files

All results are saved to `experiments/results/` directory with timestamp:

1. **Visualizations** (PNG, 300 DPI):
   - `exp1_results_YYYYMMDD_HHMMSS.png` - Privacy Kill Switch test
   - `exp2_results_YYYYMMDD_HHMMSS.png` - Saving the Genius
   - `exp3_results_YYYYMMDD_HHMMSS.png` - Dual Differential Privacy

2. **Data Files**:
   - `all_results_YYYYMMDD_HHMMSS.json` - Complete numerical results
   - `summary_YYYYMMDD_HHMMSS.csv` - Summary table for paper

## Experiment Details

### Experiment 1: Privacy Kill Switch Test

**Hypothesis**: Turning off Ortho eliminates privacy while preserving general capability.

**Metrics Collected**:
- Training losses (Phase 1: General, Phase 2: Mixed)
- Privacy error vs Alpha (0.0 to 1.0)
- General error vs Alpha
- Privacy error ratio (α=0 vs α=1)
- General error ratio (α=0 vs α=1)
- Ortho sparsity

**Success Criteria**:
- Privacy error ratio > 1.5x (privacy forgotten)
- General error ratio < 2.0x (general capability preserved)

**Visualizations**:
1. Training progress (both phases)
2. Error vs Alpha curve (privacy and general)
3. Error ratio bar chart
4. Summary table

### Experiment 2: Saving the Genius

**Hypothesis**: Aggressive Base quantization doesn't destroy Ortho genius.

**Metrics Collected**:
- Training losses (Base and Full)
- Common error by quantization level (Before, INT4, INT3, INT2)
- Genius error by quantization level
- Common degradation ratio (INT3)
- Genius survival ratio (INT3)
- Relative preservation (Genius/Common)

**Success Criteria**:
- Relative preservation < 0.5 (Genius degrades less than half of Common)

**Visualizations**:
1. Training progress
2. Error by quantization level (bar chart)
3. Degradation ratios
4. Summary table

### Experiment 3: Dual Differential Privacy

**Hypothesis**: DP only on Ortho preserves better utility than global DP.

**Metrics Collected**:
- Training losses (Base and Full)
- Public error by epsilon (0.5, 1.0, 2.0)
- Private error by epsilon
- Public utility ratio (Global/Dual)
- Degradation ratios

**Success Criteria**:
- Public utility ratio > 1.1x (Dual-DP at least 10% better)

**Visualizations**:
1. Training progress
2. Public error vs Epsilon (Global vs Dual)
3. Utility ratio by epsilon
4. Summary table

## Using Results in Paper

### For Tables

Use the CSV summary file:
```python
import pandas as pd
df = pd.read_csv('experiments/results/summary_YYYYMMDD_HHMMSS.csv')
print(df.to_latex(index=False))
```

### For Figures

Include the PNG files directly in LaTeX:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{experiments/results/exp1_results_YYYYMMDD_HHMMSS.png}
    \caption{Experiment 1: Privacy Kill Switch Test}
    \label{fig:exp1}
\end{figure}
```

### For Numerical Results

Load JSON and extract specific metrics:
```python
import json
with open('experiments/results/all_results_YYYYMMDD_HHMMSS.json') as f:
    results = json.load(f)

# Example: Get privacy ratio
privacy_ratio = results['exp1']['privacy_ratio']
print(f"Privacy error ratio: {privacy_ratio:.2f}x")
```

## Customization

### Change Random Seed

Edit the script to use different seeds:
```python
torch.manual_seed(42)  # Change to different seed
np.random.seed(42)
```

### Adjust Epsilon Values (Exp3)

Modify in `_run_exp3_enhanced()`:
```python
epsilon_values = [0.5, 1.0, 2.0]  # Add more values
```

### Change Quantization Levels (Exp2)

Modify in `_run_exp2_enhanced()`:
```python
quantization_levels = ['INT4', 'INT3', 'INT2']  # Add INT1, etc.
```

### Adjust Visualization Style

Modify at the top of the script:
```python
sns.set_style("whitegrid")  # Try "darkgrid", "white", etc.
plt.rcParams['figure.figsize'] = (12, 8)  # Adjust size
plt.rcParams['font.size'] = 11  # Adjust font size
```

## Troubleshooting

### Import Errors

Make sure you're in the project root:
```bash
cd /path/to/libortho
python experiments/run_experiments_with_visualization.py
```

### Matplotlib Backend Issues

If you get display errors, the script uses 'Agg' backend (non-interactive). If you need interactive plots:
```python
# Comment out this line:
# matplotlib.use('Agg')
```

### Memory Issues

If you run out of memory, reduce batch sizes in the experiment scripts:
```python
BATCH_GEN = 1000  # Reduce to 500
BATCH_PRIV = 10   # Reduce to 5
```

## Statistics

The script collects detailed numerical data for statistical analysis:

- **Mean and Standard Deviation**: Run multiple times with different seeds
- **Confidence Intervals**: Use bootstrap or t-test
- **Significance Testing**: Compare ratios using paired t-test

Example statistical analysis:
```python
import numpy as np
from scipy import stats

# Load multiple runs
ratios = [run1['privacy_ratio'], run2['privacy_ratio'], ...]

# Calculate statistics
mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios)
ci = stats.t.interval(0.95, len(ratios)-1, loc=mean_ratio, scale=stats.sem(ratios))

print(f"Privacy ratio: {mean_ratio:.2f} ± {std_ratio:.2f}")
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

## Integration with Paper

### Abstract

Use key numbers from summary CSV:
- Privacy ratio: X.XXx
- Relative preservation: X.XXx
- Public utility ratio: X.XXx

### Results Section

1. **Experiment 1**: 
   - "Privacy error increased by X.XXx when α=0.0"
   - "General error increased by only X.XXx"
   - Include Figure X (visualization)

2. **Experiment 2**:
   - "Genius error degraded by X.XXx vs Common error by X.XXx"
   - "Relative preservation: X.XXx"
   - Include Figure Y

3. **Experiment 3**:
   - "Dual-DP preserves X.XXx better utility than Global-DP"
   - "At ε=X.XX, public error degradation: X.XXx vs X.XXx"
   - Include Figure Z

### Discussion

Reference specific metrics:
- "Our experiments show privacy ratio of X.XXx (Table X)"
- "The relative preservation of X.XXx confirms our hypothesis"
- "Dual-DP achieves X.XXx utility ratio (Figure X)"

## Next Steps

1. **Run Multiple Trials**: Execute script multiple times with different seeds
2. **Statistical Analysis**: Calculate means, std, confidence intervals
3. **Generate Tables**: Convert CSV to LaTeX tables
4. **Create Paper Figures**: Use PNG files directly or recreate in LaTeX
5. **Write Results Section**: Use collected metrics in paper

---

**Questions?** Check the main README or open an issue.

