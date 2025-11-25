"""
Quick script to generate paper-ready figures from experiment results.

Usage:
    python experiments/generate_paper_figures.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments_with_visualization import ExperimentCollector

def main():
    """Generate all paper figures."""
    print("="*80)
    print("Generating Paper Figures")
    print("="*80)
    
    collector = ExperimentCollector()
    
    # Run experiments and generate figures
    print("\nRunning Experiment 1...")
    collector.collect_exp1_results()
    
    print("\nRunning Experiment 2...")
    collector.collect_exp2_results()
    
    print("\nRunning Experiment 3...")
    collector.collect_exp3_results()
    
    # Save results
    collector.save_results()
    
    print("\n" + "="*80)
    print("All figures generated successfully!")
    print("="*80)
    print("\nFigures saved to: experiments/results/")
    print("\nFor paper, use:")
    print("  - exp1_results_*.png -> Figure X: Privacy Kill Switch")
    print("  - exp2_results_*.png -> Figure Y: Saving the Genius")
    print("  - exp3_results_*.png -> Figure Z: Dual Differential Privacy")
    print("\nData files:")
    print("  - all_results_*.json -> Complete numerical data")
    print("  - summary_*.csv -> Summary table for paper")
    
    return 0

if __name__ == "__main__":
    exit(main())

