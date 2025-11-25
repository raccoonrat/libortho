#!/bin/bash
# Quick start script for GTX 4050 real model experiments

set -e

echo "============================================================"
echo "GTX 4050 Real Model Experiments"
echo "============================================================"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import torch; import transformers; import bitsandbytes" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install torch transformers bitsandbytes datasets accelerate
}

# Run experiments
echo ""
echo "============================================================"
echo "Running GTX 4050 Optimized Experiments"
echo "============================================================"
echo ""

python3 experiments/real_model_experiments_gtx4050.py \
    --model meta-llama/Llama-2-1B-hf \
    --experiment all \
    --device cuda \
    --quantization-bits 4

echo ""
echo "============================================================"
echo "Experiments completed!"
echo "============================================================"
echo ""
echo "Results saved in: experiments/results/gtx4050_results_*.json"
echo ""
echo "For more options, see: experiments/GTX4050_REAL_MODEL_EXPERIMENTS.md"

