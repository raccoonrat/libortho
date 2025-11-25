#!/bin/bash
# Run complete real model experiments for paper

set -e

MODEL_PATH="${1:-/home/mpcblock/models/Llama-3.2-3B}"
EXPERIMENT="${2:-all}"
OUTPUT_DIR="${3:-experiments/results}"

echo "=========================================="
echo "Running Complete Real Model Experiments"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Experiment: $EXPERIMENT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ] && [[ ! "$MODEL_PATH" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Usage: $0 [model_path] [experiment] [output_dir]"
    echo "  model_path: Local path or HuggingFace model ID"
    echo "  experiment: 1, 2, 3, 4, or all (default: all)"
    echo "  output_dir: Output directory (default: experiments/results)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run experiments
python experiments/complete_real_model_experiments.py \
    --model "$MODEL_PATH" \
    --experiment "$EXPERIMENT" \
    --device cuda \
    --no-quantization \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

