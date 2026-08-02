#!/bin/bash
# Run all ablation experiments sequentially
cd "$(dirname "$0")/../.."

ABLATIONS=(
    "experiments/configs/baseline.yaml"
    "experiments/configs/atr_utility_only.yaml"
    "experiments/configs/atr_no_sequence.yaml"
    "experiments/configs/atr_full.yaml"
)

for cfg in "${ABLATIONS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    python -m pyvision.rl.train \
        --config "$cfg" \
        --reward atr \
        --output-dir "experiments/results/$name" \
        || echo "WARNING: $name failed, continuing..."
done

echo "All ablations complete."
