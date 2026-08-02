#!/bin/bash
# Run ATR training (with ablation selection)
cd "$(dirname "$0")/../.."

CONFIG=${1:-experiments/configs/atr_full.yaml}
NAME=$(basename "$CONFIG" .yaml)

python -m pyvision.rl.train \
    --config "$CONFIG" \
    --reward atr \
    --output-dir "experiments/results/$NAME" \
    "${@:2}"
