#!/bin/bash
# Run baseline PyVision-RL training
cd "$(dirname "$0")/../.."

python -m pyvision.rl.train \
    --config experiments/configs/baseline.yaml \
    --output-dir experiments/results/baseline \
    "$@"
