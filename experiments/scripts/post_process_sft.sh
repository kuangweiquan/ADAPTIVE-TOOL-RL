#!/usr/bin/env bash
# SFT 数据后处理:过滤 → 转换(采集 4 个任务全部完成后运行)
# 用法: bash experiments/scripts/post_process_sft.sh
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p log/sft_collect

TOOLREQ_GLOB="experiments/results/sft_collect/toolreq_*/trajectories_*.jsonl"
ANSWER_GLOB="experiments/results/sft_collect/answerfirst/trajectories_*.jsonl"

echo "[$(date +%H:%M:%S)] POST-PROCESS: filter"
python experiments/scripts/filter_sft_trajectories.py \
    --input "$TOOLREQ_GLOB" \
    --input "$ANSWER_GLOB" \
    --tool_required "$TOOLREQ_GLOB" \
    --output experiments/results/sft_collect/sft_candidates.jsonl \
    > log/sft_collect/filter.log 2>&1
echo "[$(date +%H:%M:%S)] filter exit=$? (see log/sft_collect/filter.log)"
tail -25 log/sft_collect/filter.log

echo "[$(date +%H:%M:%S)] POST-PROCESS: convert"
python experiments/scripts/trajectories_to_sft.py \
    --input experiments/results/sft_collect/sft_candidates.jsonl \
    --out_dir datasets/vstar_bench/sft \
    --images_dir datasets/vstar_bench/sft_images \
    > log/sft_collect/convert.log 2>&1
echo "[$(date +%H:%M:%S)] convert exit=$? (see log/sft_collect/convert.log)"
tail -15 log/sft_collect/convert.log

echo "[$(date +%H:%M:%S)] SFT DATA READY: datasets/vstar_bench/sft/{train,val}.jsonl + sft_images/"
