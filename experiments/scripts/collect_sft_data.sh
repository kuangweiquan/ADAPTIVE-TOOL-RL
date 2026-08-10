#!/usr/bin/env bash
# SFT 冷启动数据采集:397B 生成工具轨迹 + 直接作答轨迹(计划 §1)
#   强制工具模式 191 样本 × 3 温度(0.7 / 0.9 / 1.1)
#   先答后验模式 191 样本 × 1 种子(0.0)
# 用法: nohup bash experiments/scripts/collect_sft_data.sh > log/sft_collect/collector.log 2>&1 &
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export ATR_MODEL="Qwen/Qwen3.5-397B-A17B"
mkdir -p log/sft_collect

run_one() {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] START $name ($*)"
    python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
        --output_dir "experiments/results/sft_collect/$name" "$@" \
        > "log/sft_collect/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE $name exit=$?"
}

# 4 个任务并行(每个约 1h)
run_one toolreq_t07 --tool_required --temperature 0.7 &
P1=$!
run_one toolreq_t09 --tool_required --temperature 0.9 &
P2=$!
run_one toolreq_t11 --tool_required --temperature 1.1 &
P3=$!
run_one answerfirst --temperature 0.0 &
P4=$!

FAIL=0
for p in "$P1" "$P2" "$P3" "$P4"; do
    wait "$p" || FAIL=1
done
echo "[$(date +%H:%M:%S)] ALL DONE fail=$FAIL"

# ========== 采集完成后自动处理(过滤 + 转换) ==========
# 4 个采集任务全部结束后执行;失败任务不影响已成功部分
echo "[$(date +%H:%M:%S)] POST-PROCESS: filter + convert"

TOOLREQ_GLOB="experiments/results/sft_collect/toolreq_*/trajectories_*.jsonl"

python experiments/scripts/filter_sft_trajectories.py \
    --input "$TOOLREQ_GLOB" \
    --input "experiments/results/sft_collect/answerfirst/trajectories_*.jsonl" \
    --tool_required "$TOOLREQ_GLOB" \
    --output experiments/results/sft_collect/sft_candidates.jsonl \
    > log/sft_collect/filter.log 2>&1
echo "[$(date +%H:%M:%S)] filter exit=$? (see log/sft_collect/filter.log)"

python experiments/scripts/trajectories_to_sft.py \
    --input experiments/results/sft_collect/sft_candidates.jsonl \
    --out_dir datasets/vstar_bench/sft \
    --images_dir datasets/vstar_bench/sft_images \
    > log/sft_collect/convert.log 2>&1
echo "[$(date +%H:%M:%S)] convert exit=$? (see log/sft_collect/convert.log)"
echo "[$(date +%H:%M:%S)] SFT DATA READY: datasets/vstar_bench/sft/{train,val}.jsonl + sft_images/"
