#!/usr/bin/env bash
# SFT 数据采集守护脚本(自愈):4 个任务各自循环运行,
# 进程死亡/异常退出时自动补跑(--skip_existing 断点续跑),直到每个任务 191 条完成。
# 用法: setsid bash experiments/scripts/collect_sft_daemon.sh > log/sft_collect/daemon.log 2>&1 &
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export ATR_MODEL="Qwen/Qwen3.5-397B-A17B"
mkdir -p log/sft_collect

TARGET=191

run_task() {
    local name="$1"; shift
    local dir="experiments/results/sft_collect/$name"
    while :; do
        local done=0
        for f in "$dir"/trajectories_*.jsonl; do
            [ -f "$f" ] && done=$((done + $(wc -l < "$f")))
        done
        if [ "$done" -ge "$TARGET" ]; then
            echo "[$(date +%H:%M:%S)] $name COMPLETE ($done/$TARGET)"
            return 0
        fi
        echo "[$(date +%H:%M:%S)] $name run($done/$TARGET): $*"
        python experiments/scripts/run_atr_offline.py \
            --vstar_path datasets/vstar_bench --output_dir "$dir" \
            --skip_existing "$@" >> "log/sft_collect/$name.log" 2>&1
        local rc=$?
        echo "[$(date +%H:%M:%S)] $name exited rc=$rc, restarting in 15s"
        sleep 15
    done
}

run_task toolreq_t07 --tool_required --temperature 0.7 &
P1=$!
run_task toolreq_t09 --tool_required --temperature 0.9 &
P2=$!
run_task toolreq_t11 --tool_required --temperature 1.1 &
P3=$!
run_task answerfirst --temperature 0.0 &
P4=$!

for p in "$P1" "$P2" "$P3" "$P4"; do
    wait "$p"
done
echo "[$(date +%H:%M:%S)] DAEMON ALL DONE"

# 全部完成后自动后处理(过滤 + 转换)
bash experiments/scripts/post_process_sft.sh
echo "[$(date +%H:%M:%S)] DAEMON EXIT"
