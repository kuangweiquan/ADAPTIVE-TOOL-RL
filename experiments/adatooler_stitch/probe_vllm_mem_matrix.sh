#!/bin/bash
# Arm A vLLM memory probe matrix driver (run on remote GPU box, verl-tool env).
# 5 one-shot probes; each probe process exits naturally (never killed: on this
# platform killed vLLM workers leak driver-side zombie allocations). Verifies
# GPUs are clean before/after every run and aborts if not (500 MiB threshold,
# same guard as train_arm_a_4x3090.sh).
# Log: /root/autodl-tmp/probe_matrix.log  (outside the repo, not in git)
set -u
export PATH=/root/autodl-tmp/envs/verl-tool/bin:$PATH
export VLLM_USE_V1=1 HF_HUB_DISABLE_XET=1
export VLLM_LOGGING_LEVEL=INFO VLLM_LOGGER_LEVEL=INFO
PROBE=/root/code/experiments/adatooler_stitch/probe_vllm_mem.py
LOG=/root/autodl-tmp/probe_matrix.log

check_clean() {  # $1 = context label; 0 = clean, 1 = dirty
  local dirty=0
  for line in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader); do
    idx=$(echo "$line" | cut -d, -f1)
    used=$(echo "$line" | cut -d, -f2 | tr -dc '0-9')
    if [ "$used" -gt 500 ]; then
      echo "FATAL: GPU $idx dirty (${used} MiB) [$1] -> abort matrix (instance reboot needed)" >> $LOG
      dirty=1
    fi
  done
  return $dirty
}

run() {  # tag, tp, then probe args...
  local tag=$1; local tp=$2; shift 2
  echo "=== run $tag (tp=$tp): $* ===" >> $LOG
  check_clean "$tag-before" || exit 1
  local devs=0
  [ "$tp" -ge 2 ] && devs="0,1"
  CUDA_VISIBLE_DEVICES=$devs python $PROBE --tp "$tp" "$@" --tag "$tag" >> $LOG 2>&1
  local rc=$?
  echo "--- run $tag exit=$rc ---" >> $LOG
  check_clean "$tag-after" || { echo "FATAL: $tag left dirty GPUs" >> $LOG; exit 1; }
}

: > $LOG
run mirror_verl 2 --gmem 0.5                          # verl mirror: sleep+car+chunked+mns256 (the decisive number)
run sleep_off  2 --gmem 0.5 --no-sleep               # isolates vLLM 0.11 sleep-mode pool
run mns64      2 --gmem 0.5 --max-num-seqs 64        # does KV cache respond to max_num_seqs?
run old_cfg    1 --gmem 0.85                         # calibration anchor vs 08-15 measured 21.3-21.4G
run gmem85     2 --gmem 0.85                         # is the budget knob alive? (init OOM is valid data, self-exits clean)
echo "=== matrix done ===" >> $LOG
SUMMARY=/root/autodl-tmp/probe_matrix_summary.txt
echo "--- summary (grep -a, log has control chars) ---" > $SUMMARY
grep -aE "^(===|---)|\[probe\] (SUMMARY|FATAL)|Model weights take|KV|CPU/CUDA memory" $LOG >> $SUMMARY
cat $SUMMARY
