# 14 新会话提示词：Arm A 续战三（#28 双杠杆收官）

> 用户将在新会话窗口粘贴本文档「第 3 节提示词」执行。本文档同时是新会话的交接事实源。
> 前置：13 号文档已执行完毕（探针 gmem45 + #24-#27 四次启动诊断链，全部结论见 13 号 §4）。
> 状态：死亡点已从 compute_log_prob 推进到 update 阶段；三次照片级惜败（差 13/3/几百 MiB）；#28 双杠杆已全部落地、**未启动**（用户 2026-08-19 ~18:2x 关机）。

## 1. 当前状态快照（新会话必读）

- **GPU 已关闭（用户主动，2026-08-19 傍晚），新会话第一步 = 请用户开 GPU**。#27 是 OOM 自然死亡，关机前无残留（每卡 4 MiB，无僵尸）
- 远端：`ssh ATR` 免密直达；conda env `verl-tool`（`/root/autodl-tmp/envs/verl-tool`）；`/root/code` 已在 `b9a74fc`（与 GitHub main 一致）
- **训练脚本定稿**（`experiments/adatooler_stitch/train_arm_a_4x3090.sh`，b9a74fc）关键参数：tp=2、gmem=0.45、mns=64、**`+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False`**（新）、use_dynamic_bsz=False、log_prob_micro_batch=1、ppo_max_token_len=16384、use_remove_padding=False
- **vendored 三处分块补丁（editable 源，实例重启不丢，但启动前必须 grep 确认仍在）**：`/root/autodl-tmp/adatooler_v_review/verltool/verl/verl/utils/torch_functional.py` —— ① `entropy_from_logits` token 维分块（chunk 2048）；② `logprobs_from_logits_v2` bf16 分支**序列维行分块 chunk 512**（loop / `row_logits[i:i+512]` / `row_labels[i:i+512]` 三处必须一致！）。备份链：`/root/autodl-tmp/torch_functional.py.bak`（原始）、同目录 `.bak_2048`、`.bak_1024`。数值逐 token 一致、reward 零影响
- ckpt 为空（0 个），训练 0 步产出

## 2. 核心未决问题与 #28 双杠杆预期

诊断链全貌（详见 13 号 §4）：
`MLP 中间张量 4.62G → 逐行 softmax 4.64G → 分块 2048（#24 差 13 MiB）→ 分块 1024（#26 差 3 MiB）→ mns=64（#27 基座不变）`

**#27 的关键发现**：mns=64 让引擎装载降到 10.4G（省 0.75G ✓），但 update 基座仍 23.1G 与 #26 一字不差 → **引擎 rollout 后 workspace 膨胀 ~0.75G 吃掉收益**（sleep 模式 CuMem 池留存工作区，free_cache_engine 只还 KV 块）。update 基座另有 ±0.6-2G 的批次间方差（#24: 22.54G、#26/#27: 23.08-23.14G）。

#28 双杠杆（已落地、互不依赖）：
- **(a) enable_sleep_mode=False**（config 注入，路径已实锤）：引擎 rollout 后 workspace+KV 归还驱动 → update 基座预期 **20-21G**（若成立则余量 ~2G，一举收官）。async 模式从不调用 sleep()/wake_up()，零副作用（12 号已核）
- **(b) 行分块 512**：fp32 暂存需求 594→297 MiB；对 #27 最坏基座（free 391 MiB）单靠 (b) 也能过，但余量仅 ~100 MiB
- 目标峰值 ≤ 21.5G；< 23.57G 硬限

## 3. 提示词（粘贴到新会话窗口）

```text
你是本地 AI（Windows 开发机，无 GPU）。按 CLAUDE.md 执行，经 ssh ATR 直连远端 GPU 机。
继续执行 ai_handoff/14_arm_a_resume3.md 的 Arm A 续战三任务。13 号 §4 记录了 #24-#27 诊断链（必读），#28 双杠杆已落地未启动，当前一步之遥。

【前置】请用户开 GPU。开机后 Step 0 核对（GPU 命令一律经 ssh ATR）：
  1. ssh ATR 'nvidia-smi -L && nvidia-smi --query-gpu=memory.used --format=csv,noheader'
     —— 4 张 3090、显存干净（1-4 MiB/卡）
  2. ssh ATR 'source /etc/network_turbo && cd /root/code && git pull origin main'
     —— 应在 b9a74fc（脚本含 enable_sleep_mode=False 注入）
  3. 确认 vendored 补丁仍在（editable 源，重启实例不丢，重装环境才会丢）：
     ssh ATR 'grep -c "for i in range(0, row_logits.shape[0], 512):" /root/autodl-tmp/adatooler_v_review/verltool/verl/verl/utils/torch_functional.py'
     —— 应输出 1；同时 grep 确认 "row_logits[i:i + 512]" 与 "row_labels[i:i + 512]" 各为 1（三处切片必须一致，否则 gather 尺寸不匹配崩）
  4. 确认脚本注入行：ssh ATR 'grep -c "engine_kwargs.vllm.enable_sleep_mode=False" /root/code/experiments/adatooler_stitch/train_arm_a_4x3090.sh'
     —— 应输出 2（注入行 + 头注释各一）

【Step 1 启动第 28 次（参数已全部定稿，无需再改任何旋钮）】
  ssh ATR 'nohup bash /root/code/experiments/adatooler_stitch/train_arm_a_4x3090.sh > /root/autodl-tmp/arm_a_train.log 2>&1 < /dev/null &'
  记录启动 PID，武装监控：
  a) 日志增量监控（grep -aE "OOM|CUDA out of memory|Traceback|acc_of_this_batch|reward/|step:[0-9]|KeyError|No available memory|aborting launch|FATAL|Error"）
  b) 进程存活 + 显存采样：rollout 期每 3 分钟即可；**update 窗口（reward 落盘后 ~5-10 分钟）加密到 10-20 秒/次，并加 nvidia-smi --query-compute-apps=pid,used_memory 做 per-PID 分解**——这一步同时验证 sleep-off 假说（若 update 期引擎 PID 占用 ~9G = 假说成立；若仍 ~11G = 假说不成立，只剩 (b) 的 ~100 MiB 余量）
  step 0 需 ~10-15 分钟（micro-batch 1 下 128 次小前向），耐心盯完 old_log_prob → update → 首个 "step:0" 指标
  停机硬门（同 08/11 文档）：step 10 acc 长期 0 或 tool_call 异常 → 停机回报；step 50 pg_loss 无下行 → 停机回报
  每 10 步摘指标存档进本文档 §4 回报节。

【Step 2 Step-50 硬判定】50 步后 SIGTERM 温和停（勿 SIGKILL 引擎 worker——本平台杀死引擎进程必泄漏僵尸显存），用 50 步 ckpt 跑官方 191 题 anchor_eval（experiments/adatooler_stitch/anchor_eval.py）：
  ckpt acc ≥ 81% → Arm A 阳性，可 resume 至 150
  ckpt acc 78±3 → 「纯 acc GRPO 在本基座无增益」，转入 Arm B（同 50 步预算）

【红线】不改 reward 代码；改显存参数前征询用户；不释放实例；权重/日志不进 git；引擎/探针进程只自然退出不硬杀；vendored 补丁改动一律先备份再改（备份链已有 .bak/.bak_2048/.bak_1024）；脚本内 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 是禁区（vLLM cumem.py:150 硬断言，见 13 号 §4.3），勿再加回。

【Step 3 回报】结果写入 ai_handoff/14 §4 回报节 + 更新 knowledge-base，push。
```

## 4. 远端回报（待新会话填写）

（空：第 28 次结果 / per-PID 分解（sleep-off 假说验证）/ 训练指标 / 50 步判定结论）

## 5. 后备杠杆（仅当 #28 仍 OOM 时，改前一律征询用户）

按优先级：
1. **scoped-expandable 手术**：在 vendored `verl/workers/fsdp_workers.py` 顶部（torch import 之前）设 `os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"`——只让 FSDP 训练 worker 进程启用，引擎进程不经过该模块不受影响。风险：ray 可能在模块加载前已 import torch 导致 env 过晚（需在 worker 启动日志验证）；改前备份
2. **`data.max_prompt_length`/`max_response_length` 8192→4096**：padded 前向 [1,16k,151936]→[1,8k,151936]，logits 4.64→2.32G 及所有随 seq 缩放的瞬态减半。**reward 影响非零**（长轨迹截断），需用户单独确认
3. per-PID 分解结果指向训练侧基座（而非引擎）时：优先追 FSDP 侧固定成本（梯度检查点策略、offload 粒度），勿再动引擎旋钮
