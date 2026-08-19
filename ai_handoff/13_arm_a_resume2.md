# 13 新会话提示词：Arm A 续战二（收官 0.2G 缺口）

> 用户将在新会话窗口粘贴本文档「第 3 节提示词」执行。本文档同时是新会话的交接事实源。
> 前置：12 号文档已执行（探针矩阵 + 启动 #17-#23 的 7 次 OOM 诊断链），OOM 点已从 4.62G 削到 1.16G，**只差 ~0.1-0.2G 未进 step 1**。全部结论见 12 号 §4。

## 1. 当前状态快照（新会话必读）

- **GPU 已关闭（用户主动，2026-08-16 ~12:10），新会话第一步 = 请用户开 GPU**。关机前已温和停训（SIGTERM，显存排空 4 MiB/卡，无僵尸）；开机即重置任何残留
- 远端：`ssh ATR` 免密直达；conda env `verl-tool`（`/root/autodl-tmp/envs/verl-tool`）；`/root/code` 在 `07566c5`（训练脚本定稿版），GitHub main 在 `d20fdde`（多出的是 12 号 §4 文档更新），新会话先 `git pull`
- **训练脚本定稿**（`experiments/adatooler_stitch/train_arm_a_4x3090.sh`）：tp=2 + gmem=0.5（探针实测引擎 12.3G/卡，vLLM 遵守配置）；`use_dynamic_bsz=False`（动态打包被 verl `seqlen_balancing` 断言 `max_token_len>=max_seq_len` 毙掉）；`log_prob_micro_batch_size_per_gpu=1`；`ppo_max_token_len_per_gpu=16384`；use_remove_padding=False（FA2 stub 所限）
- **vendored 两处分块补丁（editable 源，实例重启不丢，但启动前必须 grep 确认仍在）**：`/root/autodl-tmp/adatooler_v_review/verltool/verl/verl/utils/torch_functional.py` —— ① `entropy_from_logits` token 维分块（chunk 2048，备份 `torch_functional.py.bak`）；② `logprobs_from_logits_v2` bf16 分支行循环序列维分块（2048）。两处数值逐 token 一致、reward 零影响
- ckpt 为空（0 个），训练 0 步产出；rl_ckpt/arm_a 为空

## 2. 核心未决问题：最后 ~0.2G 缺口 + 基座方差

第 23 次（行分块 2048 后）OOM：失败分配 = 1.16 GiB（2048 块的 fp32 暂存），in-use 22.54G、free ~1G。诊断链全貌（详见 12 号 §4）：
`MLP 中间张量 4.62G（8×16k）→ 逐行 softmax 4.64G → fp32 暂存 2.32G（块 4096）→ fp32 暂存 1.16G（块 2048）→ 差 0.2G`

注意基座有方差：log_prob 前向基座（除 logits/瞬态外）在 #21 实测 ~15.1G、#23 实测 ~17.3G（批次内序列组成差异/分配器碎片，样本次数不足未定因）。**留 ~2G 余量才稳**，目标峰值 ≤ 21.5G。

## 3. 提示词（粘贴到新会话窗口）

```text
你是本地 AI（Windows 开发机，无 GPU）。按 CLAUDE.md 执行，经 ssh ATR 直连远端 GPU 机。
继续执行 ai_handoff/13_arm_a_resume2.md 的 Arm A 续战二任务。12 号文档记录了探针矩阵与 #17-#23 七次启动诊断链（必读其 §4），当前只差 ~0.1-0.2G 未进 step 1。

【前置】请用户开 GPU。开机后 Step 0 核对（GPU 命令一律经 ssh ATR）：
  1. ssh ATR 'nvidia-smi -L && nvidia-smi --query-gpu=memory.used --format=csv,noheader'
     —— 4 张 3090、显存干净（开机即重置）
  2. ssh ATR 'source /etc/network_turbo && cd /root/code && git pull origin main'
     —— 同步到 d20fdde（脚本与 07566c5 相同，新到的是文档更新）
  3. 确认 vendored 两处分块补丁仍在（editable 源，实例重启不丢，重装环境才会丢）：
     ssh ATR 'grep -c "chunk the row over the seq dim" /root/autodl-tmp/adatooler_v_review/verltool/verl/verl/utils/torch_functional.py && grep -c "chunk_size: int = 2048" /root/autodl-tmp/adatooler_v_review/verltool/verl/verl/utils/torch_functional.py'
     两者都应输出 1（若 0 则从 torch_functional.py.bak 起重打两处，重打内容见 12 号 §4.3 与 13 号 §2 描述）

【Step 1 探针量化 gmem=0.45（~3 分钟，先量再动）】
  远端运行：CUDA_VISIBLE_DEVICES=0,1 VLLM_LOGGING_LEVEL=INFO /root/autodl-tmp/envs/verl-tool/bin/python /root/code/experiments/adatooler_stitch/probe_vllm_mem.py --tp 2 --gmem 0.45 --tag gmem45
  记录 SUMMARY 行（预期引擎 ~11.1G，省 ~1.2G）与 "GPU KV cache size: N tokens"（N 太小时 rollout 会变慢，N≥2 万可接受）。
  同时记录 gmem=0.5 基线已测（12.30G / 73,408 tok）无需重跑。

【Step 2 收紧显存（经用户确认后一次改一个）】
  首选：脚本 gpu_memory_utilization 0.5→0.45（纯引擎 cache 预算，reward 零影响；代价 = rollout KV cache 变小可能略慢）。
  可选叠加：脚本 export 区加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（缓解分配器碎片；OOM 信息反复建议）。
  改完 commit+push → 远端 git pull → 下一步启动。
  每次只改一个旋钮；改参前征询用户。

【Step 3 启动与监控（第 24 次）】
  ssh ATR 'nohup bash /root/code/experiments/adatooler_stitch/train_arm_a_4x3090.sh > /root/autodl-tmp/arm_a_train.log 2>&1 < /dev/null &'
  监控：grep -aE "OOM|CUDA out of memory|Error|Traceback|acc_of_this_batch|reward/|step:|KeyError|No available memory" /root/autodl-tmp/arm_a_train.log（注意 grep -a，日志含控制字符）
  前 10 步密集盯 + nvidia-smi 采样：稳态引擎 ~11-12G + 训练前向，峰值目标 ≤ 21.5G/卡（< 23.57G 硬限）
  step 0 需 ~10-15 分钟（micro-batch 1 下 128 次小前向），耐心盯完 old_log_prob → update → 首个 "step:0" 指标
  停机硬门（同 08/11 文档）：step 10 acc 长期 0 或 tool_call 异常 → 停机回报；step 50 pg_loss 无下行 → 停机回报
  每 10 步摘指标存档进本文档 §4 回报节。

【Step 4 Step-50 硬判定】50 步后 SIGTERM 温和停（勿 SIGKILL 引擎 worker——本平台杀死引擎进程必泄漏僵尸显存；温和停机让进程自退），用 50 步 ckpt 跑官方 191 题 anchor_eval（experiments/adatooler_stitch/anchor_eval.py）：
  ckpt acc ≥ 81% → Arm A 阳性，可 resume 至 150
  ckpt acc 78±3 → 「纯 acc GRPO 在本基座无增益」，转入 Arm B（同 50 步预算）

【红线】不改 reward 代码；改显存参数前征询用户；不释放实例；权重/日志不进 git；引擎/探针进程只自然退出不硬杀；vendored 补丁改动一律先备份再改。

【Step 5 回报】结果写入 ai_handoff/13 §4 回报节 + 更新 knowledge-base，push。
```

## 4. 远端回报（待新会话填写）

（空：探针 gmem45 结果 / 参数改动 / 训练指标 / 50 步判定结论）
