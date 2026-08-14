# 11 新会话提示词：Arm A 启动与监控（4×3090）

> 用户将在新会话窗口粘贴本文档「第 2 节提示词」执行。本文档同时是新会话的交接事实源。
> 前置状态：Phase 0 完成（见 `ai_handoff/10_adatooler_stitch_plan.md` §16），Gate 0 PASS（锚点 149/191=78.0%）。

## 1. 当前状态快照（新会话必读）

- 远端：`ssh ATR` 免密直达（SeetaCloud，conda env `verl-tool` 在 `/root/autodl-tmp/envs/verl-tool`，工作数据全在 `/root/autodl-tmp/`）
- 资产：SFT 模型 `/root/autodl-tmp/models/AdaTooler-V-SFT-model`（16.6GB 已验证）；子集 `/root/autodl-tmp/datasets/adatooler_v_subset/`（2900 train + 100 val parquet）；官方 VStar `/root/autodl-tmp/datasets/vstar_official/`；官方仓库 `/root/autodl-tmp/adatooler_v_review/`
- 已就绪脚本：`experiments/adatooler_stitch/train_arm_a_4x3090.sh`（已 push，远端 /root/code 需 `git pull`）
- Arm A = 开源代码原样行为（纯 acc GRPO，penalty 保持注释状态）= 受控基线；**不用** stitch 包的任何组件
- 7B gated 已决定跳过（上界锚点用论文报告值 89.8%）

## 2. 提示词（粘贴到新会话窗口）

```text
你是本地 AI（Windows 开发机，无 GPU）。按 CLAUDE.md 执行，经 ssh ATR 直连远端 GPU 机。

【任务】4×3090 已开机。执行 ai_handoff/11_arm_a_launch.md 的 Arm A 训练启动与监控。

Step 0 诊断（读 ai_handoff/11 第 1 节状态快照，逐项核对远端）：
  1. ssh ATR 'nvidia-smi -L' —— 确认 4 张 3090 可见（训练脚本硬编码 CUDA_VISIBLE_DEVICES=0,1,2,3）
  2. ssh ATR 'ls /root/autodl-tmp/envs/verl-tool/bin/python' —— env 在位
  3. ssh ATR 'cd /root/code && git pull' —— 同步最新代码（含 train_arm_a_4x3090.sh）

Step 1 启动 Arm A（在远端分离运行）：
  source /etc/network_turbo  # 外网操作前
  cp experiments/adatooler_stitch/train_arm_a_4x3090.sh 到远端（scp 或等 git pull 后用仓库内的）
  ssh ATR 'nohup bash /root/code/experiments/adatooler_stitch/train_arm_a_4x3090.sh > /root/autodl-tmp/arm_a_train.log 2>&1 < /dev/null &'
  注意：脚本内置 tool server 启动与 kill；日志含 NCCL INFO 噪音属正常

Step 2 监控协议（关键）：
  - 前 10 步密集盯：每 3-5 分钟 tail 一次 /root/autodl-tmp/arm_a_train.log，重点 grep -E "OOM|CUDA out|Error|Traceback|acc_of_this_batch|reward"
  - 显存基线：nvidia-smi 采样，单卡训练态应 ≤18G（offload+grad ckpt 已开）
  - 稳定后每 20 步摘一次指标存档（reward 均值/acc/tool_call_mean/pg_loss），写进本文档「回报节」
  - 150 步完成或任一步 OOM → 进入回报
  - 停机条件（3 门同 08 文档口径）：step 50 时 pg_loss 无下行趋势 / tool_call_mean 趋势异常 / acc 长期 0

Step 3 回报：把结果写进 ai_handoff/11 的「远端回报」节并 push；如遇 OOM 记录显存峰值与失败 step 号，不要擅自改显存参数，回报告。

红线：不改 reward 代码（Arm A 是保真基线）；不释放实例；权重/日志不进 git。
```

## 3. 远端回报（待新会话填写）

（空，待 Arm A 启动后填写：启动时间 / Step 0 诊断结果 / 前 10 步指标 / OOM 情况 / 最终 150 步指标表）
