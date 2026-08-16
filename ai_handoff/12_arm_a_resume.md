# 12 新会话提示词：Arm A 续战（探针先行定显存方案）

> 用户将在新会话窗口粘贴本文档「第 3 节提示词」执行。本文档同时是新会话的交接事实源。
> 前置：11 号文档已执行（16 次启动 + 12 项修复 + 3 次重启），step 0 管线全通，唯一阻塞 = 训练期引擎显存共存（见 11 号 §3 回报）。

## 1. 当前状态快照（新会话必读）

- **GPU 已关闭，新会话第一步 = 请用户开 GPU**（4 卡在关闭前被驱动僵尸分配堵满 21830 MiB/卡，开机即重置）
- 远端：`ssh ATR` 免密直达；conda env `verl-tool`（`/root/autodl-tmp/envs/verl-tool`）；`/root/code` 在 `65213cd`（含 tp=2+gmem=0.5 的 train_arm_a_4x3090.sh，**含启动前孤儿清理 + 显存验证护栏**）
- 数据就绪：`/root/autodl-tmp/datasets/adatooler_v_subset/`（train 2900/val 100，已补 turns_stats/valid_action_stats/active_mask 列 + extra_info numpy 清洗，备份 `.bak_turns`）
- env 适配已就位（**勿删**）：sitecustomize.py（json numpy 容错 + json.load 重试）；qwen2_vl.py else 分支补丁（+6 行，sdpa 路径必需）；flash_attn stub（上会话设计）
- ckpt 为空，训练 0 步产出；vendored verl 的 vllm_async_server.py 已恢复开源原样（`git diff -w` 验证过）

## 2. 核心未决问题：引擎显存为何不按配置收缩

现象：tp=2 + gmem=0.5 时引擎实测 **21.28G/卡**（4 进程 × 21784 MiB），与 tp=1 + gmem=0.85 时代（21.3-21.4G）几乎相同。两种可能：
a) verl_tool 引擎架构未按 tp=2 分片（每 GPU worker 实为完整模型 16.1G + cache ~4G）
b) vLLM 0.11 的 cache 预算逻辑与预期不同（0.5 应用位置/分片方式）

**探针先行**（5 轮矩阵 ~10 分钟，开 GPU 后第一步）：独立 vLLM 实例量真实显存，避免第 17 次盲启。探针脚本已入库（`experiments/adatooler_stitch/probe_vllm_mem.py` + `probe_vllm_mem_matrix.sh`，参数逐一镜像 verl_tool launch_server），判定分支见第 3 节 Step 2。

## 3. 提示词（粘贴到新会话窗口）

```text
你是本地 AI（Windows 开发机，无 GPU）。按 CLAUDE.md 执行，经 ssh ATR 直连远端 GPU 机。
继续执行 ai_handoff/12_arm_a_resume.md 的 Arm A 续战任务。11 号文档记录了此前 16 次启动的完整诊断（必读其 §3 回报）。

【前置】请用户开 GPU。开机后 Step 0 核对（GPU 命令一律经 ssh ATR）：
  1. ssh ATR 'nvidia-smi -L && nvidia-smi --query-gpu=memory.used --format=csv,noheader'
     —— 4 张 3090、显存干净（开机即重置，前次僵尸分配自动消失）
  2. ssh ATR 'source /etc/network_turbo && cd /root/code && git pull origin main'
     —— 同步最新代码（含新增探针 probe_vllm_mem.py / probe_vllm_mem_matrix.sh）

【Step 1 探针矩阵（~10 分钟，地面真相，已写进仓库）】远端分离运行：
  ssh ATR 'nohup bash /root/code/experiments/adatooler_stitch/probe_vllm_mem_matrix.sh > /root/autodl-tmp/probe_driver.log 2>&1 < /dev/null &'
  轮询 /root/autodl-tmp/probe_matrix.log 直至出现 "matrix done"。5 轮探针，每轮一个独立 vLLM 实例、跑完自然退出
  （此平台只有进程自退才干净释放显存——严禁 kill 探针；矩阵脚本自带每轮前后显存干净护栏，报 FATAL 即停并回报）：
    mirror_verl  tp=2 gmem=0.5（sleep/CAR/chunked/mns=256 全开）—— 与 verl_tool 引擎参数逐一镜像，本轮数 = 判定基准
    sleep_off    tp=2 gmem=0.5 --no-sleep          —— 隔离 vLLM 0.11 sleep 模式显存池
    mns64        tp=2 gmem=0.5 --max-num-seqs 64   —— KV cache 是否响应并发数
    old_cfg      tp=1 gmem=0.85                    —— 校准锚：应与 08-15 实测 21.3-21.4G 吻合
    gmem85       tp=2 gmem=0.85                    —— 预算旋钮是否活着（init OOM 也算有效数据，自退即干净）
  每轮记录：nvidia-smi 每卡占用 + compute-apps 每 PID 占用 + INFO 日志 "Model weights take X GiB"/KV cache 行 + [probe] SUMMARY 行。
  注意：sleep 模式开时 KV cache 可能走 CuMemAllocator 池，nvidia-smi 与 torch 计数会分歧——分歧本身即证据，以 nvidia-smi 为准。

【Step 2 按探针结果分支】判定以 mirror_verl 单卡占用为准：
  分支 A（≤13G/卡，vLLM 遵守配置）：直接 nohup 启动训练（脚本已含清理+护栏），进 Step 3。
  分支 B（仍 ~21G/卡）：用其余四轮定位，全部有现成落点（已核对远端 verl_tool 源码）：
    B1 sleep_off ≤13G 而 mirror_verl ~21G → 元凶 = verl_tool 硬编码 enable_sleep_mode=True（vllm_async_server.py launch_server 的 args 字典）。
       落点（纯配置注入，不改 vendored 代码）：训练脚本加 +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False
       —— engine_kwargs 在 launch_server 里合并于硬编码项之后（**engine_kwargs 末尾展开，False 只滤 None 不会被滤掉），且 RolloutConfig 已定义该字段；
       async 模式下 server.sleep()/wake_up() 无分支、engine.sleep() 从不被调用，关掉 sleep 模式无副作用（已核对源码）；
       若 Hydra 拒绝 + 前缀，去掉 + 再试（11 号 #2 已验证 + 前缀对空 dict 注入可行）
    B2 mns64 明显收缩 → cache 按 mns×模型长定尺寸而非预算 → engine_kwargs.vllm.max_num_seqs=64（必要时 max_num_batched_tokens=4096）；
       代价：并发 256→64，rollout 变慢，50 步冒烟可接受
    B3 五轮全 ≤13G 但此前 verl 引擎 21.28G → 元凶在 verl_tool 侧（ExternalZeroMQDistributedExecutor 经 ray zmq worker load_model / layered_summon=True / 引擎-训练重叠）。
       此时勿盲改：把探针 SUMMARY + arm_a_train.log 引擎 init 段（grep -a "parallel_config\|Engine\|GiB"）写进本文档 §4 回报，按回报方案走
    B4 gmem85 与 mirror_verl 几乎相同 → 0.11 预算公式对 tp=2 失效 → 用 gmem 扫描找经验目标（注意 tp=2 权重 8G，gmem<0.34 时预算装不下权重 init 必败，勿试更小值）
  关键约束不变：引擎 + 训练峰值必须 < 23.5G/卡（训练前向 logits 单次 4.9G = 16k×151936×bf16）。
  每次只改一个旋钮；改参前先在本文档 §4 回报节写依据。

【Step 3 训练启动与监控（分支 A 或 B 参数定稿后）】
  如需 B1/B2 注入：先改 train_arm_a_4x3090.sh → 本地 commit+push → 远端 git pull（先 source /etc/network_turbo）
  启动：ssh ATR 'nohup bash /root/code/experiments/adatooler_stitch/train_arm_a_4x3090.sh > /root/autodl-tmp/arm_a_train.log 2>&1 < /dev/null &'
  监控：grep -aE "OOM|CUDA out of memory|Error|Traceback|acc_of_this_batch|reward/|step:|KeyError|No available memory" /root/autodl-tmp/arm_a_train.log（注意 grep -a，日志含控制字符）
  前 10 步密集盯 + nvidia-smi 采样：引擎 + 训练峰值应 < 23.5G/卡（预期引擎 ≤13G + 训练 ≤9G）
  停机硬门（同 08/11 文档）：step 10 acc 长期 0 或 tool_call 异常 → 停机回报；step 50 pg_loss 无下行 → 停机回报
  每 10 步摘指标存档进本文档 §4 回报节。

【Step 4 Step-50 硬判定】50 步后 SIGTERM 温和停（**勿 SIGKILL 引擎 worker**——本平台杀死引擎进程必泄漏僵尸显存；温和停机让进程自退），用 50 步 ckpt 跑官方 191 题 anchor_eval（experiments/adatooler_stitch/anchor_eval.py）：
  ckpt acc ≥ 81% → Arm A 阳性，可 resume 至 150
  ckpt acc 78±3 → 「纯 acc GRPO 在本基座无增益」，转入 Arm B（同 50 步预算）

【红线】不改 reward 代码；改显存参数前写回报并征询；不释放实例；权重/日志不进 git；引擎/探针进程只自然退出不硬杀。

【Step 5 回报】结果写入 ai_handoff/12 §4 回报节 + 更新 knowledge-base，push。
```

## 4. 远端回报（待新会话填写）

（空：探针结果 / 分支选择 / 训练指标 / 50 步判定结论）
