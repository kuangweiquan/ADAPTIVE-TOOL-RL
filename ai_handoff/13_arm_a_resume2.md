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

## 4. 远端回报（2026-08-19 填写：探针 gmem45 + #24-#27 四次启动全链 + #28 预备）

### 4.1 探针 gmem45（Step 1，一次性通过）

| 配置 | 引擎占用/卡 | KV cache |
|---|---|---|
| gmem=0.5（基线） | 12.30 GiB | 73,408 tok |
| **gmem=0.45** | **11.16 GiB** | **29,280 tok** |

省 1.14G，KV 29,280 ≥ 2 万达标。探针自退干净（exit 0）。用户确认后脚本 gmem 0.5→0.45（`a886812`）。

### 4.2 第 24 次（gmem 0.45 单旋钮）：死亡点首次推进到 update 阶段

- **old_log_prob（compute_log_prob）首次通过**——#17-#23 的死点被 gmem 0.45 攻克
- 死于下一阶段 `actor_rollout_update_actor`（update）：同 1.16 GiB 分配（2048 块 fp32 暂存），in-use 22.54G、free 1005 MiB，差 ~160 MiB（free+reserved-unalloc 1201 MiB vs 需求 1188）
- update 阶段基座比 log_prob 阶段高（梯度+optimizer+激活），吃掉了 gmem 省下的 1.14G
- 显存自净（4 MiB/卡），无僵尸

### 4.3 第 25 次（+expandable_segments）：被 vLLM 0.11 硬断言击杀

- 用户确认双旋钮（chunk 1024 + expandable_segments，`cae7041`）；vendored 行分块 2048→1024 打上（备份 `.bak_2048`，三处切片 loop/logits/labels 对齐）
- **expandable_segments 与 vLLM 0.11 内存池硬冲突**：引擎核 init 即崩 `AssertionError: Expandable segments are not compatible with memory pool`（vLLM 源码 `cumem.py:150` 设计性硬断言，torch 已知问题 #147851；LLM() 无 envs 覆盖参数）。引擎进程继承脚本 env，无法从脚本层隔离
- **回退**（`1cac6a4`，脚本留「勿再加回」警示注释）

### 4.4 第 26 次（chunk 1024 单旋钮）：差 3 MiB 照片级惜败

- 分块生效：失败分配 1.16G→**594 MiB**；但 in-use 23.14G（比 #24 的 22.54G 高 0.6G——**基座方差**把收益吃光）
- free 395 + reserved-unalloc 202 = 597 MiB vs 需求 594 → **差 3 MiB**

### 4.5 第 27 次（mns 256→64，`bc0add8`）：锁定真元凶

- 引擎装载实测 10.4G（省 0.75G ✓），但 update 基座仍 **23.08-23.14G，与 #26 一字不差**
- 结论：**引擎 rollout 后 workspace 膨胀 ~0.75G 吃掉了 mns 收益**——sleep 模式下 CuMem 池留存 rollout 工作区，`free_cache_engine` 只还 KV 块不还 workspace，训练侧也用不到池内存
- 失败：594 MiB 分配、free 391-451 MiB

### 4.6 #28 预备（已全部落地，未启动——用户关机）

- **杠杆 (a) `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False`**（`b9a74fc`）：注入路径已实锤——vllm_async_server.py:229 硬编码 True、:237 `**engine_kwargs` 在其后展开、:202 只滤 None；RolloutConfig.engine_kwargs 字段存在（rollout.py:144）；async 模式从不调用 sleep()/wake_up()（12 号已核），零副作用。预期：rollout 后 workspace+KV 归还驱动，update 基座 ~20-21G
- **杠杆 (b) vendored 行分块 1024→512**：fp32 暂存 594→297 MiB（备份 `.bak_1024`，三处切片已对齐并核验）
- 双杠杆独立（引擎侧 vs 训练侧瞬态），任一单独生效都可能过，双开几乎必过
- **远端已就位**：脚本 b9a74fc、chunk 512 补丁、备份链 .bak/.bak_2048/.bak_1024；GitHub main = b9a74fc
- 下次会话按 14 号文档执行：开 GPU → Step 0 核对 → 第 28 次启动

### 4.7 关键教训（供 14 号/后续参考）

1. **每 3 分钟采样会漏掉峰值**：本会话 OOM 均靠日志监控即时捕获；下一步的 #28 建议在 update 窗口加 10-20s 间隔的 per-PID 采样（nvidia-smi --query-compute-apps）分解「引擎 vs 训练侧」占用，验证 sleep-off 假说
2. **照片级缺口史**：#24 差 13 MiB（free+unalloc 1201 vs 1188）、#26 差 3 MiB（597 vs 594）——分块收益每次都被基座方差（0.6-2G）吃掉；基座方差与固定 shape 并存，指向分配器状态/碎片，scoped-expandable（训练 worker 导入序注入 env）仍是后备手术
3. **备用杠杆未动**：chunk 512 已是最小有效分块（再往下 256 收益趋零）；`data.max_prompt/response_length` 8192→4096 可砍 logits 张量 4.64→2.32G 但会截断长轨迹、reward 影响非零，需用户单独确认才可试
4. **scp 直传脚本安全**：本会话多次 scp 直传（push 故障兜底），远端 0 CR 字节验证通过；`tr -cd "\r" < file | wc -c` 是可靠检测法（引号转义会骗过 grep -cP）
