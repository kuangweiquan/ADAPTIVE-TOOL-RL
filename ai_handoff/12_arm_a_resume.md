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

## 4. 远端回报（2026-08-16 填写：探针矩阵完成 → 分支 A，训练已启动）

### 4.1 探针结果（5 轮矩阵，干净卡上每轮独立 vLLM 实例自退）

| run | 配置 | KV cache（每 worker） | 每卡实测 |
|---|---|---|---|
| mirror_verl | tp=2 gmem=0.5（=训练配置） | 1.96 GiB / 73,408 tok | **12.30 GiB** |
| sleep_off | 同上 --no-sleep | 1.96 GiB / 73,408 tok | 12.27 GiB |
| mns64 | 同上 mns=64 | 1.96 GiB / 73,408 tok | 11.33 GiB |
| old_cfg | tp=1 gmem=0.85 | 1.06 GiB / 19,904 tok | 19.09 GiB |
| gmem85 | tp=2 gmem=0.85 | 10.21 GiB / 382,336 tok | 20.56 GiB 后引擎核死亡（init OOM） |

判定表（§3 Step 2）：**分支 A 成立**——mirror_verl 12.30G ≤ 13G，vLLM 0.11 完全遵守 tp=2+gmem=0.5（8G 分片权重 + 1.96G KV + 开销 ≈ 12.3G）。B1/B2 排除：sleep 开关与 mns 均不影响 cache 池尺寸（1.96 GiB 三轮一致）。gmem85 init OOM 佐证预算公式活着（0.85 → cache 10.21G/卡 + 权重 8G + 初始化峰值 > 23.57G 可用 → 死亡）。
（工具注：首轮矩阵探针 SUMMARY 解析行有 nounits bug、5 轮均 exit=1，测量数据不受影响，已修并 push `88ae94b`。）

### 4.2 第 17 次启动 OOM 定位（21.28G 之谜的最终答案）

第 17 次（09:39，脚本原样）在 step 0 的 `compute_log_prob`（old_log_prob）复现 OOM，traceback 给出精确答案：

- 失败的 4.62 GiB 分配 = **MLP 中间张量**（`down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`），非此前推论的 logits 张量。131,072 tokens = 8 seq × 16k（`use_remove_padding=False` 全 pad）× 18944 中间维 × bf16 ≈ 4.62G，一次分配
- **每个 FSDP worker 进程稳态 21.28 GiB 在用**（4 进程一致，OOM 消息原文 "Process 16246 has 21.28 GiB memory in use"）= 引擎 12.3G（探针实测）+ FSDP 前向参数 ~4G + 梯度检查点激活 ~5G。11 号「4×21784 MiB」是**真实数字**（引擎+训练共存），§4.1 时代的僵尸污染解释撤回（僵尸问题仍真实存在，但不是 21.28G 的来源）
- 缺口：2.24G 空闲 vs 4.62G 需求 → 差 ~2.4G，仅一步之遥
- `ppo_max_token_len_per_gpu=16384` 在本版本 verl 的 log_prob 路径未生效分块（131K tokens 一次过前向）

### 4.3 修复与重启（第 18、19 次）

- **第 18 次（8→2 后）**：8→2 生效（稳态 21.28→20.63G，崩溃点从 MLP 中间张量移到逐行 softmax），但仍 OOM。定位：padded 路径 `logprobs_from_logits_v2`（torch_functional.py:129）逐行 `F.log_softmax(row_logits)`，行 = 填充后的 16k 序列 → [16384, 151936]×bf16 = **4.64 GiB 瞬态分配**，叠 ~16G 基座（引擎 12.3G + FSDP 前向 + 激活）必爆；该瞬态由填充长度决定，micro-batch 大小无法解决
- **修复 #2（用户已确认，`baf0763`）**：`use_dynamic_bsz` False→True（actor/rollout/ref 三处一并）+ `ppo_max_token_len_per_gpu` 16384→8192——打包前向无 padding、逐 token 行（softmax 瞬态趋零）、8k 块（logits 2.32G + entropy 临时 ~4.6G），峰值 ≈18.4G 余量 ~5G。逐 token log prob 与 per-seq attention 掩码不变，reward 零影响；log_prob 前向略慢（打包开销）
- **第 19 次（打包路径）失败**：verl `seqlen_balancing.py:295` 硬断言 `max_token_len >= max_seq_len`（8192 < 16384）——本版本动态打包不支持切分长序列，8k 块方案毙掉；块=16384 时 logits 4.64G + entropy 临时 9.28G 又超预算
- **修复 #2 定稿（用户已确认，`07566c5` + vendored 一行）**：回到 padded 路径（`use_dynamic_bsz=False`、`ppo_max_token_len_per_gpu` 还原 16384）+ `log_prob_micro_batch_size_per_gpu` 1（logits [1,16k,151936]=4.64G + 逐行 softmax 4.64G，峰值 ≈20.7G）+ vendored 一行 `calculate_entropy=True→False`（fsdp_workers.py:978；entropy_coeff=0 下只丢一条日志指标，reward 零影响；备份 `/root/autodl-tmp/fsdp_workers.py.bak_entropy`）
- **第 20 次失败（关熵方案毙掉）**：entropys=None 流入 `DataProto.from_dict` → AttributeError（ray_trainer:1176 每步无条件读 `batch["entropys"]` 算 actor/entropy 指标，无 entropy_coeff 守卫）→ 关熵需要改两处 vendored 且丢指标
- **修正（分块熵替代关熵，同记忆目标更优）**：还原 fsdp_workers.py:978，改 vendored `entropy_from_logits`（torch_functional.py:145）为 token 维分块（chunk 2048，逐 token 数值完全一致）：熵临时 9.28G→1.16G，峰值 ≈20.7G（log_probs 逐行 softmax 时）不变。备份 `/root/autodl-tmp/torch_functional.py.bak`
- 第 21 次启动 2026-08-16 ~11:05（远端 pid 68371），监控硬门同 §3 Step 3/4；后续指标逐 10 步续填本节
