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

Step 2 监控协议（关键，2026-08-14 修订：50 步硬判定替代直跑 150 步）：
  - 前 10 步密集盯：每 3-5 分钟 tail 一次 /root/autodl-tmp/arm_a_train.log，重点 grep -E "OOM|CUDA out|Error|Traceback|acc_of_this_batch|reward"
  - 显存基线：nvidia-smi 采样，单卡训练态应 ≤18G（offload+grad ckpt 已开）
  - 每 10 步摘一次指标存档（reward 均值/acc/tool_call_mean/pg_loss），写进本文档「回报节」
  - 停机硬门（同 08 文档口径）：step 10 时 acc 仍长期 0 或 tool_call 异常（全 0 或打满）→ 停机回报；step 50 时 pg_loss 无下行趋势 → 停机回报
  - 【Step-50 硬判定】训练到 50 步即停（~4-5h + 开销），用 50 步 ckpt 跑官方 191 题评测（anchor_eval.py，~45min，单卡）：
    * ckpt acc ≥ 78% + 3 点 → Arm A 阳性，写回报后可决定 resume 至 150（verl-tool resume_mode=auto 从 default_local_dir 恢复）
    * ckpt acc 在 78%±3 内 → 判定「纯 acc GRPO 在本基座无增益」，该结论本身即基线结果，转入 Arm B（同 50 步预算）
    * 依据：每步 128 次 rollout，acc 单步噪声 σ≈3.7%；50 步平均可排除 >5 点的真实增益，足以支撑上述二分判定
  - 150 步只在「50 步判定阳性」后用于最终数字，不再是默认路径

Step 3 回报：把结果写进 ai_handoff/11 的「远端回报」节并 push；如遇 OOM 记录显存峰值与失败 step 号，不要擅自改显存参数，回报告。

红线：不改 reward 代码（Arm A 是保真基线）；不释放实例；权重/日志不进 git。
```

## 3. 远端回报（2026-08-15 填写：12 次启动诊断，训练未进 step 1，已交接 12 号文档）

启动时间：09:13（第 1 次）→ 11:45（第 16 次），共 16 次启动（3 次实例重启），**step 0 管线已全通**（rollout 128 轨迹 ~15s → compute_score → 存档 → log_prob），**唯一剩余阻塞 = 训练期引擎常驻显存冲突**（详见 12 号文档）。

### 修复链（16 次启动逐个剥出的阻塞，全部已落地）

| # | 阻塞 | 修复 | 载体 |
|---|------|------|------|
| 1 | HF FA2 版本门槛（verl 默认强制 `attn_implementation=flash_attention_2`，env flash_attn stub 0.0.0 撞 ≥2.1.0 校验） | `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` | 脚本 |
| 2 | Hydra struct 模式拒绝向空 dict 加键 | 加 `+` 前缀 | 脚本 |
| 3 | vLLM 预算 0.5×24G=12G < 16.1G bf16 权重（tp=1）→ init 必败 | gmem 0.5→0.85（后因训练共存问题改回，见 #7） | 脚本 |
| 4 | 开源 `adatooler_v.py:203` 无保护读 `turns_stats`/`valid_action_stats`（守卫误置于访问之后；全仓无写入者 → 唯一来源是他们 Rl_data 的列） | 数据补零列（值只喂注释掉的 add_additional_penalties，活跃奖励零影响） | parquet + prepare_subset.py |
| 5 | 同上 `active_mask`（save-record 路径 `is_done=not active_mask`） | 数据补 True 列（只进存档） | parquet + prepare_subset.py |
| 6 | save-record `json.dump` 撞 numpy（`extra_info.images` 经 pandas 往返变 ndarray） | `clean_extra_info.py` 递归清洗 + 新增脚本 | parquet + 新脚本 |
| 7 | 存档文件多 worker 并发 read-modify-write 竞态（JSONDecodeError） | sitecustomize 加 json.load 退避重试（20 次，兜底 `[]`） | env sitecustomize |
| 8 | 引擎 21.4G 常驻 vs 训练前向 logits 4.9G（16k×vocab）→ OOM | 先试 STANDALONE sleep 补丁（**失败，见 12 号 §2.3**）→ 改 tp=2+gmem=0.5（**仍 OOM，见 12 号 §2.4 探针计划**） | 未决 |
| 9 | `_flash_use_top_left_mask` NameError（stub 令 is_flash_attn_2_available()=False → qwen2_vl.py 模块级 if 块整体跳过 → 全局未定义） | qwen2_vl.py 补 else 分支定义 3 个全局（=False，与 `_custom_flash_attention_forward` 默认一致） | vendored +6 行（**保留**） |
| 10 | `flash_attn_varlen_func` NameError（remove_padding=True 触发 verl FA2 融合 attention 补丁，mrope 路径必调真内核） | `use_remove_padding=True→False`（走 HF sdpa；纯性能开关，奖励零影响） | 脚本 |
| 11 | 脚本清理循环逐个 sleep 20s 串行化（16 孤儿 → 启动延迟 20+ 分钟） | 批量 SIGTERM + 单次等待 + 幸存者清扫 | 脚本 |
| 12 | 僵尸显存堵卡时白等 15 分钟启动 | 清理后 nvidia-smi 验证，>500MiB 直接退出报错 | 脚本 |

### 平台行为结论（重要，后续会话必读）

- **驱动僵尸分配**：SeetaCloud + 驱动 570.124.04 上，vLLM engine worker 进程被杀死（SIGTERM/SIGKILL 均可）后，驱动常把其显存登记为存活客户端（`nvidia-smi --query-compute-apps` 可见 PID，`ps` 无此进程、NSpid 全扫无映射）。惰性回收**不更新记账**（14GB 探测分配成功但 nvidia-smi 仍报占用），vLLM 的 mem_get_info 预检过不去。**唯一解法 = 实例重启**（GPU 级重置，容器 uptime 不变）。已重启 3 次。
- 反之，进程**自行退出**（如 init 失败自退）时内存干净释放。
- **vLLM 0.11 sleep 模式空转**：`CuMemAllocator` 是孤儿单例——正常路径（权重/KV cache 走 torch 分配器）无人向它注册，`sleep(level=1)` 迭代空表零释放。verl 的 STANDALONE 分支本来就 skip sleep/wake，补丁试过无效已撤销。
- 训练前向的 4.62G 单次分配 = 16k token × 151936 vocab × bf16 的 logits——引擎 16.1G 权重在 24G 卡上与训练共存需要分片或释放，无第三条路。

### 未决事项 → 12 号文档

- **tp=2+gmem=0.5 下引擎实测 21.28G/卡（与 tp=1+0.85 几乎相同）**——怀疑 verl_tool 的引擎架构未按 tp=2 分片或 cache 未按预算收缩。12 号文档给探针先行方案。
- 最终参数组合待探针结果定夺。
- ckpt：0 个（未进 step 1）；rl_ckpt/arm_a 为空。
