# 08 远端回报：GRPO RL 训练至 step50 存档停止 + 修复链完整验证

> 本文件为**远端 GPU AI** 写回报（2026-08-13 晚），本地 AI 收阅。训练按用户指令于 **step 50 存档后停止**（实际完成到 step 52 后终止，详见 §5）。

---

## 1. 任务背景与指令

- 延续 07_rl_debug_report：step35 OOM 二次崩溃的修复（commit 7b02be3）后重启训练，从 step 30 恢复
- 用户指令：「每5步保存一个checkpoint」「保持监控」「完成到50步保存checkpoint停止训练」「给本地AI写详细回报，数据日志一定要完整详实」
- 训练实况：**2026-08-13 18:40 前后重启 → 21:26 终止，全程 0 次 OOM / 0 次 Traceback**

## 2. 崩溃史与修复链（本次训练前）

| 崩溃 | 时间 | 根因 | 修复 | commit |
|------|------|------|------|--------|
| step14 OOM | 08-13 早 | gpu_memory_utilization 0.6 下 vLLM 实占 17.2G 超预算，训练段余量 <8.6G 刚需 | 0.6→0.5；save_freq 25→10 | e4af22e |
| step28 OOM | 08-13 晚 | 训练段物理余量贴顶（post-bwd free 0.07-0.55G），13312 长样本 backward 的 674MiB 梯度 cast 踩线 | +enable_gradient_checkpointing（36 层激活重算，free 回 3-5G）；ppo_max_token_len_per_gpu 2048→1536；save_freq 10→5 | 48bc86b, 9b253c2 |
| step35 OOM | 08-13 19:39 | 双根因：① FSDP mixed_precision 默认 reduce_dtype=fp32 → unshard/reduce 瞬态翻倍（post-fwd 7.5G→post-bwd 0.11G）；② optimizer step 后池碎片不归还 → reserved 跨步膨胀 31.3→36.7G | 修复 a：`fsdp_config.mixed_precision.reduce_dtype=bf16`（瞬态减半；全项显式 bf16 曾致 FSDP 初始化 OOM，回退只留 reduce 单项）；修复 b：dp_actor 双 empty_cache（optimizer step 前 + update 循环后每步还池） | 7b02be3（+ec3ca2a 回退修正） |

**修复后验证结论：step14/28/35 三个历史崩溃点全部越过，22 步无崩溃。**

## 3. 本次训练完整指标（step 31-52）

> 数据源：`logs/qwen3vl_8b_sftv2_grpo_4gpu.log`（本次重启后全程，tee 覆盖写）

| step | acc | reward | timing_s | tok/s | resp_len | prompt_len | max_alloc_G |
|------|-----|--------|----------|-------|----------|------------|-------------|
| 31 | 0.156 | 0.369 | 239.9 | 85.1 | 42.5 | 1911 | 26.89 |
| 32 | 0.188 | 0.391 | 242.5 | 84.0 | 41.0 | 1911 | 27.18 |
| 33 | 0.031 | 0.302 | 245.2 | 84.7 | 47.0 | 1913 | 27.52 |
| 34 | 0.125 | 0.330 | 244.7 | 84.9 | 44.7 | 1910 | 27.87 |
| 35 | 0.125 | **0.485** | 257.2 | 81.4 | 44.0 | 1914 | 28.17 |
| 36 | 0.031 | 0.298 | 244.5 | 87.4 | 47.5 | 1906 | 28.53 |
| 37 | 0.219 | 0.437 | 244.7 | 81.5 | 41.0 | 1910 | 28.88 |
| 38 | 0.094 | 0.328 | 258.7 | 78.6 | 45.0 | 1911 | 29.19 |
| 39 | 0.125 | 0.350 | 312.7 | 65.6 | 43.8 | 1912 | 29.53 |
| 40 | 0.188 | 0.369 | 303.7 | 64.8 | 39.8 | 1909 | 29.85 |
| 41 | 0.156 | 0.392 | 231.4 | 89.3 | 43.0 | 1904 | 30.15 |
| 42 | 0.156 | 0.391 | 231.3 | 87.9 | 43.7 | 1915 | 30.53 |
| 43 | 0.094 | 0.359 | 248.5 | 84.6 | 46.0 | 1911 | 30.90 |
| 44 | 0.188 | 0.363 | 233.5 | 89.2 | 42.6 | 1907 | 31.21 |
| 45 | 0.281 | **0.487** | 237.7 | 81.4 | 39.1 | 1906 | 31.50 |
| 46 | 0.094 | 0.424 | 239.3 | 89.5 | 46.6 | 1915 | 31.55 |
| 47 | 0.188 | 0.419 | 228.7 | 90.1 | 42.5 | 1904 | 31.55 |
| 48 | 0.188 | 0.438 | 228.6 | 87.7 | 42.2 | 1909 | 31.55 |
| 49 | 0.156 | 0.353 | 234.7 | 87.9 | 43.5 | 1906 | 31.55 |
| 50 | 0.094 | 0.267 | 257.9 | 78.9 | 45.7 | 1905 | 31.55 |
| 51 | 0.125 | 0.410 | 242.1 | 87.1 | 44.6 | 1911 | 31.80 |
| 52 | 0.125 | 0.377 | 230.6 | 86.7 | 39.5 | 1918 | 32.08 |

**汇总**（22 步）：
- reward 区间 0.267-0.487，均值 **0.377**，无退化趋势
- acc 区间 0.031-0.281，均值 0.143；峰谷交替为 32 样本高方差（每步 batch=8 样本 × n=4 rollout）
- timing 均值 ~245s/步（4.1 分钟），throughput 65-90 tok/s（长样本 step 如 39/40 略慢）
- 步速预期：总步数 = 171 条 ÷ 8 × 30 epochs ≈ **630 步**；已跑 52 步 ≈ 8.3%

## 4. 显存账（修复链的量化验证）

**MB diag 全程统计（506 条 post-bwd 记录）**：

| 指标 | 值 |
|------|-----|
| post-bwd free min | **0.03G**（53 条 < 674MiB 阈值，占 10.5%，全部通过） |
| post-bwd free p10 | 0.65G |
| post-bwd free median | **1.74G** |
| post-bwd free max | 12.58G |
| post-bwd free < 1G | 90 条（17.8%） |
| post-bwd free ≥ 1G | 416 条（82.2%） |
| post-fwd free（forward 前） | 3.85-5.85G 常态 |
| max_memory_reserved_gb | 恒 34.268G（虚拟池上限，无跨步膨胀——修复 b 生效） |

**关键结论**：
1. **修复 a（reduce_dtype bf16）生效**：backward 瞬态从崩前 7.4G 降至 ~5.4G，cast 674MiB 在 backward 内部 free 尚足时完成，post-bwd 余量（0.03-0.65G）是 cast **完成之后**的状态，不再直接构成崩溃条件
2. **修复 b（双 empty_cache）生效**：reserved 稳定在 34.27G 封顶，无 31.3→36.7G 式跨步膨胀；post-bwd 后紧跟 empty_cache 兜底 optimizer step 的 4G 状态加载
3. **残余风险（已如实记录）**：长样本 step 的 post-bwd free 仍可低至 0.03G（53 条 < 674MiB），若 cast 时机偏移仍概率性踩线。已连续 22 步验证不崩，但非绝对保证。降险选项：cast 674MiB 分块化（类似 entropy 分块先例，改 torch FSDP `_runtime_utils.py:1026`），需本地端评估后决定是否实施

## 5. 停止过程与 ckpt 状态

- 21:20 前后 step 50 完成并存档（`global_step_50`，32G，save_freq=5 第 10 个存档点）
- 21:25:42 向主进程（PID 161827，`python3 -m verl.trainer.main_ppo`）发 SIGTERM；21:26:14 确认全部退出，**4 卡显存归零，无残留进程**（Ray/vLLM/WorkerDict 均干净）
- 停止信号处理前训练又完成了 step 51、52（未存档，非 5 的倍数），latest 存档仍为 **50**

**ckpt 目录现状**（`/root/autodl-tmp/rl_ckpt/vstar_atr/qwen3vl_8b_sftv2_grpo_4gpu/`）：

| 目录 | 大小 | 说明 |
|------|------|------|
| global_step_30 | 32G | 重启前遗留（旧配置时代） |
| global_step_35 | 4.0K | **空壳**（内容已删，目录残留） |
| global_step_40 | 4.0K | **空壳**（同上） |
| global_step_45 | 32G | 完整 |
| global_step_50 | 32G | 完整 |
| latest_checkpointed_iteration.txt | — | **50** |

**两个已知现象**：
1. **淘汰列表内存态**：`max_actor_ckpt_to_keep=2` 的淘汰列表（`fsdp_checkpoint_manager.py` 的 `previous_saved_paths`）是进程内存态，重启后从空开始 → 重启前遗留的 step30 不在淘汰范围（已与用户确认：存储充足，暂不删除，稳态 3 个 32G = 96G/350G 无压力）
2. **空壳目录**：被淘汰的 35/40 留下 4.0K 空目录（rmtree 删内容但目录残留，非致命）。恢复走 `latest_checkpointed_iteration.txt`，45/50 是安全档

## 6. 本轮代码/配置状态（均已 commit + push）

- **commit 7b02be3**（修复主体）：`pyvision-rl/verl_agents/verl/workers/actor/dp_actor.py` — `_optimizer_step` 前 + `offload_fsdp_optimizer` 后各加 `torch.cuda.empty_cache()`
- **commit ec3ca2a**（回退修正）：`run_vstar_full.sh` 只留 `mixed_precision.reduce_dtype=bf16`，param/buffer 保持隐式默认（全项显式曾致 FSDP 初始化 OOM，vLLM cache 失败连锁）
- 当前 `run_vstar_full.sh` 关键配置：`gpu_memory_utilization=0.5`、`ppo_max_token_len_per_gpu=1536`、`enable_gradient_checkpointing=True`、`save_freq=5`、`max_actor_ckpt_to_keep=2`、`total_epochs=30`、Adafactor（bf16 状态）、agent max_turns=8
- 显存账注释 (1)-(7) 已同步至脚本头部

## 7. 已知问题与下一步建议（供本地端决策）

1. **C=0 结构性缺陷**（待本地调查）：所有 step 的 `C=0.000`（成本惩罚恒 0），reward 实际只由 acc+U 驱动。本地端此前已知，建议排查 reward 计算路径
2. **post-bwd 贴顶残余风险**：53 条 < 674MiB 记录虽全部通过，属概率性踩线。候选降险：torch FSDP `_cast_grad_to_param_dtype`（`_runtime_utils.py:1026`）的 674MiB 一次性 cast 分块化（先例：verl entropy 分块）
3. **多模态不切分样本**（7ff1ebb 已知）：`ppo_max_token_len_per_gpu` 对含图长样本无效，长样本全量 13312 进 backward——本次贴顶的根源，后续可考虑 vLLM 侧序列截断策略
4. **ckpt 空壳与遗留**：重启后手动清理一次空壳 + 旧档可保持 2 个稳态（本次按用户指示未删）
5. 训练仅完成 8.3%（52/630），若重启续跑：`resume_mode=auto` 从 latest=50 自动恢复，`bash run_vstar_full.sh` 即可

## 8. 回报要求

- 本档已 commit + push（`[remote]` 前缀），本地 AI pull 后收阅
- 日志原文保留于远端 `logs/qwen3vl_8b_sftv2_grpo_4gpu.log`（不进 git），需要原始 MB diag 数据可另行搬运
- 下一步（继续训练 or 先修 C=0/cast 分块）由用户 + 本地端决定
