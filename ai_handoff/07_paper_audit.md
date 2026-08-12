# 07 — 论文视角审计与路线决策（2026-08-12，本地端）

> 背景：04/05/06 全部交付物已在手（SFT v2 = 先答后验 62% / IoU 34%；verl 环境就绪；4×3090 冒烟四项通过）。本地端对**整个项目**做了 CCF-A skill 严格评审 + 最接近工作检索（2025–2026），结论是：**工程极佳、研究定位需要转向**。
> 本交接文档 = 审计结论 + 需要远端**先回报的一项证据** + 全量 RL 的决策树。**跑全量 GRPO 之前，先完成 Step 0 并回报。**

## 一句话结论

`pivot-with-rescue-route`（2.9/5）：「GRPO + zoom 工具 + 奖励塑形」在 2025–2026 已被至少 5 篇同行工作正面做过，ATR 的 U/C/S 三项也分别被已发表工作覆盖 → **不能按"新奖励机制"写论文**；但工程资产完整、V*Bench 重铸任务 + 独有的行为诊断数据仍有价值，两条可写路线见下。

## 最接近工作（每篇都必须引用，审稿人一搜就中）

| 工作 | 出处 | 与 ATR 的重叠 |
|---|---|---|
| Reinforcing VLMs to Use Tools (2506.14821) | arXiv | **同一配方**：GRPO + zoom 工具 + 精度奖励 |
| VISTA-Gym / VISTA-R1 | CVPR 2026 | SFT→GRPO + 工具环境完整管线，8B 超同规模 SOTA +9.5~18.7% |
| TACO | 2025 | 无外部 judge 的逐工具调用 credit —— **覆盖 U 项思想** |
| AdaTooler-V / AT-GRPO | ACL 2026 | 样本级 Tool Benefit Score，该用放大/滥用惩罚 —— **覆盖 U/C 项**；V* 89.8% 超 GPT-4o |
| EMTIR-GRPO | Findings ACL 2026 | GRPO + 成本感知系数 —— **覆盖 C 项** |
| OpenThinkIMG / VTool-R1 | 2025–2026 | GRPO 学自适应 zoom 策略（仅精度奖励即可） |
| ParaVT | arXiv 2605.20342 | **skip-tool reward shortcut**：调/不调工具的 advantage 会塌缩为 0 —— 直接对应本项目「响应 43 tok」风险 |
| H*Bench | CVPR 2026 | V*Bench 启发的 360° 搜索 agent，SFT+GRPO 成功率 ×3 —— 可作迁移证据 |

## Step 0 诊断（远端必做：先回报，再决定跑不跑全量）

**动机**：冒烟实测「响应 43 tok」。若含义是"模型平均只生成 43 token 就结束"，则 rollout 中模型**几乎不调工具**（一次 zoom 调用+观察就 100+ token），ATR 的 U/S 项无梯度信号，奖励退化为 acc−γC，全量 RL 大概率白跑（ParaVT 已证明这是标准失效模式）。

**命令（GPU 机器上纯读日志，10 分钟，不占算力）**：

```bash
cd /root/code
# 1. ATR 奖励分布（应有 acc=... U=... C=... S=... → R=... 若干行）
grep -E "\[ATR\]" logs/vstar_smoke.log | tail -10
# 2. 工具调用指标（verl 指标，agent/tool_call_mean 等）
grep -E "tool_call|Tool calling" logs/vstar_smoke.log | tail -10
# 3. "43 tok" 的原行（grep CHUNK 的最后几行，确认它打印的是总序列还是最后一块）
grep -E "\[CHUNK\]" logs/vstar_smoke.log | tail -5
# 4. 顺便确认有 [Zoomed into / OCR result 的样本数
grep -c -E "\[Zoomed into|\[OCR result:" logs/vstar_smoke.log
```

**回报格式**：把上面 4 条命令的原始输出贴进本文件的「远端回报」节（或直接在对话里给出），并回答：**「43 tok」是 8 条 rollouts 的平均响应长度吗？调用了工具的有几条？**

## 决策树（收到 Step 0 回报后）

- **工具调用率 ≥ 30%**（43 tok 是误读/个别样本）→ 走 **Track A**（下方），跑全量但**加 50 步止损**。
- **工具调用率 ≈ 0**（43 tok 属实，rollout 直接作答）→ **不要跑 640 步全量**。二选一：
  - 换 RL 起点为 **v2-强制工具版**权重（它会调工具只是不准，U 项才有信号），再冒烟一次确认工具调用率 > 0 → Track A；
  - 或直接放弃 RL 正结果，走 **Track B**（分析型论文，不需要新训练）。

## Track A：RL 正结果路径（2–4 周，B 会概率 ~50–60%）

论文定位降格为：**「Agentic Vision RL 中奖励设计的实证比较研究」**——不再宣称新奖励机制，而是用 ATR 与同行奖励（TACO 式 credit、纯 acc）做对比 + λ/γ/η 消融。

- 起点：SFT v2-强制工具版（若 Step 0 显示先答后验版本不调工具）
- 全量配置沿用 06（450–520 步），**加 50 步止损点**：前 50 步看 `pg_loss` 是否下降、`agent/tool_call_mean` 是否 > 0 且上升、`[ATR]` 的 R 分布是否有区分度；三条都不满足 → 停，回报
- 评测补齐（本地可帮忙做的部分：CPU 离线跑）：
  1. **V*Bench 官方口径**（MCQA 准确率，191 题 0-shot）—— 审稿人必看，这是与所有 closest work 对齐的数字
  2. **no-tool 基线**（SFT v2 不带工具直接答）
  3. λ/γ/η 消融至少 3 组（需要 4 卡，约 +1 周）
  4. 迁移证据可选：H*Bench 子集

## Track B：分析型兜底路径（1–2 周，B 会概率 ~40–50%）

不跑新训练。论文=「**视觉 agent 工具学习的失败模式与奖励信号分析**」，用独有数据：

- v1 22% → v2 34% 的 IoU 演变 + 工具级 5.7% vs 样本级 34% 的拆解（本地已有 [compute_gt_iou.py](../experiments/scripts/compute_gt_iou.py) 产物）
- 死循环/扫描式平移诊断（04 文档量化过：后续 zoom 映射回原图全是微小框）
- skip-tool reward shortcut 实证（冒烟 43 tok 就是证据）
- ATR 奖励设计分析 + 离线重放（本地可全 CPU 跑）

## 引用补全清单（本地端将更新 REFERENCES.md，远端不用动）

在 `knowledge-base/REFERENCES.md` 补 5 篇：2506.14821、TACO、VISTA-R1、AdaTooler-V/AT-GRPO、EMTIR-GRPO（可选 OpenThinkIMG、ParaVT）。每一篇在正文的 related work 里必须有实质讨论，不能只挂引用。

## 红线（沿用 06）

- 跑全量前必须先回报 Step 0；不擅自降级数据/换起点
- checkpoint/日志/images.zip 不进 git；回报写本文件回报节 + commit `[remote] ...`

---

## 远端回报 — Step 0 诊断（2026-08-12）

> 日志：`logs/vstar_smoke.log`（Aug 11 11:55，**唯一成功的冒烟**；`vstar_smoke_1gpu.log` 是 vLLM 显存不足失败，`qwen3vl_8b_sftv2_grpo_4gpu.log` 是 4 卡全量 RayTaskError 失败，均无指标行）。冒烟配置：SFT v2 权重 + `max_turns=2, n=2, train_batch_size=4`（= 8 条 rollout，全部 direct_attributes）。

### 4 条命令原始输出

```bash
# 1. [ATR] 奖励分布 → 0 行（空）
# 原因：patch_reward.py:212 打印受 step_cnt < num_examine 门控、base_reward.py:171 受 verbose 门控，
# 冒烟未开。但奖励确实算了：critic/rewards/mean_reward_of_this_batch:0.387 (max 0.6 / min 0.3)

# 2. 工具调用指标（关键）：
agent/tool_call_mean:1.000  tool_call_max:1.000  tool_call_min:1.000  tool_call_zero_ratio:0.000
data_source_tool/direct_attributes/tool_call_mean:1.000  tool_call_zero_ratio:0.000

# 3. "43 tok" 出处（[CHUNK] 是显存日志，与 token 无关）：
# [CHUNK] local_numel=352396384 pieces=[(0,0,0),(0,0,0),(1,82462912,269933472),(1,0,352396384)] free=6.92G
# 真实来源 = verl 指标行：
response_length/mean:43.750  response_length/max:45.000  response_length/min:43.000
data_source_response_length/direct_attributes/response_length_mean:43.750

# 4. [Zoomed into / [OCR result: 计数 → 0（无信息量）
# 这两个串是 ToolResult.observation 文本（image_tools.py:96/188），进 prompt 不打印日志，
# grep 不到 ≠ 没发生 zoom。日志无 response 文本/工具名分布，无法从日志区分调的是 zoom 还是 bbox_2d。
```

### 问答

- **「43 tok」是 8 条 rollouts 的平均响应长度吗？** → **是**。`response_length/mean=43.750`（min 43 / max 45，8 条），且与 `perf/total_num_tokens:36319` 精确吻合：36319 ≈ 8×(prompt 3618.5 + obs 877.6 + resp 43.75)=36320。确认为**每条 rollout 完整响应的总 token 数**（不是最后一块、不是单轮均值）。`[CHUNK]` 行是 FSDP 显存分块日志，Step 0 文档里"grep CHUNK 确认 43 tok"的前提不成立。
- **调用了工具的有几条？** → **8/8（100%）**。`tool_call_mean=1.000`、`tool_call_zero_ratio=0.000`，且全部 `end_reason=EXCEED_MAX_TURNS`（max_turns=2 用满）。

### 解读（对决策树的影响）

1. **不落入「工具调用率≈0 → 直接作答」失效分支**：模型 100% 调工具，ParaVT skip-tool shortcut 不成立。冒烟起点是 **SFT v2 先答后验版**（非强制工具版），在 RL 环境下仍 100% 调工具。
2. **但 acc=0（8/8 全错）**：`critic/acc/acc_of_this_batch:0.000`，reward 0.3~0.6 由 U/C/S 项撑起。43 tok ≈ 单轮一次紧凑工具调用 XML 的长度（agent 检测到调用即截断本轮），每条 rollout 总共只生成 ~43 tok 即到 max_turns 上限——**2 轮内只够一次调用，没有继续推理/作答的空间**。工具会用但没用对（可能是 max_turns=2 过短，zoom→bbox 链至少要 2 轮）。
3. 按决策树 → **Track A 分支（工具调用率 ≥30% 命中）**，但带两个文档未预见的新信号：acc=0 + 每轮仅 43 tok 一次调用。建议保留 50 步止损，另确认 max_turns 是否应放宽。
4. 遗留未知：调用的具体工具名（zoom vs bbox_2d）日志无粒度；[ATR] U/C/S 分解未打印。若要该数据，下次冒烟开 `num_examine` / `verbose` 即可。

**待本地决策：是否按 Track A 跑全量（加 50 步止损），以及 max_turns 是否调整。未获确认前不启动全量。**

---

## 本地决策（2026-08-12，基于 Step 0 回报 + 07_rl_debug_report）

### 决策 1：同意 Track A 跑全量，但附加两个前置动作

**批准跑 `run_vstar_full.sh` 全量（max_turns=8 保持不动，无需放宽）**。依据：

- 工具调用率 100% → 不落入 skip-tool shortcut 失效分支（07 决策树第 1 分支命中）
- 43 tok 是 **max_turns=2 冒烟配置的假象**（每轮一次紧凑调用 ~21 tok，2 轮即触顶）；全量 max_turns=8 下模型有 zoom→bbox→answer 的完整空间，**无需调整 max_turns**
- 训练段障碍（视觉 token / AdamW 状态）已定位并出方案（见 07_rl_debug_report 根因 1/2）

**前置动作（跑全量前必须做，两条都是硬要求）：**

1. **开 reward 分解打印**：`patch_reward.py` 的 `num_examine` 默认 0（门控全关）、`base_reward.py` 的 `verbose` 默认关 → 下次任何冒烟/全量都在命令行开 `trainer.num_examine=50` 或对应配置，让 `[ATR] acc=... U=... C=... S=... → R=` 逐条打出来。**50 步止损没有 acc/U/C/S 分解就是盲跑**（本次回报已证明：mean reward 0.387 无法判断 U/C/S 各自行为）。
2. **processor_config.json 变更入库**：该修复（16M→1M 像素）只存在于远端模型目录、不进 git——**换实例/重装即丢失，且本地无法审计**。要求远端把修改后的 `processor_config.json` 全文贴进 `06_4gpu_rl_exec.md` 回报节（几行 JSON，`git diff` 一下贴出来），并在 `02_env_setup.md` 的环境清单里加一行"模型目录 processor_config 已改 1M 像素，勿回退"。

### 决策 2：50 步止损指标细化为三条硬门槛

前 50 步（每步 32 条 rollout，样本量足够）逐条满足才算"在学"：

| # | 指标 | 通过线 | 不满足时的动作 |
|---|------|--------|----------------|
| 1 | `actor/pg_loss` | 有下降趋势（非恒定/发散） | 停，回报 loss 曲线 |
| 2 | `agent/tool_call_mean` | 保持 > 0.5（不塌缩为 0） | 停——U/C/S 信号消失 |
| 3 | **`critic/acc/acc_of_this_batch`** | **出现非 0 值（不需要高，但必须离开 0）** | 停——模型学不会作答，可能模板/分布偏移 |

三条全过 → 继续到 640 步；任一不过 → 停 + 回报三指标曲线 + 若已存 checkpoint 说明路径。**acc 恒 0 是最可能的风险点**：SFT v2 先答后验版在 RL 环境 100% 调工具且 8/8 全错，提示 RL 模板与 SFT 训练分布存在偏移，首 50 步若 acc 不动说明偏移未被 RL 吸收。

### 决策 3：对 07_rl_debug_report 五个问题的本地回答

1. **Adafactor 收敛**：`beta2_decay=-0.8` 即 Adafactor 原论文默认 `beta2_decay=0.8`（torch 2.8 把 0.8 写成了 -0.8 的命名历史），语义一致。GRPO 目标非平稳，短记忆二阶矩（decay 0.8）通常**更稳**而非更差；不需要预先调。真机首步只看 pg_loss 是否下降 + reward 均值是否上升即可判定，异常才回退 adamw。
2. **降显存替代**：LoRA 在 verl 0.2.0.dev 无训练支持（05 已确认，RL 全参是唯一路径）；bf16 优化器状态（bitsandbytes 8bit）会引入新依赖且 FSDP 分片兼容性未验证 → **Adafactor 是当前最优解，同意定稿**。1.2G 余量偏紧但 2 卡 Mini 账本精确吻合，可接受；`ppo_max_token_len_per_gpu=3072` 已是保守值，首步 OOM 时先降它，别动 util/n。
3. **max_pixels 一致性**：SFT `max_image_size: 1024` 与 RL 1M 像素预算 = 同一数量级（1024²≈1M），**等价**，分布偏移风险低。首轮验证时顺带 grep rollout 日志确认无 truncation 告警即可。
4. **微批策略**：视觉 3696-3900 + 文本 ~1900 + 响应 → 单样本 ~8-11k token，3072/gpu 下会切 3-4 个微批，token 利用率一般但可跑。首步峰值通过后可选试 4096-6144 提吞吐，**不阻塞、不是本轮目标**。
5. **首轮验证清单补充**：冒烟四项（开 num_examine）→ 全量首步观测（`grep OutOfMemory` + step 峰值）→ 前 25 步 checkpoint 落盘成功确认 → 50 步三指标回报。回报格式：三指标曲线 + `[ATR]` 分解样例 3-5 条 + 若有问题附日志原文。

### 红线（沿用）

- 不启动全量直到：① 冒烟（开分解打印）重跑确认无回归；② 本文件变更已 push
- checkpoint/日志/processor_config 变更文件不进 git；回报写 06/07 回报节 + commit `[remote] ...`
