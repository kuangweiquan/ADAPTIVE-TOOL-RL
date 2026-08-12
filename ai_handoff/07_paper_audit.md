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
