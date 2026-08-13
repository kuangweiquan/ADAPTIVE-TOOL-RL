# AdaTooler-V 拼接路线可行性评估（2026-08-13，本地端）

> 依据：AdaTooler-V 开源仓库深读（`D:\my_project\adatooler_v_review`，commit = 2026-08-13 当日 clone）、论文 arXiv:2512.16918（ACL 2026 Findings）、本项目 07 审计与 52 步 RL 实测。
> 审查框架：ccf-idea-reviewer（标准模式）+ ccf-experiment-designer（实验包设计）。

## 1. 一句话结论

**AdaTooler-V 是当前 closest work 中唯一"同基准（V*）+ 同规模（7B）+ 代码/权重/数据全开源"的基座；且其开源 reward 代码中论文招牌机制（ΔS 门控 AT 奖励）处于停用状态——拼接路线的可行性被两个具体事实坐实：基座可复现、增量有空位。加权评分 3.60/5（对比：自家 Track A 续跑 2.78、Track B 分析 3.4）。推荐 accept-to-develop。**

## 2. 仓库深读关键发现

### 发现 1：开源 reward 代码中，AT-GRPO 招牌机制被注释/桩化

- `verl_tool/workers/reward_manager/adatooler_v.py:326`：`add_additional_penalties` 调用被注释 → **开源代码实际训练 = 纯 acc GRPO**
- 同文件 `:229`：`TB_score = 1.0` 硬编码，真值来源 `extra_info["Tool_Benefit_Score"]` 被注释
- 但公式完整保留在 `compute_ATReward`（`:143-160`）：`R = ΔS·exp(−γ((n_tool−n_max)/n_max)²)`，参数 α=0.6、β=0.05、γ=2、n_max=6、group_tool_call_rate 下限 H=0.3、冗余上限 n_vo=1 全部在位
- 含义：**重实现 + 重新启用 + 消融 = 有明确空位的正事**，不是硬凑

### 发现 2：V* 评测脚本未随仓库开源

- `examples/train/adatooler_v/eval.sh` 是 **0 字节空文件**；`benchmarks/` 只有数学/代码评测子模块
- 89.8% 的官方复现路径缺失 → 评测口径需用官方 V*（191 题 MCQA）自行重建（本项目已有 VStar 数据与评测基础设施，见 §4）

### 其余事实

| 项 | 状态 |
|----|------|
| SFT 权重 `AdaTooler-V-SFT-model` | 开源 → **跳过 100k SFT 冷启动** |
| 最终模型 `AdaTooler-V-7B` | 开源 → 上界锚点 |
| RL 数据 300k / SFT 数据 100k | 开源（HF） |
| 训练配置 | 8×H100/A100 80G，batch 64，n=8，max_turns=2，prompt 16k，tensor_parallel=2 |
| 工具 | zoom_in（含 0.1 padding 的 crop）/ path_tracer / select_frames，单图任务实际只用 zoom_in |
| 数据格式 | verl parquet，problem_type ∈ {numerical, multiple choice, OCR, free-form}，V* = multiple choice |
| 评测数据集 `AdaTooler-V-eval` | 开源（HF）→ 交叉校验用 |

## 3. 可结合性矩阵（本项目资产 → 拼接论文中的位置）

| 本项目资产 | 在拼接论文中的位置 | 状态 |
|-----------|------------------|------|
| ATR U/C/S 规则型奖励 | 消融臂 C：「零 judge 成本规则信号」 | 可直接移植（reward manager 接口同构） |
| C≡0 诊断协议（per-type 计数器 + 协议匹配基线） | 方法论章节核心；**发现 1 已是第一个战果**——SOTA 开源代码存在静默奖励死亡 | 故事升级：从"自家管线失败"→"开源 SOTA 也存在，这是检测协议" |
| 4×3090 显存账（梯度跨 micro、优化器延迟加载、grad ckpt） | 复现可行性章节：8×80G → 4×24G 降配方法 | 52 步实测验证 |
| VStar 数据 + 官方评测基础设施 | V* 评测重建（评测协议 + 数据） | 现成 |
| 52 步失败经验（模板偏移、acc 塌缩） | related work / 讨论章节对比素材 | 现成 |

**拼接点定位**：AdaTooler-V 的 ΔS 需要"调工具 vs 不调工具"成对 rollout（2× 训练期推理成本）；ATR U/C/S 是零成本规则信号。论文问题 = **「tool-use RL 的奖励信号性价比：judge 式 ΔS vs 免费规则塑形 vs 纯 acc」**——他们论文动机（反 blind tool-use）与我们的 C 项同源，问题成立且无人做过系统对比。

## 4. 可行性五问

| 问 | 答 |
|----|----|
| SFT 阶段要重跑吗？ | 不要。SFT 权重已开源 |
| RL 数据多大？ | 300k 开源；**单图 V* 相关子集 2-5k 即可**（视频 64 帧样本排除——视觉 token 会炸 24G） |
| 4×3090 跑得动吗？ | 可以。7B + max_turns=2（响应比我们 8 轮短）+ 我们的显存账（grad ckpt / offload / mini_batch=1-2）。降配：batch 64→16-32、n=8→4、prompt 16k→8k 过滤、tensor_parallel 2→1 |
| 评测怎么做？ | 官方 V* 191 题 MCQA（我们的 VStar 数据 + 官方协议）+ 他们 eval 数据集交叉校验；锚点 = 开源 SFT 模型与 7B 模型在我们口径下的实测值 |
| ΔS 臂怎么做？ | 论文公式重实现；ΔS 用 SFT 模型对 1k 子集离线预计算（tool vs no-tool 成对 rollout），训练期查表 |

## 5. 风险与对策

| 风险 | 等级 | 对策 |
|------|------|------|
| 审稿人认为"重实现+消融"增量薄 | 中 | 静默奖励死亡发现 + 信号性价比框架撑厚度 |
| ΔS 成对 rollout 2× 成本（1k 样本 ≈ 17-33 GPU 时） | 中 | ΔS 臂用 500-1k 子集，其余臂全量 |
| verl-tool 环境在远端装不上 | 中 | **Fallback：四臂搬回我们 pyvision-rl verl 跑**（ATR 组件原生、AT 惩罚+ΔS 移植量小、起点用他们 SFT 权重；故事弱化但仍成立） |
| 四臂全无差异 | 中 | 负面结果框架（"已冷启动模型上奖励塑形边际效应≈0"）+ 并入 Track B——数据仍可写 |
| 撞车（他们团队自己做） | 低 | 代码不完整说明已 move on；先发优势仍在 |
| 开源 SFT 模型在我们口径下表现差 | 低 | Phase 0 先测锚点，Gate 0 卡死 |

## 6. 与其它路线的对比

| 路线 | 加权分 | B 会概率（本端判断） | 时间 | 主要消耗 |
|------|------|------|------|------|
| **拼接 AdaTooler-V（本方案）** | **3.60** | 40-55% | 5-7 周（含写作） | 4×3090 约 4-5 天 GPU |
| Track B（自家失败分析） | 3.4 | 30-40% | 1-2 周 | 纯 CPU |
| 续跑自家 Track A | 2.78 | 25-35% | 1.5-2 个月 | 40+ GPU 时且结果不确定 |

**执行计划见 `ai_handoff/10_adatooler_stitch_plan.md`。**

## 7. 审核声明

- Mode：standard idea-review + experiment-design 前置
- No-invention：§2 两发现均指到仓库具体行号；89.8%/78.5% 为论文报告值（标注来源），无任何捏造数字
- 置信度 3/5：不确定点 = V* 评测重建难度、ΔS 重实现保真度、消融显著性（Gate 0/1 逐一消解）
- Next CCFA owner：ccf-integrity-auditor（Phase 3 结果回收后）
