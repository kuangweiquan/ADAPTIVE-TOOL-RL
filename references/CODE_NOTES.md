# Code Notes — 对照原始代码的分析记录

> 从论文和公开资料分析 CodeV / AdaReasoner 的实际实现，
> 对比我们 `atr/` 中的当前实现，找出差异和优化机会。

---

## CodeV — Tool-Aware Policy Optimization (TAPO)

### 实际实现

| 维度 | CodeV 实际做法 | 我们的 utility.py |
|------|---------------|-------------------|
| **评估方式** | Rule-based check + **Judge model** 验证 tool output 是否包含 **问题所需的证据** | 纯 rule-based：检查 output 是否出现在 answer 中 |
| **奖励对象** | Step-wise dense reward，每个 tool output 单独打分 | Σ 所有 utility 得分 |
| **惩罚对象** | "Lazy" tool: 大而无意义的 crop、错误的 tool 操作 | 有 cost.py 惩罚重复，但没惩罚"大而无意义" |
| **关键差异** | 检查的是 **"output 提供了问题需要的证据吗"** | 我们检查的是 **"output 出现在了最终答案里吗"** |

### ⚠ 我们的关键缺陷

**utility.py 逻辑：** `if output text in answer → 加分`

但这是错的！反例：
- 模型 crop 了 A 区域，拿到"A 是红色"；然后又 crop B 区域拿到"B 是蓝色"。最终答案说"红色"。B 的 output 没出现在答案里，但 B 的 crop 是对问题"哪个是红色"的有用对比证据。
- 或者模型先做了一次错误的 crop，拿到错误信息后纠正了 → 错误的 output 不在答案里，但它是推理过程的一部分。

**正确的做法（CodeV 思路）：** 检查 tool output **是否提供了回答问题的证据**，而不是是否出现在最终答案里。

---

## AdaReasoner — Tool-GRPO

### 实际实现

| 维度 | AdaReasoner 实际做法 | 我们的 sequence.py |
|------|---------------------|-------------------|
| **奖励结构** | `R = R_format · (λ_tool·R_tool + λ_acc·R_acc)` | `R = acc + λU - γC + ηS` |
| **R_format** | **乘性门控**: 任何一步格式错误 → R_format=0 → 总 reward=0 | 没有 format gate |
| **R_tool** | 层次化评分 (0-4): Structure → Name → Parameter Name → Parameter Content | pattern-based 顺序评分 |
| **R_acc** | 不对称自适应: 答对时忽略工具使用；答错时根据工具质量给分 | 答对给 acc，答错不给 |
| **关键差异** | **格式是乘性的、工具调用质量是层次化的、acc 和 tool 是不对称的** | 全都是加性的，没有层次结构 |

### ⚠ 我们的关键缺陷

1. **没有 format gate** — AdaReasoner 用乘法 gate 确保格式正确是硬约束。一个格式错误代表整个推理无效。
2. **没有层次化工具评分** — 他们打分是 Structure→Name→Params→Content 逐层递进的。我们只看了"用了哪些工具的顺序"，没看"工具调用本身是否正确"。
3. **没有不对称自适应** — 他们答对时鼓励简洁（不扣工具），答错时奖励好的工具使用。我们答错就只给 acc=0，放弃了工具质量的信号。

---

## 需要修改的内容

### utility.py (受 CodeV 启发)
- [ ] 改核心逻辑: **检查 tool output 是否提供了回答问题的证据**（而不是是否在答案中）
- [ ] 可选: 集成 judge model 验证（降级为 rule-based fallback）
- [ ] 增加对大而无意义 crop 的 detection

### sequence.py (受 AdaReasoner 启发)  
- [ ] 增加 **format gate** 概念: 工具调用格式检查
- [ ] 增加 **工具调用层次化评分**: tool name → params 逐级检查
- [ ] 增加 **不对称自适应**: R_total 根据 acc 调整

### base_reward.py
- [ ] 引入乘性 format gate
- [ ] 不对称 reward 融合

---

## 修改原则

1. **不改接口**: `AdaptiveToolReward.compute()` 的入参和出参不变
2. **向后兼容**: ablation 配置（utility_only, cost_only, sequence_only）行为不变
3. **不引入额外模型**: 保持无额外推理成本
