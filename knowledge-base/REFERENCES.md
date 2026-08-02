# References & Code Mapping

> 核心原则：**每篇论文借一个思想，在 `atr/` 中用自定义方式实现**，而不是搬运代码。
> 详细代码对照见 `references/CODE_NOTES.md`

---

## PyVision-RL (Base Framework)

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv.org/abs/2602.20739](https://arxiv.org/abs/2602.20739) |
| **代码** | `pyvision-rl/` — [GitHub](https://github.com/agents-x-project/PyVision-RL) |
| **复用方式** | 整个 RL 训练框架：Python Runtime、GRPO、Oversampling-Filtering-Ranking |
| **修改点** | 仅替换 reward 函数，不动其他模块 |
| **ATR 映射** | `atr/adapter/patch_reward.py` 注入新 reward |

**关键引用**：
- Accumulative Tool Reward: `R = acc + 0.1 × n_tc` → 我们的 `R = (R_format)·[acc + λU − γC + ηS]`

---

## CodeV — Tool-Aware Policy Optimization

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv 2511.19661](https://arxiv.org/abs/2511.19661) — CVPR 2026 Oral |
| **代码** | [github.com/RenlyH/CodeV](https://github.com/RenlyH/CodeV) |
| **借用的思想** | **Question-aware step-wise utility** |
| **ATR 映射** | `atr/reward/utility.py` |

### 具体借用点

| CodeV 的做法 | 我们在 utility.py 中的实现 |
|-------------|--------------------------|
| Judge model 验证 tool output 是否包含问题需要的证据 | **Rule-based 问题关联度评分**：检查 output 中的实体是否回答问题的询问方向（颜色/数量/位置等） |
| 惩罚大而无意义的 crop | **Lazy crop 检测**：crop 面积 > 85% 原图 → 扣分 |
| 奖励提供了证据的 tool | **Question-aware 关键词匹配** + **信息增益**（OCR 提取数值/结构化数据） |
| step-wise dense reward | 逐 tool 计算 utility 后求和 |

### 关键差异（创新点）

| 维度 | CodeV | 我们的 ATR |
|------|-------|-----------|
| Utility 评估 | Judge model（需要额外 LLM） | **纯 rule-based**（零推理成本） |
| 工具类型 | Code execution（Python） | **Vision tools**（crop/OCR/zoom） |
| 整合方式 | TAPO 独立的 policy opt | **作为 reward 项**与 C、S 联合 |

---

## AdaReasoner — Dynamic Tool Orchestration

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv 2601.18631](https://arxiv.org/abs/2601.18631) |
| **代码** | [github.com/ssmisya/AdaReasoner](https://github.com/ssmisya/AdaReasoner) |
| **借用的思想** | **Hierarchical scoring + Format gate + Asymmetric fusion** |
| **ATR 映射** | `atr/reward/sequence.py` |

### 具体借用点

| AdaReasoner 的做法 | 我们在 sequence.py 中的实现 |
|-------------------|--------------------------|
| `R_format = ∏ R_format(τ_i)` 乘性格式门控 | `compute_format_gate()` — 层次化(0-3)打分后归一化为乘性因子 |
| 层次化工具评分（Structure→Name→Params→Content） | `compute_per_call_scores()` — 同样 0-3 层次化 |
| 不对称自适应：正确→忽略工具；错误→奖励工具质量 | `compute_asymmetric_fusion()` — 融合 S 和 call validity |
| `R_total = R_format · (λ_tool · R_tool + λ_acc · R_acc)` | 在 `base_reward.py` 中实现 `total *= format_gate` |

### 关键差异（创新点）

| 维度 | AdaReasoner | 我们的 ATR |
|------|-------------|-----------|
| 序列评估 | Hierarchical（逐调用质量） | **Hierarchical + Pattern-based**（不仅质量，还有顺序合理性） |
| Format gate | 严格二值（有错=0） | **可配置** strict/soft |
| Asymmetric | 二态（correct/wrong） | 同 AdaReasoner + 额外保留 η·S 调节 |
| Vision 适配 | 通用工具 | **Vision 专用模式**（crop→OCR 好于 OCR→crop→OCR） |

---

## 实验配置变体

| Config | Utility | Cost | Sequence | Format Gate | Asymmetric |
|--------|---------|------|----------|-------------|------------|
| `baseline` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `utility_only` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `cost_only` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `sequence_only` | ❌ | ❌ | ✅ | ❌ | ❌ |
| `no_sequence` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `atr_additive` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `atr_format_gate` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `atr_full` | ✅ | ✅ | ✅ | ✅ | ✅ |
