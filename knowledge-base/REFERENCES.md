# References & Code Mapping

> 核心原则：**每篇论文借一个核心思想，在 `atr/` 中自定义实现，而不是直接搬运代码。**

---

## PyVision-RL — RL Training Framework

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv.org/abs/2602.20739](https://arxiv.org/abs/2602.20739) |
| **代码** | `pyvision-rl/` — [GitHub](https://github.com/agents-x-project/PyVision-RL) |
| **复用方式** | RL 训练框架：GRPO、trajectory generation、training pipeline |
| **修改点** | 保留训练流程，仅替换 reward function |
| **ATR 映射** | `atr/adapter/patch_reward.py` |

### 关键引用

PyVision-RL 中的 Tool Reward：

\[
R = acc + 0.1 \times n_{tc}
\]

ATR 扩展为：

\[
R = R_{format} \cdot [R_{acc}+\lambda U-\gamma C+\eta S]
\]

其中：

- `U`: Tool Utility
- `C`: Tool Cost
- `S`: Tool Sequence Quality

---

# LLaVA-Plus — Vision Tool Calling Framework

| 项目 | 内容 |
|------|------|
| **论文** | LLaVA-Plus: Large Language and Vision Assistant with Plugins |
| **代码** | LLaVA-Plus official implementation |
| **借用的思想** | Vision tool library + tool calling paradigm |
| **ATR 映射** | `atr/tools/` + `atr/reward/` |

## 具体借用点

| LLaVA-Plus 的做法 | ATR 中的实现 |
|-------------|-------------|
| 将视觉能力封装为独立 plugins/tools | 将 crop、OCR、zoom 等视觉能力封装为 tools |
| Agent 根据任务选择工具 | RL 优化 tool selection policy |
| Tool execution 返回 observation | 记录 tool trajectory |
| Tool description 定义调用接口 | Tool schema 作为 action space |

## 关键差异（创新点）

| 维度 | LLaVA-Plus | ATR |
|------|-------------|-----------|
| Tool 使用方式 | Tool calling | RL-based adaptive tool calling |
| 优化目标 | 提升视觉任务能力 | 提升 tool efficiency 和 evidence acquisition |
| Tool reward | 无显式 tool efficiency optimization | Utility + Cost + Sequence reward |
| 应用场景 | 通用视觉助手 | Vision Agent evidence acquisition |

---

# CodeV — Tool Utility Evaluation Inspiration

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv 2511.19661](https://arxiv.org/abs/2511.19661) |
| **代码** | [github.com/RenlyH/CodeV](https://github.com/RenlyH/CodeV) |
| **借用的思想** | Step-wise tool utility evaluation |
| **ATR 映射** | `atr/reward/utility.py` |

## 具体借用点

| CodeV 的做法 | ATR 中的实现 |
|-------------|-------------|
| 判断 tool output 是否包含任务相关证据 | Rule-based question-aware utility scoring |
| 惩罚无效视觉操作 | Lazy crop detection |
| 对每一步 tool call 计算 utility | Step-wise utility accumulation |
| Tool result 作为中间 evidence | Evidence-oriented reward |

## 关键差异（创新点）

| 维度 | CodeV | ATR |
|------|-------|-----------|
| Tool 类型 | Code execution | Explicit vision tools |
| Utility 评估 | Judge model | Rule-based zero-cost evaluation |
| 优化方式 | Tool-aware policy optimization | Reward shaping for RL |
| Reward 组成 | Step utility | Utility + Cost + Sequence |

---

# AdaReasoner — Tool Sequence Reward

| 项目 | 内容 |
|------|------|
| **论文** | [arxiv 2601.18631](https://arxiv.org/abs/2601.18631) |
| **代码** | [github.com/ssmisya/AdaReasoner](https://github.com/ssmisya/AdaReasoner) |
| **借用的思想** | Hierarchical scoring + Format gate + Asymmetric fusion |
| **ATR 映射** | `atr/reward/sequence.py` |

## 具体借用点

| AdaReasoner 的做法 | ATR 中的实现 |
|-------------------|--------------|
| Hierarchical tool evaluation | `compute_per_call_scores()` |
| Format gate | `compute_format_gate()` |
| Asymmetric fusion | `compute_asymmetric_fusion()` |
| Tool quality aggregation | Sequence reward calculation |

## 关键差异（创新点）

| 维度 | AdaReasoner | ATR |
|------|-------------|-----------|
| Sequence evaluation | Tool call correctness | Tool order + evidence acquisition pattern |
| Format gate | Reasoning format | Tool trajectory validity |
| Application | General reasoning | Vision tool workflow |
| Reward | Format + tool score | Utility + Cost + Sequence |

---

# ATR Overall Mapping

## 工具定义 (atr/tools/) — 单一真值源

工具定义集中于 `atr/tools/` 注册表(VisualTool + ToolRegistry),离线管线的
SYSTEM_PROMPT 生成、工具分发、轨迹记录与奖励层的工具名校验全部由注册表驱动。

| 工具 | 参数 | 状态 | 来源 |
|------|------|------|------|
| `crop` | `bbox_2d` | ✅ 实现(PIL) | LLaVA-Plus 工具库范式 |
| `zoom` | `bbox_2d` | ✅ 实现(PIL,alias `zoom_in`) | LLaVA-Plus 工具库范式 |
| `rotate` | `angle` | ✅ 实现(PIL) | 参照 pyvision-rl VisualToolBox |
| `ocr` | `bbox_2d`? | ✅ 实现(pytesseract,可选) | LLaVA-Plus 工具库范式 |
| `select` | — | ❌ 已移除(假实现,无真实选择能力) | — |
| `read_frame`/`extract_frames`/`zoom_out`/`search` | — | ❌ 已移除(有名无实) | — |

- 调用格式:`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`(JSON action)
- 轨迹记录:`ToolTrace` → `{tool_name, arguments, output, bbox?}`(ATR reward 直接消费)
- 奖励层的 `valid_tools`/`valid_params`/`spatial_tools` 由注册表派生
  (`atr/reward/sequence.py` / `utility.py` / `cost.py`)

## 验证

- `experiments/scripts/smoke_tool_refactor.py` — 注册表/schema/执行/Trace/Reward/prompt 全链路冒烟
- 旧轨迹回归:`--analyze_only` 重跑 `trajectories_20260729_175933.jsonl`:
  utility/cost 不变(±0),sequence 因 select 移除按预期下降(~0.019)
