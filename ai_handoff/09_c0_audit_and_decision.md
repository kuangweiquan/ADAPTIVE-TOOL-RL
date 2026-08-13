# 09 本地审核：08 step50 停训回报 + C=0 代码级审计 + 下一步决策

> 本地 AI 撰写（2026-08-13）。收阅对象：用户 + 远端 GPU AI。
> 依据：`ai_handoff/08_step50_stop_report.md`、`atr/reward/{base_reward,cost,utility}.py`、`atr/adapter/patch_reward.py`、`atr/tools/{base,image_tools}.py`、`atr/prompts.py`、`run_vstar_full.sh`、06/07 交接文档。

---

## 1. 审核结论（一句话）

**训练工程侧：修复链可靠、汇报诚实、可恢复（latest=50）。论文侧：C 项全程恒 0，当前 52 步训练的是 `acc+λU+ηS` 而不是 ATR 完整公式——以本轮结果支撑论文 C 项主张是不可发表的；且三门槛中两门（pg_loss、tool_call_mean）在最终报告缺证据。**

---

## 2. 数值交叉验证（复算 vs 报告）

| 报告声称 | 本地复算 | 判定 |
|---------|---------|------|
| reward 均值 0.377（区间 0.267-0.487） | 0.379 ✓ 区间 ✓ | 一致（舍入） |
| acc 均值 0.143（0.031-0.281） | 0.142 ✓ | 一致 |
| timing 均值 ~245s（4.1 分钟） | 247.2 ✓ | 一致 |
| tok/s 65-90 | 表内 64.8-90.1 ✓ | 一致 |
| MB diag：53 条<674MiB 占 10.5% | 53/506=10.47% ✓ | 一致 |
| 90 条<1G 占 17.8%；416 条≥1G 占 82.2% | 17.79% / 82.21% ✓ | 一致 |
| 52/630 ≈ 8.3% | 52/630=8.25% ✓ | 一致 |
| ckpt 96G/350G | 3×32G ✓ | 一致 |
| **总步数 ≈ 630** | **171÷8×30 = 641**（脚本头注释写 ≈640） | 小偏差，无实质影响 |
| **「reward 无退化趋势」** | **尾部 step45-50：0.487→0.424→0.419→0.438→0.353→0.267（step50=全程最低）** | **措辞偏乐观**：无单调恶化，但尾部 6 步下倾，应表述为「待观察」 |
| **max_alloc 26.89→32.08G 单调爬升** | step52 已达 reserved 上限 34.268G 的 **93.6%** | **报告未讨论**，见 §4 风险 |

---

## 3. C=0 结构性缺陷：本地代码审计（已定位到行）

远端确认「所有 step C=0.000」（22 步 × 32 样本 ≈ 4000+ 次工具调用，C 一次都没触发）。本地逐行审计 reward 路径后，找到 **3 个具体缺陷**，两个可能的根因假设：

### 缺陷 1：工具名从未规范化（alias 静默失效）

- `patch_reward.py:53` 解析时存**原始名**：`record = {"tool_name": call_data.get("name", "unknown")}`——模型写 `zoom_in` 就存 `zoom_in`
- `cost.py:42`（Type 1 重复空间操作）要求 `tool_name in registry.spatial_tools`，而 `base.py:136` 的 `spatial_tools = {"crop", "zoom"}`（**规范名**，不含 alias）
- `cost.py:80`（Type 4 zoom 振荡）硬编码 `prev == "zoom"`
- 注册表**有** `registry.canonical()`（`base.py:113`）但 reward 路径全程不调用 → `zoom_in`（alias，`image_tools.py:65`）的调用会让 Type 1/4 静默失效
- 当前 SYSTEM_PROMPT 教的是规范名 `zoom`（`prompts.py:63`），所以**可能不是本轮 C=0 的主因，但它是真实的脆弱点**，utility.py（93/137/162/201 行）同样受影响

### 缺陷 2：检测器覆盖与观测行为不匹配（C 设计盲区）

C 的四类检测器全部是「冗余模式」检测：
- IoU>0.5 才算重复 crop/zoom；text_sim>0.85 才算重复 OCR；仅「连续同名」才算 spam；zoom 振荡需严格 zoom-crop-zoom 三元组

而观测到的真实行为是：**tool_call_mean 5.6-6.5 / max_turns=8（几乎用满工具预算）+ SFT 史上有 zoom 死循环（44/61 卡满 8 轮）**。C 里**没有任何「过度调用/预算浪费」检测器**——对当前 RL 分布天然无判别力。即使检测器零 bug，C 也大概率恒 0。

### 缺陷 3：无法区分「parse 漏解析」vs「检测器不触发」

- 假设 A：multi-turn 响应里 `parse_tool_trajectory` 漏掉了大部分 tool_call 块（如实际只解析到 1-2 个）→ 检测器没素材，C=0 是**解析失败**
- 假设 B：解析正常（~6 个 call 全进来），但模型严格交替工具名 → 检测器全不触发，C=0 是**设计盲区**
- 现有日志无法区分两者。**判定材料已在远端日志里**：`num_examine=50` 会打印 50 条 `[ATR] acc=… U=… C=… S=…` 行（`patch_reward.py:211-214`），加上 2-3 条原始 response 样本即可判定

### 附带确认

`base_reward.py:157-162`：标准加性融合下 R = acc + 1.0·U − 0.5·C + 0.3·S。C=0 时本轮 reward≈0.377 中，acc≈0.14 只占约 1/3 质量，**主导信号是规则型 U/S shaping**——模型在学「满足工具使用规则」，不是在学「答对题」。这与 acc 长期徘徊 14%（SFT-v2 直接作答基线 62%，`knowledge-base/TOOL_SUMMARY.md`）互相印证。

---

## 4. 三门槛逐门判定（依据 06 文档 256-262 行定义）

| # | 门槛 | 本轮证据 | 判定 |
|---|------|---------|------|
| 1 | `actor/pg_loss` 下降趋势 | **08 报告无 step31-52 的 pg_loss 数据**（只在 07 报告 step1-5 出现过） | **证据缺失** |
| 2 | `agent/tool_call_mean` > 0.5 | **08 报告无 step31-52 数据**（step1-5 为 5.6-6.5） | **证据缺失**（大概率过，但未记录） |
| 3 | `acc_of_this_batch` 离开 0 | 0.031-0.281 全程非零 ✓ | **通过** |

**门槛总判定：未通过（2/3 缺证据）。** 不是训练失败，是汇报不完整——远端需要补这两列数据（日志都在，一条 grep 的事）。

---

## 5. 残留风险清单（排序）

1. **C≡0（HIGH，论文级）**：见 §3。核心公式未完整训练，本轮结果不能支撑 C 项主张。
2. **max_alloc 单调爬升（MEDIUM）**：26.89→32.08G（93.6% 上限）。reserved 池恒 34.27G 属实，但 allocated 若按此趋势继续，~20 步后会触顶扩池，post-bwd free 0.03G 的贴顶状态可能复发。续跑时监控 `max_alloc_G`。
3. **post-bwd 贴顶概率性踩线（MEDIUM）**：53 条 <674MiB 全过是事实，但 0.03G 谷值没有安全余量。cast 分块化（`_runtime_utils.py:1026`）可做但优先级低于 C=0。
4. **ckpt 空壳 + step30 遗留（LOW）**：已如实记录，远端重启前手动清一次即可；`resume_mode` 未显式写但 verl 默认 auto → latest=50 恢复成立。
5. **本地 `images/` 目录未 gitignore（LOW）**：见 git status，待入 ignore 规则。

---

## 6. 下一步决策（推荐路线）

### Step A：本地修 C=0（现在可做，纯 CPU）

1. `patch_reward.py` parse 后用 `registry.canonical()` 规范化 tool_name（一处调用，修复缺陷 1）
2. `cost.py` 增加**调用预算成本**：`n_calls > budget（如 3）` 时按超出量给 C（对齐观测行为「5.6-6.5 次/8 轮」——这是当前分布里最真实的成本信号）
3. `cost.py` 加 per-type 计数器进 components（区分四类检测器各触发几次 → 直接区分假设 A/B）
4. 跑 `experiments/scripts/smoke_vstar_rl.py` + 新增冗余轨迹合成用例（CPU 可跑），确认 C 能在合成冗余轨迹上触发

### Step B：向远端要判定材料（用户搬运）

- 远端日志里 `grep "\[ATR\]" logs/qwen3vl_8b_sftv2_grpo_4gpu.log | head -50` → 50 条 ATR 分解行（看 U/S 大小 → 推断 parse 是否正常）
- 2-3 条原始 response 字符串样本（看 tool_call 块实际格式与数量）
- step31-52 的 `pg_loss` 与 `tool_call_mean` 两列（补门槛证据）

### Step C：补协议匹配基线（判定 14.3% 的含义）

SFT-v2 模型在**相同 agentic rollout 协议**下的 acc（rollout-only，不训练，1 epoch）。没有它，RL 的 14.3% 不可解释（对比锚只有 62% 直接作答，协议不同不可比）。此步不依赖 C 修复，可与 Step A 并行。

### Step D：修复后重启而非续跑（推荐）

若 C 修复改变了 reward 分布 → **从 SFT-v2 重启**（52 步 ≈ 3.5 GPU 小时，成本远低于「训练在错误 reward 下」的论文可信度损失）。续跑仅当判定 C 修复对已训 52 步的 reward 无实质影响时考虑（需 Step B 材料支撑）。

### Step E：Track B 触发条件

修复 + 重启后 step50 若 acc 仍 ≈14%（无上升趋势），且基线（Step C）显示协议匹配基线 ≥14% → **RL 未产生正收益，pivot Track B**：failure-mode 分析论文。已有素材充分（C=0 缺陷史、zoom 死循环、模板偏移、OOM 修复链、视觉 token 根因）。

---

## 7. 回报要求（远端）

按顺序给：① `[ATR]` 行 50 条 + 2-3 条原始 response；② pg_loss / tool_call_mean 两列（step31-52）；③ Step C 基线 acc（若本地确认执行）。不进 git，随回报粘贴或走日志搬运。

## 8. 审核模式声明

- Mode：claim-audit + numeric-audit（本报告无引用条目，citation-audit 不适用）
- No-invention：所有数值从 08 报告表格直接复算；C=0 缺陷指到具体代码行；无法本地验证的（日志原文、MB diag 原始数据、显存归零）明确标为「远端证据未入库，暂按报告采信」
- Next CCFA owner：ccf-experiment-designer（Step C 基线协议设计）
