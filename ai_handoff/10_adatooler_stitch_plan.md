# 10 执行计划：AdaTooler-V 拼接 — 奖励信号消融实验包（待用户批准后启动）

> 本地 AI 撰写（2026-08-13）。依据：`knowledge-base/ADATOOLER_V_FEASIBILITY.md`（可行性评估，先行阅读）。
> 设计框架：ccf-experiment-designer（Mode: design + result-template）。
> 状态：**待批准**。批准后按 Phase 顺序执行，每 Phase 结束过 Gate。

---

## 1. Mode / Venue / 假设

- **Mode**: design + result-template（所有数值 TBD，禁止捏造）
- **目标会议族**: ACL/NAACL Findings 级、CCF-B 会（COLING/ICASSP/ICPR）、CCF-C 会（NLPCC/ICONIP/IJCNN）的复现-消融类论文
- **假设**: 4×3090 24G×4 为唯一 GPU 资源；AdaTooler-V-SFT-model 权重可下载；V* 官方 191 题可用（本项目已有）

## 2. Claim–Evidence 矩阵

| # | 论文主张 | 证据 | 指标 | 产出位置 |
|---|---------|------|------|---------|
| C1 | 重新启用 AT 调用惩罚（论文公式，ΔS=1）能减少工具调用且不伤 acc | Arm B vs A | mean calls、EXCEED_MAX_TURNS%、V* acc | Table 1 |
| C2 | 规则型 U/C/S 以零 judge 成本达到与调用惩罚可比的效果 | Arm C vs B vs D | 同上 + firing 诊断 | Table 1/2/3 |
| C3 | judge ΔS 的收益来自逐样本门控，代价是 2× 训练期推理 | Arm D vs A/B + 成本实测 | acc 增益/成本 | Table 3 |
| C4 | 开源 SOTA 代码存在静默奖励死亡，检测协议可发现 | 静态审计发现（TB_score=1.0 stub、penalty 调用被注释）+ 各臂 firing 计数器 | 代码证据 + Table 2 | §方法论 |
| C5 | 免费规则信号 vs judge 信号的性价比边界存在（效率框架） | C/D 成本-收益对比 | GPU 时 vs acc | Table 3 |

## 3. 数据集 / 基准需求

| 用途 | 数据 | 规模 | 说明 |
|------|------|------|------|
| RL 训练（四臂共用） | AdaTooler-V-300k 的**单图子集** | 2-5k 样本 | 排除视频（64 帧视觉 token 炸 24G）与多图；filter >8192 token；V* 相关 benchmark（V*、GQA、等单图 MCQ）优先 |
| ΔS 预计算（仅 Arm D） | 同上子集的前 1k | 1k × 2（tool/no-tool 成对） | 用 AdaTooler-V-SFT-model 离线 rollout，ΔS=acc(tool)−acc(no-tool) 存 extra_info |
| 评测 | V* 官方 191 题 MCQA | 191 | 我们的 VStar 数据 + 官方协议；他们 `AdaTooler-V-eval` 作交叉校验 |

## 4. 方法版本（confirmed identities）

- **RL 起点（四臂共同）**: `AdaTooler-V/AdaTooler-V-SFT-model`（HF 已发布权重，跳过 SFT 阶段）
- **Arm A 纯 acc**: = 开源代码实际行为（penalty 调用保持注释状态）→ 受控基线
- **Arm B 调用惩罚**: 重启用 `add_additional_penalties`，TB_score=1.0，公式 `R=acc+0.6·exp(−2·((n−6)/6)²)`（论文方程，代码在 `adatooler_v.py:143-160`）
- **Arm C 规则型 U/C/S**: 本项目 ATR 组件移植到 verl-tool reward manager；配置 λ_u=1.0、γ_c=0.5、η_s=0.3 + **修复 C 缺陷后版本**（canonicalize + 调用预算检测器 + per-type 计数，见 09 文档 §3）
- **Arm D judge ΔS**: 按论文重实现 ΔS 门控（ΔS 离线预计算查表）；ΔS=0 样本惩罚、ΔS>0 放大
- **工具**: 只用 zoom_in（单图任务），path_tracer/select_frames 不参与

## 5. Baseline 矩阵

| Baseline | 为何包含 | 来源 | 公平性约束 | 预期指标 |
|----------|---------|------|-----------|---------|
| AdaTooler-V-SFT-model | RL 起点锚点 | HF 已发布 | 我们口径实测 | TBD（Phase 0 出） |
| AdaTooler-V-7B | 论文上界（报 89.8% V*） | HF 已发布 | 我们口径实测 | TBD |
| Qwen2.5-VL-7B base | 论文报 78.5% V* | 论文报告值 | 引用标注 | 78.5%（论文值） |
| Arm A 纯 acc GRPO | 受控基线（开源代码实际行为） | 本实验 | 同种子/步数/batch | TBD |

## 6. 主实验（四臂，全部匹配预算）

共同约束：同训练子集、同 seed、同 batch=16-32、同 n=4、同 LR=1e-6、同 max_turns=2、同步数（150-250 步，视吞吐）、同起点权重、同 4×3090 降配（grad ckpt=True、优化器 offload、mini_batch=1-2、prompt 8k 过滤、tensor_parallel=1）。

| 臂 | 奖励 | 预期差异化问题 |
|----|------|--------------|
| A | 纯 acc（开源原样） | 基线 |
| B | acc + 调用惩罚（重启用） | 惩罚能否压调用量？ |
| C | acc + U/C/S（规则，零 judge） | 规则信号能否替代惩罚/ΔS？ |
| D | acc + judge ΔS（重实现） | 逐样本门控值不值 2× 成本？ |

顺序：A→B→C→D（A/B 先出，过 Gate 1 再决定 C/D 规模）。

## 7. 消融 / 鲁棒性（只保留主张相关的）

- **C 分量内部消融（Arm C 内）**: C=调用预算 vs C=冗余模式 vs C=全关 → 支撑"哪类规则信号有效"（对应 09 文档缺陷 2 修复后的新能力）
- **ΔS 子集消融（Arm D 内）**: ΔS 预计算样本量 500 vs 1k → 支撑成本曲线
- **种子**: 四臂各 1 seed（GPU 预算约束，论文注明）；关键臂（胜者）补 1 seed
- 不做: 全 12 基准泛化实验（超预算，related work 引用即可）、视频基准（24G 不可行）

## 8. 诊断协议（方法论章节的数据来源，贯穿所有臂）

1. 每个 reward manager 打 **per-type firing 计数**（C 四类各自、U 均值/触发率、S 均值、ΔS 分布）每步入库
2. 训练期同时记录 `acc_of_this_batch`、`tool_call_mean`、`EXCEED_MAX_TURNS%`
3. 每臂完成后静态审计其 reward 代码路径（激活检查）——**"无静默死亡"声明需要此证据**

## 9. 结果表模板（全部 TBD，禁止预填）

**Table 1 — 主结果**: rows=A/B/C/D/起点SFT；cols = V* acc(官方191题) | mean tool calls/样本 | EXCEED_MAX_TURNS% | 冗余调用率 | mean resp len | reward mean

**Table 2 — 奖励分量 firing 诊断**: rows=四臂；cols = U 触发率 | C-重复spatial | C-振荡 | C-连续同名 | C-调用预算 | S 均值 | ΔS 均值 | **任一分量恒 0 则高亮**

**Table 3 — 信号性价比**: rows=四臂；cols = 训练 GPU 时 | 训练期额外 judge 推理成本 | V* acc | 每 1 GPU 时的 acc 增益

## 10. 执行阶段与决策门

### Phase 0 侦察（2-3 天，远端为主，本地并行准备评测）

1. 下载 `AdaTooler-V-SFT-model` + `AdaTooler-V-7B`（用户负责搬运，HF 可能需要代理）
2. 用我们现有 VStar 评测管线跑两个权重的官方口径 acc → **锚点数字**
3. 提取 300k 单图子集（2-5k）+ 验证 parquet 格式与本地方案一致
4. verl-tool 环境安装冒烟（conda env + vllm，不跑训练）

**Gate 0（go/no-go）**: 锚点合理（SFT 模型 V* acc 显著 > 随机 33%）+ 子集提取成功 + 环境可 import → 进入 Phase 1。否则转 Fallback（§12）。

### Phase 1 本地代码（3-5 天，纯 CPU，本地）

1. 移植 ATR U/C/S → verl-tool reward manager 接口（含 09 文档 C 缺陷修复：canonicalize + 调用预算 + per-type 计数）
2. 重实现 AT 奖励启用（Arm B）与 ΔS 门控（Arm D）
3. ΔS 预计算脚本（SFT 模型成对 rollout，输出 extra_info 表）
4. CPU 冒烟：合成轨迹上验证四臂 reward 计算 + firing 计数（复用 smoke_vstar_rl.py 模式）

### Phase 2 远端训练（5-7 天 GPU）

1. 降配脚本：8×80G 配置 → 4×3090（复用 06/08 显存账全部修复）
2. Arm A → Gate 1 → Arm B → Gate 1' → Arm C/D（规模按 Gate 决定）
3. 每臂 ~150-250 步，save_freq=10，诊断数据每步入库

**Gate 1（A/B 完成后）**: B 是否明显改变行为（calls ↓ 且 acc 不显著降）？
- 是 → C/D 全规模，进 Phase 3
- 否 → C/D 减半规模（500-1k 样本），论文框架切换为负面结果 + Track B 合并

### Phase 3 评测（1-2 天，远端）

四臂最终 ckpt + 起点 SFT + 开源 7B，统一官方 V* 口径 + 效率指标 → 填 Table 1/2/3。

**Gate 2（写作决策）**: 是否存在可发表故事？
- 存在（C≈B 零成本 / D 显著胜但贵 / 任一臂显著胜 A）→ 写拼接论文
- 全臂无差异 → 负面结果框架 + Track B 合并（"已冷启动模型上奖励塑形边际效应≈0"），仍可写

### Phase 4 写作（2-3 周，本地为主）

贡献结构：① 复现+重启用+四臂消融；② 静默奖励死亡检测协议（含发现 1 证据）；③ 奖励信号性价比框架；④ 8×80G→4×24G 复现方法。

## 11. 算力预算（4×3090）

| 项 | 估算 |
|----|------|
| Phase 0 推理 | 1-2 GPU 时 |
| Arm A/B/C 各 ~200 步 × 4-6min | 各 13-20 GPU 时 |
| Arm D ΔS 预计算 1k×2×~40s | 22 GPU 时 + 训练同 A |
| Phase 3 评测 | 2-3 GPU 时 |
| **总计** | **~4-5 天 GPU 连续 + 1-2 天评测**，墙钟 1-2 周（含排队/重启） |

对比：自家 Track A 续跑 640 步 ≈ 40+ GPU 时且结果不确定——本计划每臂产出独立数据点，中途任何一臂失败不拖垮全局。

## 12. Fallback（两个独立逃生通道）

1. **verl-tool 装不上** → 四臂搬回本项目 pyvision-rl verl（52 步已验证）：ATR 组件原生，AT 惩罚 + ΔS 移植量小（reward manager 一个文件），起点仍用他们 SFT 权重。故事从"在其开源 pipeline 上消融"弱化为"以 SOTA 权重为起点的奖励消融"，仍成立。
2. **四臂全无差异** → 负面结果论文 + Track B 合并（见 Gate 2）。

## 13. 分工

| 端 | 任务 |
|----|------|
| 用户 | 批准本计划；权重/数据包搬运（HF→远端） |
| 本地 | Phase 1 全部代码 + 冒烟；Phase 0 的评测管线准备；本文档与 09 文档维护；Phase 4 写作 |
| 远端 | Phase 0 下载/环境/锚点评测；Phase 2 训练 + Gate 数据；Phase 3 评测；回报写本文件「远端回报」节 + `[remote]` commit |

## 14. 红线

- Gate 0 未过不启动训练；四臂未完成不写结论
- 所有结果表 TBD 状态保持到 Phase 3 实测数据到位；禁止用简化跑法预填
- checkpoint/日志不进 git；新代码遵循本项目规范（工具名 canonicalize、CPU 冒烟先行）

## 15. 声明

- **No-fabrication**: 本文件所有数值为 TBD 或论文报告值（89.8%/78.5% 标注来源 arXiv:2512.16918）
- **Next CCFA owner**: Phase 3 结果回收后 → ccf-integrity-auditor（数字与主张一致性）；写作阶段 → ccf-paper-writer

---

## 16. 远端回报（Phase 0 完成，2026-08-14，本地 SSH 直连执行）

**执行方式变更**：用户批准本地 AI 经 `ssh ATR` 直连远端执行（替代远端 AI 交接循环，CLAUDE.md 已更新）。

**资产落地（全部在 `/root/autodl-tmp/`）**：
- `models/AdaTooler-V-SFT-model`：16.6GB，safetensors 校验 ALL_OK（729 tensors）。7B gated，用户决定跳过（上界用论文值 89.8%）
- `datasets/adatooler_v_subset/`：2900 train + 100 val parquet（verl-tool 格式），单图 MCQ，来源 Spatial_Image + General_Image（候选 11,015，seed 42 水库采样）；图片 100% 唯一（源数据 150 个重复 problem_id 用行号命名隔离）
- `datasets/AdaTooler-V-300k/`：rl.json + rl_with_deltaS.json（15k 条官方 ΔS 表，本子集命中仅 204 → **Arm D 用自算 ΔS**（delta_s_precompute.py），官方表作对照）
- `datasets/vstar_official/`：官方 191 图 + test_questions.jsonl（答案 label）
- `adatooler_v_review/`：官方仓库 clone；verl-tool env 建于 `/root/autodl-tmp/envs/verl-tool`（conda clone atr + editable 安装 + 补依赖 math_verify/mathruler/jiwer/rouge_score/qwen_omni_utils + `llm_agent`→`agent_loop` shim + transformers 4.57.6）；全链 import OK（reward/tool server/agent loop/trainer），tool server 冒烟 /docs 200

**Gate 0 判定：PASS** —— 锚点 acc = **149/191 = 78.0%**（direct_attributes 78.3% / relative_position 77.6%，随机 33%）；子集提取 ✓；环境 import ✓。

**已知坑（写进过程以防踩回）**：无卡模式 cgroup 内存上限 2GB（全局共享，必须串行）；两个 HF 仓库走 Xet 存储（匿名 `hf download` 401 → `HF_HUB_DISABLE_XET=1` 或直连 curl/aria2）；300k 图片存类别 zip（内部前缀 `home/sig95vg/remote_data/wangcy/`）；hf download 多个 `--include` 标志会互相覆盖（须单标志多值）；transformers 4.57.2 本地 Qwen2.5-VL tokenizer 加载 bug（钉 4.57.6）。

**Phase 2 就绪**：`experiments/adatooler_stitch/train_arm_a_4x3090.sh`（Arm A 纯 acc，batch32/n4/prompt8k/TP1/gmem0.5/save10/150 步/console，等 4×3090 开机）。
