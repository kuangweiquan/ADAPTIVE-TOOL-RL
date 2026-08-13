# adatooler_stitch — AdaTooler-V 拼接消融的四臂奖励实现

对应 `ai_handoff/10_adatooler_stitch_plan.md`（Phase 1 本地代码交付物）。

## 文件

| 文件 | 用途 | 运行位置 |
|------|------|---------|
| `arm_logic.py` | 四臂奖励纯逻辑（无 verl/torch 依赖）+ 静默死亡检测 | 本地/远端 |
| `verl_reward_manager.py` | verl-tool reward manager 适配器（注册名 `adatooler_v_stitch`）+ 像素坐标归一化 + 本地 MCQ 打分兜底 | 远端训练用，逻辑可本地冒烟 |
| `delta_s_precompute.py` | Arm D 的 ΔS 离线预计算（SFT 模型 tool/no-tool 成对 rollout） | 远端 GPU |
| `smoke_stitch_reward.py` | 四臂 + 格式适配 CPU 冒烟（16 项） | 本地 |

## 四臂

| arm | 奖励 | 说明 |
|-----|------|------|
| `a` | 纯 acc | = 他们开源代码实际行为（penalty 调用被注释，`adatooler_v.py:326`） |
| `b` | acc + 0.6·exp(−2·((n−6)/6)²) | 重启用他们的 `add_additional_penalties`（TB_score=1.0 桩值，`:229`） |
| `c` | acc + λ_u·U − γ_c·C + η_s·S | 我们的规则型信号（零 judge）；C 已修复（09 文档 §3：alias 规范化 + 调用预算 + per-type 计数） |
| `d` | acc + 0.6·ΔS·exp(−2·((n−6)/6)²) | 论文意图的 judge 门控；ΔS 来自 `delta_s_precompute.py` 产物 |

## 远端接入步骤（Phase 2）

1. 把本目录复制到 verl-tool 环境（或 mount 本项目仓库），确保项目根在 `PYTHONPATH`
2. `delta_s_precompute.py` 先跑出 `delta_s_map.jsonl`（仅 arm d 需要）
3. 训练命令加：
   ```bash
   reward_model.reward_manager=adatooler_v_stitch \
   +reward_model.reward_kwargs.arm=c \
   # arm d 额外加:
   +reward_model.reward_kwargs.delta_s_map=/path/delta_s_map.jsonl
   ```
4. 每步日志看 `[ATR-STITCH]` 行与 `DEAD_COMPONENTS` 告警——任何分量恒死立即回报（诊断协议核心）

## 已确认的适配决策（审计记录）

- **alias 桥接**：他们工具名 `crop_image`/`crop_image_normalized` → 我们的规范名 `crop`（`atr/tools/image_tools.py` alias）；`zoom_in` 既有 alias 生效
- **坐标空间**：他们的 prompt 教像素坐标 → `verl_reward_manager.normalize_bboxes` 在奖励路径归一化到 [0,1]（ATR 核心保持归一化假设不动）
- **公式保真**：Arm B/D 的 exp 惩罚按他们 `compute_ATReward`（`:143-160`）逐字复现；已知"错答+n=0 仍得 0.081"是他们公式的固有性质（复现臂不修，写论文时如实报告）
- **打分口径**：远端用他们的 `adatooler_reasoner_score`（lazy import）；本地冒烟用 MCQ 兜底实现

## 冒烟

```bash
python experiments/adatooler_stitch/smoke_stitch_reward.py   # 16 项
python experiments/scripts/smoke_c_fix.py                    # C 修复 11 项
python experiments/scripts/smoke_vstar_rl.py                 # 原管线回归 34 项
```
