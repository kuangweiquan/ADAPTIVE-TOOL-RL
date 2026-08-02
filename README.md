# PyVision-ATR: Adaptive Tool Reward for Agentic Vision RL

> 不奖励"Tool 数量"，而奖励 **Tool Utility**、**Tool Efficiency**、**Tool Sequence**

---

## 项目结构

```
Adaptive-Tool-RL/
├── pyvision-rl/          # 🔵 原始 PyVision-RL (vendored 依赖,随仓库推送)
│   └── pyvision/rl/
│       └── reward.py     # [被替换] 原始 reward 函数
│
├── atr/                  # 🟢 核心贡献代码
│   ├── reward/
│   │   ├── utility.py   # U: Tool Utility（源自 CodeV 思想）
│   │   ├── cost.py      # C: Redundant Cost（重复检测）
│   │   ├── sequence.py  # S: Sequence Quality（源自 AdaReasoner 思想）
│   │   └── base_reward.py  # R = acc + λU - γC + ηS
│   ├── adapter/
│   │   └── patch_reward.py # 注入 ATR reward 到 PyVision-RL
│   └── config/
│       └── atr_config.py   # λ, γ, η 等超参数
│
├── experiments/
│   ├── configs/         # 实验配置 (YAML)
│   ├── scripts/         # 运行脚本
│   └── results/         # 实验结果输出
│
├── knowledge-base/      # 📚 论文资料 & 引用映射
│   ├── PyVision-RL.pdf
│   ├── CodeV.pdf
│   ├── AdaReasoner.pdf
│   └── REFERENCES.md   # 论文→代码映射记录
│
├── log/                 # 训练日志
└── README.md
```

## Reward 设计

| 项 | 符号 | 定义 | 来源 |
|---|------|------|------|
| Accuracy | acc | 答案是否正确 (0/1) | PyVision-RL |
| Utility | U | 有效工具数（输出被答案引用） | CodeV 思想 |
| Cost | C | 冗余工具数（重复 crop/OCR/frame） | 自制规则 |
| Sequence | S | 工具序列合理性（crop→OCR 好于 OCR→crop→OCR） | AdaReasoner 思想 |

**公式**：`R = acc + λ·U − γ·C + η·S`

## 快速开始

```bash
# 1. 安装依赖
cd pyvision-rl
pip install -e .
cd ..

# 3. 运行 baseline
bash experiments/scripts/run_baseline.sh

# 4. 运行 ATR
bash experiments/scripts/run_atr.sh experiments/configs/atr_full.yaml

# 5. 运行 ablation
bash experiments/scripts/run_ablation.sh
```

## 实验结果指标

| 指标 | 目标 |
|------|------|
| Accuracy | 不低于 PyVision-RL baseline |
| Tool calls | 下降 20–40% |
| Visual tokens | 下降 30%+ |
| Latency | 下降 15%+ |

## 环境

- 2×4090 (最好) / 1×A800
- Qwen2.5-VL-7B + LoRA (rank 16)
- RL steps: 200

## 引用映射

详见 [knowledge-base/REFERENCES.md](knowledge-base/REFERENCES.md) — 记录每篇论文的思想如何在 `atr/` 中实现。
