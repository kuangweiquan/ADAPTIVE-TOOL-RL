# ATR 离线实验快速入门

验证 Adaptive Tool Reward 能否比 PyVision-RL 原始奖励更合理。

## 1. 下载 VStar 数据集

```bash
# 克隆 VStar repo（包含图片和标注）
git clone https://github.com/penghao-wu/vstar /path/to/vstar_bench
```

目录结构预期：
```
vstar_bench/
├── direct_attributes/
│   ├── sa_4690.jpg
│   ├── sa_4690.json
│   └── ...
└── relative_position/
    ├── ...
```

## 2. 安装依赖

```bash
pip install -r experiments/scripts/requirements_offline.txt

# 可选：安装 Tesseract OCR（如果要做 OCR 工具调用验证）
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS:   brew install tesseract
# Linux:   sudo apt install tesseract-ocr
```

## 3. 运行实验

```bash
# 快速测试（10 条样本）
python experiments/scripts/run_atr_offline.py \
    --vstar_path /path/to/vstar_bench \
    --quick 10

# 完整运行（191 条全部）
python experiments/scripts/run_atr_offline.py \
    --vstar_path /path/to/vstar_bench

# 只做分析（复用之前保存的轨迹）
python experiments/scripts/run_atr_offline.py \
    --vstar_path /path/to/vstar_bench \
    --analyze_only \
    --trajectories_file experiments/results/trajectories_20260729_120000.jsonl
```

## 4. 输出说明

| 文件 | 内容 |
|---|---|
| `results/trajectories_{timestamp}.jsonl` | 原始轨迹（每行一个样本） |
| `results/analysis_{timestamp}.json` | 分析报告（统计数据 + 对比） |

## 5. 数据流

```
VStar 原始图片 ──→ 硅基流动 API (Qwen3-VL-8B-Instruct)
                      ↓
                <tool_call> 多轮交互
                      ↓
                本地工具执行 (PIL crop/OCR)
                      ↓
                保存轨迹 JSONL
                      ↓
                ATR 奖励计算 ←→ 原始奖励计算
                      ↓
                对比分析报告
```

## 6. 环境变量配置

脚本默认内置了你的 API key。如果想通过环境变量覆盖（更安全）：

```bash
# Windows PowerShell
$env:SILICONFLOW_API_KEY="sk-your-key-here"

# Linux/macOS
export SILICONFLOW_API_KEY="sk-your-key-here"

# 可选：自定义 API URL
export SILICONFLOW_API_URL="https://api.siliconflow.cn/v1"
```

## 7. 常见问题

**OCR 不可用？**
不影响实验，OCR 会返回 `[No text detected]`，utility 评分中 OCR 相关的信息增益项会偏低。如果你主要验证 crop/zoom/spatial 行为，可以接受。

**API Key 在哪里配置？**
脚本内置了你的 SiliconFlow key。如果需要更换，编辑 `run_atr_offline.py` 中的 `SILICONFLOW_API_KEY` 常量。

**VStar 的答案格式？**
VStar 约定 `options[0]` 是正确答案，模型需要输出字母（A/B/C/D）。脚本会处理字母到文本的匹配。
