# 提示词 1/3 —— 项目介绍（先发给远端 AI）

你是本项目（PyVision-ATR）的远端执行 AI。项目压缩包已解压到服务器当前工作目录（解压后顶层目录就是 `atr/`、`datasets/`、`experiments/`、`knowledge-base/`、`pyvision-rl/` 等）。你的任务：先完整理解项目，再按后续提示词完成环境配置与 SFT 训练。

## 项目一句话

**Adaptive Tool Reward for Agentic Vision RL（自适应工具奖励的智能体视觉强化学习）**：
8B 视觉模型零样本跑不动工具循环（必调工具、乱定位、zoom 死循环），397B 大模型能跑对（80% 正确率）。方案是用 397B 生成的高质量工具轨迹做 8B 的 **SFT 冷启动**，让 8B 学会「工具调用格式 + 归一化定位 + 收敛作答」，之后再进入 ATR-RL（verl + GRPO，自适应奖励）。

## 当前进度（重要）

- ✅ 397B 数据采集、过滤、格式转换已完成
- ✅ SFT 数据已就绪：`datasets/vstar_bench/sft/train.jsonl`（392 条）/ `val.jsonl`（43 条），重放图像在 `datasets/vstar_bench/sft_images/`
- ✅ 训练配置已写好：`experiments/configs/sft_qwen3vl_lora.yaml`、`merge_lora.yaml`、`dataset_info_vstar.json`
- ⏭️ **下一步：在你的服务器上执行 LoRA SFT 训练**（见提示词 2、3）

## 目录结构

| 目录 | 说明 |
|------|------|
| `atr/` | 核心库：`tools/`（工具注册表+执行器+轨迹）、`adapter/patch_reward.py`（ATR 奖励接口）、`reward/`（utility/cost/sequence 奖励分量）、`config/` |
| `experiments/` | 脚本（`run_atr_offline.py` 离线评估管线、`trajectories_to_sft.py` 数据转换等）+ `configs/`（SFT 配置） |
| `datasets/vstar_bench/` | VStar 基准数据：`direct_attributes/`、`relative_position/`（原始图）、`crops/`（bbox 裁剪）、`sft/`（训练 JSONL）、`sft_images/`（工具响应重放图） |
| `pyvision-rl/` | RL 训练框架（verl + GRPO），`verl_agents/`，后续 RL 阶段使用 |
| `knowledge-base/` | **项目文档，先读这些**：`SFT_PLAN.md`（总体方案）、`SFT_TRAIN_README.md`（训练执行手册）、`TOOL_SUMMARY.md`（工具接口）、`REFERENCES.md`（参考文献）、`gpt-chat-history*.md`（决策过程记录）、论文 PDF |
| `references/CODE_NOTES.md` | 代码笔记 |

## 关键概念

- **工具接口**：模型输出 `<tool_call>{"name": ..., "arguments": {"bbox_2d": [归一化坐标...]}}</tool_call>`，坐标是归一化到 [0,1] 的
- **两种推理模式**：强制工具模式（`--tool_required`）和先答后验模式（工具可用但先直接作答）。RL 推理统一用先答后验 prompt
- **ATR 奖励** = 准确率 + λ·工具效用 − γ·工具成本 + η·序列质量（详见 `atr/reward/` 和 `SFT_PLAN.md` §5）
- **VStar**：191 个样本的视觉推理基准（属性/相对位置问答，带 GT bbox）

## 工作原则

1. **先读 `knowledge-base/SFT_PLAN.md` 和 `SFT_TRAIN_README.md`**，理解方案后再动手
2. 每做一步先报告计划和预期，执行后报告结果；不确定时问我
3. 所有训练/长任务日志保存到文件，便于回溯
4. 不要修改数据文件本身（`datasets/` 只读）；不要删除不了解的文件
5. 模型权重下载走国内镜像（`HF_ENDPOINT=https://hf-mirror.com`），放 `/root/autodl-tmp`（数据盘，空间大）

---

请先汇报：你看到的项目目录结构、`nvidia-smi` 的 GPU 情况、磁盘空间，然后等待下一份提示词。现在是autodl的无卡开机模式，没有使用gpu，等所有准备做好了我再打开gpu
