# 提示词 4/4 —— 第二轮 SFT（发给远端 AI）

背景：第一轮 SFT 训练成功但评估未过门槛（正确率 12%、IoU>0.1 22%、≤3 轮收敛 12%）。根因分析已在本机完成：

> **训练数据指令矛盾**：357/435 条工具轨迹是用「先答后验」system prompt 转换的，但样本行为是强制工具模式的行为（上来就调工具）。模型学到的是矛盾信号；评估时强制工具的 system prompt 是训练时从未见过的指令语境，模型不知道何时该停。

**修复**：已重新转换数据集 —— 工具轨迹改用**强制工具 system prompt**（与行为、与评估口径一致），直接作答轨迹保持先答后验。新数据在包里的 `datasets/vstar_bench/sft/train.jsonl`（392 条）和 `val.jsonl`（43 条），只替换这两个文件，**sft_images 不变**。

## Step 0：诊断评估（先做，验证根因，约 10 分钟）

用当前合并模型（Qwen3-VL-8B-ATR-SFT）跑**先答后验模式**（不带 --tool_required）quick 50：

```bash
cd /root/code
conda activate atr
python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
    --quick 50 --output_dir experiments/results/eval_sft_answerfirst
```

汇报：准确率、直接作答率（不调工具就答的占比）、平均轮次。**这组数据用于验证「prompt 错配」假设**（训练数据就是先答后验 prompt，若该模式准确率显著高于强制模式的 12%，则错配是主因之一）。

## Step 1：替换数据

```bash
cp /root/code/datasets/vstar_bench/sft/train.jsonl \
   /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/train.jsonl
cp /root/code/datasets/vstar_bench/sft/val.jsonl \
   /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/val.jsonl
wc -l /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/train.jsonl   # 392
wc -l /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/val.jsonl     # 43
```

> 若之前用了方案 B（sed 改图片路径），新文件也要做同样的路径替换。

## Step 2：重训（新 checkpoint，别覆盖 v1）

复制 `sft_qwen3vl_lora.yaml` 为 `sft_qwen3vl_lora_v2.yaml`，修改：

```yaml
lora_rank: 64            # 32 → 64，更多容量学定位与停止策略
lora_alpha: 128
num_train_epochs: 5      # 3 → 5（数据量小，5 epochs 仍便宜）
output_dir: saves/qwen3vl-8b-lora/vstar_tool_sft_v2   # 新目录
```

```bash
cd /root/autodl-tmp/LLaMA-Factory
nohup CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /root/code/experiments/configs/sft_qwen3vl_lora_v2.yaml \
    > /root/autodl-tmp/sft_train_v2.log 2>&1 &
```

监控同前（loss 应下降；eval_loss 参考 v1 的 0.344）。汇报训练时长与 loss 曲线。

## Step 3：合并 v2

修改 `merge_lora.yaml`（复制为 `merge_lora_v2.yaml`）：`adapter_name_or_path` → v2 最佳 checkpoint，`export_dir` → `/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2`。复用你第一轮修复 export 字段问题的处理（tokenizer/config.json 修复）。

```bash
llamafactory-cli export /root/code/experiments/configs/merge_lora_v2.yaml
```

## Step 4：重评（两个口径都跑）

```bash
# vLLM 服务切到 v2 模型（同第一轮参数：TP=2、--max-model-len 12288、--served-model-name）
# 口径 1：强制工具（对门槛表）
python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
    --quick 50 --tool_required --output_dir experiments/results/eval_sft_v2
# 口径 2：先答后验（RL 将用的模式，参考）
python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
    --quick 50 --output_dir experiments/results/eval_sft_v2_answerfirst
```

## Step 5：汇报格式

```
[诊断] 先答后验口径(旧模型)：准确率 X% / 直接作答率 Y% / 平均轮次 Z
[训练v2] 时长 / loss 曲线 / eval_loss
[评估v2-强制] 准确率 / IoU>0.1% / ≤3轮收敛%   ← 对照门槛表
[评估v2-先答后验] 准确率 / 直接作答率 / 平均轮次
[死循环分析] 若仍有 max_turns_exceeded：最后 2 轮的典型动作是什么？
   （重复同一 bbox zoom？zoom 到极限仍 zoom？工具类型单一？挑 5 条举例）
[判定] 通过 / 未通过
```

## 门槛（不变）

正确率 ≥50% | IoU>0.1 ≥30% | ≤3 轮收敛 ≥80%（强制工具口径）。

若仍不达标：报告死循环分析结果，我们会决定下一步（397B 多温度扩采 / 多步 zoom 教学数据 / HRBench），**不要擅自重训或加数据**。
