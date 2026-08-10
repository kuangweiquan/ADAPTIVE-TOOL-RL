# 提示词 3/3 —— SFT 执行计划（发给远端 AI）

目标：用 LLaMA-Factory 在 GPU 服务器上完成 Qwen3-VL-8B-Instruct 的 LoRA SFT，合并权重，并用工具循环评估验证。**每一步先报计划、再执行、后报结果。**

## 训练目标与通过门槛（SFT_PLAN.md §4.2）

| 指标 | SFT 后门槛 |
|------|-----------|
| 正确率（50 条强制工具） | **≥ 50%**（零样本仅 6.6%） |
| 工具定位 | 真实调用 + **IoU > 0.1 比例 ≥ 30%** |
| 收敛性 | **≥ 80% 样本 ≤ 3 轮作答**（零样本 44/61 死循环） |

不满足 → 报告并讨论（扩数据/调 LoRA），不要擅自决定。

## Step 1：准备（假设提示词 2 已完成）

确认：GPU 数量 N、模型在 `/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`、数据已注册、项目解压路径（下文以 `/root/code` 为例，按实际调整）。

## Step 2：调整训练配置

把 `experiments/configs/sft_qwen3vl_lora.yaml` 拷到 LLaMA-Factory 目录（或直接用绝对路径），按实际修改：

- `model_name_or_path: /root/autodl-tmp/models/Qwen3-VL-8B-Instruct`（本地路径）
- `dataset_dir: data`（保持 LLaMA-Factory 内部 data 目录）
- `output_dir: saves/qwen3vl-8b-lora/vstar_tool_sft`（保持相对 LLaMA-Factory 根，或改绝对路径）
- 若 GPU 数 ≠ 2：`per_device_train_batch_size` 保持不变，N 卡时等效 batch = 2×N×8，可接受（≤ 64 即可）
- **若 LLaMA-Factory 报参数错误**（如 `max_image_size` 不支持）→ 删掉该行，加 `--model_max_image_size 1024` 或 `--resolution 1024`（按 LLaMA-Factory 实际支持项）
- **若 template 报错**（`qwen2_vl` 不认 Qwen3-VL）→ 依次尝试 `qwen3_vl`、`qwen_vl`、`qwen2_5_vl`

## Step 3：启动训练（后台 + 日志）

```bash
cd /root/autodl-tmp/LLaMA-Factory
conda activate atr
# 日志必须保留：
nohup CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /root/code/experiments/configs/sft_qwen3vl_lora.yaml \
    > /root/autodl-tmp/sft_train.log 2>&1 &
echo $! > /root/autodl-tmp/sft_train.pid
tail -f /root/autodl-tmp/sft_train.log
```

预期：约 1-2 小时/卡（3 epochs、392 条、batch 等效 32、8192 截断）。`logging_steps:10`、`eval_steps:200`、`save_steps:500`。

## Step 4：监控

- `tail -f` 日志：loss 应稳定下降；eval 看 val 集指标
- **OOM（显存不足）** → 降 `per_device_train_batch_size` 到 1 并升 `gradient_accumulation_steps` 到 16，或改用 QLoRA（`quantization_bit: 4`）
- **训练中断** → 报告，不擅自重来
- 训练完成后汇报：loss 曲线（`plot_loss: true` 生成 `loss.png`）、eval 结果、checkpoint 路径

## Step 5：合并 LoRA

调整 `experiments/configs/merge_lora.yaml`：`model_name_or_path` 同上、`adapter_name_or_path` 指向最佳 checkpoint（或最后 save）、`export_dir` 设为 `/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT`。

```bash
llamafactory-cli export /root/code/experiments/configs/merge_lora.yaml
ls /root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT   # 确认 config.json + 权重齐全
```

## Step 6：工具循环评估（对照门槛）

```bash
pip install vllm   # 若未装；vLLM 需与 torch cu128 匹配版本
nohup vllm serve /root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 > /root/autodl-tmp/vllm.log 2>&1 &

export ATR_MODEL="Qwen3-VL-8B-ATR-SFT"
export SILICONFLOW_API_URL="http://localhost:8000/v1"
export SILICONFLOW_API_KEY="dummy"
cd /root/code
conda activate atr
python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
    --quick 50 --tool_required --output_dir experiments/results/eval_sft
```

- 若评估管线报错（缺依赖/接口变化），读 `knowledge-base/TOOL_SUMMARY.md` 和 `references/CODE_NOTES.md` 自查；解决不了就报告
- 统计并对照 Step 0 门槛表：准确率、IoU>0.1 比例、≤3 轮收敛比例

## Step 7：汇报格式

```
[训练] checkpoint 路径 / loss 曲线截图 / 训练时长
[评估] 50 条结果：准确率 X% / IoU>0.1 Y% / 收敛 Z%
[判定] 通过 / 未通过（未通过项 + 你的分析）
```

## 备选与风险（SFT_PLAN.md §6）

| 情况 | 处理 |
|------|------|
| LLaMA-Factory 对 Qwen3-VL 兼容问题 | 改用 ms-swift：`pip install ms-swift`，`swift sft --model Qwen/Qwen3-VL-8B-Instruct --dataset ... --lora true` |
| 训练 OOM | QLoRA 4bit + grad_accum 提升 |
| 评估不达标 | 报告，候选方案：扩数据（HRBench）、调 LoRA rank/lr、单轮动作空间 |
| 任何不确定 | **先问，不要猜** |
