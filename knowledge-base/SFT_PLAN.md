# Qwen3-VL-8B 工具能力 SFT 冷启动方案

> 背景:实验证明 8B 零样本跑不动工具循环(必调工具、乱定位、zoom 死循环、答案随机),
> 而 397B 在相同接口(强制工具 + 归一化坐标)下达 80% 正确率、100% 真实工具调用。
> 方案:用 397B 生成的高质量工具轨迹做 8B 的 SFT 冷启动(CodeV 标准做法),
> 让 8B 学会"工具调用格式 + 归一化定位 + 收敛作答",再进入 ATR-RL。

---

## 0. 决策与目标

| 项 | 值 |
|----|-----|
| 训练目标模型 | **Qwen3-VL-8B-Instruct**(不用 Qwen2.5-VL-7B) |
| SFT 目的 | ① 工具调用格式(JSON + 归一化 bbox)② 归一化定位 ③ 收敛作答(直接答 or 工具核实后答) |
| 教学数据来源 | Qwen3.5-397B-A17B 生成(已验证:80% 正确、工具真实) |
| 训练方式 | LoRA(2×4090,48GB)→ 合并 → vLLM 评估 |
| 后续 | verl + GRPO + ATR reward(patch_reward 已就绪,接口不变) |

**RL 推理时的统一 prompt = 先答后验策略**(工具可用 + 先直接作答)。SFT 数据混合两类:
- 工具轨迹(教"何时用工具、怎么用")——来自 397B 强制工具模式
- 直接作答轨迹(教"何时不用工具")——来自 397B 先答后验模式

这直接针对 8B 的两个崩溃点:**工具必调偏置**(用直接作答样本纠正)和 **zoom 死循环**(用正确的工具→作答样本纠正)。

---

## 1. 数据采集(397B 生成,一次性成本)

### 1.1 工具轨迹(强制工具模式)

```
VStar 全 191 样本 × 3 个温度种子(0.7 / 0.9 / 1.1)≈ 573 条
```

- 命令:`ATR_MODEL=Qwen/Qwen3.5-397B-A17B python run_atr_offline.py --vstar_path datasets/vstar_bench --tool_required --output_dir ...`
- 多温度增广:修改 `run_atr_offline.py` 的 `TEMPERATURE` 为参数(小改:加 `--temperature` CLI 或环境变量)
- **过滤条件**(写入过滤脚本):
  - `status == "success"`(排除 max_turns_exceeded/error)
  - `accuracy == 1.0`(只教正确答案走向)
  - 工具调用真实(复用 `verify_real_tool_calls.py` 的判定逻辑)
  - **GT 覆盖率 ≥ 50%**(证据:20 条审计中 success 样本 94% 满足,加严只多排除 1 条,教学数据更纯)
  - 预期保留率 ~60-70% → **~350-400 条**

### 1.2 直接作答轨迹(先答后验模式,教"不调工具")

```
VStar 全 191 样本 × 1 种子(temperature 0.0)≈ 191 条
```

- 默认模式跑一遍,过滤 `accuracy == 1.0` → **~75 条**

### 1.3 数据量决策

| 类别 | 目标量 | 作用 |
|------|--------|------|
| 工具轨迹(正确) | ~350 | 工具格式 + 定位 + 工具→作答 |
| 直接作答(正确) | ~75 | 何时不调工具(抑制工具必调偏置) |
| **合计** | **~425** | 8B 格式级冷启动足够(CodeV 同类 SFT 量级) |

若数据量不足或需要更强效果,扩展:HRBench/VSI-Bench(README 已列,需确认数据可得性)。

---

## 2. 数据格式转换(轨迹 JSONL → SFT 对话 JSONL)

新建 `experiments/scripts/trajectories_to_sft.py`。

### 2.1 消息重建(关键:工具响应图像重放)

轨迹 JSONL **未存**模型当时的完整响应文本和工具响应图像,需确定性重建:

1. **逐条重放状态机**:用 `atr.tools`(execute + ToolEnv 同款逻辑)按记录重新执行,
   生成与模型当时所见一致的显示图(`resize_for_display`,DISPLAY_MAX=1024)
2. **assistant 工具消息重建**:记录的 arguments 是**执行后像素空间**,
   按该步 `current_image.size` 反推归一化:
   `normalized = pixel / size`,重建 `<tool_call>{"name": X, "arguments": {"bbox_2d": [norm...]}}</tool_call>`
   (教给模型的就是归一化接口)
3. **user 工具响应消息**:记录的 output 文本 + 重放的显示图(与 `run_atr_offline.py`
   的 tool_response 构造一致:`<tool_response>` + image + output + `</tool_response>`)
4. **最终 assistant 消息**:`<answer>{predicted_answer}</answer>`
5. 直接作答样本:仅两轮 [user → assistant <answer>]

### 2.2 统一 system prompt

所有样本使用 **先答后验** prompt(与 RL 推理一致):`build_system_prompt(tool_required=False)`。

### 2.3 输出格式(LLaMA-Factory multimodal 格式)

```json
{"messages": [
  {"role": "system", "content": "<先答后验 system prompt>"},
  {"role": "user", "content": [
      {"type": "image", "image": "datasets/vstar_bench/direct_attributes/sa_17.jpg"},
      {"type": "text", "text": "Question: ...\nOptions: ...\nAnswer directly ..."}]},
  {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
  {"role": "user", "content": [{"type": "image", "image": "<重放的显示图路径>"},
                                {"type": "text", "text": "<tool_response>...</tool_response>"}]},
  {"role": "assistant", "content": "<answer>A</answer>"}
]}
```

- 重放图落盘 `datasets/vstar_bench/sft_images/<sample>/step<i>.jpg`,JSONL 引用路径
- 90/10 切分 train/val

---

## 3. 训练(LLaMA-Factory,2×4090)

### 3.1 环境

```bash
# LLaMA-Factory(2026 版已支持 Qwen3-VL)
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e .[torch,bitsandbytes]
# 模型:Qwen3-VL-8B-Instruct(HF 权重,需下载或从 SiliconFlow 镜像)
```

### 3.2 LoRA 配置(48GB 双卡)

| 参数 | 值 |
|------|-----|
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| 目标模块 | all linear(q,k,v,o,gate,up,down) |
| 学习率 / 调度 | 2e-5 / cosine,warmup 0.03 |
| epochs / 截断 | 3 / 8192 tokens(多轮+图) |
| batch | per_device 2 × 2 卡,grad_accum 8(~32 等效) |
| 精度 | bf16(4090 支持) |
| 最大图像分辨率 | 1024(与管线 DISPLAY_MAX 一致) |

### 3.3 训练命令(yaml 要点)

```yaml
model_name_or_path: Qwen3-VL-8B-Instruct
dataset: vstar_tool_sft           # 上面转换的 JSONL,注册进 dataset_info.json
template: qwen2_vl                # LLaMA-Factory 的 Qwen-VL 模板
finetuning_type: lora
lora_target: all
cutoff_len: 8192
learning_rate: 2e-5
num_train_epochs: 3.0
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
bf16: true
```

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train sft_qwen3vl_lora.yaml
llamafactory-cli export merge_lora.yaml   # 合并 LoRA → Qwen3-VL-8B-ATR-SFT
```

### 3.4 备选框架

LLaMA-Factory 若对 Qwen3-VL 支持有问题 → **ms-swift**(Qwen 官方生态,对 Qwen3-VL 支持最稳)。

---

## 4. 验证(SFT 后工具循环测试)

### 4.1 部署 + 评估

```bash
# vLLM 起本地服务(1 卡即可)
vllm serve Qwen3-VL-8B-ATR-SFT --port 8000 --max-model-len 8192

# 离线管线指向本地端点(OpenAI 兼容)
export ATR_MODEL="Qwen3-VL-8B-ATR-SFT"
export SILICONFLOW_API_URL="http://localhost:8000/v1"
export SILICONFLOW_API_KEY="dummy"
python run_atr_offline.py --vstar_path datasets/vstar_bench --quick 50 \
    --tool_required --output_dir experiments/results/eval_sft
```

### 4.2 通过门槛(与 8B 零样本对照)

| 指标 | 8B 零样本(对照) | SFT 后门槛 |
|------|-----------------|-----------|
| 正确率(50 条) | 6.6%(强制) | **≥ 50%**(显著>随机 33.8%) |
| 工具调用真实 | ✓(但乱定位) | ✓ + **IoU > 0.1 比例 ≥ 30%** |
| 收敛性 | 44/61 死循环 | **≥ 80% 样本 ≤ 3 轮作答** |
| 直接作答率 | 0%(必调) | 允许 0-40%(先答后验混合) |

不满足门槛 → 回到第 1 步扩数据或调 LoRA,再验。

### 4.3 全量数据回归(可选)

SFT 后再生成 100 条新轨迹,跑 ATR 奖励分析(`--analyze_only`),确认奖励分布合理。

---

## 5. 进入 RL(衔接,无需改动)

1. pyvision-rl 训练管线(verl + GRPO),reward 用 `atr/adapter/patch_reward.py` 的 `ATRRewardManager`(已就绪,`AdaptiveToolReward.compute()` 接口不变)
2. 训练 prompt = 先答后验(system prompt 与 SFT 一致,`build_system_prompt()`)
3. ATR 奖励:acc + λU − γC + ηS,utility 空间精度项可传入 VStar GT bbox(`ground_truth["gt_bbox"]`)启用
4. 冷启动权重 = SFT checkpoint(LoRA 合并后),不用原版 8B

---

## 6. 风险与备选

| 风险 | 备选 |
|------|------|
| SFT 后 8B 仍不收敛(数据量/模型能力) | ① 扩数据(HRBench 等)② 单轮动作空间:bbox 提议+答案同轮,绕开多轮 |
| 397B 数据生成成本(573 条 × ~20s) | ~3-4 小时,一次性;可先 191×2 种子(→~250 条)验证 |
| LLaMA-Factory 对 Qwen3-VL 兼容性 | ms-swift 备选 |
| 2×4090 训练 OOM | QLoRA(4bit)+ grad_accum 提升 |
| VStar options[0]=GT 约定导致 A 偏置混入准确率 | 论文披露;评估时用打乱选项顺序交叉验证 |
| 多温度增广需要脚本改动 | 给 `TEMPERATURE` 加 CLI 参数(小改) |

---

## 7. 执行顺序(依赖关系)

1. `run_atr_offline.py` 加 `--temperature` 参数(小改)
2. 397B 数据采集:强制工具 191×3 种子 + 先答后验 191(t≈4h,后台)
3. 过滤脚本:success + correct + 真实调用 → 生成 SFT 数据清单
4. `trajectories_to_sft.py`:重放 + 归一化反推 + 消息重建 → train/val JSONL
5. LLaMA-Factory LoRA SFT(≈1-2h/卡)
6. 合并 → vLLM → 50 条工具循环评估 → 对照门槛
7. 通过 → 接入 verl + ATR-RL
