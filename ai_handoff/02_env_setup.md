# 提示词 2/3 —— 环境配置（发给远端 AI）

目标：把服务器配置成可执行 LLaMA-Factory LoRA SFT 训练的环境。逐步执行，每步汇报结果。

## Step 0：确认 GPU 与磁盘

```bash
nvidia-smi                                # 确认 GPU 型号、数量、显存（预期 ≥1 块，如 2×4090 共 48G）
df -h / /root/autodl-tmp                  # 确认空间（模型 ~16G 放数据盘 autodl-tmp）
```

- 若无 GPU 或 nvidia-smi 异常，**停下来报告**，不要继续
- 记录 GPU 数量 N（后续 CUDA_VISIBLE_DEVICES 和 batch 按 N 调整）

## Step 1：conda 环境

```bash
# 若无 conda：Miniconda 安装见官方文档；AutoDL 镜像通常自带 /root/miniconda3
conda create -n atr python=3.10 -y
conda activate atr
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 必须 True
```

## Step 2：项目依赖

```bash
pip install openai tqdm pillow numpy
pip install -r pyvision-rl/pv_requirements.txt   # 离线评估管线依赖
```

（atr 是源码包，脚本通过 sys.path 引用，**不需要** pip install atr）

## Step 3：LLaMA-Factory

```bash
cd /root/autodl-tmp
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
# torch 已装，避免重装 torch：
pip install -e . --no-deps 2>/dev/null || pip install -e .
# 补齐轻量依赖（若上面 --no-deps 生效了）：
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece
llamafactory-cli version   # 验证安装
```

> 若 pip 慢：`pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`（或清华镜像）

## Step 4：下载模型权重（~16G）

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
ls /root/autodl-tmp/models/Qwen3-VL-8B-Instruct   # 确认 config.json + safetensors 齐全
```

> 镜像失败时试：`pip install modelscope && modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ...`

> ⚠️ **模型目录 `processor_config.json` 已修改（勿回退）**：`size` 从 `longest_edge=16777216`（16M 像素≈无限制）改为 `longest_edge=1003520, shortest_edge=3136`（≈1M 像素，与 SFT `max_image_size: 1024` 同量级）。此修复是 4 卡视觉 token OOM 的根因解决方案，只存在于远端模型目录、不进 git——**换实例/重装模型后必须重新打这个补丁**，否则 RL 视觉 token 每样本 9964-32400、必然 OOM。改前先 `git diff` 对比 06 文档「回报 5」里的标准内容。

## Step 5：注册 SFT 数据集

```bash
# 把项目里 experiments/configs/dataset_info_vstar.json 的内容合并进 LLaMA-Factory/data/dataset_info.json
# （下方即该文件的完整内容）
```

注册内容（追加到 `data/dataset_info.json` 的 datasets 字典里）：

```json
{
  "vstar_tool_sft_train": {
    "file_name": "vstar_tool_sft/train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"}
  },
  "vstar_tool_sft_val": {
    "file_name": "vstar_tool_sft/val.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"}
  }
}
```

把数据放到位并**处理图片路径**。先记下项目解压路径 `$PROJECT`（例如 `/root/code`）。JSONL 内图片路径是相对项目根的 `datasets/vstar_bench/sft_images/...`，二选一：

```bash
# 方案 A（推荐）：整个项目保持原结构，从项目根目录运行训练
mkdir -p /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft
cp $PROJECT/datasets/vstar_bench/sft/train.jsonl \
   $PROJECT/datasets/vstar_bench/sft/val.jsonl \
   /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/

# 方案 B：把图片路径改成本机绝对路径（$PROJECT 换成实际路径）
sed -i "s|datasets/vstar_bench/sft_images|$PROJECT/datasets/vstar_bench/sft_images|g" \
    /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/*.jsonl
```

> 若 JSONL 里 `"image"` 是相对路径（如 `datasets/vstar_bench/direct_attributes/xx.jpg` 的原始问题图），同样按方案 B 处理前缀。

## Step 6：验证

```bash
wc -l /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/train.jsonl   # 期望 392
wc -l /root/autodl-tmp/LLaMA-Factory/data/vstar_tool_sft/val.jsonl     # 期望 43
python -c "from transformers import AutoModelForVision2Seq, AutoProcessor; \
m=AutoModelForVision2Seq.from_pretrained('/root/autodl-tmp/models/Qwen3-VL-8B-Instruct', trust_remote_code=True); \
print('model load OK', sum(p.numel() for p in m.parameters())/1e9, 'B')"
```

## 完成标准

以上 Step 0-6 全部通过。汇报：GPU 型号/数量、conda env 名、LLaMA-Factory 版本、模型路径、数据注册情况，然后索要第三份提示词。
