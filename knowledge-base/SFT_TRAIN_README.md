# SFT 训练执行手册(LLaMA-Factory,2×4090)

> 数据采集/过滤/转换在 Windows 开发机完成;本手册为 GPU 服务器(2×4090)训练步骤。
> 计划见 [SFT_PLAN.md](SFT_PLAN.md) §3-4。

## 1. 环境安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,bitsandbytes]"
# 模型权重:Qwen/Qwen3-VL-8B-Instruct(HF 或镜像下载)
```

## 2. 数据注册

把转换产物的 `train.jsonl` / `val.jsonl` 拷到 `LLaMA-Factory/data/vstar_tool_sft/`。
随压缩包已提供注册片段 `experiments/configs/dataset_info_vstar.json`,
把它合并进 `data/dataset_info.json` 即可(内容同下):

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

**重要**:JSONL 内图像路径是相对项目根的(`datasets/vstar_bench/sft_images/...`)。
训练前把 `datasets/` 整个拷到服务器上,或在服务器上把路径前缀改成绝对路径:
```bash
# 例:把 sft_images 放在 /data/vstar_sft/ 下
sed -i 's|datasets/vstar_bench/sft_images|/data/vstar_sft/sft_images|g' \
    data/vstar_tool_sft/train.jsonl data/vstar_tool_sft/val.jsonl
```

## 3. 训练

```bash
# 配置文件在本仓库 experiments/configs/(拷到服务器后按需调整路径)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train sft_qwen3vl_lora.yaml

# 合并 LoRA → Qwen3-VL-8B-ATR-SFT
llamafactory-cli export merge_lora.yaml
```

若 LLaMA-Factory 对 Qwen3-VL 支持异常(如模板/图像参数报错),改用
**ms-swift**(Qwen 官方生态):`swift sft --model Qwen/Qwen3-VL-8B-Instruct
--dataset <转换后的格式> --lora true ...`(计划 §3.4 备选)。

## 4. 评估(SFT 后工具循环,计划 §4)

```bash
# vLLM 起本地服务(1 卡即可,OpenAI 兼容)
vllm serve /path/to/Qwen3-VL-8B-ATR-SFT --port 8000 --max-model-len 8192

# 离线管线指向本地端点
export ATR_MODEL="Qwen3-VL-8B-ATR-SFT"
export SILICONFLOW_API_URL="http://localhost:8000/v1"
export SILICONFLOW_API_KEY="dummy"
python experiments/scripts/run_atr_offline.py --vstar_path datasets/vstar_bench \
    --quick 50 --tool_required --output_dir experiments/results/eval_sft
```

通过门槛(与 8B 零样本对照):
| 指标 | 门槛 |
|------|------|
| 正确率(50 条) | ≥ 50% |
| IoU > 0.1 比例 | ≥ 30% |
| ≤3 轮作答比例 | ≥ 80% |
| 直接作答率 | 0-40% 可接受 |

不满足 → 回 §1 扩数据或调 LoRA 再验。

## 5. 数据生成方(Windows 开发机,已完成部分)

| 步骤 | 命令 | 输出 |
|------|------|------|
| 采集 | `nohup bash experiments/scripts/collect_sft_data.sh &` | `experiments/results/sft_collect/`(4 目录) |
| 过滤 | `python experiments/scripts/filter_sft_trajectories.py ...` | `sft_candidates.jsonl` |
| 转换 | `python experiments/scripts/trajectories_to_sft.py --input sft_candidates.jsonl` | `datasets/vstar_bench/sft/{train,val}.jsonl` + `sft_images/` |

过滤/转换脚本用法详见各自 docstring。
