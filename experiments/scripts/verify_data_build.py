"""CPU 验证:复刻 rl_dataset 数据构建,打印截断后的真实输入构成。

检查点:
  1. processor(1M 像素预算)产出的 input_ids 长度与视觉 token 数
  2. postprocess_data(max_length=5120, truncation=right) 截断后:
     图像占位 token(image_token id 151652)还剩几个 vs grid_thw 乘积
  3. get_rope_index 输出形状(4, seq)
  4. 171 样本的 input_ids 构成分布(文本 vs 视觉 vs 截断)
"""
import json, os, sys

import torch

PROJECT_ROOT = "/root/code"
VERL_ROOT = os.path.join(PROJECT_ROOT, "pyvision-rl", "verl_agents")
for p in (PROJECT_ROOT, VERL_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoProcessor  # noqa: E402
from verl.utils.torch_functional import postprocess_data  # noqa: E402
from verl.models.transformers.qwen2_vl import get_rope_index  # noqa: E402

MODEL = "/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2"
IMG_TOKEN = 151652

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok = proc.tokenizer
print("image_processor size:", proc.image_processor.size)

samples = json.load(open("/root/code/datasets/vstar_bench/rl/train.json"))
stats = {"n": 0, "truncated": 0, "img_token_lt_grid": 0, "over_5120": 0}
rows = []
for i, s in enumerate(samples):
    messages = s["prompt"]
    raw_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images = [s["image"]]
    mi = proc(text=[raw_prompt], images=images, return_tensors="pt")
    input_ids, attn = mi["input_ids"], mi["attention_mask"]
    grid = mi.get("image_grid_thw")
    grid_patch = int(grid[0, 1] * grid[0, 2]) if grid is not None else 0
    img_tok_before = int((input_ids[0] == IMG_TOKEN).sum())
    seq_before = input_ids.shape[1]
    input_ids, attn = postprocess_data(
        input_ids, attn, max_length=5120,
        pad_token_id=tok.pad_token_id, left_pad=True, truncation="right",
    )
    seq = input_ids.shape[1]
    img_tok_after = int((input_ids[0] == IMG_TOKEN).sum())
    stats["n"] += 1
    if seq == 5120:
        stats["truncated"] += 1
    if img_tok_after < grid_patch:
        stats["img_token_lt_grid"] += 1
    text_tok = seq - img_tok_after - int((input_ids[0] == tok.pad_token_id).sum())
    rows.append(dict(i=i, seq=seq, img_tok=img_tok_after, grid=grid_patch,
                     text=text_tok, img0=img_tok_before, seq0=seq_before))
    if i < 5:
        print(f"[{i}] seq0={seq_before} (img_tok0={img_tok_before}, grid={grid_patch}) "
              f"-> seq={seq} img_tok_after={img_tok_after} text≈{text_tok}")

print("\n统计:", stats)
# 文本 token 分布(截断后)
texts = [r["text"] for r in rows]
import statistics
print(f"文本(截后) min={min(texts)} median={statistics.median(texts):.0f} max={max(texts)}")
print(f"图像占位截后分布: min={min(r['img_tok'] for r in rows)} "
      f"median={statistics.median([r['img_tok'] for r in rows]):.0f} "
      f"max={max(r['img_tok'] for r in rows)}")
lt = [r for r in rows if r["img_tok"] < r["grid"]]
print(f"image_token 数 < grid_patch 的样本: {len(lt)}")
for r in lt[:5]:
    print(f"  idx={r['i']} img_tok={r['img_tok']} grid={r['grid']}")
