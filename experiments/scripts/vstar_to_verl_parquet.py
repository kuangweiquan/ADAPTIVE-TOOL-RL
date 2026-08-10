#!/usr/bin/env python3
"""vstar_bench → verl RL 数据集(parquet)。

对接 verl_agents 的 RLHFDataset(with_mm_hint=True)与 VStarToolEnv:

  - prompt:      chat 消息列表(system = atr.prompts 先答后验版;
                 user 含 <image> 占位 + 问题 + 选项)
  - mm_hint:     {"hint_type": "image", "hint_path": <绝对路径>}
  - env_name:    "vstar_tool_env"(ToolBase 注册名,agent rollout 用)
  - data_source: direct_attributes / relative_position
  - reward_model:{"ground_truth": options[0], "style": "model"}
  - extra_info:  {question, options, gt_bbox(归一化 [x1,y1,x2,y2]),
                  image_size(原始像素 [W,H]), index}
  - uid:         样本唯一 id

GT bbox 约定(VStar 标注 [x, y, w, h] 像素):
  - 取 bbox[0](第一个目标物体) → [x1,y1,x2,y2] → 除以原图 (W, H) 归一化
  - 模型在初始视图输出归一化坐标,与 gt_bbox 同空间 → utility IoU 匹配有效
  - 注意:zoom 之后模型坐标属于缩放视图,与原始图 GT 不再同空间
    (仅首轮定位有 IoU 奖励;渐进细化靠 utility 的 refinement 奖励)

用法:
  python vstar_to_verl_parquet.py --vstar_path datasets/vstar_bench \
      --output_dir datasets/vstar_bench/rl \
      --image_root /root/code/datasets/vstar_bench   # 远端路径时改写 hint_path

依赖: pyarrow pandas(PIL 仅用于取图尺寸)
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from atr.prompts import build_system_prompt, build_user_prompt

TEST_TYPES = ["direct_attributes", "relative_position"]


def anno_to_gt_bbox(anno: Dict[str, Any], img_size: Tuple[int, int]):
    """VStar 标注 [x, y, w, h] → 归一化 [x1, y1, x2, y2](取第一个 bbox)。"""
    W, H = img_size
    boxes = anno.get("bbox") or []
    if not boxes:
        return None
    x, y, w, h = map(float, boxes[0])
    return [x / W, y / H, (x + w) / W, (y + h) / H]


def discover_samples(vstar_path: str) -> List[Tuple[str, str, Dict]]:
    """(image_path, image_file, anno) 列表,与离线评估 discover_vstar_samples 一致。"""
    samples = []
    for test_type in TEST_TYPES:
        test_dir = os.path.join(vstar_path, test_type)
        if not os.path.isdir(test_dir):
            print(f"  [Warning] {test_dir} not found, skipping")
            continue
        for f in sorted(os.listdir(test_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(f)[0]
            json_path = os.path.join(test_dir, f"{base}.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path, encoding="utf-8") as fh:
                anno = json.load(fh)
            samples.append((os.path.join(test_dir, f), f, anno))
    return samples


def build_row(
    image_path: str,
    img_file: str,
    anno: Dict[str, Any],
    test_type: str,
    image_root: str,
    index: int,
) -> Dict[str, Any]:
    question = anno["question"]
    options = anno["options"]
    ground_truth = options[0]  # VStar 约定:第一个选项正确

    with Image.open(image_path) as img:
        W, H = img.size

    # hint_path 必须是远端可读的绝对路径(训练时由 mm_hint 加载)
    hint_path = os.path.join(image_root, test_type, img_file)
    if not os.path.isfile(hint_path):
        # image_root 缺省时退回样本实际路径
        hint_path = image_path

    user_content = "<image>\n" + build_user_prompt(question, options)
    prompt = [
        {"role": "system", "content": build_system_prompt(tool_required=False)},
        {"role": "user", "content": user_content},
    ]
    return {
        "prompt": prompt,
        "mm_hint": {"hint_type": "image", "hint_path": hint_path},
        "data_source": test_type,
        "env_name": "vstar_tool_env",
        "reward_model": {"ground_truth": ground_truth, "style": "model"},
        "ground_truth": ground_truth,
        "extra_info": {
            "question": question,
            "options": options,
            "gt_bbox": anno_to_gt_bbox(anno, (W, H)),
            "image_size": [W, H],
            "index": index,
        },
        "uid": f"{test_type}/{img_file}",
    }


def main():
    parser = argparse.ArgumentParser(description="vstar_bench → verl RL parquet")
    parser.add_argument("--vstar_path", required=True, help="vstar_bench 目录")
    parser.add_argument("--output_dir", required=True, help="parquet 输出目录")
    parser.add_argument("--image_root", default=None,
                        help="hint_path 前缀(远端路径,如 /root/code/datasets/vstar_bench;默认 vstar_path)")
    parser.add_argument("--val_size", type=int, default=20, help="验证集条数(种子抽样)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_root = args.image_root or args.vstar_path

    samples = discover_samples(args.vstar_path)
    print(f"Found {len(samples)} samples: {[t for t in TEST_TYPES]}")

    rows = []
    for i, (image_path, img_file, anno) in enumerate(samples):
        test_type = next(t for t in TEST_TYPES if image_path.replace("\\", "/").split("/")[-2] == t)
        rows.append(build_row(image_path, img_file, anno, test_type, image_root, i))

    random.seed(args.seed)
    random.shuffle(rows)
    val = rows[: args.val_size]
    train = rows[args.val_size:]

    table = pa.Table.from_pylist(train)
    pq.write_table(table, os.path.join(args.output_dir, "train.parquet"))
    pq.write_table(pa.Table.from_pylist(val), os.path.join(args.output_dir, "val.parquet"))
    print(f"train={len(train)} val={len(val)} → {args.output_dir}/{{train,val}}.parquet")

    # 展示一条(校验格式)
    r = train[0]
    print("\n--- sample row ---")
    print("uid:", r["uid"], "| data_source:", r["data_source"])
    print("env_name:", r["env_name"])
    print("mm_hint:", r["mm_hint"])
    print("ground_truth:", r["ground_truth"])
    print("gt_bbox(normalized):", r["extra_info"]["gt_bbox"])
    print("prompt[0].role:", r["prompt"][0]["role"], "| prompt[1] head:",
          r["prompt"][1]["content"][:60].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
