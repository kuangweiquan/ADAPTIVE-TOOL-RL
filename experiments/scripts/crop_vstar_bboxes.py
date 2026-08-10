#!/usr/bin/env python3
"""
按 VStar 标注 bbox 截取目标区域,方便人工核对轨迹时比对 ground-truth 区域。

VStar 标注格式(注意 bbox 是 [x1, y1, w, h],不是 [x1, y1, x2, y2]):
    {
      "target_object": ["little girl's shirt"],
      "bbox": [[1682, 1175, 28, 31]],
      "question": "...",
      "options": ["(正确答案)", "...", ...]
    }

输出:
    datasets/vstar_bench/crops/<test_type>/<sample>.jpg          # 单个 bbox
    datasets/vstar_bench/crops/<test_type>/<sample>_<i>.jpg      # 多个 bbox 时
    datasets/vstar_bench/crops/index.jsonl                       # 裁剪索引(供人工核对)

用法:
  python experiments/scripts/crop_vstar_bboxes.py [--vstar_path datasets/vstar_bench]
"""

import os
import sys
import json
import argparse
from PIL import Image

VSTAR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "vstar_bench")
TEST_TYPES = ["direct_attributes", "relative_position"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop VStar bbox regions for manual verification")
    parser.add_argument("--vstar_path", type=str, default=VSTAR_DEFAULT)
    args = parser.parse_args()

    vstar = os.path.abspath(args.vstar_path)
    crops_dir = os.path.join(vstar, "crops")
    index_path = os.path.join(crops_dir, "index.jsonl")
    os.makedirs(crops_dir, exist_ok=True)

    n_total = 0
    n_missing_img = 0
    n_no_bbox = 0
    n_multi = 0
    with open(index_path, "w", encoding="utf-8") as idx:
        for test_type in TEST_TYPES:
            test_dir = os.path.join(vstar, test_type)
            if not os.path.isdir(test_dir):
                print(f"[skip] {test_dir} 不存在")
                continue
            out_dir = os.path.join(crops_dir, test_type)
            os.makedirs(out_dir, exist_ok=True)

            jsons = sorted(f for f in os.listdir(test_dir) if f.endswith(".json"))
            for jf in jsons:
                base = os.path.splitext(jf)[0]
                img_path = os.path.join(test_dir, f"{base}.jpg")
                if not os.path.isfile(img_path):
                    img_path = os.path.join(test_dir, f"{base}.png")
                if not os.path.isfile(img_path):
                    n_missing_img += 1
                    print(f"[warn] {jf} 缺少图片文件,跳过")
                    continue

                anno = json.load(open(os.path.join(test_dir, jf), encoding="utf-8"))
                bboxes = anno.get("bbox", [])
                if not bboxes:
                    n_no_bbox += 1
                    print(f"[warn] {jf} 无 bbox 标注")
                    continue

                img = Image.open(img_path).convert("RGB")
                W, H = img.size

                for i, bb in enumerate(bboxes):
                    if len(bb) != 4:
                        print(f"[warn] {jf} bbox[{i}] 长度异常: {bb}")
                        continue
                    x1, y1, w, h = map(int, bb)
                    # clamp 到图像边界(VStar 标注可能越界)
                    x1c = max(0, min(x1, W))
                    y1c = max(0, min(y1, H))
                    x2c = max(0, min(x1 + w, W))
                    y2c = max(0, min(y1 + h, H))
                    if x2c <= x1c or y2c <= y1c:
                        print(f"[warn] {jf} bbox[{i}] 越界/空区域: {bb},图像 {W}x{H}")
                        continue

                    crop = img.crop((x1c, y1c, x2c, y2c))
                    crop_name = f"{base}.jpg" if len(bboxes) == 1 else f"{base}_{i}.jpg"
                    crop_path = os.path.join(out_dir, crop_name)
                    crop.save(crop_path, "JPEG", quality=95)

                    record = {
                        "crop_path": os.path.relpath(crop_path, vstar),
                        "test_type": test_type,
                        "source_image": f"{base}.jpg",
                        "annotation_file": jf,
                        "bbox_xywh": [x1, y1, w, h],
                        "bbox_clamped": [x1c, y1c, x2c, y2c],
                        "target_object": anno.get("target_object"),
                        "question": anno.get("question"),
                        "ground_truth": anno["options"][0] if anno.get("options") else None,
                    }
                    idx.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_total += 1
                    if len(bboxes) > 1:
                        n_multi += 1

    print(f"\n完成: 共裁剪 {n_total} 个 bbox 区域 → {crops_dir}")
    print(f"  (多 bbox 样本: {n_multi} | 缺图: {n_missing_img} | 无 bbox: {n_no_bbox})")
    print(f"索引文件: {index_path}")

    # 展示几个例子
    print("\n示例(前 5 条):")
    with open(index_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            r = json.loads(line)
            print(f"  {r['crop_path']} ← {r['source_image']} | Q: {r['question'][:50]} | GT: {r['ground_truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
