#!/usr/bin/env python3
"""
从轨迹文件中挑选 10 条代表性样本,整理成便于人工核对的文件夹:
    crops/trajectory_check/<sample>/
        ├── 00_original.jpg          # 原始图片
        ├── 01_gt_bbox_crop.jpg      # 标注 bbox 裁剪(ground truth 区域)
        ├── 02_tool_<i>_<tool>.jpg   # 模型实际工具调用的 bbox 区域(从原图裁)
        ├── 03_overlay.jpg           # 原图 + GT bbox(绿) + 工具 bbox(红,带序号)
        ├── <sample>_trajectory.json # 该样本轨迹(展开的多行 JSON)
        └── ...(多个 GT bbox 时多个 01_ 文件)
    crops/trajectory_check/README.md  # 10 条样本一览表

挑选标准:覆盖 crop/zoom/rotate/ocr 四工具、真实 OCR 样本优先、
正确/错误混合、工具序列多样性。

用法:
  python experiments/scripts/select_trajectories_for_check.py \
      [--trajectories experiments/results/run_20260805/trajectories_20260805_175118.jsonl] \
      [--crops_dir datasets/vstar_bench/crops] [--n 10]
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DEFAULT_TRAJ = os.path.join(PROJECT_ROOT, "experiments", "results", "run_20260805",
                            "trajectories_20260805_175118.jsonl")
DEFAULT_CROPS = os.path.join(PROJECT_ROOT, "datasets", "vstar_bench", "crops")
DEFAULT_N = 10


def load_trajectories(path: str) -> list:
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return [json.loads(l) for l in f if l.strip()]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码 {path}")


def load_gt_index(crops_dir: str) -> dict:
    """crops 索引 → {test_type/name: [记录...]}"""
    idx = {}
    idx_path = os.path.join(crops_dir, "index.jsonl")
    if os.path.isfile(idx_path):
        for line in open(idx_path, encoding="utf-8"):
            r = json.loads(line)
            idx.setdefault(r["test_type"] + "/" + os.path.splitext(r["source_image"])[0], []).append(r)
    return idx


def pick_samples(trajectories: list, n: int) -> list:
    """挑选代表性样本:真实 OCR 优先,其次 rotate,正确/错误均衡,工具多样。"""
    def toolset(t):
        return tuple(sorted({c["tool_name"] for c in t.get("tool_calls", [])}))

    def score(t):
        tools = toolset(t)
        s = 0.0
        if "ocr" in tools:
            s += 100  # OCR 样本稀有,优先
        if "rotate" in tools:
            s += 10
        if len(tools) >= 2:
            s += 3
        n_calls = len(t.get("tool_calls", []))
        if 2 <= n_calls <= 5:
            s += 2
        if t.get("status") == "success" and t.get("predicted_answer"):
            s += 1
        return s

    ranked = sorted(trajectories, key=score, reverse=True)
    picked, seen_tools, n_correct, n_wrong = [], set(), 0, 0
    for t in ranked:
        tools = toolset(t)
        if len(picked) >= n:
            break
        # 均衡:正确/错误各不超过 n//2 + 1
        if t.get("accuracy", 0) > 0.5:
            if n_correct >= n // 2 + 1:
                continue
            n_correct += 1
        else:
            if n_wrong >= n // 2 + 1:
                continue
            n_wrong += 1
        picked.append(t)
        seen_tools.update(tools)
    # 若工具覆盖不完整(如没选到 ocr),补选一个
    for t in ranked:
        if len(picked) >= n:
            break
        if t in picked:
            continue
        picked.append(t)
    return picked


def draw_overlay(orig: Image.Image, gt_boxes: list, tool_boxes: list, out_path: str) -> None:
    """GT bbox(绿) + 工具 bbox(红,带序号)画在原图上。"""
    draw = ImageDraw.Draw(orig)
    for x1, y1, x2, y2 in gt_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
    for i, (x1, y1, x2, y2) in enumerate(tool_boxes):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, max(0, y1 - 16)), f"T{i}", fill="red")
    orig.save(out_path, "JPEG", quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default=DEFAULT_TRAJ)
    parser.add_argument("--crops_dir", default=DEFAULT_CROPS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    traj_path = os.path.abspath(args.trajectories)
    crops_dir = os.path.abspath(args.crops_dir)
    vstar = os.path.dirname(crops_dir)

    trajectories = load_trajectories(traj_path)
    gt_index = load_gt_index(crops_dir)
    picked = pick_samples(trajectories, args.n)

    out_root = os.path.join(crops_dir, "trajectory_check")
    os.makedirs(out_root, exist_ok=True)

    summary = []
    for t in picked:
        name = os.path.splitext(t["image"])[0]
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) 原始图片
        test_type = "direct_attributes" if os.path.isfile(os.path.join(vstar, "direct_attributes", t["image"])) \
            else "relative_position"
        src_path = os.path.join(vstar, test_type, t["image"])
        orig = Image.open(src_path).convert("RGB")
        orig.save(os.path.join(out_dir, "00_original.jpg"), "JPEG", quality=92)

        # 2) GT bbox 裁剪
        gt_records = gt_index.get(f"{test_type}/{name}", [])
        for i, rec in enumerate(gt_records):
            x1, y1, w, h = rec["bbox_xywh"]
            crop = orig.crop((x1, y1, x1 + w, y1 + h))
            crop.save(os.path.join(out_dir, f"01_gt_bbox_crop_{i}.jpg"), "JPEG", quality=95)

        # 3) 模型实际工具调用区域(从原图裁;旋转/缩放后的 bbox 为近似位置)
        tool_boxes = []
        for i, c in enumerate(t.get("tool_calls", [])):
            bbox = c.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                tool_boxes.append((x1, y1, x2, y2))
                if x2 > x1 and y2 > y1 and x2 <= orig.width and y2 <= orig.height:
                    crop = orig.crop((x1, y1, x2, y2))
                    crop.save(os.path.join(out_dir, f"02_tool_{i}_{c['tool_name']}.jpg"), "JPEG", quality=95)

        # 4) 叠加标注图
        draw_overlay(orig.copy(), [tuple(r["bbox_clamped"]) for r in gt_records], tool_boxes,
                     os.path.join(out_dir, "03_overlay.jpg"))

        # 5) 展开的轨迹 JSON
        with open(os.path.join(out_dir, f"{name}_trajectory.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2, ensure_ascii=False)

        summary.append({
            "dir": f"crops/trajectory_check/{name}",
            "image": t["image"],
            "acc": t["accuracy"],
            "tools": [c["tool_name"] for c in t.get("tool_calls", [])],
            "question": t["question"][:70],
            "ground_truth": t.get("ground_truth", "")[:50],
            "predicted": str(t.get("predicted_answer", ""))[:50],
        })

    # README 一览表
    with open(os.path.join(out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Trajectory Check — 人工核对一览\n\n")
        f.write("| # | 文件夹 | acc | 工具序列 | 问题 | GT | 预测 |\n")
        f.write("|---|--------|-----|----------|------|----|------|\n")
        for i, s in enumerate(summary):
            f.write(f"| {i} | {s['dir']} | {s['acc']} | {','.join(s['tools']) or '-'} "
                    f"| {s['question']} | {s['ground_truth']} | {s['predicted']} |\n")
        f.write("\n> 说明:01_ 为标注 GT bbox 区域;02_ 为模型工具调用区域(从原图裁,"
                "旋转/缩放后的调用坐标为近似);03_overlay.jpg 绿框=GT、红框=模型调用。\n")

    print(f"已生成 {len(picked)} 个样本 → {out_root}")
    for i, s in enumerate(summary):
        tools = ",".join(s["tools"]) or "-"
        print(f"  [{i}] {s['image']} acc={s['acc']} tools=[{tools}]")
    print(f"一览表: {os.path.join(out_root, 'README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
