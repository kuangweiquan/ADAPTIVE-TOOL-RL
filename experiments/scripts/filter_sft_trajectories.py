#!/usr/bin/env python3
"""
SFT 数据过滤(计划 §1.1/§1.2):从 397B 采集的轨迹中筛选教学数据。

工具轨迹(tool_required 文件)过滤条件:
  1. status == "success"(排除 max_turns_exceeded / error)
  2. accuracy == 1.0(只教正确答案走向)
  3. 工具调用真实(与 verify_real_tool_calls.py 判定逻辑一致:
     规范工具名、output 非 [Error/[Unknown、空间工具带有效 bbox)
  4. 至少 1 次真实调用,且 **GT 覆盖率 ≥ 50%**:
     coverage(调用 bbox, GT bbox) = 交集面积 / GT 面积,取所有原图空间调用
     × 所有 GT bbox 的最大值。全图调用(bbox 覆盖整幅图)不参与计算
     —— 全图 OCR/全图区域不算"定位到了目标"。
     (zoom/rotate 会切换视图,之后的调用坐标不再与原图可比,不参与。)

直接作答轨迹(先答后验文件)过滤条件:
  1. status == "success"
  2. accuracy == 1.0
  3. 工具调用数 == 0(纯直接作答样本,教"何时不调工具")

用法:
  python experiments/scripts/filter_sft_trajectories.py \
      --input experiments/results/sft_collect/toolreq_t07/trajectories_*.jsonl \
      --input experiments/results/sft_collect/toolreq_t09/trajectories_*.jsonl \
      --input experiments/results/sft_collect/toolreq_t11/trajectories_*.jsonl \
      --input experiments/results/sft_collect/answerfirst/trajectories_*.jsonl \
      --output experiments/results/sft_collect/sft_candidates.jsonl \
      --tool_required experiments/results/sft_collect/toolreq_t07/trajectories_*.jsonl
      --tool_required experiments/results/sft_collect/toolreq_t09/trajectories_*.jsonl
      --tool_required experiments/results/sft_collect/toolreq_t11/trajectories_*.jsonl
      --vstar_path datasets/vstar_bench
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

VALID_TOOLS = {"crop", "zoom", "rotate", "ocr"}
STATE_TOOLS = {"zoom", "rotate"}   # 切换视图的工具(之后坐标空间不可比)
MIN_GT_COVERAGE = 0.5


def load_trajectories(path: str) -> list:
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return [json.loads(l) for l in f if l.strip()]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码 {path}")


def is_real_call(call: dict) -> bool:
    """判定一次调用是否真实执行(与 verify_real_tool_calls.py 一致)。"""
    name = call.get("tool_name", "")
    out = str(call.get("output", ""))
    if name not in VALID_TOOLS:
        return False
    if out.startswith("[Unknown tool") or out.startswith("[Error"):
        return False
    if name in ("crop", "zoom"):
        bbox = call.get("bbox")
        return (isinstance(bbox, list) and len(bbox) == 4
                and bbox[2] > bbox[0] and bbox[3] > bbox[1])
    if name == "rotate":
        return out.startswith("[Rotated by") and call.get("bbox") is None
    if name == "ocr":
        return out.startswith("[OCR result") or out == "[No text detected in region]"
    return False


def load_gt_bboxes(vstar_path: str, image_file: str) -> list:
    """VStar 标注 bbox([x1,y1,w,h])→ [x1,y1,x2,y2],按图片名在两个 test_type 中查找。"""
    for test_type in ("direct_attributes", "relative_position"):
        json_path = os.path.join(vstar_path, test_type, os.path.splitext(image_file)[0] + ".json")
        if os.path.isfile(json_path):
            anno = json.load(open(json_path, encoding="utf-8"))
            out = []
            for bb in anno.get("bbox", []):
                x1, y1, w, h = map(int, bb)
                out.append([x1, y1, x1 + w, y1 + h])
            return out
    return []


def bbox_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def bbox_coverage(a, gt):
    """coverage(a, gt) = 交集面积 / gt 面积。"""
    g_area = bbox_area(gt)
    if g_area <= 0:
        return 0.0
    ix1, iy1 = max(a[0], gt[0]), max(a[1], gt[1])
    ix2, iy2 = min(a[2], gt[2]), min(a[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / g_area


def is_full_image(call: dict, img_size: tuple) -> bool:
    """调用 bbox 是否覆盖整幅图(全图 OCR/全图区域)——不算定位。"""
    bbox = call.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    w, h = img_size
    return (bbox[0] <= 0 and bbox[1] <= 0
            and bbox[2] >= w - 1 and bbox[3] >= h - 1)


def max_gt_coverage(traj: dict, gt_boxes: list) -> float:
    """原图空间调用 × GT bbox 的最大覆盖率(zoom/rotate 之后的调用不参与)。"""
    if not gt_boxes:
        return 0.0
    img_size = tuple(traj.get("image_size", (0, 0)))
    best = 0.0
    in_original_space = True
    for call in traj.get("tool_calls", []):
        name = call.get("tool_name", "")
        bbox = call.get("bbox")
        if in_original_space and bbox and len(bbox) == 4 and not is_full_image(call, img_size):
            for gt in gt_boxes:
                best = max(best, bbox_coverage(bbox, gt))
        if name in STATE_TOOLS:
            in_original_space = False
    return best


def filter_tool_trajectory(traj: dict, gt_boxes: list) -> dict:
    """工具轨迹过滤,返回 (是否通过, 覆盖率, 原因)。"""
    if traj.get("status") != "success":
        return None, 0.0, "status"
    if traj.get("accuracy", 0) != 1.0:
        return None, 0.0, "accuracy"
    calls = traj.get("tool_calls", [])
    if not calls:
        return None, 0.0, "no_tool_calls"
    if not all(is_real_call(c) for c in calls):
        return None, 0.0, "unreal_call"
    cov = max_gt_coverage(traj, gt_boxes)
    if cov < MIN_GT_COVERAGE:
        return None, cov, f"gt_coverage<{MIN_GT_COVERAGE}"
    return traj, cov, "pass"


def filter_answerfirst_trajectory(traj: dict) -> dict:
    """直接作答轨迹过滤:success + 正确 + 0 工具调用。"""
    if traj.get("status") != "success":
        return None, "status"
    if traj.get("accuracy", 0) != 1.0:
        return None, "accuracy"
    if traj.get("tool_calls"):
        return None, "has_tool_calls"
    return traj, "pass"


def main():
    global MIN_GT_COVERAGE
    parser = argparse.ArgumentParser(description="SFT 轨迹过滤")
    parser.add_argument("--input", action="append", required=True,
                        help="轨迹 JSONL(支持 glob);--tool_required 中的文件按工具轨迹过滤")
    parser.add_argument("--tool_required", action="append", default=[],
                        help="标记为强制工具模式的轨迹文件(glob),应用工具过滤条件")
    parser.add_argument("--vstar_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets", "vstar_bench"))
    parser.add_argument("--output", type=str, required=True,
                        help="过滤结果清单 JSONL 输出路径")
    parser.add_argument("--min_gt_coverage", type=float, default=MIN_GT_COVERAGE)
    args = parser.parse_args()
    MIN_GT_COVERAGE = args.min_gt_coverage

    tool_required_files = set()
    for pat in args.tool_required:
        tool_required_files.update(glob.glob(pat))

    stats = Counter()
    kept = []
    tool_kept = 0
    answer_kept = 0

    for pat in args.input:
        for path in sorted(glob.glob(pat)):
            is_tool = os.path.abspath(path) in {os.path.abspath(f) for f in tool_required_files}
            trajs = load_trajectories(path)
            stats[f"files:{os.path.basename(path)}"] = len(trajs)
            seen_images = set()
            for t in trajs:
                # 同文件内按 image 去重(断点续跑旧/新文件可能含重复样本)
                if t["image"] in seen_images:
                    stats["dup_skipped"] += 1
                    continue
                seen_images.add(t["image"])
                if is_tool:
                    gt = load_gt_bboxes(args.vstar_path, t["image"])
                    rec, cov, reason = filter_tool_trajectory(t, gt)
                    if rec is None:
                        stats[f"drop_tool:{reason}"] += 1
                        continue
                    stats["keep_tool"] += 1
                    tool_kept += 1
                    rec["_filter"] = {"source": path, "kind": "tool", "gt_coverage": round(cov, 3)}
                else:
                    rec, reason = filter_answerfirst_trajectory(t)
                    if rec is None:
                        stats[f"drop_answer:{reason}"] += 1
                        continue
                    stats["keep_answer"] += 1
                    answer_kept += 1
                    rec["_filter"] = {"source": path, "kind": "answerfirst", "gt_coverage": None}
                kept.append(rec)

    # 去重:同一样本在同一温度多轨迹时保留全部(多温度增广是数据量来源);
    # 跨文件重复的样本(不同温度)有意保留。
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"工具轨迹保留: {tool_kept} | 直接作答保留: {answer_kept} | 合计: {len(kept)}")
    print("来源统计:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
