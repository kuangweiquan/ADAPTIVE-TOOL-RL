#!/usr/bin/env python3
"""从离线评估轨迹补算工具 bbox 与 VStar GT bbox 的 IoU 统计。

背景:run_atr_offline.py 的 analysis JSON 无 IoU 独立输出(accuracy 即 0/1 判定),
本轮(04 提示词 v2)汇报中 IoU 缺失,本脚本从已有 trajectories_*.jsonl 离线补算,
无需重跑模型 / API,零成本。

坐标语义(run_atr_offline.ToolEnv):
  - 模型输出归一化 bbox [0,1],换算到执行空间像素 = 归一化 × current_image.size
  - current_image 尺寸恒等于原图尺寸(zoom 裁剪后 resize 回原尺寸)
  - 因此轨迹 bbox = 视图归一化 × 原图尺寸,语义上是"当前视图"的局部坐标;
    zoom 后继续调用时,该坐标对应原图的子区域,需沿视图链逆映射回原图
  - crop 不更新视图(锚点保持主视图);rotate 旋转后视图方向变化(标记中断)

两种口径:
  A. 记录坐标直比(近似,视同原图坐标)—— 与历史 22%(v1,来源脚本已不可考)最可能同口径
  B. 视图链逆映射(严格)—— zoom 链逆映射回原图坐标后与 GT 比

统计(两种口径各出):
  - 样本级命中:样本任一调用 bbox 与任一 GT bbox 的 IoU>0.1
  - 工具级命中:全部调用中 IoU>0.1 的比例
  - 样本平均最大 IoU(任调用 × 任 GT 的最大值)
  - 首次调用命中(仅口径 B,诊断"第一步定位"质量)

GT: datasets/vstar_bench/<test_type>/<sample>.json,bbox 为 VStar xywh 格式
    [x1, y1, w, h](注意不是 xyxy!),relative_position 含多个目标取 max。

用法:
  python experiments/scripts/compute_gt_iou.py --traj experiments/results/eval_sft_v2/trajectories_*.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

IOU_THRESHOLD = 0.1


def xywh_to_xyxy(b: List[float]) -> List[float]:
    """VStar xywh → xyxy。"""
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou(box1: List[float], box2: List[float]) -> float:
    """标准 IoU,输入均为 [x1, y1, x2, y2]。"""
    x1, y1, x2, y2 = box1
    u1, v1, u2, v2 = box2
    ix1, iy1 = max(x1, u1), max(y1, v1)
    ix2, iy2 = min(x2, u2), min(y2, v2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (u2 - u1) * (v2 - v1)
    return inter / (area1 + area2 - inter)


def load_gt(vstar_path: str, image_name: str) -> List[List[float]]:
    """从 direct_attributes / relative_position 任一目录读 GT bbox(xyxy)。"""
    for test_type in ("direct_attributes", "relative_position"):
        jf = os.path.join(vstar_path, test_type, os.path.splitext(image_name)[0] + ".json")
        if os.path.isfile(jf):
            with open(jf, "r", encoding="utf-8") as f:
                ann = json.load(f)
            return [xywh_to_xyxy(b) for b in ann.get("bbox", [])]
    return []


def map_to_original(bbox_px: List[float], view_chain: List[Tuple[tuple, tuple]]) -> Optional[List[float]]:
    """把"当前视图局部像素 bbox"沿视图链逆映射回原图坐标。

    view_chain: [(frame_px, view_size)],frame_px 为当前视图被 zoom 时的原图系窗口,
                view_size 为该视图的像素尺寸(w,h)。
    映射:视图归一化坐标 = bbox_px / view_size,逐层外推:
        orig = frame.x1 + nx * frame_w; orig_y = frame.y1 + ny * frame_h
    """
    w, h = view_chain[0][1] if view_chain else (0, 0)
    # 当前视图尺寸(恒为原图尺寸,取最后一层视图尺寸)
    if view_chain:
        cur_w, cur_h = view_chain[0][1]
    else:
        return None
    nx1, ny1, nx2, ny2 = bbox_px[0] / cur_w, bbox_px[1] / cur_h, bbox_px[2] / cur_w, bbox_px[3] / cur_h
    # chain = [zoomN, ..., zoom1](最新在前),映射需从 zoomN 逐步外推到 zoom1(原图坐标)
    for (frame, vw_vh) in view_chain:
        fx1, fy1, fx2, fy2 = frame
        vw, vh = vw_vh
        nx1, ny1, nx2, ny2 = (
            fx1 + nx1 * (fx2 - fx1) / vw, fy1 + ny1 * (fy2 - fy1) / vh,
            fx1 + nx2 * (fx2 - fx1) / vw, fy1 + ny2 * (fy2 - fy1) / vh,
        )
    return [nx1, ny1, nx2, ny2]


def compute(traj_path: str, vstar_path: str) -> Dict:
    """对一份轨迹文件跑两种口径统计。"""
    recs = [json.loads(l) for l in open(traj_path, encoding="utf-8") if l.strip()]
    # 口径 A/B 各自的样本级命中、工具级命中、平均最大 IoU、首调命中
    stat_a = {"hit": 0, "calls": 0, "call_hit": 0, "max_iou_sum": 0.0, "first_hit": 0}
    stat_b = {"hit": 0, "calls": 0, "call_hit": 0, "max_iou_sum": 0.0, "first_hit": 0}
    n_gt_missing = 0
    for rec in recs:
        gts = load_gt(vstar_path, rec["image"])
        calls = rec.get("tool_calls", [])
        calls = [c for c in calls if c.get("bbox")]
        if not gts:
            n_gt_missing += 1
            continue
        # 视图链重建(与 ToolEnv 相同语义)
        chain: List[Tuple[tuple, tuple]] = []  # [(原图窗口, 该视图尺寸)]
        chain_ok = True
        max_iou_a = max_iou_b = 0.0
        hit_a = hit_b = first_hit_a = first_hit_b = False
        for i, call in enumerate(calls):
            name = call["tool_name"]
            bbox = call["bbox"]
            stat_a["calls"] += 1
            if chain_ok:
                stat_b["calls"] += 1
            # 口径 A:直比
            iou_a = max(iou(bbox, g) for g in gts)
            max_iou_a = max(max_iou_a, iou_a)
            if iou_a > IOU_THRESHOLD:
                stat_a["call_hit"] += 1
                hit_a = True
                if i == 0:
                    first_hit_a = True
            # 口径 B:逆映射(chain_ok 时)
            if chain_ok and name in ("crop", "zoom", "zoom_in"):
                mb = map_to_original(bbox, chain) if chain else bbox
                if mb:
                    iou_b = max(iou(mb, g) for g in gts)
                    max_iou_b = max(max_iou_b, iou_b)
                    if iou_b > IOU_THRESHOLD:
                        stat_b["call_hit"] += 1
                        hit_b = True
                        if i == 0:
                            first_hit_b = True
            # 更新视图链
            if name in ("zoom", "zoom_in"):
                # 当前视图尺寸 = chain 首层尺寸(原图尺寸)或原图尺寸
                if chain:
                    cur_w, cur_h = chain[0][1]
                else:
                    cur_w, cur_h = rec.get("image_size", [0, 0])
                if cur_w and cur_h and chain_ok:
                    chain.insert(0, (tuple(bbox), (cur_w, cur_h)))
            elif name == "rotate":
                chain_ok = False  # 旋转后视图方向变化,锚点断裂
        if hit_a:
            stat_a["hit"] += 1
            if first_hit_a:
                stat_a["first_hit"] += 1
        if hit_b:
            stat_b["hit"] += 1
            if first_hit_b:
                stat_b["first_hit"] += 1
        stat_a["max_iou_sum"] += max_iou_a
        stat_b["max_iou_sum"] += max_iou_b

    n = len(recs)
    res = {"n_samples": n, "n_gt_missing": n_gt_missing}
    for name, s in (("A_直比", stat_a), ("B_逆映射", stat_b)):
        n_hit = s["hit"]
        res[f"{name}_样本命中"] = f"{n_hit}/{n} ({n_hit / n * 100:.1f}%)"
        res[f"{name}_工具命中"] = f"{s['call_hit']}/{s['calls']} ({s['call_hit'] / max(1, s['calls']) * 100:.1f}%)"
        res[f"{name}_平均最大IoU"] = f"{s['max_iou_sum'] / max(1, n):.3f}"
        res[f"{name}_首调命中"] = f"{s['first_hit']}/{n} ({s['first_hit'] / n * 100:.1f}%)"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="补算轨迹工具 bbox 与 GT 的 IoU 统计")
    ap.add_argument("--traj", required=True, help="trajectories_*.jsonl 路径")
    ap.add_argument("--vstar_path", default="datasets/vstar_bench")
    args = ap.parse_args()
    res = compute(args.traj, args.vstar_path)
    print(f"== {os.path.basename(args.traj)} ==")
    for k, v in res.items():
        print(f"  {k:<16} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
