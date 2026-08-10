#!/usr/bin/env python3
"""
轨迹 JSONL → SFT 对话 JSONL(计划 §2)。

轨迹文件未存模型当时的完整响应文本与工具响应图像,本脚本确定性重建:

1. **逐条重放状态机**:用 atr.tools.execute(与 run_atr_offline.py 的 ToolEnv
   同款逻辑)按记录的 tool_calls 重新执行,生成与模型当时所见一致的显示图
   (resize_for_display, DISPLAY_MAX=1024),落盘 sft_images/<sample>/step<i>.jpg。
2. **assistant 工具消息重建**:记录的 arguments 是执行空间(像素)坐标,
   按该步执行前 current_image.size 反推归一化:
   normalized = pixel / size,重建 <tool_call>{"name": X, "arguments": {...}}</tool_call>。
3. **user 工具响应消息**:重放的显示图 + output 文本,
   构造与 run_atr_offline.py 的 tool_response 逐字节一致。
4. **最终 assistant 消息**:<answer>{predicted_answer}</answer>。
5. 直接作答样本(无工具调用):仅两轮 [user → assistant <answer>]。

所有样本使用**先答后验** system prompt(与 RL 推理一致)。
输出 LLaMA-Factory multimodal 格式,90/10 切分 train/val。

用法:
  python experiments/scripts/trajectories_to_sft.py \
      --input experiments/results/sft_collect/sft_candidates.jsonl \
      --vstar_path datasets/vstar_bench \
      --out_dir datasets/vstar_bench/sft \
      --images_dir datasets/vstar_bench/sft_images
"""

import os
import sys
import json
import argparse
import random
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

from atr.tools import execute as execute_tool, registry
from experiments.scripts.run_atr_offline import (
    build_system_prompt,
    build_user_prompt,
    resize_for_display,
    DISPLAY_MAX,
)

# 先答后验 prompt(RL 推理统一 prompt,计划 §2.2)
SYSTEM_PROMPT = build_system_prompt(tool_required=False)


def find_source_image(vstar_path: str, image_file: str) -> str:
    for test_type in ("direct_attributes", "relative_position"):
        p = os.path.join(vstar_path, test_type, image_file)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"找不到原图: {image_file} in {vstar_path}")


def safe_relpath(path: str) -> str:
    """项目内路径用相对路径(可移植),跨盘时退化为绝对路径。"""
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:  # Windows 跨盘
        return path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def normalize_bbox(pixel_bbox, size):
    """执行空间像素 → 归一化 [0,1](显示空间与执行空间等比缩放,比例一致)。"""
    w, h = size
    x1, y1, x2, y2 = pixel_bbox
    return [round(x1 / w, 6), round(y1 / h, 6), round(x2 / w, 6), round(y2 / h, 6)]


def rebuild_tool_call(call: dict, view_size) -> str:
    """重建 assistant 的 <tool_call> 消息。

    记录的 arguments 是执行空间像素;bbox 类参数反推为归一化坐标,
    其余参数(如 rotate 的 angle)原样传递。
    """
    args = {}
    for k, v in call.get("arguments", {}).items():
        key = "bbox_2d" if k in ("bbox_2d", "bbox") else k
        if key == "bbox_2d" and isinstance(v, list) and len(v) == 4:
            args[key] = normalize_bbox(v, view_size)
        else:
            args[key] = v
    payload = {"name": call.get("tool_name", "unknown"), "arguments": args}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def build_tool_response_image(result, current: Image.Image) -> Image.Image:
    """重放该步的显示图:与 rollout 中 _last_image_b64 的编码对象一致。

    状态工具(zoom/rotate)产生新视图 → 新视图的显示图;
    非状态工具(crop)产生观察图 → 观察图的显示图。
    """
    if result.image is None:
        return None
    display_img, _ = resize_for_display(result.image)
    return display_img


def replay_and_build(traj: dict, vstar_path: str, images_dir: str,
                     system_prompt: Optional[str] = None) -> dict:
    """重放一条轨迹并重建对话消息。返回 (messages, 重放图路径列表)。

    system_prompt: 按轨迹来源传入对应的 system prompt(强制工具轨迹用
    tool_required=True 版,直接作答轨迹用先答后验版),保证指令与行为一致。
    """
    image_file = traj["image"]
    src_path = find_source_image(vstar_path, image_file)
    sample_name = os.path.splitext(image_file)[0]
    sample_dir = os.path.join(images_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    current = Image.open(src_path).convert("RGB")

    # 初始显示图(模型第一轮所见)
    display0, _ = resize_for_display(current)
    img0_path = os.path.join(sample_dir, "step0.jpg")
    display0.save(img0_path, "JPEG", quality=85)

    # user 首轮消息(与 rollout 的 build_user_prompt 一致)
    user_prompt = build_user_prompt(traj["question"], traj.get("options"))
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": safe_relpath(img0_path)},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    calls = traj.get("tool_calls", [])
    for i, call in enumerate(calls):
        # 该步执行前的视图尺寸(坐标空间锚点,反推归一化用)
        view_size = current.size

        # 1) assistant 工具调用消息
        messages.append({"role": "assistant", "content": rebuild_tool_call(call, view_size)})

        # 2) 重放执行(记录的 arguments 已是执行空间像素,直接执行)
        try:
            result = execute_tool(call["tool_name"], call.get("arguments", {}), current)
        except Exception as e:  # 理论不发生(轨迹来自真实执行)
            print(f"  [WARN] 重放失败 {sample_name} step{i}: {e},跳过该样本")
            return None, None

        # 3) 更新状态(与 ToolEnv 一致:状态工具切换 current_image)
        if result.image is not None and registry.get(result.canonical_name).updates_state:
            current = result.image

        # 4) 工具响应显示图
        display = build_tool_response_image(result, current)
        step_path = os.path.join(sample_dir, f"step{i + 1}.jpg")
        user_content = [{"type": "text", "text": "<tool_response>"}]
        if display is not None:
            display.save(step_path, "JPEG", quality=85)
            user_content.append({
                "type": "image",
                "image": safe_relpath(step_path),
            })
        user_content.append({"type": "text", "text": result.output})
        user_content.append({"type": "text", "text": "</tool_response>"})
        user_content.append({"type": "text", "text":
                             "\nContinue analyzing. Call another tool or answer with <answer>...</answer>"})
        messages.append({"role": "user", "content": user_content})

    # 最终答案消息
    messages.append({"role": "assistant", "content": f"<answer>{traj.get('predicted_answer', '')}</answer>"})

    return messages, sample_dir


def main():
    parser = argparse.ArgumentParser(description="轨迹 → SFT 对话 JSONL")
    parser.add_argument("--input", type=str, required=True, help="过滤后的轨迹清单(sft_candidates.jsonl)")
    parser.add_argument("--vstar_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets", "vstar_bench"))
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets", "vstar_bench", "sft"),
                        help="SFT JSONL 输出目录")
    parser.add_argument("--images_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets", "vstar_bench", "sft_images"),
                        help="重放图落盘目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.images_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        trajs = [json.loads(l) for l in f if l.strip()]
    print(f"载入 {len(trajs)} 条候选轨迹")

    records = []
    n_skipped = 0
    for t in trajs:
        # 指令与行为对齐:工具轨迹 → 强制工具 prompt;直接作答 → 先答后验 prompt
        kind = t["_filter"]["kind"]
        sys_prompt = build_system_prompt(tool_required=(kind == "tool"))
        messages, sample_dir = replay_and_build(t, args.vstar_path, args.images_dir,
                                                system_prompt=sys_prompt)
        if messages is None:
            n_skipped += 1
            continue
        records.append({
            "id": os.path.splitext(t["image"])[0],
            "messages": messages,
            "_meta": {
                "kind": t["_filter"]["kind"],
                "source": t["_filter"]["source"],
                "gt_coverage": t["_filter"].get("gt_coverage"),
                "n_tool_calls": len(t.get("tool_calls", [])),
                "image": t["image"],
            },
        })
    print(f"重建完成: {len(records)} 条(跳过 {n_skipped})")

    # 90/10 切分(按轨迹随机,seed 固定)
    random.seed(args.seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * 0.1))
    val_records, train_records = records[:n_val], records[n_val:]

    def dump(recs, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kind = sum(1 for r in recs if r["_meta"]["kind"] == "tool")
        print(f"  {os.path.basename(path)}: {len(recs)} 条(工具轨迹 {kind},直接作答 {len(recs) - kind})")

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    print(f"切分(train/val 90/10, seed={args.seed}):")
    dump(train_records, train_path)
    dump(val_records, val_path)

    print(f"\n重放图: {args.images_dir}")
    print(f"完成!train: {train_path}\nval: {val_path}")


if __name__ == "__main__":
    main()
