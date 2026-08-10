#!/usr/bin/env python3
"""
验证轨迹文件中的所有工具调用都是真实调用。

判定标准:
  1. tool_name 必须是 atr.tools 注册的规范名(crop/zoom/rotate/ocr),无 alias/未知名
  2. output 不得以 "[Unknown tool" 或 "[Error" 开头(假调用/失败调用的标记)
  3. 空间工具(crop/zoom)必须带有效 bbox(4 个数,正面积)—— clamp 后的真实执行结果
  4. rotate 必须返回 "[Rotated by ...]" 且不记 bbox(设计如此,避免 lazy-crop 惩罚)
  5. ocr 必须返回 "[OCR result ...]" 或 "[No text detected in region]"——真实执行
     (安装 tesseract 后应是真实识别结果)
  6. 状态:样本 status 应为 success(允许 max_turns_exceeded,不允许 error)

用法:
  python experiments/scripts/verify_real_tool_calls.py <trajectories.jsonl>
退出码:0 = 全部真实;1 = 存在非真实调用
"""

import sys
import os
import json
from collections import Counter

VALID_TOOLS = {"crop", "zoom", "rotate", "ocr"}


def verify_file(path: str) -> int:
    # 轨迹文件可能以 utf-8 或 Windows 默认编码(GBK)写入,自动探测
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = [l.strip() for l in f if l.strip()]
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[FAIL] 无法解码轨迹文件: {path}")
        return 1
    trajectories = [json.loads(l) for l in lines]

    n_samples = len(trajectories)
    n_tool_calls = 0
    tool_counter = Counter()
    problems = []          # (file, idx, reason, detail)
    errors = []            # (file, msg)

    for traj in trajectories:
        fname = traj.get("image", "?")
        status = traj.get("status", "?")
        if status == "error":
            errors.append((fname, "status=error"))
        calls = traj.get("tool_calls", [])
        n_tool_calls += len(calls)
        for i, c in enumerate(calls):
            name = c.get("tool_name", "")
            out = str(c.get("output", ""))
            tool_counter[name] += 1

            if name not in VALID_TOOLS:
                problems.append((fname, i, f"未知工具名: {name!r}", out[:60]))
                continue
            if out.startswith("[Unknown tool") or out.startswith("[Error"):
                problems.append((fname, i, f"假/失败调用: {out[:50]}", out[:80]))
                continue
            if name in ("crop", "zoom"):
                bbox = c.get("bbox")
                ok_bbox = (isinstance(bbox, list) and len(bbox) == 4
                           and bbox[2] > bbox[0] and bbox[3] > bbox[1])
                if not ok_bbox:
                    problems.append((fname, i, f"{name} 缺有效 bbox: {bbox}", out[:60]))
            elif name == "rotate":
                if not out.startswith("[Rotated by"):
                    problems.append((fname, i, f"rotate 输出异常: {out[:50]}", out[:80]))
                if c.get("bbox") is not None:
                    problems.append((fname, i, "rotate 不应记 bbox", out[:60]))
            elif name == "ocr":
                if not (out.startswith("[OCR result") or out == "[No text detected in region]"):
                    problems.append((fname, i, f"ocr 输出异常: {out[:50]}", out[:80]))

    print(f"样本数: {n_samples}, 工具调用总数: {n_tool_calls}")
    print(f"工具分布: {dict(tool_counter)}")
    if errors:
        print(f"\n[FAIL] error 样本: {len(errors)}")
        for fname, msg in errors[:10]:
            print(f"  - {fname}: {msg}")
    if problems:
        print(f"\n[FAIL] 非真实调用: {len(problems)}")
        for fname, i, reason, detail in problems[:15]:
            print(f"  - {fname}[{i}]: {reason}")
        return 1
    print("\n[PASS] 所有工具调用均为真实调用")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(verify_file(sys.argv[1]))
