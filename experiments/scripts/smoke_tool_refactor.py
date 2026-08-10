#!/usr/bin/env python3
"""
Smoke test — 工具重构(atr/tools 注册表 + 奖励层派生 + 离线管线)全链路验证。

覆盖:
  1. 注册表 / schema:crop/zoom/rotate/ocr 四名、无 select、alias zoom_in
  2. 工具执行:合成图上 crop/zoom/rotate/ocr 的往返(含错误路径)
  3. ToolTrace 记录与期望 dict 逐字段一致(旧轨迹格式字节兼容)
  4. Reward round-trip:新旧风格混合记录 → AdaptiveToolReward.compute 无异常;
     select → 1 分、合法 crop → 3 分;format_gate < 1(有 select 时)
  5. SYSTEM_PROMPT(离线管线生成):含 rotate、无 select

用法:
  python experiments/scripts/smoke_tool_refactor.py
退出码:0 = 全部通过;1 = 有失败
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from PIL import Image

from atr.tools import registry, get_tool_schemas, execute, ToolTrace
from atr.reward.sequence import SequenceQuality
from atr.reward.base_reward import AdaptiveToolReward
from atr.config import ATRConfig

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def section(title: str):
    print(f"\n== {title} ==")


def main() -> int:
    # ============================================================
    # 1. Registry / schemas
    # ============================================================
    section("1. Registry / schemas")
    check("工具名 = [crop, zoom, rotate, ocr]",
          registry.tool_names == ["crop", "zoom", "rotate", "ocr"],
          str(registry.tool_names))
    check("无 select", "select" not in registry.all_names,
          str(sorted(registry.all_names)))
    check("all_names 含 alias zoom_in", "zoom_in" in registry.all_names)
    check("canonical(zoom_in) == zoom", registry.canonical("zoom_in") == "zoom")
    check("canonical(select) is None", registry.canonical("select") is None)
    check("spatial_tools == {crop, zoom}",
          registry.spatial_tools == frozenset({"crop", "zoom"}))

    schemas = get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    check("schema 顺序 = 注册顺序", names == ["crop", "zoom", "rotate", "ocr"], str(names))
    rot = [s for s in schemas if s["function"]["name"] == "rotate"][0]["function"]
    check("rotate schema 含 angle 且 required", "angle" in rot["parameters"]["properties"]
          and "angle" in rot["parameters"]["required"])

    # ============================================================
    # 2. Tool execution round-trips
    # ============================================================
    section("2. Tool execution")
    img = Image.new("RGB", (512, 512), "white")

    r = execute("crop", {"bbox_2d": [10, 10, 110, 210]}, img)
    check("crop 输出前缀", r.output.startswith("[Cropped region"))
    check("crop bbox 记录", r.bbox == [10, 10, 110, 210], str(r.bbox))
    check("crop 产生裁剪图(供模型查看)", r.image is not None
          and r.image.size == (100, 200), str(r.image.size if r.image else None))
    check("crop 规范名", r.canonical_name == "crop")

    r = execute("crop", {"bbox_2d": [5, 5, 5, 5]}, img)
    check("crop 空 bbox 报错文本", r.output == "[Crop: invalid bbox, region is empty]")

    r = execute("zoom", {"bbox_2d": [10, 10, 110, 110]}, img)
    check("zoom 输出", r.output == "[Zoomed into (10,10)-(110,110)]")
    check("zoom 图尺寸回原视图", r.image is not None and r.image.size == (512, 512))
    check("zoom bbox 记录", r.bbox == [10, 10, 110, 110])

    r = execute("zoom_in", {"bbox_2d": [10, 10, 110, 110]}, img)
    check("zoom_in alias → 规范名 zoom", r.canonical_name == "zoom")

    r = execute("rotate", {"angle": 90}, img)
    check("rotate 输出", r.output == "[Rotated by 90 degrees]")
    check("rotate 产生新图且尺寸不变", r.image is not None and r.image.size == (512, 512))
    check("rotate 不记 bbox(避免 lazy-crop)", r.bbox is None)

    try:
        execute("rotate", {}, img)
        check("rotate 缺 angle 抛 ValueError", False)
    except ValueError as e:
        check("rotate 缺 angle 抛 ValueError", str(e) == "rotate requires angle", str(e))

    try:
        execute("select", {}, img)
        check("select → KeyError", False)
    except KeyError:
        check("select → KeyError", True)

    r = execute("ocr", {"bbox_2d": [0, 0, 50, 50]}, img)
    check("ocr 输出(无 tesseract 时为 No-text)",
          r.output.startswith("[OCR result") or r.output == "[No text detected in region]",
          r.output[:40])
    r = execute("ocr", {}, img)
    check("ocr 无 bbox → 全图记录", r.bbox == [0, 0, 512, 512], str(r.bbox))

    # ============================================================
    # 3. ToolTrace byte-compat
    # ============================================================
    section("3. ToolTrace 记录兼容")
    trace = ToolTrace()
    r = execute("crop", {"bbox_2d": [10, 10, 50, 60]}, img)
    trace.record(r.canonical_name, r.arguments, r.output, r.bbox)
    expected = {
        "tool_name": "crop",
        "arguments": {"bbox_2d": [10, 10, 50, 60]},
        "output": f"[Cropped region (10,10)-(50,60), size 40×50]",
        "bbox": [10, 10, 50, 60],
    }
    check("crop 记录逐字段一致", trace.records[0] == expected, json.dumps(trace.records[0]))

    r = execute("ocr", {}, img)
    trace.record(r.canonical_name, r.arguments, r.output, r.bbox)
    check("ocr 无 bbox 记录 arguments/bbox 为全图",
          trace.records[1]["arguments"] == {"bbox_2d": [0, 0, 512, 512]}
          and trace.records[1]["bbox"] == [0, 0, 512, 512])

    # ============================================================
    # 4. Reward round-trip(新旧混合)
    # ============================================================
    section("4. Reward round-trip")
    mixed = [
        {"tool_name": "crop", "arguments": {"bbox_2d": [10, 10, 110, 210]},
         "output": "[Cropped region (10,10)-(110,210), size 100×200]", "bbox": [10, 10, 110, 210]},
        {"tool_name": "zoom_in", "arguments": {"bbox_2d": [20, 20, 80, 80]},
         "output": "[Zoomed into (20,20)-(80,80)]", "bbox": [20, 20, 80, 80]},   # 旧 alias
        {"tool_name": "rotate", "arguments": {"angle": 90}, "output": "[Rotated by 90 degrees]"},
        {"tool_name": "ocr", "arguments": {"bbox_2d": [10, 10, 50, 50]},
         "output": "[OCR result: \"12 mm\"]", "bbox": [10, 10, 50, 50]},
        {"tool_name": "select", "arguments": {"label": "x"}, "output": "[Selected: x]"},  # 旧 select
    ]
    atr = AdaptiveToolReward(config=ATRConfig())
    total, comps = atr.compute(
        tool_calls=mixed,
        final_answer="12 mm",
        accuracy=1.0,
        question="What is the width?",
        image_size=(512, 512),
    )
    check("compute 无异常且返回组件", set(["utility", "cost", "sequence", "total"]).issubset(comps),
          str(sorted(comps.keys())))

    sq = SequenceQuality()
    scores, _ = sq.compute_per_call_scores(mixed)
    check("per-call: crop=3", scores[0] == 3, str(scores))
    check("per-call: zoom_in(alias)=3", scores[1] == 3, str(scores))
    check("per-call: rotate=3", scores[2] == 3, str(scores))
    check("per-call: ocr=3", scores[3] == 3, str(scores))
    check("per-call: select=1(未注册)", scores[4] == 1, str(scores))
    gate = sq.compute_format_gate(mixed)
    check("format_gate < 1(有 select)", gate < 1.0, str(gate))

    # ============================================================
    # 5. SYSTEM_PROMPT(离线管线)
    # ============================================================
    section("5. SYSTEM_PROMPT")
    # run_atr_offline 模块级会重包 sys.stdout,先刷新避免缓冲丢失
    sys.stdout.flush()
    sys.path.insert(0, SCRIPT_DIR)
    import run_atr_offline as rao
    check("prompt 含 rotate", "rotate" in rao.SYSTEM_PROMPT)
    check("prompt 含规范名 zoom", '"name": "zoom"' in rao.SYSTEM_PROMPT)
    check("prompt 无 select", "select" not in rao.SYSTEM_PROMPT)

    # ============================================================
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} 项未通过: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
