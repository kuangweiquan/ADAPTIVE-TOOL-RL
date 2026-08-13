#!/usr/bin/env python3
"""本机冒烟测试:C=0 缺陷修复验证(纯 CPU)。

覆盖(对应 09 文档 §3 的三个缺陷):
  1. alias 规范化:模型写 "zoom_in" → parse 后规范名为 "zoom",
     Type 1(重复空间操作)能触发(修复前静默失效)
  2. 调用预算检测器(Type 6):超出 cost_call_budget 的调用被惩罚
  3. per-type 计数器进 components(区分"解析失败 vs 检测器盲区")
  4. 回归:规范名 "zoom" 重复检测、干净轨迹 C=0 不变
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from atr.adapter.patch_reward import parse_tool_trajectory, approximate_tool_outputs  # noqa: E402
from atr.reward import AdaptiveToolReward  # noqa: E402
from atr.config import ATRConfig  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def parse(response):
    calls = parse_tool_trajectory(response)
    return approximate_tool_outputs(response, calls)


atr = AdaptiveToolReward(ATRConfig())  # 默认 lambda_u=1.0, gamma_c=0.5, eta_s=0.3, budget=3

print("=== 1. alias 规范化: zoom_in → zoom, Type 1 重复检测触发 ===")
resp_alias_dup = '''<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [0.10, 0.10, 0.30, 0.30]}}
</tool_call>
The region shows a red object.
<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [0.11, 0.11, 0.31, 0.31]}}
</tool_call>
The region shows a red object again.
<answer>
A
</answer>'''
calls = parse(resp_alias_dup)
check("alias 规范化为 zoom", calls[0]["tool_name"] == "zoom", f"got {calls[0]['tool_name']!r}")
r, comp = atr.compute(tool_calls=calls, final_answer="A", accuracy=1.0)
check("重复 zoom_in(alias) 触发 C>0", comp["cost"] > 0, f"C={comp['cost']:.3f}")
check("dup_spatial 计数=1", comp.get("cost_dup_spatial") == 1, f"={comp.get('cost_dup_spatial')}")

print("=== 2. 调用预算检测器(Type 6) ===")
resp_spam = '''<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.1, 0.1, 0.2, 0.2]}}
</tool_call>
a
<tool_call>
{"name": "crop", "arguments": {"bbox_2d": [0.2, 0.2, 0.3, 0.3]}}
</tool_call>
b
<tool_call>
{"name": "ocr", "arguments": {"bbox_2d": [0.3, 0.3, 0.4, 0.4]}}
</tool_call>
c
<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.4, 0.4, 0.5, 0.5]}}
</tool_call>
d
<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.5, 0.5, 0.6, 0.6]}}
</tool_call>
e
<answer>
B
</answer>'''
calls = parse(resp_spam)
r, comp = atr.compute(tool_calls=calls, final_answer="B", accuracy=1.0)
check("5 调用超过预算 3 → C>0", comp["cost"] > 0, f"C={comp['cost']:.3f}")
check("call_budget 计数=2", comp.get("cost_call_budget") == 2, f"={comp.get('cost_call_budget')}")
check("consecutive_same 计数=1(第4、5次连续 zoom)",
      comp.get("cost_consecutive_same") == 1, f"={comp.get('cost_consecutive_same')}")

print("=== 3. 干净轨迹 C=0 且计数器全零(无误报) ===")
resp_clean = '''<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.10, 0.10, 0.30, 0.30]}}
</tool_call>
The object is here.
<tool_call>
{"name": "ocr", "arguments": {"bbox_2d": [0.10, 0.10, 0.30, 0.30]}}
</tool_call>
Text reads "apple".
<answer>
C
</answer>'''
calls = parse(resp_clean)
r, comp = atr.compute(tool_calls=calls, final_answer="C", accuracy=1.0)
check("干净轨迹 C=0", comp["cost"] == 0.0, f"C={comp['cost']}")
check("计数器全零", all(comp.get(f"cost_{k}") == 0 for k in
      ("dup_spatial", "dup_ocr", "oscillation", "consecutive_same", "call_budget")),
      str({k: comp.get(f"cost_{k}") for k in ("dup_spatial", "dup_ocr", "oscillation",
                                               "consecutive_same", "call_budget")}))

print("=== 4. 回归:规范名 zoom 重复 + 无工具调用 ===")
resp_zoom_dup = '''<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.10, 0.10, 0.30, 0.30]}}
</tool_call>
x
<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.12, 0.12, 0.32, 0.32]}}
</tool_call>
y
<answer>
A
</answer>'''
calls = parse(resp_zoom_dup)
r, comp = atr.compute(tool_calls=calls, final_answer="A", accuracy=1.0)
check("规范名 zoom 重复仍触发(回归)", comp["cost"] > 0, f"C={comp['cost']:.3f}")

r, comp = atr.compute(tool_calls=[], final_answer="B", accuracy=0.0)
check("无工具调用 C=0 且计数存在", comp["cost"] == 0.0 and comp.get("cost_dup_spatial") == 0,
      str(comp["cost"]))

print("=== 5. 未知工具名保持原样(canonicalize 不吞未知名) ===")
resp_unknown = '''<tool_call>
{"name": "some_future_tool", "arguments": {"bbox_2d": [0.1, 0.1, 0.2, 0.2]}}
</tool_call>
<answer>
A
</answer>'''
calls = parse(resp_unknown)
check("未知名原样保留", calls[0]["tool_name"] == "some_future_tool", calls[0]["tool_name"])

print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
