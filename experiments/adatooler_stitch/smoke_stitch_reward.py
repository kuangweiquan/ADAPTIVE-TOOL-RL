#!/usr/bin/env python3
"""本机冒烟测试:四臂奖励逻辑 + AdaTooler-V 格式适配(纯 CPU,无 verl)。

覆盖(对应 ai_handoff/10 §6 四臂):
  1. Arm A 纯 acc: 对=1 错=0
  2. Arm B 调用惩罚: n=6 时罚项最大 0.6;n=0 时 0.081(论文公式复现)
  3. Arm C 规则 U/C/S: 用 AdaTooler-V 工具名(crop_image)与像素坐标,
     验证 alias 规范化 + bbox 归一化后 C/U 能触发
  4. Arm D judge ΔS: ΔS 门控(正/负/1.0 三种)
  5. silent_death_check 能识别恒死分量
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "experiments", "adatooler_stitch")):
    if p not in sys.path:
        sys.path.insert(0, p)

from arm_logic import compute_arm_reward, silent_death_check  # noqa: E402
from verl_reward_manager import compute_sample_reward, normalize_bboxes  # noqa: E402
from atr.reward import AdaptiveToolReward  # noqa: E402
from atr.config import ATRConfig  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


atr = AdaptiveToolReward(ATRConfig())

print("=== 1. Arm A: 纯 acc ===")
r, c = compute_arm_reward("a", 1.0, tool_calls=[], atr=atr)
check("A 对=1", abs(r - 1.0) < 1e-9, f"r={r}")
r, c = compute_arm_reward("a", 0.0, tool_calls=[], atr=atr)
check("A 错=0", r == 0.0, f"r={r}")
check("A 无 at_penalty", c["at_penalty"] == 0.0)

print("=== 2. Arm B: AT 调用惩罚(论文公式复现) ===")
r6, c6 = compute_arm_reward("b", 1.0, tool_calls=[{}] * 6, atr=atr)
check("B n=6 罚项=0.6(顶点)", abs(c6["at_penalty"] - 0.6) < 1e-9, f"penalty={c6['at_penalty']:.4f}")
r0, c0 = compute_arm_reward("b", 0.0, tool_calls=[], atr=atr)
check("B n=0 罚项=0.0811(0.6*e^-2)", abs(c0["at_penalty"] - 0.08107) < 1e-3,
      f"penalty={c0['at_penalty']:.4f}")
check("B 错+n=0 仍有奖励(公式原样复现,设计问题另记)",
      r0 > 0.0, f"r={r0:.4f}")

print("=== 3. Arm C: AdaTooler-V 格式适配(alias + 像素坐标归一化) ===")
# 他们的工具名 crop_image / zoom_in(像素坐标),重复调用 + 超预算
resp_their = '''<tool_call>
{"name": "crop_image", "arguments": {"bbox_2d": [100, 100, 300, 300], "target_image": 1}}
</tool_call>
The region shows a red object.
<tool_call>
{"name": "crop_image", "arguments": {"bbox_2d": [110, 110, 310, 310], "target_image": 1}}
</tool_call>
The region shows a red object again.
<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [200, 200, 400, 400], "target_image": 1}}
</tool_call>
x
<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [300, 300, 500, 500], "target_image": 1}}
</tool_call>
y
<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [400, 400, 600, 600], "target_image": 1}}
</tool_call>
z
<answer>
B
</answer>'''

reward, components = compute_sample_reward(
    arm="c",
    response_str=resp_their,
    ground_truth=["B"],
    problem_type="multiple choice",
    tool_interact_info=None,
    extra_info={"images": [], "problem_type": "multiple choice"},
    atr=atr,
)
# 无 image_size → 像素坐标不归一化,但 alias 规范化仍应生效:
# crop_image→crop 后 Type 1 的 IoU 判定在像素空间仍一致(同尺度)
check("crop_image 轨迹 C>0(alias 生效)", components.get("cost", 0.0) > 0,
      f"C={components.get('cost', 0):.3f}")
check("dup_spatial 计数>0", components.get("cost_dup_spatial", 0) > 0,
      f"={components.get('cost_dup_spatial')}")
check("call_budget 计数=2(5 调用,预算 3)", components.get("cost_call_budget") == 2,
      f"={components.get('cost_call_budget')}")

# 像素坐标归一化检查(600x800 图)
calls = [{"tool_name": "crop", "bbox": [100, 100, 300, 300]}]
normalize_bboxes(calls, (600, 800))
check("像素 bbox 归一化", all(abs(v - t) < 1e-6 for v, t in zip(
    calls[0]["bbox"], [1/6, 1/8, 1/2, 3/8])), str(calls[0]["bbox"]))
calls_norm = [{"tool_name": "crop", "bbox": [0.1, 0.1, 0.3, 0.3]}]
normalize_bboxes(calls_norm, (600, 800))
check("已归一化 bbox 不动", calls_norm[0]["bbox"] == [0.1, 0.1, 0.3, 0.3])

print("=== 4. Arm D: judge ΔS 门控 ===")
r, c = compute_arm_reward("d", 1.0, tool_calls=[{}] * 6, delta_s=1.0, atr=atr)
check("D ΔS=1 同 B(罚项 0.6)", abs(c["at_penalty"] - 0.6) < 1e-9)
r, c = compute_arm_reward("d", 1.0, tool_calls=[{}] * 6, delta_s=-1.0, atr=atr)
check("D ΔS=-1 罚项转负", c["at_penalty"] < 0, f"penalty={c['at_penalty']:.4f}")
r, c = compute_arm_reward("d", 1.0, tool_calls=[{}] * 6, delta_s=0.0, atr=atr)
check("D ΔS=0 罚项归零(工具无益时不奖励)", abs(c["at_penalty"]) < 1e-9)

print("=== 5. silent_death_check ===")
report = silent_death_check([
    {"acc": 1.0, "at_penalty": 0.6, "n_tool": 6.0},
    {"acc": 0.0, "at_penalty": 0.6, "n_tool": 3.0},
])
check("恒量 at_penalty 判死", report["at_penalty"]["dead"] is True, str(report["at_penalty"]))
check("变量 acc 判活", report["acc"]["dead"] is False, str(report["acc"]))

print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
