#!/usr/bin/env python3
"""本机冒烟测试:VStarToolEnv + ATR reward 闭环 + 数据转换(纯 CPU)。

覆盖:
  1. VStarToolEnv 注册(ToolBase.create)与生命周期(zoom 状态更新/crop 观察/ocr/answer)
  2. RL 轨迹文本 → parse_tool_trajectory + approximate_tool_outputs
  3. compute_vstar_score 答案判定
  4. atr.compute 闭环:GT 对齐 zoom 有 utility、同坐标重复 zoom 有 cost
  5. vstar_to_verl_parquet 端到端(train/val parquet 生成 + 回读)
"""

import os
import sys
import json
import shutil
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VERL_ROOT = os.path.join(PROJECT_ROOT, "pyvision-rl", "verl_agents")
for p in (PROJECT_ROOT, VERL_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

VSTAR = os.path.join(PROJECT_ROOT, "datasets", "vstar_bench")

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# —————————————————————————————————————
# 1. VStarToolEnv
# —————————————————————————————————————
print("=== 1. VStarToolEnv ===")
from atr.adapter.vstar_env import VStarToolEnv, parse_tool_call, extract_answer  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from verl.workers.agent.tool_envs import ToolBase  # noqa: E402
    check("env 已注册(ToolBase.registry)",
          "vstar_tool_env" in ToolBase.registry,
          f"registered={ToolBase.registry.get('vstar_tool_env').__name__ if 'vstar_tool_env' in ToolBase.registry else 'NO'}")
    env = ToolBase.create("vstar_tool_env")
except ImportError:
    print("  [SKIP] verl 未安装,注册检查跳过(env 逻辑照常测试)")
    env = VStarToolEnv(_name=None, _desc=None, _params=None)

demo_img = os.path.join(VSTAR, "direct_attributes", "sa_10033.jpg")
env.reset(raw_prompt="", multi_modal_data={"image": [Image.open(demo_img).convert("RGB")]},
          origin_multi_modal_data=None)
check("reset 载入原图", env.current_image is not None, f"size={env.current_image.size}")

# zoom(归一化 → 像素执行,含图像观测)
obs, reward, done, info = env.execute(
    '<tool_call>\n{"name": "zoom", "arguments": {"bbox_2d": [0.15, 0.25, 0.28, 0.38]}}\n</tool_call>')
check("zoom 返回 Format3 dict", isinstance(obs, dict) and "prompt" in obs and "multi_modal_data" in obs)
check("zoom 观测含 1 张 PIL 图",
      len(obs["multi_modal_data"]["image"]) == 1 and isinstance(obs["multi_modal_data"]["image"][0], Image.Image))
check("zoom 状态更新(current_image 切换)", env.current_image is not None)
check("zoom 观测文本含结果描述", "[Zoomed into" in obs["prompt"], obs["prompt"][:50].replace("\n", "\\n"))

# crop(非状态工具,不更新视图)
obs2, reward2, done2, _ = env.execute(
    '<tool_call>\n{"name": "crop", "arguments": {"bbox_2d": [0.1, 0.1, 0.5, 0.5]}}\n</tool_call>')
check("crop 返回 Format3 dict", isinstance(obs2, dict) and "multi_modal_data" in obs2)
check("crop 不更新状态图像", env.current_image.size == env.original_image.size)

# ocr(纯文本观测;本机无 tesseract 也能走通,输出 [No text detected])
obs3, reward3, done3, _ = env.execute('<tool_call>\n{"name": "ocr", "arguments": {}}\n</tool_call>')
check("ocr 返回文本观测", isinstance(obs3, str) and "<tool_response>" in obs3, obs3[:40].replace("\n", "\\n"))

# 未知工具/坏参数 → 错误文本,不结束轨迹
obs4, _, done4, _ = env.execute('<tool_call>\n{"name": "nuke", "arguments": {}}\n</tool_call>')
check("未知工具报错不结束", "[Unknown tool" in obs4 and not done4)

# answer → done
obs5, reward5, done5, _ = env.execute("<answer>The flag is white.</answer>")
check("answer 结束轨迹", done5 and obs5 == "")

# 无工具调用 → 提示作答,不结束
obs6, _, done6, _ = env.execute("I think I need to look closer.")
check("无工具调用提示作答", "Please provide your answer" in obs6 and not done6)

# 解析器
check("parse_tool_call 兼容 ```json", parse_tool_call('```json\n{"name": "zoom", "arguments": {}}\n```')["name"] == "zoom")
check("extract_answer", extract_answer("<answer> A </answer>") == "A")

# —————————————————————————————————————
# 2. 轨迹解析 + 答案判定
# —————————————————————————————————————
print("=== 2. 轨迹解析 + compute_score ===")
from atr.adapter.patch_reward import parse_tool_trajectory, approximate_tool_outputs  # noqa: E402
from atr.adapter.score import compute_vstar_score  # noqa: E402

traj_text = """<tool_call>
{"name": "zoom", "arguments": {"bbox_2d": [0.5, 0.5, 0.7, 0.7]}}
</tool_call>
<tool_response>
<image>
[Zoomed into (x1,y1)-(x2,y2)]
</tool_response>
Continue analyzing.
<tool_call>
{"name": "ocr", "arguments": {}}
</tool_call>
<tool_response>
[OCR result: "RED"]
</tool_response>
<answer>The flag is red.</answer>"""
calls = approximate_tool_outputs(traj_text, parse_tool_trajectory(traj_text))
check("解析出 2 个工具调用", len(calls) == 2, [c["tool_name"] for c in calls])
check("zoom 带归一化 bbox", calls[0]["bbox"] == [0.5, 0.5, 0.7, 0.7])
check("obs 文本近似填充", "OCR result" in calls[1].get("output", ""))

gt_text = "The flag is red."
opts = ["The flag is red.", "The flag is white."]
check("字母答案判定正确", compute_vstar_score("direct_attributes", "<answer>A</answer>", gt_text, {"options": opts})["is_answer_right"])
check("文本答案判定正确", compute_vstar_score("direct_attributes", "<answer>The flag is red.</answer>", gt_text, {"options": opts})["is_answer_right"])
check("无答案判定错误", not compute_vstar_score("direct_attributes", "no answer", gt_text, {"options": opts})["is_answer_right"])

# —————————————————————————————————————
# 3. ATR reward 闭环(RL 语义:归一化坐标 + gt_bbox 归一化)
# —————————————————————————————————————
print("=== 3. ATR reward 闭环 ===")
from atr.reward import AdaptiveToolReward  # noqa: E402
from atr.config import ATRConfig  # noqa: E402

atr = AdaptiveToolReward(config=ATRConfig(lambda_u=1.0, gamma_c=0.5, eta_s=0.3))
extra_info = {
    "question": "Is the flag red or white?",
    "options": ["The flag is red.", "The flag is white."],
    "gt_bbox": [0.55, 0.55, 0.75, 0.75],   # 归一化 GT(演示值)
}

# 3a. 正确路径:1 次 GT 对齐 zoom(IOU 0.5+)+ 答案正确
good_traj = [
    {"tool_name": "zoom", "bbox": [0.55, 0.55, 0.75, 0.75], "output": "[Zoomed into region]"},
]
r_good, comp_good = atr.compute(
    tool_calls=good_traj, final_answer="<answer>The flag is red.</answer>",
    accuracy=1.0, ground_truth=extra_info, question=extra_info["question"],
    image_size=(1.0, 1.0))
check("GT 对齐 zoom 有 utility", comp_good["utility"] > 0, f"U={comp_good['utility']:.3f}")
check("正确轨迹奖励 > 1(acc + U)", r_good > 1.0, f"R={r_good:.3f}")

# 3b. 死循环:同坐标 zoom ×3 + 答案错误
loop_traj = [
    {"tool_name": "zoom", "bbox": [0.2, 0.2, 0.4, 0.4], "output": "[Zoomed into region]"},
    {"tool_name": "zoom", "bbox": [0.2, 0.2, 0.4, 0.4], "output": "[Zoomed into region]"},
    {"tool_name": "zoom", "bbox": [0.2, 0.2, 0.4, 0.4], "output": "[Zoomed into region]"},
]
r_loop, comp_loop = atr.compute(
    tool_calls=loop_traj, final_answer="<answer>blue</answer>",
    accuracy=0.0, ground_truth=extra_info, question=extra_info["question"],
    image_size=(1.0, 1.0))
check("重复 zoom 有 cost", comp_loop["cost"] > 0, f"C={comp_loop['cost']:.3f}")
check("死循环 + 错误奖励为负", r_loop < 0, f"R={r_loop:.3f}")
check("GT 对齐优于死循环", r_good > r_loop, f"{r_good:.3f} vs {r_loop:.3f}")

# 3c. 定位偏差:zoom 偏离 GT(IOU<0.2)
miss_traj = [{"tool_name": "zoom", "bbox": [0.0, 0.0, 0.05, 0.05], "output": "[Zoomed into region]"}]
_, comp_miss = atr.compute(
    tool_calls=miss_traj, final_answer="<answer>blue</answer>",
    accuracy=0.0, ground_truth=extra_info, question=extra_info["question"],
    image_size=(1.0, 1.0))
check("定位偏差 zoom utility 低/0", comp_miss["utility"] <= 0.2, f"U={comp_miss['utility']:.3f}")

# —————————————————————————————————————
# 4. 数据转换端到端
# —————————————————————————————————————
print("=== 4. vstar_to_verl_parquet ===")
tmpdir = tempfile.mkdtemp(prefix="vstar_rl_")
try:
    from experiments.scripts.vstar_to_verl_parquet import discover_samples, build_row  # noqa: E402
    samples = discover_samples(VSTAR)
    check("发现 191 个样本", len(samples) == 191, f"n={len(samples)}")

    row = build_row(*samples[0], "direct_attributes", VSTAR, 0)
    check("prompt 含 <image> 与 system", row["prompt"][0]["role"] == "system" and "<image>" in row["prompt"][1]["content"])
    check("mm_hint 绝对路径", row["mm_hint"]["hint_type"] == "image" and os.path.isabs(row["mm_hint"]["hint_path"]))
    check("gt_bbox 归一化 [0,1]", all(0 <= v <= 1 for v in row["extra_info"]["gt_bbox"]))
    check("reward_model.ground_truth = options[0]", row["reward_model"]["ground_truth"] == row["extra_info"]["options"][0])

    out = os.path.join(tmpdir, "parquet")
    os.makedirs(out)
    rc = os.system(
        f'python "{PROJECT_ROOT}/experiments/scripts/vstar_to_verl_parquet.py" '
        f'--vstar_path "{VSTAR}" --output_dir "{out}" --image_root "{VSTAR}" --val_size 20 '
        f'> {tmpdir}/conv.log 2>&1')
    conv_tail = ""
    if os.path.exists(f"{tmpdir}/conv.log"):
        with open(f"{tmpdir}/conv.log", encoding="utf-8", errors="replace") as fh:
            conv_tail = fh.read().strip().splitlines()[-1]
    check("转换脚本退出码 0", rc == 0, conv_tail)
    import pyarrow.parquet as pq  # noqa: E402
    tr = pq.read_table(os.path.join(out, "train.parquet"))
    vl = pq.read_table(os.path.join(out, "val.parquet"))
    check("train=171 / val=20", tr.num_rows == 171 and vl.num_rows == 20, f"{tr.num_rows}/{vl.num_rows}")
    check("列齐全", set(["prompt", "mm_hint", "env_name", "data_source", "reward_model", "extra_info", "uid"]) <= set(tr.column_names))
    check("env_name 全部为 vstar_tool_env",
          set(tr.column("env_name").to_pylist()) == {"vstar_tool_env"})
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print(f"\n===== RESULT: {len(PASS)} passed, {len(FAIL)} failed =====")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
