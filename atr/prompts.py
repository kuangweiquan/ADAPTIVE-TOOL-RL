"""Prompt 构建 — RL 数据集与 VStarToolEnv 的单一真值源。

与离线评估 experiments/scripts/run_atr_offline.py 的 build_system_prompt /
build_user_prompt 输出**逐字节一致**(RL 训练语境 = SFT 先答后验语境),
<tools> 段由 atr.tools 注册表生成。
"""

import json
from typing import List, Optional, Tuple

from .tools import get_tool_schemas


def build_system_prompt(tool_required: bool = False) -> str:
    """由 atr.tools 注册表生成 SYSTEM_PROMPT(含坐标空间约定与工作流示例)。

    tool_required=True 时切换为"先工具核实后作答"策略(用于采集工具轨迹)。
    RL 训练使用默认先答后验版(tool_required=False)。
    """
    schemas = "\n".join(json.dumps(s) for s in get_tool_schemas())
    if tool_required:
        answer_policy = (
            "# Answer policy (IMPORTANT)\n"
            "1. BEFORE answering, you MUST call a tool to verify the target object:\n"
            "   zoom into the relevant region (or crop it) and inspect the returned image.\n"
            "2. Call ONE tool per turn; you may call several tools in sequence.\n"
            "3. After inspecting tool results, give your final answer in <answer> tags."
        )
    else:
        answer_policy = (
            "# Answer policy (IMPORTANT)\n"
            "1. FIRST, answer directly from the full image: <answer>your answer</answer>\n"
            "2. ONLY if you cannot determine the answer from the full image\n"
            "   (object too small, text unreadable, details unclear) may you call a tool\n"
            "   to inspect. Call ONE tool per turn, inspect the returned image, then answer.\n"
            "3. After inspecting tool results, give your final answer in <answer> tags."
        )
    return f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{schemas}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Coordinate system (IMPORTANT)
All bbox_2d values MUST be NORMALIZED coordinates: each value is a fraction
of the image size in [0, 1], where (0,0) = top-left corner and (1,1) = bottom-right
corner of the image you are currently viewing. Do NOT use pixel values.
Your coordinates are scaled automatically by the tool executor.

{answer_policy}

Tools available:
<tool_call>
{{"name": "zoom", "arguments": {{"bbox_2d": [0.15, 0.25, 0.28, 0.38]}}}}
</tool_call>
Use ocr to read text, rotate only if the image orientation is wrong.

When you have enough evidence, provide your answer inside <answer> tags:
<answer>your answer</answer>"""


def build_user_prompt(
    question: str,
    options: Optional[List[str]] = None,
    display_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Build user prompt with question and options(与离线评估逐字节一致)。"""
    prompt = f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
        for i, opt in enumerate(options):
            prompt += f"{abc_map.get(i + 1, str(i + 1))}. {opt}\n"
    prompt += ("\nAnswer directly from the full image with the correct option letter in <answer> tags. "
               "Use a tool only if you cannot determine the answer from the full image.")
    return prompt
