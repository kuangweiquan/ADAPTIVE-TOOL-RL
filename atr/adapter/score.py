"""VStar 答案判定 — verl custom_reward_function 的 compute_score。

verl 契约:compute_score(data_source, solution_str, ground_truth, extra_info)
→ {"is_answer_right": bool, "score": float}。

与离线评估 run_atr_offline.py 的 extract_answer / check_answer_correct
逻辑一致(选项精确匹配 + 字母映射)。在 05 手稿的配置里通过
`reward_model.custom_reward_function.path=atr.adapter.score,
 name=compute_vstar_score` 挂载。
"""

import re
from typing import Dict, Any, List, Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags(与离线评估一致)。"""
    pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def check_answer_correct(predicted: str, ground_truth: str, options: List[str]) -> bool:
    """Check if predicted answer matches ground truth(与离线评估逐字一致)。

    Handles letter answers (A, B, C, D) and text answers.
    """
    predicted = predicted.strip()

    # Direct match
    if predicted.lower() == ground_truth.lower():
        return True

    # Check if ground truth text appears in predicted answer
    if ground_truth.lower() in predicted.lower():
        return True

    # Letter match: if options are given, check if predicted letter maps to ground truth
    abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
    for i, opt in enumerate(options):
        letter = abc_map.get(i + 1, "")
        if letter and predicted.upper() == letter and opt == ground_truth:
            return True

    # If predicted is a letter but ground_truth is text, check what that letter maps to
    if len(predicted) == 1 and predicted.upper() in "ABCDEF":
        abc_rev = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
        idx = abc_rev.get(predicted.upper(), -1) - 1
        if 0 <= idx < len(options) and options[idx] == ground_truth:
            return True

    return False


def compute_vstar_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """verl 兼容的 compute_score:VStar 选项匹配。

    Args:
        data_source: 数据集列(如 "direct_attributes"),当前不参与判定。
        solution_str: 模型完整回复(含 <answer> 标签与工具轨迹)。
        ground_truth: reward_model.ground_truth(= options[0],VStar 约定)。
        extra_info: 数据集列,须含 "options"(选项列表)供字母映射匹配。
    """
    predicted = extract_answer(solution_str) or ""
    options = (extra_info or {}).get("options", []) or []
    is_right = bool(predicted) and check_answer_correct(predicted, ground_truth, options)
    return {"is_answer_right": is_right, "score": 1.0 if is_right else 0.0}


if __name__ == "__main__":
    # 自测
    cases = [
        ("<answer>The flag is red.</answer>", "The flag is red.", []),
        ("<answer>A</answer>", "The flag is red.",
         ["The flag is red.", "The flag is white."]),
        ("no answer here", "The flag is red.",
         ["The flag is red.", "The flag is white."]),
        ("<tool_call>{\"name\": \"zoom\", \"arguments\": {}}</tool_call>\n<answer>B</answer>",
         "The flag is red.", ["The flag is red.", "The flag is white."]),
    ]
    for sol, gt, opts in cases:
        r = compute_vstar_score("direct_attributes", sol, gt, {"options": opts})
        print(f"{sol[:50]!r} gt={gt!r} -> {r}")
