"""verl-tool reward manager adapter for the four-arm stitch ablation.

Drop-in for their `AdatoolerVRewardManager` (reward_manager=adatooler_v):
set `reward_model.reward_manager=adatooler_v_stitch` and pass
`reward_model.reward_kwargs` with `arm` ("a"|"b"|"c"|"d") plus optional
`atr_config_dict` (arm c) and `delta_s_map` (arm d, path to the JSONL
produced by delta_s_precompute.py).

Per-sample logic is factored into `compute_sample_reward` (no DataProto
dependency) so the local CPU smoke can cover it without verl installed.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# project root on sys.path so `atr` and `experiments.adatooler_stitch` import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJECT_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.adatooler_stitch.arm_logic import (  # noqa: E402
    ARM_NAMES,
    compute_arm_reward,
    silent_death_check,
)

from atr.adapter.patch_reward import (  # noqa: E402
    approximate_tool_outputs,
    parse_tool_trajectory,
)
from atr.config import ATRConfig  # noqa: E402
from atr.reward import AdaptiveToolReward  # noqa: E402


# ————————————————————————————————————————————————
#  Local answer scoring (mirror of their choices_compute_score,
#  adatooler_v.py:67-82 — used when verl_tool is not importable)
# ————————————————————————————————————————————————
def _choices_score_local(predict: str, ground_truth: List[str]) -> float:
    m = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    predict_strip = m.group(1).strip() if m else predict.strip()
    if predict_strip == "":
        return 0.0
    answer = [a.strip() for a in predict_strip.split(",")]
    gt = [re.sub(r"</?answer>", "", str(x)).strip() for x in ground_truth]
    if len(answer) != len(gt):
        return 0.0
    return 1.0 if all(a in gt for a in answer) else 0.0


def _load_their_scorer():
    """Lazy import their adatooler_reasoner_score; None if unavailable."""
    try:
        from verl_tool.workers.reward_manager.adatooler_v import (
            adatooler_reasoner_score,
        )
        return adatooler_reasoner_score
    except Exception:
        return None


def resolve_image_size(extra_info) -> Optional[Tuple[int, int]]:
    """Read (w, h) of the first image referenced in extra_info.

    Their prompts teach pixel-space bbox_2d ("minimum 0, maximum width/height"),
    our atr core assumes normalized [0,1] — so pixel coords must be converted
    before rule-based U/C/S see them. Returns None when unavailable; the
    silent-death counters will then flag any dead spatial detector.
    """
    try:
        from PIL import Image
        if isinstance(extra_info, dict):
            images = extra_info.get("images") or extra_info.get("image") or []
            if isinstance(images, (str, dict)):
                images = [images]
            for img in images:
                if isinstance(img, str) and os.path.exists(img):
                    with Image.open(img) as im:
                        return im.size
                if isinstance(img, dict) and "image" in img and os.path.exists(str(img["image"])):
                    with Image.open(str(img["image"])) as im:
                        return im.size
    except Exception:
        return None
    return None


def normalize_bboxes(tool_calls: List[Dict[str, Any]],
                     image_size: Optional[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Convert pixel-space bboxes to normalized [0,1] in place (shallow copy).

    Detection: any coordinate > 1.0 ⇒ pixel space (their prompt convention).
    """
    if not image_size or not tool_calls:
        return tool_calls
    w, h = image_size
    for call in tool_calls:
        bbox = call.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        try:
            vals = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if any(v > 1.0 for v in vals):
            call["bbox"] = [vals[0] / w, vals[1] / h, vals[2] / w, vals[3] / h]
    return tool_calls


def compute_sample_reward(
    arm: str,
    response_str: str,
    ground_truth: Any,
    problem_type: str,
    tool_interact_info: Optional[List[Dict[str, Any]]],
    extra_info: Optional[Dict[str, Any]],
    atr: AdaptiveToolReward,
    delta_s_lookup=None,
    their_scorer=None,
) -> Tuple[float, Dict[str, float]]:
    """Reward for one decoded rollout response (verl-free)."""
    # 1. accuracy via their scorer (multiple choice etc.), local MCQ fallback
    if their_scorer is not None:
        score = their_scorer(response_str, ground_truth, problem_type)
    elif problem_type == "multiple choice":
        score = _choices_score_local(response_str, list(ground_truth))
    else:
        raise ValueError(
            f"problem_type={problem_type!r} requires verl_tool scorer on remote; "
            "only 'multiple choice' has a local fallback (CPU smoke)"
        )
    accuracy = 1.0 if score > 0 else 0.0

    # 2. tool calls: structured stats from the agent loop are authoritative;
    #    parsed + canonicalized records drive the rule-based C/U/S signals.
    #    Their prompts teach pixel-space bboxes → normalize before ATR sees them.
    tool_calls = parse_tool_trajectory(response_str)
    tool_calls = approximate_tool_outputs(response_str, tool_calls)
    tool_calls = normalize_bboxes(tool_calls, resolve_image_size(extra_info))
    num_valid_actions = None
    if tool_interact_info:
        num_valid_actions = sum(
            1 for t in tool_interact_info if t.get("valid_action", False)
        )

    # 3. precomputed Delta-S (arm d); default 1.0 = their released stub
    delta_s = 1.0
    if extra_info and "Tool_Benefit_Score" in extra_info:
        delta_s = float(extra_info["Tool_Benefit_Score"])
    elif delta_s_lookup is not None:
        uid = None
        if extra_info and isinstance(extra_info, dict):
            uid = extra_info.get("id") or extra_info.get("uid")
        if uid is not None and uid in delta_s_lookup:
            delta_s = float(delta_s_lookup[uid])

    # 4. arm reward
    return compute_arm_reward(
        arm=arm,
        accuracy=accuracy,
        tool_calls=tool_calls,
        num_valid_actions=num_valid_actions,
        delta_s=delta_s,
        atr=atr,
        final_answer=response_str,
    )


# ————————————————————————————————————————————————
#  verl-tool reward manager (registered name: adatooler_v_stitch)
# ————————————————————————————————————————————————
class AdatoolerVStitchRewardManager:
    """Mirrors AdatoolerVRewardManager's interface (adatooler_v.py:173-183)."""

    name = "adatooler_v_stitch"

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.arm = str(kwargs.get("arm", "a"))
        if self.arm not in ARM_NAMES:
            raise ValueError(f"arm must be one of {sorted(ARM_NAMES)}, got {self.arm!r}")
        self.atr = AdaptiveToolReward(ATRConfig(**(kwargs.get("atr_config_dict") or {})))
        self.their_scorer = _load_their_scorer()
        self.delta_s_lookup = None
        delta_s_map_path = kwargs.get("delta_s_map")
        if delta_s_map_path and os.path.exists(delta_s_map_path):
            with open(delta_s_map_path) as f:
                self.delta_s_lookup = {
                    str(item["uid"]): float(item["delta_s"])
                    for item in (json.loads(line) for line in f if line.strip())
                }
        self.step = 0

    def __call__(self, data, return_dict=False):
        import torch
        from collections import defaultdict

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        batch_components: List[Dict[str, float]] = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            problem_type = (
                extra_info.get("problem_type", "multiple choice")
                if isinstance(extra_info, dict) else "multiple choice"
            )
            tool_interact_info = data_item.non_tensor_batch.get("tool_interact_info", None)

            reward, components = compute_sample_reward(
                arm=self.arm,
                response_str=response_str,
                ground_truth=ground_truth,
                problem_type=problem_type,
                tool_interact_info=tool_interact_info,
                extra_info=extra_info,
                atr=self.atr,
                delta_s_lookup=self.delta_s_lookup,
                their_scorer=self.their_scorer,
            )
            reward_tensor[i, valid_response_length - 1] = reward
            batch_components.append(components)

            for key, value in components.items():
                reward_extra_info[key].append(value)
            reward_extra_info["is_answer_right"].append(components["acc"] > 0.5)
            reward_extra_info["acc_score"].append(components["acc"])

            if self.num_examine and i < self.num_examine:
                cstr = " ".join(
                    f"{k}={v}" for k, v in components.items()
                    if k.startswith("cost_") or k in ("acc", "n_tool", "at_penalty", "delta_s", "total")
                )
                print(f"[ATR-STITCH arm={self.arm}] {cstr}")

        # silent-death audit: per-batch constant components get no GRPO signal
        death = silent_death_check(batch_components)
        dead_keys = [k for k, v in death.items() if v["dead"]]
        if dead_keys:
            print(f"[ATR-STITCH arm={self.arm}] DEAD_COMPONENTS(step {self.step}): {dead_keys}")
        self.step += 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


# register with verl_tool when importable (remote)
try:
    from verl.workers.reward_manager import register
    register("adatooler_v_stitch")(AdatoolerVStitchRewardManager)
except Exception:
    pass  # local CPU smoke: registration happens on the remote side
