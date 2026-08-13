"""AdaTooler-V stitch ablation: pure reward logic for the four arms.

No verl / torch dependency — this module must be importable on the local
CPU machine for smoke tests. The verl-tool adapter lives in
`verl_reward_manager.py` (remote only).

Arm definitions (see ai_handoff/10_adatooler_stitch_plan.md §6):
  a: plain accuracy        — = released-code behavior (their penalty call is
                             commented out, adatooler_v.py:326)
  b: AT call-count penalty — re-enable `add_additional_penalties` with
                             TB_score=1.0 (their adatooler_v.py:229 stub)
  c: ATR rule U/C/S        — our components, zero-judge rule-based signals
  d: judge Delta-S         — paper-intent gating: precomputed per-sample
                             Tool Benefit Score multiplies the AT penalty
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

ARM_NAMES = {
    "a": "plain_acc",
    "b": "at_call_penalty",
    "c": "atr_rule_ucs",
    "d": "judge_delta_s",
}

# AT reward hyperparameters (their adatooler_v.py:186-193, verbatim)
AT_ALPHA = 0.6
AT_BETA = 0.05       # paper's beta; not used by the released compute_ATReward
AT_GAMMA = 2.0
AT_N_MAX = 6         # action_max_limit


def compute_at_reward(delta_s: float, n_tool: int, n_max: int = AT_N_MAX,
                      gamma: float = AT_GAMMA) -> float:
    """Their compute_ATReward (adatooler_v.py:143-160), verbatim math.

    R = delta_s * exp(-gamma * ((n_tool - n_max) / n_max)^2)
    """
    ratio = (n_tool - n_max) / n_max
    return delta_s * math.exp(-gamma * ratio ** 2)


def compute_arm_reward(
    arm: str,
    accuracy: float,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    num_valid_actions: Optional[int] = None,
    delta_s: float = 1.0,
    atr=None,
    final_answer: str = "",
) -> Tuple[float, Dict[str, float]]:
    """Compute the reward for one trajectory under one arm.

    Args:
        arm: "a" | "b" | "c" | "d"
        accuracy: 0.0 / 1.0 (their choices_compute_score ∈ {0, 1} for MCQ)
        tool_calls: parsed + canonicalized tool-call records (our parse path)
        num_valid_actions: structured tool count from the agent loop's
            tool_interact_info (their reward path); falls back to len(tool_calls)
        delta_s: precomputed Tool Benefit Score (arm d); 1.0 = re-enable
            without gating (arm b, their released stub value)
        atr: AdaptiveToolReward instance for arm c

    Returns:
        (reward, components) where components always carries the silent-death
        diagnostics: per-type C counters (arm c) or at_penalty value (b/d).
    """
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}, choose from {sorted(ARM_NAMES)}")

    tool_calls = tool_calls or []
    if num_valid_actions is None:
        num_valid_actions = len(tool_calls)
    n = max(0, int(num_valid_actions))

    components: Dict[str, float] = {
        "acc": float(accuracy),
        "n_tool": float(n),
        "arm": float({"a": 0, "b": 1, "c": 2, "d": 3}[arm]),
    }

    if arm == "a":
        # = released-code behavior: plain accuracy GRPO
        components["at_penalty"] = 0.0
        reward = float(accuracy)

    elif arm == "b":
        # re-enable their add_additional_penalties with the released TB stub
        at_penalty = AT_ALPHA * compute_at_reward(1.0, n)
        components["at_penalty"] = at_penalty
        reward = float(accuracy) + at_penalty

    elif arm == "c":
        # our rule-based U/C/S (zero-judge). atr.compute returns the full
        # additive reward; the C fix (09 doc) must be present in the atr
        # package for the call-budget detector to fire.
        if atr is None:
            raise ValueError("arm c requires an AdaptiveToolReward instance")
        # bboxes must already be normalized [0,1] by the caller (see
        # normalize_bboxes in verl_reward_manager); atr core assumes
        # normalized space, image_size=(1.0, 1.0) matches patch_reward.
        reward, atr_components = atr.compute(
            tool_calls=tool_calls,
            final_answer=final_answer,
            accuracy=accuracy,
            image_size=(1.0, 1.0),
        )
        components.update({k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in atr_components.items()
                           if isinstance(v, (int, float))})
        components["at_penalty"] = 0.0

    elif arm == "d":
        # judge-gated AT reward: paper intent, Delta-S from precomputed
        # extra_info["Tool_Benefit_Score"] (see delta_s_precompute.py)
        at_penalty = AT_ALPHA * compute_at_reward(delta_s, n)
        components["at_penalty"] = at_penalty
        components["delta_s"] = float(delta_s)
        reward = float(accuracy) + at_penalty

    components["total"] = float(reward)
    return float(reward), components


def silent_death_check(batch_components: List[Dict[str, float]]) -> Dict[str, Any]:
    """Batch-level diagnostic: which reward components fired (varied).

    A component is 'dead' if it is constant across the batch — constant
    components contribute zero advantage signal to GRPO. This is the
    protocol from ai_handoff/09 §3 defect 3, applied per step.
    """
    keys = set()
    for c in batch_components:
        keys.update(c.keys())
    report = {}
    for k in sorted(keys):
        vals = [c[k] for c in batch_components if k in c]
        if len(vals) < 2:
            continue
        unique = len(set(vals))
        report[k] = {
            "n": len(vals),
            "unique_values": unique,
            "dead": unique <= 1,
            "min": min(vals),
            "max": max(vals),
        }
    return report
