"""Adaptive Tool Reward — the core contribution.

    R = (R_format) · [acc + λ·U − γ·C + η·S]

where:
  U = Tool Utility (CodeV-inspired: question-aware evidence scoring)
  C = Tool Cost   (redundancy detection)
  S = Sequence Quality (AdaReasoner-inspired: hierarchical + pattern-based)
  R_format = Format gate (AdaReasoner-inspired: multiplicative)

AdaReasoner asymmetric fusion (optional):
    If correct: reward dominated by accuracy
    If wrong: reward comes from tool quality (U, S)
"""

from typing import List, Dict, Any, Optional, Tuple

from .utility import ToolUtility
from .cost import ToolCost
from .sequence import SequenceQuality
from ..config import ATRConfig


class AdaptiveToolReward:
    """Replace PyVision-RL's fixed 0.1-per-tool reward with Adaptive Tool Reward."""

    def __init__(self, config: Optional[ATRConfig] = None):
        self.config = config or ATRConfig()
        self.utility_evaluator = ToolUtility(
            iou_threshold=self.config.iou_threshold,
            max_lazy_crop_ratio=0.85,
        )
        self.cost_evaluator = ToolCost(
            iou_threshold=self.config.iou_threshold,
            text_sim_threshold=self.config.text_sim_threshold,
            frame_time_threshold=self.config.frame_time_threshold,
            call_budget=self.config.cost_call_budget,
        )
        self.sequence_evaluator = SequenceQuality()

    def compute(
        self,
        tool_calls: List[Dict[str, Any]],
        final_answer: str,
        accuracy: float,
        ground_truth: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        image_size: Optional[tuple] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the full adaptive reward for one trajectory.

        Args:
            tool_calls: List of tool invocation records.
            final_answer: The model's final answer string.
            accuracy: 1.0 if correct, 0.0 if wrong.
            ground_truth: Optional GT info (may contain 'question' text).
            question: The question text (for question-aware utility).
            image_size: (width, height) for lazy crop detection.

        Returns:
            total_reward: The scalar reward for this trajectory.
            components: Breakdown dict for debugging.
        """
        components = {}
        acc = float(accuracy)

        # — Extract question from ground_truth if not passed separately —
        if question is None and ground_truth and isinstance(ground_truth, dict):
            question = ground_truth.get("question", None)

        # ————————————
        #  1. Format Gate R_format  (AdaReasoner-inspired)
        # ————————————
        format_gate = 1.0
        if self.config.enable_format_gate and tool_calls:
            if self.config.format_gate_strict:
                # Strict: any invalid structure → R_format = 0
                _, all_valid = self.sequence_evaluator.compute_per_call_scores(tool_calls)
                format_gate = 1.0 if all_valid else 0.0
            else:
                # Soft: continuous [0, 1] based on per-call validity
                format_gate = self.sequence_evaluator.compute_format_gate(tool_calls)
        components["format_gate"] = format_gate

        # ————————————
        #  2. Utility U  (CodeV-inspired: question-aware)
        # ————————————
        if self.config.enable_utility and tool_calls:
            utility_kwargs = dict(
                tool_calls=tool_calls,
                final_answer=final_answer,
                ground_truth=ground_truth,
            )
            if self.config.enable_question_aware_utility:
                utility_kwargs["question"] = question
                utility_kwargs["image_size"] = image_size

            utility_scores = self.utility_evaluator.compute(**utility_kwargs)
            U = self.utility_evaluator.aggregate(utility_scores)
        else:
            U = 0.0
        components["utility"] = U

        # ————————————
        #  3. Cost C
        # ————————————
        if self.config.enable_cost and tool_calls:
            cost_scores, cost_stats = self.cost_evaluator.compute_with_stats(tool_calls)
            C = self.cost_evaluator.aggregate(cost_scores)
        else:
            cost_scores, cost_stats = [], {}
            C = 0.0
        components["cost"] = C
        # 静默奖励死亡诊断:每类检测器的触发次数进 components,训练期按步聚合
        for stat_key in ("dup_spatial", "dup_ocr", "oscillation",
                         "consecutive_same", "call_budget"):
            components[f"cost_{stat_key}"] = cost_stats.get(stat_key, 0)

        # ————————————
        #  4. Sequence S  (AdaReasoner-inspired)
        # ————————————
        S = 0.0
        per_call_validity = []
        if self.config.enable_sequence and len(tool_calls) >= 2:
            # Pattern-based ordering score
            S = self.sequence_evaluator.compute(tool_calls)

            # Per-call validity scores
            if self.config.enable_asymmetric_fusion:
                per_call_validity, _ = self.sequence_evaluator.compute_per_call_scores(tool_calls)

        components["sequence"] = S
        components["per_call_validity"] = (
            sum(per_call_validity) / len(per_call_validity) if per_call_validity else 0.0
        )

        # ————————————
        #  5. Total Reward
        # ————————————

        if self.config.enable_asymmetric_fusion and len(tool_calls) >= 1:
            # AdaReasoner-style asymmetric:
            #   Correct → acc dominates, tool quality is bonus
            #   Wrong → reward comes from tool quality
            seq_reward, s_score, call_score = self.sequence_evaluator.compute_asymmetric_fusion(
                sequence_score=S,
                per_call_scores=per_call_validity or [0],
                accuracy=acc,
                lambda_s=self.config.asymmetric_lambda_s,
                lambda_call=self.config.asymmetric_lambda_call,
            )

            if acc > 0.5:
                # Correct: full accuracy + utility/cost modulation
                total = acc + self.config.lambda_u * U - self.config.gamma_c * C
            else:
                # Wrong: reward only from tool quality
                total = seq_reward + self.config.lambda_u * max(0, U) - self.config.gamma_c * C

            components["asymmetric_seq"] = seq_reward
        else:
            # Standard additive fusion
            total = (
                acc
                + self.config.lambda_u * U
                - self.config.gamma_c * C
                + self.config.eta_s * S
            )

        # — Apply format gate (multiplicative) —
        total *= format_gate

        components["total"] = total

        if self.config.verbose:
            print(
                f"[ATR] acc={acc:.1f}  U={U:.3f}  C={C:.3f}  S={S:.3f}  "
                f"gate={format_gate:.2f}  → R={total:.3f}"
            )

        return total, components

    @staticmethod
    def original_pyvision_reward(accuracy: float, n_tool_calls: int, tool_weight: float = 0.1) -> float:
        """Original PyVision-RL reward for comparison."""
        return accuracy + tool_weight * n_tool_calls if accuracy > 0 else 0.0
