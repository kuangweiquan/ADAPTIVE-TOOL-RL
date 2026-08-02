"""Configuration for Adaptive Tool Reward."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ATRConfig:
    """Adaptive Tool Reward hyperparameters.

    R = acc + λ·U − γ·C + η·S

    Attributes:
        lambda_u: Weight for tool utility (positive reward for useful tools).
        gamma_c: Weight for tool cost (penalty for redundant tools).
        eta_s: Weight for sequence quality (reward for good tool ordering).
        utility_mode: How to compute utility — "rule_based" (no extra model) or "learned".
        cost_mode: How to detect redundancy — "exact_dup", "iou_threshold", or "semantic".
        sequence_mode: How to evaluate sequence — "pattern_based" or "learned".
        iou_threshold: IoU threshold for duplicate crop detection.
        text_sim_threshold: Text similarity threshold for duplicate OCR detection.
        frame_time_threshold: Frame timestamp proximity for duplicate frame reading.
        verbose: Whether to log per-step reward components for debugging.
    """
    lambda_u: float = 1.0
    gamma_c: float = 0.5
    eta_s: float = 0.3

    utility_mode: str = "rule_based"
    cost_mode: str = "exact_dup"
    sequence_mode: str = "pattern_based"

    iou_threshold: float = 0.5
    text_sim_threshold: float = 0.85
    frame_time_threshold: float = 1.0  # seconds

    verbose: bool = False

    # ——————— PyVision-RL original ———————
    original_tool_reward: float = 0.1

    # ——————— Ablation flags ———————
    enable_utility: bool = True
    enable_cost: bool = True
    enable_sequence: bool = True

    # ——————— AdaReasoner-inspired features ———————
    enable_format_gate: bool = False      # multiplicative format gate (R_format)
    format_gate_strict: bool = False      # True = any format error → R=0
    enable_asymmetric_fusion: bool = False  # AdaReasoner-style asymmetric fusion
    asymmetric_lambda_s: float = 0.3      # S weight when answer is wrong
    asymmetric_lambda_call: float = 0.7   # per-call validity weight when wrong

    # ——————— CodeV-inspired features ———————
    enable_question_aware_utility: bool = True  # question-aware utility scoring
    enable_lazy_crop_penalty: bool = True       # penalty for uninformative large crops

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def baseline(cls) -> "ATRConfig":
        """Return a config equivalent to PyVision-RL's original reward."""
        return cls(
            lambda_u=0.0, gamma_c=0.0, eta_s=0.0,
            enable_utility=False, enable_cost=False, enable_sequence=False,
        )

    @classmethod
    def utility_only(cls) -> "ATRConfig":
        """Ablation: only utility, no cost or sequence."""
        return cls(
            gamma_c=0.0, eta_s=0.0,
            enable_utility=True, enable_cost=False, enable_sequence=False,
        )

    @classmethod
    def cost_only(cls) -> "ATRConfig":
        """Ablation: only cost, no utility or sequence."""
        return cls(
            lambda_u=0.0, eta_s=0.0,
            enable_utility=False, enable_cost=True, enable_sequence=False,
        )

    @classmethod
    def sequence_only(cls) -> "ATRConfig":
        """Ablation: only sequence, no utility or cost."""
        return cls(
            lambda_u=0.0, gamma_c=0.0,
            enable_utility=False, enable_cost=False, enable_sequence=True,
        )

    @classmethod
    def no_sequence(cls) -> "ATRConfig":
        """Ablation: utility + cost, no sequence."""
        return cls(
            eta_s=0.0,
            enable_utility=True, enable_cost=True, enable_sequence=False,
        )
