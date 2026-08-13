"""Tool Cost — detects redundant / meaningless tool calls.

Penalizes:
  - Duplicate crop of the same region (IoU-based)
  - Duplicate OCR on the same text
  - Zoom-in then zoom-out (pathological oscillation)
  - Consecutive identical tools (spam)
"""

from typing import List, Dict, Any, Tuple
import re

from ..tools import registry


class ToolCost:
    """Detect and penalize redundant tool calls in a trajectory."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        text_sim_threshold: float = 0.85,
        frame_time_threshold: float = 1.0,
        call_budget: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.text_sim_threshold = text_sim_threshold
        self.frame_time_threshold = frame_time_threshold
        # Type 6: calls beyond this budget are penalized (over-calling detector).
        self.call_budget = call_budget

    def compute(self, tool_calls: List[Dict[str, Any]]) -> List[float]:
        """Return a cost score in [0, 1] per tool call.

        Returns:
            cost_scores: list of floats, same length as tool_calls.
                Higher = more redundant (more costly).
        """
        costs, _ = self.compute_with_stats(tool_calls)
        return costs

    def compute_with_stats(self, tool_calls: List[Dict[str, Any]]):
        """Compute per-call costs plus per-type firing counts.

        The counts are the silent-reward-death diagnostic: a cost term that
        never fires on the actual policy distribution contributes zero
        gradient, and per-type counts tell us WHICH detector is blind.
        """
        costs = [0.0] * len(tool_calls)
        stats = {
            "dup_spatial": 0,
            "dup_ocr": 0,
            "oscillation": 0,
            "consecutive_same": 0,
            "call_budget": 0,
        }

        # === Type 1: Duplicate spatial operations ===
        # 空间工具集合由 atr.tools 注册表派生(当前: crop, zoom)
        spatial_indices = [
            i for i, c in enumerate(tool_calls)
            if c.get("tool_name", "").lower() in registry.spatial_tools
               and c.get("bbox") is not None
        ]
        for i in range(len(spatial_indices)):
            for j in range(i + 1, len(spatial_indices)):
                idx_i = spatial_indices[i]
                idx_j = spatial_indices[j]
                iou = self._compute_iou(
                    tool_calls[idx_i]["bbox"],
                    tool_calls[idx_j]["bbox"],
                )
                if iou > self.iou_threshold:
                    # Penalize the later duplicate
                    costs[idx_j] = max(costs[idx_j], 0.6)
                    stats["dup_spatial"] += 1

        # === Type 2: Duplicate OCR on similar text ===
        ocr_indices = [
            i for i, c in enumerate(tool_calls)
            if c.get("tool_name", "").lower() == "ocr"
               and c.get("output")
        ]
        for i in range(len(ocr_indices)):
            for j in range(i + 1, len(ocr_indices)):
                idx_i = ocr_indices[i]
                idx_j = ocr_indices[j]
                sim = self._text_similarity(
                    tool_calls[idx_i]["output"],
                    tool_calls[idx_j]["output"],
                )
                if sim > self.text_sim_threshold:
                    costs[idx_j] = max(costs[idx_j], 0.6)
                    stats["dup_ocr"] += 1

        # === Type 3: (removed) repeated frame reads — 视频工具未实现 ===
        # === Type 4: Zoom oscillation (zoom in → zoom out → same region) ===
        for i in range(1, len(tool_calls) - 1):
            prev = tool_calls[i - 1].get("tool_name", "").lower()
            curr = tool_calls[i].get("tool_name", "").lower()
            nxt = tool_calls[i + 1].get("tool_name", "").lower()
            if prev == "zoom" and curr in ("zoom", "crop") and nxt == "zoom":
                costs[i] = max(costs[i], 0.4)
                costs[i + 1] = max(costs[i + 1], 0.4)
                stats["oscillation"] += 1

        # === Type 5: Consecutive identical tools (spam) ===
        for i in range(1, len(tool_calls)):
            if tool_calls[i].get("tool_name") == tool_calls[i - 1].get("tool_name"):
                costs[i] = max(costs[i], 0.3)
                stats["consecutive_same"] += 1

        # === Type 6: Call budget (over-calling detector) ===
        # 观测行为:tool_call_mean 5.6-6.5/max_turns=8 的过量调用。
        # 预算外的每次调用给温和惩罚,让 C 对"调用量"有判别力。
        for i in range(len(tool_calls)):
            if i >= self.call_budget:
                costs[i] = max(costs[i], 0.2)
                stats["call_budget"] += 1

        return costs, stats

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """Simple token-overlap similarity between two text outputs."""
        if not t1 or not t2:
            return 0.0
        tokens1 = set(re.findall(r'\w+', t1.lower()))
        tokens2 = set(re.findall(r'\w+', t2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        return len(intersection) / max(len(tokens1), len(tokens2))

    @staticmethod
    def aggregate(cost_scores: List[float]) -> float:
        """Aggregate per-call costs into the overall C term.

        C = Σ (positive cost values)
        """
        return sum(max(0.0, c) for c in cost_scores)
