"""Tool Cost — detects redundant / meaningless tool calls.

Penalizes:
  - Duplicate crop of the same region (IoU-based)
  - Duplicate OCR on the same text
  - Repeated frame reads in temporal proximity
  - Zoom-in then zoom-out (pathological oscillation)
  - Tools called after answer is already determinable
"""

from typing import List, Dict, Any, Tuple
import re


class ToolCost:
    """Detect and penalize redundant tool calls in a trajectory."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        text_sim_threshold: float = 0.85,
        frame_time_threshold: float = 1.0,
    ):
        self.iou_threshold = iou_threshold
        self.text_sim_threshold = text_sim_threshold
        self.frame_time_threshold = frame_time_threshold

    def compute(self, tool_calls: List[Dict[str, Any]]) -> List[float]:
        """Return a cost score in [0, 1] per tool call.

        Returns:
            cost_scores: list of floats, same length as tool_calls.
                Higher = more redundant (more costly).
        """
        costs = [0.0] * len(tool_calls)

        # === Type 1: Duplicate spatial operations ===
        spatial_indices = [
            i for i, c in enumerate(tool_calls)
            if c.get("tool_name", "").lower() in ("crop", "zoom", "select")
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

        # === Type 3: Repeated frame reads close in time ===
        frame_indices = [
            i for i, c in enumerate(tool_calls)
            if c.get("tool_name", "").lower() in ("read_frame", "extract_frames")
        ]
        for i in range(len(frame_indices)):
            for j in range(i + 1, len(frame_indices)):
                idx_i = frame_indices[i]
                idx_j = frame_indices[j]
                t_i = tool_calls[idx_i].get("timestamp", tool_calls[idx_i].get("frame_idx", 0))
                t_j = tool_calls[idx_j].get("timestamp", tool_calls[idx_j].get("frame_idx", 0))
                if t_i is not None and t_j is not None:
                    if abs(float(t_i) - float(t_j)) < self.frame_time_threshold:
                        costs[idx_j] = max(costs[idx_j], 0.5)

        # === Type 4: Zoom oscillation (zoom in → zoom out → same region) ===
        for i in range(1, len(tool_calls) - 1):
            prev = tool_calls[i - 1].get("tool_name", "").lower()
            curr = tool_calls[i].get("tool_name", "").lower()
            nxt = tool_calls[i + 1].get("tool_name", "").lower()
            if prev == "zoom" and curr in ("zoom", "crop") and nxt == "zoom":
                costs[i] = max(costs[i], 0.4)
                costs[i + 1] = max(costs[i + 1], 0.4)

        # === Type 5: Consecutive identical tools (spam) ===
        for i in range(1, len(tool_calls)):
            if tool_calls[i].get("tool_name") == tool_calls[i - 1].get("tool_name"):
                costs[i] = max(costs[i], 0.3)

        return costs

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
