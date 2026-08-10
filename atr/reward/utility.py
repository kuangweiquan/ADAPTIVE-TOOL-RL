"""Tool Utility — inspired by CodeV's Tool-Aware Policy Optimization (TAPO).

CodeV insight: evaluate each tool call by whether it provides EVIDENCE
needed to ANSWER THE QUESTION, not just whether its output appears in
the final answer text.

Our implementation: rule-based (zero extra model cost).
  - Question-aware: tool output provides evidence for answering the question
  - Spatial utility: targeted crop/zoom (not lazy full-image crops)
  - Information gain: OCR/video frame provides new information
  - Progressive refinement: spatial tools that zoom in are rewarded
"""

from typing import List, Dict, Any, Optional
import re

from ..tools import registry


class ToolUtility:
    """Compute utility score for each tool call in a trajectory.

    A tool is "useful" if its output provides evidence needed to answer
    the question — regardless of whether that evidence appears verbatim
    in the final answer.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        # — New: lazy crop detection —
        max_lazy_crop_ratio: float = 0.85,
    ):
        """
        Args:
            iou_threshold: IoU threshold for GT bbox matching.
            max_lazy_crop_ratio: Max (crop area / image area) before
                a crop is deemed "lazy" (uninformatively large).
        """
        self.iou_threshold = iou_threshold
        self.max_lazy_crop_ratio = max_lazy_crop_ratio

    def compute(
        self,
        tool_calls: List[Dict[str, Any]],
        final_answer: str,
        ground_truth: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,   # ← NEW: question text
        image_size: Optional[tuple] = None,  # ← NEW: (width, height) for lazy crop detection
    ) -> List[float]:
        """Return a utility score in [0, 1] per tool call.

        Args:
            tool_calls: List of tool invocation records.
            final_answer: The model's final answer text.
            ground_truth: Optional GT info.
            question: The input question text (for question-aware scoring).
            image_size: Original image (width, height) for lazy crop detection.

        Returns:
            utility_scores: List of floats, same length as tool_calls.
        """
        scores = []
        for call in tool_calls:
            score = self._score_single_call(
                call=call,
                answer=final_answer,
                question=question,
                gt=ground_truth,
                image_size=image_size,
                all_calls=tool_calls,
            )
            scores.append(score)
        return scores

    def _score_single_call(
        self,
        call: Dict[str, Any],
        answer: str,
        question: Optional[str] = None,
        gt: Optional[Dict[str, Any]] = None,
        image_size: Optional[tuple] = None,
        all_calls: Optional[List[Dict]] = None,
    ) -> float:
        """Score a single tool call's utility.

        Scoring dimensions (max ~1.0):
          - Evidence-to-question relevance: +0.4
          - Spatial precision: +0.3
          - Information gain: +0.3
          - Anti-lazy-crop penalty: -0.3 cap
        """
        tool = call.get("tool_name", "").lower()
        output = str(call.get("output", ""))
        score = 0.0

        # ================================================================
        # Dimension 1: Evidence-to-Question Relevance  (max +0.4)
        # Does the tool output contain information that helps answer
        # the question?  Independent of whether it appears in `answer`.
        # ================================================================
        if output and output.lower() != "none" and output.lower() != "null":
            # 1a. Question-aware keyword match (preferred)
            if question:
                match_score = self._question_relevance(output, question)
                score += match_score * 0.4
            else:
                # 1b. Fallback: output appears in answer (original logic)
                key_tokens = [t for t in re.findall(r'\b\w{3,}\b', output.lower())]
                if key_tokens:
                    answer_lower = answer.lower()
                    match_ratio = sum(1 for t in key_tokens if t in answer_lower) / len(key_tokens)
                    if match_ratio > 0.3:
                        score += 0.2  # reduced weight

            # 1c. Output contains numerical/categorical data likely to be the answer
            #     (e.g., OCR reading "42" when question asks "How many...")
            numbers_output = set(re.findall(r'\d+\.?\d*', output))
            if numbers_output and question:
                if any(kw in question.lower() for kw in ["how many", "count", "number", "what is", "what's"]):
                    score += 0.15

            # 1d. Output disambiguates — contains distinctive content
            #     (non-stopword tokens not in question, suggesting new info)
            if question:
                q_words = set(question.lower().split())
                out_phrases = set(re.findall(r'\b\w{4,}\b', output.lower()))
                novel_info = out_phrases - q_words
                if len(novel_info) >= 3:
                    score += 0.15

        # ================================================================
        # Dimension 2: Spatial Precision  (max +0.3)
        # Does spatial tool target the correct region?
        # 空间工具集合由 atr.tools 注册表派生(当前: crop, zoom)
        # ================================================================
        if tool in registry.spatial_tools:

            # 2a. Matches GT bbox
            if gt and "gt_bbox" in gt:
                bbox = call.get("bbox")
                if bbox and len(bbox) == 4 and len(gt["gt_bbox"]) == 4:
                    iou = self._compute_iou(bbox, gt["gt_bbox"])
                    if iou > self.iou_threshold:
                        score += 0.3
                    elif iou > 0.2:
                        score += 0.1  # partial overlap

            # 2b. Anti-lazy-crop: penalize near-full-image crops
            if image_size and call.get("bbox"):
                w, h = image_size
                bx1, by1, bx2, by2 = call["bbox"]
                crop_area = (bx2 - bx1) * (by2 - by1)
                img_area = w * h
                if img_area > 0 and (crop_area / img_area) > self.max_lazy_crop_ratio:
                    score -= 0.3  # penalty caps utility at 0.7 max

                    # Extra penalty: if this exact large crop was done before
                    if all_calls:
                        prev_bboxes = [
                            c.get("bbox") for c in all_calls
                            if c.get("tool_name", "").lower() in registry.spatial_tools
                            and c.get("bbox") and c is not call
                        ]
                        for pb in prev_bboxes:
                            if self._compute_iou(call["bbox"], pb) > 0.9:
                                score -= 0.2
                                break

        # ================================================================
        # Dimension 3: Information Gain  (max +0.3)
        # Does the tool provide NEW info not already available?
        # ================================================================
        if tool == "ocr" and output:
            # OCR that extracts numbers/measurements from an image is high-value
            if re.search(r'\d+', output):
                score += 0.15
            # OCR that extracts structured data (labels, readings)
            if re.search(r':|°|%|mm|cm|kg|hz', output, re.IGNORECASE):
                score += 0.15
            # OCR not already available from prior OCR calls
            if all_calls:
                for prev in all_calls:
                    if prev is call:
                        break
                    if prev.get("tool_name", "").lower() == "ocr":
                        prev_out = str(prev.get("output", ""))
                        if self._text_similarity(output, prev_out) > 0.85:
                            score -= 0.2  # redundant OCR
                            break

        # ================================================================
        # Progressive refinement bonus
        # A crop that significantly reduces area from previous crop is
        # evidence of targeted analysis → bonus
        # ================================================================
        if tool in registry.spatial_tools and call.get("bbox") and all_calls:
            for prev in reversed(all_calls):
                if prev is call:
                    break
                if prev.get("tool_name", "").lower() in registry.spatial_tools and prev.get("bbox"):
                    prev_area = (prev["bbox"][2] - prev["bbox"][0]) * (prev["bbox"][3] - prev["bbox"][1])
                    cur_area = (call["bbox"][2] - call["bbox"][0]) * (call["bbox"][3] - call["bbox"][1])
                    if prev_area > 0 and cur_area < prev_area * 0.5:
                        score += 0.15  # reward progressive zoom-in
                    break

        return max(min(score, 1.0), -0.5)

    @staticmethod
    def _question_relevance(output: str, question: str) -> float:
        """Estimate how relevant tool output is to answering the question.

        Uses keyword overlap between question entities and output content.
        Returns 0.0–1.0 score.
        """
        # Extract key nouns/entities from question
        q_lower = question.lower()

        # Question type words that indicate what to look for
        # "what color" → look for color words in output
        # "how many" → look for numbers
        # "is there" → look for existence words
        # "where is" → look for location/spatial words
        q_words = set(re.findall(r'\b\w{3,}\b', q_lower))
        o_words = set(re.findall(r'\b\w{3,}\b', output.lower()))

        if not q_words or not o_words:
            return 0.0

        # Direct keyword overlap
        overlap = q_words & o_words
        if overlap:
            return 0.6

        # Semantic proximity: question asks about a specific attribute
        # and output provides that type of information
        if re.search(r'color|colour', q_lower) and re.search(r'red|blue|green|yellow|white|black|pink|purple|orange|brown|gray|grey', output.lower()):
            return 0.8
        if re.search(r'number|count|how many', q_lower) and re.search(r'\d+', output):
            return 0.7
        if re.search(r'where|location|position|side', q_lower) and re.search(r'left|right|top|bottom|center|middle|front|back|behind|above|below', output.lower()):
            return 0.7
        if re.search(r'shape|size', q_lower) and re.search(r'large|small|big|tiny|round|square|rectangular|circular', output.lower()):
            return 0.6
        if re.search(r'yes|no|is there|does|contains|has|have', q_lower):
            # Existence question — if output is non-empty, it's likely relevant
            if len(output) > 20:
                return 0.5

        return 0.0

    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """Simple token-overlap similarity."""
        if not t1 or not t2:
            return 0.0
        tokens1 = set(re.findall(r'\w+', t1.lower()))
        tokens2 = set(re.findall(r'\w+', t2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        inter = tokens1 & tokens2
        return len(inter) / max(len(tokens1), len(tokens2))

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two axis-aligned boxes [x1,y1,x2,y2]."""
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
    def aggregate(utility_scores: List[float]) -> float:
        """Aggregate per-call utilities into the overall U term.

        U = Σ utility scores (allow negative to reflect cost)
        """
        return sum(utility_scores)
