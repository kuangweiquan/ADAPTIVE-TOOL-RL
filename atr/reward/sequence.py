"""Sequence Quality — inspired by AdaReasoner's Tool-GRPO framework.

AdaReasoner introduces TWO key ideas we adopt here:

1. **Hierarchical per-call scoring** (Structure → Name → Params → Content):
   Each tool call is scored on a 0–4 scale based on its structural validity,
   not just its position in the sequence.

2. **Format gate** via R_format (multiplicative):
   Any tool call with invalid format zeros out the entire reward.
   We implement this as a format_validity flag consumed by base_reward.py.

We retain our own pattern-based sequence ordering evaluation and combine
it with AdaReasoner's per-call quality scoring.
"""

from typing import List, Dict, Any, Tuple


class SequenceQuality:
    """Evaluate tool sequence quality combining:
      - AdaReasoner-style hierarchical per-call validity scoring
      - Pattern-based inter-call ordering evaluation
      - Format gate for multiplicative reward modulation
    """

    # — Preferred patterns (our original) —
    PREFERRED_PATTERNS: List[Tuple[Tuple[str, ...], str, float]] = [
        (("crop", "ocr"), "crop-then-read", 0.4),
        (("crop", "zoom", "ocr"), "locate-zoom-read", 0.6),
        (("zoom", "ocr"), "zoom-then-read", 0.4),
        (("read_frame", "ocr"), "frame-then-ocr", 0.4),
        (("select", "crop", "ocr"), "select-crop-read", 0.5),
        (("crop", "ocr", "answer"), "crop-read-answer", 0.5),
    ]

    # — Anti-patterns (our original) —
    ANTI_PATTERNS: List[Tuple[Tuple[str, ...], str, float]] = [
        (("ocr", "crop", "ocr"), "read-before-locate", -0.4),
        (("ocr", "zoom", "ocr"), "read-before-zoom", -0.4),
        (("crop", "crop", "crop"), "repeated-crop", -0.3),
        (("ocr", "ocr"), "repeated-ocr-no-spatial", -0.3),
    ]

    def compute(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> float:
        """Return sequence quality score S in [-1, 1].

        This is the ORIGINAL pattern-based scoring (unchanged).
        """
        if len(tool_calls) < 2:
            return 0.0

        names = [c.get("tool_name", "").lower() for c in tool_calls]
        score = 0.0

        for i in range(len(names)):
            for pattern, _, reward in self.PREFERRED_PATTERNS:
                if self._match_at(names, pattern, i):
                    score += reward

        for i in range(len(names)):
            for pattern, _, penalty in self.ANTI_PATTERNS:
                if self._match_at(names, pattern, i):
                    score += penalty

        unique_tools = len(set(names))
        if unique_tools >= 3:
            score += 0.2
        elif unique_tools >= 2:
            score += 0.1
        if unique_tools == 1 and len(names) > 1:
            score -= 0.3

        return max(min(score, 1.0), -1.0)

    # ================================================================
    # NEW: AdaReasoner-inspired hierarchical per-call scoring
    # ================================================================

    def compute_per_call_scores(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[float], bool]:
        """Score each tool call on hierarchical validity (0–4 scale).

        AdaReasoner hierarchy (Structure → Name → Param Name → Param Content):
          Level 3: Correct structure + name + params + content
          Level 2: Correct structure + name + params (content wrong)
          Level 1: Correct structure + name (params wrong)
          Level 0: Structure wrong

        Also returns `all_valid`: True if ALL tool calls pass basic format check
        (AdaReasoner's R_format gate concept).

        Returns:
            per_call_validity: List of scores in [0, 3], same length as tool_calls.
            all_format_valid:  True if every call scored ≥ 2 (structure + name ok).
        """
        if not tool_calls:
            return [], True

        scores = []
        for call in tool_calls:
            scores.append(self._score_single_call_validity(call))

        # Format gate: true if all calls have valid structure + name
        all_format_valid = all(s >= 2 for s in scores)

        return scores, all_format_valid

    def _score_single_call_validity(self, call: Dict[str, Any]) -> int:
        """Score a single tool call on 0–3 hierarchical scale.

        3 = All correct (structure + name + params + content)
        2 = Structure + name correct (params may have issues)
        1 = Structure correct, name unknown/invalid
        0 = Structure wrong (can't parse)
        """
        raw = call.get("_raw", "")
        tool_name = call.get("tool_name", "")

        # — Level 0 check: Structure —
        # Must have a tool_name and (if _raw available) proper format
        if not tool_name:
            return 0
        if raw and not raw.strip().startswith("{"):
            return 0

        # — Level 1 check: Tool name —
        valid_tools = {"crop", "zoom", "ocr", "select", "read_frame", "extract_frames",
                       "zoom_in", "zoom_out", "answer", "search"}
        if tool_name.lower() not in valid_tools:
            return 1

        # — Level 2 check: Parameter names —
        params = call.get("arguments", {}) or call.get("params", {})
        if not isinstance(params, dict):
            return 2

        # Check if param names are valid for this tool
        valid_params = {
            "crop": {"bbox", "bbox_2d", "region", "x1", "y1", "x2", "y2", "area"},
            "zoom": {"bbox", "bbox_2d", "region", "scale", "factor"},
            "zoom_in": {"bbox", "bbox_2d", "region", "scale", "factor"},
            "zoom_out": {"factor"},
            "ocr": {"bbox", "bbox_2d", "region", "area"},
            "select": {"bbox", "bbox_2d", "region", "label", "object"},
            "read_frame": {"frame_idx", "timestamp", "frame", "index"},
            "extract_frames": {"start", "end", "interval", "count", "frame_idx"},
            "search": {"query", "term", "keyword"},
        }
        allowed = valid_params.get(tool_name.lower(), set())
        if allowed and params:
            param_names = set(k.lower() for k in params.keys())
            unknown = param_names - allowed
            # Allow at most 1 unknown param
            if len(unknown) > 1:
                return 2

        # — Level 3 check: Parameter values (content) —
        if tool_name.lower() in ("crop", "zoom", "zoom_in", "select"):
            bbox = call.get("bbox") or params.get("bbox") or params.get("bbox_2d")
            if bbox:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # Bbox should have positive area
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        return 3
                    else:
                        return 2  # invalid bbox
                else:
                    return 2
            return 2  # valid structure but no bbox for spatial tool
        elif tool_name.lower() in ("read_frame",):
            frame = call.get("frame_idx") or params.get("frame_idx") or params.get("timestamp")
            if frame is not None and isinstance(frame, (int, float)):
                return 3
            return 2
        elif tool_name.lower() == "ocr":
            return 3  # OCR without params is valid (reads whole selected region)

        return 3

    def compute_format_gate(self, tool_calls: List[Dict[str, Any]]) -> float:
        """Compute multiplicative format gate R_format in [0, 1].

        AdaReasoner: R_format = ∏ R_format(τ_i)
        A single format error → R_format = 0.

        We soften to continuous: average per-call validity / 3.
        Returns 0 if any call has invalid structure (score=0).
        """
        if not tool_calls:
            return 1.0

        scores, _ = self.compute_per_call_scores(tool_calls)

        # Strict gate: any score=0 → format death
        if any(s == 0 for s in scores):
            return 0.0

        # Soft gate: average validity (normalized to [0, 1])
        return sum(scores) / (len(scores) * 3.0) if scores else 1.0

    def compute_asymmetric_fusion(
        self,
        sequence_score: float,
        per_call_scores: List[float],
        accuracy: float,
        lambda_s: float = 0.3,
        lambda_call: float = 0.7,
    ) -> Tuple[float, float, float]:
        """Compute sequence reward with AdaReasoner-style asymmetric fusion.

        AdaReasoner insight:
          - When answer is CORRECT: reward is dominated by accuracy,
            sequence quality is a small bonus (don't penalize diff strategies).
          - When answer is WRONG: reward comes from tool quality,
            sequence matters more.

        Returns:
            sequence_reward: Fused sequence reward.
            s_score: Pattern score component.
            call_score: Per-call validity component.
        """
        # — Pattern-based sequence score (our original S) —
        s_score = max(0, sequence_score)  # clamp negative to 0 for fusion

        # — Average per-call validity (AdaReasoner's R_tool) —
        if per_call_scores:
            call_score = sum(per_call_scores) / (len(per_call_scores) * 3.0)
        else:
            call_score = 0.0

        # — Asymmetric fusion —
        if accuracy > 0.5:
            # Correct: mainly accuracy, S is bonus
            # AdaReasoner: "correct answer gets full reward regardless of tool use"
            seq_reward = 0.0  # don't add S, let base_reward handle via acc
        else:
            # Wrong: reward good tool usage
            # AdaReasoner: "wrong answer depends entirely on tool usage quality"
            seq_reward = lambda_s * s_score + lambda_call * call_score

        return seq_reward, s_score, call_score

    @staticmethod
    def _match_at(names: List[str], pattern: Tuple[str, ...], start: int) -> bool:
        if start + len(pattern) > len(names):
            return False
        for i, p in enumerate(pattern):
            if names[start + i] != p:
                return False
        return True
