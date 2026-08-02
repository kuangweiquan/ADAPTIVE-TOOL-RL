#!/usr/bin/env python3
"""
Offline Experiment: Validate Adaptive Tool Reward (ATR) on VStar Benchmark.

Pipeline:
  1. Load VStar data (images + questions)
  2. For each sample, do multi-turn model interaction via SiliconFlow API:
     - Model sees image + question
     - Model calls tools (crop, zoom, ocr, etc.) via <tool_call> blocks
     - We execute tools locally (PIL + pytesseract)
     - We feed tool results back to model as <tool_response>
     - Model eventually outputs <answer>
  3. Save complete trajectories to JSONL
  4. Compute both ATR and original PyVision-RL rewards for each trajectory
  5. Output comparison tables + statistics

Usage:
  # Quick test with 10 samples
  python run_atr_offline.py --vstar_path /path/to/vstar_bench --quick 10

  # Full run
  python run_atr_offline.py --vstar_path /path/to/vstar_bench

  # Analyze only (reuse saved trajectories)
  python run_atr_offline.py --vstar_path /path/to/vstar_bench --analyze_only

Requires:
  pip install openai pillow pytesseract tqdm
  Also need Tesseract-OCR installed on system (for OCR tool):
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - macOS: brew install tesseract
    - Linux: apt install tesseract-ocr
"""

import os
import sys
import json
import re
import math
import argparse
import base64
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Add project root to path (for importing ATR modules)
# ============================================================
# Force UTF-8 output for Windows console compatibility
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# ATR imports (used in analysis stage)
# ============================================================
# These are imported later (analysis functions) to keep rollout
# functional even without the full atr package installed.
# from atr.reward import AdaptiveToolReward
# from atr.config import ATRConfig

# ============================================================
# Configuration
# ============================================================

# --- SiliconFlow API ---
# Priority: env var > hardcoded default (hardcoded key is for convenience)
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-gicdpzromcmlkhsltvqogmpqtczzlcqqvnpnanzcpssxywsc"
)
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1")

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# --- Interaction ---
MAX_TOOL_TURNS = 8           # Max tool calls per sample
MAX_RETRIES = 3               # API retry on failure
TEMPERATURE = 0.0
MAX_TOKENS = 10240

# --- VStar test types ---
TEST_TYPES = ["direct_attributes", "relative_position"]

# --- Image processing ---
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


# ============================================================
#  System Prompt — define available tools + output format
# ============================================================

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"crop","description":"Crop a region of the image to examine details.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box as [x1, y1, x2, y2]."}},"required":["bbox_2d"]}}}
{"type":"function","function":{"name":"ocr","description":"Extract text from a region of the image.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The region to read as [x1, y1, x2, y2]."}},"required":[]}}}
{"type":"function","function":{"name":"zoom_in","description":"Zoom into a region to see fine details.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The region to zoom into as [x1, y1, x2, y2]."}},"required":["bbox_2d"]}}}
{"type":"function","function":{"name":"select","description":"Select and identify an object in the image.","parameters":{"type":"object","properties":{"label":{"type":"string","description":"Description of what to select."}},"required":["label"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Example:
<tool_call>
{"name": "crop", "arguments": {"bbox_2d": [100, 200, 300, 400]}}
</tool_call>

After using tools, provide your answer inside <answer> tags:
<answer>your answer</answer>"""


def build_user_prompt(question: str, options: Optional[List[str]] = None) -> str:
    """Build user prompt with question and options."""
    prompt = f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
        for i, opt in enumerate(options):
            prompt += f"{abc_map.get(i + 1, str(i + 1))}. {opt}\n"
    prompt += "\nUse tools to examine the image, then answer with the correct option letter in <answer> tags."
    return prompt


# ============================================================
#  Image Utilities
# ============================================================

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """Resize while respecting pixel budget (copied from qwen-vl-utils)."""
    def round_by_factor(n, f):
        return round(n / f) * f
    def ceil_by_factor(n, f):
        return math.ceil(n / f) * f
    def floor_by_factor(n, f):
        return math.floor(n / f) * f

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_file_to_base64(image_path: str, max_size: int = 768) -> str:
    """Encode an image file to base64, optionally resizing to max_size on longest side."""
    img = Image.open(image_path)
    w, h = img.size
    if max_size and (w > max_size or h > max_size):
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ============================================================
#  Tool Environment — executes tools on images locally
# ============================================================

class ToolEnv:
    """Lightweight tool execution environment.

    Maintains the current image state (starts as full image, transforms
    with each spatial tool call). Records all tool calls for later reward
    computation.
    """

    def __init__(self, image_path: str):
        self.original_image = Image.open(image_path).convert("RGB")
        self.current_image = self.original_image.copy()
        self.original_size = self.original_image.size  # (width, height)
        self.tool_records: List[Dict[str, Any]] = []
        self._ocr_available = self._check_ocr()

    def _check_ocr(self) -> bool:
        """Check if pytesseract is available."""
        try:
            import pytesseract
            # Quick test
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _record(self, tool_name: str, arguments: dict,
                output: str, bbox: Optional[List] = None):
        """Record a tool call in the format ATR expects."""
        record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "output": output,
        }
        if bbox:
            record["bbox"] = bbox
        self.tool_records.append(record)

    def crop(self, bbox_2d: List[float]) -> Tuple[Image.Image, str]:
        """Crop the current image to the specified region.

        Args:
            bbox_2d: [x1, y1, x2, y2] in pixel coordinates of current image.

        Returns:
            (cropped_image, description_string)
        """
        x1, y1, x2, y2 = map(int, bbox_2d)
        # Clamp to image bounds
        w, h = self.current_image.size
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            desc = "[Crop: invalid bbox, region is empty]"
            self._record("crop", {"bbox_2d": bbox_2d}, output=desc, bbox=[x1, y1, x2, y2])
            return self.current_image, desc

        cropped = self.current_image.crop((x1, y1, x2, y2))
        desc = f"[Cropped region ({x1},{y1})-({x2},{y2}), size {x2-x1}×{y2-y1}]"
        self._record("crop", {"bbox_2d": bbox_2d}, output=desc, bbox=[x1, y1, x2, y2])
        return cropped, desc

    def zoom_in(self, bbox_2d: List[float]) -> Tuple[Image.Image, str]:
        """Zoom into a region (crop + resize to original view size)."""
        x1, y1, x2, y2 = map(int, bbox_2d)
        w, h = self.current_image.size
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            desc = "[ZoomIn: invalid bbox]"
            self._record("zoom", {"bbox_2d": bbox_2d}, output=desc, bbox=[x1, y1, x2, y2])
            return self.current_image, desc

        cropped = self.current_image.crop((x1, y1, x2, y2))
        # Resize back to original view size for zoom effect
        zoomed = cropped.resize((w, h), Image.BICUBIC)
        desc = f"[Zoomed into ({x1},{y1})-({x2},{y2})]"
        self._record("zoom", {"bbox_2d": bbox_2d}, output=desc, bbox=[x1, y1, x2, y2])
        return zoomed, desc

    def ocr(self, bbox_2d: Optional[List[float]] = None) -> str:
        """Extract text from a region (or full image if no bbox)."""
        if bbox_2d:
            x1, y1, x2, y2 = map(int, bbox_2d)
            w, h = self.current_image.size
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            target_img = self.current_image.crop((x1, y1, x2, y2))
        else:
            target_img = self.current_image
            bbox_2d = [0, 0, *self.current_image.size]

        text = ""
        if self._ocr_available:
            try:
                import pytesseract
                # Convert PIL to format tesseract expects
                text = pytesseract.image_to_string(target_img).strip()
            except Exception:
                text = ""
        else:
            text = ""

        if not text:
            text = "[No text detected in region]"
        else:
            text = f"[OCR result: \"{text}\"]"

        self._record("ocr", {"bbox_2d": bbox_2d}, output=text, bbox=list(bbox_2d))
        return text

    def select(self, label: str) -> str:
        """Select/identify an object."""
        output = f"[Selected: {label}]"
        self._record("select", {"label": label}, output=output)
        return output


# ============================================================
#  API Interaction
# ============================================================

def create_client():
    """Create OpenAI-compatible client for SiliconFlow."""
    return OpenAI(
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        timeout=120.0,  # 2 min per API call
        max_retries=2,
    )


def call_model(client, messages: List[Dict]) -> Optional[str]:
    """Call the model via SiliconFlow API with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [API Error] {e}, retrying ({attempt + 1}/{MAX_RETRIES})...")
                import time
                time.sleep(2 ** attempt)
            else:
                print(f"  [API Error] {e}, max retries exceeded")
                return None


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first <tool_call> block from model response text.

    Returns dict with "name" and "arguments", or None if not found.
    """
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None

    try:
        call_data = json.loads(match.group(1))
        if isinstance(call_data, list):
            call_data = call_data[0] if call_data else {}
        return {
            "name": call_data.get("name", "unknown"),
            "arguments": call_data.get("arguments", {}),
        }
    except (json.JSONDecodeError, Exception):
        # Fallback to eval
        try:
            call_data = eval(match.group(1))
            if isinstance(call_data, list):
                call_data = call_data[0] if call_data else {}
            return {
                "name": call_data.get("name", "unknown"),
                "arguments": call_data.get("arguments", {}),
            }
        except Exception:
            return None


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def check_answer_correct(predicted: str, ground_truth: str, options: List[str]) -> bool:
    """Check if predicted answer matches ground truth.

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


# ============================================================
#  Single Sample Rollout
# ============================================================

def run_single_sample(
    client,
    image_path: str,
    anno: Dict[str, Any],
    max_turns: int = MAX_TOOL_TURNS,
) -> Dict[str, Any]:
    """Run the full multi-turn interaction for one VStar sample.

    Returns a dict with:
        - image: filename
        - question: question text
        - options: list of options
        - ground_truth: correct answer (options[0])
        - predicted_answer: model's final answer
        - accuracy: 1.0 if correct, 0.0 if wrong
        - tool_calls: list of tool call records (for ATR)
        - trajectory: full message history (for debugging)
        - status: "success" or "error"
        - image_size: (width, height) of original image
    """
    question = anno["question"]
    options = anno["options"]
    ground_truth = options[0]  # VStar convention: first option is correct

    # Encode image
    base64_image = encode_image_file_to_base64(image_path)

    # Initialize tool env
    env = ToolEnv(image_path)

    # Build initial messages
    user_prompt = build_user_prompt(question, options)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # Debug log (without actual image data)
    debug_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    predicted_answer = ""
    status = "success"
    answer_extracted = False

    for turn in range(max_turns):
        # Call model
        response_text = call_model(client, messages)
        if response_text is None:
            status = "error"
            break

        # Debug log
        debug_messages.append({"role": "assistant", "content": response_text})

        # Check for answer
        answer = extract_answer(response_text)
        if answer:
            predicted_answer = answer
            answer_extracted = True
            break

        # Check for tool call
        tool_call = parse_tool_call(response_text)
        if tool_call is None:
            # No tool call and no answer — model might be thinking.
            # If nothing to do, break.
            debug_messages.append({
                "role": "user",
                "content": "Please provide your answer using <answer>...</answer> tags."
            })
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Please provide your answer using <answer>...</answer> tags."
            })
            continue

        # Execute tool
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        tool_result = ""

        try:
            if tool_name in ("crop",):
                bbox = tool_args.get("bbox_2d") or tool_args.get("bbox")
                if bbox:
                    _, desc = env.crop(bbox)
                    tool_result = desc
                else:
                    tool_result = "[Error: crop requires bbox_2d]"

            elif tool_name in ("zoom_in", "zoom"):
                bbox = tool_args.get("bbox_2d") or tool_args.get("bbox")
                if bbox:
                    new_img, desc = env.zoom_in(bbox)
                    env.current_image = new_img
                    # Encode the new image to send back to model
                    new_img_b64 = pil_to_base64(new_img)
                    tool_result = desc
                else:
                    tool_result = "[Error: zoom_in requires bbox_2d]"

            elif tool_name in ("ocr",):
                bbox = tool_args.get("bbox_2d") or tool_args.get("bbox")
                tool_result = env.ocr(bbox)

            elif tool_name in ("select",):
                label = tool_args.get("label", "object")
                tool_result = env.select(label)

            else:
                tool_result = f"[Unknown tool: {tool_name}]"
                env._record(tool_name, tool_args, output=tool_result)

        except Exception as e:
            tool_result = f"[Error executing {tool_name}: {e}]"
            env._record(tool_name, tool_args, output=tool_result)

        # Format tool response
        tool_response_content = [
            {"type": "text", "text": "<tool_response>"},
        ]

        # If the tool returned a new image, include it
        if tool_name in ("zoom_in", "zoom") and tool_result.startswith("[Zoomed"):
            tool_response_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{new_img_b64}"},
            })

        tool_response_content.append({"type": "text", "text": tool_result})
        tool_response_content.append({"type": "text", "text": "</tool_response>"})
        tool_response_content.append({"type": "text", "text": "\nContinue analyzing. Call another tool or answer with <answer>...</answer>"})

        # Add to messages
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": tool_response_content})

        # Debug log
        debug_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"<tool_response>\n{tool_result}\n</tool_response>"},
            ],
        })

    else:
        # Max turns reached without answer
        predicted_answer = ""
        status = "max_turns_exceeded"

    # Compute accuracy
    accuracy = 1.0 if check_answer_correct(predicted_answer, ground_truth, options) else 0.0

    # Build result
    result = {
        "image": os.path.basename(image_path),
        "question": question,
        "options": options,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "accuracy": accuracy,
        "tool_calls": env.tool_records,
        "image_size": env.original_size,
        "status": status,
        "debug_messages": debug_messages,
    }

    return result


# ============================================================
#  Load VStar Data
# ============================================================

def discover_vstar_samples(vstar_path: str) -> List[Tuple[str, str, Dict]]:
    """Discover all samples in the VStar benchmark directory.

    Returns:
        List of (image_path, image_file, annotation_dict)
    """
    samples = []
    for test_type in TEST_TYPES:
        test_dir = os.path.join(vstar_path, test_type)
        if not os.path.isdir(test_dir):
            print(f"  [Warning] {test_dir} not found, skipping")
            continue

        # Find all .jpg files (also handles .png, .jpeg)
        image_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not os.path.isdir(os.path.join(test_dir, f))
        ])

        for img_file in image_files:
            # Find annotation: same name with .json extension
            base_name = os.path.splitext(img_file)[0]
            json_path = os.path.join(test_dir, f"{base_name}.json")
            if not os.path.isfile(json_path):
                # Try other common patterns
                json_path = os.path.join(test_dir, f"{base_name}.json")
                if not os.path.isfile(json_path):
                    continue

            with open(json_path, "r") as f:
                anno = json.load(f)

            samples.append((os.path.join(test_dir, img_file), img_file, anno))

    return samples


# ============================================================
#  Reward Computation (ATR + Original)
# ============================================================

def compute_rewards_for_trajectory(
    tool_calls: List[Dict],
    final_answer: str,
    accuracy: float,
    question: str,
    image_size: Tuple[int, int],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """Compute both ATR and original PyVision-RL rewards.

    Returns dict with:
        - atr_reward: Full ATR reward
        - atr_utility: U component
        - atr_cost: C component
        - atr_sequence: S component
        - atr_format_gate: R_format
        - original_reward: Original R = acc + 0.1 * n_tool_calls
        - n_tool_calls: number of tool calls
    """
    from atr.reward import AdaptiveToolReward
    from atr.config import ATRConfig

    n_tool_calls = len(tool_calls)

    # --- Original PyVision-RL reward ---
    original_reward = AdaptiveToolReward.original_pyvision_reward(accuracy, n_tool_calls, 0.1)

    # --- ATR reward (full) ---
    atr = AdaptiveToolReward(config=ATRConfig())
    atr_reward, components = atr.compute(
        tool_calls=tool_calls,
        final_answer=final_answer,
        accuracy=accuracy,
        ground_truth={"question": question, "gt_answer": ground_truth} if ground_truth else None,
        question=question,
        image_size=image_size,
    )

    # --- ATR variants (ablation) ---
    # Utility only
    atr_util = AdaptiveToolReward(config=ATRConfig.utility_only())
    r_util, _ = atr_util.compute(tool_calls, final_answer, accuracy,
                                  question=question, image_size=image_size)

    # No sequence (U + C)
    atr_no_seq = AdaptiveToolReward(config=ATRConfig.no_sequence())
    r_no_seq, _ = atr_no_seq.compute(tool_calls, final_answer, accuracy,
                                      question=question, image_size=image_size)

    return {
        "n_tool_calls": n_tool_calls,
        "original_reward": original_reward,
        "atr_reward": atr_reward,
        "atr_utility": components.get("utility", 0.0),
        "atr_cost": components.get("cost", 0.0),
        "atr_sequence": components.get("sequence", 0.0),
        "atr_format_gate": components.get("format_gate", 1.0),
        "atr_utility_only": r_util,
        "atr_no_sequence": r_no_seq,
    }


# ============================================================
#  Analysis & Reporting
# ============================================================

def analyze_trajectories(trajectories: List[Dict]) -> Dict[str, Any]:
    """Run full reward analysis on all trajectories.

    Returns dict with results for all samples and aggregate stats.
    """
    results = []
    for traj in tqdm(trajectories, desc="Computing rewards"):
        # Skip errored trajectories
        if traj.get("status") == "error":
            continue

        rewards = compute_rewards_for_trajectory(
            tool_calls=traj["tool_calls"],
            final_answer=traj.get("predicted_answer", ""),
            accuracy=traj["accuracy"],
            question=traj["question"],
            image_size=tuple(traj["image_size"]),
            ground_truth=traj["ground_truth"],
        )

        results.append({**traj, **rewards})

    return _compute_statistics(results)


def _compute_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics."""
    if not results:
        return {}

    n = len(results)

    # Accuracy
    accuracies = [r["accuracy"] for r in results]
    mean_accuracy = np.mean(accuracies)

    # Tool calls
    n_tools = [r["n_tool_calls"] for r in results]
    mean_tool_calls = np.mean(n_tools)
    total_tool_calls = sum(n_tools)

    # Rewards
    orig_rewards = [r["original_reward"] for r in results]
    atr_rewards = [r["atr_reward"] for r in results]
    atr_utilities = [r["atr_utility"] for r in results]
    atr_costs = [r["atr_cost"] for r in results]
    atr_sequences = [r["atr_sequence"] for r in results]
    atr_util_only = [r["atr_utility_only"] for r in results]
    atr_no_seq = [r["atr_no_sequence"] for r in results]

    # Correlation: ATR vs Original (pairwise comparison)
    spearman_corr = _spearman_rank_corr(orig_rewards, atr_rewards)

    # Find "misclassified" trajectories:
    #   - Original says good (high reward) but ATR says bad (low reward) = tool spam
    #   - Original says bad (low reward) but ATR says good (high reward) = efficient usage
    orig_scores = np.array(orig_rewards)
    atr_scores = np.array(atr_rewards)
    orig_mean = np.mean(orig_scores)
    atr_mean = np.mean(atr_scores)

    orig_high_atr_low = []
    orig_low_atr_high = []

    for i, r in enumerate(results):
        if orig_scores[i] > orig_mean and atr_scores[i] < atr_mean:
            orig_high_atr_low.append({
                "index": i,
                "file": r.get("image", ""),
                "question": r.get("question", "")[:60],
                "n_tool_calls": r["n_tool_calls"],
                "accuracy": r["accuracy"],
                "original": orig_scores[i],
                "atr": atr_scores[i],
            })
        elif orig_scores[i] < orig_mean and atr_scores[i] > atr_mean:
            orig_low_atr_high.append({
                "index": i,
                "file": r.get("image", ""),
                "question": r.get("question", "")[:60],
                "n_tool_calls": r["n_tool_calls"],
                "accuracy": r["accuracy"],
                "original": orig_scores[i],
                "atr": atr_scores[i],
            })

    # Redundancy rate: % of tool calls flagged with cost > 0
    redundancy_rates = []
    for r in results:
        if r["n_tool_calls"] > 0:
            # Approximate: cost per tool call
            per_call_cost = r["atr_cost"] / max(r["n_tool_calls"], 1)
            redundancy_rates.append(min(1.0, per_call_cost * 2))  # normalize
    mean_redundancy = np.mean(redundancy_rates) if redundancy_rates else 0.0

    return {
        "n_samples": n,
        "n_correct": int(sum(accuracies)),
        "n_wrong": n - int(sum(accuracies)),
        "accuracy_mean": mean_accuracy,

        "tool_calls_mean": mean_tool_calls,
        "tool_calls_total": total_tool_calls,

        "original_reward_mean": np.mean(orig_rewards),
        "original_reward_std": np.std(orig_rewards),
        "atr_reward_mean": np.mean(atr_rewards),
        "atr_reward_std": np.std(atr_rewards),
        "atr_utility_mean": np.mean(atr_utilities),
        "atr_cost_mean": np.mean(atr_costs),
        "atr_sequence_mean": np.mean(atr_sequences),
        "atr_utility_only_mean": np.mean(atr_util_only),
        "atr_no_sequence_mean": np.mean(atr_no_seq),

        "spearman_corr": spearman_corr,
        "orig_high_atr_low_count": len(orig_high_atr_low),
        "orig_low_atr_high_count": len(orig_low_atr_high),
        "orig_high_atr_low": orig_high_atr_low[:10],  # top 10 examples
        "orig_low_atr_high": orig_low_atr_high[:10],

        "redundancy_rate_mean": mean_redundancy,

        "per_sample_results": results,
    }


def _spearman_rank_corr(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        # Fallback: manual computation
        n = len(x)
        if n < 3:
            return 0.0
        x_ranks = np.argsort(np.argsort(x))
        y_ranks = np.argsort(np.argsort(y))
        d = x_ranks - y_ranks
        rho = 1 - (6 * sum(d ** 2)) / (n * (n ** 2 - 1))
        return rho


def print_report(stats: Dict[str, Any]):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("  ATR Offline Analysis Report")
    print("=" * 70)

    print(f"\n  Dataset: VStar ({stats['n_samples']} samples)")
    print(f"  Accuracy: {stats['n_correct']}/{stats['n_samples']} ({stats['accuracy_mean']:.1%})")
    print(f"  Avg Tool Calls: {stats['tool_calls_mean']:.2f} (total: {stats['tool_calls_total']})")

    print(f"\n  {'=' * 50}")
    print(f"  {'Reward Method':<25} {'Mean':<10} {'Std':<10} {'Delta vs Orig':<12}")
    print(f"  {'-' * 50}")
    print(f"  {'PyVision-RL (Original)':<25} {stats['original_reward_mean']:<10.3f} {stats['original_reward_std']:<10.3f} {'-':<12}")
    print(f"  {'ATR (Utility Only)':<25} {stats['atr_utility_only_mean']:<10.3f} {'':<10} {stats['atr_utility_only_mean'] - stats['original_reward_mean']:<+12.3f}")
    print(f"  {'ATR (U + C, no seq)':<25} {stats['atr_no_sequence_mean']:<10.3f} {'':<10} {stats['atr_no_sequence_mean'] - stats['original_reward_mean']:<+12.3f}")
    print(f"  {'ATR (Full)':<25} {stats['atr_reward_mean']:<10.3f} {stats['atr_reward_std']:<10.3f} {stats['atr_reward_mean'] - stats['original_reward_mean']:<+12.3f}")
    print(f"  {'-' * 50}")

    print(f"\n  --- ATR Component Breakdown ---")
    print(f"  Utility (U):    {stats['atr_utility_mean']:.3f}  (question-aware tool usefulness)")
    print(f"  Cost (C):       {stats['atr_cost_mean']:.3f}  (redundancy penalty)")
    print(f"  Sequence (S):   {stats['atr_sequence_mean']:.3f}  (tool ordering quality)")
    print(f"  Redundancy Rate: {stats['redundancy_rate_mean']:.1%}")

    print(f"\n  --- Correlation ---")
    print(f"  Spearman ρ(ATR, Original): {stats['spearman_corr']:.3f}")
    print(f"    (Low correlation = original and ATR rank trajectories differently)")

    print(f"\n  --- Disagreement Analysis ---")
    print(f"  Original↑ ATR↓ (tool spam?): {stats['orig_high_atr_low_count']} samples")
    for ex in stats['orig_high_atr_low'][:5]:
        print(f"    - {ex['file']}: {ex['n_tool_calls']} tools, "
              f"acc={ex['accuracy']:.0f}, orig={ex['original']:.2f}, atr={ex['atr']:.2f}")
        print(f"      Q: {ex['question'][:60]}")

    print(f"\n  Original↓ ATR↑ (efficient?): {stats['orig_low_atr_high_count']} samples")
    for ex in stats['orig_low_atr_high'][:5]:
        print(f"    - {ex['file']}: {ex['n_tool_calls']} tools, "
              f"acc={ex['accuracy']:.0f}, orig={ex['original']:.2f}, atr={ex['atr']:.2f}")
        print(f"      Q: {ex['question'][:60]}")

    # Top discrepancies table
    print(f"\n  --- Per-Sample Detail (first 10) ---")
    print(f"  {'File':<20} {'Tools':<6} {'Acc':<5} {'R_orig':<8} {'R_atr':<8} {'U':<6} {'C':<6} {'S':<6}")
    print(f"  {'-' * 65}")
    for r in stats['per_sample_results'][:10]:
        fname = os.path.basename(r.get("image", "?"))[:18]
        print(f"  {fname:<20} {r['n_tool_calls']:<6} {r['accuracy']:<5.0f} "
              f"{r['original_reward']:<8.3f} {r['atr_reward']:<8.3f} "
              f"{r['atr_utility']:<6.2f} {r['atr_cost']:<6.2f} {r['atr_sequence']:<6.2f}")

    print("\n" + "=" * 70)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ATR Offline Experiment on VStar")
    parser.add_argument("--vstar_path", type=str, required=True,
                        help="Path to VStar benchmark directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "results"),
                        help="Output directory for trajectories and reports")
    parser.add_argument("--quick", type=int, default=0,
                        help="Run on N samples only (for quick testing)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip rollout, analyze existing trajectories")
    parser.add_argument("--trajectories_file", type=str, default=None,
                        help="Path to existing trajectories JSONL (for --analyze_only)")
    args = parser.parse_args()

    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========== Rollout Stage ==========
    if not args.analyze_only:
        print(f"\n{'=' * 60}")
        print(f"  Stage 1: Rollout (collecting trajectories)")
        print(f"  Model: {MODEL_NAME}")
        print(f"  VStar: {args.vstar_path}")
        print(f"{'=' * 60}\n")

        # Discover VStar samples
        samples = discover_vstar_samples(args.vstar_path)
        print(f"  Found {len(samples)} VStar samples in {TEST_TYPES}")

        if not samples:
            print("  Error: No VStar samples found!")
            sys.exit(1)

        # Quick mode: take subset
        if args.quick > 0:
            import random
            random.seed(42)
            samples = random.sample(samples, min(args.quick, len(samples)))
            print(f"  Quick mode: using {len(samples)} samples")

        # Create client
        client = create_client()

        # Run rollout with incremental saving
        traj_file = os.path.join(args.output_dir, f"trajectories_{timestamp}.jsonl")
        trajectories = []
        for image_path, img_file, anno in tqdm(samples, desc="Processing VStar"):
            result = run_single_sample(client, image_path, anno)
            trajectories.append(result)

            # Print summary for each sample (ASCII-safe for Windows)
            status_mark = "[OK]" if result["accuracy"] > 0 else "[NO]"
            answer_preview = result.get("predicted_answer", "")[:40].encode('ascii', errors='replace').decode()
            tqdm.write(f"  {status_mark} {img_file}: "
                       f"{len(result['tool_calls'])} tools, "
                       f"acc={result['accuracy']:.0f}, "
                       f"answer='{answer_preview}'")

            # Incrementally save after each sample (crash-safe)
            with open(traj_file, "a") as f:
                save_t = {k: v for k, v in result.items() if k != "debug_messages"}
                f.write(json.dumps(save_t, ensure_ascii=False) + "\n")
                f.flush()

        print(f"\n  Saved {len(trajectories)} trajectories to: {traj_file}")

    else:
        # Load existing trajectories
        traj_file = args.trajectories_file
        if not traj_file:
            print("  Error: --analyze_only requires --trajectories_file")
            sys.exit(1)
        if not os.path.isfile(traj_file):
            print(f"  Error: trajectories file not found: {traj_file}")
            sys.exit(1)

        with open(traj_file, "r") as f:
            trajectories = [json.loads(line) for line in f if line.strip()]
        print(f"\n  Loaded {len(trajectories)} trajectories from: {traj_file}")

    # ========== Analysis Stage ==========
    print(f"\n{'=' * 60}")
    print(f"  Stage 2: Reward Analysis")
    print(f"{'=' * 60}\n")

    valid_trajs = [t for t in trajectories if t.get("status") != "error"]
    print(f"  Valid trajectories: {len(valid_trajs)} / {len(trajectories)}")

    if valid_trajs:
        stats = analyze_trajectories(valid_trajs)
        print_report(stats)

        # Save analysis report
        report_file = os.path.join(args.output_dir, f"analysis_{timestamp}.json")
        # Remove bulky per-sample raw data from saved report
        save_stats = {k: v for k, v in stats.items() if k != "per_sample_results"}
        save_stats["trajectories_file"] = traj_file
        with open(report_file, "w") as f:
            json.dump(save_stats, f, indent=2, default=str)
        print(f"  Analysis report saved to: {report_file}")
    else:
        print("  No valid trajectories to analyze.")

    print(f"\n  Done! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
