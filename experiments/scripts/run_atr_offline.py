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
# ATR tool registry imports (单一真值源:工具定义/prompt/轨迹)
# ============================================================
from atr.tools import get_tool_schemas, execute as execute_tool, ToolTrace, registry

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
# 可用环境变量 ATR_MODEL 覆盖(换模型不改代码)
MODEL_NAME = os.environ.get("ATR_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

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

# --- 模型显示尺寸(坐标空间锚点) ---
# 所有发给模型的图像统一等比缩放到最长边 DISPLAY_MAX;
# 模型输出的 bbox_2d 坐标约定为"当前显示图像空间",
# ToolEnv 自动换算到执行空间(原始分辨率),模型无需心算。
DISPLAY_MAX = 1024


# ============================================================
#  System Prompt — define available tools + output format
#  <tools> 段由 atr.tools 注册表生成(单一真值源)
# ============================================================

def build_system_prompt(tool_required: bool = False) -> str:
    """由 atr.tools 注册表生成 SYSTEM_PROMPT(含坐标空间约定与工作流示例)。

    tool_required=True 时切换为"先工具核实后作答"策略(用于采集工具轨迹)。
    """
    schemas = "\n".join(json.dumps(s) for s in get_tool_schemas())
    if tool_required:
        answer_policy = (
            "# Answer policy (IMPORTANT)\n"
            "1. BEFORE answering, you MUST call a tool to verify the target object:\n"
            "   zoom into the relevant region (or crop it) and inspect the returned image.\n"
            "2. Call ONE tool per turn; you may call several tools in sequence.\n"
            "3. After inspecting tool results, give your final answer in <answer> tags."
        )
    else:
        answer_policy = (
            "# Answer policy (IMPORTANT)\n"
            "1. FIRST, answer directly from the full image: <answer>your answer</answer>\n"
            "2. ONLY if you cannot determine the answer from the full image\n"
            "   (object too small, text unreadable, details unclear) may you call a tool\n"
            "   to inspect. Call ONE tool per turn, inspect the returned image, then answer.\n"
            "3. After inspecting tool results, give your final answer in <answer> tags."
        )
    return f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{schemas}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Coordinate system (IMPORTANT)
All bbox_2d values MUST be NORMALIZED coordinates: each value is a fraction
of the image size in [0, 1], where (0,0) = top-left corner and (1,1) = bottom-right
corner of the image you are currently viewing. Do NOT use pixel values.
Your coordinates are scaled automatically by the tool executor.

{answer_policy}

Tools available:
<tool_call>
{{"name": "zoom", "arguments": {{"bbox_2d": [0.15, 0.25, 0.28, 0.38]}}}}
</tool_call>
Use ocr to read text, rotate only if the image orientation is wrong.

When you have enough evidence, provide your answer inside <answer> tags:
<answer>your answer</answer>"""


SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(question: str, options: Optional[List[str]] = None,
                      display_size: Optional[Tuple[int, int]] = None) -> str:
    """Build user prompt with question and options."""
    prompt = f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
        for i, opt in enumerate(options):
            prompt += f"{abc_map.get(i + 1, str(i + 1))}. {opt}\n"
    prompt += ("\nAnswer directly from the full image with the correct option letter in <answer> tags. "
               "Use a tool only if you cannot determine the answer from the full image.")
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


def resize_for_display(img: Image.Image, max_side: int = DISPLAY_MAX) -> Tuple[Image.Image, Tuple[int, int]]:
    """等比缩放到最长边 max_side,返回 (缩放后图像, (显示宽, 显示高))。

    所有发给模型的图像(初始图、crop/zoom/rotate 结果)统一走此函数,
    保证模型每轮看到的视图尺寸与坐标空间锚点一致。
    """
    w, h = img.size
    if w <= max_side and h <= max_side:
        return img.copy(), (w, h)
    ratio = max_side / max(w, h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.LANCZOS), new_size


def encode_image_file_to_base64(image_path: str, max_size: int = DISPLAY_MAX) -> str:
    """Encode an image file to base64, resizing to max_size on longest side."""
    img = Image.open(image_path)
    display_img, _ = resize_for_display(img, max_size)
    buffered = BytesIO()
    display_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ============================================================
#  Tool Environment — 薄封装:持有图像状态,执行委托 atr.tools 注册表
#  坐标约定:模型在"当前显示图像空间"输出 bbox,ToolEnv 自动换算到执行空间。
# ============================================================

class ToolEnv:
    """轻量工具执行环境。

    维护当前图像状态(初始为原图,zoom/rotate 等状态工具更新它)。
    所有工具的执行逻辑与轨迹记录委托 atr.tools 注册表(ToolTrace),
    轨迹格式与 ATR reward 层完全一致。

    坐标空间:
      - 模型看到的图像统一缩放到最长边 DISPLAY_MAX(显示空间)
      - 模型输出的 bbox 是显示空间坐标,_to_current_space 换算到执行空间
      - 轨迹记录 bbox 为执行空间(原始分辨率),与旧数据/GT 标注一致
    """

    def __init__(self, image_path: str):
        self.original_image = Image.open(image_path).convert("RGB")
        self.current_image = self.original_image.copy()
        self.original_size = self.original_image.size  # (width, height)
        self.trace = ToolTrace()
        self._last_image_b64: Optional[str] = None
        # 模型当前所见图像的显示尺寸(坐标空间锚点)
        _, self.display_size = resize_for_display(self.original_image)

    def _to_current_space(self, arguments: dict) -> dict:
        """把模型输出的归一化 bbox [0,1] 换算到执行空间(current_image 像素)。

        归一化坐标与分辨率无关,任何视图状态下换算都一致;
        轨迹记录 bbox 为像素(执行空间),与旧数据/GT 标注保持一致。
        """
        args = dict(arguments)
        bbox = args.get("bbox_2d") or args.get("bbox")
        if bbox and len(bbox) == 4:
            cw, ch = self.current_image.size
            x1, y1, x2, y2 = bbox
            args["bbox_2d"] = [x1 * cw, y1 * ch, x2 * cw, y2 * ch]
        return args

    def execute(self, tool_name: str, arguments: dict) -> Tuple[str, bool]:
        """执行一个工具调用(注册表分派,显示空间 → 执行空间自动换算)。

        Returns:
            (output_desc, produced_new_image): 返回给模型的文本描述;
            工具是否产生了新图像(crop/zoom/rotate 时 True,并已编码为 b64)。
        """
        args = self._to_current_space(arguments)
        try:
            result = execute_tool(tool_name, args, self.current_image)
        except KeyError:
            msg = f"[Unknown tool: {tool_name}]"
            self.trace.record(tool_name, arguments, output=msg)
            return msg, False
        except ValueError as e:
            # 与旧 dispatch guard 的错误文本一致,如 "[Error: crop requires bbox_2d]"
            msg = f"[Error: {e}]"
            self.trace.record(tool_name, arguments, output=msg)
            return msg, False
        except Exception as e:
            msg = f"[Error executing {tool_name}: {e}]"
            self.trace.record(tool_name, arguments, output=msg)
            return msg, False

        if result.image is not None:
            # 统一缩放到显示尺寸后发给模型
            if registry.get(result.canonical_name).updates_state:
                # 状态工具(zoom/rotate):工作图像切换,坐标锚点同步更新
                self.current_image = result.image
                display_img, self.display_size = resize_for_display(result.image)
            else:
                # 非状态工具(crop):裁剪图仅作放大观察,坐标锚点保持主视图不变,
                # 模型继续在主视图空间输出坐标(约定见 system prompt)
                display_img, _ = resize_for_display(result.image)
            self._last_image_b64 = pil_to_base64(display_img)
        self.trace.record(result.canonical_name, result.arguments, result.output, result.bbox)
        return result.output, result.image is not None


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

    # Build initial messages(含当前视图尺寸坐标锚点)
    user_prompt = build_user_prompt(question, options, display_size=env.display_size)
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

        # Execute tool (注册表分派,内部已捕获所有异常并记录轨迹)
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        tool_result, new_image = env.execute(tool_name, tool_args)

        # Format tool response
        tool_response_content = [
            {"type": "text", "text": "<tool_response>"},
        ]

        # If the tool returned a new image, include it (zoom/rotate)
        if new_image and env._last_image_b64:
            tool_response_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{env._last_image_b64}"},
            })

        tool_response_content.append({"type": "text", "text": tool_result})
        tool_response_content.append({"type": "text", "text": "</tool_response>"})
        tool_response_content.append({"type": "text", "text":
            "\nContinue analyzing. Call another tool or answer with <answer>...</answer>"})

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
        "tool_calls": env.trace.records,
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
    global SYSTEM_PROMPT, TEMPERATURE
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
    parser.add_argument("--tool_required", action="store_true",
                        help="强制先工具核实后作答(采集工具轨迹;默认先答后验)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="采样温度(默认 %(default)s;多温度增广用 0.7/0.9/1.1)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="断点续跑:跳过 output_dir 中已有轨迹文件里完成的样本")
    args = parser.parse_args()

    # 切换策略:tool_required → 强制工具模式;温度增广
    SYSTEM_PROMPT = build_system_prompt(tool_required=args.tool_required)
    TEMPERATURE = args.temperature

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

        # 断点续跑:跳过已有轨迹中的样本(编码探测:utf-8 → gbk,Windows 写入可能为 GBK)
        if args.skip_existing:
            existing = set()
            for f in os.listdir(args.output_dir):
                if not (f.startswith("trajectories_") and f.endswith(".jsonl")):
                    continue
                p = os.path.join(args.output_dir, f)
                for enc in ("utf-8", "gbk"):
                    try:
                        for line in open(p, encoding=enc):
                            try:
                                existing.add(json.loads(line)["image"])
                            except Exception:
                                pass
                        break
                    except UnicodeDecodeError:
                        continue
            if existing:
                before = len(samples)
                samples = [s for s in samples if s[1] not in existing]
                print(f"  断点续跑:跳过 {before - len(samples)} 个已完成样本,剩余 {len(samples)}")

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
