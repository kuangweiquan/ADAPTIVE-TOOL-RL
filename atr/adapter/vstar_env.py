"""VStarToolEnv — 将 VStar 工具循环包装成 verl agent 环境的 ToolBase。

对接 verl_agents 的 agent rollout 协议(参考模板:
  pyvision-rl/verl_agents/verl/workers/agent/envs/visual_agent/vl_agent_v3.py):

    env = ToolBase.create(env_name)          # 由数据集 env_name 列选择
    env.reset(raw_prompt, multi_modal_data, origin_multi_modal_data)
    observation, reward, done, info = env.execute(action_string)

状态逻辑与离线评估 ToolEnv(run_atr_offline.py) 一致:
  - 初始视图 = 原图缩放到最长边 DISPLAY_MAX(模型坐标空间锚点)
  - zoom/rotate 更新工作图像;crop 仅作观察(坐标锚点不变)
  - 模型输出 NORMALIZED bbox [0,1],自动换算到当前视图像素空间执行
  - 工具结果文本与离线 `<tool_response>...</tool_response>` 格式一致

观测返回格式(verl agent 协议):
  - 含新图像(crop/zoom/rotate): Format 3 dict
      {"prompt": "...<image>...", "multi_modal_data": {"image": [PIL]}}
  - 纯文本(ocr/错误/提示): Format 1 str
  - 检测到 <answer>: ("", 0.0, True, {})  → 轨迹结束
每轮 env_reward 恒为 0.0 —— 奖励全部由 ATRRewardManager 集中计算。

注册:继承 ToolBase 并定义 name 即自动注册(metaclass),训练脚本 import 本模块即可。
"""

from typing import Any, Dict, List, Optional, Tuple

import re
import json
from PIL import Image

try:
    # verl_agents 已装(训练机):继承 ToolBase,导入即注册(name 属性)
    from verl.workers.agent.tool_envs import ToolBase
except ImportError:
    # 本机冒烟/离线:无 verl 时退化为普通类(状态机逻辑不受影响)
    ToolBase = object

from ..tools import execute as execute_tool, registry, ToolTrace


# —————————————————————————————————————
# 显示尺寸(与离线评估一致)
# —————————————————————————————————————
DISPLAY_MAX = 1024


def resize_for_display(img: Image.Image, max_side: int = DISPLAY_MAX):
    """等比缩放到最长边 max_side,返回 (缩放后图像, (显示宽, 显示高))。"""
    w, h = img.size
    if w <= max_side and h <= max_side:
        return img.copy(), (w, h)
    ratio = max_side / max(w, h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.LANCZOS), new_size


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """解析模型回复中的第一个 <tool_call> JSON(兼容 ```json 代码块)。

    与离线评估 run_atr_offline.py 的 parse_tool_call 一致。
    """
    for block_re in (
        re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL),
        re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL),
    ):
        match = block_re.search(text)
        if not match:
            continue
        try:
            call_data = json.loads(match.group(1))
            if isinstance(call_data, list):
                call_data = call_data[0] if call_data else {}
            return {
                "name": call_data.get("name", "unknown"),
                "arguments": call_data.get("arguments", {}),
            }
        except (json.JSONDecodeError, Exception):
            try:
                call_data = eval(match.group(1))
                if isinstance(call_data, list):
                    call_data = call_data[0] if call_data else {}
                return {
                    "name": call_data.get("name", "unknown"),
                    "arguments": call_data.get("arguments", {}),
                }
            except Exception:
                continue
    return None


def extract_answer(text: str) -> Optional[str]:
    """从 <answer>...</answer> 提取最终答案(与离线评估一致)。"""
    pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


TOOL_RESPONSE_TEMPLATE = (
    "<tool_response>\n{output}\n</tool_response>\n"
    "Continue analyzing. Call another tool or answer with <answer>...</answer>"
)
ANSWER_PROMPT = "Please provide your answer using <answer>...</answer> tags."


class VStarToolEnv(ToolBase):
    name = "vstar_tool_env"

    def __init__(self, _name, _desc, _params, **kwargs):
        self.current_image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.display_size: Optional[Tuple[int, int]] = None
        self.trace = ToolTrace()
        if ToolBase is not object:
            super().__init__(name=self.name)

    # ------------------------------------------------------------------
    #  verl agent 协议
    # ------------------------------------------------------------------

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        # origin_multi_modal_data 是原始分辨率图像(与离线 ToolEnv.original_image 等价)
        mm = origin_multi_modal_data or multi_modal_data or {}
        images = mm.get("image", [])
        assert images, f"[VStarToolEnv] no image in multi_modal_data: {mm.keys()=}"
        self.original_image = images[0].convert("RGB")
        self.current_image = self.original_image.copy()
        _, self.display_size = resize_for_display(self.original_image)
        self.trace = ToolTrace()
        return None

    def execute(self, action_string, **kwargs):
        # 1. 已作答 → 轨迹结束(答案文本已在 response 中,reward 由 reward manager 结算)
        if extract_answer(action_string):
            return '', 0.0, True, {}

        # 2. 解析工具调用
        tool_call = parse_tool_call(action_string)
        if tool_call is None:
            # 无工具调用也无答案:提示作答(与离线评估行为一致,不结束轨迹)
            return ANSWER_PROMPT, 0.0, False, {}

        # 3. 执行工具(归一化 → 当前视图像素自动换算)
        tool_name = tool_call["name"]
        args = self._to_current_space(tool_call["arguments"])
        try:
            result = execute_tool(tool_name, args, self.current_image)
        except KeyError:
            msg = f"[Unknown tool: {tool_name}]"
            self.trace.record(tool_name, tool_call["arguments"], output=msg)
            return TOOL_RESPONSE_TEMPLATE.format(output=msg), 0.0, False, {}
        except ValueError as e:
            msg = f"[Error: {e}]"
            self.trace.record(tool_name, tool_call["arguments"], output=msg)
            return TOOL_RESPONSE_TEMPLATE.format(output=msg), 0.0, False, {}
        except Exception as e:
            msg = f"[Error executing {tool_name}: {e}]"
            self.trace.record(tool_name, tool_call["arguments"], output=msg)
            return TOOL_RESPONSE_TEMPLATE.format(output=msg), 0.0, False, {}

        self.trace.record(result.canonical_name, result.arguments, result.output, result.bbox)

        # 4. 组装观测
        if result.image is not None:
            if registry.get(result.canonical_name).updates_state:
                # 状态工具(zoom/rotate):工作图像切换,坐标锚点同步更新
                self.current_image = result.image
                display_img, self.display_size = resize_for_display(result.image)
            else:
                # 非状态工具(crop):裁剪图仅作放大观察,坐标锚点保持主视图不变
                display_img, _ = resize_for_display(result.image)
            # Format 3:prompt 内 <image> 占位 + PIL 图像列表(verl 回传 vLLM)
            prompt = (
                "<tool_response>\n<image>\n"
                f"{result.output}\n"
                "</tool_response>\n"
                "Continue analyzing. Call another tool or answer with <answer>...</answer>"
            )
            return {"prompt": prompt, "multi_modal_data": {"image": [display_img]}}, 0.0, False, {}

        # 纯文本观测(ocr 结果/错误)
        return TOOL_RESPONSE_TEMPLATE.format(output=result.output), 0.0, False, {}

    def close(self):
        self.current_image = None
        self.original_image = None
        self.display_size = None

    # ------------------------------------------------------------------
    #  内部工具
    # ------------------------------------------------------------------

    def _to_current_space(self, arguments: dict) -> dict:
        """模型输出的归一化 bbox [0,1] → 当前视图像素坐标(与离线 ToolEnv 一致)。"""
        args = dict(arguments)
        bbox = args.get("bbox_2d") or args.get("bbox")
        if bbox and len(bbox) == 4:
            cw, ch = self.current_image.size
            x1, y1, x2, y2 = bbox
            args["bbox_2d"] = [x1 * cw, y1 * ch, x2 * cw, y2 * ch]
        return args

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """本样本的工具调用轨迹(调试用;reward 层从 response 文本自行解析)。"""
        return self.trace.records


if __name__ == "__main__":
    # 快速自测(不依赖 verl 训练管线;先 import 本模块注册 env)
    import os

    demo_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "datasets", "vstar_bench", "direct_attributes", "sa_10033.jpg",
    )
    tool = VStarToolEnv(_name=None, _desc=None, _params=None)
    tool.reset(raw_prompt="", multi_modal_data={"image": [Image.open(demo_img).convert("RGB")]},
               origin_multi_modal_data=None)

    action = '<tool_call>\n{"name": "zoom", "arguments": {"bbox_2d": [0.15, 0.25, 0.28, 0.38]}}\n</tool_call>'
    obs, reward, done, info = tool.execute(action_string=action)
    print(f"[zoom] type={type(obs).__name__} reward={reward} done={done}")
    if isinstance(obs, dict):
        print(f"  prompt[:60]={obs['prompt'][:60]!r} n_images={len(obs['multi_modal_data']['image'])}")
        print(f"  current_image.size={tool.current_image.size}")

    action2 = '<tool_call>\n{"name": "ocr", "arguments": {}}\n</tool_call>'
    obs2, reward2, done2, _ = tool.execute(action_string=action2)
    print(f"[ocr] type={type(obs2).__name__} reward={reward2} done={done2}")
    print(f"  obs[:80]={obs2[:80]!r}")

    action3 = "<answer>A</answer>"
    obs3, reward3, done3, _ = tool.execute(action_string=action3)
    print(f"[answer] obs={obs3!r} reward={reward3} done={done3}")
