"""Adapter: Inject ATR reward into PyVision-RL's training loop.

Integration point: `NaiveRewardManager.__call__()` in
  pyvision-rl/verl_agents/verl/workers/reward_manager/naive.py

Strategy: Create a custom reward manager that subclasses NaiveRewardManager
and replaces the env_reward + acc_reward logic with ATR.compute().
"""

from typing import Optional, List, Dict, Any
import re
import json

from ..reward import AdaptiveToolReward
from ..config import ATRConfig


# —————————————————————————————————————
#  Tool trajectory parser
# —————————————————————————————————————
def parse_tool_trajectory(response_str: str) -> List[Dict[str, Any]]:
    """Parse tool calls from the model's response text.

    PyVision-RL tools use format:
        <tool_call>
        {"name": "zoom_in", "arguments": {"bbox_2d": [x1,y1,x2,y2], "label": "..."}}
        </tool_call>

    Returns a list of dicts with:
        - tool_name: str
        - bbox: [x1,y1,x2,y2] (if available; 模型输出的 NORMALIZED [0,1] 坐标)
        - output: str  (the tool's return observation, approximated)

    注:bbox 是模型写在文本里的归一化坐标(显示空间),与 reward 层
    gt_bbox(归一化)/lazy-crop 判定直接同空间比较,无需换算。
    """
    tool_calls = []

    # Match <tool_call> blocks
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(response_str):
        try:
            call_data = json.loads(match.group(1))
        except Exception:
            try:
                call_data = eval(match.group(1))
            except Exception:
                continue
        if isinstance(call_data, list):
            call_data = call_data[0] if call_data else {}
        if not isinstance(call_data, dict):
            continue
        record = {"tool_name": call_data.get("name", "unknown")}
        args = call_data.get("arguments", {})
        if isinstance(args, dict):
            if "bbox_2d" in args:
                record["bbox"] = args["bbox_2d"]
            if "label" in args:
                record["label"] = args["label"]
        tool_calls.append(record)

    return tool_calls


def approximate_tool_outputs(response_str: str, tool_calls: List[Dict]) -> List[Dict]:
    """Fill the 'output' field of each tool call with surrounding text."""
    if not tool_calls:
        return tool_calls

    markers = ["<tool_call>", "</tool_call>"]
    enriched = []
    parts = re.split(r'(<tool_call>.*?</tool_call>)', response_str, flags=re.DOTALL)

    call_idx = 0
    for part in parts:
        if part.startswith("<tool_call>"):
            if call_idx < len(tool_calls):
                enriched.append(tool_calls[call_idx])
                call_idx += 1
        else:
            if enriched:
                # The text after a tool_call block is its output
                enriched[-1]["output"] = part.strip()

    return enriched


# —————————————————————————————————————
#  ATR Reward Manager (drop-in replacement)
# —————————————————————————————————————
class ATRRewardManager:
    """Replacement for NaiveRewardManager that uses Adaptive Tool Reward.

    Usage in training script (main_ppo.py):

        from atr.adapter import ATRRewardManager
        trainer.reward_manager = ATRRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=original_compute_score,
            atr_config=ATRConfig(),
        )

    Or monkey-patch at module level:

        import verl.workers.reward_manager.naive as naive
        from atr.adapter import ATRRewardManager
        naive.NaiveRewardManager = ATRRewardManager
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        atr_config: Optional[ATRConfig] = None,
        atr_config_dict: Optional[Dict[str, Any]] = None,  # yaml reward_kwargs 直传
        tool_cumulative_reward: float = 0.0,   # set to 0 to disable original tool reward
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        if atr_config is None and atr_config_dict:
            atr_config = ATRConfig(**atr_config_dict)
        self.atr = AdaptiveToolReward(config=atr_config or ATRConfig())
        self.tool_cumulative_reward = tool_cumulative_reward
        self.step_cnt = 0

    def __call__(self, data, return_dict=False):
        import torch
        from collections import defaultdict

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = data.batch.get('action_mask', data.batch['attention_mask'])

        # — Parse env_reward (original tool count reward, kept for comparison) —
        env_reward_tensor = data.batch.get('env_reward', None)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            uid = data_item.non_tensor_batch.get("uid", None)

            # — Step 1: Compute accuracy (original) —
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            accuracy = 1.0 if score.get("is_answer_right", False) else 0.0

            # — Step 2: Parse tool calls from response —
            tool_calls = parse_tool_trajectory(response_str)
            tool_calls = approximate_tool_outputs(response_str, tool_calls)

            # — Extract question from extra_info —
            question = None
            if extra_info and isinstance(extra_info, dict):
                question = extra_info.get("question", None)

            # — Step 3: Compute ATR reward (question-aware) —
            # extra_info 契约(数据集侧保证):
            #   question: 问题文本(utility 的 question-aware 打分)
            #   gt_bbox:  目标物体 GT bbox,归一化 [x1,y1,x2,y2](utility 的 IoU 匹配)
            #   options:  选项列表(compute_score 的字母映射匹配)
            #   image_size: 原始图像尺寸(仅存档,不参与计算)
            atr_reward, components = self.atr.compute(
                tool_calls=tool_calls,
                final_answer=response_str,
                accuracy=accuracy,
                ground_truth=extra_info,
                question=question,
                # 模型 bbox 是归一化 [0,1] 坐标,lazy-crop 面积比按归一化
                # 面积直接计算 → image_size 恒为 (1,1)。
                # (离线评估 trace 记录像素坐标,语义不同,勿混用)
                image_size=(1.0, 1.0),
            )
            components["accuracy_raw"] = score.get("score", 0.0)

            # — Step 4: Write reward to EOS position —
            reward_tensor[i, valid_response_length - 1] = atr_reward

            # — Collect extra info —
            for key, value in components.items():
                reward_extra_info[key].append(value)
            # naive.py 原版会把 score dict 全部键写入 extra_info；
            # val 路径（ray_trainer._validate）硬读 is_answer_right / acc_score，缺键即 KeyError
            reward_extra_info['is_answer_right'].append(bool(accuracy))
            reward_extra_info['acc_score'].append(components['accuracy_raw'])
            reward_extra_info['ability'].append(extra_info.get('ability', 'unknown') if isinstance(extra_info, dict) else 'unknown')
            reward_extra_info['ground_truth'].append(ground_truth)
            reward_extra_info['data_source'].append(data_source)
            reward_extra_info['uid'].append(uid)

            if self.step_cnt < self.num_examine:
                print(f"[ATR] acc={accuracy}  U={components['utility']:.3f}  "
                      f"C={components['cost']:.3f}  S={components['sequence']:.3f}  "
                      f"→ R={atr_reward:.3f}")

            self.step_cnt += 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


# —————————————————————————————————————
#  Quick patch for main_ppo.py (推荐集成方式)
# —————————————————————————————————————
PATCH_INSTRUCTIONS = '''
To use ATRRewardManager in verl training (verl_agents), patch
`verl/trainer/main_ppo.py` to add an "atr" branch:

--- a/verl_agents/verl/trainer/main_ppo.py
+++ b/verl_agents/verl/trainer/main_ppo.py
@@
         if reward_manager_name == "naive":
             from verl.workers.reward_manager import NaiveRewardManager

             reward_manager_cls = NaiveRewardManager
+        elif reward_manager_name == "atr":
+            from atr.adapter.patch_reward import ATRRewardManager

+            reward_manager_cls = ATRRewardManager
         elif reward_manager_name == "prime":

Then in the launch config set:
  reward_model.reward_manager: atr
  reward_model.reward_kwargs: {atr_config_dict: {lambda_u: 1.0, gamma_c: 0.5, eta_s: 0.3}}
  reward_model.custom_reward_function.path: atr.adapter.score
  reward_model.custom_reward_function.name: compute_vstar_score
The atr package is importable because /root/code (project root) is on PYTHONPATH.
'''

if __name__ == "__main__":
    # Quick self-test
    sample_response = '''
<tool_call>
{"name": "zoom_in", "arguments": {"bbox_2d": [100, 200, 300, 400], "label": "screen"}}
</tool_call>
The cropped region shows a blue button.
<tool_call>
{"name": "ocr", "arguments": {"bbox_2d": [100, 200, 300, 400]}}
</tool_call>
The text reads "Submit".
<answer>
\\boxed{Submit}
</answer>
'''
    calls = parse_tool_trajectory(sample_response)
    calls = approximate_tool_outputs(sample_response, calls)
    print(f"Parsed {len(calls)} tool calls:")
    for c in calls:
        print(f"  - {c.get('tool_name')}: output={c.get('output', '')[:50]}")
