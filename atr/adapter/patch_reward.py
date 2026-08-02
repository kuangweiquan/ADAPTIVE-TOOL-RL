"""Adapter: Inject ATR reward into PyVision-RL's training loop.

Integration point: `NaiveRewardManager.__call__()` in
  pyvision-rl/verl_agents/verl/workers/reward_manager/naive.py

Strategy: Create a custom reward manager that subclasses NaiveRewardManager
and replaces the env_reward + acc_reward logic with ATR.compute().
"""

from typing import Optional, List, Dict, Any
import re

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
        - bbox: [x1,y1,x2,y2] (if available)
        - output: str  (the tool's return observation, approximated)
    """
    tool_calls = []

    # Match <tool_call> blocks
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(response_str):
        try:
            call_data = eval(match.group(1))
            if isinstance(call_data, list):
                call_data = call_data[0] if call_data else {}
            record = {"tool_name": call_data.get("name", "unknown")}
            args = call_data.get("arguments", {})
            if "bbox_2d" in args:
                record["bbox"] = args["bbox_2d"]
            if "label" in args:
                record["label"] = args["label"]
            # Approximate output: the text between this tool_call and next
            tool_calls.append(record)
        except Exception:
            continue

    # Fill in approximate outputs (text between consecutive tool calls)
    segments = pattern.split(response_str)
    for i, call in enumerate(tool_calls):
        # Output is the text after </tool_call> and before next <tool_call>
        idx_in_segments = segments.index  # This won't work properly
        pass

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
        tool_cumulative_reward: float = 0.0,   # set to 0 to disable original tool reward
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
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
            atr_reward, components = self.atr.compute(
                tool_calls=tool_calls,
                final_answer=response_str,
                accuracy=accuracy,
                ground_truth=extra_info,
                question=question,
                # image_size can be extracted from env if available
            )
            components["accuracy_raw"] = score.get("score", 0.0)

            # — Step 4: Write reward to EOS position —
            reward_tensor[i, valid_response_length - 1] = atr_reward

            # — Collect extra info —
            for key, value in components.items():
                reward_extra_info[key].append(value)
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
#  Quick patch for naive.py
# —————————————————————————————————————
PATCH_INSTRUCTIONS = '''
To replace NaiveRewardManager with ATRRewardManager in PyVision-RL:

--- a/verl_agents/verl/workers/reward_manager/naive.py
+++ b/verl_agents/verl/workers/reward_manager/naive.py
@@ -1,3 +1,11 @@
+import sys
+sys.path.insert(0, "..")  # add project root
+from atr.adapter.patch_reward import ATRRewardManager
+from atr.config import ATRConfig
+
 # Replace the class at module level:
-NaiveRewardManager
+NaiveRewardManager = ATRRewardManager
+
+# Or for runtime config:
+# NaiveRewardManager = lambda *a, **kw: ATRRewardManager(*a, **kw, atr_config=ATRConfig(lambda_u=1.0, gamma_c=0.5, eta_s=0.3))
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
