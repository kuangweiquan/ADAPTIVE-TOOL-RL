#!/usr/bin/env python3
"""Delta-S (Tool Benefit Score) offline precompute — remote GPU script (Arm D).

Runs in the verl-tool conda env on the remote 4x3090 machine. For each sample,
rolls out the SFT model twice:
  TOOL mode    : system prompt with <tools> section + 2-turn zoom_in loop
                 (mtrl observation format replicated from verl_tool agent_loop)
  NO-TOOL mode : guideline prompt without tools, single direct answer
Then Delta-S = acc(tool) - acc(no-tool), matching the paper's per-sample
Tool Benefit Score definition (arXiv:2512.16918). Output JSONL is consumed
by verl_reward_manager.py via reward_kwargs `delta_s_map`.

Faithfulness note: this reimplementation replicates the released agent-loop
prompt/mtrl format (verl_tool) and the released tool server observations;
the paper's internal Delta-S computation was not released (stub TB_score=1.0
in adatooler_v.py:229), so this is our re-implementation, documented as such.

Usage (remote, inside verl-tool env):
  bash examples/train/adatooler_v/  # start tool server first:
  python -m verl_tool.servers.serve --host 127.0.0.1 --port 30001 \
      --tool_type adatooler_v --workers_per_tool 8 &
  python delta_s_precompute.py \
      --model AdaTooler-V/AdaTooler-V-SFT-model \
      --dataset AdaTooler-V/AdaTooler-V-300k \
      --max_samples 1000 --tool_server http://127.0.0.1:30001 \
      --out delta_s_map.jsonl
"""

import argparse
import json
import re
import sys
import time

SYSTEM_TOOL = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

GUIDELINE = """Guidelines: Understand the given visual information and the user query.

Reason with the visual information step by step.
You should:
1. Explain your reasoning.
2. Provide the final answer.

Place your text reasoning process within the <think> </think> tags.
Place your final answer within the <answer> </answer> tags.
Do NOT call any tools — answer directly."""

TYPE_TEMPLATE = {
    "multiple choice": "{question}\nPlease answer with the option's letter from the given choices directly.",
    "numerical": "{question}\nPlease answer with a number directly.",
    "OCR": "{question}\nPlease transcribe the text exactly.",
    "free-form": "{question}\nPlease answer in a short phrase.",
}

# replicated from verl_tool agent_loop (verltool_agent_loop.py:180-186):
# mtrl_sep = turn_end_token + "\n" + chat_template(system, content="{obs}")
MTRL_OBS_FORMAT = "<|im_end|>\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
ACTION_STOP = "</tool_call>"
MAX_TURNS = 2


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def choices_score(predict: str, ground_truth: list) -> float:
    answer = [a.strip() for a in extract_answer(predict).split(",")]
    gt = [re.sub(r"</?answer>", "", str(x)).strip() for x in ground_truth]
    if len(answer) != len(gt):
        return 0.0
    return 1.0 if all(a in gt for a in answer) else 0.0


def call_tool_server(tool_server: str, trajectory_id: str, action: str, extra_fields: dict):
    import requests
    resp = requests.post(
        f"{tool_server}/get_observation",
        json={
            "tool_type": "adatooler_v",
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "extra_fields": [extra_fields],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()  # {"observations": [...], "dones": [...], "valids": [...]}


def rollout_once(llm, processor, sample, mode: str, tool_server: str, traj_id: str):
    """One rollout for one sample. mode: 'tool' | 'no_tool'."""
    images = sample["images"]
    question = sample["question"]
    problem_type = sample["problem_type"]
    system = SYSTEM_TOOL if mode == "tool" else GUIDELINE
    user_text = TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["multiple choice"]).format(
        question=question)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    image_paths = images[:1]  # single-image samples only
    n_tool_calls = 0

    for turn in range(MAX_TURNS if mode == "tool" else 1):
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False)
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image_paths}},
            sampling_params=None,  # set below in main via LLM defaults
        )
        text = outputs[0].outputs[0].text if mode == "tool" else outputs[0].outputs[0].text
        if mode == "no_tool":
            return extract_answer(text), 0

        # tool turn: did it emit a call before stopping?
        call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if not call_match:
            break
        try:
            call = json.loads(call_match.group(1))
        except json.JSONDecodeError:
            break
        action_xml = f'<tool_call>\n{json.dumps(call)}\n</tool_call>'
        result = call_tool_server(tool_server, traj_id, action_xml,
                                  {"image": image_paths[0]})
        observations = result.get("observations") or [f"[tool error]"]
        obs = observations[0]
        n_tool_calls += 1
        messages.append({"role": "assistant", "content": text + MTRL_OBS_FORMAT.format(obs=obs)})
        # continue the loop: next turn generates after the observation

    # final turn: answer
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False)
    outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image_paths}})
    text = outputs[0].outputs[0].text
    return extract_answer(text), n_tool_calls


def load_samples(dataset_ref: str, max_samples: int):
    """Load single-image samples from the HF RL dataset (raw format)."""
    from datasets import load_dataset
    ds = load_dataset(dataset_ref, split="train")
    samples = []
    for ex in ds:
        images = ex.get("image") or ex.get("images") or []
        if isinstance(images, dict):
            images = [images]
        # raw HF entries may be path strings or PIL objects; keep first image
        if not images:
            continue
        problem_type = ex.get("problem_type", "multiple choice")
        if problem_type == "video":
            continue
        samples.append({
            "uid": ex.get("id") or ex.get("uid") or str(len(samples)),
            "question": ex.get("question", ""),
            "images": images,
            "problem_type": problem_type,
            "ground_truth": ex.get("answer") or ex.get("ground_truth", []),
        })
        if len(samples) >= max_samples:
            break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--tool_server", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"[dS] loading model {args.model} ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048,
                                     stop=[ACTION_STOP])

    samples = load_samples(args.dataset, args.max_samples)
    print(f"[dS] {len(samples)} samples loaded", flush=True)

    t0 = time.time()
    with open(args.out, "w") as f:
        for idx, s in enumerate(samples):
            try:
                ans_tool, n_calls = rollout_once(llm, processor, s, "tool",
                                                 args.tool_server, f"dS-{s['uid']}")
                ans_notool, _ = rollout_once(llm, processor, s, "no_tool",
                                             args.tool_server, f"dS-{s['uid']}-nt")
            except Exception as e:
                print(f"[dS] sample {s['uid']} FAILED: {e}", flush=True)
                continue
            acc_tool = choices_score(ans_tool, s["ground_truth"])
            acc_notool = choices_score(ans_notool, s["ground_truth"])
            rec = {"uid": s["uid"], "delta_s": acc_tool - acc_notool,
                   "acc_tool": acc_tool, "acc_no_tool": acc_notool,
                   "n_tool_calls": n_calls}
            f.write(json.dumps(rec) + "\n")
            if (idx + 1) % 20 == 0:
                el = time.time() - t0
                print(f"[dS] {idx+1}/{len(samples)} done, {el:.0f}s elapsed "
                      f"(~{el/(idx+1):.1f}s/sample)", flush=True)

    print(f"[dS] done -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
