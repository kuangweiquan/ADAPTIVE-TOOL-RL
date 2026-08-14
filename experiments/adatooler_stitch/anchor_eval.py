"""Phase 0 anchor eval: official V* 191-question direct-answer MCQA.

Official protocol (paper 89.8% reference): question text with lettered options
from craigwu/vstar_bench test_questions.jsonl (their VStar.json is equivalent),
answer letter parsed from <answer>X</answer>. Runs on GPU (single GPU suffices,
7B bf16 ~15GB).

Usage (remote, GPU mode):
  python anchor_eval.py \
    --model /root/autodl-tmp/models/AdaTooler-V-SFT-model \
    --images_dir /root/autodl-tmp/datasets/vstar_official \
    --test_jsonl /root/autodl-tmp/datasets/vstar_official_test.jsonl \
    --out /root/autodl-tmp/eval_anchor_sft.jsonl \
    --limit 5   # smoke first, drop --limit for full 191
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.S)


def extract_letter(text: str):
    m = ANSWER_RE.search(text or "")
    if m:
        return m.group(1).upper()
    for ch in re.findall(r"\b([A-E])\b", text or ""):
        return ch
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    print(f"questions: {len(items)}", flush=True)

    print("loading model...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    img_dir = Path(args.images_dir)
    results, correct = [], 0
    for i, item in enumerate(items):
        img = Image.open(img_dir / Path(item["image"]).name).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": item["text"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        generated = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_letter(generated)
        hit = (pred == item["label"].upper())
        correct += hit
        results.append({"qid": item["question_id"], "label": item["label"],
                        "pred": pred, "hit": bool(hit), "generated": generated[:300]})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)} acc {correct / (i + 1):.3f}", flush=True)

    acc = correct / len(items)
    print(f"FINAL acc: {correct}/{len(items)} = {acc:.4f}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"acc": acc, "correct": correct, "n": len(items),
                   "results": results}, f, indent=1, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
