"""Convert vstar_bench RL parquet -> json array readable by vendored RLHFDataset.

The vendored verl (PyVision-RL 0.2.0.dev) RLHFDataset only reads json arrays
(json.load), not parquet. Fields are already in the expected shape:
  prompt: chat messages list (system + user, user contains <image> placeholder)
  mm_hint: {"hint_type": "image", "hint_path": absolute_path}
  reward_model: {"ground_truth": ..., "style": "model"}
  extra_info: {question, options, gt_bbox, image_size, index}
Additionally we add an `image` column (data.image_key, yaml default "image")
pointing to the same absolute path so `_build_messages_pyvision` converts the
<image> placeholder into an image content item and the processor renders a real
image token in input_ids.
"""
import argparse
import json

import numpy as np
import pandas as pd


def to_native(obj):
    """Recursively convert numpy scalars/arrays to native python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_native(v) for v in obj]
    return obj


def convert(parquet_path: str, json_path: str) -> None:
    df = pd.read_parquet(parquet_path)
    records = []
    for row in df.to_dict("records"):
        rec = to_native(dict(row))
        rec["image"] = rec["mm_hint"]["hint_path"]  # image_key used by _build_messages_pyvision
        records.append(rec)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(records)} records -> {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_path", required=True)
    ap.add_argument("--json_path", required=True)
    args = ap.parse_args()
    convert(args.parquet_path, args.json_path)
