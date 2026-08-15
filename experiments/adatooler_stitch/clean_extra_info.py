"""Convert numpy types in subset parquet extra_info to plain Python types.

Why: released adatooler_v.py save-record path does a plain json.dump of
extra_info; after a pandas read_parquet roundtrip, list fields (images) come
back as ndarray and int fields may come back as np.int64, both of which fail
json.dumps. Their Rl_data parquet was written so that no numpy objects
survive. Run this after prepare_subset.py (re)generation.

Usage (remote, verl-tool env):
    python clean_extra_info.py /root/autodl-tmp/datasets/adatooler_v_subset
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def clean(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for name in ("train.parquet", "val.parquet"):
        f = out / name
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if "extra_info" in df.columns:
            df["extra_info"] = df["extra_info"].apply(clean)
        df.to_parquet(f, index=False)
        print(f"cleaned {f} ({len(df)} rows)")


if __name__ == "__main__":
    main()
