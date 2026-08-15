"""Env-level json tolerance for numpy objects (verl-tool env, Arm A infra fix).

The released adatooler_v.py save-record path does a plain json.dump of rollout
records; runtime fields (from the verl_tool agent-loop DataProto) can carry
numpy scalars/arrays which json refuses. We keep the released repo code
byte-identical and make the *environment* tolerant instead (same spirit as the
flash_attn 0.0.0 stub). Reward semantics are untouched: records are analysis
artifacts only.

Install on the remote (verl-tool env):

    cp sitecustomize_json_numpy.py \
        /root/autodl-tmp/envs/verl-tool/lib/python3.11/site-packages/sitecustomize.py

Diagnostic: the first ndarray conversion per process prints
`[json-np-tol] ndarray shape=... dtype=... -> list` to stderr (lands in the
training log) so the offending field is identifiable from the records.
"""
import json
import sys

import numpy as np

_orig_default = json.JSONEncoder.default
_reported = False


def _json_default(self, o):
    global _reported
    if isinstance(o, np.ndarray):
        if not _reported:
            _reported = True
            print(f"[json-np-tol] ndarray shape={o.shape} dtype={o.dtype} -> list", file=sys.stderr)
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return _orig_default(self, o)


json.JSONEncoder.default = _json_default


# --- json.load retry: released adatooler_v.py merges per-step record files with
# read-modify-write across concurrent RewardManagerWorkers; a load can hit a
# file being written by another worker (empty or half-written) -> JSONDecodeError.
# Retry with backoff; on exhaustion return [] (records are analysis artifacts,
# training must not die on them).
import time as _time

_orig_load = json.load


def _json_load_retry(fp, *args, **kwargs):
    for attempt in range(20):
        try:
            return _orig_load(fp, *args, **kwargs)
        except json.JSONDecodeError as e:
            if attempt == 19:
                print(f"[json-load-retry] giving up after 20 attempts: {e}", file=sys.stderr)
                return []
            try:
                if fp.seekable():
                    fp.seek(0)
            except Exception:
                pass
            _time.sleep(0.1 * (attempt + 1))
    return []


json.load = _json_load_retry
