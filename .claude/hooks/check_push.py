#!/usr/bin/env python3
"""PrePush guard for the Adaptive-Tool-RL repo (runs on local & remote).

Blocks a push if it would add files that violate the sync protocol
(see CLAUDE.md "Git 同步协议"):
  - files > 50 MB (GitHub hard limit is 100 MB per file)
  - files under blocked data / output dirs
  - files with blocked binary / data extensions

Data and artifacts never enter git — they travel via bundle tar / cloud drive.
Exit 0 = ok, 2 = blocked (Claude Code PrePush hook semantics).
"""

import os
import subprocess
import sys

SIZE_LIMIT = 50 * 1024 * 1024  # 50 MiB

BLOCKED_DIRS = (
    "datasets/vstar_bench/direct_attributes/",
    "datasets/vstar_bench/relative_position/",
    "datasets/vstar_bench/crops/",
    "datasets/vstar_bench/sft_images/",
    "datasets/vstar_bench/sft_v1_backup/",
    "log/",
    "experiments/results/",
    "checkpoints/",
)

BLOCKED_EXT = (
    ".tar.gz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".parquet",
    ".npy",
    ".npz",
    ".h5",
)

GITKEEP = ".gitkeep"


def run_git(args):
    try:
        proc = subprocess.run(["git"] + args, capture_output=True, text=True)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def files_to_check():
    """Files that would leave the repo on the next push:
    committed-but-unpushed (origin/main..HEAD) + staged-but-uncommitted.
    Deleted files are irrelevant (they shrink the repo)."""
    files = []
    for ref in ("origin/main..HEAD",):
        out = run_git(["diff", "--name-only", "--diff-filter=ACMR", ref])
        if out is not None:
            files.extend(out.splitlines())
    out = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if out is not None:
        files.extend(out.splitlines())
    return list(dict.fromkeys(f.strip() for f in files if f.strip()))


def is_blocked(path):
    """Return a reason string if the file violates the protocol, else None."""
    low = path.replace("\\", "/").lower()
    if low.endswith(GITKEEP):
        return None
    if any(low.endswith(ext) for ext in BLOCKED_EXT):
        return "blocked extension"
    for d in BLOCKED_DIRS:
        if low.startswith(d):
            return "blocked dir"
    try:
        if os.path.exists(path) and os.path.getsize(path) > SIZE_LIMIT:
            return "> 50 MB"
    except OSError:
        pass
    return None


def main():
    try:
        problems = []
        for f in files_to_check():
            reason = is_blocked(f)
            if reason:
                problems.append("  %s  (%s)" % (f, reason))
        if not problems:
            return 0
        print(
            "Push blocked by .claude/hooks/check_push.py — these files violate the sync "
            "protocol (数据/产物不进 git, see CLAUDE.md):\n"
            + "\n".join(problems)
            + "\nFix: `git rm --cached <file>` and commit, or move the file out of a "
            "blocked path.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # fail-open on unexpected errors, never brick pushes
        print("check_push.py warning (push not blocked): %s" % exc, file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
