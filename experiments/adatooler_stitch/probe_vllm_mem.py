#!/usr/bin/env python
"""Ground-truth vLLM 0.11 memory probe for Arm A (GPU-only; run on the remote box).

Answers the open question from ai_handoff/12 §2: does plain vLLM 0.11 obey
tp=2 + gpu_memory_utilization=0.5 (~12G/card expected) or land at ~21G/card
like the verl_tool async engine measured on 2026-08-15 (21.28G/card)?

One config per invocation; the process exits naturally when done. On this
platform (SeetaCloud, drv 570) KILLED vLLM workers leak driver-side zombie
allocations, so never kill a running probe - let it finish or fail on its own
(a failed init self-exits cleanly).

Default knobs mirror the verl_tool engine exactly (vllm_async_server.py
launch_server + train_arm_a_4x3090.sh): tp=2, gmem=0.5, max_model_len=16384
(8192 prompt + 8192 response), max_num_seqs=256, max_num_batched_tokens=10000,
enforce_eager=True, enable_chunked_prefill=True, enable_sleep_mode=True
(hardcoded by verl_tool), disable_custom_all_reduce=True (hardcoded by
verl_tool), dtype=bfloat16, load_format=safetensors.

Note: with sleep mode ON the KV cache may live in a CuMemAllocator pool that
torch does not see - nvidia-smi and torch numbers can diverge; that divergence
is itself the signal. nvidia-smi (--query-gpu / --query-compute-apps) is the
authoritative measurement; the coordinator-process torch numbers are expected
to be near zero because vLLM 0.11 V1 runs the engine core in a child process.
"""
import argparse
import os
import subprocess
import sys


def nvidia_smi(query):
    return subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/root/autodl-tmp/models/AdaTooler-V-SFT-model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gmem", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=10000)
    ap.add_argument("--sleep", action=argparse.BooleanOptionalAction, default=True,
                    help="mirrors verl_tool hardcoded enable_sleep_mode=True")
    ap.add_argument("--custom-allreduce", action=argparse.BooleanOptionalAction, default=True,
                    help="mirrors verl_tool hardcoded disable_custom_all_reduce=True")
    ap.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True,
                    help="mirrors verl RolloutConfig default enable_chunked_prefill=True")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # fail fast if this box does not even have tp GPUs clean (driver also guards)
    import torch  # noqa: E402  (GPU-only module, imported lazily for local py_compile)
    visible = torch.cuda.device_count()
    if visible < args.tp:
        print(f"[probe] FATAL: {visible} visible GPU(s) < tp={args.tp} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        return 1
    print(f"[probe] start tag={args.tag} tp={args.tp} gmem={args.gmem} sleep={args.sleep} "
          f"car={args.custom_allreduce} chunked={args.chunked_prefill} "
          f"mml={args.max_model_len} mns={args.max_num_seqs} mnbt={args.max_num_batched_tokens} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gmem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        enable_chunked_prefill=args.chunked_prefill,
        enable_sleep_mode=args.sleep,
        disable_custom_all_reduce=args.custom_allreduce,
        trust_remote_code=True,
        dtype="bfloat16",
        load_format="safetensors",
    )

    from vllm import LLM  # noqa: E402
    sleep_applied = args.sleep
    try:
        llm = LLM(**kwargs)
    except TypeError as e:  # older vLLM API without enable_sleep_mode
        if "enable_sleep_mode" not in str(e):
            raise
        kwargs.pop("enable_sleep_mode")
        sleep_applied = False
        print("[probe] enable_sleep_mode rejected by LLM(), retrying without", flush=True)
        llm = LLM(**kwargs)

    # engine-core child process holds weights/KV cache; coordinator torch counters stay ~0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"[probe] torch gpu{i} (coordinator proc): reserved={torch.cuda.memory_reserved(i)/2**30:.2f}GiB "
              f"allocated={torch.cuda.memory_allocated(i)/2**30:.2f}GiB "
              f"used={(total-free)/2**30:.2f}GiB/{total/2**30:.2f}GiB", flush=True)
    print("[probe] nvidia-smi gpus:\n" + nvidia_smi("--query-gpu=index,memory.used,memory.total"), flush=True)
    apps = nvidia_smi("--query-compute-apps=pid,used_memory")
    print("[probe] nvidia-smi compute-apps:\n" + (apps or "(none)"), flush=True)

    vllm_config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    if vllm_config is not None:
        print("[probe] parallel_config:", getattr(vllm_config, "parallel_config", None), flush=True)
        print("[probe] scheduler_config:", getattr(vllm_config, "scheduler_config", None), flush=True)
        print("[probe] cache_config:", getattr(vllm_config, "cache_config", None), flush=True)

    smi_used = {}
    for row in nvidia_smi("--query-gpu=index,memory.used").splitlines():
        idx, used = [c.strip() for c in row.split(",")]
        smi_used[int(idx)] = int(used)
    cells = [f"gpu{i}:smi_used={smi_used.get(i, '?')}MiB" for i in range(torch.cuda.device_count())]
    print(f"[probe] SUMMARY tag={args.tag} tp={args.tp} gmem={args.gmem} "
          f"mml={args.max_model_len} mns={args.max_num_seqs} mnbt={args.max_num_batched_tokens} "
          f"sleep={sleep_applied} car={args.custom_allreduce} chunked={args.chunked_prefill} "
          + " ".join(cells), flush=True)
    # natural exit: on this platform only self-exits release GPU memory cleanly
    return 0


if __name__ == "__main__":
    sys.exit(main())
