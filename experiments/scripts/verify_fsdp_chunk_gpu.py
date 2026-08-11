"""GPU 验证：chunk 级权重收集在真实 FSDP1（world>1, nccl）下工作。

与生产 `_vstar_chunked_forward`（dp_actor.py）完全相同的实现，迷你模型
（hidden=64, vocab=8192）经 FSDP1（use_orig_params=False）分片到 2 进程，
共享同一张 GPU：

  1. 裸参数 junk(37,64) 放在 lm_head 前 → flat 顺序 = 注册顺序 → shard
     边界 (total/2) 恰好落在 lm_head 某行中间（真实分片行分裂场景）
  2. patch lm_head.forward = chunked 实现，走完整 model(x)（FSDP pre-hook
     unshard → patch forward 读 _local_shard → chunk all_gather → backward
     FSDP post-hook reduce-scatter → optimizer step）
  3. 断言：
     - logprobs vs 全量 log_softmax 参考 maxdiff < 1e-3
     - optimizer step 后 lm_head 权重变化 > 0（梯度真正回传 FSDP flat shard）

用法（1 卡 2 进程，nccl 同设备）：
  NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=2 --master_port=29599 verify_fsdp_chunk_gpu.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VERL_ROOT = os.path.join(PROJECT_ROOT, "pyvision-rl", "verl_agents")
for p in (PROJECT_ROOT, VERL_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from verl.utils.torch_functional import logprobs_from_hidden  # noqa: E402  (生产钩子函数)

HIDDEN = 64
VOCAB = 8192
TEMP = 0.7
CHUNK = 4096  # logprobs_from_hidden 默认 chunk → 2 次 gather_chunk 调用


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        # 裸参数放 lm_head 前：flat 顺序 = 注册顺序。
        # total = 37*64 + 8192*64 = 526656 → shard 边界 263328 落在
        # lm_head 行 (263328-2368)/64 = 4077.5 → 行被 shard 边界切开。
        self.junk = nn.Parameter(torch.randn(37, HIDDEN))
        self.head = nn.Linear(HIDDEN, VOCAB)
        nn.init.normal_(self.head.weight, std=0.02)
        self.head.bias.data.zero_()

    def forward(self, x):
        return self.head(x)


def chunked_forward_impl(model, lm_head_module, x):
    """与 dp_actor._vstar_chunked_forward 相同的实现（生产代码复制）。"""
    flat = model._flat_param
    target = None
    for i, info in enumerate(flat._param_infos):
        if info.module is lm_head_module:
            target = i
            break
    if target is None:
        raise RuntimeError("[CHUNK] lm_head not found in FSDP flat params")
    si = flat._shard_param_infos[target]
    hidden_dim = x.shape[-1]
    world = dist.get_world_size()
    local = flat._local_shard.to(x.device)
    in_shard = int(si.in_shard)
    off = int(si.offset_in_shard or 0)
    numel = int(si.numel_in_shard or 0)
    meta = torch.tensor([in_shard, off, numel], dtype=torch.long, device=x.device)
    metas = [torch.empty(3, dtype=torch.long, device=x.device) for _ in range(world)]
    dist.all_gather(metas, meta)
    total_numel = sum(int(m[2]) for m in metas if int(m[0]))
    vocab_size = total_numel // hidden_dim
    if dist.get_rank() == 0:
        print(
            f"[CHUNK] pieces={[(int(m[0]), int(m[1]), int(m[2])) for m in metas]} vocab={vocab_size}",
            flush=True,
        )
    if in_shard and numel > 0:
        g0 = off // hidden_dim
        g1 = (off + numel - 1) // hidden_dim + 1
        n_rows = g1 - g0
        row_tensor = torch.zeros(n_rows, hidden_dim, dtype=local.dtype, device=x.device)
        elem_s = max(off, g0 * hidden_dim)
        elem_e = min(off + numel, g1 * hidden_dim)
        if elem_s < elem_e:
            dst_s = elem_s - g0 * hidden_dim
            row_tensor.view(-1)[dst_s : dst_s + (elem_e - elem_s)] = local[elem_s - off : elem_e - off]
        g_idx = torch.arange(g0, g1, device=x.device)
    else:
        row_tensor = None
        g_idx = None

    def gather_chunk(start, end):
        contrib = torch.zeros(end - start, hidden_dim, dtype=local.dtype, device=x.device)
        if row_tensor is not None:
            mask = (g_idx >= start) & (g_idx < end)
            if mask.any():
                contrib[g_idx[mask] - start] = row_tensor[mask]
        gathered = torch.distributed.nn.all_gather(contrib)  # autograd-aware
        return torch.stack(gathered).sum(dim=0)

    return logprobs_from_hidden(
        hidden=x,
        lm_head_weight=None,
        labels=lm_head_module._labels,
        temperature=lm_head_module._temp,
        weight_gather_fn=gather_chunk,
        vocab_size=vocab_size,
    )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    if rank == 0:
        print(f"[VERIFY] world={dist.get_world_size()} device=0 (2 procs share 1 GPU)", flush=True)

    model = Mini().cuda()
    model = FSDP(model, use_orig_params=False)

    # 参考全量权重（summon 后取）
    with torch.no_grad():
        with FSDP.summon_full_params(model):
            w_ref = model.head.weight.detach().clone()

    x = torch.randn(5, HIDDEN, device="cuda")  # 模拟 last_hidden_state（需梯度）
    labels = torch.randint(0, VOCAB, (5,), device="cuda")
    model.head._labels = labels
    model.head._temp = TEMP
    model.head.forward = lambda h: chunked_forward_impl(model, model.head, h)

    out = model(x)  # 完整 FSDP forward（pre-hook unshard → patch forward → post-hook）
    lp = out[torch.arange(5), labels]

    # 参考：全量 log_softmax
    logits_ref = (x.detach() @ w_ref.T) / TEMP
    lp_ref = torch.log_softmax(logits_ref, dim=-1)[torch.arange(5), labels]
    err = (lp - lp_ref).abs().max().item()
    if rank == 0:
        print(f"[VERIFY] logprobs maxdiff vs full-softmax: {err:.3e}", flush=True)
    assert err < 1e-3, f"logprob mismatch {err}"

    # backward（FSDP post-hook 自动 reduce-scatter）+ optimizer step
    loss = -lp.sum()
    loss.backward()
    before = {}
    with FSDP.summon_full_params(model):
        for n, p in model.named_parameters():
            before[n] = p.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    after = {}
    with FSDP.summon_full_params(model):
        for n, p in model.named_parameters():
            after[n] = p.detach().clone()
    head_delta = (after["head.weight"] - before["head.weight"]).abs().max().item()
    junk_delta = (after["junk"] - before["junk"]).abs().max().item()
    if rank == 0:
        print(f"[VERIFY] head.max_delta={head_delta:.6e} junk.max_delta={junk_delta:.6e}", flush=True)
    assert head_delta > 0, "lm_head 权重未更新——梯度未回传 FSDP flat shard！"
    if rank == 0:
        print("[VERIFY] ALL OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
