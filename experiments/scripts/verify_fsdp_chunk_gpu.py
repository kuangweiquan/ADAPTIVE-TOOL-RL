"""GPU 验证:chunk 级 lm_head 在真实 FSDP1(world>1, nccl)下工作。

与生产 dp_actor + verl qwen3_vl.forward_with_normal_backend 相同的实现与
调用时机——**FSDP 根 forward 内**(verl 的 return_hidden_states 分支在模型
forward 期间调用 patch 过的 lm_head,此时 flat 处于 unshard 状态,模块
参数 = 全量 view):迷你模型(hidden=64, vocab=8192)经 FSDP1
(use_orig_params=False)分片到 2 进程:

  1. 裸参数 junk(37,64) 放在 lm_head 前 → flat 顺序 = 注册顺序 → shard
     边界 (total/2) 恰好落在 lm_head 某行中间(真实分片行分裂场景)
  2. Mini.forward 调用 patch 过的 self.head(模拟 verl return_hidden_states
     分支在模型 forward 内调用 lm_head)→ chunk 级 logprobs 直接在 FSDP
     forward 内算出(权重全量 view 直接切片,无手工 all_gather)
  3. 断言:
     - logprobs vs 全量 log_softmax 参考 maxdiff < 1e-3
     - optimizer step 后 lm_head 权重变化 > 0(梯度经 FSDP AllGather
       backward reduce-scatter 正常回传 flat shard)

排障史(2026-08-12 4x3090 实机):
  - 1 卡 2 进程 NCCL duplicate GPU 不可行;2 卡 2 进程需 set_device(local_rank)
  - **FSDP forward 外调用在结构上不可行**:reshard 后 flat 的 shape 恒为
    「全量」(534848)而数据只是本地分片(267424)。分片坐标切片 → 数据对
    但 SliceBackward 输出 shard 大小梯度与 flat 期望的全量形状冲突
    (`got [267424] but expected [534848]`,实机复现);全量坐标切片 → 梯度
    形状对但数据错位;`_local_shard`(=flat.data, detached)与
    `FSDP.summon_full_params`(纯数据交换, 无 autograd Function)梯度全断
  - 手工 all_gather 组装(shard 坐标 + intra_param_start_idx + index_copy)
    数值正确(9.5e-07)但梯度无法与 FSDP 的归约机制兼容——Flat 的梯度
    只能由 FSDP 自己的 AllGather backward 产生
  - **正解 = 移进 FSDP forward 内**:模型 forward 期间 flat unshard,模块
    参数是全量 view,直接切片算 chunk logprobs,梯度走 AllGather backward
    reduce-scatter(本脚本即此路径的验证)

用法(2 卡 2 进程):
  CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
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


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        # 裸参数放 lm_head 前:flat 顺序 = 注册顺序。
        # total = 37*64 + 8192*64 = 526656 → shard 边界 263328 落在
        # lm_head 行 (263328-2368)/64 = 4077.5 → 行被 shard 边界切开。
        self.junk = nn.Parameter(torch.randn(37, HIDDEN))
        self.head = nn.Linear(HIDDEN, VOCAB)
        nn.init.normal_(self.head.weight, std=0.02)
        self.head.bias.data.zero_()

    def forward(self, x):
        # 模拟 verl qwen3_vl.forward_with_normal_backend 的
        # return_hidden_states 分支:模型 forward 内调用 (patch 过的) lm_head
        return self.head(x)


def chunked_forward_impl(lm_head_module, x):
    """与生产 dp_actor._vstar_chunked_forward 相同的实现(生产代码复制)。

    在 FSDP 根 forward 内被调用:flat 处于 unshard 状态,模块参数
    `lm_head.weight` 是 (vocab, hidden) 的全量 view,直接按 chunk 切片
    (weight_gather_fn=None, logprobs_from_hidden 内部切片)即可;梯度经
    view → AllGather backward(reduce-scatter)回 flat shard,与 FSDP 的
    归约机制完全兼容。无需手工 all_gather / index_copy / 坐标换算。
    """
    if x.dim() == 3:  # verl 模型 forward 传 (1, nnz, hidden)
        x = x.squeeze(0)
    calc_entropy = getattr(lm_head_module, "_calc_entropy", False)
    if calc_entropy:
        lp, ent = logprobs_from_hidden(
            hidden=x,
            lm_head_weight=lm_head_module.weight,
            labels=lm_head_module._labels,
            temperature=lm_head_module._temp,
            compute_entropy=True,
        )
        lm_head_module._entropy_out = ent
        return lp
    return logprobs_from_hidden(
        hidden=x,
        lm_head_weight=lm_head_module.weight,
        labels=lm_head_module._labels,
        temperature=lm_head_module._temp,
    )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    if rank == 0:
        print(f"[VERIFY] world={dist.get_world_size()} device={local_rank}", flush=True)

    model = Mini().cuda()
    model = FSDP(model, use_orig_params=False)

    # 参考全量权重(summon 后取)
    with torch.no_grad():
        with FSDP.summon_full_params(model):
            w_ref = model.head.weight.detach().clone()

    x = torch.randn(1, 5, HIDDEN, device="cuda", requires_grad=True)  # 模拟 3D last_hidden_state
    labels = torch.randint(0, VOCAB, (5,), device="cuda")

    # 生产等价:patch lm_head.forward(在第一次模型 forward 前完成,生产在
    # dp_actor.__init__ 里做),然后 FSDP forward 内自动调用 chunked 实现
    model.head._labels = labels
    model.head._temp = TEMP
    model.head.forward = lambda h: chunked_forward_impl(model.head, h)
    lp = model(x)  # FSDP forward 内得到 chunked logprobs

    # 参考:全量 log_softmax
    logits_ref = (x.squeeze(0).detach() @ w_ref.T) / TEMP
    lp_ref = torch.log_softmax(logits_ref, dim=-1)[torch.arange(5), labels]
    err = (lp - lp_ref).abs().max().item()
    if rank == 0:
        print(f"[VERIFY] logprobs maxdiff vs full-softmax: {err:.3e}", flush=True)
    assert err < 1e-3, f"logprob mismatch {err}"

    # entropy 微批(生产 calculate_entropy=True):同一 chunked forward 带
    # compute_entropy,额外返回分块累计的策略熵;参考 = 全量
    # entropy_from_logits(temperature-scaled logits,与生产旧路径一致)
    from verl.utils.torch_functional import entropy_from_logits  # noqa: E402

    model.head._calc_entropy = True
    lp2 = model(x)
    ent = model.head._entropy_out
    ent_ref = entropy_from_logits(logits_ref)
    ent_err = (ent - ent_ref).abs().max().item()
    if rank == 0:
        print(f"[VERIFY] entropy maxdiff vs full: {ent_err:.3e}", flush=True)
    assert ent_err < 1e-3, f"entropy mismatch {ent_err}"

    # backward + optimizer step(梯度经 FSDP AllGather backward 回 flat shard;
    # 两个图都回传,模拟训练中 logprobs 与 entropy 微批交替)
    loss = -(lp.sum() + lp2.sum())
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
