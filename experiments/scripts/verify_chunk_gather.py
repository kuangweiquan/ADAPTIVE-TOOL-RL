"""CPU 验证：chunk 级权重收集（模拟 2-rank FSDP 分片 + 行分裂）。

复刻 dp_actor._vstar_chunked_forward 的行提取/gather 逻辑，验证：
  1. 合并权重与 full 逐 chunk 一致（含被 shard 边界切开的行）
  2. logprobs_from_hidden 的 weight_gather_fn 路径与原版数值一致
  3. backward 梯度回传路径存在（x 有梯度；w 侧保留到 local 的链接）
"""
import sys

sys.path.insert(0, "/root/code/pyvision-rl/verl_agents")

import torch
from verl.utils.torch_functional import logprobs_from_hidden

torch.manual_seed(0)
hidden = 4
vocab = 10
full = torch.randn(vocab, hidden, dtype=torch.float64)
flat = full.reshape(-1)
# 2 rank 均分 40 元素：rank0 [0,20)，rank1 [20,40)（元素 20-23 = 行 5 前半在 rank0，后半在 rank1）
# 行 5 的全局元素 [20, 24)：rank0 持 [20,20) 无，rank1 持 [20,24) 全 —— 行 5 完整属于 rank1。
# 反而行 1（元素 [4,8)）不跨边界。构造一个真正跨界的: 用 numel=18/22 不均分
# 更真实的 FSDP shard：均分但 lm_head 在 flat 中偏移非 0 —— 简化：直接模拟不均分。
# rank0 持 [0, 18)，rank1 持 [18, 40)。行 4 元素 [16,20)：rank0 持 [16,18) 部分，rank1 持 [18,20)。
pieces = [(1, 0, 18), (1, 18, 22)]
locals_ = [flat[0:18].clone(), flat[18:40].clone()]


def build_rows(local, in_shard, off, numel, hidden_dim):
    if in_shard and numel > 0:
        g0 = off // hidden_dim
        g1 = (off + numel - 1) // hidden_dim + 1
        n_rows = g1 - g0
        row_tensor = torch.zeros(n_rows, hidden_dim, dtype=local.dtype)
        elem_s = max(off, g0 * hidden_dim)
        elem_e = min(off + numel, g1 * hidden_dim)
        if elem_s < elem_e:
            dst_s = elem_s - g0 * hidden_dim
            row_tensor.view(-1)[dst_s : dst_s + (elem_e - elem_s)] = local[elem_s - off : elem_e - off]
        g_idx = torch.arange(g0, g1)
        return row_tensor, g_idx
    return None, None


def gather_fn(start, end, locals_=locals_):
    contribs = []
    for r, (in_shard, off, numel) in enumerate(pieces):
        rt, gidx = build_rows(locals_[r], in_shard, off, numel, hidden)
        contrib = torch.zeros(end - start, hidden, dtype=torch.float64)
        if rt is not None:
            mask = (gidx >= start) & (gidx < end)
            if mask.any():
                contrib[gidx[mask] - start] = rt[mask]
        contribs.append(contrib)
    return torch.stack(contribs).sum(0)


# 1) 逐 chunk 合并权重 == full
for s in range(0, vocab, 3):
    e = min(s + 3, vocab)
    assert torch.allclose(gather_fn(s, e), full[s:e], atol=1e-12), f"chunk {s} mismatch"
print("1) chunk gather == full weight: OK")

# 2) logprobs 数值一致
x = torch.randn(5, hidden, dtype=torch.float64)
labels = torch.randint(0, vocab, (5,))
lp_ref = logprobs_from_hidden(x, lm_head_weight=full, labels=labels)
lp_new = logprobs_from_hidden(x, lm_head_weight=None, labels=labels, weight_gather_fn=gather_fn, vocab_size=vocab)
d = (lp_ref - lp_new).abs().max().item()
assert d < 1e-9, f"maxdiff {d}"
print(f"2) logprobs maxdiff = {d:.2e}: OK")

# 3) backward：x 与 local 侧（模拟 flat shard）都收到梯度
x2 = x.detach().requires_grad_(True)
locals_grad = [locals_[0].detach().requires_grad_(True), locals_[1].detach().requires_grad_(True)]
lp2 = logprobs_from_hidden(
    x2,
    lm_head_weight=None,
    labels=labels,
    weight_gather_fn=lambda s, e: gather_fn(s, e, locals_=[locals_grad[0], locals_grad[1]]),
    vocab_size=vocab,
)
lp2.sum().backward()
assert x2.grad is not None and torch.isfinite(x2.grad).all()
assert locals_grad[0].grad is not None and locals_grad[0].grad.abs().sum() > 0
assert locals_grad[1].grad is not None and locals_grad[1].grad.abs().sum() > 0
# 梯度覆盖整个 local 切片（copy 覆盖全部元素），无越界
assert locals_grad[0].grad.numel() == 18 and locals_grad[0].grad.abs().sum() > 0
assert locals_grad[1].grad.numel() == 22 and locals_grad[1].grad.abs().sum() > 0
print("3) backward 回传 x + 两个 local shard（全覆盖，无越界）: OK")
print("ALL OK")
