"""2 卡 GPU 验证:真实 8B + 真实多模态数据 + 生产微批路径的峰值与数值。

模拟 dp_actor.update_policy 的一个微批(bsz=1,含图):
  rl_dataset 构建(processor 1M 像素预算 + postprocess_data + get_rope_index)
  -> 拼响应 -> pad 到 (1, 13312) -> unpad(rmpad) -> FSDP 根 forward
  (verl forward_with_normal_backend, return_hidden_states=True,
  chunked lm_head 在 forward 内算 logprobs) -> loss.backward()

目的:
  1. 确认新 processor(视觉 token 13160 -> 3800)下真实微批的峰值(4 卡崩时
     post-fwd 36-37.5G / 微批内部序列 ~17-18k tokens)
  2. 确认 chunked lm_head + 多模态 + FSDP(2 进程)在真实模型上数值/梯度正确
  3. 若 2 卡 24G 能跑通最坏微批(prompt 1900 + 响应 3224 + 视觉 3800),
     4 卡 24G 大概率稳(每卡分片减半 + 峰值相近)

用法(2 卡):
  CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
  torchrun --nproc_per_node=2 --master_port=29598 verify_microbatch_2gpu.py
"""
import functools
import json
import os
import re
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VERL_ROOT = os.path.join(PROJECT_ROOT, "pyvision-rl", "verl_agents")
for p in (PROJECT_ROOT, VERL_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import transformers  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: E402

from verl.models.transformers import qwen3_vl as verl_qwen3_vl  # noqa: E402
from verl.models.transformers.qwen2_vl import get_rope_index  # noqa: E402
from flash_attn.bert_padding import index_first_axis, unpad_input  # noqa: E402
from verl.utils.torch_functional import (  # noqa: E402
    logprobs_from_hidden,
    postprocess_data,
)

MODEL = "/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2"
DATA = "/root/code/datasets/vstar_bench/rl/train.json"
MAX_PROMPT = 5120
RESP_LEN = int(os.environ.get("RESP_LEN", "3224"))  # 最坏:prompt 1900 + 响应 3224 = nnz 5124(4 卡日志上限)
PAD = 13312  # max_prompt + max_response


def build_messages_pyvision(example):
    messages = list(example["prompt"])
    for message in messages:
        content = message["content"]
        cl = []
        for segment in re.split("(<image>|<video>)", content):
            if segment == "<image>":
                cl.append({"type": "text", "text": "<image_clue_0>"})
                cl.append({"type": "image"})
                cl.append({"type": "text", "text": "</image_clue_0>"})
            else:
                cl.append({"type": "text", "text": segment})
        message["content"] = cl
    return messages


def build_micro_batch(processor, tok, example, response_len):
    """复刻 rl_dataset._process_single_row -> 微批拼装(prompt + response)。"""
    messages = build_messages_pyvision(example)
    raw_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    mi = processor(text=[raw_prompt], images=[example["image"]], return_tensors="pt")
    input_ids, attn = mi.pop("input_ids"), mi.pop("attention_mask")
    grid = mi.get("image_grid_thw")
    grid_patch = int(grid[0, 1] * grid[0, 2]) if grid is not None else 0
    input_ids, attn = postprocess_data(
        input_ids, attn, max_length=MAX_PROMPT,
        pad_token_id=tok.pad_token_id, left_pad=True, truncation="right",
    )
    vp = get_rope_index(
        processor, input_ids=input_ids[0], image_grid_thw=mi.get("image_grid_thw"),
        video_grid_thw=mi.get("video_grid_thw"),
        second_per_grid_ts=mi.get("second_per_grid_ts"), attention_mask=attn[0],
    )
    valid_mask = attn[0].bool()
    text_pos = torch.ones((1, input_ids.shape[1]), dtype=torch.long)
    text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item())
    pos = torch.cat((text_pos, vp), dim=0)  # (4, prompt_seq)

    # 拼 response(模拟 rollout 输出;响应是纯文本,position 续 prompt)
    prompt_nnz = int(valid_mask.sum().item())
    resp_ids = torch.randint(0, 10000, (1, response_len), dtype=torch.long)
    resp_attn = torch.ones((1, response_len), dtype=attn.dtype)
    resp_pos = torch.arange(response_len).expand(4, response_len) + prompt_nnz  # 续接
    input_ids = torch.cat([input_ids, resp_ids], dim=1)
    attn = torch.cat([attn, resp_attn], dim=1)
    pos = torch.cat([pos, resp_pos], dim=1)

    # pad 到 (1, PAD)(生产 dataloader 行为)
    pad_n = PAD - input_ids.shape[1]
    if pad_n > 0:
        input_ids = torch.cat([input_ids, torch.full((1, pad_n), tok.pad_token_id, dtype=torch.long)], dim=1)
        attn = torch.cat([attn, torch.zeros((1, pad_n), dtype=attn.dtype)], dim=1)
        pos = torch.cat([pos, torch.zeros(4, pad_n, dtype=pos.dtype)], dim=1)
    else:
        input_ids, attn, pos = input_ids[:, :PAD], attn[:, :PAD], pos[:, :, :PAD]
    mm = {k: v for k, v in mi.items() if v is not None}
    return input_ids, attn, pos, mm, grid_patch, prompt_nnz


def mem_report(tag):
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = (torch.cuda.mem_get_info()[0]) / 1e9
    print(f"[{tag}] alloc={allocated:.2f}G free={free:.2f}G (total {total:.0f}G)", flush=True)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    dev = f"cuda:{local_rank}"

    if rank == 0:
        print(f"[VERIFY] world={dist.get_world_size()} model={MODEL}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tok = proc.tokenizer
    if rank == 0:
        print(f"[VERIFY] image_processor size={proc.image_processor.size}", flush=True)

    # 与生产一致:monkey_patch verl 的 forward_with_normal_backend
    Qwen3VLForConditionalGeneration.forward = verl_qwen3_vl.forward_with_normal_backend

    samples = json.load(open(DATA))
    # 取「响应最长」场景:直接构造最坏微批
    input_ids, attn, pos, mm, grid_patch, prompt_nnz = build_micro_batch(proc, tok, samples[0], RESP_LEN)
    if rank == 0:
        print(f"[VERIFY] input_ids {tuple(input_ids.shape)} grid_patch={grid_patch} "
              f"prompt_nnz={prompt_nnz} resp={RESP_LEN}", flush=True)

    # 模型加载(与生产一致:bf16 + param_offload + FSDP1 use_orig_params=False)
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.api import CPUOffload
    from torch.distributed.fsdp.wrap import _or_policy

    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.requires_grad_(True)  # 与生产一致:全量参数参与训练(use_orig_params=False 要求 requires_grad 统一)
    if os.environ.get("CKPT", "1") == "1":
        # 与生产一致:ppo_trainer.yaml 默认 enable_gradient_checkpointing=True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # 与生产 get_fsdp_wrap_policy 相同:按 _no_split_modules 解析类名
    from verl.utils.fsdp_utils import get_module_class_from_name  # noqa: E402

    transformer_cls_to_wrap = set()
    for layer_name in ("Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"):
        transformer_cls_to_wrap.add(get_module_class_from_name(model, layer_name))
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )
    auto_wrap_policy = functools.partial(_or_policy, policies=[wrap_policy])

    model = FSDP(
        model,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                       buffer_dtype=torch.bfloat16),
        cpu_offload=CPUOffload(offload_params=os.environ.get("OFFLOAD", "1") == "1"),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        forward_prefetch=False,
    )
    if rank == 0:
        n_fsdp = 0
        n_layers = 0
        total_params = 0.0

        def walk(m):
            nonlocal n_fsdp, n_layers, total_params
            if isinstance(m, FSDP):
                n_fsdp += 1
                total_params += sum(p.numel() for p in m.parameters()) / 1e9
            if m.__class__.__name__ == "Qwen3VLTextDecoderLayer":
                n_layers += 1
            for c in m.children():
                walk(c)

        walk(model)
        print(f"[VERIFY] FSDP units={n_fsdp} decoder_layers={n_layers} "
              f"total_params={total_params:.2f}G", flush=True)
        lm = model.model.language_model
        print(f"[VERIFY] lang_ckpt={lm.gradient_checkpointing}", flush=True)
    if os.environ.get("OFFLOAD", "1") == "0":
        model.to(dev)  # 非 offload 模式需要手动移 GPU
    torch.cuda.empty_cache()
    mem_report("model loaded")

    # 生产 lm_head chunked patch(同 dp_actor.__init__)
    lm_head_module = model.lm_head
    lm_head_module._vstar_orig_forward = lm_head_module.forward

    def _vstar_chunked_forward(x):
        if getattr(lm_head_module, "_vstar_labels", None) is None:
            return lm_head_module._vstar_orig_forward(x)
        if x.dim() == 3:
            x = x.squeeze(0)
        if getattr(lm_head_module, "_vstar_calc_entropy", False):
            log_probs, entropy = logprobs_from_hidden(
                hidden=x, lm_head_weight=lm_head_module.weight,
                labels=lm_head_module._vstar_labels,
                temperature=lm_head_module._vstar_temp, compute_entropy=True,
            )
            lm_head_module._vstar_entropy = entropy
            return log_probs
        return logprobs_from_hidden(
            hidden=x, lm_head_weight=lm_head_module.weight,
            labels=lm_head_module._vstar_labels, temperature=lm_head_module._vstar_temp,
        )

    lm_head_module.forward = _vstar_chunked_forward

    # ---- 生产微批 forward(照抄 dp_actor._forward_micro_batch)----
    batch_size, seqlen = input_ids.shape
    input_ids = input_ids.to(dev)
    attn = attn.to(dev)
    pos = pos.to(dev)
    pos = pos.unsqueeze(0)  # (4, seq) -> (1, 4, seq), 生产 3D 分支
    if os.environ.get("NOIMG", "0") == "1":
        mm = {}  # 对照实验:不传视觉
    else:
        mm = {k: v.to(dev) for k, v in mm.items()}

    torch.cuda.empty_cache()
    mem_report("pre-forward")
    if rank == 0 and os.environ.get("HOOKS", "0") == "1":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

        hook_ct = [0]

        def _h(module, args, kwargs):
            hook_ct[0] += 1
            if hook_ct[0] in (1, 5, 20, 36):
                inp = args[0]
                print(f"[HOOK] layer {hook_ct[0]} input {tuple(inp.shape)} "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}G "
                      f"alloc={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

        handles = []
        for i, m in enumerate(model.model.language_model.layers):
            inner = m._fsdp_wrapped_module if isinstance(m, FSDP) else m
            handles.append(inner.register_forward_hook(_h))
            if isinstance(m, FSDP):
                handles.append(m.register_forward_hook(
                    lambda mod, a, kw, i=i: print(
                        f"[FSDP-post] layer {i+1} alloc={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
                    if i + 1 in (1, 5, 20) else None))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attn)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, nnz)
        # 生产: rearrange(position_ids, 'c b s ... -> (b s) c ...') == permute+reshape
        position_ids_rmpad = (
            index_first_axis(pos.permute(1, 2, 0).reshape(-1, 4), indices)
            .transpose(0, 1)
            .unsqueeze(1)
        )  # (4, 1, nnz)
        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
        nnz = input_ids_rmpad_rolled.numel()
        lm_head_module._vstar_labels = input_ids_rmpad_rolled
        lm_head_module._vstar_temp = 0.7
        lm_head_module._vstar_calc_entropy = False

        output = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            **mm,
            use_cache=False,
            return_hidden_states=True,
        )
        log_probs = lm_head_module._vstar_log_probs
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[VERIFY] post-fwd nnz={nnz} log_probs {tuple(log_probs.shape)} "
                  f"nan={bool(torch.isnan(log_probs).any().item())} "
                  f"max={log_probs.abs().max().item():.3f} "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}G "
                  f"lang_ckpt={model.model.language_model.gradient_checkpointing}", flush=True)
        mem_report("post-fwd")

        loss = -log_probs.sum()
        loss.backward()
        torch.cuda.synchronize()
        mem_report("post-bwd")
        if rank == 0:
            g = model.lm_head.weight.grad
            print(f"[VERIFY] lm_head grad (local shard) absmax={g.abs().max().item():.3e} "
                  f"nnz_grad={int((g != 0).sum().item())}", flush=True)
        assert g.abs().max().item() > 0, "lm_head 梯度为零!"
        assert not torch.isnan(loss).item(), "loss nan"

    if rank == 0:
        print("[VERIFY] 2GPU MICROBATCH ALL OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
