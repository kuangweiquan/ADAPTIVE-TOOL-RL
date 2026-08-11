# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.nn  # autograd-aware all_gather for chunked lm_head

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty, compute_policy_loss_with_filter_mask
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_hidden, logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
# from verl.trainer.ppo.filter_fn_utils import max_interaction_budget_filter_fn

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    # 2x24G: skip lm_head when entropy is not needed; the
                    # chunked logprobs_from_hidden then keeps logits/dlogits
                    # memory bounded to a single vocab chunk
                    return_hidden_states=not calculate_entropy,
                )  # prevent model thinks we are generating

                if calculate_entropy:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=False
                    )
                    # compute entropy
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                else:
                    hidden_f = output.hidden_states.squeeze(0)
                    # FSDP flattens lm_head into a 1D shard; calling the
                    # module (instead of reading .weight) runs inside the
                    # FSDP pre/post hooks, where the weight is the full
                    # unsharded flat view. Patch the instance forward to do
                    # the chunked log-softmax there so gradients are
                    # reduced by FSDP normally.
                    lm_head_module = self.actor_module.lm_head
                    if not getattr(lm_head_module, "_vstar_chunked_fwd", False):
                        def _vstar_chunked_forward(x):
                            # FSDP shards lm_head across ranks; a rank may
                            # hold zero rows of the vocab (flat-param
                            # boundary). The full weight must be identical
                            # on every rank for the DP-parallel log_probs.
                            # FSDP1 flattens lm_head with embed_tokens etc.
                            # into one outer flat; the rank-local view is a
                            # 1D element slice whose boundary can split a
                            # row. Assembling the whole weight at once
                            # (all_gather of the full flat, ~2.8G on 4x3090)
                            # OOMs in the train segment, so we assemble it
                            # per-vocab-chunk: each rank contributes the rows
                            # it owns (zero-filled), all_gather moves only
                            # O(chunk x hidden) per call, and each row is
                            # summed to full value (split rows complete each
                            # other since only the owner rank is nonzero).
                            flat = self.actor_module._flat_param
                            target = None
                            for i, info in enumerate(flat._param_infos):
                                if info.module is lm_head_module:
                                    target = i
                                    break
                            if target is None:
                                raise RuntimeError("[CHUNK] lm_head not found in FSDP flat params")
                            si = flat._shard_param_infos[target]
                            hidden_dim = x.shape[-1]
                            world = torch.distributed.get_world_size()
                            if world == 1:
                                w = lm_head_module.weight.view(-1, hidden_dim)
                                return logprobs_from_hidden(
                                    hidden=x,
                                    lm_head_weight=w,
                                    labels=lm_head_module._vstar_labels,
                                    temperature=lm_head_module._vstar_temp,
                                )
                            # NOTE: keep the autograd link (no `.data`): the
                            # chunk gather must credit gradients back to the
                            # FSDP flat shard, or lm_head never updates.
                            local = flat._local_shard.to(x.device)
                            in_shard = int(si.in_shard)
                            off = int(si.offset_in_shard or 0)
                            numel = int(si.numel_in_shard or 0)
                            meta = torch.tensor([in_shard, off, numel], dtype=torch.long, device=x.device)
                            metas = [torch.empty(3, dtype=torch.long, device=x.device) for _ in range(world)]
                            torch.distributed.all_gather(metas, meta)  # tiny, non-autograd
                            total_numel = sum(int(m[2]) for m in metas if int(m[0]))
                            vocab_size = total_numel // hidden_dim
                            print(
                                f"[CHUNK] pieces={[(int(m[0]), int(m[1]), int(m[2])) for m in metas]} "
                                f"vocab={vocab_size}",
                                flush=True,
                            )
                            # Row-tensor of the local element slice (edge rows
                            # zero-filled) plus their global row ids.
                            if in_shard and numel > 0:
                                g0 = off // hidden_dim
                                g1 = (off + numel - 1) // hidden_dim + 1
                                n_rows = g1 - g0
                                row_tensor = torch.zeros(n_rows, hidden_dim, dtype=local.dtype, device=x.device)
                                elem_s = max(off, g0 * hidden_dim)
                                elem_e = min(off + numel, g1 * hidden_dim)
                                if elem_s < elem_e:
                                    dst_s = elem_s - g0 * hidden_dim
                                    row_tensor.view(-1)[dst_s : dst_s + (elem_e - elem_s)] = local[
                                        elem_s - off : elem_e - off
                                    ]
                                g_idx = torch.arange(g0, g1, device=x.device)
                            else:
                                row_tensor = None
                                g_idx = None

                            def gather_chunk(start, end):
                                contrib = torch.zeros(
                                    end - start, hidden_dim, dtype=local.dtype, device=x.device
                                )
                                if row_tensor is not None:
                                    mask = (g_idx >= start) & (g_idx < end)
                                    if mask.any():
                                        contrib[g_idx[mask] - start] = row_tensor[mask]
                                gathered = torch.distributed.nn.all_gather(contrib)  # autograd-aware
                                return torch.stack(gathered).sum(dim=0)  # each row owned by 1 rank

                            return logprobs_from_hidden(
                                hidden=x,
                                lm_head_weight=None,
                                labels=lm_head_module._vstar_labels,
                                temperature=lm_head_module._vstar_temp,
                                weight_gather_fn=gather_chunk,
                                vocab_size=vocab_size,
                            )

                        lm_head_module.forward = _vstar_chunked_forward
                        lm_head_module._vstar_chunked_fwd = True
                    lm_head_module._vstar_labels = input_ids_rmpad_rolled
                    lm_head_module._vstar_temp = temperature
                    log_probs = lm_head_module(hidden_f)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        indices = None  # set by the dynamic-bsz branch; multi-modal branch chunks by micro_batch_size

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            response_mask = micro_batch["attention_mask"][:, -micro_batch["responses"].size(-1) :]
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        # multi-modal branch splits by micro_batch_size and produces no
        # `indices`; only rearrange when the dynamic-bsz branch ran
        if use_dynamic_bsz and indices is not None:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "action_mask", "tool_cnt"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys, strict=False).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys, strict=False).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys, strict=False).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    action_or_attn_mask = data['action_mask'] if 'action_mask' in data.keys() else data['attention_mask']

                    response_mask = action_or_attn_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    overbudget_masking = self.config.overbudget_masking
                    interaction_budget = self.config.interaction_budget

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    free_mem, _ = torch.cuda.mem_get_info()
                    print(
                        f"[MB diag] post-fwd: alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G free={free_mem/1e9:.2f}G",
                        flush=True,
                    )

########################################################################################################################
                    # print(f"shape of response mask: {response_mask.shape}")
                    # print(f"len of data: {len(data)}")
                    # print(f"keys of data: {data.keys()}")
                    tool_cnt = data['tool_cnt'].detach().cpu().tolist()[0][0]
                    tool_cnt = int(tool_cnt)
                    # print(f"tool_cnt: {tool_cnt}")
                    # print(f"shape of response mask: {response_mask.shape}")
                    # print(f"len of data: {len(data)}")
                    # print(f"keys of data: {data.keys()}")
                    # if tool_cnt == interaction_budget:
                    #     filter_mask = torch.zeros_like(response_mask)
                    #     # print("tool_cnt is ")
                    # else:
                    #     filter_mask = response_mask

                    # if overbudget_masking:

                    #     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_with_filter_mask(
                    #         old_log_prob=old_log_prob,
                    #         log_prob=log_prob,
                    #         advantages=advantages,
                    #         response_mask=response_mask,
                    #         filter_mask=filter_mask,
                    #         cliprange=clip_ratio,
                    #         cliprange_low=clip_ratio_low,
                    #         cliprange_high=clip_ratio_high,
                    #         clip_ratio_c=clip_ratio_c,
                    #         loss_agg_mode=loss_agg_mode,
                    #     )

########################################################################################################################

                    # else:
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    free_mem, _ = torch.cuda.mem_get_info()
                    print(
                        f"[MB diag] pre-bwd: alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G free={free_mem/1e9:.2f}G",
                        flush=True,
                    )
                    # 2x24G: free reserved-but-unallocated physical pages
                    # before backward; every MB counts at this margin
                    torch.cuda.empty_cache()
                    loss.backward()
                    free_mem, _ = torch.cuda.mem_get_info()
                    print(
                        f"[MB diag] post-bwd: alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G free={free_mem/1e9:.2f}G",
                        flush=True,
                    )

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
