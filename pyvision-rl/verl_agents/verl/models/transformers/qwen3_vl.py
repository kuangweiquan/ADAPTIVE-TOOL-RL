# Qwen3-VL backend forward for vendored verl (VStar RL remote).
#
# verl's monkey_patch.py references verl.models.transformers.qwen3_vl
# (forward_with_normal_backend + qwen3_vl_base_forward), but the vendored
# checkout does not ship the module. Qwen3-VL-8B is the DeepStack variant
# (visual encoder returns image_embeds + per-layer deepstack_visual_embeds
# fused inside Qwen3VLTextModel), so we do NOT replicate the qwen2_vl
# _get_input_embeds pattern — instead we capture the pristine transformers
# Qwen3VLModel.forward and forward through it, keeping native deepstack /
# mrope / rmpad semantics identical to upstream transformers 4.57.

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Capture the pristine forward before monkey_patch replaces it.
_original_qwen3_vl_model_forward = Qwen3VLModel.forward


def qwen3_vl_base_forward(
    self: "Qwen3VLModel",
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position=None,
    **kwargs,
):
    """Identical to upstream transformers Qwen3VLModel.forward."""
    return _original_qwen3_vl_model_forward(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        cache_position=cache_position,
        **kwargs,
    )


def qwen3_vl_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Call the (monkey-patched) Qwen3VLModel forward with verl's call shape."""
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )


def forward_with_normal_backend(
    self: Qwen3VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    return_hidden_states: bool = False,
    **kwargs,
) -> "Qwen3VLCausalLMOutputWithPast":
    outputs = qwen3_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]  # last_hidden_state
    if return_hidden_states:
        # 2x24G: compute the log-softmax chunked over the vocab instead of
        # materializing the full (nnz, vocab) logits + dlogits during
        # backward. This runs INSIDE the FSDP root forward, where the flat
        # is unsharded and lm_head.weight is the full view — the only spot
        # where a sliced lm_head weight is autograd-compatible with FSDP
        # (outside the forward the flat is resharded: slice data
        # misaligns / shard-sized grads clash with the full-sized flat,
        # reproduced in verify_fsdp_chunk_gpu). The (patched) chunked
        # forward stashes the log_probs on the module; dp_actor picks them
        # up after the model returns.
        log_probs = self.lm_head(hidden_states)
        self.lm_head._vstar_log_probs = log_probs
        return Qwen3VLCausalLMOutputWithPast(logits=None, hidden_states=hidden_states)
    logits = self.lm_head(hidden_states)
    return Qwen3VLCausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(*args, **kwargs):
    raise NotImplementedError(
        "Qwen3-VL torch fused backend (FusedLinearForPPO) is not available in the "
        "vendored verl checkout; verl.utils.experimental.torch_functional is missing. "
        "Run with use_fused_kernels=False (default)."
    )


def forward_with_triton_backend(*args, **kwargs):
    raise NotImplementedError(
        "Qwen3-VL triton fused backend (linear_cross_entropy) is not available in the "
        "vendored verl checkout; verl.utils.kernel.linear_cross_entropy is missing. "
        "Run with use_fused_kernels=False (default)."
    )
