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

import inspect
import logging
import os

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage

from .base import BaseShardingManager
from .patch import patched_ds_v3_load_weights, patched_qwen_moe_load_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        print(f"[DIAG] enter sm: allocated={torch.cuda.memory_allocated()/1e9:.2f}G reserved={torch.cuda.memory_reserved()/1e9:.2f}G", flush=True)

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        params = self.module.state_dict()
        # 2x24G: state_dict ran while the vLLM engine was still asleep (GPU
        # free). Stash the copy on CPU now so wake_up's memory-pool remap
        # (weights + KV ~9.5G) doesn't race with the 8.8G shard copy;
        # update_params pulls each param back to GPU layer by layer.
        params = {k: v.to("cpu") for k, v in params.items()}
        # 2x24G: state_dict() unshards ALL params (17.5G peak); those cached
        # blocks stay in the torch allocator and would starve wake_up below
        torch.cuda.empty_cache()
        print(f"[DIAG] after state_dict: allocated={torch.cuda.memory_allocated()/1e9:.2f}G reserved={torch.cuda.memory_reserved()/1e9:.2f}G", flush=True)
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(
                f"[DIAG] before wake: free={free_mem/1e9:.2f}G total={total_mem/1e9:.2f}G",
                flush=True,
            )
            try:
                alloc = self.inference_engine.llm_engine.model_executor.driver_worker.worker.device_allocator
                segs = [(t.tag, h[1] / 1e9) for t, h in alloc.pointer_to_data.items()]
                print(f"[DIAG] cumem segments: {segs}", flush=True)
            except Exception as e:
                print(f"[DIAG] no allocator access: {e}", flush=True)
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            torch.cuda.empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            free_before, _ = torch.cuda.mem_get_info()
            self.inference_engine.sleep(level=1)
            free_after, _ = torch.cuda.mem_get_info()
            print(
                f"[DIAG] __exit__ sleep: free {free_before/1e9:.2f}G -> {free_after/1e9:.2f}G "
                f"(freed {(free_after-free_before)/1e9:.2f}G)",
                flush=True,
            )

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        world_size = torch.distributed.get_world_size()
        if model.config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
            loaded_params = patched_ds_v3_load_weights(
                model,
                (
                    (name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param)
                    for name, param in updated_params.items()
                ),
            )
        elif model.config.architectures[0] in ["Qwen2MoeForCausalLM"]:
            loaded_params = patched_qwen_moe_load_weights(
                model,
                (
                    (name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param)
                    for name, param in updated_params.items()
                ),
            )
        else:
            # 2x24G: state_dict was gathered from the CPU-shard (params offloaded);
            # move the local shard back to GPU before full_tensor()/load_weights
            loaded_params = model.load_weights(
                (
                    (name, param.to("cuda").full_tensor() if world_size != 1 else param)
                    for name, param in updated_params.items()
                )
            )
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")
