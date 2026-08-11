#!/bin/bash
# VStar RL 冒烟训练：train_batch=2, n=2, max_turns=2, total_epochs=1（05 手稿 B4，不存盘）
# 4x3090 版（2026-08-11）：TP=4 vocab 151936/4=37984 整除；FSDP shard 4.3G/卡，backward 峰值 ~12G
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate atr
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/root/code:$PYTHONPATH
export NCCL_P2P_DISABLE=1   # 3090 无 NVLink(SYS topo)，隔离下 NCCL P2P 探测报 217 且不 fallback
export NCCL_SHM_DISABLE=1   # 双禁后走 Socket transport（实测隔离下 all_reduce 正常）
# 注意: 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments —— vLLM 0.11 CuMemAllocator 不兼容
EXPERIMENT_NAME="vstar_smoke"
mkdir -p /root/code/logs

MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2
# 冒烟用最短样本（train 3415-3796 tokens），跑通四项验证 + 实测响应长度（定全量 max_response）
TRAIN_DATA=/root/code/datasets/vstar_bench/rl/train_smoke_short.json
VAL_DATA=/root/code/datasets/vstar_bench/rl/val_smoke_short.json

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=[${TRAIN_DATA}] \
    data.val_files=[${VAL_DATA}] \
    data.train_batch_size=4 \
    data.max_prompt_length=5120 \
    data.max_response_length=2048 \
    data.truncation=right \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.max_turns=2 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.max_vllm_images=2 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=1000 \
    trainer.project_name=vstar_atr \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=/root/autodl-tmp/rl_ckpt_smoke/vstar_atr/vstar_smoke \
    trainer.total_epochs=1 2>&1 | tee /root/code/logs/${EXPERIMENT_NAME}.log
