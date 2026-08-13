#!/bin/bash
# VStar RL 全量 GRPO（05 手稿 B7 配置，4x3090 修正版，2026-08-11）
# 相对手稿的修改（全部由 2x24G 冒烟阶段的实测/修复得出）：
#   1. max_prompt_length 2048 -> 5120（2048 是 doc2len bug 时代的旧值，真实 prompt 3415-9061）
#   2. max_model_len 12288 -> 14336（5120+8192=13312 + 1K 系统余量）
#   3. 数据改用 train.json/val.json（进 git 无需搬运 parquet；parquet 格式相同）
#   4. n_gpus_per_node 2 -> 4（TP=4，vocab 151936/4=37984 整除）
#   5. checkpoint 改存 /root/autodl-tmp/rl_ckpt（系统盘容量有限）
#   6. +fsdp_config.model_dtype=bf16（冒烟验证必需）
#   7. +use_dynamic_bsz / ppo_max_token_len_per_gpu=8192（micro batch 按 token 切，激活余量稳）
# 显存账（4x3090，实测修正 2026-08-11）：
#   (1) vLLM 启动时 FSDP 已占 ~4.3G/卡（空闲 18.22G）→ gpu_memory_utilization 必须 < 0.77，0.85 启动检查即失败。
# 显存账（4x3090，实测修正 2026-08-11，三版演进）：
#   (1) vLLM 启动时 FSDP 已占 ~4.3G/卡 → gpu_memory_utilization 必须 < 0.77（0.85 启动检查失败）。
#   (2) vLLM KV pool 训练段不释放 → 0.6 下 vLLM 实占 17.2G，训练段只剩 6.4G < 8.6G 刚需 → 必须 sleep。
#       vLLM 0.11 sleep(level=1) 由 fsdp_vllm.py 的 sm_guard __exit__ 无条件调用（不依赖配置）：
#       权重 offload CPU + 丢 KV，实测释放 14.93G（CPU 需 >16G，本机 755G ✓）。
#   (3) 训练段峰值（micro batch T tokens）：param 4.3 + grad 4.3 + 激活(36 层无 ckpt, ~T×1.4G/4096)
#       + logits(T×151936×2B) + entropy 峰值(~4.5×logits)。实测 4096 tokens 峰值 ≈ 24G > 23.57G OOM。
#       → ppo_max_token_len_per_gpu=3072：峰值 ~21G < 23.57G ✓（余量 2.5G；再低训练更慢）
#   (4) 2026-08-11 三轮修复后的预算（单样本 8120 tok 独占 micro，peaks ≈ 22.6G）：
#       entropy 分块版（torch_functional.py，chunk=16384，峰值 5.55G→~200M）
#       + lm_head chunk 级权重收集（dp_actor.py gather_chunk + weight_gather_fn：
#         all_gather 2.63G→~134M/chunk；并修 .data 切断的权重梯度回传）。
#   (5) 2026-08-12 优化器账本（此前未计入！）：AdamW 状态 fp32 2x 分片 = 16.1G/卡，
#       step 峰值 29.6G > 24G → 4x24G 全参训练此前任何 step 都不可能过。
#       换 Adafactor（yaml actor.optim.name=adafactor，状态 1x = 8.0G/卡）：
#       step 峰值 ≈ 参数 4.4 + 梯度 8.0 + 状态 8.0 + 临时 ~2 = 22.4G < 23.57G ✓（余量 ~1.2G）。
#       2 卡 Mini 同构验证：AdamW round1 step OOM(24.4G)，Adafactor 两轮全通过(峰值 20.8G)。
#   rollout 段：权重 4.3 + KV（max_num_batched_tokens 8192）实测 17.2G < 0.6×23.57=14.1G 预算上限。
# 注意：total_epochs=30，实际步数 = 滤超长后样本数/8 × 30（171 条全量≈640 步；滤掉>5120 后约 120-140 条≈450-520 步）
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate atr
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/root/code:$PYTHONPATH
export NCCL_P2P_DISABLE=1   # 3090 无 NVLink(SYS topo)
export NCCL_SHM_DISABLE=1   # 双禁后走 Socket transport
# 注意: 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments —— vLLM 0.11 CuMemAllocator 不兼容
# 注意: 注释行不能插在 python3 命令的 `\` 续行链中间（会断链），只能放这里或命令之前
# 冒烟实测单次 ckpt ~29G，25 次 ≈725G 会写爆数据盘 → max_actor_ckpt_to_keep=2 只留最近 2 个
PROJECT_NAME="vstar_atr"
EXPERIMENT_NAME="qwen3vl_8b_sftv2_grpo_4gpu"
mkdir -p /root/code/logs

MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2
TRAIN_DATA=/root/code/datasets/vstar_bench/rl/train.json
VAL_DATA=/root/code/datasets/vstar_bench/rl/val.json

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=[${TRAIN_DATA}] \
    data.val_files=[${VAL_DATA}] \
    data.train_batch_size=8 \
    data.max_prompt_length=5120 \
    data.max_response_length=8192 \
    data.truncation=right \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=14336 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.max_turns=8 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.max_vllm_images=2 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    +trainer.num_examine=50 \
    +reward_model.reward_kwargs.atr_config_dict.verbose=true \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=1000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=/root/autodl-tmp/rl_ckpt/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=30 2>&1 | tee /root/code/logs/${EXPERIMENT_NAME}.log
