#!/bin/bash
# Arm A (plain acc GRPO, released-code fidelity) downscaled 8x80G -> 4x3090 24G.
# Mirror of verltool/examples/train/adatooler_v/train_qwen25vl.sh with plan-10 changes:
#   n_gpus 8->4, tensor_parallel 2 (released value, see note below), gpu_mem_util 0.8->0.5->0.45,
#   batch 64->32, n 8->4,
#   prompt/response/obs 16384->8192, save_freq 50->10, total_steps 150, logger console-only,
#   local model/data paths, val_batch 512->100.
#   + model.override_config.attn_implementation=sdpa: verl defaults to flash_attention_2 but this
#   env ships a deliberate flash_attn stub (ver 0.0.0, see 05_verl_env_precheck) -> HF >=2.1.0 gate
#   raises ImportError at AutoConfig time; sdpa keeps the qwen3_vl torch-backend path (no FA2 kernels).
#   tp=2 + gpu_mem_util 0.45 (08-16 fix #3, user-confirmed, probe gmem45): the async rollout
#   engine stays resident during the training phase (vLLM 0.11 sleep mode is a no-op on this
#   setup), so engine+training forward must fit together: tp=2 shards weights to ~8G/card,
#   0.45 budget=11.16G (probe-measured, vs 12.30G at 0.5) leaves ~1.2G KV cache (29,280 tok,
#   vs 73,408 at 0.5); #23 OOM was 0.1-0.2G over -> this 1.14G saving lands peak ~21.4G,
#   inside the <=21.5G target. tp=1 would need 16.1G weights alone -> OOM (0.85 tried).
#   use_remove_padding True->False (08-15 fix): remove_padding=True makes verl install its
#   fused FA2 attention patch (qwen2_vl_attn_forward), which needs real flash_attn kernels that
#   this env deliberately lacks (flash_attn 0.0.0 stub, see 05_verl_env_precheck). False keeps the
#   HF sdpa attention path (perf-only knob; masked padding tokens, same softmax for valid tokens).
#   log_prob_micro_batch_size_per_gpu 8->2 (08-16 fix, user-approved): compute_log_prob OOM at
#   step-0 old_log_prob — with remove_padding=False each seq is fully padded to 16k, 8 seqs =
#   131k tokens in one forward -> MLP intermediate 4.62 GiB single alloc on top of 21.28G steady
#   (engine 12.3G + FSDP fwd + activations) -> OOM. 2 seqs = 32k tokens -> intermediate ~1.2G,
#   peak ~18.8G. Per-token log probs are independent of micro-batch size (padding masked
#   identically in every split), zero reward impact.
#   log_prob_micro_batch_size_per_gpu 8->1 + calculate_entropy=False (08-16 fix #2, user-approved):
#   padded path computes full micro-batch logits [B, 16384, 151936] then logprobs_from_logits_v2
#   loops rows = padded seqs, each F.log_softmax row = 4.64 GiB transient -> OOM at B>=2;
#   B=1 fits logits+softmax (peak ~20.7G) but entropy_from_logits temps (softmax + pd*logits,
#   9.3G) still blow up -> entropy skipped via vendored one-liner in fsdp_workers.py:978
#   (entropy_coeff=0, metric-only, zero reward impact; backup fsdp_workers.py.bak_entropy).
#   NOTE: use_dynamic_bsz=True (packed) was rejected by verl's seqlen_balancing assert
#   (max_token_len >= max_seq_len; 8192 < 16384) and 16k chunks + entropy still OOM.
#   ppo_max_token_len_per_gpu is ignored by the padded path (kept at prompt+response).
set -x
export PATH=/root/autodl-tmp/envs/verl-tool/bin:$PATH

# --- pre-launch cleanup of orphaned processes from previously failed runs ---
# When a verl_tool run dies abnormally, its vLLM engine cores (multiprocessing spawn
# children) and per-replica tool servers survive and hold GPU memory; SIGKILLing them
# on this platform (SeetaCloud, drv 570) can leave permanent driver-side zombie
# allocations, so SIGTERM first and only SIGKILL survivors. Only orphans (PPID=1) are
# touched; never matches the running script itself (its cmdline lacks these patterns).
term_orphans() {
    local pids
    pids=$(pgrep -f "$1" 2>/dev/null)
    [ -n "$pids" ] || return 0
    kill $pids 2>/dev/null
    sleep 25
    for p in $pids; do
        if kill -0 $p 2>/dev/null; then kill -9 $p 2>/dev/null; fi
    done
}
term_orphans "verl_tool.servers.tool_server"
term_orphans "verl_tool.servers.serve"
# batch SIGTERM all orphaned spawn workers, one wait, then sweep survivors
# (a per-orphan sleep in the loop would serialize into many minutes of startup delay)
ORPHAN_PIDS=""
for p in $(pgrep -f "from multiprocessing.spawn import spawn_main"); do
    if [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ]; then
        ORPHAN_PIDS="$ORPHAN_PIDS $p"
    fi
done
if [ -n "$ORPHAN_PIDS" ]; then
    kill $ORPHAN_PIDS 2>/dev/null
    sleep 20
    for p in $ORPHAN_PIDS; do
        if kill -0 $p 2>/dev/null; then kill -9 $p 2>/dev/null; fi
    done
fi
sleep 3

# --- verify GPUs are actually free before launching (zombie driver allocations from
# SIGKILLed vLLM cores report as used and are not reclaimable on this platform;
# launching into them wastes a full 15-min startup before vLLM's init check fails) ---
FREE_OK=1
for line in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader); do
    idx=$(echo $line | cut -d, -f1)
    used=$(echo $line | cut -d, -f2 | tr -dc '0-9')
    if [ "$used" -gt 500 ]; then
        echo "WARN: GPU $idx still shows ${used} MiB used -> zombie allocation; aborting launch (reboot needed)" >&2
        FREE_OK=0
    fi
done
[ "$FREE_OK" = "1" ] || exit 1

train_data=/root/autodl-tmp/datasets/adatooler_v_subset/train.parquet
val_data=/root/autodl-tmp/datasets/adatooler_v_subset/val.parquet
model_name=/root/autodl-tmp/models/AdaTooler-V-SFT-model

rl_alg=grpo
n_gpus_per_node=4
n_nodes=1
n=4
batch_size=32
ppo_mini_batch_size=32
max_prompt_length=8192
max_response_length=8192
max_obs_length=8192
max_action_length=4096
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp"
action_stop_tokens='</tool_call>'
max_turns=2
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=adatooler_v
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=2
gpu_memory_utilization=0.45
do_offload=True
use_dynamic_bsz=False
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=10000
run_name_postfix="arm_a_4x3090"
run_name="eval_${reward_manager}-${strategy}-agent-${run_name_postfix}-${rl_alg}-n${n}-b${batch_size}-t${temperature}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
rollout_mode='async'

action_stop_tokens_file="/root/autodl-tmp/arm_a_action_stop_tokens"
echo -n "$action_stop_tokens" > $action_stop_tokens_file

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "adatooler_v" --workers_per_tool 16 &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

PYTHONUNBUFFERED=1 python -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files="[$train_data]" \
    data.val_files="[$val_data]" \
    data.dataloader_num_workers=2 \
    data.train_batch_size=$batch_size \
    data.val_batch_size=100 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_local_dir=/root/autodl-tmp/rl_ckpt/arm_a \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=150 \
    trainer.val_only=False

kill -9 $server_pid
