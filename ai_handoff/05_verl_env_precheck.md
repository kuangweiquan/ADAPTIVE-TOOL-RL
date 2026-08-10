# 05 verl_agents 环境预检报告（远端，2026-08-10）

状态：**环境就绪 + 全链路验证通过 + 显存基线实测**。本地端 1–4 交付后可直接进入冒烟训练。

## 1. 环境安装结果

| 项 | 状态 |
|---|---|
| 机器 | 2× RTX 3090 24GB（无 NVLink，P2P 不可用，TP 通信走 PCIe） |
| 环境 | conda `atr`：torch 2.8.0+cu128 / vllm 0.11.0 / transformers 4.57.2 / ray 2.56.1 |
| verl | vendored `pyvision-rl/verl_agents`（0.2.0.dev），已 `pip install -e --no-deps` + 补 tensordict 0.6.2 / codetiming / hydra-core / pybind11 / pylatexenc / scikit-image / qwen-vl-utils |
| import 验证 | `agent_rollout_loop`、`vLLMRollout`、`ParallelEnv`、`VLAgentEnvV3`、`ToolBase` 全部通过 |
| Qwen3-VL 支持 | verl `models/transformers/monkey_patch.py` 含 qwen3_vl 分支（torch backend，无需 flash_attn）✓ |
| 跳过 | flash-attn（不需要，vLLM 自带 FlashAttention backend）、liger-kernel（可选加速）、gymnasium/playwright（可选 env，注册失败自动跳过） |

## 2. agent 模式核验（与本地端审计一致）

- 配置：`verl/trainer/config/ppo_trainer.yaml` 的 `agent:` 段——activate_agent / single_response_max_tokens=32768 / max_turns=50 / concurrent_workers=1 / tool_name_key=env_name / custom_stop=['</code>'] / max_vllm_images=32
- 循环：`verl/workers/agent/agent_rollout_loop`（parallel_env.py:136）驱动多轮工具循环，vllm_rollout_spmd.py:288 挂载
- 协议：`ToolBase.reset(raw_prompt, multi_modal_data)` + `execute(action_string) → (tool_result, reward, done, info)`；模板 [vl_agent_v3.py](../pyvision-rl/verl_agents/verl/workers/agent/envs/visual_agent/vl_agent_v3.py) 含 bbox 解析/缩放 + `<answer></answer>` 检测 + 单图回传（max_images_per_round=1）
- 参考训练脚本：`pyvision-rl/verl_agents/examples/agent/final_merged_v1v8_thinklite_single_node.sh`（GRPO + kl_coef=0 + FSDP param/optimizer offload + vllm rollout + agent 模式）——**2×24G 冒烟配置应以此为底**，train_batch 64→8、n=16→4、max_turns 20→8
- **verl 0.2.0.dev 无 LoRA 训练**（fsdp_utils 的 is_lora 仅是 wrap policy 分支）→ RL 走**全参 FSDP + param/optimizer offload**

## 3. vLLM 显存实测（Qwen3-VL-8B-ATR-SFT-v2，TP=2）

| 配置 | 结果 |
|---|---|
| util 0.5 / 0.6 / 0.65 + eager | KV cache 无空间，引擎起不来（eager 下 profile 激活峰值巨大；TP=1 + eager 直接 OOM 23.4/24GB） |
| **util 0.75 + CUDA graph（默认）** | **✓ 启动 48.7s（含 graph 捕获 ~9s），KV 2.42GiB** |
| util 0.65 + max_model_len 6144 | KV 仅 0.06GiB ✗（常驻与 max_model_len 基本无关） |

**结论：纯推理常驻 ~15.55GB/卡（TP=2）是硬基线**——权重分片 8.6 + CUDA graph 缓冲 + encoder cache（151250 token 预算）+ context。RL 同卡训练侧只剩 ~8.5GB/卡。

**全链路验证通过**：processor chat template（AutoProcessor.apply_chat_template + `<image>` 占位符）→ `llm.generate(multi_modal_data={"image": ...})` → 输出 `'white'`（GT 正确）✓

## 4. 冒烟训练配置草案（2×24G，待本地端 env/数据到位后执行）

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=[vstar_grpo.parquet] \
    data.val_files=[...] \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=12288 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.max_turns=8 \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=4096 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=200 \
    trainer.total_epochs=8
```

冒烟版：train_batch=2、n=2、max_turns=2、total_epochs=1、save_freq=10000（先不存盘）。

## 5. 风险清单（按优先级）

1. **同卡共存无余量**：推理 15.55GB/卡 + 训练仅剩 ~8.5GB（FSDP 全 offload + micro_batch=1 理论可行但紧）→ **冒烟训练第一步就验证**；若 OOM，备选：rollout util 降到 0.7 同时 max_model_len 降 8192。
2. **vllm 0.11 + vendored verl 兼容**：rollout 层已改用 vllm 新 API（`from vllm import LLM`），单卡直测通过，但 verl 的 DataProto 装配、agent 循环内采样参数路径未实跑 → 冒烟验证。
3. **磁盘**：`/root/autodl-tmp` 仅剩 13G；8B hf 全量 checkpoint ~16G 装不下。处理：`trainer.save_freq` 调大 + checkpoint contents 只存 `['model']`；或删 `/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`（17G，基线原版，SFT-v2 已含其能力）腾空间。
4. **max_turns=8 的序列预算**：8 轮 ×（1 新图 ~1.3k token + 文本）可能超 max_prompt_length 8192 → 冒烟时观察 truncation；必要时 max_turns 6 或提升 max_prompt_length。
5. **无 P2P**：TP=2 通信走 PCIe，多卡吞吐打折；batch 小影响有限。

## 6. 交接点

- 本地端 1–4 交付物（vstar_env.py / patch_reward 修复 / vstar_to_verl_parquet.py / compute_score）到位后，远端执行：冒烟训练（train_batch=2）→ 全量 GRPO。
- 远端已就绪，无阻塞。磁盘清理决策（删 Instruct 基线？）请本地端拍板或远端默认执行。
