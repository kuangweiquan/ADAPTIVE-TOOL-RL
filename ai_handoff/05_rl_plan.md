# 05 — VStar RL：verl_agents 环境预装 + GRPO 训练流程

> 本地已完成 RL 集成并全部通过冒烟（34 passed, 0 failed），本手稿是远端执行指令。
> **Part A 立即执行**（环境预装）；Part A 通过后再执行 Part B（训练流程）。
> 模型：Qwen3-VL-8B-ATR-SFT-v2（04 合并产物，RL 起点权重）。

---

## Part A：verl_agents 环境预装

### A1 现状盘点（先跑，结果一并汇报）

```bash
nvidia-smi                       # 确认 2×24G 可用显存
ls /root/code                    # 项目目录结构（pyvision-rl 是否已在）
ls /root/autodl-tmp/models/      # Qwen3-VL-8B-ATR-SFT-v2 是否存在
conda env list && python --version
pip list 2>/dev/null | grep -iE "torch|vllm|tensordict|^ray|transformers|datasets|pyarrow"
```

### A2 同步代码包

**首选走 git**（本地已 push，远端直接拉，带版本记录）：

```bash
cd /root/code && git pull origin main
git config core.hooksPath .claude/hooks   # 启用 pre-push 大数据拦截（与本地一致）
```

> 远端若连不上 GitHub 才用备用包：`atr_project_bundle.tar.gz`（纯代码 ~0.5MB，含 atr/、pyvision-rl/、新增脚本、全部手稿；不含数据与 checkpoint），解压到 /root/code 即可。若本地有更新代码未推，以 git 为准。

### A3 安装 verl_agents（不碰已有 torch/vllm）

```bash
cd /root/code/pyvision-rl/verl_agents
pip install -e . --no-deps --no-build-isolation
# 补齐缺失依赖（逐个验证，已有则跳过；禁止装新的 torch/cuda 包）
pip install "tensordict<=0.6.2" "ray[default]>=2.10" datasets pyarrow hydra-core omegaconf wandb
```

- 注意：本机已有 vllm 0.11.0 + torch 2.8.0+cu128，setup.py 的 `vllm<=0.8.5` 只是 extras 限制，`--no-deps` 安装不影响已装 vllm。
- 若 `import tensordict` 与 torch 2.8 版本冲突 → 改用 `pip install tensordict`（让 pip 解析兼容版本）。

### A4 导入验证（Part A 通过的硬门槛）

```bash
cd /root/code && PYTHONPATH=/root/code python -c "
import tensordict, ray, datasets, pyarrow, hydra, omegaconf; print('deps ok')
import vllm; print('vllm', vllm.__version__)
import verl; print('verl ok')
"
```

```bash
cd /root/code && PYTHONPATH=/root/code python -c "
from verl.workers.agent.tool_envs import ToolBase
import atr.adapter.vstar_env          # 触发注册 vstar_tool_env
print('registry:', sorted(ToolBase.registry.keys()))
assert 'vstar_tool_env' in ToolBase.registry, 'env not registered!'
print('VStarToolEnv registration OK')
"
```

```bash
# env 生命周期自测（34 项冒烟，需要 vstar_bench 在 /root/code/datasets 下）
cd /root/code && PYTHONPATH=/root/code python experiments/scripts/smoke_vstar_rl.py
# 期望输出:===== RESULT: 34 passed, 0 failed =====
```

> 若远端 vstar_bench 不完整（缺 relative_position 等），从本地补同步 `datasets/vstar_bench`（只要图片+json，不要 crops/sft/rl 子目录）。

### A5 Part A 汇报格式

```
[环境] GPU 显存: xxx GB free ×2 | torch X.X / vllm X.X / tensordict X.X
[目录] /root/code 结构: ...
[模型] Qwen3-VL-8B-ATR-SFT-v2: 存在/缺失
[验证] deps ok: yes/no | verl ok: yes/no | VStarToolEnv 注册: yes/no | 冒烟: N passed
```

---

## Part B：RL 训练流程（Part A 通过后执行）

### B1 数据转换（vstar_bench → verl parquet）

```bash
cd /root/code && PYTHONPATH=/root/code python experiments/scripts/vstar_to_verl_parquet.py \
    --vstar_path datasets/vstar_bench \
    --output_dir datasets/vstar_bench/rl \
    --image_root /root/code/datasets/vstar_bench \
    --val_size 20
# 期望输出: Found 191 samples ... train=171 val=20 → datasets/vstar_bench/rl/{train,val}.parquet
```

检查打印的 sample row：prompt 含 system+user、mm_hint 为绝对路径、gt_bbox 归一化 [0,1]、ground_truth = options[0]。

### B2 打补丁（3 处小改，每处都有原注释锚点）

**Patch 1 — 注册 VStarToolEnv**（agent rollout worker 进程里 metaclass 注册依赖此 import）：

文件 `pyvision-rl/verl_agents/verl/workers/agent/__init__.py`，在文件末尾 `from .parallel_env import agent_rollout_loop` 之后追加：

```python
# VStar RL: register VStarToolEnv so agent rollout workers can find it
try:
    from atr.adapter.vstar_env import VStarToolEnv  # noqa: F401
except Exception as err:
    print(f' [ERROR] Failed to register VStarToolEnv : {err=}')
```

**Patch 2 — main_ppo.py 增加 "atr" reward manager 分支**：

文件 `pyvision-rl/verl_agents/verl/trainer/main_ppo.py`，在
`if reward_manager_name == "naive": ... reward_manager_cls = NaiveRewardManager` 块之后插入：

```python
        elif reward_manager_name == "atr":
            from atr.adapter.patch_reward import ATRRewardManager

            reward_manager_cls = ATRRewardManager
```

**Patch 3 — ppo_trainer.yaml 配置默认值**：

文件 `pyvision-rl/verl_agents/verl/trainer/config/ppo_trainer.yaml`：

```yaml
reward_model:
  ...
  reward_manager: naive      # ← 改为 atr
  reward_kwargs:             # ← 新增（main_ppo 读 config.reward_model.reward_kwargs 直传构造函数）
    atr_config_dict:
      lambda_u: 1.0
      gamma_c: 0.5
      eta_s: 0.3

custom_reward_function:
  path: null                 # ← 改为 atr.adapter.score
  name: compute_score        # ← 改为 compute_vstar_score
```

（也可不改 yaml，改用 launch 参数：`reward_model.reward_manager=atr`、`+reward_model.reward_kwargs=...`、`custom_reward_function.path=atr.adapter.score`、`custom_reward_function.name=compute_vstar_score`；改 yaml 更直观。）

### B3 训练脚本 run_vstar_grpo.sh（2×24G 调参版）

```bash
#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/root/code:$PYTHONPATH
export SAVE_CHECKPOINT_DIR=/root/code/rl_ckpt
PROJECT_NAME="vstar_atr"
EXPERIMENT_NAME="qwen3vl_8b_sftv2_grpo_2gpu"
mkdir -p ./logs

MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2
TRAIN_DATA=/root/code/datasets/vstar_bench/rl/train.parquet
VAL_DATA=/root/code/datasets/vstar_bench/rl/val.parquet

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=[${TRAIN_DATA}] \
    data.val_files=[${VAL_DATA}] \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.max_turns=8 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=1000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=30 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
```

要点（配置依据）：
- **GRPO**：`adv_estimator=grpo` + `rollout.n=4`（组内采样比较）+ temperature 0.7
- **agent 协议**：`activate_agent=True`、`tool_name_key=env_name`（env 名取数据列 env_name="vstar_tool_env"）、`max_turns=8`（answer-first 平均 3.76 轮，留足上限）、`single_response_max_tokens=2048`（单轮回复上限）
- **2×24G 显存**：FSDP param+optimizer offload 到 CPU（actor 显存占用降到几 GB），vLLM 拿 0.85×24G；`max_model_len=12288` 与评估期 vLLM 一致
- **reward 接线**（Patch 2/3 已生效）：`ATRRewardManager` 每样本算 R = acc + λU − γC + ηS，写入 response EOS 位置；死循环（同坐标重复 zoom、扫描平移、crop 微缩）因 C 项被压低，GT 对齐 zoom + 正确作答奖励最高
- **GRPO 无 KL**：`kl_coef=0.0`（数据少、起点已 SFT，靠 reward 直接优化）

### B4 冒烟（1 步，先验证再全量）

```bash
cd /root/code && bash run_vstar_grpo.sh > /dev/null  # 观察前 1 个 step
# 或先加临时参数跑 1 步: +debug=True
```

冒烟通过标准（日志中依次确认）：
1. agent 注册生效：`[DEBUG agent] num_agent=8, num_non_agent=0`（batch=8 条全部进 agent 循环）
2. 工具真正执行：日志/rollout 里出现 `[Zoomed into`、`[OCR result:` 等工具输出（vLLM 回传图像正常）
3. reward 非全 0 且分布有意义：`[ATR] acc=... U=... C=... S=... → R=...`（对齐 zoom 的 R 应明显高于死循环/错误轨迹）
4. 不 OOM、能跑完 1 个 step 并打印 loss

任一不满足 → 排查并修好后再全量（排查顺序：agent 注册 → env 执行 → reward 接线 → 显存）。

### B5 全量训练

- 171 条 / batch 8 → 22 步/epoch，30 epochs ≈ 640 步；用冒烟实测单步时间 × 640 估算总时长，先报告
- 观测：console 每步的 `[ATR]` 采样 + loss；期望趋势：mean reward 上升、死循环轨迹占比下降、accuracy 不崩
- checkpoint 默认存 hf_model（Patch 3 已含 `hf_model`），供评估直接加载

### B6 训练后评估（与 04 同口径）

1. 取最后一个 checkpoint 的 hf_model → 合并权重 → vLLM 起服务（同 04 Step 4 的方式）
2. `run_atr_offline.py` 双口径评估（forced-tool / answer-first，各 50 题）
3. 对比 SFT v2 基线（forced 34% / answer-first 62%）：
   - **目标 1**：answer-first acc ≥ 62%（不倒退）
   - **目标 2**：forced 场景显著提升（RL 的核心增益：学会“什么时候该用工具、用对工具”）
   - **目标 3**：死循环比例下降（04 报告 33/50 forced、15/50 answer-first）
4. **本轮补齐 IoU 指标**：v1 报告过但 v2 缺失；用评估的 trace 对比 direct_attributes/*.json 的 GT bbox 报同口径 IoU

### B7 风险与排障

| 现象 | 处理 |
|---|---|
| 显存 OOM | 依次降：`n=2`、`max_num_batched_tokens=4096`、`gpu_memory_utilization=0.9`、`train_batch_size=4` |
| reward 全 0 | 先看 `[ATR]` 日志是否存在 → env 是否执行（agent 日志）→ compute_score 是否拿到 extra_info（改 num_examine 看 detail） |
| vLLM 图编译/模型加载失败 | `enforce_eager=True` 已开；确认 model.path 是合并后的 hf 模型（非 LoRA adapter 目录） |
| agent 循环卡死/超长 | max_turns=8 兜底；单条轨迹超长会在 response_length 截断 |
| tensordict 与 torch 2.8 冲突 | 换 pip 可解析的版本（A3 注） |
| 训练发散 | lr 1e-6 起步（SFT 已收敛，RL 只微调）；必要时开 kl_coef=1e-3 |

### B8 汇报格式

```
[冒烟] 1-step 通过: yes/no | agent 注册: num_agent= | 工具执行: yes/no | 单步耗时: Xs
[训练] 总步数/预计: 640 / Xh | 最终 mean reward: X | acc 曲线: ... | 死循环占比: SFTv2 X% → RL X%
[评估] forced: acc X% (SFTv2 34%) | answer-first: acc X% (SFTv2 62%) | 平均轮次 | 死循环 | IoU
[checkpoint] 路径
[结论] 是否达到目标 1/2/3，建议下一步
```

---

## 与 04 的衔接

- 判定：answer-first 62% ≥ 30% 阈值 → 决策树分支 1，直接进入 RL（本手稿）
- RL 起点 = SFT v2 合并权重；SFT v1/v2 的 checkpoint 与 LoRA 权重保持不动
- 04 遗留项：**v2 的 IoU 指标缺失** → B6 第 4 步补上
