# 06 — 4×3090 全量 GRPO 执行计划（2026-08-11）

> 背景：2×24G 冒烟阶段完成全部代码修复，但 backward 峰值（参数+梯度双驻留 17.5G）超出 24G 物理上限，判定**必须换实例**。已升级 4×3090（96G）。
> 本交接文档 = 开卡后完整执行路径。**阶段 0 全部无 GPU 可做；阶段 1 必须 GPU。**

## 已有成果（代码已提交本仓库，git pull 即得）

- **chunked lm_head**（8B 模型在 24G 卡上 forward/backward 的内存救星，4 卡下依然生效更稳）：
  `qwen3_vl.py`（return_hidden_states 分支）+ `torch_functional.py`（logprobs_from_hidden 分块 log_softmax）+ `dp_actor.py`（lm_head 实例 patch + FSDP 空 shard broadcast）
- rl_dataset.py doc2len 修复（mm_hint 读图）、fsdp_vllm.py wake_up OOM 修复、fsdp_workers.py 训练前 empty_cache、ppo_trainer.yaml std_sort_enable=False
- 冒烟脚本 `run_vstar_smoke.sh`（已 4 卡化）+ 全量脚本 `run_vstar_full.sh`（新写，B7 修正版）
- 数据 json：`datasets/vstar_bench/rl/{train,val,train_smoke_short,val_smoke_short}.json`（train=171/val=20，短样本 4 条 3415-3796 tokens）

---

## 阶段 0：开卡前 — 无 GPU（新实例上先做，别等开卡）

1. **代码同步**：`git pull origin main`（或当前分支 sync_remote）
2. **图片数据搬运**（关键！训练会读图，缺图直接挂）：
   - 191 张图原始路径：`/root/code/datasets/vstar_bench/{direct_attributes,relative_position,crops}/`（rl/train.json 的 mm_hint.hint_path 指向这些目录）
   - 数据包 `images.zip`（269M，本机已打包）或网盘同步，**不进 git**
   - 校验：`python3 -c "import json; [json.load(open(f)) for f in ['datasets/vstar_bench/rl/train.json','datasets/vstar_bench/rl/val.json']]; print('json ok')"` + 抽查一张图存在
3. **环境预检**（无 GPU 也能跑大部分）：
   - `conda activate atr` 存在？vLLM 0.11、transformers、torch≥2.4 版本符合（照 `ai_handoff/02_env_setup.md`）
   - 模型在 `/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2`？（SFT v2 权重，含 processor）
   - `datasets/vstar_bench/rl/` 下 4 个 json 就位（本仓库已含）
4. 熟悉两个脚本参数（见下），确认无需再改

## 阶段 1：开卡后 — GPU 部分

### Step 1 冒烟（B4，~10 分钟）
```bash
cd /root/code
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $pid; done
setsid nohup bash run_vstar_smoke.sh > /dev/null 2>&1 < /dev/null &
# 轮询（间隔 1-2 分钟）：grep -E "\[CHUNK\]|\[MB diag\]|pg_loss|\[ATR\] acc|OutOfMemory" logs/vstar_smoke.log | tail
# 注意：Bash 工具 2 分钟 timeout 会杀进程组，必须 setsid；启动 ~4 分钟到训练步
```
验证四项（全部通过才进 Step 2）：
1. **agent 注册**：`grep -E "\[DEBUG agent\] num_agent" logs/vstar_smoke.log`
2. **工具执行**：`grep -E "\[Zoomed into|\[OCR result:" logs/vstar_smoke.log`（或 Tool calling 100%）
3. **ATR reward 非全 0**：`grep "\[ATR\] acc=" logs/vstar_smoke.log` → 形如 `acc=...U=...C=...S=...→R=`，R 不全是 0（全 0 = reward 管道断）
4. **不 OOM + loss 打印**：日志出现 `actor/pg_loss`（4 卡下 backward 峰值 ~12G，应有大余量）

**附带动作：实测响应长度**（决定全量 max_response）：
`grep -E "\[CHUNK\] hidden" logs/vstar_smoke.log | tail -2` → hidden=(nnz, 4096)，响应≈nnz−prompt(3415-3796)。若响应均值 <4096 → 全量 max_response 可保持 8192；若接近 2048 上限（生满）→ 全量时降 max_response 至 4096-5120 减 KV。

### Step 2 全量 GRPO（B7，约 640 步）
```bash
setsid nohup bash run_vstar_full.sh > /dev/null 2>&1 < /dev/null &
```
- 配置要点（相对 05 手稿 B7 的修正）：max_prompt 5120（旧值 2048 是 bug 时代）、max_model_len 14336、n_gpus_per_node=4（TP=4，vocab 整除）、dynamic bsz + max_token_len_per_gpu=8192、checkpoint `/root/autodl-tmp/rl_ckpt/vstar_atr/qwen3vl_8b_sftv2_grpo_4gpu`、save_freq=25
- **预期步数**：filter_overlong_prompts 会滤掉 >5120 tokens 的样本，实际步数 = 滤后样本数/8 × 30 ≈ 450-640 步；启动后看第一行 Total training steps 确认
- **显存风险点（理论上已避开）**：训练段 ~15G/卡、rollout 段 ~20G/卡（KV 60G/4）。若 OOM 依次：`ppo_max_token_len_per_gpu 8192→6144`、`gpu_memory_utilization 0.85→0.75`、`n=4→3`
- 中途检查：每 25 步存盘（checkpoint 在 /root/autodl-tmp，不动系统盘）；`grep "pg_loss"` 看 loss 下降；reward 波动正常

### Step 3 回报（B8 格式）
1. 结果写 `ai_handoff/05_rl_plan.md` 回报节：指标（reward 均值、acc、pg_loss 曲线要点）、checkpoint 路径、问题与解决
2. `git add`（只 add 源码/文档/json，**不要 add images.zip/parquet/checkpoint/logs**）→ commit `[remote] rl: 全量 GRPO 完成 acc=...` → push（pre-push 拦截 >50MB = 有大数据混入，`git rm --cached` 修正，不是故障）
3. 回报给用户：训练完成、产物路径、下一步（评测/继续训练）

## 不做什么（红线）
- 不删改本地需要的源码（本仓库代码是唯一真值）
- 不把 checkpoint/大文件推 git
- 不在未验证冒烟四项前直接跑全量

## 回报 2（2026-08-11 第 7 次启动前，远端）

**状态：代码侧 OOM 修复链全部完成并 CPU 验证通过；训练启动受阻于实例 GPU 状态（见下）。**

### 本轮 4 项修复（解决训练段最后两个 OOM）
1. **entropy 分块版**（`pyvision-rl/verl_agents/verl/utils/torch_functional.py`）：
   `entropy_from_logits` 按 vocab chunk=16384 两趟计算（sum_exp + sum(pd·logits)），
   峰值 5.55G → ~200M。maxdiff 1.16e-4（bf16 精度内）。
2. **lm_head chunk 级权重收集**（`dp_actor.py` `_vstar_chunked_forward` 重写）：
   不再全量 all_gather（2.82G 分配目标，free 1.4G 时 OOM）；每 rank 只提取自己
   持有的行（`_shard_param_infos` 的 off/numel，0 填充被 shard 切开的行），
   `gather_chunk(start,end)` 每次只 all_gather O(chunk×hidden)≈134M，行由唯一
   owner 非零贡献、sum 合并（跨 rank 分裂的行互相补全）。
3. **权重梯度回传修复**：原 `flat._local_shard.data` 的 `.data` 切断 autograd 图
   → lm_head 权重永不更新；去掉 `.data` 后梯度经 copy/all_gather 回传 FSDP flat shard。
4. **`logprobs_from_hidden` 钩子**：新增 `weight_gather_fn` + `vocab_size` 参数
   （weight_gather_fn 模式下 lm_head_weight=None，vocab 由调用方传入）。

### CPU 验证（/tmp/verify_chunk.py，conda atr 环境，全部通过）
- chunk 合并权重 == 全量权重（含跨 rank 分裂行）✓
- weight_gather_fn 路径与原版 logprobs maxdiff = 0 ✓
- backward 梯度回传 x 与两个 local shard（全覆盖、无越界）✓
- 两个文件 py_compile 通过 ✓

### 阻塞：实例 GPU 不可用（用户需处理）
- 实例于 18:43 重启（PID1 均为 18:43），`/dev/nvidia0-3` 缺失（已 mknod 仍无效）、
  `/proc/driver/nvidia/gpus/` 不存在、`nvidia-smi` → "No devices were found"、
  `torch.cuda.is_available()=False`。驱动版本可读（NVRM 570.124.04）但无 GPU 绑定。
- 判断：AutoDL 控制台侧未分配 GPU（无卡模式开机/重启未挂 GPU）。容器内无法修复。
- **需要用户**：AutoDL 控制台确认实例 GPU 模式，必要时重启实例；GPU 恢复后
  直接跑 `setsid nohup bash run_vstar_full.sh > /dev/null 2>&1 < /dev/null &`，
  10 秒级轮询 `grep -E "\[CHUNK\]|pg_loss|\[ATR\] acc|OutOfMemory" logs/qwen3vl_8b_sftv2_grpo_4gpu.log`
  等 STEP1_DONE（actor/pg_loss + [ATR] acc）。

---

## 回报 3（2026-08-12，远端 2 卡调试，GPU 不足期）

> 用户期 GPU 不足（2×24G），先做能做的调试。结论：**4 卡 OOM 根因 = 视觉 token 无限制（已修复），但 4×24G 全参训练另有物理极限（step 峰值 33.6G），必须换 8 卡或降训练参数量。**

### ① 4 卡 OOM 根因（确凿，已修复）
- **processor_config.json 的 `size` 语义 = 像素预算**：原配置 `longest_edge=16777216`（16M 像素≈无限制）且无 max_pixels
  → Qwen2VLImageProcessorFast 按原始分辨率处理：171 张图全部 ≥9964 视觉 token（中位 13254，最大 32400，尺寸 1500×1827~5759×1440），而文本 nnz 仅 4121-5124。
  **视觉 token 从没进过显存账本 = 4 卡微批 6-8 OOM 根因**（不同样本视觉 token 差异 13160-32400 → 微批间峰值差 3-4G）。
- **修复**（远端模型目录，本地无此文件）：`/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2/processor_config.json`
  → `size={"longest_edge": 1003520, "shortest_edge": 3136}`（≈1M 像素 = transformers max_pixels 默认值，≈1024 边长，与 SFT `max_image_size: 1024` 一致）
  → 视觉 token 全样本降到 3696-3900（降 8 倍）。vLLM 采样端 + FSDP 训练端共用此 processor，一处改两处生效。
- 数据机制（verify_data_build.py 钉死）：input_ids 仅 1 个 image_pad(151652) 占位，grid_patch(≈3800) 由模型 forward 内部展开；postprocess_data 右截断/pad 5120 不触发截断（文本 ~1900）。

### ② 4×24G 全参训练物理极限（本轮新发现，与视觉无关）
- **真实参数量**：safetensors 17.5G（bf16）= **8.75B**（GQA：qkv 21M + o 16.8M + MLP 151M = 189M/层 ×36 = 6.8B + emb 1.24B + vision 0.71B）。
- **生产 FSDP 语义**（fsdp_workers.py）：actor 模型 **fp32 创建**（179 行注释：bf16 优化器不正确）、
  `use_orig_params=True`（245-248 行：视觉塔 `requires_grad_(False)` 强制）、MixedPrecision bf16、
  梯度 fp32（reduce_dtype 默认 fp32）、AdamW 状态惰性创建（第一轮 step 前不在 GPU）。
- **第一轮 update_policy 的 optimizer.step() 峰值（4 卡）**：
  参数 bf16 全量 unshard 8.04G（FSDP use_orig_params=True step 时 unshard）+ 视觉 frozen 全量 1.42G
  + 梯度 fp32 分片 8.04G + Adam 状态 fp32 分片 16.08G ≈ **33.6G > 24G，激活为 0 也崩**。
- **Mini 同构复现**（/tmp/verify_accum.py，2 卡 24G，3.04G 迷你模型 + 生产语义全模拟）：
  微批循环 forward/backward 能过（ckpt 生效时 post-fwd 7.77G），**round1 optimizer.step() 创建状态 + unshard 时 OOM**
  （崩点 in use 23.11G ≈ 参数 1.52+3.04 全量 + 梯度 6.08 + 状态 12.2，精确匹配）→ 与 8B 4 卡同构。
- **Q3 顺带结论**：「ckpt 生效仍 +0.68G/层逐层累积」= **无 ckpt 的正常行为**（Mini CKPT=0 同构递增，CKPT=1 平坦降半）；
  8B 2 卡观察到的累积是 ckpt 未生效（生产中 enable_gradient_checkpointing=True 于语言模型，生效时曲线平坦）。
- **因此**：视觉 token 修复（3800）只能让微批循环通过；**第一轮 step 必 OOM → 4×24G 全参 8B 物理不可行**，不是 bug。

### ③ 下一步选项（需用户拍板）
| 方案 | 4 卡账本(step) | 改动 | 备注 |
|------|---------------|------|------|
| **8×24G 或 4×32G+** | 8 卡: 参数 1.0+视觉 1.42+梯度 2.0+状态 4.0 ≈ 8.4G ✓ | 无 | 最稳，视觉 1.42G 全量每卡不变 |
| **LoRA**（verl 是否支持需查） | 可训练 ~0.5B → 全部 <4G ✓ | 中 | RL 中 LoRA 收敛质量待验证 |
| **Adafactor**（状态 1×） | 4.02+1.42+8.04+4.0 ≈ 17.5G ✓ | 小（fsdp_workers.py 319 行换 optimizer） | 收敛行为差异大，RL 少见 |
| 冻结 decoder 部分层 | 线性降 | 小 | 训练质量未知 |
| **2 卡 + 更大显存** | 不可能（2 卡 8B 连纯文本 1902 token 都 OOM，物理极限实锤） | - | 已排除 |

### ④ 本轮新增脚本（已入库）
- `experiments/scripts/verify_data_build.py`：CPU 验证 rl_dataset 数据构建（processor 1M 像素 + postprocess_data + get_rope_index），171/171 样本 img_tok(1) < grid(3696-3900)。
- `experiments/scripts/verify_microbatch_2gpu.py`：2 卡真实 8B + 真实数据 + 生产微批路径（unpad/position_ids/chunked lm_head/FSDP），RESP_LEN/NOIMG/HOOKS/OFFLOAD/CKPT 环境变量控制。
- 用法（需 GPU）：见文件头 docstring。

---

## 回报 4（2026-08-12，4×24G 降显存方案定稿：Adafactor 优化器）

> 用户澄清最大负荷 = **4×24G 3090**。回报 3 结论：4 卡账本 AdamW step 峰值 29.6G > 24G，全参训练物理不可行。
> **本回报 = 降显存方案：优化器 AdamW → Adafactor（状态 2× fp32 → 1× fp32，省 8.0G/卡），4 卡可跑。**

### 4×24G 最终账本（生产配置：model_dtype=bf16、视觉冻结、FSDP FULL_SHARD 4 卡）
| 项 | dtype | 每卡 | 说明 |
|----|-------|------|------|
| 参数(8.04B 可训练) | bf16 分片 | 4.02G | model_dtype=bf16(脚本已配) |
| 视觉(0.71B frozen) | bf16 分片 | 0.36G | **FSDP 对 frozen 参数也分片**(FROZEN=1 实验证实，非全量复制) |
| 梯度 | fp32 分片 | 8.04G | FSDP1 梯度存储固定 fp32(reduce_dtype 只影响通信，实测证实) |
| **Adafactor 状态** | fp32 分片 | **8.04G** | 1× fp32(vs AdamW 16.08G 2×) |
| step 峰值 | | **≈22.4G** | 常驻 20.5G + 临时 ~2G < 23.57G ✓ 余量 ~1.2G |

### 2 卡 Mini 同构验证（/tmp/verify_accum.py，生产语义全模拟，两轮 update_policy）
- **AdamW**：round1 微批循环能过（ckpt 生效 post-fwd 7.77G），**step 崩（24.4G ≈ post-bwd 12.17 + 状态 12.16）**
- **Adafactor + 视觉冻结**：状态仅 6.08G（分片参数 1.52G × 4B），round1/round2 全流程峰值 20.8G ✓
- 附带结论：FSDP step **无参数 unshard**（峰值与账本精确吻合）；冻结参数不产生梯度/状态（峰值 20.8→17.2G）

### 改动清单（已入库，git pull 即得）
1. `pyvision-rl/verl_agents/verl/workers/fsdp_workers.py`：`optim_config.name` 分支——
   `adafactor` 用 `optim.Adafactor(lr, weight_decay, beta2_decay=-0.8)`（torch 2.8 新签名，显式 lr 生效）；默认仍 adamw
2. `pyvision-rl/verl_agents/verl/trainer/config/ppo_trainer.yaml`：`actor.optim.name: adafactor`（含账本注释）
3. `run_vstar_full.sh`：显存账注释更新（(5) 段：optimizer 账本此前未计入 + Adafactor 修正）

### 注意事项
- **Adafactor 状态也是标准 Tensor**（offload_fsdp_optimizer/load 兼容，round2 load 生效已实测）；惰性初始化与 AdamW 相同（第一轮 step 前状态不在 GPU）
- **收敛差异**：beta2_decay=-0.8（decay 0.8）vs AdamW beta2=0.999，二阶矩衰减快；GRPO 首轮训练观察 loss/reward，若异常回退 `name: adamw`（yaml 一行）
- checkpoint 的 optimizer state 保存/恢复（FSDPCheckpointManager）对 Adafactor 通用（标准 torch state_dict）
- 训练启动前 `grep -E "Total steps|pg_loss|\[ATR\] acc|OutOfMemory" logs/qwen3vl_8b_sftv2_grpo_4gpu.log`；首轮 step 是关键（此前从未跑过 step 阶段）

---

## 回报 5（2026-08-13，本地决策前置动作②：processor_config.json 全文入库）

> 文件：`/root/autodl-tmp/models/Qwen3-VL-8B-ATR-SFT-v2/processor_config.json`
> 修改内容（相对 Qwen3-VL-8B 官方原版，仅 `image_processor.size` 一处）：`longest_edge 16777216 → 1003520`、新增 `shortest_edge: 3136`（≈1M 像素预算，与 SFT `max_image_size: 1024` 同量级；video_processor 未动）。
> 作用：Qwen2VLImageProcessorFast 按像素预算缩放 → 视觉 token 全样本从 9964-32400 降到 3696-3900，是 4 卡 OOM 根因修复（见回报 3 ①）。**vLLM 采样端 + FSDP 训练端共用此 processor，一处改两处生效。**
> 注意：此文件不进 git，换实例/重装模型后按下方全文重新打补丁（对照 02_env_setup.md 环境清单）。

```json
{
  "image_processor": {
    "do_convert_rgb": true,
    "do_normalize": true,
    "do_rescale": true,
    "do_resize": true,
    "image_mean": [
      0.5,
      0.5,
      0.5
    ],
    "image_processor_type": "Qwen2VLImageProcessor",
    "image_std": [
      0.5,
      0.5,
      0.5
    ],
    "merge_size": 2,
    "patch_size": 16,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
      "longest_edge": 1003520,
      "shortest_edge": 3136
    },
    "temporal_patch_size": 2
  },
  "processor_class": "Qwen3VLProcessor",
  "video_processor": {
    "do_convert_rgb": true,
    "do_normalize": true,
    "do_rescale": true,
    "do_resize": true,
    "do_sample_frames": true,
    "fps": 2,
    "image_mean": [
      0.5,
      0.5,
      0.5
    ],
    "image_std": [
      0.5,
      0.5,
      0.5
    ],
    "max_frames": 768,
    "merge_size": 2,
    "min_frames": 4,
    "patch_size": 16,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "return_metadata": false,
    "size": {
      "longest_edge": 25165824,
      "shortest_edge": 4096
    },
    "temporal_patch_size": 2,
    "video_processor_type": "Qwen3VLVideoProcessor"
  }
}
```

### 本轮代码变更（前置动作①：reward 分解打印，已入库）

- `pyvision-rl/verl_agents/verl/trainer/main_ppo.py`：`num_examine=0` 硬编码 → `config.trainer.get("num_examine", 0)`（默认 0 不变，命令行可开）
- `run_vstar_full.sh`：新增两行 → `trainer.num_examine=50`（patch_reward 的 `[ATR] acc=... U=... C=... S=... → R=` 门控）+ `reward_model.reward_kwargs.atr_config_dict.verbose=true`（base_reward 每次 compute 都打）
- 已 CPU 验证：`ATRConfig(**{"lambda_u": 1.0, "gamma_c": 0.5, "eta_s": 0.3, "verbose": True})` 构造通过

### 全量启动后监控（50 步止损三门槛，07 本地决策 2）

| # | 指标 | 通过线 | 不满足 → 停 |
|---|------|--------|-------------|
| 1 | `actor/pg_loss` | 下降趋势（非恒定/发散） | 停，回报 loss 曲线 |
| 2 | `agent/tool_call_mean` | 保持 > 0.5 | 停——U/C/S 信号消失 |
| 3 | `critic/acc/acc_of_this_batch` | 出现非 0 值 | 停——模型学不会作答，可能模板/分布偏移 |

监控命令：`grep -E "pg_loss|tool_call_mean|acc_of_this_batch|\[ATR\] acc" logs/qwen3vl_8b_sftv2_grpo_4gpu.log | tail`

---

## 回报 6（2026-08-13，全量启动 OOM 根因与修复：梯度跨 micro 累积超限）

> 全量实际启动后连续两次在 **step 2 的 update_policy backward OOM**（`_cast_grad_to_param_dtype` 分配 674 MiB 失败，in use 23.0G/23.57G）。per-process 探针（每 5 秒 nvidia-smi）完整捕获后确诊，**不是偶发，是账本结构性漏算**。

### 根因（探针实测，step 2 微批循环逐帧显存）

| micro | 显存 | 构成 |
|-------|------|------|
| 第 1 个 post-fwd | ~15G | 基座 8.9G + 临时（logits 分块/激活） |
| 第 2-3 个 post-fwd | 20.3→23.4G | + 梯度逐步累积（每个 backward +~2G） |
| 第 4 个 post-fwd | 23.9G | 基座 8.9 + 梯度 6G + 临时 |
| 第 4 个 backward | **>23.57G 崩** | + grad cast 674M + 新梯度 |

- **基座 8.9G/卡** = 参数 bf16 分片 4.02 + 视觉 0.36 + **Adafactor 状态 bf16 分片 4.02**（实测验证：torch Adafactor 状态随参数 dtype，`variance` 为 bf16 —— **06 回报 4 账本把状态按 fp32 记 8.04G，高估 4G**）+ CUDA context ~0.5
- **梯度 fp32 分片满 8.04G** 是跨 micro batch 逐步累积的（`zero_grad` 在 mini_batch 层，ppo_trainer 语义），**账本只算了「梯度常驻」没算「梯度累积 + 微批临时并存」** —— 第 4 个 micro 峰值 = 基座 8.9 + 梯度满 8.04 + 临时 ~4.3 ≈ 23.9G > 23.57G，**必然崩**
- 附带确认：**`ppo_max_token_len_per_gpu` 对多模态样本无效**（dp_actor update_policy 的 `has_multi_modal_inputs` 分支按样本切分，dynamic bsz 分支被绕过）——两次 OOM 的 `allocated 35.82G` 分毫不差就是证据，3072→2048 改动无效

### 修复（已入库，正在验证）

`run_vstar_full.sh`：`ppo_mini_batch_size 4 → 2` → 梯度只累积 2 个 micro（满 4.02G）→ 峰值预算 ≈ 基座 8.9 + 梯度 4.02 + 临时 4.3 ≈ **17.2G < 23.57G（余 6G）**。
- 训练语义影响：GRPO 每步 rollout 仍 32 条，仅每次 update 的梯度来自 2 条样本（原 4 条），梯度噪声 ×√2，可接受
- 实测效果：step 1 通过（306s/步，比 335s 略快）、**step 2 通过（历史崩点清除）**、训练段 GPU ~17G 与预测吻合
- 账本修正后注释已同步（06 回报 4 的 22.4G 峰值 → 实际 23.9G 会崩，原因如上）

### 已验证通过

- step 1/2 均无 OOM，`timing_s/step` 306s，GPU ~17G/卡
- vLLM seed 固定 → rollout 确定性复现（step 1 指标与首轮逐位相同，非日志残留）

### 待验证（50 步止损三门槛，07 本地决策 2）

继续监控 step 5/10/50：pg_loss 下降趋势 + tool_call_mean > 0.5 + acc_of_this_batch 离开 0（step 1 已见 15.6%）。

## 回报 7（2026-08-13，OOM 结构性修复定稿：优化器状态延迟加载，step 1-5 全过）

### 演进结论：mini_batch 4→2→1 只是推迟崩溃

| 配置 | 崩溃点 | 观察 |
|------|--------|------|
| mini_batch=4 | step 1（原始） | 梯度 8.04G 满额 + 基座 = 必然超限 |
| mini_batch=2 | step 3 | 梯度 4.02G，但样本 real nnz 波动（1901-2762）决定生死 |
| mini_batch=1 | step 5 | 每步 2 个 mini_batch 各 1 micro，峰值仅随样本波动 |
| **状态延迟加载** | **step 5+ 全过** | 见下 |

### 最终根因（代码级）

`fsdp_workers.update_actor` 在 backward 前就把 **Adafactor 状态（bf16 分片 4.02G/卡）load 回 GPU**，
backward 全程驻留 → 峰值 = 基座(参数 4.02+视觉 0.36+context 0.5) + 梯度 2.01 + **状态 4.02** + 激活/临时 ≈ 23.3G，
贴 23.57G 物理极限，样本一波动（nnz 2762）即崩。

### 修复（代码级，训练语义零变化）

把状态搬运时机从「backward 前」挪到「step 前」：
- `dp_actor.py`：`_optimizer_step` 内 step() 前 `load_fsdp_optimizer`、step() 后立即 `offload_fsdp_optimizer`
  （每 mini_batch 一次 load→step→offload；PCIe 开销 ~0.7s/step，可忽略）
- `fsdp_workers.py`：删除 update_actor 开头的 `load_fsdp_optimizer`（551 行）
- 效果：backward 峰值期状态在 CPU（省 4.02G），step 峰值期激活已释放（状态 4.02G 放得下）

### 验证（2026-08-13 第 5 次启动，mini_batch=1 + 状态延迟加载）

- **step 1-5 全部通过，0 OOM**（历史崩点 step 3 / step 5 均清除）
- 训练段物理占用 17-18G/卡（修复前 23.3G），post-bwd free 最低 5.1G（修复前 0.5G 崩）
- rollout 段峰值最高 21.9G（vLLM，sleep 后回落 16G）
- 节奏 ~306-330s/步

### 三指标（step 1-5，50 步止损三门槛初判）

| step | pg_loss | acc_of_this_batch | tool_call_mean |
|------|---------|-------------------|----------------|
| 1 | -0.241 | 0.156 | 5.60 |
| 2 | +0.400 | 0.188 | 5.95 |
| 3 | +0.280 | 0.188 | 5.83 |
| 4 | -0.250 | 0.188 | 6.42 |
| 5 | -0.108 | 0.062 | 6.50 |

- pg_loss 波动（-0.25~+0.40）无单调恶化趋势 ✓
- tool_call_mean 5.6-6.5 远超 0.5 ✓（8 轮 max_turns 内用完是正常初始态）
- **acc 离开 0**：0.156-0.188 稳定非零 ✓（step 5 回落 0.062 为样本波动，继续观察）

### 待验证

继续监控 step 10/25/50：pg_loss 趋势 + acc 是否保持离开 0 + checkpoint（save_freq=25）落地。
