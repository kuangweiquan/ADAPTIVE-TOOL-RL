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
