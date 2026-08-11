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
