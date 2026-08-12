# 07 — SFT + RL 训练尝试完整报告（远端 → 本地联合分析）

> 写于 2026-08-12，远端 GPU 服务器（AutoDL，4×3090 24G 上限）。
> 目的：把 SFT 两轮与 RL 全部尝试（含 3 个核心根因与 1 个已定稿的降显存方案）完整同步给本地 AI 联合分析。
> 代码已全部推 GitHub（sync_remote 分支，最新 `1b5004f`），`git pull` 即得。

---

## 一、总览

| 阶段 | 状态 | 一句话 |
|------|------|--------|
| 数据 | ✅ 完成 | 397B 高质量轨迹 → VStar 基准训练/验证集（171 训练 / 20 验证） |
| SFT 第 1 轮 | ✅ 完成 | 强制 34% |
| SFT 第 2 轮 | ✅ 完成 | 先答后验 62% |
| RL 集成 | ✅ 完成 | VStarToolEnv + patch_reward + vstar→verl parquet + compute_score，冒烟 34/34 通过 |
| RL 训练 | ❌ **从未跑通过任何 step** | 2 卡物理装不下；4 卡曾 OOM 于两个根因，已逐一修复/定方案，**等 4 卡恢复验证** |

**一句话现状**：RL 训练段的所有显存障碍（视觉 token 无限制、AdamW 优化器状态超限）都已定位并处理完毕，代码已入库；尚未在 4 卡上真机验证第一轮 step。

---

## 二、SFT 部分（已完成，两轮冷启动）

- 数据：397B 高质量轨迹（工具调用 + ATR 奖励），筛选/过滤后用于 SFT
- 第 1 轮：**强制 34%**（强制工具调用正确率 34%）
- 第 2 轮：**先答后验 62%**（先答后验策略下正确率 62%）
- 产物：`Qwen3-VL-8B-ATR-SFT-v2`（远端 `/root/autodl-tmp/models/`，本地无此目录，通过数据包/网盘同步）
- 与 RL 的衔接：SFT 后模型已具备工具调用格式能力，RL 阶段在其上做 GRPO 强化（奖励 = acc + λU − γC + ηS）

---

## 三、RL 部分尝试史（重点）

### 3.1 集成阶段（已完成，纯代码）

- VStarToolEnv（图像环境：直接属性/相对位置/裁剪工具）、patch_reward（ATR 奖励管道）、vstar→verl parquet 转换、compute_score
- 冒烟 34/34 通过（纯 CPU + 小模型路径全绿）
- 工具定义单一真值源（atr/tools/ 注册表）已建立

### 3.2 2×24G 阶段（物理极限，已排除）

- 8B 全参训练在 2 卡 24G 上**物理装不下**：参数分片 8.2G/卡 + 逐层激活累积 → 连纯文本 1902 token 都 OOM（alloc 22.68G）
- 测试矩阵：offload ON/OFF × ckpt ON/OFF 均 ~22.3G OOM
- 结论：2 卡只能做调试/冒烟，不能训练 → 升级 4 卡

### 3.3 4×24G 阶段（多次启动受阻 + 两个真根因）

启动史：
1. 第 1-6 次：训练段 OOM（当时误以为只差激活/权重收集），先后修复：entropy 分块、lm_head chunk 级权重收集（2.63G→134M）、.data 梯度回传
2. 第 7 次：实例无 GPU（设备节点缺失），中断
3. 2 卡启动 4 卡配置：ValueError（GPUs 2 < 4），配置未动
4. 用户 GPU 不足期：2 卡调试 → 完成 3.4/3.5 两个根因

### 3.4 根因 1：视觉 token 无限制（✅ 已修复）

- **现象**：4 卡微批 6-8 OOM，峰值比微批 1-5 高 3-4G，post-fwd 峰值 36-37.5G
- **根因**：processor_config.json 的 `size` 语义 = **像素预算**。原配置 `longest_edge=16777216`（16M 像素 ≈ 无限）且无 max_pixels → 图像按原始分辨率处理：
  - 171 张图全部 ≥9964 视觉 token（中位 13254，最大 32400，尺寸 1500×1827 ~ 5759×1440）
  - 文本 nnz 仅 4121-5124 → **视觉 token 从没进过显存账本**
- **修复**（远端模型目录，本地无此文件）：`processor_config.json` → `size={"longest_edge": 1003520, "shortest_edge": 3136}`（≈1M 像素 = transformers max_pixels 默认，≈1024 边长，与 SFT `max_image_size: 1024` 一致）→ 视觉 token 全样本降到 3696-3900（降 8 倍）。vLLM 采样端 + FSDP 训练端共用此 processor，一处改两处生效
- 数据机制（verify_data_build.py 钉死）：input_ids 只有 1 个 image_pad 占位，grid_patch 由模型 forward 内部展开；postprocess_data pad 5120 不触发截断（文本 ~1900）

### 3.5 根因 2：AdamW 优化器状态超限（✅ 方案已定稿，等真机验证）

- **现象**：即使视觉修复，第一轮 `optimizer.step()` 也会 OOM
- **根因**（账本，4 卡，生产配置 model_dtype=bf16 + 视觉冻结 + FSDP FULL_SHARD）：
  - 参数 bf16 分片 4.02G + 视觉 0.36G（frozen 也分片，非全量复制）
  - 梯度 fp32 分片 8.04G（FSDP1 梯度存储固定 fp32，reduce_dtype 只影响通信）
  - **AdamW 状态 fp32 2× 分片 16.08G**（惰性创建：第一轮 step 才出现）
  - **step 峰值 29.6G > 24G，激活为 0 也崩** → 4 卡全参训练此前任何 step 都不可能过
- **方案**：optimizer **AdamW → Adafactor**（torch 2.8 标准库，零新依赖；状态 2× fp32 → 1× fp32 = 8.0G/卡）→ **step 峰值 22.4G < 23.57G ✓ 余量 ~1.2G**
- **2 卡 Mini 同构验证**（/tmp/verify_accum.py，3.04G 迷你模型 + 生产语义全模拟：fp32 创建/use_orig_params=True/MixedPrecision/手动 offload-load，两轮 update_policy）：
  - AdamW：round1 微批循环能过，**step 崩（24.4G ≈ post-bwd 12.17 + 状态 12.16）**
  - Adafactor：状态仅 6.08G（分片参数 1.52G × 4B），round1/round2 全流程峰值 20.8G ✓ 两轮全通
  - 附带：FSDP step 无参数 unshard（峰值与账本精确吻合）；冻结参数不产生梯度/状态
- **改动**（已入库 1b5004f）：
  1. `fsdp_workers.py`：`optim_config.name` 分支（adafactor 用 `Adafactor(lr, weight_decay, beta2_decay=-0.8)`，torch 2.8 新签名显式 lr 生效；默认仍 adamw 可回退）
  2. `ppo_trainer.yaml`：`actor.optim.name: adafactor`
  3. `run_vstar_full.sh`：账本注释更新
- **风险**：Adafactor 的 beta2_decay=-0.8（decay 0.8）vs AdamW beta2=0.999，二阶矩衰减快，收敛行为可能不同；GRPO 首轮训练需观察 loss/reward，异常则 yaml 一行回退 adamw（需先降显存别的方式）

### 3.6 当前代码状态

- 全部修复/方案已提交推 GitHub（sync_remote 分支，最新 `1b5004f`）
- 新增验证脚本：verify_data_build.py（CPU）、verify_microbatch_2gpu.py（2 卡 8B 真实微批）、verify_fsdp_chunk_gpu.py（2 进程 FSDP chunk）
- 运行配置：run_vstar_full.sh（ppo_max_token_len_per_gpu=3072、gpu_memory_utilization=0.6、n=4、max_prompt 5120、max_model_len 14336、checkpoint /root/autodl-tmp）
- processor_config.json 改动在远端模型目录（不进 git，本地 AI 无法直接看到——需在 SFT 侧同样确认/对齐）

---

## 四、待本地 AI 联合分析的问题（重点）

1. **Adafactor 在 GRPO 场景的收敛风险**：decay 0.8 的短记忆二阶矩对 RL 这种非平稳目标是否可接受？是否建议用更大的 decay_rate（torch 2.8 的 beta2_decay 参数如何映射回 0.999 语义）？或先跑小规模（如 20 样本）验证 loss 下降趋势？
2. **降显存是否有更优替代**：本地 AI 是否有其他已验证的手段（如 bf16 优化器状态 + 精度补偿、LoRA/QLoRA 路线、冻结策略）值得在真机验证前评估？当前方案余量仅 1.2G，是否够稳？
3. **max_pixels=1M 与 SFT 1024 边长的一致性**：SFT 侧 `max_image_size: 1024` 与 RL 侧 1M 像素预算是否等价？会不会引入训练/推理分布偏移（如 SFT 用小图、RL 用大图）？
4. **微批序列策略**：视觉 3800 + 文本 1900 + 响应（max 8192 截断），3072 token/gpu 微批下训练质量是否受影响（token 利用率）？是否需要调 ppo_max_token_len_per_gpu？
5. **首轮 step 验证清单**：4 卡开机后的验证顺序（冒烟 → 全量 → 首轮 step 峰值观测），本地 AI 是否有补充检查点？

---

## 五、下一步计划（等 4 卡恢复）

1. `git pull` + 数据包校验（191 张图 + 模型目录 + 4 个 json）
2. 冒烟 `run_vstar_smoke.sh`：验证四项（agent 注册/工具执行/ATR reward 非全 0/不 OOM + pg_loss）
3. 全量 `run_vstar_full.sh`：重点观测**首轮 update_policy 的 step 峰值**（历史从未跑过 step 阶段）
4. 若首轮通过：观察 loss/reward 趋势（Adafactor 收敛验证）
5. 结果回报 → 本地 pull → knowledge-base 更新
