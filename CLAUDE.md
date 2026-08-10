# CLAUDE.md — PyVision-ATR 项目协作规范

> 本文件由**本地 AI** 与**远端 GPU AI** 共同阅读。先判定自己是谁：
>
> - **本地 AI（Windows 开发机）**：用户面前的机器，**无 GPU**，禁止一切 GPU 负载 → 按「本地职责」执行
> - **远端 AI（GPU 服务器，AutoDL/SeetaCloud）**：跑训练与评测的机器，环境见 `ai_handoff/02_env_setup.md` → 按「远端职责」执行
>
> 用户负责两端的物理搬运（数据包、网盘）。沟通载体 = **GitHub 仓库 + `ai_handoff/` 交接文档**。

## 项目一句话

Qwen3-VL-8B 工具调用 + **ATR 奖励**（`R = acc + λU − γC + ηS`）的智能体视觉 RL。
流程：397B 高质量轨迹 → SFT 冷启动（两轮已完成，强制 34% / 先答后验 62%）→ **GRPO RL**（verl，当前阶段）。
RL 集成已完成（VStarToolEnv + patch_reward + vstar→verl parquet + compute_score，冒烟 34/34 通过），交接文档 `ai_handoff/05_rl_plan.md` 已写好，等远端执行。

## 双端分工

| 端 | 机器 | 做什么 | 绝不做什么 |
|----|------|--------|-----------|
| 本地 | Windows 开发机（无 GPU） | 写代码；改 tools/reward/adapter；**CPU 冒烟测试**（`experiments/scripts/smoke_*.py`）；SFT 数据过滤/检查；vstar→verl parquet 转换（纯 CPU）；写 `ai_handoff/` 交接文档；git 提交推送 | **一切 GPU 负载**：模型推理/训练/评测（`run_atr_offline.py`、vLLM、verl、torchrun、deepspeed、`nvidia-smi` 等）。`.claude/settings.local.json` 已配置 deny 拦截，不要绕过。也不直接 SSH 连远端（用户托管） |
| 远端 | GPU 服务器 | SFT/RL 训练；vLLM 服务；`run_atr_offline.py` 评测；模型合并导出；按交接文档执行并回报结果 | 不删改本地需要的源码与数据（远端是执行端，成果回写交接文档 + push） |

## 协作流程（每轮远端任务）

1. 本地把任务写成编号交接文档 `ai_handoff/0N_主题.md`（格式：背景 → 当前进度 → Step 0 诊断 → 具体命令 → 回报要求），commit + push
2. 用户把代码/数据带到远端（远端 `git pull` + 数据包搬运）
3. 远端 AI 执行交接文档 → 结果（指标、产物路径、问题）写回报节，尽量 push 回仓库（无凭据则回报给用户）
4. 本地 pull，结论整理进 `knowledge-base/`，进入下一轮

当前阶段 = `ai_handoff/` 最新编号文档；`knowledge-base/` 是方案事实源（`SFT_TRAIN_README.md` 训练手册、`TOOL_SUMMARY.md` 工具接口、`REFERENCES.md` 引用映射）。

## Git 同步协议（GitHub）

origin = `https://github.com/kuangweiquan/ADAPTIVE-TOOL-RL`（已配置，main）

**进 git**：`atr/`、`pyvision-rl/`、`experiments/configs`、`experiments/scripts`、`knowledge-base/`、`ai_handoff/`、`datasets/vstar_bench/sft/*.jsonl`、`CLAUDE.md`、`.claude/`（settings.local.json 除外）

**不进 git**（.gitignore + PrePush hook 双重拦截）：
- 大图片数据：`datasets/vstar_bench/{direct_attributes,relative_position,crops,sft_images,sft_v1_backup}/`
- 产物：`log/`、`experiments/results/`、checkpoints、`*.pt/*.pth/*.ckpt/*.safetensors`
- 数据格式：`*.parquet`（含 vstar→verl 转换输出，**走数据包**）、`*.npy`、`*.tar.gz`
- 所有 >50 MB 的文件（GitHub 单文件硬上限 100 MB）

**提交信息格式**：`[local|remote] 范围: 摘要`，例：`[local] atr: 新增 VStarToolEnv`、`[remote] sft: 第2轮训练完成 acc=34%`。
push 被 PrePush hook 拒绝 = 有大数据/禁用路径混进提交，先 `git rm --cached` 修正再推，不是故障。

## 代码约定

- **工具定义单一真值源**：`atr/tools/` 注册表。新增/改名工具必须同步改：注册表 → SYSTEM_PROMPT 生成 → `run_atr_offline.py` 工具分发 → reward 工具名校验（见 README 工具表）
- 坐标一律归一化 `[0,1]`（`bbox_2d`）
- 文档用中文，代码注释/命名用英文
- 冒烟测试必须**纯 CPU 可跑**；改 tools/reward/adapter 后先跑通冒烟再写交接文档
- 新逻辑尽量落成本地可跑测试（CPU 模拟环境），不要只等远端验证

## 环境速查

- 远端：conda env `atr`，工作目录 `/root/code`（详见 `ai_handoff/02_env_setup.md`）
- 模型：Qwen3-VL-8B（SFT 后 `Qwen3-VL-8B-ATR-SFT`）
- 数据：VStar 基准原始图 338 MB，走 `atr_project_bundle*.tar.gz` 数据包 / 网盘同步，不进 git
