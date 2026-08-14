# CLAUDE.md — PyVision-ATR 项目协作规范

> 本文件由**本地 AI** 与**远端 AI（可选后备）**共同阅读。先判定自己是谁：
>
> - **本地 AI（Windows 开发机）**：用户面前的机器，**无 GPU**，禁止本机一切 GPU 负载 → 按「本地职责」执行；GPU 任务经 **`ssh ATR` 直连远端**执行
> - **远端 AI（GPU 服务器上运行的 AI，可选后备）**：仅当本地 AI 离线时按交接文档执行并回报；环境见 `ai_handoff/02_env_setup.md`
>
> 沟通载体 = **GitHub 仓库 + `ai_handoff/` 交接文档**；本地与远端之间靠 SSH/SCP 直连，用户只负责实例开关与大文件物理搬运。

## 项目一句话

Qwen3-VL-8B 工具调用 + **ATR 奖励**（`R = acc + λU − γC + ηS`）的智能体视觉 RL。
流程：397B 高质量轨迹 → SFT 冷启动（两轮已完成，强制 34% / 先答后验 62%）→ **GRPO RL**（verl，当前阶段）。
RL 集成已完成（VStarToolEnv + patch_reward + vstar→verl parquet + compute_score，冒烟 34/34 通过），交接文档 `ai_handoff/05_rl_plan.md` 已写好，等远端执行。

## 双端分工

| 端 | 机器 | 做什么 | 绝不做什么 |
|----|------|--------|-----------|
| 本地 | Windows 开发机（无 GPU） | 写代码；改 tools/reward/adapter；**CPU 冒烟测试**（`experiments/scripts/smoke_*.py`）；SFT 数据过滤/检查；vstar→verl parquet 转换（纯 CPU）；写 `ai_handoff/` 交接文档；git 提交推送；**SSH 直连远端（`ssh ATR`，密钥免密）**：同步代码、启动/监控训练、scp 数据、读日志 | **本机一切 GPU 负载**（`run_atr_offline.py`、vLLM、verl、torchrun、deepspeed、`nvidia-smi` 等；`.claude/settings.local.json` 已配置 deny 拦截，不要绕过）。GPU 命令只能经 `ssh ATR` 发往远端执行 |
| 远端 | GPU 服务器（SeetaCloud，`ssh ATR` 可达） | 执行本地经 SSH 下发的命令：SFT/RL 训练、vLLM 服务、评测、模型合并导出；长任务 nohup/screen 分离自跑 | GPU 开关由用户控制台操作；关机≠释放，权重回传本地验证前绝不释放 |
| 远端 AI | GPU 服务器上运行的 AI | **可选后备**：仅当本地 AI 离线/无网络时，按交接文档执行并回报 | 不再承担日常执行角色 |

## 协作流程（本地 SSH 直连远端）

1. 本地写代码 + CPU 冒烟 → commit + push
2. `ssh ATR 'cd /root/code && git pull'` 同步代码；不进 git 的数据（图片/parquet/ckpt）用 scp/rsync 搬运
3. 本地经 SSH 启动训练（**必须 nohup/screen 分离**：本地 Bash 单命令 ≤10 分钟，长任务靠远端分离自跑）→ 轮询日志/指标 → 回改
4. 关键结论仍写 `ai_handoff/` 回报节 + `knowledge-base/` 归档（保留记录习惯）

当前阶段 = `ai_handoff/` 最新编号文档；`knowledge-base/` 是方案事实源（`SFT_TRAIN_README.md` 训练手册、`TOOL_SUMMARY.md` 工具接口、`REFERENCES.md` 引用映射）。

## SSH 远端速查

- 连接：`ssh ATR`（`~/.ssh/config` Host ATR → connect.nmb2.seetacloud.com:25419，密钥认证）
- 远端环境：conda env `atr`（`/root/miniconda3/envs/atr`）、工作目录 `/root/code`、数据盘 `/root/autodl-tmp`
- **远端任何外网操作（git/pip/HF）前先 `source /etc/network_turbo`**（只加速 github/huggingface，会拖慢其他站点）
- 远端有 nohup/screen/rsync，**无 tmux**
- GPU 开机/无卡模式切换由用户在控制台操作；无卡模式仍可 SSH 传文件、改代码、看盘

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

- 远端：conda env `atr`，工作目录 `/root/code`，SSH 直连 `ssh ATR`（详见 `ai_handoff/02_env_setup.md`）
- 模型：Qwen3-VL-8B（SFT 后 `Qwen3-VL-8B-ATR-SFT`）
- 数据：VStar 基准原始图 338 MB，走 `atr_project_bundle*.tar.gz` 数据包 / 网盘同步，不进 git
