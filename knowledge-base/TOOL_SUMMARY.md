# Tool 实现方式与实验问题总结

> 2026-08 实验记录。从工具定义重构到 8B→397B 模型选型的完整过程。

---

## 一、Tool 实现方式

### 1.1 架构:atr/tools/ 单一真值源

工具定义集中在一个注册表模块,离线管线的 prompt 生成、工具分发、轨迹记录与奖励层的工具名校验全部由它驱动:

```
atr/tools/
├── base.py          # VisualTool ABC(name/aliases/schema/spatial/updates_state)
│                    # + ToolResult + ToolRegistry(规范名/alias 索引)
├── image_tools.py   # CropTool / ZoomTool / RotateTool / OCRTool(真实实现)
├── trace.py         # ToolTrace → {tool_name, arguments, output, bbox?}(ATR 直接消费)
└── __init__.py      # registry / get_tool_schemas() / execute()
```

**四个真实工具**(全部有真实执行,无 mock):

| 工具 | 参数 | 行为 | 输出 |
|------|------|------|------|
| `crop` | `bbox_2d` | 裁剪区域,裁剪图回显给模型,不更新状态 | `[Cropped region (x1,y1)-(x2,y2), size W×H]` |
| `zoom` | `bbox_2d` | 裁剪 + BICUBIC 缩放回视图尺寸,更新状态 | `[Zoomed into (x1,y1)-(x2,y2)]` |
| `rotate` | `angle` | PIL 旋转(expand=False,画布不变) | `[Rotated by N degrees]` |
| `ocr` | `bbox_2d`? | pytesseract(conda-forge 安装),模块级惰性探测 | `[OCR result: "..."]` / `[No text detected...]` |

### 1.2 奖励层集成(单一真值源)

- `sequence.py`:`valid_tools`/`valid_params` 改为从注册表派生(`registry.canonical` + `VALID_PARAMS`),0-3 层次打分保留;未注册工具名 → 1 分
- `utility.py` / `cost.py`:硬编码 `("crop","zoom","select")` → `registry.spatial_tools`
- `AdaptiveToolReward.compute()` 接口不变(项目原则)
- 轨迹记录保持 `{tool_name, arguments, output, bbox?}` 格式,与旧数据字节兼容

### 1.3 坐标接口演进(三轮迭代)

1. **原始(像素坐标,无锚点)**:模型输出原图像素坐标,但只看到 768px 缩放图 → 208/243 次调用超出视图,全乱定位
2. **显示空间锚定**:prompt 注入"当前视图尺寸"+ ToolEnv 自动缩放 → 模型不遵循(仍输出原图尺度坐标)
3. **归一化坐标(最终)**:模型输出 `[0,1]` 归一化 bbox,ToolEnv 换算到执行空间(像素),记录保持像素与 GT 兼容 → 孤立测试中心距离 0.009(几乎正中)

### 1.4 策略双模式(CLI 开关 `--tool_required`)

- **默认(先答后验)**:先直接作答,工具仅"看不清时核实"
- **强制工具模式**:必须先调用工具核实再作答(采集工具轨迹)

### 1.5 清理的假实现/有名无实工具

- 移除 `select`(纯回显 mock,139 次轨迹调用评价失真)
- 移除 `read_frame`/`extract_frames`/`zoom_out`/`search`(只有名字,无实现)

---

## 二、实验遇到的问题(时间线)

### 2.1 问题一:工具定义三层脱节

执行层(pyvision-rl)、离线管线(ToolEnv)、奖励层(硬编码字符串)三套定义互不相通。select 是回显 mock 却支撑 139 次调用的效用/成本评价;4 个工具只有名字。

**解决**:atr/tools/ 注册表统一(见 1.1)。回归验证:旧轨迹重跑 utility/cost 不变(Δ=0.0000),sequence 按预期下降 0.0188。

### 2.2 问题二:8B 模型工具循环系统性失败(核心问题)

三个消融实验(同 100 样本):

| 运行 | 策略 | 正确率 | 工具行为 | 工具-GT IoU |
|------|------|--------|---------|-------------|
| ①强制工具+像素坐标 | 工具必调 | 31%(z=-0.76,≈随机) | 288 次,79% 样本 | 0.000 |
| ②先答后验+归一化 | 先答,工具可选 | 38%(z=+0.42) | 0 次 | — |
| ③强制工具+归一化 | 工具必调 | 6.6% | 371 次 zoom,44/61 死循环 | 0.000 |

**孤立能力测试与循环内行为的巨大反差**:

| 能力 | 孤立测试 | 工具循环内 |
|------|---------|-----------|
| 看图描述 | ✅ 准确 | — |
| 直接 VQA | ✅ 5/5 | 38%(③模式下 6.6%) |
| 归一化定位 | ✅ 中心距离 0.009 | ❌ IoU=0.000 |

**诊断**:Qwen3-VL-8B 一进工具循环就系统性退化——必调工具(5/5)、不定位、不收敛。排查过的候选原因:坐标空间错位(部分成立,但修正无效)、prompt 复杂度(消融排除)、视觉通道(排除,描述准确)、多图传输(排除,crop 图能看见)、模型能力(确认)。**工具循环是 8B 模型 agentic 能力的硬限制。**

### 2.3 问题三:策略纠偏的副作用

"先答后验"把准确率从随机救回 38-41%,但模型对 VStar 所有样本都自信直接答,**工具调用归零**——没有工具轨迹,ATR 无从评估。

### 2.4 问题四:强制工具的崩溃形态

强制模式下 8B 出现 **zoom 死循环**(44/61 样本卡满 8 轮上限不作答)、准确率掉到 6.6%——比随机还低。

### 2.5 问题五:模型选择

- `Qwen/Qwen3.5-9B` 是**纯文本模型**(ID 无 -VL),视觉测试全空
- SiliconFlow 无 Qwen2.5-VL(训练目标模型需本地 vLLM 测试)
- **Qwen3.5-397B-A17B**(用户确认平台标注有视觉能力)实测通过:

**最终结果(20 条,强制工具 + 归一化)**:

| 指标 | 8B(对照) | 397B(最终) |
|------|---------|-----------|
| 正确率 | 6.6% | **80%**(随机期望 33.8%,z=+3.56) |
| 工具调用 | 死循环 | **100% 样本,41 次全部真实(PASS)** |
| 收敛性 | 44/61 卡死 | 17/20 一次 zoom 即作答 |

### 2.6 遗留问题

1. **397B 工具-GT IoU 仍低**(均值 0.04):zoom 大致区域有重叠但精度不足;答对率 80% 说明工具+整图信息结合够用,但"精确定位"仍是短板
2. **3/20 样本卡死**:难题上模型反复 zoom 不收敛
3. **RL 训练可行性(未解决)**:训练目标是 7B 小模型,但小模型零样本跑不动工具循环

---

## 三、结论与建议

### 已验证的事实

1. 工具定义架构(atr/tools/ 注册表 + 归一化坐标 + 真实实现)正确,通过冒烟测试、旧数据回归、真实调用验证
2. 8B 零样本工具循环失败是模型能力问题,不是配置问题;397B 在相同接口下成功
3. VStar 的颜色/属性题整图可答,本身不适合作为"工具使用"研究基准


### 环境备忘

- conda env `atr`(Python 3.11):tesseract 5.5.2 + pytesseract,需 `PATH` 含 `Library/bin`、`TESSDATA_PREFIX` 指向 `share/tessdata`
- 模型切换:`ATR_MODEL` 环境变量(如 `Qwen/Qwen3.5-397B-A17B`)
- 运行:离线管线 `run_atr_offline.py --quick N [--tool_required]`;验证 `verify_real_tool_calls.py`;冒烟 `smoke_tool_refactor.py`;bbox 截取 `crop_vstar_bboxes.py`;10 条核对 `select_trajectories_for_check.py`
