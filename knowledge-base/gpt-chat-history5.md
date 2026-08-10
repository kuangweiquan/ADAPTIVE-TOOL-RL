CodeV 的核心思想是：**Tool 不是普通 Python 函数注册，而是让 VLM 生成可执行 Python code，通过代码调用视觉操作工具**。也就是说：

```
VLM Agent
   |
   | generate python code
   ↓
Python Sandbox
   |
   | execute image operations
   ↓
Tool Output Image
   |
   ↓
VLM继续reasoning
```

CodeV 论文中强调的是 **code-based visual agent**，工具调用不是固定 JSON action，而是 executable Python code；TAPO 再针对每一步 tool input/output 做 reward。([arXiv][1])

例如模型生成：

```python
crop = image.crop((100,50,300,200))
display(crop)
```

而不是：

```json
{
 "tool":"crop",
 "bbox":[100,50,300,200]
}
```

---

## 1. CodeV 风格 Tool 基础实现

建议你的 ATR 也采用这种结构：

```
tools/
├── base.py
├── image_tool.py
├── sandbox.py
└── registry.py
```

---

## base.py

定义所有视觉工具：

```python
from abc import ABC, abstractmethod


class VisualTool(ABC):

    name = None

    description = None


    @abstractmethod
    def run(self, *args, **kwargs):
        pass
```

---

## 2. Image Tool 实现

例如 CodeV 最基础的 crop：

### tools/image_tool.py

```python
from PIL import Image
from .base import VisualTool


class CropTool(VisualTool):

    name = "crop"

    description = """
    Crop a region from image.

    Args:
        image: PIL image
        box:
        [left, upper, right, lower]

    Return:
        cropped image
    """


    def run(
        self,
        image,
        box
    ):

        crop_img = image.crop(
            tuple(box)
        )

        return crop_img
```

---

Zoom：

```python
class ZoomTool(VisualTool):

    name="zoom"


    description="""
    Zoom image region.
    """


    def run(
        self,
        image,
        box,
        scale=2
    ):

        region=image.crop(
            tuple(box)
        )


        w,h=region.size


        return region.resize(
            (
            w*scale,
            h*scale
            )
        )
```

---

## 3. Tool Registry

类似 CodeV 的 tool library：

```python
from .image_tool import (
    CropTool,
    ZoomTool
)


TOOLS={

"crop":
CropTool(),

"zoom":
ZoomTool()

}
```

---

## 4. 最关键：Code Executor

CodeV 的关键不是 tool registry，而是：

> LLM 生成代码，然后 sandbox 执行

所以需要：

```
generated code
       |
       ↓
executor
       |
       ↓
tool result
```

例如：

```python
import ast


class CodeExecutor:


    def __init__(
        self,
        tools
    ):

        self.tools=tools



    def execute(
        self,
        code,
        image
    ):


        env={

            "image":image

        }


        for name,tool in self.tools.items():

            env[name]=tool.run


        exec(
            code,
            {},
            env
        )


        return env
```

---

模型输出：

```python
result = crop(
    image,
    [100,100,400,400]
)

```

executor：

```python
executor.execute(
code,
image
)
```

执行：

```python
crop()
```

得到：

```
result=image region
```

---

# 5. Agent Loop

完整 CodeV-style loop：

```python
while True:


    response=model.generate(
        prompt,
        image
    )


    if "<python>" in response:


        code=parse_code(
            response
        )


        output=executor.execute(
            code,
            image
        )


        image=output["result"]


        prompt += """

        Tool result:
        <image>

        Continue reasoning.

        """


    else:

        answer=response
        break
```

---

# 6. 对你的 ATR，需要增加 Reward Hook

因为你的创新是 Adaptive Tool RL。

所以 executor 返回：

```python
{
"tool":"crop",

"input":
{
"box":[100,100,400,400]
},

"output":crop_image,

"cost":0.2
}
```

例如：

```python
class ToolTrace:


    def record(
        self,
        tool,
        args,
        output
    ):


        self.logs.append({

        "tool":tool,

        "args":args,

        "output":output

        })
```

然后 reward：

```python
reward = (

answer_reward

+

lambda*tool_usefulness(trace)

-

gamma*tool_cost(trace)

)
```

---

# 7. 如果迁移到你的工业视频任务

CodeV:

```
crop()
zoom()
rotate()
```

你可以扩展：

```
video_tools/

frame_search()

temporal_crop()

track_object()

ocr()

detect PPE()

compare_state()

```

例如：

Agent生成：

```python
frames = temporal_search(
    video,
    query="worker wearing helmet"
)


crop = crop_person(
    frames[20]
)


result = helmet_detector(
    crop
)
```

trace：

```
temporal_search
       |
       |
crop_person
       |
       |
helmet_detector
```

reward：

```
+ 找到正确证据
+ 工具链合理

- 无意义frame search
- 多余crop
```

这和 CodeV 的 TAPO 思路一致。([arXiv][1])

---

## 对你的 ADAPTIVE-TOOL-RL，我建议不要完全复制 CodeV

CodeV 的：

```
Python code generation
```

比较适合通用视觉推理。

你的工业 SOP 审查更适合：

```
Hybrid:

固定domain tools
+
code executor

```

例如：

```
Agent
 |
 | choose
 |
 +---- frame_search()
 |
 +---- crop()
 |
 +---- detector()
 |
 +---- verifier()
```

原因：

工业场景工具空间有限，完全自由生成 Python 安全性和训练成本都更高。

如果目标是 CCF-B，推荐实现：

**Tool Registry + Sandbox Executor + Tool Trace + Adaptive Reward**

这比单纯增加 reward 更像一篇完整 Vision Agent RL 工作。

[1]: https://arxiv.org/abs/2511.19661?utm_source=chatgpt.com "CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization"
