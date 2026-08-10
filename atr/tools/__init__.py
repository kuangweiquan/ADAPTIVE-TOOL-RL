"""atr.tools — 工具定义的单一真值源。

用法:
    from atr.tools import registry, get_tool_schemas, execute, ToolTrace

    schemas = get_tool_schemas()          # → prompt <tools> 块的 JSON 行
    result = execute("zoom", {"bbox_2d": [...]}, image)   # → ToolResult
    trace.record(result.canonical_name, result.arguments, result.output, result.bbox)
"""

from typing import Dict, Any
from PIL import Image

from .base import VisualTool, ToolRegistry, ToolResult, registry
from .image_tools import CropTool, ZoomTool, RotateTool, OCRTool
from .trace import ToolTrace

# 注册顺序 = prompt <tools> 顺序
registry.register(CropTool)
registry.register(ZoomTool)
registry.register(RotateTool)
registry.register(OCRTool)


def get_tool_schemas() -> list:
    """全部工具的 function-calling JSON schema(注册顺序)。"""
    return registry.schemas()


def execute(tool_name: str, arguments: Dict[str, Any], image: Image.Image) -> ToolResult:
    """按名(规范名或 alias)执行工具。未知工具 raise KeyError。

    参数先经 normalize_arguments 归一化(alias → 规范键、丢弃未知键)。
    """
    cls = registry.get(tool_name)
    if cls is None:
        raise KeyError(f"Unknown tool: {tool_name}")
    return cls().run(image, cls.normalize_arguments(arguments))


__all__ = [
    "registry",
    "get_tool_schemas",
    "execute",
    "VisualTool",
    "ToolRegistry",
    "ToolResult",
    "ToolTrace",
]
