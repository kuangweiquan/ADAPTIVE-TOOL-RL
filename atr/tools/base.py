"""Tool 基类与注册表 — 工具定义的单一真值源 (Single Source of Truth).

所有工具的名称、描述、参数 schema 与执行逻辑集中于此，
离线管线的 prompt 生成、工具分发、轨迹记录与奖励层的工具名校验
全部由注册表驱动。

风格参考:CodeV 的 tool library 思想(聊天记录 gpt-chat-history5)，
但调用方式保持本项目现有的 `<tool_call>` JSON action 格式。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


@dataclass
class ToolResult:
    """一次工具执行的结果。

    image:   仅当工具更新图像状态时非 None(zoom/rotate)，调用方应
             将 current_image 更新为它。
    output:  返回给模型的文本描述(与旧 ToolEnv 输出格式逐字节一致)。
    bbox:    空间工具的 clamp 后 bbox [x1,y1,x2,y2];非空间工具为 None。
    arguments: 归一化后的参数(alias 已解析、未知键已丢弃)。
    canonical_name: 规范工具名(alias 已归一化)，用于轨迹记录。
    """
    image: Optional[Image.Image] = None
    output: str = ""
    bbox: Optional[List[float]] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    canonical_name: str = ""


class VisualTool:
    """视觉工具基类。子类覆盖类属性与 run() 即可注册使用。

    类属性:
        name:           规范名，如 "zoom"
        aliases:        兼容旧名，如 ("zoom_in",)
        description:    工具描述(进 prompt schema)
        parameters:     JSON schema "properties"
        required:       JSON schema "required"
        updates_state:  是否更新图像状态(zoom/rotate → True)
        spatial:        是否为空间工具(参与 IoU / lazy-crop 判定)
        param_aliases:  参数别名映射，如 {"bbox": "bbox_2d"}
    """

    name: str = ""
    aliases: Tuple[str, ...] = ()
    description: str = ""
    parameters: Dict[str, Any] = {}
    required: List[str] = []
    updates_state: bool = False
    spatial: bool = False
    param_aliases: Dict[str, str] = {}

    def run(self, image: Image.Image, arguments: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Schema / 参数工具方法
    # ------------------------------------------------------------------

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """function-calling JSON schema — 即 prompt `<tools>` 块里的一行 JSON。"""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": cls.parameters,
                    "required": cls.required,
                },
            },
        }

    @classmethod
    def normalize_arguments(cls, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """参数别名解析为规范键;丢弃 schema 之外的未知键。"""
        out: Dict[str, Any] = {}
        for k, v in (arguments or {}).items():
            key = cls.param_aliases.get(k, k)
            if key in cls.parameters:
                out[key] = v
        return out


class ToolRegistry:
    """工具注册表:按规范名 + alias 索引。"""

    def __init__(self) -> None:
        self._tools: Dict[str, type] = {}          # canonical name → class
        self._alias_map: Dict[str, str] = {}       # alias → canonical
        self._order: List[str] = []                # 注册顺序(即 prompt 顺序)

    def register(self, tool_cls: type) -> None:
        if tool_cls.name in self._tools:
            raise ValueError(f"Tool already registered: {tool_cls.name}")
        self._tools[tool_cls.name] = tool_cls
        self._order.append(tool_cls.name)
        for alias in tool_cls.aliases:
            if alias in self._alias_map:
                raise ValueError(f"Alias conflict for {alias!r}")
            self._alias_map[alias] = tool_cls.name

    def get(self, name: str) -> Optional[type]:
        """按规范名或 alias 查找工具类;未知返回 None。"""
        return self._tools.get(name) or self._tools.get(self._alias_map.get(name, ""))

    def canonical(self, name: str) -> Optional[str]:
        """任意名 → 规范名;未知返回 None。"""
        if name in self._tools:
            return name
        return self._alias_map.get(name)

    # ------------------------------------------------------------------
    # 供奖励层使用的派生属性(单一真值源)
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> List[str]:
        """规范工具名列表(注册顺序)。"""
        return list(self._order)

    @property
    def all_names(self) -> set:
        """规范名 + 所有 alias。"""
        return set(self._order) | set(self._alias_map.keys())

    @property
    def spatial_tools(self) -> frozenset:
        """空间工具名集合(参与 IoU / lazy-crop 判定)。"""
        return frozenset(t.name for t in self._tools.values() if t.spatial)

    def param_names(self, name: str) -> set:
        """某工具规范参数的键集合。"""
        cls = self._tools.get(name)
        return set(cls.parameters.keys()) if cls else set()

    def param_aliases(self, name: str) -> Dict[str, str]:
        cls = self._tools.get(name)
        return dict(cls.param_aliases) if cls else {}

    # ------------------------------------------------------------------
    # Prompt 生成
    # ------------------------------------------------------------------

    def schemas(self) -> List[Dict[str, Any]]:
        """全部工具的 function schema(注册顺序)。"""
        return [self._tools[n].schema() for n in self._order]

    def schema(self, name: str) -> Dict[str, Any]:
        cls = self._tools.get(self.canonical(name) or "")
        if cls is None:
            raise KeyError(f"Unknown tool: {name}")
        return cls.schema()


# 全局唯一注册表(atr.tools 包级单例)
registry = ToolRegistry()
