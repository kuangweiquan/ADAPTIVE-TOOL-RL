"""ToolTrace — 工具调用轨迹记录(ATR reward 直接消费的数据格式)。

记录格式与旧 ToolEnv._record 逐字节一致:
    {"tool_name": ..., "arguments": ..., "output": ...[, "bbox": ...]}
"""

from typing import Any, Dict, List, Optional


class ToolTrace:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(
        self,
        tool_name: str,
        arguments: dict,
        output: str,
        bbox: Optional[list] = None,
    ) -> None:
        """记录一次工具调用。bbox 保留 `if bbox:` 真值语义(与旧实现一致)。"""
        record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "output": output,
        }
        if bbox:
            record["bbox"] = bbox
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)
