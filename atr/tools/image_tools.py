"""视觉工具实现 — crop / zoom / rotate / ocr。

行为与输出文本与旧离线管线(run_atr_offline.py 的 ToolEnv)逐字节一致:
  - bbox clamp 到图像边界
  - 空 bbox 区域 → 无效描述
  - 错误通过 ValueError 抛出,文本与旧 dispatch 的 guard 一致
  - ocr 依赖 pytesseract,惰性探测可用性(模块导入不依赖 tesseract)
"""

from typing import Any, Dict, Optional
from PIL import Image

from .base import VisualTool, ToolResult


def _clamp_bbox(bbox_2d, w: int, h: int):
    """Clamp [x1,y1,x2,y2] 到图像边界,返回 int 列表。"""
    x1, y1, x2, y2 = map(int, bbox_2d)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return [x1, y1, x2, y2]


class CropTool(VisualTool):
    name = "crop"
    description = "Crop a region of the image to examine details."
    parameters = {
        "bbox_2d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "The region as NORMALIZED [x1, y1, x2, y2]: each value in [0,1], "
                           "0,0 = top-left corner of the image, 1,1 = bottom-right.",
        },
    }
    required = ["bbox_2d"]
    spatial = True
    param_aliases = {"bbox": "bbox_2d"}

    def run(self, image: Image.Image, arguments: Dict[str, Any]) -> ToolResult:
        bbox_2d = arguments.get("bbox_2d")
        if not bbox_2d:
            raise ValueError("crop requires bbox_2d")
        w, h = image.size
        x1, y1, x2, y2 = _clamp_bbox(bbox_2d, w, h)

        if x2 <= x1 or y2 <= y1:
            desc = "[Crop: invalid bbox, region is empty]"
            return ToolResult(output=desc, bbox=[x1, y1, x2, y2],
                              arguments={"bbox_2d": bbox_2d}, canonical_name=self.name)

        # image 返回给模型查看;updates_state=False → 不更新 current_image
        # (轨迹记录格式不变,仍为 {tool_name, arguments, output, bbox})
        cropped = image.crop((x1, y1, x2, y2))
        desc = f"[Cropped region ({x1},{y1})-({x2},{y2}), size {x2-x1}×{y2-y1}]"
        return ToolResult(image=cropped, output=desc, bbox=[x1, y1, x2, y2],
                          arguments={"bbox_2d": bbox_2d}, canonical_name=self.name)


class ZoomTool(VisualTool):
    name = "zoom"
    aliases = ("zoom_in",)
    description = "Zoom into a region to see fine details."
    parameters = {
        "bbox_2d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "The region to zoom into as NORMALIZED [x1, y1, x2, y2]: "
                           "each value in [0,1], 0,0 = top-left, 1,1 = bottom-right.",
        },
    }
    required = ["bbox_2d"]
    updates_state = True
    spatial = True
    param_aliases = {"bbox": "bbox_2d"}

    def run(self, image: Image.Image, arguments: Dict[str, Any]) -> ToolResult:
        bbox_2d = arguments.get("bbox_2d")
        if not bbox_2d:
            raise ValueError("zoom_in requires bbox_2d")
        w, h = image.size
        x1, y1, x2, y2 = _clamp_bbox(bbox_2d, w, h)

        if x2 <= x1 or y2 <= y1:
            desc = "[ZoomIn: invalid bbox]"
            return ToolResult(output=desc, bbox=[x1, y1, x2, y2],
                              arguments={"bbox_2d": bbox_2d}, canonical_name=self.name)

        cropped = image.crop((x1, y1, x2, y2))
        zoomed = cropped.resize((w, h), Image.BICUBIC)
        desc = f"[Zoomed into ({x1},{y1})-({x2},{y2})]"
        return ToolResult(image=zoomed, output=desc, bbox=[x1, y1, x2, y2],
                          arguments={"bbox_2d": bbox_2d}, canonical_name=self.name)


class RotateTool(VisualTool):
    name = "rotate"
    description = "Rotate the image by a specified angle."
    parameters = {
        "angle": {
            "type": "integer",
            "description": "Rotation angle in degrees (e.g., 90, 180, 270). "
                           "Positive values rotate counter-clockwise.",
        },
    }
    required = ["angle"]
    updates_state = True
    # 注意:不记 bbox — 全图 bbox 面积比 1.0 会触发 utility 的 lazy-crop 惩罚
    spatial = False

    def run(self, image: Image.Image, arguments: Dict[str, Any]) -> ToolResult:
        angle = arguments.get("angle")
        if angle is None:
            raise ValueError("rotate requires angle")
        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        desc = f"[Rotated by {angle} degrees]"
        return ToolResult(image=rotated, output=desc,
                          arguments={"angle": angle}, canonical_name=self.name)


# ------------------------------------------------------------------
# OCR — pytesseract 惰性探测
# ------------------------------------------------------------------

_OCR_AVAILABLE = None  # None=未探测;True/False=已缓存


def _check_ocr() -> bool:
    global _OCR_AVAILABLE
    if _OCR_AVAILABLE is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _OCR_AVAILABLE = True
        except Exception:
            _OCR_AVAILABLE = False
    return _OCR_AVAILABLE


class OCRTool(VisualTool):
    name = "ocr"
    description = "Extract text from a region of the image."
    parameters = {
        "bbox_2d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "The region to read as NORMALIZED [x1, y1, x2, y2]: "
                           "each value in [0,1], 0,0 = top-left, 1,1 = bottom-right. "
                           "Omit to OCR the whole image.",
        },
    }
    required = []
    param_aliases = {"bbox": "bbox_2d"}

    def run(self, image: Image.Image, arguments: Dict[str, Any]) -> ToolResult:
        bbox_2d = arguments.get("bbox_2d")
        if bbox_2d:
            w, h = image.size
            x1, y1, x2, y2 = _clamp_bbox(bbox_2d, w, h)
            target_img = image.crop((x1, y1, x2, y2))
            args = {"bbox_2d": bbox_2d}
            bbox = [x1, y1, x2, y2]
        else:
            # 与旧行为一致:无 bbox 时全文 OCR,arguments/bbox 记为全图
            target_img = image
            w, h = image.size
            args = {"bbox_2d": [0, 0, w, h]}
            bbox = [0, 0, w, h]

        text = ""
        if _check_ocr():
            try:
                import pytesseract
                text = pytesseract.image_to_string(target_img).strip()
            except Exception:
                text = ""

        if not text:
            output = "[No text detected in region]"
        else:
            output = f'[OCR result: "{text}"]'

        return ToolResult(output=output, bbox=bbox, arguments=args,
                          canonical_name=self.name)
