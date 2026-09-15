"""Unicode 安全的图像读写（engine 级统一入口）。

背景（2026-09-11 壁布品类验证时暴露）：`cv2.imread` / `cv2.imwrite` 在 Windows
上遇到含非 ASCII 字符的路径会**静默失败**——imread 返回 None、imwrite 返回 True
但文件不存在。工作区路径本身含中文（如「PSD图层处理 - WorkBuddy」），
用户从 IDE / 一键 bat 传入绝对路径是常态，因此入口读图必须走本模块。

原理：`np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`，
均以字节流方式处理路径，与 Locale 无关。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，支持含非 ASCII 字符的路径；失败返回 None（与 cv2.imread 语义一致）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


# 支持的扩展名（修复 2026-09-15：此前未知扩展名一律回落 jpg，
# 会把 .tif/.bmp/.webp 等无损/专用格式悄悄编成有损 jpg 字节）
_SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _ext_for(path: str) -> str:
    """根据扩展名选择编码格式；未知扩展名默认 **PNG**（无损，避免有损误编码）。"""
    low = path.lower()
    for e in _SUPPORTED_EXTS:
        if low.endswith(e):
            return ".jpg" if e in (".jpg", ".jpeg") else e
    return ".png"


def imwrite_unicode(path: str, img: np.ndarray,
                    params: Optional[list] = None) -> bool:
    """写出图像，支持含非 ASCII 字符的路径。params 透传 cv2.imencode（如 JPEG 质量）。

    扩展名 → 编码格式：.png/.bmp/.tif/.tiff/.webp 各按自身格式，
    .jpg/.jpeg 用 JPEG；未知扩展名默认 PNG（无损）。
    """
    ext = _ext_for(path)
    ok, buf = cv2.imencode(ext, img, params) if params else cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except OSError:
        return False
