import os
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple

class ColorManager:
    """Handles professional color conversions, CMYK mapping, TAC control, and pytoshop raw inversion."""

    @staticmethod
    def bgr_to_cmyk_raw(bgr_image: np.ndarray, icc_path: Optional[str] = None) -> np.ndarray:
        """BGR → pytoshop 磁盘反码 CMYK (4, H, W)。

        约定：pytoshop raw CMYK 中 255 = 0% 墨（空白纸），0 = 100% 墨。

        色彩转换路径（2026-09-10 增补）：
          - 提供 `icc_path` 且文件存在时，走 **ICC 驱动的转换**（ImageCms，相对比色意图），
            这是印前应有做法：色彩由目标印刷条件决定，而非设备无关的朴素公式；
          - 否则回退 PIL 的朴素 `convert('CMYK')`，并在调用方 manifest 中如实标注
            "无色彩管理"（该回退仅用于开发/验证，不应用于正式交付）。

        注意：ICC profile **不含 TAC 上限**，TAC 属印刷工艺参数，
        见 `engine/core/ink_limiter.py` 的说明。
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if icc_path and os.path.isfile(icc_path):
            cmyk_np = ColorManager._rgb_to_cmyk_icc(rgb, icc_path)
        else:
            cmyk_np = np.array(Image.fromarray(rgb).convert('CMYK'))  # (H,W,4) 0..255 墨量

        # 墨量 → 磁盘反码
        return (255 - cmyk_np).transpose(2, 0, 1).astype(np.uint8)

    # ICC transform 缓存：buildTransform 是重操作，而生产链路会对
    # Section 5 + 背景层 + 每个图层分别做 CMYK 转换（实测 13 次）。
    # 若不缓存，Step 6 写盘耗时从 66.9s 暴增到 217.5s。
    _TRANSFORM_CACHE: dict = {}

    @staticmethod
    def _get_icc_transform(icc_path: str):
        """获取（并缓存）sRGB → 目标 CMYK 的 ImageCms transform。"""
        from PIL import ImageCms

        key = os.path.abspath(icc_path)
        xform = ColorManager._TRANSFORM_CACHE.get(key)
        if xform is None:
            dst = ImageCms.getOpenProfile(key)
            src = ImageCms.createProfile("sRGB")
            xform = ImageCms.buildTransform(
                src, dst, "RGB", "CMYK",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
            ColorManager._TRANSFORM_CACHE[key] = xform
        return xform

    @staticmethod
    def clear_icc_cache() -> None:
        """清空 transform 缓存（切换 ICC / 测试时用）。"""
        ColorManager._TRANSFORM_CACHE.clear()

    @staticmethod
    def _rgb_to_cmyk_icc(rgb: np.ndarray, icc_path: str) -> np.ndarray:
        """用 ICC profile 做 sRGB → 目标 CMYK 的转换，返回 (H, W, 4) 墨量 0..255。"""
        from PIL import ImageCms

        xform = ColorManager._get_icc_transform(icc_path)
        out = ImageCms.applyTransform(Image.fromarray(rgb), xform)
        arr = np.array(out)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f"ICC 转换未返回 4 通道 CMYK：{arr.shape}")
        return arr.astype(np.uint8)

    @staticmethod
    def get_ivory_substrate_cmyk_raw(height: int, width: int,
                                     c_pct: float = 2.0, m_pct: float = 3.0,
                                     y_pct: float = 8.0, k_pct: float = 0.0) -> np.ndarray:
        """
        Generates calibrated luxury ivory base substrate channels in pytoshop raw format.
        """
        raw_c = int(round(255 - (c_pct / 100.0) * 255))
        raw_m = int(round(255 - (m_pct / 100.0) * 255))
        raw_y = int(round(255 - (y_pct / 100.0) * 255))
        raw_k = int(round(255 - (k_pct / 100.0) * 255))

        c = np.full((height, width), raw_c, dtype=np.uint8)
        m = np.full((height, width), raw_m, dtype=np.uint8)
        y = np.full((height, width), raw_y, dtype=np.uint8)
        k = np.full((height, width), raw_k, dtype=np.uint8)

        return np.stack([c, m, y, k], axis=0)

    @staticmethod
    def cmyk_raw_to_bgr_preview(cmyk_raw_channels: np.ndarray, max_dim: int = 1600) -> np.ndarray:
        """
        Converts pytoshop raw CMYK channels (4, H, W) to a high-quality BGR preview image.
        """
        c, m, y, k = cmyk_raw_channels[0], cmyk_raw_channels[1], cmyk_raw_channels[2], cmyk_raw_channels[3]
        h, w = c.shape

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)
            k = cv2.resize(k, (nw, nh), interpolation=cv2.INTER_AREA)

        ink_c = 255 - c
        ink_m = 255 - m
        ink_y = 255 - y
        ink_k = 255 - k

        cmyk_stack = np.stack([ink_c, ink_m, ink_y, ink_k], axis=2)
        rgb = np.array(Image.fromarray(cmyk_stack, mode='CMYK').convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def calculate_tac(cmyk_raw_channels: np.ndarray) -> Tuple[float, float]:
        """
        Calculates Total Area Coverage (TAC) percentage (0% ~ 400%).
        Returns: (max_tac_pct, mean_tac_pct)
        """
        c, m, y, k = cmyk_raw_channels[0], cmyk_raw_channels[1], cmyk_raw_channels[2], cmyk_raw_channels[3]
        tac = ((255 - c).astype(float) + (255 - m).astype(float) + (255 - y).astype(float) + (255 - k).astype(float)) / 255.0 * 100.0
        return float(np.max(tac)), float(np.mean(tac))
