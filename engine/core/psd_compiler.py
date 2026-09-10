"""
v2.2 PSD / PSB 编译器（ADR-002 / ADR-003 / ADR-011 / R6）

约定
----
1. **反码只在这里做一次**：工作数组是逻辑墨量（0 = 0% 墨，255 = 100% 墨），
   PSD 磁盘存反码。`_to_disk()` 是唯一转换点，算子内禁止反复反转。
2. **默认 RLE**（ADR-003）：边长 > 30000 px 或预估 > 1.5 GB 自动切 PSB（`version_2`）。
3. **双产品线**（ADR-011）：PLATE 走 CMYK 四通道 + 冲孔掩模；DESIGN 走 RGB + alpha + bbox（§6.7）。
4. **Section 5 合成图由图层叠加算出**（R6 层合成等价性）：不另算一套，
   保证"PSD 里看到的"与"分层叠加出来的"是同一份像素。
5. 图层名在写入前统一去 `\x00`（pytoshop Pascal 串会补零，V4-05）。
"""

from __future__ import annotations

import os
import struct
import time
from typing import Any, Optional

import numpy as np
import pytoshop
from pytoshop import enums, image_resources
from pytoshop.user import nested_layers

from engine.core.models import LayerDescriptor, ProcessingContext

# ADR-003 切换阈值
PSB_SIDE_LIMIT = 30000
PSB_SIZE_LIMIT_BYTES = 1_500_000_000

# 设计线合成底色
DESIGN_BG = 255


class PsdCompiler:
    """把 ProcessingContext.layers 编译为 PSD / PSB。"""

    # ---------------- 公共入口 ----------------
    @staticmethod
    def compile_psd(
        context: ProcessingContext,
        output_path: str,
        compression: Optional[enums.Compression] = None,
        output_mode: Optional[str] = None,
    ) -> str:
        t0 = time.time()
        mode = (output_mode or getattr(context, "output_mode", "PLATE") or "PLATE").upper()
        if mode not in ("PLATE", "DESIGN"):
            raise ValueError(f"未知 output_mode: {mode}（只允许 PLATE / DESIGN）")

        compression, comp_reason = PsdCompiler.resolve_compression(compression)
        canvas_w, canvas_h = context.canvas_px
        PsdCompiler._assert_layers(context.layers, mode)

        # 1) 图层 → pytoshop 结构
        psd_layers = [
            PsdCompiler._build_image_layer(lyr, mode, canvas_w, canvas_h)
            for lyr in context.layers
        ]

        # 2) 版本（PSD / PSB）与压缩
        version = PsdCompiler.choose_version(canvas_w, canvas_h, len(context.layers))
        color_mode = enums.ColorMode.cmyk if mode == "PLATE" else enums.ColorMode.rgb

        psd = nested_layers.nested_layers_to_psd(
            psd_layers,
            color_mode=color_mode,
            version=version,
            compression=compression,
            size=(canvas_w, canvas_h),
        )

        # 3) Section 5 合成图（R6：由图层叠加推导）
        if mode == "PLATE":
            comp = PsdCompiler._composite_plate(context.layers, canvas_w, canvas_h)
            comp = PsdCompiler._to_disk(comp)
        else:
            comp = PsdCompiler._composite_design(context.layers, canvas_w, canvas_h)
        psd.image_data._channels = np.ascontiguousarray(comp)

        # 4) 分辨率信息块（ID 1005）
        PsdCompiler._write_resolution(psd, float(context.ppi))

        # 5) 落盘
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            psd.write(f)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        context.qa_metrics.update(
            {
                "output_mode": mode,
                "psd_version": "PSB" if version == enums.Version.version_2 else "PSD",
                "psd_compression": str(compression),
                "psd_compression_reason": comp_reason,
                "layer_count": len(context.layers),
                "compile_elapsed_sec": round(elapsed, 3),
                "output_file_size_mb": round(size_mb, 3),
            }
        )
        context.metadata["compile_elapsed_sec"] = elapsed
        context.metadata["output_file_size_mb"] = size_mb
        return output_path

    # ---------------- 压缩决策 ----------------
    @staticmethod
    def rle_supported() -> bool:
        """
        ADR-003 默认 RLE，但 pytoshop 的 RLE 依赖 C 扩展 `pytoshop.packbits`；
        纯 wheel 安装（无 MSVC）时该扩展未编译，`from . import packbits` 会被静默吞掉，
        直到写文件才抛 `NameError: name 'packbits' is not defined`。这里提前探测。
        """
        try:
            from pytoshop import packbits  # noqa: F401

            return hasattr(packbits, "encode")
        except Exception:
            return False

    @staticmethod
    def resolve_compression(
        preferred: Optional[enums.Compression],
    ) -> tuple[int, str]:
        """RLE（首选）→ ZIP（无预测的 zlib 兜底）→ RAW。返回 (compression, reason)。"""
        if preferred is None:
            preferred = enums.Compression.rle
        preferred = enums.Compression(int(preferred))
        if preferred == enums.Compression.rle and not PsdCompiler.rle_supported():
            return int(enums.Compression.zip), "rle_unavailable_packbits_not_built"
        if preferred == enums.Compression.zip_prediction and not PsdCompiler.rle_supported():
            return int(enums.Compression.zip), "zip_prediction_unavailable_packbits_not_built"
        return int(preferred), "as_requested"

    # ---------------- 版本决策 ----------------
    @staticmethod
    def choose_version(width: int, height: int, layer_count: int) -> int:
        """ADR-003：超限自动切 PSB。"""
        if max(width, height) > PSB_SIDE_LIMIT:
            return enums.Version.version_2
        est = int(width) * int(height) * max(1, layer_count) * 0.35  # RLE 经验压缩率
        if est > PSB_SIZE_LIMIT_BYTES:
            return enums.Version.version_2
        return enums.Version.version_1

    # ---------------- 图层构建 ----------------
    @staticmethod
    def _build_image_layer(
        lyr: LayerDescriptor, mode: str, canvas_w: int, canvas_h: int
    ) -> nested_layers.Image:
        name = str(lyr.name).rstrip("\x00")[:255]

        top, bottom, left, right = PsdCompiler._place(lyr, canvas_w, canvas_h)

        if mode == "DESIGN" or lyr.layer_type == "element_rgba":
            ch = PsdCompiler._rgba_channels(lyr, top, bottom, left, right)
        else:
            ch = PsdCompiler._cmyk_channels(lyr, top, bottom, left, right)

        return nested_layers.Image(
            name=name,
            visible=bool(lyr.visible),
            opacity=int(lyr.opacity),
            blend_mode=PsdCompiler._blend_mode(lyr.blend_mode),
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            channels=ch,
        )

    @staticmethod
    def _cmyk_channels(lyr: LayerDescriptor, top: int, bottom: int, left: int, right: int) -> dict:
        if lyr.cmyk_channels is None:
            raise ValueError(f"PLATE 线图层缺少 cmyk_channels：{lyr.name}")
        h, w = bottom - top, right - left
        ch: dict[int, np.ndarray] = {}
        for i in range(4):
            plane = lyr.cmyk_channels[i]
            ch[i] = PsdCompiler._to_disk(PsdCompiler._crop(plane, h, w))
        if lyr.alpha is not None:
            ch[-1] = PsdCompiler._crop(lyr.alpha, h, w).astype(np.uint8)
        return ch

    @staticmethod
    def _rgba_channels(lyr: LayerDescriptor, top: int, bottom: int, left: int, right: int) -> dict:
        h, w = bottom - top, right - left
        if lyr.rgba is None:
            # 无 RGBA 时退化为透明层，保证结构完整
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            rgba = PsdCompiler._fit_hwc(lyr.rgba, h, w)
        ch = {0: rgba[:, :, 0], 1: rgba[:, :, 1], 2: rgba[:, :, 2]}
        ch[-1] = rgba[:, :, 3] if rgba.shape[2] > 3 else np.full((h, w), 255, np.uint8)
        return {k: np.ascontiguousarray(v) for k, v in ch.items()}

    # ---------------- Section 5 合成 ----------------
    @staticmethod
    def _composite_plate(layers: list[LayerDescriptor], w: int, h: int) -> np.ndarray:
        """逻辑墨量叠加：0 = 白纸。冲孔层以 alpha==0 标记打穿。"""
        comp = np.zeros((4, h, w), dtype=np.uint8)
        substrate = next((l for l in layers if l.layer_type == "substrate_cmyk"), None)
        if substrate is not None and substrate.cmyk_channels is not None:
            comp[:] = PsdCompiler._crop_fit(substrate.cmyk_channels, h, w)

        for lyr in layers:
            if not lyr.visible or lyr.cmyk_channels is None:
                continue
            if lyr.layer_type == "diecut_mask":
                continue
            top, bottom, left, right = PsdCompiler._place(lyr, w, h)
            vh, vw = bottom - top, right - left
            ink = PsdCompiler._crop_fit(lyr.cmyk_channels, vh, vw)
            if lyr.alpha is None:
                comp[:, top:bottom, left:right] = ink
            else:
                act = PsdCompiler._crop_fit(lyr.alpha, vh, vw) > 0
                if act.any():
                    for i in range(4):
                        comp[i, top:bottom, left:right][act] = ink[i][act]

        # 冲孔：alpha 为 0 的位置回落到承印物（无承印物则为 0 墨量 = 白纸透空）
        for lyr in layers:
            if lyr.layer_type != "diecut_mask" or not lyr.visible or lyr.alpha is None:
                continue
            top, bottom, left, right = PsdCompiler._place(lyr, w, h)
            vh, vw = bottom - top, right - left
            holes = PsdCompiler._crop_fit(lyr.alpha, vh, vw) == 0
            if substrate is not None and substrate.cmyk_channels is not None:
                sub_crop = PsdCompiler._crop_fit(substrate.cmyk_channels, h, w)
                for i in range(4):
                    comp[i, top:bottom, left:right][holes] = sub_crop[i, top:bottom, left:right][holes]
            else:
                comp[:, top:bottom, left:right][:, holes] = 0
        return comp

    @staticmethod
    def _composite_design(layers: list[LayerDescriptor], w: int, h: int) -> np.ndarray:
        """RGB 合成：自底向上 alpha 混合，底色白。"""
        comp = np.full((3, h, w), DESIGN_BG, dtype=np.uint8)
        for lyr in layers:
            if not lyr.visible or lyr.rgba is None:
                continue
            top, bottom, left, right = PsdCompiler._place(lyr, w, h)
            vh, vw = bottom - top, right - left
            rgba = PsdCompiler._fit_hwc(lyr.rgba, vh, vw)
            a = rgba[:, :, 3].astype(np.float32) / 255.0
            dst = comp[:, top:bottom, left:right].astype(np.float32)
            src = rgba[:, :, :3].astype(np.float32)
            for i in range(3):
                dst[i] = dst[i] * (1.0 - a) + src[:, :, i] * a
            comp[:, top:bottom, left:right] = dst.astype(np.uint8)
        return comp

    # ---------------- 辅助 ----------------
    @staticmethod
    def _to_disk(arr: np.ndarray) -> np.ndarray:
        """逻辑墨量 → PSD 磁盘反码。唯一转换点。"""
        return (255 - np.asarray(arr, dtype=np.uint8)).astype(np.uint8)

    @staticmethod
    def _crop(plane: np.ndarray, h: int, w: int) -> np.ndarray:
        arr = np.asarray(plane)
        if arr.shape[:2] == (h, w):
            return arr
        return np.ascontiguousarray(arr[:h, :w])

    @staticmethod
    def _crop_fit(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """裁剪或零填充到 (h, w)，末尾维度不变。"""
        a = np.asarray(arr)
        if a.shape[-2:] == (h, w):
            return a
        if a.ndim == 2:
            out = np.zeros((h, w), dtype=a.dtype)
            out[: min(h, a.shape[0]), : min(w, a.shape[1])] = a[:h, :w]
        else:
            out = np.zeros(a.shape[:-2] + (h, w), dtype=a.dtype)
            out[..., : min(h, a.shape[-2]), : min(w, a.shape[-1])] = a[..., :h, :w]
        return out

    @staticmethod
    def _fit_hwc(rgba: np.ndarray, h: int, w: int) -> np.ndarray:
        """通道在后（H, W, C）的裁剪/零填充，专供设计线 RGBA 使用。"""
        a = np.asarray(rgba)
        if a.shape[:2] == (h, w):
            return a
        c = a.shape[2] if a.ndim == 3 else 1
        out = np.zeros((h, w, c), dtype=a.dtype)
        out[: min(h, a.shape[0]), : min(w, a.shape[1])] = a[:h, :w]
        return out

    @staticmethod
    def _place(lyr: LayerDescriptor, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
        """图层在画布上的落位；无 bbox 时占满画布，越界即裁剪。"""
        if lyr.bbox is None:
            return 0, canvas_h, 0, canvas_w
        left, top, right, bottom = (int(v) for v in lyr.bbox)
        left = max(0, min(left, canvas_w))
        right = max(left, min(right, canvas_w))
        top = max(0, min(top, canvas_h))
        bottom = max(top, min(bottom, canvas_h))
        return top, bottom, left, right

    @staticmethod
    def _alpha_mask(lyr: LayerDescriptor, w: int, h: int) -> Optional[np.ndarray]:
        if lyr.alpha is None:
            return None
        return PsdCompiler._crop(lyr.alpha, h, w) > 0

    @staticmethod
    def _blend_mode(name: str) -> bytes:
        return {
            "normal": enums.BlendMode.normal,
            "multiply": enums.BlendMode.multiply,
            "screen": enums.BlendMode.screen,
        }.get(str(name).lower(), enums.BlendMode.normal)

    @staticmethod
    def _write_resolution(psd: Any, ppi: float) -> None:
        dpi = int(round(ppi))
        fixed = dpi << 16
        data = struct.pack(">IHHIHH", fixed, 1, 2, fixed, 1, 2)
        psd.image_resources.blocks.append(
            image_resources.GenericImageResourceBlock(resource_id=1005, name="", data=data)
        )

    @staticmethod
    def _assert_layers(layers: list[LayerDescriptor], mode: str) -> None:
        if not layers:
            raise ValueError("图层列表为空，拒绝写出 PSD")
        for lyr in layers:
            if mode == "PLATE" and lyr.cmyk_channels is None and lyr.layer_type != "element_rgba":
                raise ValueError(f"PLATE 线图层缺少 cmyk_channels：{lyr.name}")
            if mode == "DESIGN" and lyr.rgba is None:
                raise ValueError(f"DESIGN 线图层缺少 rgba：{lyr.name}")
