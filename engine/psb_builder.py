"""引擎 → 内核 的数据适配层（G1 内核合流，2026-09-10）。

合流前的状态
------------
`engine/psb_builder.py` 直接操作 pytoshop 写盘，`engine/core/psd_compiler.py`
从未被生产引用——所谓"内核"实际是摆设（且一跑就崩，见尽调报告 §十）。

合流后的职责划分
----------------
- **本类**：把引擎的 dict 层结构（生产语义：mask + 颜色源 + 混合模式）
  转换为内核的 `LayerDescriptor`（逻辑墨量 / RGBA + bbox），并准备 Section 5；
- **core.PsdCompiler**：唯一的 PSD/PSB 写盘方——通道构造、反码转换、
  版本与压缩决策、分辨率块、图层组装全部由内核负责（G1 达成）。

反码约定（易错点，务必区分）
----------------------------
- `ColorManager.bgr_to_cmyk_raw()` 返回 **PSD 磁盘反码**（255 = 0% 墨），
  TAC 压制（`ink_limiter.limit_ink`）在该空间进行；
- `LayerDescriptor.cmyk_channels` 是**逻辑墨量**（0 = 0% 墨），
  编译器在 `_to_disk()` 统一反转。
⇒ 适配层必须把压制后的反码再转回逻辑墨量：`logical = 255 - raw`。
"""

from __future__ import annotations

import cv2
import numpy as np
from pytoshop import enums

from engine.codecs_accelerator import install_psb_codec_accelerator
from engine.core.color_manager import ColorManager
from engine.core.ink_limiter import INK_MAX, limit_ink
from engine.core.models import LayerDescriptor, ProcessingContext
from engine.core.psd_compiler import PsdCompiler

install_psb_codec_accelerator()


class UniversalPSBBuilder:
    """引擎数据 → 内核 LayerDescriptor 的适配层（写盘委托 core.PsdCompiler）。"""

    def __init__(self, target_w=16000, target_h=7808, dpi=150.0,
                 compression=enums.Compression.rle, color_mode="rgb",
                 icc_path=None, tac_policy=None):
        install_psb_codec_accelerator()
        self.target_w = target_w
        self.target_h = target_h
        self.dpi = float(dpi)
        self.compression = compression
        self.color_mode = str(color_mode).lower()
        if self.color_mode not in ("rgb", "cmyk"):
            raise ValueError(f"未知 color_mode: {color_mode}（只允许 rgb / cmyk）")
        #: 目标印刷条件的 ICC profile（分色与黑版生成由它驱动）
        self.icc_path = icc_path
        #: TAC 工艺策略（上限来自印刷条件，不是 ICC —— ICC 里没有该字段）
        self.tac_policy = tac_policy
        #: 最近一次写盘的 TAC 审计结果 (max_pct, mean_pct)
        self.last_tac = None
        #: 最近一次写盘的详细油墨统计（供 manifest 披露）
        self.last_ink_stats = None
        #: 内核回填的质检指标
        self.last_qa: dict = {}

    @property
    def is_cmyk(self) -> bool:
        return self.color_mode == "cmyk"

    # ---------------- CMYK 转换 + TAC 压制 ----------------
    def _to_cmyk_limited(self, bgr: np.ndarray) -> np.ndarray:
        """BGR → CMYK 磁盘反码，并执行 TAC 压制。

        为什么每条 CMYK 数据（含 Section 5、背景层、各图层）都要压制：
        TAC 是「最终输出像素」的属性，任何一条超限都会在印厂糊版；
        统一用同一 policy 压制可保证各层墨量一致、可复算。
        """
        raw = ColorManager.bgr_to_cmyk_raw(bgr, icc_path=self.icc_path)
        if self.tac_policy is not None:
            # inplace=True：转换产物用完即弃，可安全原地修改（16K 下省一次 500MB 复制）
            raw, stats = limit_ink(raw, self.tac_policy, inplace=True)
            self.last_ink_stats = stats
        return raw

    def _raw_to_logical(self, raw: np.ndarray) -> np.ndarray:
        """磁盘反码 → 内核逻辑墨量（LayerDescriptor 的约定空间）。"""
        return (INK_MAX - raw.astype(np.int16)).astype(np.uint8)

    # ---------------- 主入口（签名与合流前一致，调用方零改动）----------------
    def build_psb(self, output_path: str, src_hr_bgr: np.ndarray,
                  bg_hr_bgr: np.ndarray, sorted_layers: list,
                  hr_masks_dict: dict, bg_layer_name: str | None = None) -> str:
        ctx = ProcessingContext(ppi=self.dpi,
                                canvas_px=(self.target_w, self.target_h))
        ctx.output_mode = "PLATE" if self.is_cmyk else "DESIGN"

        # 1) Section 5：印前预览的权威像素来自超分结果（不从图层重新叠加）
        if self.is_cmyk:
            raw = self._to_cmyk_limited(src_hr_bgr)                 # 反码
            ctx.section5_planes = self._raw_to_logical(raw)         # → 逻辑墨量
            self.last_tac = ColorManager.calculate_tac(raw)
        else:
            rgb = cv2.cvtColor(src_hr_bgr, cv2.COLOR_BGR2RGB)
            ctx.section5_planes = np.ascontiguousarray(rgb.transpose(2, 0, 1))

        # 2) 底板层（画布最底、全幅）
        ctx.layers.append(self._make_layer(
            name=bg_layer_name or "02_纯净金箔大底板_Gold_Base_Clean",
            color_bgr=bg_hr_bgr,
            mask=np.full((self.target_h, self.target_w), 255, dtype=np.uint8),
            full_canvas=True,
        ))

        # 3) 内容图层（紧凑 bbox）
        for layer_info in sorted_layers:
            name = str(layer_info.get("name", "")).strip()
            if not name:
                continue
            mask = hr_masks_dict.get(name)
            if mask is None:
                continue
            # 颜色源：补全层优先用 inpainted 内容，否则源超分图
            if layer_info.get("fixed_color") is not None:
                fc = layer_info["fixed_color"]
                color = np.empty((*mask.shape, 3), dtype=np.uint8)
                color[:, :, 0], color[:, :, 1], color[:, :, 2] = fc[0], fc[1], fc[2]
            else:
                # 注意：不能用 `x or y`——numpy 数组参与布尔运算会抛
                # "truth value of an array is ambiguous"，必须显式判 None
                color = layer_info.get("inpainted_hr_bgr")
                if color is None:
                    color = src_hr_bgr
            ctx.layers.append(self._make_layer(
                name=name, color_bgr=color, mask=mask,
                blend_mode=layer_info.get("blend_mode", "NORMAL"),
                opacity=int(layer_info.get("opacity", 255)),
            ))

        # 4) 委托内核编译（写盘唯一入口，G1）
        PsdCompiler.compile_psd(
            ctx, output_path,
            compression=self.compression,
            output_mode=ctx.output_mode,
        )
        self.last_qa = dict(ctx.qa_metrics)
        return output_path

    # ---------------- dict 层 → LayerDescriptor ----------------
    def _make_layer(self, name: str, color_bgr: np.ndarray, mask: np.ndarray,
                    full_canvas: bool = False,
                    blend_mode: str = "NORMAL", opacity: int = 255) -> LayerDescriptor:
        h, w = color_bgr.shape[:2]
        # 紧凑包围盒（与合流前 psb_builder 的 bbox 语义一致）
        pts = cv2.findNonZero(mask)
        if pts is not None and not full_canvas:
            x, y, bw, bh = cv2.boundingRect(pts)
            top, bottom, left, right = y, y + bh, x, x + bw
        else:
            top, bottom, left, right = 0, h, 0, w

        crop = color_bgr[top:bottom, left:right]
        alpha = mask[top:bottom, left:right]

        if self.is_cmyk:
            raw = self._to_cmyk_limited(crop)               # 反码（含 TAC 压制）
            cmyk = self._raw_to_logical(raw)                # → 逻辑墨量（内核约定）
            return LayerDescriptor(
                name=name,
                layer_type="substrate_cmyk" if full_canvas else "print_cmyk",
                cmyk_channels=np.ascontiguousarray(cmyk),
                alpha=np.ascontiguousarray(alpha),
                visible=True,
                opacity=int(opacity),
                blend_mode=str(blend_mode).lower(),
                bbox=(left, top, right, bottom),
            )
        rgba = np.empty((*crop.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgba[:, :, 3] = alpha
        return LayerDescriptor(
            name=name,
            layer_type="element_rgba",
            rgba=np.ascontiguousarray(rgba),
            visible=True,
            opacity=int(opacity),
            blend_mode=str(blend_mode).lower(),
            bbox=(left, top, right, bottom),
        )
