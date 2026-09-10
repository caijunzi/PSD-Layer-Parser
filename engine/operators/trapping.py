"""
印前专色陷印/爆边算子 (TrappingOperator)。
对应 SSOT §6.2 / ADR-017 / 断言 V4-02。

工业背景与工程目标：
在制版印刷（PLATE 线）中，烫金、玫瑰铜专色与四色底色套印时，印刷机机械套准公差（0.1~0.3mm）
极易产生肉眼可见的漏白缝隙（Flashing）。
本算子根据物理输出 PPI 与工艺毫米公差自适应派生陷印宽度：
    W_px = round((trap_mm / 25.4) * ppi)
在专色与周边油墨交界带执行变尺寸智能爆边（Spread/Choke），生成独立的 _Spot 专色通道层。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class TrappingOperator(BaseOperator):
    """印前专色变尺寸陷印/爆边算子。"""

    name: str = "trapping"

    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        # 1. 提取专色掩模与参考邻域掩模
        spot_mask = None
        adjacent_mask = None

        if params and "spot_mask" in params:
            spot_mask = params["spot_mask"]
        if params and "adjacent_mask" in params:
            adjacent_mask = params["adjacent_mask"]

        # 兼容从 ctx 中提取
        if spot_mask is None and hasattr(ctx, "layers"):
            for lyr in ctx.layers:
                lname = getattr(lyr, "name", "").lower()
                if "spot" in lname or "foil" in lname or "gold" in lname or "copper" in lname:
                    if getattr(lyr, "alpha", None) is not None:
                        spot_mask = (lyr.alpha > 0).astype(np.uint8) * 255
                        break

        if spot_mask is None:
            return OperatorResult(
                success=False,
                message="TrappingOperator: No spot mask provided or found in context",
            )

        h, w = spot_mask.shape[:2]

        # 2. 读取工艺参数，严禁魔法数字（SSOT §6.1 / ADR-017）
        ppi = float(get_param(params, "ppi", default=getattr(ctx, "ppi", 150.0), min_val=10.0, max_val=2400.0))
        trap_mm = float(get_param(params, "trap_mm", default=0.2, min_val=0.05, max_val=2.0))
        trap_type = str(get_param(params, "trap_type", default="spread"))  # spread | choke
        trap_density = int(get_param(params, "trap_density", default=255, min_val=1, max_val=255))

        # 3. 物理毫米到像素宽度的严格推导
        # 1 inch = 25.4 mm
        calc_px = (trap_mm / 25.4) * ppi
        trap_width_px = max(1, int(round(calc_px)))

        # 4. 形态学变尺寸陷印计算
        ksize = trap_width_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        if trap_type == "spread":
            # 专色外扩（Spread）：专色轻微膨胀压入邻域
            dilated_spot = cv2.dilate(spot_mask, kernel)
            if adjacent_mask is not None:
                # 仅在有邻域油墨/底色处外扩，避免漫溢到空白无墨区
                trap_band = cv2.bitwise_and(dilated_spot, adjacent_mask)
                trapped_spot_mask = cv2.bitwise_or(spot_mask, trap_band)
            else:
                trapped_spot_mask = dilated_spot
                trap_band = cv2.subtract(trapped_spot_mask, spot_mask)
        else:
            # 背景收缩（Choke）：底色内缩为专色留出嵌合余量
            eroded_spot = cv2.erode(spot_mask, kernel)
            trapped_spot_mask = eroded_spot
            trap_band = cv2.subtract(spot_mask, trapped_spot_mask)

        # 5. 生成标准 8-bit _Spot 专色通道
        # 逻辑墨量：0=无墨，255=100% 墨（SSOT §4.3）
        spot_channel = (trapped_spot_mask > 0).astype(np.uint8) * trap_density

        # 6. 统计质检指标
        spot_active_px = int(np.count_nonzero(trapped_spot_mask))
        trap_band_px = int(np.count_nonzero(trap_band))
        total_px = h * w
        spot_ratio = float(spot_active_px / max(1, total_px))
        band_ratio = float(trap_band_px / max(1, spot_active_px))

        metrics = {
            "ppi": round(ppi, 1),
            "trap_mm": round(trap_mm, 3),
            "calculated_px": round(calc_px, 3),
            "trap_width_px": int(trap_width_px),
            "trap_type": trap_type,
            "spot_active_pixels": int(spot_active_px),
            "trap_band_pixels": int(trap_band_px),
            "spot_coverage_ratio": round(spot_ratio, 4),
            "trap_expansion_ratio": round(band_ratio, 4),
        }

        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["trapping"] = metrics

        return OperatorResult(
            success=True,
            data={
                "trapped_spot_mask": trapped_spot_mask,
                "trap_band_mask": trap_band,
                "spot_channel": spot_channel,
                "trap_width_px": trap_width_px,
            },
            metrics=metrics,
            message=f"Trapping applied: {trap_type} {trap_mm}mm ({trap_width_px}px @ {ppi:.1f} PPI), trap band: {trap_band_px} px",
        )
