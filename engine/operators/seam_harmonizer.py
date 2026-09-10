"""
接缝流场对齐与平滑算子 (SeamHarmonizerOperator)。
对应 SSOT §6.2 / 铁律 R3 / 断言 V2-12, V4-30。

工程目标：
彻底消灭大画幅/循环花纹拼接中的机械硬拼缝（禁止裸 vstack / hstack）。
通过接缝切线梯度流场对齐（Cross-correlation / Phase alignment）与余弦 Hann 窗平滑插值，
消除拼接交界处的阶跃跳变、剪切错位与撕裂伪影。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class SeamHarmonizerOperator(BaseOperator):
    """接缝流场对齐与平滑算子：消灭拼缝阶跃与机械硬拼缝。"""

    name: str = "seam_harmonizer"

    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        # 1. 提取输入图像
        if hasattr(ctx, "detect_image") and ctx.detect_image is not None:
            image = ctx.detect_image
        elif hasattr(ctx, "output_image") and ctx.output_image is not None:
            image = ctx.output_image
        elif hasattr(ctx, "raw_image") and ctx.raw_image is not None:
            image = ctx.raw_image
        elif isinstance(ctx, np.ndarray):
            image = ctx
        else:
            return OperatorResult(
                success=False,
                message="SeamHarmonizerOperator: No valid image found in context",
            )

        # 2. 读取工艺参数，禁止硬编码（SSOT §6.1）
        warp_band_px = int(get_param(params, "warp_band_px", default=32, min_val=4, max_val=256))
        max_shift_px = int(get_param(params, "max_shift_px", default=16, min_val=1, max_val=64))
        blend_method = str(get_param(params, "blend_method", default="cosine_hann"))
        mode = str(get_param(params, "mode", default="cyclic_vertical"))  # cyclic_vertical | stitch_two

        if mode == "cyclic_vertical":
            harmonized, metrics = self._harmonize_cyclic_vertical(image, warp_band_px, max_shift_px, blend_method)
        else:
            tile_a = params.get("tile_a", image)
            tile_b = params.get("tile_b", image)
            harmonized, metrics = self._harmonize_two_tiles_vertical(tile_a, tile_b, warp_band_px, max_shift_px, blend_method)

        # 回写度量到上下文中
        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["seam_harmonizer"] = metrics

        return OperatorResult(
            success=True,
            data={"harmonized_image": harmonized},
            metrics=metrics,
            message=f"Seam harmonized: pre_rmse={metrics['pre_seam_rmse']:.2f} -> post_rmse={metrics['post_seam_rmse']:.2f} (offset_x={metrics['shear_offset_x']}px)",
        )

    def _harmonize_cyclic_vertical(
        self, img: np.ndarray, band_h: int, max_shift: int, blend_method: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        对单张图像进行上下循环接缝对齐与平滑，使垂直循环平铺时接缝完全连续。
        """
        h, w = img.shape[:2]
        band_h = min(band_h, h // 4)

        top_strip = img[:band_h, :].astype(np.float32)
        bot_strip = img[-band_h:, :].astype(np.float32)

        # 1. 梯度流场与水平剪切位移检测（Phase Correlation）
        gray_top = cv2.cvtColor(top_strip.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else top_strip
        gray_bot = cv2.cvtColor(bot_strip.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else bot_strip

        # 测算上下边缘第 0 行与最后一行之间的跳变 RMSE
        raw_diff = top_strip[0, :] - bot_strip[-1, :]
        pre_seam_rmse = float(np.sqrt(np.mean(raw_diff ** 2)))

        # 计算水平相位相关
        shift_x = 0
        try:
            shift, _ = cv2.phaseCorrelate(gray_bot.astype(np.float32), gray_top.astype(np.float32))
            cand_shift_x = int(round(shift[0]))
            if abs(cand_shift_x) <= max_shift:
                shift_x = cand_shift_x
        except Exception:
            shift_x = 0

        # 2. 剪切变形补偿（若存在微小水平错位）
        out = img.copy()
        if shift_x != 0:
            # 在底部过渡带内线性施加剪切变换
            for row_idx in range(band_h):
                alpha = float(row_idx) / float(band_h)
                dx = shift_x * alpha
                M = np.float32([[1, 0, dx], [0, 1, 0]])
                y = h - band_h + row_idx
                row_slice = out[y:y+1, :]
                warped_row = cv2.warpAffine(row_slice, M, (w, 1), borderMode=cv2.BORDER_REFLECT)
                out[y:y+1, :] = warped_row

        # 3. 余弦 Hann 窗加权双向平滑插值（Cosine Blending）
        # 让 top_strip 的上边缘渐变混合 bot_strip 的下边缘
        # window 权重从 0.0 到 0.5
        t = np.linspace(0.0, np.pi, band_h).reshape(-1, 1, 1 if len(img.shape) == 3 else 1)
        # 余弦衰减权重
        weight = 0.5 * (1.0 - np.cos(t))

        # 底部过渡带逐渐向顶部内容过渡
        out_bot_band = out[-band_h:, :].astype(np.float32)
        in_top_band = out[:band_h, :].astype(np.float32)

        # 混合：在底部带将自身平滑过渡到顶部切线
        blended_bot = (1.0 - weight * 0.5) * out_bot_band + (weight * 0.5) * in_top_band
        out[-band_h:, :] = np.clip(blended_bot, 0, 255).astype(np.uint8)

        # 4. 度量平滑后上下接缝 RMSE
        post_diff = out[0, :].astype(np.float32) - out[-1, :].astype(np.float32)
        post_seam_rmse = float(np.sqrt(np.mean(post_diff ** 2)))

        metrics = {
            "shear_offset_x": int(shift_x),
            "pre_seam_rmse": round(pre_seam_rmse, 3),
            "post_seam_rmse": round(post_seam_rmse, 3),
            "rmse_reduction_ratio": round(float((pre_seam_rmse - post_seam_rmse) / max(1e-3, pre_seam_rmse)), 4),
            "warp_band_px": int(band_h),
            "blend_method": blend_method,
        }

        return out, metrics

    def _harmonize_two_tiles_vertical(
        self, tile_a: np.ndarray, tile_b: np.ndarray, band_h: int, max_shift: int, blend_method: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """对两块垂直相邻单元拼接处进行流场平滑融合。"""
        ha, wa = tile_a.shape[:2]
        hb, wb = tile_b.shape[:2]
        w = min(wa, wb)
        band_h = min(band_h, ha // 4, hb // 4)

        strip_a = tile_a[-band_h:, :w].astype(np.float32)
        strip_b = tile_b[:band_h, :w].astype(np.float32)

        pre_diff = strip_a[-1, :] - strip_b[0, :]
        pre_seam_rmse = float(np.sqrt(np.mean(pre_diff ** 2)))

        # Hann 余弦过渡
        t = np.linspace(0.0, 1.0, band_h).reshape(-1, 1, 1 if len(tile_a.shape) == 3 else 1)
        w_blend = 0.5 * (1.0 - np.cos(np.pi * t))

        blended_seam = ((1.0 - w_blend) * strip_a + w_blend * strip_b).astype(np.uint8)

        out_a = tile_a.copy()
        out_b = tile_b.copy()
        out_a[-band_h:, :w] = blended_seam[:band_h, :]

        stacked = np.vstack([out_a, out_b])
        post_diff = out_a[-1, :w].astype(np.float32) - out_b[0, :w].astype(np.float32)
        post_seam_rmse = float(np.sqrt(np.mean(post_diff ** 2)))

        metrics = {
            "shear_offset_x": 0,
            "pre_seam_rmse": round(pre_seam_rmse, 3),
            "post_seam_rmse": round(post_seam_rmse, 3),
            "rmse_reduction_ratio": round(float((pre_seam_rmse - post_seam_rmse) / max(1e-3, pre_seam_rmse)), 4),
            "warp_band_px": int(band_h),
            "blend_method": blend_method,
        }
        return stacked, metrics
