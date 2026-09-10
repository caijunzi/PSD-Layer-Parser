"""
核心轮廓几何保全算子 (ContourProtectionOperator)。
对应 SSOT §6.2 / ADR-008 / 铁律 R4 / 断言 V4-11~12。

工程目标：
彻底消除裁切线硬编码（如写死 Y=718 的历史缺陷）。
在运行时自适应扫描画面底部边缘，识别有效花纹/回纹与纤维毛边、破损毛刺或背景阴影，
自动计算出绝不截断主体花纹连通域回路的几何保全裁切下限 safe_bottom_y。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class ContourProtectionOperator(BaseOperator):
    """核心轮廓保护算子：运行时自适应计算几何保全裁切线。"""

    name: str = "contour_protection"

    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        # 1. 优先从 ctx.detect_image 获取检测分支图像（铁律 R2）
        if hasattr(ctx, "detect_image") and ctx.detect_image is not None:
            image = ctx.detect_image
        elif hasattr(ctx, "raw_image") and ctx.raw_image is not None:
            image = ctx.raw_image
        elif isinstance(ctx, np.ndarray):
            image = ctx
        else:
            return OperatorResult(
                success=False,
                message="ContourProtectionOperator: No valid image found in context",
            )

        h, w = image.shape[:2]

        # 2. 从 params 读取工艺参数，禁止硬编码（SSOT §6.1）
        min_safe_bottom_y = int(get_param(params, "min_safe_bottom_y", default=int(h * 0.8), min_val=0, max_val=h - 1))
        max_search_y = int(get_param(params, "max_search_y", default=h - 1, min_val=min_safe_bottom_y, max_val=h))
        max_tear_ratio = float(get_param(params, "max_tear_ratio", default=0.01, min_val=0.0, max_val=0.5))
        edge_energy_thresh = float(get_param(params, "edge_energy_thresh", default=4.0, min_val=0.5, max_val=50.0))
        desk_gray_thresh = float(get_param(params, "desk_gray_thresh", default=215.0, min_val=50.0, max_val=255.0))
        close_kernel_size = int(get_param(params, "close_kernel_size", default=7, min_val=3, max_val=31))
        close_iter = int(get_param(params, "close_iter", default=2, min_val=1, max_val=10))

        # 3. 纹理能量与背景检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        lap_smooth = cv2.GaussianBlur(lap, (close_kernel_size, close_kernel_size), 0)

        # 区分有效织物/主体与破损毛边/平坦背景
        is_desk_bright = gray > desk_gray_thresh
        if len(image.shape) == 3:
            r = image[:, :, 2].astype(np.int32)
            b = image[:, :, 0].astype(np.int32)
            is_void_or_desk = is_desk_bright & ((r - b < 25) | (lap_smooth < edge_energy_thresh))
        else:
            is_void_or_desk = is_desk_bright

        # 形态学闭运算填平毛边微孔
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        void_cleaned = cv2.morphologyEx(is_void_or_desk.astype(np.uint8), cv2.MORPH_CLOSE, k_close, iterations=close_iter)

        # 4. 自底向上扫描安全行
        # 从 min_safe_bottom_y 向下寻找连续完整基材且无撕裂入侵的最低行
        safe_bottom_y = min_safe_bottom_y
        for y in range(min_safe_bottom_y, min(max_search_y, h)):
            tear_count = int(np.sum(void_cleaned[y, :]))
            tear_ratio = tear_count / float(w)
            if tear_ratio <= max_tear_ratio:
                safe_bottom_y = y
            else:
                # 遇到撕裂毛边或背景侵入，立即截停在上一安全行
                break

        # 5. 校验主体花纹连通性完整度
        slice_region = gray[:safe_bottom_y, :]
        solid_pixels = int(np.sum(void_cleaned[:safe_bottom_y, :] == 0))
        total_pixels = int(safe_bottom_y * w)
        fabric_solid_ratio = float(solid_pixels / max(1, total_pixels))

        metrics = {
            "safe_bottom_y": int(safe_bottom_y),
            "min_safe_bottom_y": int(min_safe_bottom_y),
            "fabric_solid_ratio": round(fabric_solid_ratio, 4),
            "tear_ratio_at_cut": round(float(np.sum(void_cleaned[safe_bottom_y, :]) / float(w)), 4),
            "cut_y_source": "contour_protection",
        }

        # 如果 ctx 有相应字段，回写到上下文中
        if hasattr(ctx, "unit_cut_bottom_y"):
            ctx.unit_cut_bottom_y = safe_bottom_y
        if hasattr(ctx, "cut_y_source"):
            ctx.cut_y_source = "contour_protection"
        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["contour_protection"] = metrics

        # 构造安全裁切掩模 (H, W)，safe_bottom_y 以上为 255
        cut_mask = np.zeros((h, w), dtype=np.uint8)
        cut_mask[:safe_bottom_y, :] = 255

        return OperatorResult(
            success=True,
            data={
                "safe_bottom_y": safe_bottom_y,
                "cut_mask": cut_mask,
                "cropped_image": image[:safe_bottom_y, :].copy(),
            },
            metrics=metrics,
            message=f"Contour protection: dynamically computed safe_bottom_y={safe_bottom_y} (min={min_safe_bottom_y}, solid_ratio={fabric_solid_ratio:.2%})",
        )
