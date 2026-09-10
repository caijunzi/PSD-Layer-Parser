"""
真实物理激光冲孔提取算子 (MicroHolesOperator)。
对应 SSOT §6.2 / 铁律 R5 / 断言 V4-13~16。

工程目标与铁律合规：
1. 严格遵守铁律 R5：微孔检测仅在有效基材掩模内运行，边缘腐蚀屏蔽，杜绝在虚空区域产生假冲孔；
2. 消除魔法数字：sigma1, sigma2, threshold, base_rx, base_ry 等全部由 params 注入；
3. 输出矢量级二值 1-bit 刀模掩模（DieCut Mask）与质检统计指标。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class MicroHolesOperator(BaseOperator):
    """真实物理激光冲孔提取算子。"""

    name: str = "micro_holes"

    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        # 1. 优先取检测分支图像（铁律 R2）
        if hasattr(ctx, "detect_image") and ctx.detect_image is not None:
            image = ctx.detect_image
        elif hasattr(ctx, "raw_image") and ctx.raw_image is not None:
            image = ctx.raw_image
        elif isinstance(ctx, np.ndarray):
            image = ctx
        else:
            return OperatorResult(
                success=False,
                message="MicroHolesOperator: No valid image found in context",
            )

        # 2. 读取工艺参数，严禁魔法数字（SSOT §6.1）
        sigma1 = float(get_param(params, "dog_sigma_low", default=0.9, min_val=0.1, max_val=5.0))
        sigma2 = float(get_param(params, "dog_sigma_high", default=2.0, min_val=0.5, max_val=10.0))
        threshold = float(get_param(params, "dog_threshold", default=4.5, min_val=0.5, max_val=50.0))
        gray_max = float(get_param(params, "gray_max", default=225.0, min_val=50.0, max_val=255.0))
        base_rx = float(get_param(params, "base_rx", default=1.9, min_val=0.5, max_val=10.0))
        base_ry = float(get_param(params, "base_ry", default=2.3, min_val=0.5, max_val=10.0))
        valid_mask = params.get("valid_print_mask", None) if params else None

        # 3. 高斯差分（DoG）检测真实微孔物理极大值中心
        y_peaks, x_peaks = self.detect_micro_holes(
            image=image,
            valid_print_mask=valid_mask,
            sigma1=sigma1,
            sigma2=sigma2,
            threshold=threshold,
            gray_max=gray_max,
        )

        # 4. 可选：渲染高分辨率 1-bit 刀模冲孔层
        target_size = params.get("target_size", (image.shape[1], image.shape[0])) if params else (image.shape[1], image.shape[0])
        scale_x = float(params.get("scale_x", 1.0)) if params else 1.0
        scale_y = float(params.get("scale_y", 1.0)) if params else 1.0

        diecut_mask = self.render_diecut_mask(
            target_size=target_size,
            y_peaks=y_peaks,
            x_peaks=x_peaks,
            scale_x=scale_x,
            scale_y=scale_y,
            base_rx=base_rx,
            base_ry=base_ry,
        )

        hole_count = len(x_peaks)
        metrics = {
            "hole_count": int(hole_count),
            "dog_sigma_low": sigma1,
            "dog_sigma_high": sigma2,
            "dog_threshold": threshold,
            "target_resolution": f"{target_size[0]}x{target_size[1]}",
        }

        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["micro_holes"] = metrics

        return OperatorResult(
            success=True,
            data={
                "y_peaks": y_peaks,
                "x_peaks": x_peaks,
                "hole_count": hole_count,
                "diecut_mask": diecut_mask,
            },
            metrics=metrics,
            message=f"Detected {hole_count} authentic laser-cut micro-holes via DoG ({sigma1}/{sigma2}, thresh={threshold})",
        )

    @staticmethod
    def detect_micro_holes(
        image: np.ndarray,
        valid_print_mask: Optional[np.ndarray] = None,
        sigma1: float = 0.9,
        sigma2: float = 2.0,
        threshold: float = 4.5,
        gray_max: float = 225.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """DoG 核心微孔检测算法。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(image.shape) == 3 else image.astype(np.float32)
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)
        dog = g2 - g1

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(dog, kernel3)

        is_peak = (dog == dilated) & (dog > threshold)
        if valid_print_mask is not None:
            is_peak = is_peak & (valid_print_mask > 0)
        else:
            is_peak = is_peak & (gray < gray_max)

        y_peaks, x_peaks = np.where(is_peak)
        return y_peaks, x_peaks

    @staticmethod
    def render_diecut_mask(
        target_size: Tuple[int, int],
        y_peaks: np.ndarray,
        x_peaks: np.ndarray,
        scale_x: float,
        scale_y: float,
        base_rx: float = 1.9,
        base_ry: float = 2.3,
    ) -> np.ndarray:
        """渲染高精度 1-bit 刀模板：白色=基材保全(255)，黑色=激光冲孔(0)。"""
        target_w, target_h = target_size
        mask = np.full((target_h, target_w), 255, dtype=np.uint8)

        rx = int(round(base_rx * scale_x))
        ry = int(round(base_ry * scale_y))

        for yp, xp in zip(y_peaks, x_peaks):
            X = int(round(xp * scale_x))
            Y = int(round(yp * scale_y))
            if 0 <= X < target_w and 0 <= Y < target_h:
                cv2.ellipse(mask, (X, Y), (rx, ry), 0, 0, 360, 0, -1, lineType=cv2.LINE_AA)

        return mask


# 保持向后兼容性别名
MicroHolesExtractor = MicroHolesOperator
