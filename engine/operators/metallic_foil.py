"""
多金属分色与烫印分层算子 (MetallicFoilOperator)。
对应 SSOT §6.2 / 断言 V4-17~19。

工程目标与铁律合规：
1. 精准解耦玫瑰铜箔（Rose Copper Foil）与复古金（Antique Gold Metallic Ink），防止色相串色与边缘溢出；
2. 消除魔法数字：所有色度对比度、Lab A* 阈值、空间限定区间均由 params 注入；
3. 输出独立的二值印刷版掩模与面积统计度量。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class MetallicFoilOperator(BaseOperator):
    """多金属分色与烫金掩模提取算子。"""

    name: str = "metallic_foil"

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
                message="MetallicFoilOperator: No valid image found in context",
            )

        h, w = image.shape[:2]

        # 2. 读取工艺参数，严禁魔法数字（SSOT §6.1）
        valid_print_mask = params.get("valid_print_mask", None) if params else None
        if valid_print_mask is None:
            valid_print_mask = np.full((h, w), 255, dtype=np.uint8)

        lab_a_min = float(get_param(params, "lab_a_min", default=140.0, min_val=0.0, max_val=255.0))
        rb_diff_min = float(get_param(params, "rb_diff_min", default=90.0, min_val=0.0, max_val=255.0))
        rg_diff_min = float(get_param(params, "rg_diff_min", default=35.0, min_val=0.0, max_val=255.0))
        band_y_min = params.get("band_y_min", None) if params else None
        band_y_max = params.get("band_y_max", None) if params else None
        close_ksize = int(get_param(params, "close_ksize", default=5, min_val=3, max_val=25))
        open_ksize = int(get_param(params, "open_ksize", default=3, min_val=1, max_val=25))

        # 3. 核心分色算法
        copper_mask, gold_mask = self.separate_foil_and_gold(
            image=image,
            valid_print_mask=valid_print_mask,
            lab_a_min=lab_a_min,
            rb_diff_min=rb_diff_min,
            rg_diff_min=rg_diff_min,
            band_y_min=band_y_min,
            band_y_max=band_y_max,
            close_ksize=close_ksize,
            open_ksize=open_ksize,
        )

        copper_px = int(np.count_nonzero(copper_mask))
        gold_px = int(np.count_nonzero(gold_mask))
        total_px = h * w

        metrics = {
            "copper_pixels": copper_px,
            "gold_pixels": gold_px,
            "copper_ratio": round(copper_px / max(1, total_px), 4),
            "gold_ratio": round(gold_px / max(1, total_px), 4),
            "lab_a_min": lab_a_min,
            "rb_diff_min": rb_diff_min,
            "rg_diff_min": rg_diff_min,
        }

        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["metallic_foil"] = metrics

        return OperatorResult(
            success=True,
            data={
                "copper_mask": copper_mask,
                "gold_mask": gold_mask,
            },
            metrics=metrics,
            message=f"Metallic separation complete: copper={copper_px} px ({metrics['copper_ratio']:.2%}), gold={gold_px} px ({metrics['gold_ratio']:.2%})",
        )

    @staticmethod
    def separate_foil_and_gold(
        image: np.ndarray,
        valid_print_mask: np.ndarray,
        lab_a_min: float = 140.0,
        rb_diff_min: float = 90.0,
        rg_diff_min: float = 35.0,
        band_y_min: Optional[int] = None,
        band_y_max: Optional[int] = None,
        close_ksize: int = 5,
        open_ksize: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离玫瑰铜箔与复古金掩模。
        返回: (copper_mask, gold_mask)，均为 uint8 二值掩模 (255=激活, 0=非激活)。
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        A = lab[:, :, 1]
        R = image[:, :, 2].astype(float)
        G = image[:, :, 1].astype(float)
        B = image[:, :, 0].astype(float)

        # 玫瑰铜箔具有高 A* 红度分量以及显著红蓝/红绿差值
        is_copper_seed = (
            (valid_print_mask > 0)
            & (R - B > rb_diff_min)
            & (R - G > rg_diff_min)
            & (A >= lab_a_min)
        )

        # 如果工艺预设中限定了工艺带垂直区间 (band_y_range)
        if band_y_min is not None:
            is_copper_seed[:int(band_y_min), :] = False
        if band_y_max is not None:
            is_copper_seed[int(band_y_max):, :] = False

        # 形态学平滑：铜箔是片状/块状金属，过滤孤立噪点
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        copper_cleaned = cv2.morphologyEx(
            is_copper_seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close
        )
        if open_ksize > 1:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
            copper_mask = cv2.morphologyEx(copper_cleaned, cv2.MORPH_OPEN, k_open)
        else:
            copper_mask = copper_cleaned

        # 古金色为有效印刷区域除去铜箔后的剩余部分
        gold_mask = (valid_print_mask > 0).astype(np.uint8) * 255
        gold_mask[copper_mask > 0] = 0

        return copper_mask, gold_mask


# 保持向后兼容性别名
MetallicFoilSeparator = MetallicFoilOperator
