"""
受控 2.5D 遮挡补全算子 (DeocclusionOperator)。
对应 SSOT §6.2 / §6.8 / ADR-010 / 铁律 R1 / 断言 V4-33。

工程目标与铁律合规：
1. 严格遵守铁律 R1：任何生成/补全内容必须显式标记为重建区（reconstruction_mask）；
2. 严格遵守铁律 R6：层合成等价性，前景元素隐藏后底层结构自然延展；
3. 受控可追溯（断言 V4-33）：统计并向质检单披露重建区像素总数、面积占比与最大单域直径，超限告警。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from engine.operators.base import BaseOperator, OperatorResult, get_param


class DeocclusionOperator(BaseOperator):
    """受控 2.5D 遮挡补全算子，强制标定重建区。"""

    name: str = "deocclusion"

    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        # 1. 获取输入图像与图层列表
        img_bgr = None
        if params and "image" in params:
            img_bgr = params["image"]
        elif hasattr(ctx, "output_image") and ctx.output_image is not None:
            img_bgr = ctx.output_image
        elif hasattr(ctx, "raw_image") and ctx.raw_image is not None:
            img_bgr = ctx.raw_image
        elif isinstance(ctx, np.ndarray):
            img_bgr = ctx

        if img_bgr is None:
            return OperatorResult(
                success=False,
                message="DeocclusionOperator: No valid image found in context",
            )

        layers = params.get("layers", getattr(ctx, "layers", []))
        if not layers:
            return OperatorResult(
                success=False,
                message="DeocclusionOperator: No layers provided to deocclude",
            )

        h, w = img_bgr.shape[:2]

        # 2. 读取工艺参数，严禁魔法数字（SSOT §6.1 / §6.8）
        extension_pixels = int(get_param(params, "extension_pixels", default=20, min_val=5, max_val=100))
        inpaint_radius = int(get_param(params, "inpaint_radius", default=7, min_val=1, max_val=31))
        max_area_ratio = float(get_param(params, "max_area_ratio", default=0.08, min_val=0.01, max_val=0.30))
        inpainting_provider = params.get("inpainting_provider", None)

        # 3. 构造定向延展核
        ext_ksize = max(5, (extension_pixels // 2) * 2 + 1)
        kernel_ext = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ext_ksize, ext_ksize))

        accumulated_foreground = np.zeros((h, w), dtype=np.uint8)
        global_reconstruction_mask = np.zeros((h, w), dtype=np.uint8)
        deoccluded_layers_info = []

        for i, layer in enumerate(layers):
            lname = layer.name.lower() if hasattr(layer, "name") else layer.get("name", "").lower()
            lmask = (layer.alpha > 0).astype(np.uint8) * 255 if hasattr(layer, "alpha") and layer.alpha is not None else layer.get("mask", None)

            if lmask is None:
                continue

            # 底色画布与边框/印章不作为被遮挡底层处理
            if "base" in lname or "canvas" in lname or "frame" in lname or "seal" in lname or "seam" in lname:
                continue

            layer_recon_mask = np.zeros((h, w), dtype=np.uint8)
            inpainted_bgr = None

            if np.count_nonzero(accumulated_foreground) > 0:
                # 寻找当前层与已有前景重叠/邻接区域
                dilated_layer = cv2.dilate(lmask, kernel_ext)
                extension_zone = cv2.bitwise_and(dilated_layer, accumulated_foreground)
                ext_count = int(np.count_nonzero(extension_zone))

                if ext_count > 20:
                    # 执行受控补全（优先神经网络提供者，次选 Telea 确定性补全）
                    if inpainting_provider is not None:
                        try:
                            inpainted_bgr = inpainting_provider.inpaint(img_bgr, extension_zone)
                        except Exception:
                            inpainted_bgr = cv2.inpaint(img_bgr, extension_zone, inpaint_radius, cv2.INPAINT_TELEA)
                    else:
                        inpainted_bgr = cv2.inpaint(img_bgr, extension_zone, inpaint_radius, cv2.INPAINT_TELEA)

                    layer_recon_mask = extension_zone
                    global_reconstruction_mask = cv2.bitwise_or(global_reconstruction_mask, extension_zone)

                    # 回写到图层对象中（铁律 R1）
                    if hasattr(layer, "alpha"):
                        layer.alpha = cv2.bitwise_or(layer.alpha, extension_zone)
                        layer.is_recon = True
                    elif isinstance(layer, dict):
                        layer["mask"] = cv2.bitwise_or(layer["mask"], extension_zone)
                        layer["inpainted_bgr"] = inpainted_bgr
                        layer["reconstruction_mask"] = extension_zone

            deoccluded_layers_info.append({
                "name": lname,
                "recon_pixels": int(np.count_nonzero(layer_recon_mask)),
                "has_inpaint": inpainted_bgr is not None
            })

            # 累积当前层为上层前景
            accumulated_foreground = cv2.bitwise_or(accumulated_foreground, lmask)

        # 4. 重建区连通域与度量分析（断言 V4-33）
        total_recon_px = int(np.count_nonzero(global_reconstruction_mask))
        total_canvas_px = h * w
        recon_ratio = float(total_recon_px / max(1, total_canvas_px))

        # 统计最大连通域外接圆直径
        max_diameter = 0.0
        if total_recon_px > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(global_reconstruction_mask)
            for lbl in range(1, num_labels):
                bw = stats[lbl, cv2.CC_STAT_WIDTH]
                bh = stats[lbl, cv2.CC_STAT_HEIGHT]
                diam = float(np.hypot(bw, bh))
                if diam > max_diameter:
                    max_diameter = diam

        controlled = recon_ratio <= max_area_ratio
        metrics = {
            "reconstruction_pixel_count": int(total_recon_px),
            "reconstruction_area_ratio": round(recon_ratio, 5),
            "max_area_ratio_threshold": round(max_area_ratio, 5),
            "max_component_diameter_px": round(max_diameter, 2),
            "controlled": bool(controlled),
            "deoccluded_layers_count": len(deoccluded_layers_info),
        }

        if hasattr(ctx, "qa_metrics") and isinstance(ctx.qa_metrics, dict):
            ctx.qa_metrics["deocclusion"] = metrics

        return OperatorResult(
            success=True,
            data={
                "layers": layers,
                "deoccluded_layers_info": deoccluded_layers_info,
            },
            reconstruction_mask=global_reconstruction_mask,
            metrics=metrics,
            message=f"De-occlusion complete: recon_area={recon_ratio:.2%} (max_thresh={max_area_ratio:.2%}), max_diam={max_diameter:.1f}px, controlled={controlled}",
        )
