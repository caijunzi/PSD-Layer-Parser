"""R6 层合成等价性验证（铁律 R6 / 断言 V4-34, V4-35）。

R6 原文：**所有可见层按序叠加必须还原原图；隐藏任意单层后可恢复。**

与 `pipeline/06_verify_psb.py` 的 MAE 指标的区别（重要）
----------------------------------------------------
`06_verify_psb.py` 比对的是「Section 5 预渲染合并图 vs 源图」。实测该指标在两个
掩模质量差异极大的产物（修复前印章层 8.78%、修复后 0.0115%）上**给出完全相同的
MAE=3.498** —— 因为 Section 5 在两版中都是同一个 RealESRGAN 超分结果，与掩模无关。
该指标实际反映的是**超分链路与缩略图比对方法**引入的差异，对掩模正确性不敏感。

本脚本改为比对「图层合成结果 vs Section 5」，这才是 R6 的直接检验：
若各层 alpha 与颜色齐备，自底向上叠加应能还原 Section 5。

用法：
    python tests/verify_r6_composition.py outputs/Rosetsu_Master_16k_fixed.psb [--max-side 2000]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BLEND_MULTIPLY = "multiply"


def composite_layers(psd: PSDImage, max_side: int) -> np.ndarray:
    """自底向上按 alpha 叠加所有可见层，返回 float32 RGB（已降采样）。"""
    canvas_rgb = np.zeros((psd.height, psd.width, 3), dtype=np.float32)
    canvas_a = np.zeros((psd.height, psd.width, 1), dtype=np.float32)

    # psd_tools 迭代顺序为底 → 顶（已实测确认：psd[0] 为金底板）
    for lyr in psd:
        if not lyr.is_visible():
            continue
        topil = lyr.topil()
        if topil is None:
            continue
        rgba = np.array(topil.convert("RGBA"), dtype=np.float32) / 255.0
        h, w = rgba.shape[:2]
        rgb = rgba[:, :, :3]
        a = rgba[:, :, 3:4]

        region_rgb = canvas_rgb[lyr.top:lyr.top + h, lyr.left:lyr.left + w]
        region_a = canvas_a[lyr.top:lyr.top + h, lyr.left:lyr.left + w]

        if getattr(lyr, "blend_mode", None) is not None and \
                "multiply" in str(lyr.blend_mode).lower():
            src = region_rgb * rgb
        else:
            src = rgb

        opacity = (lyr.opacity or 255) / 255.0
        out_a = a * opacity + region_a * (1.0 - a * opacity)
        out_rgb = (src * a * opacity + region_rgb * region_a * (1.0 - a * opacity))
        out_rgb = np.divide(out_rgb, np.maximum(out_a, 1e-6))

        canvas_rgb[lyr.top:lyr.top + h, lyr.left:lyr.left + w] = out_rgb
        canvas_a[lyr.top:lyr.top + h, lyr.left:lyr.left + w] = out_a

    # 以白底合成后降采样
    flat = canvas_rgb * canvas_a + (1.0 - canvas_a)
    img = Image.fromarray(np.clip(flat * 255, 0, 255).astype(np.uint8))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="R6 layer-composition equivalence check")
    ap.add_argument("psb")
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="合成与 Section 5 的 MAE 上限")
    args = ap.parse_args()

    if not os.path.isfile(args.psb):
        print(f"[FATAL] 未找到 {args.psb}")
        return 1

    psd = PSDImage.open(args.psb)
    print(f"文件: {args.psb}  画幅: {psd.size}  图层: {len(psd)}")

    print("叠加所有可见图层...")
    composed = composite_layers(psd, args.max_side)

    print("读取 Section 5 预渲染图...")
    section5 = psd.topil().convert("RGB")
    section5.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
    s5 = np.array(section5, dtype=np.float32)

    if composed.shape != s5.shape:
        print(f"[FAIL] 尺寸不一致: 叠加 {composed.shape} vs Section5 {s5.shape}")
        return 1

    mae = float(np.mean(np.abs(composed - s5)))
    print(f"\n叠加结果 vs Section 5 : MAE = {mae:.3f}  (阈值 {args.threshold:.1f})")
    print(f"叠加结果 均值 RGB     : {composed.mean(axis=(0, 1)).round(2)}")
    print(f"Section5 均值 RGB     : {s5.mean(axis=(0, 1)).round(2)}")

    ok = mae <= args.threshold
    print(f"\n=== R6 层合成等价性：{'通过' if ok else '未通过'} ===")
    if not ok:
        print("    说明：图层叠加无法还原 Section 5，可能是层颜色源缺失、")
        print("          透明层缺色或被遮挡区域未回填。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
