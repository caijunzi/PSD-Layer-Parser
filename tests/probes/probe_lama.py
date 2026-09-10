# -*- coding: utf-8 -*-
"""探针：LaMaInpaintingProvider 在不同 ROI 形状下是否真的走到神经推理。

背景：生产日志出现
  [Circuit Breaker] Batch inference on openvino_GPU.1 failed:
  model input (shape=[?,3,512,512]) and tensor (shape=[2,3,486,512]) are incompatible
怀疑 _inpaint_tiled 在 ROI 某一维 < 512 时，切片被截断产生非 512 的 tile，
导致 batch stack 尺寸不齐 -> OpenVINO 拒绝 -> 整批回退 Telea，即 LaMa 实际未生效。
"""
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.providers.inpainting_provider import LaMaInpaintingProvider

SRC = "inputs/source_4000.jpg"


ROI_PAD = 48  # 与 _inpainting_provider.inpaint() 的 pad 一致


def make_case(h: int, w: int, hole_w: int, hole_h: int) -> tuple:
    """构造 (h, w) 的图与矩形洞掩模。

    为什么用矩形：ROI 由 mask 的 boundingRect + 2*48px pad 决定，
    只有精确控制洞的宽高，才能构造出「高<512 宽>512」这类边界形状。
    用圆形小洞会得到 ~183x183 的 ROI，直接走 _inpaint_single，到不了 tiled 分支。

    洞面积必须 > 2500 px，否则 inpaint() 第 116 行的微孔快速旁路会直接返回 Telea。
    """
    img = cv2.resize(cv2.imread(SRC), (w, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, (w - hole_w) // 2)
    y0 = max(0, (h - hole_h) // 2)
    mask[y0:y0 + hole_h, x0:x0 + hole_w] = 255
    return img, mask


def main() -> int:
    print("[probe] 初始化 LaMaInpaintingProvider (GPU.1)...")
    p = LaMaInpaintingProvider(preferred_device="GPU.1")
    print(f"[probe] backend = {p.backend}")
    if p.backend is None or "openvino" not in str(p.backend):
        print("[WARN] 神经后端未激活，后续结论不成立")
    print()

    # (标签, 画布 h, 画布 w, 洞宽, 洞高)
    cases = [
        ("ROI 双方 <=512 -> 单切片", 900, 900, 200, 200),
        ("ROI 高<512 宽>512 -> tiled", 900, 1200, 600, 300),
        ("ROI 高>512 宽<512 -> tiled", 1200, 900, 300, 600),
        ("ROI 双方 >512 整倍数", 1400, 1400, 600, 600),
        ("ROI 双方 >512 非整倍数", 1400, 1400, 700, 700),
        ("ROI 双方 >512 大洞", 2000, 2500, 1100, 900),
    ]

    print(f"{'场景':<34}{'ROI(h x w)':>16}{'熔断?':>8}{'耗时s':>8}  说明")
    print("-" * 100)
    for label, h, w, hw, hh in cases:
        img, mask = make_case(h, w, hw, hh)
        roi_h, roi_w = hh + 2 * ROI_PAD, hw + 2 * ROI_PAD
        # 捕获 stdout 中的熔断信息
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        t0 = __import__("time").time()
        try:
            _ = p.inpaint(img, mask)
            err = ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        dt = __import__("time").time() - t0
        sys.stdout = old
        log = buf.getvalue()
        tripped = "YES" if ("Circuit Breaker" in log or "failed" in log) else "no"
        shape_note = ""
        if "tensor (shape=" in log:
            shape_note = "tensor=" + log.split("tensor (shape=")[1].split(")")[0]
        print(f"{label:<34}{f'{roi_h} x {roi_w}':>16}{tripped:>8}{dt:>8.2f}  {shape_note or err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
