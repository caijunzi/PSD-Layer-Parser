"""弥散门与语义覆盖披露测试。

背景（2026-09-11 山水图验收）：SAM 对远山/峭壁等无边界山水元素产出
全画布碎片掩模（bbox 覆盖 ~75%），合成时雾化污染。弥散门 = 跨图通用
可靠性规则（非 per-image 补丁）：内容层外接框覆盖比 > 50% 即拒绝，
底板/外框/折痕等天然全画布层豁免。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent  # 仅为路径一致性
ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.providers.grounded_sam_provider import (
    GroundedSAMProvider,
    MAX_BBOX_AREA_RATIO,
    is_bbox_exempt,
)

H, W = 7808, 16000  # 山水图输出画幅


def diffuse_mask():
    """弥散掩模：外接框覆盖 ~75% 画布，像素稀疏散布（碎片化）。"""
    m = np.zeros((H, W), dtype=np.uint8)
    rng = np.random.RandomState(0)
    for _ in range(400):
        y = rng.randint(int(H * 0.1), int(H * 0.85))
        x = rng.randint(int(W * 0.1), int(W * 0.85))
        m[y:y + 12, x:x + 12] = 255
    return m


def compact_mask(cx_ratio=0.3, cy_ratio=0.4, w_ratio=0.2, h_ratio=0.15):
    """集中掩模：单一连通块。"""
    m = np.zeros((H, W), dtype=np.uint8)
    y0, x0 = int(H * cy_ratio), int(W * cx_ratio)
    m[y0:y0 + int(H * h_ratio), x0:x0 + int(W * w_ratio)] = 255
    return m


class TestDiffuseGate(unittest.TestCase):

    def test_diffuse_mask_rejected(self):
        m = diffuse_mask()
        ys, xs = np.nonzero(m > 127)
        bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        self.assertGreater(bbox / (W * H), MAX_BBOX_AREA_RATIO)  # 前置：构造正确
        reason = GroundedSAMProvider._diffuse_check("04D_远山淡墨晴岚_Distant_Soft_Mountain", m)
        self.assertIsNotNone(reason)
        self.assertIn("弥散", reason)

    def test_compact_mask_passes(self):
        self.assertIsNone(
            GroundedSAMProvider._diffuse_check("07_高士侍童人物_Figures_Scholar_Attendant", compact_mask())
        )

    def test_frame_full_canvas_exempt(self):
        """外框/底板/折痕天然全画布 —— 必须豁免，否则误杀装饰层。"""
        for name in ("10A_外框与织锦绫边_Brocade_Outer_Frame",
                     "02_纯净金箔大底板_Gold_Base_Clean",
                     "10B_屏风折痕折缝_Panel_Fold_Seams"):
            self.assertTrue(is_bbox_exempt(name))
            full = np.full((H, W), 255, dtype=np.uint8)
            self.assertIsNone(GroundedSAMProvider._diffuse_check(name, full))

    def test_exempt_check_negative(self):
        self.assertFalse(is_bbox_exempt("04A_前景墨岩峭壁_Foreground_Dark_Cliffs"))

    def test_empty_mask_no_crash(self):
        self.assertIsNone(
            GroundedSAMProvider._diffuse_check("08_芦雁群禽_Geese_Flock", np.zeros((H, W), dtype=np.uint8))
        )


if __name__ == "__main__":
    unittest.main()
