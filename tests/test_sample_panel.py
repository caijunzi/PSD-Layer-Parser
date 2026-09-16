# -*- coding: utf-8 -*-
"""实物样品照「实体样块」边界检测回归（2026-09-16 新增）。

覆盖三类：
  1. 真实样品照 `inputs/damask_sample.png` —— 命中已知真值边界
     （rows 48~52/968~970、cols 75~78/1454~1458，上轮诊断脚本实测）。
  2. 合成样块（平滑内外 + 锐边）—— 命中；高内缩/无外框 —— 安全回退 None。
  3. `resolve_sample_panel_roi` 的配置开关与掩码形状。

背景：现有 `_detect_painting_roi` 依赖灰度对比，对"墙面底衬与样块同类同色系纹样"
的样品照会失效（实测灰度差仅 3.3 < 10）→ 改用**长直边持续性**判别。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.sample_panel import (  # noqa: E402
    detect_sample_panel_bbox,
    resolve_sample_panel_roi,
    sample_panel_roi,
)

DAMASK = ROOT / "inputs" / "damask_sample.png"


def _smooth_panel(h=400, w=600, top=60, bottom=340, left=90, right=510, seed=0):
    """合成：平滑内外 + 锐利样块边。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 100.0, np.float32) + rng.normal(0, 3, (h, w))
    img[top:bottom, left:right] = 180.0 + rng.normal(0, 3, (bottom - top, right - left))
    return np.clip(img, 0, 255).astype(np.uint8)


class TestDetectSamplePanelRealPhoto(unittest.TestCase):
    """真实壁布样品照：必须命中已知真值边界。"""

    @classmethod
    def setUpClass(cls):
        if not DAMASK.is_file():
            raise unittest.SkipTest(f"缺少样本 {DAMASK}")
        import cv2
        from engine.core.io_utils import imread_unicode
        img = imread_unicode(str(DAMASK))
        assert img is not None, "读取 damask_sample.png 失败"
        cls.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cls.img = img

    def test_bbox_matches_ground_truth(self):
        bb = detect_sample_panel_bbox(self.gray)
        self.assertIsNotNone(bb, "应检出实体样块边界")
        top, bottom, left, right = bb
        # 真值：rows 48~52 / 968~970，cols 75~78 / 1454~1458（容差 ±6px）
        self.assertLessEqual(abs(top - 49), 6, f"上边应≈49，实际 {top}")
        self.assertLessEqual(abs(bottom - 968), 6, f"下边应≈968，实际 {bottom}")
        self.assertLessEqual(abs(left - 76), 6, f"左边应≈76，实际 {left}")
        self.assertLessEqual(abs(right - 1457), 6, f"右边应≈1457，实际 {right}")

    def test_roi_is_inset_not_full_canvas(self):
        roi = sample_panel_roi(self.gray)
        self.assertIsNotNone(roi)
        frac = float(roi.mean())
        # 样块应是画幅主体但非全幅（墙面底衬占比实测 ≈19.1%）
        self.assertGreater(frac, 0.70)
        self.assertLess(frac, 0.95)
        # 四角必在样块之外（背景带）
        self.assertFalse(roi[0, 0])
        self.assertFalse(roi[-1, -1])

    def test_bgr_config_resolver(self):
        roi = resolve_sample_panel_roi(self.img, {"enabled": True})
        self.assertIsNotNone(roi)
        self.assertEqual(roi.shape, self.gray.shape)


class TestDetectSamplePanelSynthetic(unittest.TestCase):
    def test_smooth_panel_detected(self):
        bb = detect_sample_panel_bbox(_smooth_panel())
        self.assertIsNotNone(bb)
        top, bottom, left, right = bb
        for got, exp in ((top, 60), (bottom, 340), (left, 90), (right, 510)):
            self.assertLessEqual(abs(got - exp), 3, f"期望≈{exp} 实际 {got}")

    def test_high_frequency_texture_falls_back(self):
        """内外皆高频纹理时无法可靠区分 → 必须安全回退 None（不得乱切）。"""
        rng = np.random.default_rng(0)
        img = rng.integers(80, 120, (400, 600)).astype(np.uint8)
        img[60:340, 90:510] = rng.integers(160, 200, (280, 420)).astype(np.uint8)
        self.assertIsNone(detect_sample_panel_bbox(img))

    def test_uniform_and_noise_return_none(self):
        self.assertIsNone(detect_sample_panel_bbox(np.full((400, 600), 128, np.uint8)))
        rng = np.random.default_rng(1)
        self.assertIsNone(detect_sample_panel_bbox(rng.integers(0, 255, (400, 600)).astype(np.uint8)))

    def test_panel_filling_canvas_returns_none(self):
        """样块占满画幅（无外部）→ 无背景带可言，返回 None。"""
        self.assertIsNone(detect_sample_panel_bbox(_smooth_panel(top=2, bottom=398, left=2, right=598)))

    def test_tiny_image_returns_none(self):
        self.assertIsNone(detect_sample_panel_bbox(np.zeros((8, 8), np.uint8)))


class TestResolveConfig(unittest.TestCase):
    def test_disabled_returns_none(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertIsNone(resolve_sample_panel_roi(img, {"enabled": False}))
        self.assertIsNone(resolve_sample_panel_roi(img, None))

    def test_shape_preserved(self):
        img = np.zeros((400, 600, 3), np.uint8)
        roi = resolve_sample_panel_roi(img, {"enabled": True})
        # 纯色图无线索 → None（回退全画幅由调用方处理）
        self.assertIsNone(roi)


class TestSamplePanelWiring(unittest.TestCase):
    """wiring 教训（勿重蹈「单测全绿 ≠ 功能可用」）：断言模块已接生产调用点。"""

    def test_grounded_sam_accepts_and_forwards_roi_mask(self):
        import inspect
        from engine.providers.grounded_sam_provider import GroundedSAMProvider
        sig = inspect.signature(GroundedSAMProvider.segment_objects)
        self.assertIn("roi_mask", sig.parameters, "segment_objects 必须接受 roi_mask")
        src = inspect.getsource(GroundedSAMProvider.segment_objects)
        self.assertIn("painting_roi=roi_mask", src,
                      "roi_mask 必须下传规则分割器（否则背景带层为空）")

    def test_engine_passes_panel_roi(self):
        src = (ROOT / "run_universal_engine.py").read_text(encoding="utf-8")
        self.assertIn("resolve_sample_panel_roi", src,
                      "引擎必须解析 sample_panel 配置")
        self.assertIn("roi_mask=panel_roi", src,
                      "引擎必须把样块 ROI 传给 segment_objects")

    def test_background_band_exempt_from_diffuse_gate(self):
        # 背景带天然跨全画布（bbox 覆盖 100%），若不豁免会被弥散门误杀 →
        # 方案 C 的核心交付（背景带隔离层）落不了地。
        from engine.providers.grounded_sam_provider import is_bbox_exempt
        self.assertTrue(is_bbox_exempt("01A_画面外背景带_Photo_Background"),
                        "「画面外背景带」必须在弥散门豁免名单内")

    def test_preset_exists_and_valid(self):
        import json
        from engine.schemas.preset_schema import validate_preset
        p = ROOT / "presets" / "textile_damask_photo.json"
        self.assertTrue(p.is_file(), "缺少 textile_damask_photo preset")
        d = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(validate_preset(d), [], "preset 校验未通过")
        # 关键：关闭接缝对齐（样品照不可平铺）+ 锁定模式（规避 adaptive 覆盖，根因③）
        self.assertFalse(d["seam_harmonization"]["enabled"])
        self.assertEqual(d.get("mode"), "locked")
        self.assertTrue(d["sample_panel"]["enabled"])
        # 白名单须包含「画面外背景带」与团花/卷草（否则会重蹈零掩模）
        allow = set(d["rule_class_allowlist"])
        self.assertIn("01A_画面外背景带_Photo_Background", allow)
        # name_mapping 必须把规则引擎外框名映射为背景带名（键须小写）
        self.assertEqual(
            d["layer_semantics"]["name_mapping"].get("11a_brocade_outer_frame"),
            "01A_画面外背景带_Photo_Background")


if __name__ == "__main__":
    unittest.main()
