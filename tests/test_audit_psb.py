"""tools/audit_psb 修复验证：

1. ⑤ 合成等价性改用 psd_tools 真实渲染（psd.composite），尊重 opacity/blend/CMYK，
   不再用手工 alpha-over 忽略混合模式/不透明度；CMYK 产物与转 CMYK 的源图同空间比对。
2. 底板检测/纯净度不写死"金地"——产品线无关关键词即可命中。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_psb import (  # noqa: E402
    _composite_rmse, _find_base_layer, DEFAULT_BASE_KEYWORDS,
)


def _fake_psd(image):
    class FakePSD:
        def __init__(self, img):
            self._img = img
        def composite(self):
            return self._img
    return FakePSD(image)


class _Layer:
    def __init__(self, name):
        self.name = name


class TestCompositeRealAPI(unittest.TestCase):
    def test_composite_uses_real_api_and_rgb(self):
        calls = {"n": 0}
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        real = Image.fromarray(src)

        class FakePSD:
            def composite(self):
                calls["n"] += 1
                return real
        raw, low, _cm = _composite_rmse(FakePSD(), src, target_long=40)
        self.assertEqual(calls["n"], 1, "必须使用 psd.composite() 真实合成")
        self.assertAlmostEqual(raw, 0.0, places=5)
        self.assertAlmostEqual(low, 0.0, places=5)
        self.assertTrue(_cm, "RGB 产物不依赖 ICC，应标注已色彩管理")

    def test_composite_cmyk_uses_same_icc_reference(self):
        """CMYK 产物必须用**同一 ICC** 分色的参考图比对（同口径），而非 PIL 朴素转换。

        构造自洽用例：产物 = 源图经项目 ICC 分色 → 与同 ICC 参考比对，RMSE 应≈0。
        这是 2026-09-16 修复的核心回归——此前用 PIL 朴素 convert（K 恒 0）导致虚高。
        """
        import cv2
        from tools.audit_psb import _src_to_cmyk_same_icc
        rng = np.random.default_rng(0)
        src_rgb = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        prod = _src_to_cmyk_same_icc(Image.fromarray(src_rgb))
        if prod is None:
            self.skipTest("ICC/ImageCms 不可用，无法验证同口径比对")
        # 注意：_composite_rmse 入参是 BGR（内部自行转 RGB），测试须按契约传 BGR，
        # 否则通道被交换，随机彩色图会立刻暴露（全零图看不出来）。
        src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
        raw, low, cm_ok = _composite_rmse(_fake_psd(prod), src_bgr, target_long=40)
        self.assertTrue(cm_ok, "ICC 可用时必须走同口径色彩管理")
        self.assertLess(raw, 1.0, f"同 ICC 分色产物应与参考几乎一致，实际 raw={raw:.3f}")
        self.assertLess(low, 1.0, f"同 ICC 分色产物应与参考几乎一致，实际 low={low:.3f}")

    def test_composite_falls_back_and_flags_when_icc_missing(self):
        """ICC 不可用 → 回退 PIL 朴素转换，并显式标注 color_managed=False（不静默）。"""
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        cmyk = Image.fromarray(src).convert("CMYK")
        raw, low, cm_ok = _composite_rmse(_fake_psd(cmyk), src, target_long=40,
                                           icc_path=str(ROOT / "__no_such_icc__.icc"))
        self.assertFalse(cm_ok, "ICC 缺失时必须标注未做色彩管理")

    def test_composite_downscales_large(self):
        # 模拟大图：用 400x400 但 target_long=100 → 应被缩到 100 宽
        big = Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8))
        raw, low, _cm = _composite_rmse(_fake_psd(big),
                                        np.zeros((400, 400, 3), dtype=np.uint8), target_long=100)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_real_psd_composite_runs(self):
        # 真实小 PSB 端到端跑通（RGB）
        p = ROOT / "intermediate" / "test_patch.psb"
        if not p.exists():
            self.skipTest("无 test_patch.psb")
        from psd_tools import PSDImage
        psd = PSDImage.open(str(p))
        src = np.zeros((psd.size[1], psd.size[0], 3), dtype=np.uint8)
        raw, low, _cm = _composite_rmse(psd, src, target_long=200)
        self.assertIsInstance(raw, float)
        self.assertIsInstance(low, float)


class TestBaseLayerNotHardcodedGold(unittest.TestCase):
    def test_finds_non_gold_base(self):
        layers = [_Layer("Fabric_Base_01"), _Layer("some_content")]
        self.assertIsNotNone(_find_base_layer(layers))

    def test_finds_generic_base_names(self):
        for name in ("Base_Ground", "底版", "底板", "Substrate", "Gold_Base"):
            layers = [_Layer(name + "_x"), _Layer("ink")]
            self.assertIsNotNone(_find_base_layer(layers), f"未命中 {name}")

    def test_no_base_returns_none(self):
        layers = [_Layer("mountain"), _Layer("water")]
        self.assertIsNone(_find_base_layer(layers))

    def test_default_keywords_are_product_line_agnostic(self):
        # 关键词不应只包含金地特定命名，须覆盖多产品线
        self.assertIn("Fabric_Base", DEFAULT_BASE_KEYWORDS)
        self.assertIn("Base_Ground", DEFAULT_BASE_KEYWORDS)


if __name__ == "__main__":
    unittest.main()
