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
        raw, low = _composite_rmse(FakePSD(), src, target_long=40)
        self.assertEqual(calls["n"], 1, "必须使用 psd.composite() 真实合成")
        self.assertAlmostEqual(raw, 0.0, places=5)
        self.assertAlmostEqual(low, 0.0, places=5)

    def test_composite_handles_cmyk_by_matching_source_space(self):
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        cmyk = Image.fromarray(src).convert("CMYK")  # (0,0,0,0)
        raw, low = _composite_rmse(_fake_psd(cmyk), src, target_long=40)
        # CMYK 产物与转 CMYK 的源图(全 0)同空间 → RMSE 应为 0
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_composite_downscales_large(self):
        # 模拟大图：用 400x400 但 target_long=100 → 应被缩到 100 宽
        big = Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8))
        raw, low = _composite_rmse(_fake_psd(big),
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
        raw, low = _composite_rmse(psd, src, target_long=200)
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
