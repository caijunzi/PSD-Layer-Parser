"""交付清单（manifest）契约单元测试。

覆盖双产品线隔离的关键不变量：
  - OutputMode 解析与非法值拒绝
  - 诚实指标（effective_source_ppi / upscale_factor）计算正确
  - 生成内容占比按层面积加权
  - PLATE 纯净性校验必须拒绝任何生成内容（§3.1 硬边界 1）
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.schemas.manifest import (  # noqa: E402
    GENERATIVE_UPSCALE_ALLOWED,
    LayerGenerationRecord,
    OutputMode,
    build_manifest,
    load_manifest,
)


class TestOutputMode(unittest.TestCase):
    def test_parse_valid(self):
        self.assertEqual(OutputMode.parse("plate"), OutputMode.PLATE)
        self.assertEqual(OutputMode.parse("DESIGN"), OutputMode.DESIGN)
        self.assertEqual(OutputMode.parse(" both "), OutputMode.BOTH)

    def test_parse_default(self):
        self.assertEqual(OutputMode.parse(None), OutputMode.DESIGN)
        self.assertEqual(OutputMode.parse(""), OutputMode.DESIGN)

    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            OutputMode.parse("cmyk")

    def test_plate_forbids_generative_upscale(self):
        """§3.1 硬边界 1：PLATE 线不得使用生成式超分。"""
        self.assertFalse(GENERATIVE_UPSCALE_ALLOWED[OutputMode.PLATE])
        self.assertTrue(GENERATIVE_UPSCALE_ALLOWED[OutputMode.DESIGN])


class TestHonestMetrics(unittest.TestCase):
    def _build(self, mode="design"):
        return build_manifest(
            run_id="test-1",
            output_mode=mode,
            source_path="inputs/source_4000.jpg",
            source_wh=(4000, 1952),
            output_path="outputs/x.psb",
            output_wh=(16000, 7808),
            ppi=150.0,
            color_mode="rgb",
        )

    def test_effective_ppi(self):
        """4000px 源 / 2709.3mm 物理宽 = 37.5 PPI。"""
        m = self._build()
        self.assertAlmostEqual(m.source["effective_ppi"], 37.5, places=1)
        self.assertAlmostEqual(m.output["physical_width_mm"], 2709.3, places=0)

    def test_upscale_factor(self):
        m = self._build()
        self.assertAlmostEqual(m.output["upscale_factor_x"], 4.0, places=3)
        self.assertAlmostEqual(m.output["upscale_factor_y"], 4.0, places=3)

    def test_declare_stage(self):
        m = self._build()
        m.generation_policy.declare("super_resolution", "realesrgan:ov", True, 1.0)
        self.assertTrue(m.generation_policy.stages["super_resolution"]["generative"])
        self.assertEqual(m.generation_policy.stages["super_resolution"]["coverage_ratio"], 1.0)


class TestGenerationRatio(unittest.TestCase):
    def test_ratio_weighted_by_layer_area(self):
        m = build_manifest(
            run_id="t", output_mode="design",
            source_path="s", source_wh=(100, 100),
            output_path="o", output_wh=(100, 100),
            ppi=150.0, color_mode="rgb",
        )
        # 单层覆盖全画幅、其内部 50% 为重建
        rec = LayerGenerationRecord(name="L1", bbox=(0, 0, 100, 100),
                                    generated=True, recon_ratio=0.5)
        rec.add_reason("deocclusion")
        m.layers.append(rec)
        self.assertAlmostEqual(m.generated_ratio(), 0.5, places=6)


class TestPlatePurity(unittest.TestCase):
    def _manifest(self, mode):
        return build_manifest(
            run_id="t", output_mode=mode,
            source_path="s", source_wh=(100, 100),
            output_path="o", output_wh=(100, 100),
            ppi=150.0, color_mode="cmyk",
        )

    def test_plate_with_generated_layer_fails(self):
        m = self._manifest("plate")
        rec = LayerGenerationRecord(name="L1")
        rec.add_reason("deocclusion")
        m.layers.append(rec)
        ok, msg = m.assert_plate_purity()
        self.assertFalse(ok)
        self.assertIn("生成内容", msg)

    def test_plate_with_generative_stage_fails(self):
        m = self._manifest("plate")
        m.generation_policy.declare("super_resolution", "realesrgan:ov", True, 1.0)
        ok, msg = m.assert_plate_purity()
        self.assertFalse(ok)
        self.assertIn("super_resolution", msg)

    def test_pure_plate_passes(self):
        m = self._manifest("plate")
        rec = LayerGenerationRecord(name="L1")
        m.layers.append(rec)
        m.generation_policy.declare("super_resolution", "lanczos_guided_non_generative", False, 1.0)
        ok, msg = m.assert_plate_purity()
        self.assertTrue(ok, msg)

    def test_design_not_restricted(self):
        m = self._manifest("design")
        rec = LayerGenerationRecord(name="L1")
        rec.add_reason("super_resolution(realesrgan)")
        m.layers.append(rec)
        ok, _ = m.assert_plate_purity()
        self.assertTrue(ok)


class TestRoundTrip(unittest.TestCase):
    def test_save_and_load(self):
        m = build_manifest(
            run_id="rt", output_mode="design",
            source_path="s.jpg", source_wh=(4000, 1952),
            output_path="o.psb", output_wh=(16000, 7808),
            ppi=150.0, color_mode="rgb",
        )
        rec = LayerGenerationRecord(name="L1", bbox=(0, 0, 10, 10))
        rec.add_reason("deocclusion")
        m.layers.append(rec)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "o.manifest.json")
            m.save(p)
            self.assertTrue(os.path.isfile(p))
            raw = load_manifest(p)
            self.assertEqual(raw["manifest_version"], "1.0")
            self.assertEqual(raw["output_mode"], "design")
            self.assertEqual(raw["layers"][0]["name"], "L1")
            self.assertTrue(raw["layers"][0]["generated"])
            self.assertIn("effective_source_ppi", raw["totals"])
            # 确保可 JSON 序列化且中文不转义
            with open(p, encoding="utf-8") as f:
                self.assertIn("effective_source_ppi", f.read())


class TestMaskIO(unittest.TestCase):
    """掩码读写必须支持中文路径。

    实测：cv2.imwrite 在含中文的 Windows 路径下**返回 True 但文件并不存在**，
    cv2.imread 则返回 None。写入/读取都必须走 imencode/imdecode 路径。
    """

    def test_roundtrip_non_ascii_path(self):
        import numpy as np

        from engine.schemas.manifest import read_mask_png, write_mask_png

        m = np.zeros((64, 80), np.uint8)
        m[10:30, 20:50] = 255
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "06_水榭草堂建筑_Architecture_Pavilion.recon.png")
            self.assertTrue(write_mask_png(p, m))
            self.assertTrue(os.path.isfile(p), "中文路径写入失败（cv2.imwrite 的静默失败重现）")
            back = read_mask_png(p)
            self.assertIsNotNone(back)
            self.assertTrue(np.array_equal(back, m))

    def test_read_missing_returns_none(self):
        from engine.schemas.manifest import read_mask_png

        self.assertIsNone(read_mask_png("no/such/mask.png"))


if __name__ == "__main__":
    unittest.main()
