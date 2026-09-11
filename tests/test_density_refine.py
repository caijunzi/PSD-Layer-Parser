"""密度场与精修通道单测（原型实证：04A bbox 74.4%→16.4%）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.core.density_field import ink_density, mask_stats, refine_mask
from engine.providers.grounded_sam_provider import GroundedSAMProvider

H, W = 1952, 4000


def synth_scene():
    """合成金地水墨场景：亮金地（~0.8）+ 浓墨岩壁块（~0.1）+ 淡墨远山带。"""
    rng = np.random.RandomState(7)
    gray = np.full((H, W), 0.80, dtype=np.float64) + rng.normal(0, 0.02, (H, W))
    # 浓墨岩壁（中右）
    gray[int(H*0.35):int(H*0.75), int(W*0.55):int(W*0.85)] = 0.08
    # 淡墨远山（上部，弱对比）
    yy = int(H*0.05)
    gray[yy:yy + int(H*0.12), int(W*0.3):int(W*0.7)] = 0.62
    return np.clip(gray, 0, 1)


class TestDensityField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gray = synth_scene()
        cls.D = ink_density(cls.gray, gold_percentile=88, sigma=4.0)

    def test_gold_is_low_density(self):
        """金地密度应接近 0（无墨）。"""
        gold = self.gray[0:int(H*0.02), :]
        self.assertLess(float(self.D[0:int(H*0.02), :].mean()), 0.15)

    def test_dense_rock_is_high_density(self):
        """浓墨岩壁中心密度应显著高于地板 0.12。"""
        rock = self.D[int(H*0.5), int(W*0.65):int(W*0.75)]
        self.assertGreater(float(rock.mean()), 0.25)

    def test_refine_removes_gold_noise(self):
        """精修：弥散掩模（含大量金地噪声）→ 密度地板剔除噪声保留墨迹。"""
        diffuse = np.zeros((H, W), dtype=np.uint8)
        rng = np.random.RandomState(1)
        # 弥散碎片散布全画布（含金地噪声区）
        for _ in range(500):
            y, x = rng.randint(0, H - 8), rng.randint(0, W - 8)
            diffuse[y:y + 8, x:x + 8] = 255
        # 岩壁墨迹（应保留）
        diffuse[int(H*0.45):int(H*0.7), int(W*0.6):int(W*0.8)] = 255
        s0 = mask_stats(diffuse)
        self.assertGreater(s0["bbox_ratio"], 0.5)  # 前置：弥散
        refined = refine_mask(diffuse, self.D, floor=0.12)
        s1 = mask_stats(refined)
        self.assertLess(s1["bbox_ratio"], 0.5)     # 精修后过弥散门
        # 岩壁墨迹保留
        kept = refined[int(H*0.45):int(H*0.7), int(W*0.6):int(W*0.8)]
        self.assertGreater(int(np.count_nonzero(kept > 127)), 1000)

    def test_refine_empty_on_pure_gold(self):
        """纯金地区域的掩模 → 精修后为空。"""
        pure = np.zeros((H, W), dtype=np.uint8)
        pure[0:50, 0:50] = 255  # 金地区小块
        refined = refine_mask(pure, self.D, floor=0.12)
        self.assertEqual(int(np.count_nonzero(refined > 127)), 0)


class TestRefineConfig(unittest.TestCase):

    def test_missing_config_disabled(self):
        prov = GroundedSAMProvider.__new__(GroundedSAMProvider)
        prov.preset_name = "textile_damask"  # 该 preset 无 density_refine
        cfg = prov._density_refine_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["classes"], {})

    def test_config_loaded(self):
        prov = GroundedSAMProvider.__new__(GroundedSAMProvider)
        prov.preset_name = "japanese_screen_gold"
        cfg = prov._density_refine_config()
        self.assertTrue(cfg["enabled"])
        self.assertIn("04A_前景墨岩峭壁_Foreground_Dark_Cliffs", cfg["classes"])
        self.assertEqual(cfg["classes"]["04A_前景墨岩峭壁_Foreground_Dark_Cliffs"]["floor"], 0.12)


if __name__ == "__main__":
    unittest.main()
