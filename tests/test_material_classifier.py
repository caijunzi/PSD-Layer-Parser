"""材质判别回归测试（2026-09-15 金地修复）。

背景：扫描件外框是博物馆灰底（实测 source_4000.jpg 外框 L37/b0），
而画心金地是 L80/b38；旧规则用外框当"背景"，导致金地屏风被误判为油画布。
修复：金地规则改用中心主体区统计（fingerprint.center_median_LAB）。
"""
import sys
import unittest
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.adaptive.material_classifier import classify_material_family


def _fp(bg=(50.0, 0.0, 0.0), center=None, sat=20.0, csat=None, edge=0.06, glcm=0.30):
    g = {"bg_median_LAB": tuple(bg), "saturation": float(sat)}
    if center is not None:
        g["center_median_LAB"] = tuple(center)
        g["center_saturation"] = float(csat if csat is not None else sat)
    return {
        "global_features": g,
        "texture_features": {"glcm_energy": float(glcm)},
        "structure_features": {"edge_density": float(edge)},
    }


class TestGoldScreenClassification(unittest.TestCase):
    def test_gold_screen_with_dark_museum_border(self):
        """外框深灰 + 画心金地 → 必须判为「金地屏风」（本次修复的核心场景）。"""
        fp = _fp(bg=(37.3, 0.0, 0.0), center=(79.2, 5.0, 38.0), sat=24.8, csat=34.9, edge=0.069)
        fam, conf = classify_material_family(fp)
        self.assertEqual(fam, "金地屏风")
        self.assertGreaterEqual(conf, 0.9)

    def test_backward_compat_without_center_keys(self):
        """旧指纹（无 center_* 键）应回退到外框口径且不报错。"""
        fp = _fp(bg=(79.2, 5.0, 38.0), sat=34.9, edge=0.069)
        fam, conf = classify_material_family(fp)
        self.assertEqual(fam, "金地屏风")
        self.assertGreater(conf, 0.5)

    def test_dark_center_is_not_gold(self):
        """中心区暗淡无色 → 不得判为金地（防止把暗底图误当金地）。"""
        fp = _fp(bg=(37.0, 0.0, 0.0), center=(40.0, 0.0, 0.0), sat=10.0, csat=10.0, edge=0.02)
        fam, _ = classify_material_family(fp)
        self.assertNotEqual(fam, "金地屏风")

    def test_achromatic_paper_not_gold(self):
        """中性纸白（b*≈0）→ 应判「宣纸水墨」而非金地。"""
        fp = _fp(bg=(93.0, 0.0, 0.0), center=(93.0, 0.0, 0.0), sat=5.0, csat=5.0, edge=0.02, glcm=0.10)
        fam, _ = classify_material_family(fp)
        self.assertEqual(fam, "宣纸水墨")

    def test_missing_required_fields_degrades(self):
        """缺必需字段 → 兜底「其他」，不抛异常。"""
        fam, conf = classify_material_family({})
        self.assertEqual(fam, "其他")

    def test_ink_wash_landscape_is_ink_paper(self):
        """写意水墨山水（淡黄纸地 + 稀疏笔触）→ 宣纸水墨，而非绢本工笔。

        实测样本 inputs/宋代写意山水画创作.jpeg：中心 L69.8 b15.0 edge0.132 glcm0.236。
        """
        fp = _fp(bg=(81.6, 2.0, 20.0), center=(69.8, 1.0, 15.0), sat=16.2, csat=14.5,
                 edge=0.132, glcm=0.236)
        fam, _ = classify_material_family(fp)
        self.assertEqual(fam, "宣纸水墨")

    def test_dense_fine_texture_is_silk(self):
        """细腻密集纹理（米黄 + 高 edge/glcm）→ 绢本工笔。

        实测样本 inputs/damask_sample.png：中心 L73.3 b13.0 edge0.203 glcm0.337。
        """
        fp = _fp(bg=(72.9, 2.0, 13.0), center=(73.3, 2.0, 13.0), sat=13.8, csat=14.0,
                 edge=0.203, glcm=0.337)
        fam, _ = classify_material_family(fp)
        self.assertEqual(fam, "绢本工笔")


if __name__ == "__main__":
    unittest.main()
