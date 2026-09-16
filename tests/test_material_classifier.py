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

from engine.adaptive.material_classifier import (  # noqa: E402
    classify_material_family,
    get_material_families,
)

#: LBP 直方图（16 bin）：均匀分布 → 熵 ln16≈2.77（**有结构**，真实织物/绘画实测 2.18~2.63）
LBP_STRUCT = [1] * 16
#: 近似退化直方图 → 熵≈0.6（噪声实测 1.32、纯色 0.76，均 <2.0 结构门）
LBP_FLAT = [15] + [0] * 15


def _fp(bg=(50.0, 0.0, 0.0), center=None, sat=20.0, csat=None, edge=0.06, glcm=0.30,
        lbp_hist=None):
    g = {"bg_median_LAB": tuple(bg), "saturation": float(sat)}
    if center is not None:
        g["center_median_LAB"] = tuple(center)
        g["center_saturation"] = float(csat if csat is not None else sat)
    t = {"glcm_energy": float(glcm)}
    if lbp_hist is not None:
        t["lbp_hist"] = list(lbp_hist)
    return {
        "global_features": g,
        "texture_features": t,
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

    def test_ming_ink_wash_is_ink_not_silk(self):
        """明代写意山水（稀疏 + glcm 偏高）→ 宣纸水墨，**不得**判为绢本工笔。

        实测 inputs/明代写意山水画创作.jpeg：中心 L84.3 b13.0 edge0.1229 glcm0.325。
        关键：glcm 项须以 edge 稠密为前提（否则被误判绢本）。
        """
        fp = _fp(bg=(87.5, 0.0, 15.0), center=(84.3, 0.0, 13.0), sat=14.0, csat=14.1,
                 edge=0.1229, glcm=0.325)
        fam, _ = classify_material_family(fp)
        self.assertEqual(fam, "宣纸水墨")

    def test_oil_painting_is_oil_canvas(self):
        """油画（高饱和 + 厚重笔触 + 低 GLCM）→ 油画布。

        实测 inputs/油画.jpeg：中心 L52.2 a7.0 b27.0 sat28.1 edge0.241 glcm0.159。
        """
        fp = _fp(bg=(63.9, 4.0, 16.0), center=(52.2, 7.0, 27.0), sat=27.0, csat=28.1,
                 edge=0.241, glcm=0.159)
        fam, _ = classify_material_family(fp)
        self.assertEqual(fam, "油画布")
        # 排除金地误判（b* 27 偏高，但饱和度/亮度组合应归油画）
        self.assertNotEqual(fam, "金地屏风")


class TestFabricWallcoveringClassification(unittest.TestCase):
    """织物壁布族（2026-09-16 新增）。

    背景：分类器此前只有 4 个绘画族，`inputs/工艺壁布-*.jpeg`（绗缝织物实物照）
    被强行塞进绘画族 —— 实测三张**全被误判为「宣纸水墨」**，进而推荐错的 preset。
    织物判据 = **近中性**（sat<15 且 b*<15 合取），与四种绘画基材正交。
    """

    def test_three_real_samples_are_fabric(self):
        """三张工艺壁布真值（中心区实测值）必须判为「织物壁布」。"""
        cases = [
            # (tag, center(L,a,b), center_sat, edge, glcm)
            ("工艺壁布-1", (45.9, 2.0, 7.0), 7.1, 0.133, 0.173),
            ("工艺壁布-2", (69.0, 1.0, 9.0), 10.7, 0.385, 0.207),
            ("工艺壁布-3", (84.3, 0.0, 5.0), 7.0, 0.098, 0.325),
        ]
        for tag, center, csat, edge, glcm in cases:
            fp = _fp(bg=center, center=center, sat=csat, csat=csat, edge=edge, glcm=glcm,
                     lbp_hist=LBP_STRUCT)
            fam, conf = classify_material_family(fp)
            self.assertEqual(fam, "织物壁布", f"{tag} 应判织物壁布，实际 {fam}({conf:.2f})")
            self.assertGreaterEqual(conf, 0.70, f"{tag} 置信度应 ≥0.70")

    def test_structure_gate_rejects_noise_and_flat(self):
        """结构门（LBP 熵≥2）：「中性但无结构」不得判织物。

        实测：纯随机噪声 LBP 熵 1.32、灰噪声 1.36、平滑纯色 0.76（均 <2.0）；
        而真织物 2.18~2.47。缺此门时"中性即织物"会把噪声/纯色也吞进来。
        """
        for tag, edge in (("纯随机噪声", 0.3705), ("灰噪声", 0.2804), ("平滑纯色", 0.0)):
            fp = _fp(bg=(54.0, 7.0, 5.0), center=(54.0, 7.0, 5.0), sat=8.0, csat=8.0,
                     edge=edge, glcm=0.146, lbp_hist=LBP_FLAT)
            fam, _ = classify_material_family(fp)
            self.assertNotEqual(fam, "织物壁布", f"{tag} 无结构纹理，不应判织物")

    def test_painting_samples_not_reclassified_as_fabric(self):
        """四种绘画基材真值不得被新族吸走（防回归）。"""
        cases = [
            ("绢本-1", (89.4, 0.0, 16.0), 15.3, 0.0915, 0.429, "绢本工笔"),
            ("水墨宋代", (69.8, 1.0, 15.0), 14.5, 0.132, 0.236, "宣纸水墨"),
            ("金地屏风", (79.2, 5.0, 38.0), 34.9, 0.069, 0.310, "金地屏风"),
            ("油画", (52.2, 7.0, 27.0), 28.1, 0.241, 0.159, "油画布"),
        ]
        for tag, center, csat, edge, glcm, expect in cases:
            fp = _fp(bg=center, center=center, sat=csat, csat=csat, edge=edge, glcm=glcm)
            fam, _ = classify_material_family(fp)
            self.assertEqual(fam, expect, f"{tag} 应仍判 {expect}，实际 {fam}")
            self.assertNotEqual(fam, "织物壁布", f"{tag} 不得被误判为织物壁布")

    def test_conjunction_gate_blocks_warm_low_sat(self):
        """硬合取门：暖黄低饱和（b* 高）不得判织物（否则会吸走暖色纸/绢）。"""
        fp = _fp(bg=(70.0, 2.0, 22.0), center=(70.0, 2.0, 22.0), sat=9.0, csat=9.0,
                 edge=0.12, glcm=0.30)
        fam, _ = classify_material_family(fp)
        self.assertNotEqual(fam, "织物壁布", "sat 低但 b*=22 偏暖，不应判织物")

    def test_families_list_contains_fabric(self):
        from engine.adaptive.material_classifier import get_material_families
        self.assertIn("织物壁布", get_material_families())


class TestFabricClassificationOnRealFiles(unittest.TestCase):
    """端到端：直接跑真实样本文件（缺失则跳过）。"""

    def test_real_wallcovering_files(self):
        from engine.core.io_utils import imread_unicode
        from engine.adaptive.fingerprint import extract_fingerprint
        d = ENGINE_ROOT / "inputs"
        files = sorted(d.glob("工艺壁布-*.jpeg"))
        if not files:
            self.skipTest("缺少 inputs/工艺壁布-*.jpeg")
        checked = 0
        for p in files:
            img = imread_unicode(str(p))
            if img is None:
                continue
            fam, conf = classify_material_family(extract_fingerprint(img))
            self.assertEqual(fam, "织物壁布", f"{p.name} 实际 {fam}({conf:.2f})")
            checked += 1
        if checked == 0:
            self.skipTest("工艺壁布样本读取失败")


if __name__ == "__main__":
    unittest.main()
