"""金标准回归：锁住 japanese_screen_gold 的分层配置与可靠性机制。

目的（2026-09-11 验收）：preset 是品类语义唯一真相源（G2/SSOT），
配置漂移会让产物层结构悄悄退化（曾出现 15 层 → 11 层的迁移遗漏）。
本测试把"期望的层结构与质量机制"固化为断言——preset 任何改动必须过此回归。
"""
import sys
import unittest
from pathlib import Path

import json

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.providers.grounded_sam_provider import (
    MAX_BBOX_AREA_RATIO,
    is_bbox_exempt,
)

PRESET = Path(ENGINE_ROOT, "presets", "japanese_screen_gold.json")


def _load():
    return json.loads(PRESET.read_text(encoding="utf-8"))


class TestGoldenScreenGold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = _load()
        cls.names = [c["name"] for c in cls.cfg["ai_semantic_classes"]]

    def test_semantic_classes_cover_required_objects(self):
        """你点名的对象类必须在配置中（缺类=缺层）。"""
        for prefix in ("06_水榭草堂建筑", "07_高士侍童人物", "08_芦雁群禽",
                       "09A_长泽芦雪朱红印章", "09B_题跋落款墨书", "04C_湖心独立孤石"):
            self.assertTrue(any(n.startswith(prefix) for n in self.names),
                            f"缺语义类: {prefix}")

    def test_landscape_elements_configured(self):
        """山水元素类（远山/峭壁/寒林/水波/渚上水木）必须配置，
        否则内容会静默留在底板无独立层。"""
        for prefix in ("04A_前景墨岩峭壁", "04D_远山淡墨晴岚", "05A_前景寒林枯木",
                       "03_水波微澜墨纹", "05B_中景渚上水木"):
            self.assertTrue(any(n.startswith(prefix) for n in self.names),
                            f"缺语义/密度带类: {prefix}")

    def test_no_duplicate_class_names(self):
        self.assertEqual(len(self.names), len(set(self.names)))

    def test_diffuse_gate_enabled(self):
        """弥散门必须开启（曾因缺失导致右半屏雾化污染）。"""
        self.assertLessEqual(MAX_BBOX_AREA_RATIO, 0.5)
        for exempt in ("02_纯净金箔大底板_Gold_Base_Clean",
                       "10A_外框与织锦绫边_Brocade_Outer_Frame",
                       "10B_屏风折痕折缝_Panel_Fold_Seams"):
            self.assertTrue(is_bbox_exempt(exempt))
        self.assertFalse(is_bbox_exempt("04A_前景墨岩峭壁_Foreground_Dark_Cliffs"))

    def test_density_refine_configured(self):
        """密度精修通道：04A 峭壁已实证救回（bbox 74.4%→16.4%）。"""
        dr = self.cfg["density_refine"]
        self.assertTrue(dr["enabled"])
        self.assertIn("04A_前景墨岩峭壁_Foreground_Dark_Cliffs", dr["classes"])
        self.assertAlmostEqual(
            dr["classes"]["04A_前景墨岩峭壁_Foreground_Dark_Cliffs"]["floor"], 0.12)

    def test_density_band_configured(self):
        """密度带通道：03 水波（DINO 未命中的纹理类）由区域+密度产层。"""
        band = [b["name"] for b in self.cfg.get("density_band_classes", [])]
        self.assertIn("03_水波微澜墨纹_Water_Ripples", band)
        for b in self.cfg["density_band_classes"]:
            self.assertIn("region", b)
            self.assertLess(b["density_min"], b["density_max"])

    def test_region_sam_configured(self):
        """区域先验 SAM（P1/P2 实证）：05A 寒林 75.1%→21.6%、04B 平渚 17.5% 产出。"""
        for prefix in ("05A_", "04B_"):
            cls = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith(prefix))
            self.assertIn("region", cls, f"{prefix} 必须配置 region（否则 SAM 弥散/未命中）")
            x0, y0, x1, y1 = cls["region"]
            self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
            self.assertGreaterEqual(int(cls.get("region_boxes", 1)), 2)

    def test_distant_mountain_documented_unavailable(self):
        """04D 远山：P3 三种 region 尝试均失败（外框假阳性/弥散）——必须无 region 且留有结论记录。"""
        dm = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith("04D_"))
        self.assertNotIn("region", dm, "04D 不可配 region（实测产出错误层）")
        self.assertIn("note", dm, "04D 必须保留不可行结论记录")

    def test_instance_split_for_flock(self):
        """雁群按实例拆分（用户要求"尽可能细致"）。"""
        flock = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith("08_"))
        self.assertTrue(flock.get("instance_split"))

    def test_output_policy_present(self):
        """输出策略（双线模式/复现尺度/DPI/ICC/印刷条件）必须在 preset 中显式声明。"""
        self.assertIn("mode", self.cfg["output"])
        self.assertIn(self.cfg["output"]["mode"], ("design", "plate", "both"))
        self.assertIn("icc_path", self.cfg)
        self.assertIn("print_condition", self.cfg)
        self.assertAlmostEqual(float(self.cfg["dpi"]), 150.0)
        self.assertAlmostEqual(float(self.cfg["super_res_scale"]), 4.0)
        self.assertEqual(int(self.cfg["target_w"]), 16000)


if __name__ == "__main__":
    unittest.main()
