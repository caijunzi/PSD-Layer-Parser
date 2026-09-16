# -*- coding: utf-8 -*-
"""Preset 语义契约测试 —— 2026-09-16 新增。

来历
----
`#49` 事故的核心是 **`rule_class_allowlist` 与 preset 实际产出名不相交** → 静默 0 掩模；
随后又出现反向诱惑：把规则引擎的**屏风系外来内容类目**（书法题跋/雁群/建筑/人物…）
塞进壁布 preset 的名单里"让它通过" —— 那等于产出**命名错误的内容层**（垃圾产物）。

本文件把两条原则钉死：
  A. **名单必须"可产出"**：`rule_class_allowlist` 的每一项都应能在该 preset 下真的产生
     （= preset 语义类目名 ∪ `name_mapping` 的值）。孤项 = 永远不会命中的死配置。
  B. **名单不得含外来内容类目**：壁布/织物类 preset 不得放屏风系内容类目。
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRESETS_DIR = ROOT / "presets"

#: 屏风系「内容类目」（规则引擎 `UniversalSemanticSegmenter` 的内置产出名）。
#: 它们对壁布/织物/油画等品类**语义错误**；放进这些 preset 的名单即为"假通过"。
SCREEN_CONTENT_CLASSES = {
    "02_Water_Ripples", "03_Distant_Mountains", "04_Mountains_Cliffs_Shorelines",
    "05_Trees_Vegetation", "06_Architecture_Pavilion", "07_Figures_Scholar_Attendants",
    "08_Fauna_Geese", "09_Calligraphy_Inscription", "10_Cinnabar_Seal",
}

#: 「介质结构类」= 与外框/折缝等**物理载体结构**相关，属正当可复用检测器
MEDIUM_STRUCTURE_KEYS = {"11a_brocade_outer_frame", "11b_panel_fold_seams"}

#: 不应用于壁布/织物品类的 preset（其名单不得含屏风系内容类目）
TEXTILE_PRESETS = ("textile_damask", "textile_damask_photo")


def _load(name: str) -> dict:
    return json.loads((PRESETS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _all_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


class TestAllowlistProducible(unittest.TestCase):
    """原则 A：名单每一项都必须"可产出"，否则是永不命中的死配置。"""

    def test_every_allowlist_entry_is_producible(self):
        checked = 0
        for name in _all_presets():
            cfg = _load(name)
            allow = cfg.get("rule_class_allowlist")
            if not isinstance(allow, list) or not allow:
                continue
            semantic = {c.get("name") for c in (cfg.get("ai_semantic_classes") or [])
                        if isinstance(c, dict)}
            mapped = set((cfg.get("layer_semantics") or {}).get("name_mapping", {}).values())
            producible = semantic | mapped
            orphan = [a for a in allow if a not in producible]
            self.assertEqual(
                orphan, [],
                f"{name}: rule_class_allowlist 含无法产出的条目（永远不命中）：{orphan}\n"
                f"  可产出集合 = 语义类目 {sorted(semantic)} ∪ name_mapping 值 {sorted(mapped)}")
            checked += 1
        self.assertGreater(checked, 0, "未检查到任何含 rule_class_allowlist 的 preset")

    def test_name_mapping_keys_are_lowercase(self):
        """契约：`_map_to_bilingual_names` 以小写 clean_k 查表，键必须小写。"""
        for name in _all_presets():
            mapping = (_load(name).get("layer_semantics") or {}).get("name_mapping") or {}
            for k in mapping:
                self.assertEqual(k, k.lower(),
                                 f"{name}: name_mapping 键 {k!r} 非小写，永远不会命中")


class TestNoForeignContentClasses(unittest.TestCase):
    """原则 B：壁布/织物类 preset 不得放屏风系外来**内容**类目。"""

    def test_textile_presets_reject_screen_content_classes(self):
        for name in TEXTILE_PRESETS:
            path = PRESETS_DIR / f"{name}.json"
            if not path.is_file():
                continue
            cfg = _load(name)
            allow = set(cfg.get("rule_class_allowlist") or [])
            bad = allow & SCREEN_CONTENT_CLASSES
            self.assertEqual(
                bad, set(),
                f"{name}: 名单含屏风系内容类目 {sorted(bad)} —— "
                f"这类层对壁布语义错误，纳入即产出命名错误的垃圾层")

    def test_textile_damask_allowlist_is_semantic_plus_medium_structure(self):
        """textile_damask 重建后的名单 = 本 preset 语义类目 ∪ 介质结构类（外框/折缝）。"""
        cfg = _load("textile_damask")
        allow = set(cfg["rule_class_allowlist"])
        semantic = {"02_巴洛克团花_Baroque_Medallion", "03_金箔卷草纹样_Gold_Filigree_Motif"}
        medium = {"01A_画面外框_Wallcovering_Frame", "01B_折缝拼接缝_Wallcovering_Fold_Seam"}
        self.assertEqual(allow, semantic | medium,
                         "textile_damask 名单应为「语义类目 + 介质结构类」两个子集")
        # 介质结构层的映射键必须是已知的结构类检测器
        mapping = cfg["layer_semantics"]["name_mapping"]
        self.assertEqual(set(mapping.keys()), MEDIUM_STRUCTURE_KEYS)


class TestStructureLayersSurviveDiffuseGate(unittest.TestCase):
    """介质结构层天然跨全幅 → 必须命中弥散门豁免，否则被静默拒绝。"""

    def test_frame_and_seam_are_bbox_exempt(self):
        from engine.providers.grounded_sam_provider import is_bbox_exempt
        for name in ("01A_画面外框_Wallcovering_Frame",
                     "01B_折缝拼接缝_Wallcovering_Fold_Seam",
                     "01A_画面外背景带_Photo_Background"):
            self.assertTrue(is_bbox_exempt(name), f"{name} 必须豁免弥散门")


if __name__ == "__main__":
    unittest.main()
