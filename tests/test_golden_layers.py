"""金标准回归：锁住 japanese_screen_gold 的分层配置与可靠性机制。

目的（2026-09-11 验收）：preset 是品类语义唯一真相源（G2/SSOT），
配置漂移会让产物层结构悄悄退化（曾出现 15 层 → 11 层的迁移遗漏）。
本测试把"期望的层结构与质量机制"固化为断言——preset 任何改动必须过此回归。

Stage 3 增强（2026-09-13）：核心类目必出 + 审计不退化
- 新增测试：验证核心语义类在实际运行中被检测到（基于 episode 归档）
- 新增测试：验证审计 8 维指标相对于回归基线不退化（调用 regression_tester.py）
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
from engine.adaptive.regression_tester import run_regression_test

PRESET = Path(ENGINE_ROOT, "presets", "japanese_screen_gold.json")
BASELINE = Path(ENGINE_ROOT, "tests", "baseline_audit_8d.json")


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
        """雁群逐只拆分：blob_instances（墨点连通域实例化，DINO 对 30-60px 雁漏检）。

        用户核心要求："能单独被提取的物类，应该都是单独层"。
        """
        flock = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith("08_"))
        self.assertTrue(flock.get("instance_split"))
        self.assertTrue(flock.get("blob_instances"), "雁群必须配 blob_instances 逐只成层")
        for cfg in flock["blob_instances"]:
            x0, y0, x1, y1 = cfg["region"]
            self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
            self.assertLess(cfg["min_blob_px"], cfg["max_blob_px"])

    def test_rock_instances_configured(self):
        """石矶多块拆分（region + instance_split，实测 8 块）。"""
        rock = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith("04C_"))
        self.assertTrue(rock.get("instance_split"))
        self.assertIn("region", rock)

    def test_tree_regions_multi(self):
        """枯树两丛各一层（regions 多区域 → 05A_01/_02）。"""
        tree = next(c for c in self.cfg["ai_semantic_classes"] if c["name"].startswith("05A_"))
        self.assertGreaterEqual(len(tree.get("regions", [])), 2)

    def test_output_policy_present(self):
        """输出策略（双线模式/复现尺度/DPI/ICC/印刷条件）必须在 preset 中显式声明。"""
        self.assertIn("mode", self.cfg["output"])
        self.assertIn(self.cfg["output"]["mode"], ("design", "plate", "both"))
        self.assertIn("icc_path", self.cfg)
        self.assertIn("print_condition", self.cfg)
        self.assertAlmostEqual(float(self.cfg["dpi"]), 150.0)
        self.assertAlmostEqual(float(self.cfg["super_res_scale"]), 4.0)
        self.assertEqual(int(self.cfg["target_w"]), 16000)

    # ==================== Stage 3 增强：核心类目必出 + 审计不退化 ====================

    def test_core_categories_must_be_detected(self):
        """核心类目必出：preset 中 core=true 的语义类，在**真实运行**的 episode 归档中
        必须全部被检测到（2026-09-16 启用——以脱敏的真实归档为受控夹具）。

        夹具 `tests/fixtures/episode_chinese_ink_golden.json` 来自一次真实引擎运行
        （chinese_ink_landscape_ai / 青绿山水_松亭瀑布渔舟_纸本设色 / design，8 维审计通过，路径已脱敏）。
        此前该测试永久 skipTest——因为归档在 gitignore 的 outputs/ 里；现以夹具入库，
        离线即可验证「core 类目不得静默漏检」。
        """
        import json
        fixture = ENGINE_ROOT / "tests" / "fixtures" / "episode_chinese_ink_golden.json"
        self.assertTrue(fixture.is_file(), "缺少 episode 夹具")
        ep = json.loads(fixture.read_text(encoding="utf-8"))

        preset_name = ep.get("preset")
        preset_path = Path(ENGINE_ROOT, "presets", f"{preset_name}.json")
        self.assertTrue(preset_path.is_file(), f"夹具引用的 preset 不存在: {preset_name}")
        cfg = json.loads(preset_path.read_text(encoding="utf-8"))

        core = [c["layer_name"] for c in cfg["ai_semantic_classes"] if c.get("core")]
        self.assertGreaterEqual(len(core), 5, "preset 未标记任何 core 类目，本测试失去意义")

        detected = {d.get("layer_name") for d in (ep.get("detections") or [])}
        missing = [c for c in core if c not in detected]
        self.assertEqual(
            missing, [],
            f"core 类目在真实运行中漏检：{missing}\n"
            f"  episode={ep.get('episode_id')} preset={preset_name}\n"
            f"  实际检出 {len(detected)} 层: {sorted(detected)}")

    def test_golden_baseline_is_real_data(self):
        """基线必须是**真实测量值**（2026-09-16 重提后加防腐）。

        旧基线的教训：`total_size_mb=0.0`、`backend='unknown'` 从未被填真值，
        等于拿一份失真数据做"不退化"比对。本测试锁死：每条基线必须有
        正的产物体积、完整的 8 维逐维判定，且关键指标落在合理区间。
        """
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data), 5, "基线任务数不足（应覆盖 5 个金标准）")
        for tid, v in data.items():
            self.assertGreater(float(v.get("total_size_mb", 0)), 1.0,
                               f"{tid}: total_size_mb 未填真值（<=1MB）")
            self.assertTrue(v.get("passed"), f"{tid}: 基线任务本身未通过 8 维审计")
            dp = v.get("dims_passed") or {}
            self.assertEqual(len(dp), 8, f"{tid}: dims_passed 应记录 8 维（实际 {len(dp)}）")
            self.assertTrue(all(dp.values()), f"{tid}: 基线的 8 维存在失败项 {dp}")
            self.assertLess(float(v.get("lost_ratio", 1)), 0.05,
                            f"{tid}: lost_ratio 异常偏大")
            self.assertGreater(int(v.get("n_layers", 0)), 3, f"{tid}: 图层数异常")

    def test_regression_mechanism_against_real_baseline(self):
        """用真实基线跑一次回归测试器：金标准对自身必须零回归（机制 + 数据双验证）。"""
        import os
        import tempfile
        cur = tempfile.NamedTemporaryFile(suffix=".audit.json", delete=False)
        cur.close()
        try:
            # 以 golden_screen 的基线值构造一份"当前审计"（与基线同值 → 应零回归）
            base = json.loads(BASELINE.read_text(encoding="utf-8"))["golden_screen"]
            audit = {
                "passed": True, "psb": None,
                "dims": {
                    "④ 内容承载": {"passed": True, "metrics": {"lost_ratio": base["lost_ratio"]}},
                    "⑤ 合成等价性": {"passed": True, "metrics": {
                        "rmse_raw": base["rmse_raw"], "rmse_lowfreq": base["rmse_lowfreq"]}},
                    "⑦ plate 合规": {"passed": True, "metrics": {
                        "plate_purity_ok": base["plate_purity_ok"],
                        "tac_max_pct": base["tac_max_pct"]}},
                    "① 层属性": {"passed": True, "metrics": {"layer_count": base["n_layers"]}},
                },
            }
            with open(cur.name, "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False)
            r = run_regression_test(cur.name, "golden_screen",
                                    baseline_path=str(BASELINE))
            self.assertTrue(r["passed"], f"同值重放应零回归，实际 issues={r['issues']}")
        finally:
            os.unlink(cur.name)

    def test_audit_metrics_no_regression(self):
        """审计不退化：**回归测试器机制自检**（不再永久跳过）。

        此前该测试 `self.skipTest(...)` 永久跳过，而 `run_regression_test` 与
        `tests/baseline_audit_8d.json` 都已存在却从未被任何测试调用 —— 即
        「回归保护存在但从未证明有效」。本测试用**临时文件端到端**驱动
        `run_regression_test`，证明它真能：① 无退化时放行；② 各类退化能被抓到。
        （对真实产物的逐任务回归仍由端到端跑批承担。）
        """
        import json
        import tempfile
        import os
        from engine.adaptive.regression_tester import run_regression_test

        def _audit(lost, rmse_low, n_layers, purity, passed=True):
            """构造与 tools/audit_psb.py 同构的最小审计 JSON。"""
            return {
                "passed": passed,
                "dims": {
                    "④ 内容承载": {"passed": True, "metrics": {"lost_ratio": lost}},
                    "⑤ 合成等价性": {"passed": True, "metrics": {"rmse_raw": rmse_low, "rmse_lowfreq": rmse_low}},
                    "⑦ plate 合规": {"passed": True, "metrics": {"plate_purity_ok": purity, "tac_max_pct": 300.0}},
                    "① 层属性": {"passed": True, "metrics": {"layer_count": n_layers}},
                },
            }

        base = {
            "t_base": {
                "lost_ratio": 0.001, "rmse_raw": 1.0, "rmse_lowfreq": 1.0,
                "plate_purity_ok": True, "tac_max_pct": 300.0, "n_layers": 10,
                "total_size_mb": 0.0, "backend": "fallback_rule_based",
                "passed": True,
                "dims_passed": {"④ 内容承载": True, "⑤ 合成等价性": True,
                                "⑦ plate 合规": True, "① 层属性": True},
            }
        }
        tmp = tempfile.mkdtemp()
        base_path = os.path.join(tmp, "baseline.json")
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False)

        def _run(audit_obj):
            cur = os.path.join(tmp, "current.audit.json")
            with open(cur, "w", encoding="utf-8") as f:
                json.dump(audit_obj, f, ensure_ascii=False)
            return run_regression_test(cur, "t_base", baseline_path=base_path)

        # ① 无退化 → 必须放行
        r = _run(_audit(0.001, 1.0, 10, True))
        self.assertTrue(r["passed"], f"无退化应放行，实际 issues={r['issues']}")

        # ② 低频 RMSE 退化（1.0 → 2.0，+100% > 5%）→ 必须抓到
        r = _run(_audit(0.001, 2.0, 10, True))
        self.assertFalse(r["passed"], "rmse_lowfreq +100% 必须判退化")
        self.assertTrue(any("低频 RMSE" in x for x in r["issues"]), r["issues"])

        # ③ 内容丢失退化（0.001 → 0.05）→ 必须抓到
        r = _run(_audit(0.05, 1.0, 10, True))
        self.assertFalse(r["passed"], "lost_ratio 恶化必须判退化")
        self.assertTrue(any("内容丢失" in x for x in r["issues"]), r["issues"])

        # ④ 层数骤减（10 → 5）→ 必须抓到
        r = _run(_audit(0.001, 1.0, 5, True))
        self.assertFalse(r["passed"], "层数腰斩必须判退化")
        self.assertTrue(any("图层数减少" in x for x in r["issues"]), r["issues"])

        # ⑤ 底板纯度布尔翻转 → 必须抓到
        r = _run(_audit(0.001, 1.0, 10, False))
        self.assertFalse(r["passed"], "plate_purity_ok True→False 必须判退化")
        self.assertTrue(any("底板纯度退化" in x for x in r["issues"]), r["issues"])

        # ⑥ 基线缺失该任务 → 必须显式报错（不得静默通过）
        cur = os.path.join(tmp, "cur2.audit.json")
        with open(cur, "w", encoding="utf-8") as f:
            json.dump(_audit(0.001, 1.0, 10, True), f, ensure_ascii=False)
        with self.assertRaises(KeyError):
            run_regression_test(cur, "不存在的任务", baseline_path=base_path)


if __name__ == "__main__":
    unittest.main()
