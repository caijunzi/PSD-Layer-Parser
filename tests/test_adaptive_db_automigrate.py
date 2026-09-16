# -*- coding: utf-8 -*-
"""全新自适应数据库「首连自动迁移」回归测试（2026-09-16）。

背景：全新 ADAPTIVE_DB_PATH 首查即报 `no such table: category_prompts` →
自适应整条链路静默不启用（有打印、无自愈）。修复：DBManager 首连时若
`categories` 表不存在，自动按 001→002→003→004 顺序迁移（含种子与类目树）；
**已存在的库一律不触碰**。
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.adaptive.db_manager import DBManager  # noqa: E402


class TestAutoMigrate(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_db_auto_migrates(self):
        """全新库（连父目录都不存在）首连应自动迁移，迁移后 001~004 产物齐全。"""
        db = self.tmp / "fresh" / "nested" / "adaptive.db"   # 父目录不存在，必须自建
        mgr = DBManager(str(db))
        cats = mgr.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        prompts = mgr.execute("SELECT COUNT(*) AS n FROM category_prompts").fetchone()["n"]
        aliases = mgr.execute("SELECT COUNT(*) AS n FROM category_preset_aliases").fetchone()["n"]
        tree = mgr.execute(
            "SELECT COUNT(*) AS n FROM categories WHERE parent_id IS NOT NULL"
        ).fetchone()["n"]
        cols = [r["name"] for r in mgr.execute("PRAGMA table_info(categories)").fetchall()]
        mgr.close()
        self.assertGreater(cats, 0, "001/003 未产生类目")
        self.assertGreater(prompts, 0, "001 未产生类目提示词")
        self.assertGreater(aliases, 0, "002 未产生 preset 别名")
        self.assertGreater(tree, 0, "003 未建立类目树")
        self.assertIn("deleted_at", cols, "004 未加人审反馈字段")
        self.assertTrue(db.is_file())

    def test_existing_db_never_reseeded(self):
        """已存在的库必须**零触碰**：删掉一条种子后重开，不得被自动迁移重新种回。"""
        db = self.tmp / "existing" / "adaptive.db"
        mgr1 = DBManager(str(db))
        victim = mgr1.execute(
            "SELECT id FROM categories ORDER BY id LIMIT 1"
        ).fetchone()["id"]
        mgr1.execute("DELETE FROM categories WHERE id = ?", (victim,))
        mgr1.commit()
        mgr1.close()

        mgr2 = DBManager(str(db))          # 重开：categories 已存在 → 不迁移
        still_gone = mgr2.execute(
            "SELECT COUNT(*) AS n FROM categories WHERE id = ?", (victim,)
        ).fetchone()["n"]
        mgr2.close()
        self.assertEqual(still_gone, 0, "已存在的库被自动迁移重新触碰（违反零触碰约定）")

    def test_schema_check_runs_once_per_instance(self):
        """同一实例多次查询只做一次 schema 检查（幂等）。"""
        db = self.tmp / "once" / "adaptive.db"
        mgr = DBManager(str(db))
        for _ in range(3):
            mgr.execute("SELECT COUNT(*) AS n FROM categories").fetchone()
        self.assertTrue(mgr._schema_checked)
        mgr.execute("SELECT COUNT(*) AS n FROM categories").fetchone()
        self.assertTrue(mgr._schema_checked)
        mgr.close()


if __name__ == "__main__":
    unittest.main()
