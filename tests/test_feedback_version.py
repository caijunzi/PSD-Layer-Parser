"""B4：人审反馈落版本节点（create_version 接线验证）。

背景：db_manager.create_version 此前**无任何调用方**（版本链空转）。
现将人审反馈（accept/rename/delete/merge）接到版本链：每次改动落一个版本节点并激活。
"""
import sqlite3
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.adaptive.active_learner import apply_feedback_to_category

SCHEMA = ENGINE_ROOT / "engine" / "adaptive" / "schema.sql"
NOW = 1789000000


class TestFeedbackVersionChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Path(self.tmp) / "t.db"
        conn = sqlite3.connect(self.db)
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO categories (id,name_zh,name_en,name_template,confidence,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            ("water_ripples", "水波纹", "Water_Ripples", "水波纹_Water_Ripples", 0.7, NOW, NOW),
        )
        conn.execute(
            "INSERT INTO category_prompts (category_id,prompt,weight,lang,source,created_at)"
            " VALUES (?,?,?,?,?,?)",
            ("water_ripples", "water ripples", 1.0, "zh", "seed", NOW),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _version_rows(self):
        conn = sqlite3.connect(self.db)
        n = conn.execute("SELECT COUNT(*) FROM category_versions").fetchone()[0]
        act = conn.execute("SELECT version_id FROM active_version WHERE id = 1").fetchone()
        conn.close()
        return n, act

    def test_accept_creates_version_node(self):
        res = apply_feedback_to_category("water_ripples", "accept", {}, db_path=str(self.db))
        self.assertTrue(res["ok"])
        self.assertIn("version_id", res)

        n, act = self._version_rows()
        self.assertEqual(n, 1)
        self.assertIsNotNone(act)
        self.assertEqual(act[0], res["version_id"])

        conn = sqlite3.connect(self.db)
        weight, n_accept = conn.execute(
            "SELECT weight, n_accept FROM category_prompts WHERE category_id='water_ripples'"
        ).fetchone()
        conn.close()
        self.assertAlmostEqual(weight, 1.1, places=5)   # accept ×1.1
        self.assertEqual(n_accept, 1)

    def test_rename_creates_version_and_updates_name(self):
        res = apply_feedback_to_category(
            "water_ripples", "rename", {"new_name": "涟漪"}, db_path=str(self.db)
        )
        self.assertTrue(res["ok"])
        self.assertIn("version_id", res)
        n, _ = self._version_rows()
        self.assertEqual(n, 1)

        conn = sqlite3.connect(self.db)
        name_zh = conn.execute(
            "SELECT name_zh FROM categories WHERE id='water_ripples'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(name_zh, "涟漪")

    def test_failed_action_creates_no_version(self):
        res = apply_feedback_to_category("nope", "accept", {}, db_path=str(self.db))
        # accept 对不存在类目不会失败（UPDATE 影响 0 行），但版本节点仍应登记为一次操作；
        # 这里用 rename 的失败路径验证「失败不登记」
        res2 = apply_feedback_to_category(
            "water_ripples", "rename", {}, db_path=str(self.db)
        )
        self.assertFalse(res2["ok"])
        self.assertNotIn("version_id", res2)


if __name__ == "__main__":
    unittest.main()
